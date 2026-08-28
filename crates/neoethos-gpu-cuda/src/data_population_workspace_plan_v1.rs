//! One-shot CUDA admission for the currently implemented Discovery stage:
//! exact resident Data materialization followed by one resident population
//! evaluator. This is deliberately not a `FullDiscovery` plan; later
//! validation phases require their own reviewed planners before they can share
//! an arena with this stage.

use crate::population::{PopulationGeneStorePlanV1, PopulationMetricsOnlyPlanV1};
use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionRequestV3, GpuOnlyRunDeviceAdmissionV3,
    seal_gpu_only_run_device_admission_v3,
};
use crate::run_device_admission_v1::{
    DiscoveryRunDeviceAdmissionErrorV1, SealedCudaNativeBuildIdentityV1,
    SealedDiscoveryRunDeviceAdmissionV1, SealedNativeCudaRunDeviceAdmissionV1,
};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityV3, ResidentWorkingSetExtentV3,
};
use sha2::{Digest, Sha256};
use thiserror::Error;

const DATA_POPULATION_WORKSPACE_PLAN_SCHEMA_V1: &str =
    "neoethos.data-population-gpu-workspace-plan.v1";
pub const DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1: u64 = 64 * 1024 * 1024;
pub const DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1: &str =
    crate::resident_feature_store_v3::RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3;

#[derive(Debug)]
pub struct DataPopulationWorkspacePreflightRequestV1 {
    pub native_admission_facts: SealedNativeCudaDataPopulationPreflightFactsV1,
    pub data_extent: ResidentWorkingSetExtentV3,
    pub max_ordered_index_count: usize,
    pub max_adaptive_row_count: usize,
    pub gene_plan: PopulationGeneStorePlanV1,
    pub metrics_plan: PopulationMetricsOnlyPlanV1,
    pub classic_ta_capability: ResidentProducerCapabilityV3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedDataPopulationExecutionLimitsV1 {
    workspace_plan_identity_sha256: [u8; 32],
    population_sizing_authority_sha256: [u8; 32],
    data_extent_identity_sha256: [u8; 32],
    parent_row_count: u64,
    feature_count: u64,
    max_ordered_index_count: u64,
    max_adaptive_row_count: u64,
    max_candidate_count: u64,
    max_gene_term_count: u64,
    max_concurrent_scenario_count: u64,
    month_capacity: u64,
    bounded_host_metric_readback_bytes: u64,
}

impl SealedDataPopulationExecutionLimitsV1 {
    pub const fn workspace_plan_identity_sha256(&self) -> [u8; 32] {
        self.workspace_plan_identity_sha256
    }

    pub const fn population_sizing_authority_sha256(&self) -> [u8; 32] {
        self.population_sizing_authority_sha256
    }

    pub const fn data_extent_identity_sha256(&self) -> [u8; 32] {
        self.data_extent_identity_sha256
    }

    pub const fn parent_row_count(&self) -> u64 {
        self.parent_row_count
    }

    pub const fn feature_count(&self) -> u64 {
        self.feature_count
    }

    pub const fn max_ordered_index_count(&self) -> u64 {
        self.max_ordered_index_count
    }

    pub const fn max_adaptive_row_count(&self) -> u64 {
        self.max_adaptive_row_count
    }

    pub const fn max_candidate_count(&self) -> u64 {
        self.max_candidate_count
    }

    pub const fn max_gene_term_count(&self) -> u64 {
        self.max_gene_term_count
    }

    pub const fn max_concurrent_scenario_count(&self) -> u64 {
        self.max_concurrent_scenario_count
    }

    pub const fn month_capacity(&self) -> u64 {
        self.month_capacity
    }

    pub const fn bounded_host_metric_readback_bytes(&self) -> u64 {
        self.bounded_host_metric_readback_bytes
    }
}

#[derive(Debug)]
pub struct SealedDataPopulationGpuWorkspacePlanV1 {
    data_peak_device_bytes: u64,
    data_steady_device_bytes: u64,
    population_gap_flags_bytes: u64,
    max_ordered_view_indices_bytes: u64,
    max_adaptive_base_pips_bytes: u64,
    retained_population_view_bytes: u64,
    gene_store_device_bytes: u64,
    metrics_scenario_device_bytes: u64,
    population_incremental_device_bytes: u64,
    bounded_host_metric_readback_bytes: u64,
    required_device_bytes_excluding_reserve: u64,
    allocator_context_reserve_bytes: u64,
    required_device_bytes_including_reserve: u64,
    classic_ta_implementation_sha256: [u8; 32],
    exact_math_authority: String,
    workspace_plan_identity_sha256: [u8; 32],
    native_admission_facts_identity_sha256: [u8; 32],
    data_extent_identity_sha256: [u8; 32],
    limits: SealedDataPopulationExecutionLimitsV1,
}

impl SealedDataPopulationGpuWorkspacePlanV1 {
    pub const fn data_peak_device_bytes(&self) -> u64 {
        self.data_peak_device_bytes
    }

    pub const fn data_steady_device_bytes(&self) -> u64 {
        self.data_steady_device_bytes
    }

    pub const fn population_gap_flags_bytes(&self) -> u64 {
        self.population_gap_flags_bytes
    }

    pub const fn max_ordered_view_indices_bytes(&self) -> u64 {
        self.max_ordered_view_indices_bytes
    }

    pub const fn max_adaptive_base_pips_bytes(&self) -> u64 {
        self.max_adaptive_base_pips_bytes
    }

    pub const fn retained_population_view_bytes(&self) -> u64 {
        self.retained_population_view_bytes
    }

    pub const fn gene_store_device_bytes(&self) -> u64 {
        self.gene_store_device_bytes
    }

    pub const fn metrics_scenario_device_bytes(&self) -> u64 {
        self.metrics_scenario_device_bytes
    }

    pub const fn population_incremental_device_bytes(&self) -> u64 {
        self.population_incremental_device_bytes
    }

    pub const fn bounded_host_metric_readback_bytes(&self) -> u64 {
        self.bounded_host_metric_readback_bytes
    }

    pub const fn required_device_bytes_excluding_reserve(&self) -> u64 {
        self.required_device_bytes_excluding_reserve
    }

    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
    }

    pub const fn required_device_bytes_including_reserve(&self) -> u64 {
        self.required_device_bytes_including_reserve
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; 32] {
        self.workspace_plan_identity_sha256
    }

    pub const fn native_admission_facts_identity_sha256(&self) -> [u8; 32] {
        self.native_admission_facts_identity_sha256
    }

    pub const fn data_extent_identity_sha256(&self) -> [u8; 32] {
        self.data_extent_identity_sha256
    }

    pub const fn limits(&self) -> &SealedDataPopulationExecutionLimitsV1 {
        &self.limits
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedNativeCudaDataPopulationPreflightFactsV1 {
    admission_identity_sha256: [u8; 32],
    selected_device_ordinal: u32,
    pre_materialization_free_bytes_snapshot: u64,
    allocator_context_reserve_bytes: u64,
    compute_capability_major: u16,
    compute_capability_minor: u16,
    cuda_build_manifest_sha256: [u8; 32],
    cuda_build_artifact_sha256: [u8; 32],
    facts_identity_sha256: [u8; 32],
}

impl SealedNativeCudaDataPopulationPreflightFactsV1 {
    pub const fn admission_identity_sha256(&self) -> [u8; 32] {
        self.admission_identity_sha256
    }

    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }

    pub const fn pre_materialization_free_bytes_snapshot(&self) -> u64 {
        self.pre_materialization_free_bytes_snapshot
    }

    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
    }

    pub const fn allocator_context_reserve_policy(&self) -> &'static str {
        DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1
    }

    pub const fn compute_capability_major(&self) -> u16 {
        self.compute_capability_major
    }

    pub const fn compute_capability_minor(&self) -> u16 {
        self.compute_capability_minor
    }

    pub const fn cuda_build_manifest_sha256(&self) -> [u8; 32] {
        self.cuda_build_manifest_sha256
    }

    pub const fn cuda_build_artifact_sha256(&self) -> [u8; 32] {
        self.cuda_build_artifact_sha256
    }

    pub const fn facts_identity_sha256(&self) -> [u8; 32] {
        self.facts_identity_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPopulationWorkspacePlanErrorCodeV1 {
    InvalidDataExtent,
    InvalidPopulationExtent,
    WorkspaceExtentOverflow,
    InsufficientExactOrdinalMemory,
    AdmissionFactsMismatch,
    CpuRouteCannotBindGpuWorkspace,
    RunDeviceAdmissionFailure,
}

#[derive(Debug, Error)]
#[error("Data+population GPU workspace admission failed ({code:?}): {detail}")]
pub struct DataPopulationWorkspacePlanErrorV1 {
    code: DataPopulationWorkspacePlanErrorCodeV1,
    detail: String,
}

impl DataPopulationWorkspacePlanErrorV1 {
    fn new(code: DataPopulationWorkspacePlanErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> DataPopulationWorkspacePlanErrorCodeV1 {
        self.code
    }
}

impl From<DiscoveryRunDeviceAdmissionErrorV1> for DataPopulationWorkspacePlanErrorV1 {
    fn from(error: DiscoveryRunDeviceAdmissionErrorV1) -> Self {
        Self::new(
            DataPopulationWorkspacePlanErrorCodeV1::RunDeviceAdmissionFailure,
            error.to_string(),
        )
    }
}

pub fn seal_data_population_gpu_workspace_plan_v1(
    request: DataPopulationWorkspacePreflightRequestV1,
) -> Result<SealedDataPopulationGpuWorkspacePlanV1, DataPopulationWorkspacePlanErrorV1> {
    let DataPopulationWorkspacePreflightRequestV1 {
        native_admission_facts,
        data_extent,
        max_ordered_index_count,
        max_adaptive_row_count,
        gene_plan,
        metrics_plan,
        classic_ta_capability,
    } = request;
    if data_extent.row_count() == 0
        || data_extent.column_count() == 0
        || data_extent.peak_device_bytes() < data_extent.steady_device_bytes()
        || native_admission_facts.facts_identity_sha256() == [0; 32]
        || native_admission_facts.admission_identity_sha256() == [0; 32]
        || native_admission_facts.cuda_build_manifest_sha256() == [0; 32]
        || native_admission_facts.cuda_build_artifact_sha256() == [0; 32]
        || native_admission_facts.pre_materialization_free_bytes_snapshot() == 0
        || native_admission_facts.allocator_context_reserve_bytes()
            != DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1
    {
        return Err(DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::InvalidDataExtent,
            "resident Data extent or exact native admission facts are incomplete",
        ));
    }
    let max_ordered_index_count = u64::try_from(max_ordered_index_count).map_err(|_| {
        DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::InvalidPopulationExtent,
            "ordered-view cap does not fit u64",
        )
    })?;
    let max_adaptive_row_count = u64::try_from(max_adaptive_row_count).map_err(|_| {
        DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::InvalidPopulationExtent,
            "adaptive-view cap does not fit u64",
        )
    })?;
    if max_ordered_index_count > data_extent.row_count()
        || max_adaptive_row_count > data_extent.row_count()
        || gene_plan.candidate_count() == 0
        || gene_plan.term_count() == 0
        || metrics_plan.scenario_count() == 0
        || metrics_plan.scenario_count() > gene_plan.candidate_count()
        || metrics_plan.month_capacity() == 0
        || metrics_plan.outcome_bytes() != 0
        || metrics_plan.accepted_trade_total_bytes() != 0
        || classic_ta_capability.producer() != ResidentFeatureProducerV3::ClassicTa
        || classic_ta_capability.implementation_sha256() == [0; 32]
        || classic_ta_capability
            .exact_math_authority()
            .trim()
            .is_empty()
    {
        return Err(DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::InvalidPopulationExtent,
            "resident population/view/gene/scenario/build facts are inconsistent",
        ));
    }
    let population_gap_flags_bytes = data_extent.row_count();
    let max_ordered_view_indices_bytes =
        max_ordered_index_count.checked_mul(8).ok_or_else(|| {
            DataPopulationWorkspacePlanErrorV1::new(
                DataPopulationWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "ordered-view bytes overflowed",
            )
        })?;
    let max_adaptive_base_pips_bytes = max_adaptive_row_count.checked_mul(8).ok_or_else(|| {
        DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
            "adaptive-view bytes overflowed",
        )
    })?;
    // The two allocations have independent retained capacities. A view may use
    // both at once, and sequential views do not release either capacity.
    let retained_population_view_bytes = max_ordered_view_indices_bytes
        .checked_add(max_adaptive_base_pips_bytes)
        .ok_or_else(|| {
            DataPopulationWorkspacePlanErrorV1::new(
                DataPopulationWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "retained population-view bytes overflowed",
            )
        })?;
    let population_incremental_device_bytes = population_gap_flags_bytes
        .checked_add(retained_population_view_bytes)
        .and_then(|bytes| bytes.checked_add(gene_plan.total_device_bytes()))
        .and_then(|bytes| bytes.checked_add(metrics_plan.total_device_bytes()))
        .ok_or_else(|| {
            DataPopulationWorkspacePlanErrorV1::new(
                DataPopulationWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "resident population incremental bytes overflowed",
            )
        })?;
    let resident_population_peak = data_extent
        .steady_device_bytes()
        .checked_add(population_incremental_device_bytes)
        .ok_or_else(|| {
            DataPopulationWorkspacePlanErrorV1::new(
                DataPopulationWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "Data steady plus population peak overflowed",
            )
        })?;
    let required_device_bytes_excluding_reserve = data_extent
        .peak_device_bytes()
        .max(resident_population_peak);
    let required_device_bytes_including_reserve = required_device_bytes_excluding_reserve
        .checked_add(DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1)
        .ok_or_else(|| {
            DataPopulationWorkspacePlanErrorV1::new(
                DataPopulationWorkspacePlanErrorCodeV1::WorkspaceExtentOverflow,
                "stage workspace plus allocator reserve overflowed",
            )
        })?;
    if required_device_bytes_including_reserve
        > native_admission_facts.pre_materialization_free_bytes_snapshot()
    {
        return Err(DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::InsufficientExactOrdinalMemory,
            format!(
                "Data+population stage requires {} bytes including its one reserve; admitted ordinal {} has {} bytes free in the sealed pre-materialization snapshot",
                required_device_bytes_including_reserve,
                native_admission_facts.selected_device_ordinal(),
                native_admission_facts.pre_materialization_free_bytes_snapshot()
            ),
        ));
    }
    let bounded_host_metric_readback_bytes = metrics_plan.metric_rows_bytes();
    let data_extent_identity_sha256 = data_extent.identity_sha256();
    let classic_ta_implementation_sha256 = classic_ta_capability.implementation_sha256();
    let exact_math_authority = classic_ta_capability.exact_math_authority().to_owned();
    let workspace_plan_identity_sha256 = hash_workspace_plan_v1(
        &[
            data_extent.row_count(),
            data_extent.column_count(),
            data_extent.parent_dataset_bytes(),
            data_extent.final_bar_major_value_bytes(),
            data_extent.packed_validity_allocated_bytes(),
            data_extent.fit_metadata_bytes(),
            data_extent.steady_device_bytes(),
            data_extent.merkle_scratch_bytes(),
            data_extent.max_live_producer_bytes(),
            data_extent.max_live_producer_scratch_bytes(),
            data_extent.normalization_scratch_bytes(),
            data_extent.pointer_and_schema_metadata_bytes(),
            data_extent.compact_hash_and_error_bytes(),
            data_extent.peak_device_bytes(),
            population_gap_flags_bytes,
            max_ordered_index_count,
            max_adaptive_row_count,
            max_ordered_view_indices_bytes,
            max_adaptive_base_pips_bytes,
            retained_population_view_bytes,
            gene_plan.candidate_count(),
            gene_plan.term_count(),
            gene_plan.total_device_bytes(),
            metrics_plan.scenario_count(),
            metrics_plan.month_capacity(),
            metrics_plan.total_device_bytes(),
            bounded_host_metric_readback_bytes,
            population_incremental_device_bytes,
            required_device_bytes_excluding_reserve,
            DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1,
            required_device_bytes_including_reserve,
        ],
        &native_admission_facts.facts_identity_sha256(),
        &data_extent_identity_sha256,
        &classic_ta_implementation_sha256,
        &exact_math_authority,
    );
    let population_sizing_authority_sha256 = hash_population_sizing_authority_v1(
        native_admission_facts.facts_identity_sha256(),
        workspace_plan_identity_sha256,
        gene_plan.candidate_count(),
        gene_plan.term_count(),
        metrics_plan.scenario_count(),
        metrics_plan.month_capacity(),
        required_device_bytes_excluding_reserve,
        required_device_bytes_including_reserve,
    );
    let limits = SealedDataPopulationExecutionLimitsV1 {
        workspace_plan_identity_sha256,
        population_sizing_authority_sha256,
        data_extent_identity_sha256,
        parent_row_count: data_extent.row_count(),
        feature_count: data_extent.column_count(),
        max_ordered_index_count,
        max_adaptive_row_count,
        max_candidate_count: gene_plan.candidate_count(),
        max_gene_term_count: gene_plan.term_count(),
        max_concurrent_scenario_count: metrics_plan.scenario_count(),
        month_capacity: metrics_plan.month_capacity(),
        bounded_host_metric_readback_bytes,
    };
    Ok(SealedDataPopulationGpuWorkspacePlanV1 {
        data_peak_device_bytes: data_extent.peak_device_bytes(),
        data_steady_device_bytes: data_extent.steady_device_bytes(),
        population_gap_flags_bytes,
        max_ordered_view_indices_bytes,
        max_adaptive_base_pips_bytes,
        retained_population_view_bytes,
        gene_store_device_bytes: gene_plan.total_device_bytes(),
        metrics_scenario_device_bytes: metrics_plan.total_device_bytes(),
        population_incremental_device_bytes,
        bounded_host_metric_readback_bytes,
        required_device_bytes_excluding_reserve,
        allocator_context_reserve_bytes: DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1,
        required_device_bytes_including_reserve,
        classic_ta_implementation_sha256,
        exact_math_authority,
        workspace_plan_identity_sha256,
        native_admission_facts_identity_sha256: native_admission_facts.facts_identity_sha256(),
        data_extent_identity_sha256,
        limits,
    })
}

fn hash_workspace_plan_v1(
    extents: &[u64],
    native_admission_facts_identity_sha256: &[u8; 32],
    data_extent_identity_sha256: &[u8; 32],
    classic_ta_implementation_sha256: &[u8; 32],
    exact_math_authority: &str,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DATA_POPULATION_WORKSPACE_PLAN_SCHEMA_V1.as_bytes());
    for extent in extents {
        hasher.update(extent.to_le_bytes());
    }
    hasher.update(native_admission_facts_identity_sha256);
    hasher.update(data_extent_identity_sha256);
    hasher.update(classic_ta_implementation_sha256);
    hasher.update(exact_math_authority.as_bytes());
    hasher.update(DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1.as_bytes());
    hasher.finalize().into()
}

fn hash_population_sizing_authority_v1(
    native_admission_facts_identity_sha256: [u8; 32],
    workspace_plan_identity_sha256: [u8; 32],
    candidate_count: u64,
    gene_term_count: u64,
    concurrent_scenario_count: u64,
    month_capacity: u64,
    required_device_bytes_excluding_reserve: u64,
    required_device_bytes_including_reserve: u64,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data-population-sizing-authority.v1");
    hasher.update(native_admission_facts_identity_sha256);
    hasher.update(workspace_plan_identity_sha256);
    hasher.update(candidate_count.to_le_bytes());
    hasher.update(gene_term_count.to_le_bytes());
    hasher.update(concurrent_scenario_count.to_le_bytes());
    hasher.update(month_capacity.to_le_bytes());
    hasher.update(required_device_bytes_excluding_reserve.to_le_bytes());
    hasher.update(required_device_bytes_including_reserve.to_le_bytes());
    hasher.finalize().into()
}

pub fn native_cuda_data_population_preflight_facts_v1(
    admission: &SealedNativeCudaRunDeviceAdmissionV1,
) -> SealedNativeCudaDataPopulationPreflightFactsV1 {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.native-cuda-data-population-preflight-facts.v1");
    hasher.update(admission.admission_identity_sha256);
    hasher.update(admission.ordinal.to_le_bytes());
    hasher.update(admission.free_memory_bytes_snapshot.to_le_bytes());
    hasher.update(DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1.to_le_bytes());
    hasher.update(admission.compute_capability_major.to_le_bytes());
    hasher.update(admission.compute_capability_minor.to_le_bytes());
    hasher.update(admission.cuda_build_identity.manifest_sha256);
    hasher.update(admission.cuda_build_identity.artifact_sha256);
    let facts_identity_sha256 = hasher.finalize().into();
    SealedNativeCudaDataPopulationPreflightFactsV1 {
        admission_identity_sha256: admission.admission_identity_sha256,
        selected_device_ordinal: admission.ordinal,
        pre_materialization_free_bytes_snapshot: admission.free_memory_bytes_snapshot,
        allocator_context_reserve_bytes: DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1,
        compute_capability_major: admission.compute_capability_major,
        compute_capability_minor: admission.compute_capability_minor,
        cuda_build_manifest_sha256: admission.cuda_build_identity.manifest_sha256,
        cuda_build_artifact_sha256: admission.cuda_build_identity.artifact_sha256,
        facts_identity_sha256,
    }
}

#[derive(Debug)]
#[must_use = "consume the admitted Data+population run into the resident Data materializer"]
pub struct AdmittedNativeCudaDataPopulationRunV1 {
    run_device: GpuOnlyRunDeviceAdmissionV3,
}

impl AdmittedNativeCudaDataPopulationRunV1 {
    pub fn into_gpu_only_run_device_admission_v3(self) -> GpuOnlyRunDeviceAdmissionV3 {
        self.run_device
    }
}

pub fn bind_data_population_gpu_workspace_plan_v1(
    admission: SealedDiscoveryRunDeviceAdmissionV1,
    plan: SealedDataPopulationGpuWorkspacePlanV1,
) -> Result<AdmittedNativeCudaDataPopulationRunV1, DataPopulationWorkspacePlanErrorV1> {
    admission
        .probe_counters()
        .require_exact_single_run_device_acquisition_v1()?;
    let SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native) = admission else {
        return Err(DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::CpuRouteCannotBindGpuWorkspace,
            "a no-physical-GPU CPU route cannot bind a Data+population CUDA workspace",
        ));
    };
    let recomputed_facts = native_cuda_data_population_preflight_facts_v1(&native);
    if recomputed_facts.facts_identity_sha256() != plan.native_admission_facts_identity_sha256() {
        return Err(DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::AdmissionFactsMismatch,
            "the stage plan was not sealed from this exact native CUDA admission and pre-materialization free-memory snapshot",
        ));
    }
    if plan.required_device_bytes_including_reserve() > native.free_memory_bytes_snapshot {
        return Err(DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::InsufficientExactOrdinalMemory,
            format!(
                "Data+population stage requires {} bytes including its one reserve; admitted ordinal has {} bytes free",
                plan.required_device_bytes_including_reserve(),
                native.free_memory_bytes_snapshot
            ),
        ));
    }
    let native = *native;
    let SealedNativeCudaRunDeviceAdmissionV1 {
        admission_identity_sha256,
        device_uuid,
        ordinal,
        run_stream,
        primary_context,
        cuda_build_identity,
        sass_target,
        driver_version,
        context_api_version,
        compute_capability_major,
        compute_capability_minor,
        free_memory_bytes_snapshot,
        ..
    } = native;
    let SealedCudaNativeBuildIdentityV1 {
        artifact_sha256: gpu_cuda_build_sha256,
        nvcc_version,
        ..
    } = cuda_build_identity;
    let run_device = seal_gpu_only_run_device_admission_v3(GpuOnlyRunDeviceAdmissionRequestV3 {
        source_admission_identity_sha256: admission_identity_sha256,
        workspace_plan_identity_sha256: plan.workspace_plan_identity_sha256,
        selected_device_ordinal: ordinal,
        device_uuid,
        compute_capability_major,
        compute_capability_minor,
        run_stream,
        primary_context,
        driver_version,
        context_api_version,
        nvcc_version,
        native_sass_target: sass_target,
        vector_ta_build_sha256: plan.classic_ta_implementation_sha256,
        gpu_cuda_build_sha256,
        exact_math_authority: plan.exact_math_authority,
        phase_one_free_bytes_snapshot: free_memory_bytes_snapshot,
        allocator_context_reserve_bytes: plan.allocator_context_reserve_bytes,
        data_population_limits: Some(plan.limits),
    })
    .map_err(|error| {
        DataPopulationWorkspacePlanErrorV1::new(
            DataPopulationWorkspacePlanErrorCodeV1::RunDeviceAdmissionFailure,
            error.to_string(),
        )
    })?;
    Ok(AdmittedNativeCudaDataPopulationRunV1 { run_device })
}
