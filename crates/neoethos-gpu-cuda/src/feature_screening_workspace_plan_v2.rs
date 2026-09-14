//! Exact first-phase CUDA admission for bounded resident feature screening.
//!
//! The phase keeps only the immutable parent graph, one producer batch, one
//! compact normalization/scoring batch and the native trim/prefilter state.
//! It deliberately excludes the full unfiltered feature cube and population
//! allocations. The same context and stream are later resealed for the exact
//! selected Data+population extent.

use crate::data_population_workspace_plan_v1::{
    DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1, DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1,
    SealedNativeCudaDataPopulationPreflightFactsV1,
};
use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionRequestV3, GpuOnlyRunDeviceAdmissionV3,
    SealedFullDiscoveryTrimAdmissionV1, seal_gpu_only_run_device_admission_v3,
};
use crate::resident_trim_prefilter_v1::{
    SealedResidentTrimPrefilterWorkspacePreflightV2,
    UnboundResidentTrimPrefilterWorkspacePreflightV2,
};
use crate::run_device_admission_v1::{
    DiscoveryRunDeviceAdmissionErrorV1, SealedCudaNativeBuildIdentityV1,
    SealedDiscoveryRunDeviceAdmissionV1,
};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityV3,
};
use sha2::{Digest, Sha256};
use thiserror::Error;

const FEATURE_SCREENING_WORKSPACE_PLAN_SCHEMA_V2: &str =
    "neoethos.feature-screening-gpu-workspace-plan.v2";
const VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2: u64 = 4;

#[derive(Debug)]
pub struct FeatureScreeningWorkspacePreflightRequestV2 {
    pub native_admission_facts: SealedNativeCudaDataPopulationPreflightFactsV1,
    pub parent_row_count: u64,
    pub parent_column_count: u64,
    pub parent_dataset_bytes: u64,
    pub max_live_producer_bytes: u64,
    pub max_live_producer_scratch_bytes: u64,
    pub max_batch_column_count: u64,
    pub normalization_scratch_bytes: u64,
    pub normalization_fit_metadata_bytes: u64,
    pub pointer_table_bytes: u64,
    pub trim_prefilter: UnboundResidentTrimPrefilterWorkspacePreflightV2,
    pub classic_ta_capability: ResidentProducerCapabilityV3,
}

#[derive(Debug)]
pub struct SealedFeatureScreeningGpuWorkspacePlanV2 {
    parent_row_count: u64,
    parent_column_count: u64,
    parent_dataset_bytes: u64,
    max_live_producer_bytes: u64,
    max_live_producer_scratch_bytes: u64,
    max_batch_column_count: u64,
    batch_bar_major_value_bytes: u64,
    batch_packed_validity_bytes: u64,
    normalization_scratch_bytes: u64,
    normalization_fit_metadata_bytes: u64,
    pointer_table_bytes: u64,
    global_parent_ordinal_bytes: u64,
    schema_metadata_bytes: u64,
    trim_prefilter_reserved_bytes: u64,
    screening_runtime_control_bytes: u64,
    required_device_bytes_excluding_reserve: u64,
    allocator_context_reserve_bytes: u64,
    required_device_bytes_including_reserve: u64,
    native_admission_facts_identity_sha256: [u8; 32],
    classic_ta_implementation_sha256: [u8; 32],
    exact_math_authority: String,
    workspace_plan_identity_sha256: [u8; 32],
    trim_prefilter: SealedResidentTrimPrefilterWorkspacePreflightV2,
}

impl SealedFeatureScreeningGpuWorkspacePlanV2 {
    pub const fn parent_row_count(&self) -> u64 {
        self.parent_row_count
    }

    pub const fn parent_column_count(&self) -> u64 {
        self.parent_column_count
    }

    pub const fn parent_dataset_bytes(&self) -> u64 {
        self.parent_dataset_bytes
    }

    pub const fn max_live_producer_bytes(&self) -> u64 {
        self.max_live_producer_bytes
    }

    pub const fn max_live_producer_scratch_bytes(&self) -> u64 {
        self.max_live_producer_scratch_bytes
    }

    pub const fn max_batch_column_count(&self) -> u64 {
        self.max_batch_column_count
    }

    pub const fn batch_bar_major_value_bytes(&self) -> u64 {
        self.batch_bar_major_value_bytes
    }

    pub const fn batch_packed_validity_bytes(&self) -> u64 {
        self.batch_packed_validity_bytes
    }

    pub const fn normalization_scratch_bytes(&self) -> u64 {
        self.normalization_scratch_bytes
    }

    pub const fn normalization_fit_metadata_bytes(&self) -> u64 {
        self.normalization_fit_metadata_bytes
    }

    pub const fn pointer_table_bytes(&self) -> u64 {
        self.pointer_table_bytes
    }

    pub const fn global_parent_ordinal_bytes(&self) -> u64 {
        self.global_parent_ordinal_bytes
    }

    pub const fn schema_metadata_bytes(&self) -> u64 {
        self.schema_metadata_bytes
    }

    pub const fn trim_prefilter_reserved_bytes(&self) -> u64 {
        self.trim_prefilter_reserved_bytes
    }

    pub const fn screening_runtime_control_bytes(&self) -> u64 {
        self.screening_runtime_control_bytes
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

    pub const fn classic_ta_implementation_sha256(&self) -> [u8; 32] {
        self.classic_ta_implementation_sha256
    }

    pub fn exact_math_authority(&self) -> &str {
        &self.exact_math_authority
    }

    pub const fn trim_prefilter_preflight(
        &self,
    ) -> &SealedResidentTrimPrefilterWorkspacePreflightV2 {
        &self.trim_prefilter
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureScreeningWorkspacePlanErrorCodeV2 {
    InvalidExtent,
    ArithmeticOverflow,
    InsufficientExactOrdinalMemory,
    AdmissionFactsMismatch,
    CpuRouteCannotBindGpuWorkspace,
    RunDeviceAdmissionFailure,
}

#[derive(Debug, Error)]
#[error("feature-screening GPU workspace admission failed ({code:?}): {detail}")]
pub struct FeatureScreeningWorkspacePlanErrorV2 {
    code: FeatureScreeningWorkspacePlanErrorCodeV2,
    detail: String,
}

impl FeatureScreeningWorkspacePlanErrorV2 {
    fn new(code: FeatureScreeningWorkspacePlanErrorCodeV2, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> FeatureScreeningWorkspacePlanErrorCodeV2 {
        self.code
    }
}

impl From<DiscoveryRunDeviceAdmissionErrorV1> for FeatureScreeningWorkspacePlanErrorV2 {
    fn from(error: DiscoveryRunDeviceAdmissionErrorV1) -> Self {
        Self::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::RunDeviceAdmissionFailure,
            error.to_string(),
        )
    }
}

pub fn seal_feature_screening_gpu_workspace_plan_v2(
    request: FeatureScreeningWorkspacePreflightRequestV2,
) -> Result<SealedFeatureScreeningGpuWorkspacePlanV2, FeatureScreeningWorkspacePlanErrorV2> {
    let FeatureScreeningWorkspacePreflightRequestV2 {
        native_admission_facts,
        parent_row_count,
        parent_column_count,
        parent_dataset_bytes,
        max_live_producer_bytes,
        max_live_producer_scratch_bytes,
        max_batch_column_count,
        normalization_scratch_bytes,
        normalization_fit_metadata_bytes,
        pointer_table_bytes,
        trim_prefilter,
        classic_ta_capability,
    } = request;
    if parent_row_count == 0
        || parent_column_count == 0
        || parent_dataset_bytes == 0
        || max_live_producer_bytes == 0
        || max_batch_column_count == 0
        || max_batch_column_count > parent_column_count
        || pointer_table_bytes == 0
        || trim_prefilter.parent_column_count() != parent_column_count
        || trim_prefilter.schema_metadata_bytes() == 0
        || trim_prefilter.peak_device_bytes() == 0
        || native_admission_facts.facts_identity_sha256() == [0; 32]
        || native_admission_facts.pre_materialization_free_bytes_snapshot() == 0
        || native_admission_facts.allocator_context_reserve_bytes()
            != DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1
        || classic_ta_capability.producer() != ResidentFeatureProducerV3::ClassicTa
        || classic_ta_capability.implementation_sha256() == [0; 32]
        || classic_ta_capability
            .exact_math_authority()
            .trim()
            .is_empty()
    {
        return Err(FeatureScreeningWorkspacePlanErrorV2::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::InvalidExtent,
            "screening recipe, native admission, trim preflight or CUDA capability is incomplete",
        ));
    }
    let batch_cells = parent_row_count
        .checked_mul(max_batch_column_count)
        .ok_or_else(|| overflow_v2("screening batch cells"))?;
    let batch_bar_major_value_bytes = batch_cells
        .checked_mul(8)
        .ok_or_else(|| overflow_v2("screening batch value bytes"))?;
    let batch_packed_validity_logical_bytes = batch_cells / 2 + batch_cells % 2;
    let batch_packed_validity_bytes = align_up_v2(
        batch_packed_validity_logical_bytes,
        VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2,
        "screening packed validity bytes",
    )?;
    let global_parent_ordinal_bytes = max_batch_column_count
        .checked_mul(4)
        .ok_or_else(|| overflow_v2("screening global ordinal bytes"))?;
    // One aggregate placeholder value, four validity bytes, one aggregate
    // control word, and one per-batch local control word.
    let screening_runtime_control_bytes = 8 + 4 + 4 + 4;
    let schema_metadata_bytes = trim_prefilter.schema_metadata_bytes();
    let trim_prefilter_reserved_bytes = trim_prefilter
        .peak_device_bytes()
        .checked_add(schema_metadata_bytes)
        .ok_or_else(|| overflow_v2("screening trim plus schema bytes"))?;
    let required_device_bytes_excluding_reserve = checked_sum_v2(
        &[
            parent_dataset_bytes,
            max_live_producer_bytes,
            max_live_producer_scratch_bytes,
            batch_bar_major_value_bytes,
            batch_packed_validity_bytes,
            normalization_scratch_bytes,
            normalization_fit_metadata_bytes,
            pointer_table_bytes,
            global_parent_ordinal_bytes,
            trim_prefilter_reserved_bytes,
            screening_runtime_control_bytes,
        ],
        "complete feature-screening peak",
    )?;
    let allocator_context_reserve_bytes = DATA_POPULATION_ALLOCATOR_RESERVE_BYTES_V1;
    let required_device_bytes_including_reserve = required_device_bytes_excluding_reserve
        .checked_add(allocator_context_reserve_bytes)
        .ok_or_else(|| overflow_v2("feature-screening peak plus allocator reserve"))?;
    if required_device_bytes_including_reserve
        > native_admission_facts.pre_materialization_free_bytes_snapshot()
    {
        return Err(FeatureScreeningWorkspacePlanErrorV2::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::InsufficientExactOrdinalMemory,
            format!(
                "screening requires {required_device_bytes_including_reserve} bytes including reserve; admitted ordinal {} has {} bytes free",
                native_admission_facts.selected_device_ordinal(),
                native_admission_facts.pre_materialization_free_bytes_snapshot()
            ),
        ));
    }
    let trim_prefilter = trim_prefilter
        .bind_screening_workspace_reserve_v2(required_device_bytes_excluding_reserve)
        .map_err(|error| {
            FeatureScreeningWorkspacePlanErrorV2::new(
                FeatureScreeningWorkspacePlanErrorCodeV2::InvalidExtent,
                format!("bind native trim reserve: {error:?}"),
            )
        })?;
    let classic_ta_implementation_sha256 = classic_ta_capability.implementation_sha256();
    let exact_math_authority = classic_ta_capability.exact_math_authority().to_owned();
    let workspace_plan_identity_sha256 = workspace_identity_v2(
        &[
            parent_row_count,
            parent_column_count,
            parent_dataset_bytes,
            max_live_producer_bytes,
            max_live_producer_scratch_bytes,
            max_batch_column_count,
            batch_bar_major_value_bytes,
            batch_packed_validity_bytes,
            normalization_scratch_bytes,
            normalization_fit_metadata_bytes,
            pointer_table_bytes,
            global_parent_ordinal_bytes,
            schema_metadata_bytes,
            trim_prefilter_reserved_bytes,
            screening_runtime_control_bytes,
            required_device_bytes_excluding_reserve,
            allocator_context_reserve_bytes,
            required_device_bytes_including_reserve,
        ],
        native_admission_facts.facts_identity_sha256(),
        trim_prefilter.allocation_plan_sha256(),
        classic_ta_implementation_sha256,
        &exact_math_authority,
    );
    Ok(SealedFeatureScreeningGpuWorkspacePlanV2 {
        parent_row_count,
        parent_column_count,
        parent_dataset_bytes,
        max_live_producer_bytes,
        max_live_producer_scratch_bytes,
        max_batch_column_count,
        batch_bar_major_value_bytes,
        batch_packed_validity_bytes,
        normalization_scratch_bytes,
        normalization_fit_metadata_bytes,
        pointer_table_bytes,
        global_parent_ordinal_bytes,
        schema_metadata_bytes,
        trim_prefilter_reserved_bytes,
        screening_runtime_control_bytes,
        required_device_bytes_excluding_reserve,
        allocator_context_reserve_bytes,
        required_device_bytes_including_reserve,
        native_admission_facts_identity_sha256: native_admission_facts.facts_identity_sha256(),
        classic_ta_implementation_sha256,
        exact_math_authority,
        workspace_plan_identity_sha256,
        trim_prefilter,
    })
}

#[derive(Debug)]
#[must_use = "consume the admitted screening run into Data's bounded two-pass producer"]
pub struct AdmittedNativeCudaFeatureScreeningRunV2 {
    run_device: GpuOnlyRunDeviceAdmissionV3,
}

impl AdmittedNativeCudaFeatureScreeningRunV2 {
    pub fn into_gpu_only_run_device_admission_v3(self) -> GpuOnlyRunDeviceAdmissionV3 {
        self.run_device
    }
}

pub fn bind_feature_screening_gpu_workspace_plan_v2(
    admission: SealedDiscoveryRunDeviceAdmissionV1,
    plan: SealedFeatureScreeningGpuWorkspacePlanV2,
) -> Result<AdmittedNativeCudaFeatureScreeningRunV2, FeatureScreeningWorkspacePlanErrorV2> {
    admission
        .probe_counters()
        .require_exact_single_run_device_acquisition_v1()?;
    let SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native) = admission else {
        return Err(FeatureScreeningWorkspacePlanErrorV2::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::CpuRouteCannotBindGpuWorkspace,
            "a no-physical-GPU route cannot bind feature-screening CUDA workspace",
        ));
    };
    let recomputed =
        crate::data_population_workspace_plan_v1::native_cuda_data_population_preflight_facts_v1(
            &native,
        );
    if recomputed.facts_identity_sha256() != plan.native_admission_facts_identity_sha256 {
        return Err(FeatureScreeningWorkspacePlanErrorV2::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::AdmissionFactsMismatch,
            "screening plan was not sealed from this exact native admission",
        ));
    }
    if plan.required_device_bytes_including_reserve > native.free_memory_bytes_snapshot {
        return Err(FeatureScreeningWorkspacePlanErrorV2::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::InsufficientExactOrdinalMemory,
            "screening workspace no longer fits its sealed free-memory snapshot",
        ));
    }
    let native = *native;
    let crate::run_device_admission_v1::SealedNativeCudaRunDeviceAdmissionV1 {
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
        multiprocessor_count,
        warp_size,
        free_memory_bytes_snapshot,
        ..
    } = native;
    let SealedCudaNativeBuildIdentityV1 {
        artifact_sha256: gpu_cuda_build_sha256,
        nvcc_version,
        ..
    } = cuda_build_identity;
    let workspace_plan_identity_sha256 = plan.workspace_plan_identity_sha256;
    let full_trim = SealedFullDiscoveryTrimAdmissionV1::new(
        workspace_plan_identity_sha256,
        plan.required_device_bytes_excluding_reserve,
        plan.trim_prefilter_reserved_bytes,
        plan.required_device_bytes_excluding_reserve,
    );
    let run_device = seal_gpu_only_run_device_admission_v3(GpuOnlyRunDeviceAdmissionRequestV3 {
        source_admission_identity_sha256: admission_identity_sha256,
        native_preflight_facts_identity_sha256: plan.native_admission_facts_identity_sha256,
        workspace_plan_identity_sha256,
        selected_device_ordinal: ordinal,
        device_uuid,
        compute_capability_major,
        compute_capability_minor,
        multiprocessor_count,
        warp_size,
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
        data_population_limits: None,
        full_discovery_trim_admission: Some(full_trim),
    })
    .map_err(|error| {
        FeatureScreeningWorkspacePlanErrorV2::new(
            FeatureScreeningWorkspacePlanErrorCodeV2::RunDeviceAdmissionFailure,
            error.to_string(),
        )
    })?;
    Ok(AdmittedNativeCudaFeatureScreeningRunV2 { run_device })
}

fn overflow_v2(field: &'static str) -> FeatureScreeningWorkspacePlanErrorV2 {
    FeatureScreeningWorkspacePlanErrorV2::new(
        FeatureScreeningWorkspacePlanErrorCodeV2::ArithmeticOverflow,
        format!("{field} overflowed"),
    )
}

fn checked_sum_v2(
    values: &[u64],
    field: &'static str,
) -> Result<u64, FeatureScreeningWorkspacePlanErrorV2> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value).ok_or_else(|| overflow_v2(field))
    })
}

fn align_up_v2(
    value: u64,
    alignment: u64,
    field: &'static str,
) -> Result<u64, FeatureScreeningWorkspacePlanErrorV2> {
    value
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
        .ok_or_else(|| overflow_v2(field))
}

fn workspace_identity_v2(
    extents: &[u64],
    native_facts_identity_sha256: [u8; 32],
    trim_allocation_plan_sha256: [u8; 32],
    classic_ta_implementation_sha256: [u8; 32],
    exact_math_authority: &str,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(FEATURE_SCREENING_WORKSPACE_PLAN_SCHEMA_V2.as_bytes());
    for extent in extents {
        hasher.update(extent.to_le_bytes());
    }
    hasher.update(native_facts_identity_sha256);
    hasher.update(trim_allocation_plan_sha256);
    hasher.update(classic_ta_implementation_sha256);
    hasher.update(exact_math_authority.as_bytes());
    hasher.update(DATA_POPULATION_ALLOCATOR_RESERVE_POLICY_V1.as_bytes());
    hasher.finalize().into()
}
