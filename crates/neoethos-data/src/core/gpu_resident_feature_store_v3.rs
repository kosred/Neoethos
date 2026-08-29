//! Opaque Data authority for strict GPU-only feature materialization.
//!
//! The orchestration entrypoint is exported so warning-denied builds exercise
//! the real native/runtime seam, but it fails before consuming the one-shot
//! run-device carrier until every producer in [`ResidentFeatureProducerV3::ALL`]
//! has a real resident implementation. Low-level public contract DTOs cannot
//! mint either authority below.

use std::sync::Arc;
use std::{error::Error as StdError, fmt};

use anyhow::{Context as _, Result, bail};
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureOperationTagV1, FeaturePlanV1,
};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    CANONICAL_MERKLE_CHUNK_ROWS_V3, CanonicalCudaSha256AuthorityV3,
    CudaPrimaryContextBuildIdentityV3, GpuOnlyResidentAdmissionRequestV3,
    GpuOnlyResidentAdmissionV3, ResidentFeatureContractErrorV3, ResidentFeatureProducerV3,
    ResidentFeatureRouteV3, ResidentFeatureStageV3, ResidentProducerCapabilityManifestV3,
    ResidentProducerCapabilityV3, ResidentReadyEventV3, ResidentWorkingSetBoundV3,
    ResidentWorkingSetExtentRequestV3, ResidentWorkingSetExtentV3, ResidentWorkingSetRequestV3,
    SealedResidentFeatureStoreRequestV3, SealedResidentFeatureStoreV3,
};
use neoethos_gpu_cuda::resident_classic_ta_v3::{
    ResidentClassicTaExecutorErrorV3, ResidentClassicTaPreDeviceMemoryReceiptV4,
    ResidentClassicTaRecipeV3, preflight_resident_classic_ta_memory_v4,
    resident_classic_ta_capability_v3,
};
use neoethos_gpu_cuda::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionV3, ResidentFeatureColumnBindingV3, ResidentFeatureCompactHashesV3,
    ResidentFeatureLayoutEvidenceV3, ResidentFeatureStoreAssemblerV3,
    ResidentFeatureStoreCudaErrorV3, ResidentFeatureStoreImportV3, ResidentFeatureStoreOwnerV3,
    resident_canonical_content_sha256_capability_v3,
    resident_feature_major_to_bar_major_capability_v3,
};
use neoethos_gpu_cuda::resident_footprint_v2::{
    FOOTPRINT_SEMANTIC_VERSION_V2, RESIDENT_FOOTPRINT_COLUMN_NAMES_V2,
    preflight_resident_footprint_memory_v4, resident_footprint_capability_v2,
};
use neoethos_gpu_cuda::resident_higher_timeframe_alignment_v3::resident_higher_timeframe_capability_v3;
use neoethos_gpu_cuda::resident_quant_v3::resident_quant_capability_v3;
use neoethos_gpu_cuda::resident_regime_v3::{
    RESIDENT_REGIME_COLUMN_NAMES_V3, RESIDENT_REGIME_SEMANTIC_VERSION_V3,
    preflight_resident_regime_memory_v4, resident_regime_capability_v3,
};
use neoethos_gpu_cuda::resident_robust_normalization_v2::{
    RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2, ResidentRobustNormalizationPlanV2,
    resident_robust_normalization_capability_v2,
    resident_robust_normalization_disabled_fit_sha256_v2,
};
use neoethos_gpu_cuda::resident_session_v2::resident_session_capability_v2;
use neoethos_gpu_cuda::resident_smc_v3::{
    PendingResidentSmcBatchV3, RESIDENT_SMC_COLUMN_NAMES_V3, ResidentSmcMaterializationV3,
    begin_resident_smc_store_v3, preflight_resident_smc_memory_v4, prepare_resident_smc_parent_v3,
    resident_smc_capability_v3,
};
use neoethos_gpu_cuda::{
    AdmittedNativeCudaDataPopulationRunV1, DataPopulationWorkspacePlanErrorV1,
    DataPopulationWorkspacePreflightRequestV1, FullDiscoveryWorkspacePlanErrorV1,
    PopulationGeneStorePlanV1, PopulationMetricsOnlyPlanV1, SealedDataPopulationGpuWorkspacePlanV1,
    SealedNativeCudaDataPopulationPreflightFactsV1,
    full_discovery_workspace_plan_v1::AdmittedNativeCudaFullDiscoveryRunV1,
    seal_data_population_gpu_workspace_plan_v1,
};
use sha2::{Digest, Sha256};
use vector_ta::cuda::F64_EXACT_MATH_AUTHORITY_V3;

use crate::CanonicalOhlcvFrame;

use super::features::FeatureProfile;
use super::footprint_features::{FOOTPRINT_FEATURE_NAMES, FOOTPRINT_SEMANTIC_VERSION};
use super::gpu_only_feature_workspace_preflight_v3::PreparedGpuOnlyFeatureWorkspacePreflightV3;
use super::gpu_resident_classic_ta_v3::{
    RESIDENT_CLASSIC_TA_LOCAL_ROUTE_DOMAIN_V4, ResidentClassicTaPlanV3,
    preflight_resident_classic_ta_v3,
};
use super::gpu_resident_feature_recipe_v4::{
    ResidentCanonicalParameterV4, ResidentCanonicalParameterValueV4,
    ResidentFeatureIdentityTemplateV4, ResidentFeatureRecipeErrorV4, ResidentProducerBatchDraftV4,
    ResidentProducerDraftV4, ResidentRouteDraftV4, ResidentTransformCapabilityDraftV4,
    derive_route_semantic_source_sha256_v4,
};
use super::gpu_resident_higher_timeframe_alignment_v3::{
    HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3, PendingResidentHigherTimeframeRuntimeV3,
    PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3,
    ValidatedResidentHigherTimeframeDirectParentCaptureV3,
    preflight_resident_higher_timeframe_alignment_v3,
    prepare_resident_higher_timeframe_direct_parent_owner_v3,
};
use super::gpu_resident_quant_v3::{
    PreparedResidentQuantRuntimeV3, preflight_current_native_resident_quant_v3,
};
use super::gpu_resident_regime_v3::{PreparedResidentRegimeInputV3, preflight_resident_regime_v3};
use super::gpu_resident_robust_normalization_v2::{
    PreparedResidentRobustNormalizationInputV2, prepare_resident_robust_normalization_input_v2,
};
use super::gpu_resident_session_v2::{
    PreparedResidentSessionRuntimeV2, preflight_current_native_resident_session_v2,
};
use super::hpc_ta::{
    IndicatorComputePolicy, prepare_classic_ta_gpu_exact_parity_run_plan_v3,
    prepare_classic_ta_run_plan,
};
use super::pinned_canonical_series_v1::MaterializedPinnedResidentCanonicalSourcesV1;
use super::pinned_source_projection_v1::{
    CanonicalPinnedSourceProjectionV1, derive_pinned_source_projection_v1,
};
use super::regime_detection::{
    REGIME_FEATURE_NAMES_V3, REGIME_OPERATION_SCHEDULE_V1, REGIME_SEMANTIC_V3_FIXTURE_SHA256,
    REGIME_SEMANTIC_VERSION, REGIME_V2_ARTIFACT_MIGRATION_POLICY,
};
use super::smc::SMC_SEMANTIC_VERSION;

const DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3: &str =
    "neoethos.data.gpu-only-feature-materialization-admission.v3";
const DATA_GPU_ONLY_SEALED_STORE_AUTHORITY_V3: &str =
    "neoethos.data.sealed-gpu-resident-feature-store.v3";
const EXACT_ALLOCATOR_RESERVE_POLICY_V3: &str = "neoethos.cuda.exact-allocator-context-reserve.v3";
const FEATURE_MAJOR_TO_BAR_MAJOR_COMPONENT_AUTHORITY_V3: &str =
    "neoethos.data.feature-major-to-bar-major-workspace-component.v3";
const CANONICAL_CONTENT_SHA256_COMPONENT_AUTHORITY_V3: &str =
    "neoethos.data.canonical-content-sha256-workspace-component.v3";
const FOOTPRINT_COMPONENT_AUTHORITY_V2: &str =
    "neoethos.data.footprint-workspace-component.semantic-v2";
const REGIME_COMPONENT_AUTHORITY_V3: &str = "neoethos.data.regime-workspace-component.semantic-v3";
const ROBUST_NORMALIZATION_COMPONENT_AUTHORITY_V2: &str =
    "neoethos.data.robust-normalization-workspace-component.semantic-v2";

#[derive(Debug)]
pub enum GpuOnlyFeatureMaterializationErrorV3 {
    MissingProducerCapabilities {
        missing: Vec<ResidentFeatureProducerV3>,
    },
    A2ProducerFactoryNotIntegrated {
        component: &'static str,
    },
    PrimaryContextBuildIdentityMismatch,
    Contract(ResidentFeatureContractErrorV3),
    Runtime(ResidentFeatureStoreCudaErrorV3),
    ClassicTa(ResidentClassicTaExecutorErrorV3),
    Other(anyhow::Error),
}

impl fmt::Display for GpuOnlyFeatureMaterializationErrorV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProducerCapabilities { missing } => write!(
                formatter,
                "strict resident producer capability manifest is incomplete: {missing:?}"
            ),
            Self::A2ProducerFactoryNotIntegrated { component } => write!(
                formatter,
                "strict resident A2 producer factory has no exact implementation for {component}"
            ),
            Self::PrimaryContextBuildIdentityMismatch => formatter.write_str(
                "selected CUDA primary-context/build identity does not match exact SASS authority",
            ),
            Self::Contract(error) => write!(formatter, "resident feature contract failed: {error}"),
            Self::Runtime(error) => write!(formatter, "resident CUDA runtime failed: {error}"),
            Self::ClassicTa(error) => write!(formatter, "resident Classic TA failed: {error}"),
            Self::Other(error) => error.fmt(formatter),
        }
    }
}

impl StdError for GpuOnlyFeatureMaterializationErrorV3 {}

impl From<ResidentFeatureContractErrorV3> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: ResidentFeatureContractErrorV3) -> Self {
        Self::Contract(error)
    }
}

impl From<ResidentFeatureStoreCudaErrorV3> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: ResidentFeatureStoreCudaErrorV3) -> Self {
        Self::Runtime(error)
    }
}

impl From<ResidentClassicTaExecutorErrorV3> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: ResidentClassicTaExecutorErrorV3) -> Self {
        Self::ClassicTa(error)
    }
}

impl From<anyhow::Error> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: anyhow::Error) -> Self {
        Self::Other(error)
    }
}

impl From<ResidentFeatureRecipeErrorV4> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: ResidentFeatureRecipeErrorV4) -> Self {
        Self::Other(error.into())
    }
}

impl From<FullDiscoveryWorkspacePlanErrorV1> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: FullDiscoveryWorkspacePlanErrorV1) -> Self {
        Self::Other(error.into())
    }
}

impl From<DataPopulationWorkspacePlanErrorV1> for GpuOnlyFeatureMaterializationErrorV3 {
    fn from(error: DataPopulationWorkspacePlanErrorV1) -> Self {
        Self::Other(error.into())
    }
}

/// Crate-owned resolved inputs. There is no public constructor: Data derives
/// this from the immutable feature recipe, exact route projections and
/// producer ledgers. The moved gpu-cuda run-device carrier is the sole owner of
/// runtime device identity, admission identity and free-memory evidence.
#[derive(Debug)]
pub(crate) struct ResolvedGpuOnlyFeatureMaterializationPlanV3 {
    dataset_recipe_sha256: [u8; 32],
    feature_plan_schema_sha256: [u8; 32],
    route_plan_sha256: [u8; 32],
    row_count: usize,
    planned_routes: Vec<ResidentFeatureRouteV3>,
    producer_capabilities: ResidentProducerCapabilityManifestV3,
    producer_batches: Vec<ResolvedResidentProducerBatchMemoryV3>,
    regime_scale_anchor_bits: u64,
    regime_input_identity_sha256: [u8; 32],
    normalization_scratch_bytes: u64,
    fit_metadata_bytes: u64,
    robust_normalization_input: Option<PreparedResidentRobustNormalizationInputV2>,
    feature_identity: ResidentFeatureIdentityTemplateV4,
}

#[derive(Debug)]
struct ResolvedResidentProducerBatchMemoryV3 {
    producer: ResidentFeatureProducerV3,
    first_column: usize,
    column_count: usize,
    additional_retained_bytes: u64,
    scratch_bytes: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum RobustNormalizationScratchLifetimeV2 {
    DisabledZeroExtent,
    ThroughReadyEventAndBoundedDigestReadback,
}

#[derive(Debug, PartialEq, Eq)]
enum RobustNormalizationFitLifetimeV2 {
    DisabledZeroExtent,
    AlwaysResidentThroughSearchConsumerCompletion,
}

#[derive(Debug, PartialEq, Eq)]
enum RobustNormalizationEventLifetimeV2 {
    DisabledNoEvent,
    AlwaysResidentThroughSearchConsumerCompletion,
}

/// Exact post-pack/pre-SHA allocation and bounded-control contract derived
/// only from Data's consumed canonical split and ordered feature plan.
#[derive(Debug)]
pub(crate) struct RobustNormalizationAllocationReceiptV2 {
    semantic_version: u64,
    enabled: bool,
    row_count: u64,
    feature_column_count: u64,
    training_start: u64,
    training_end: u64,
    padded_training_rows: u64,
    packed_validity_logical_bytes: u64,
    packed_validity_allocated_bytes: u64,
    normalization_scratch_device_bytes: u64,
    fit_metadata_device_bytes: u64,
    batch_count: u64,
    native_launch_count: u64,
    producer_ready_event_count: u64,
    producer_ready_event_synchronize_count: u64,
    control_error_device_bytes: u64,
    control_error_readback_count: u64,
    control_error_d2h_bytes: u64,
    fit_digest_readback_count: u64,
    fit_digest_d2h_bytes: u64,
    parent_input_h2d_bytes: u64,
    feature_value_d2h_bytes: u64,
}

/// Move-only lifetime facts for the scratch, retained fit and exact runtime
/// event. Disabled mode is represented by typed zero-work variants.
#[derive(Debug)]
pub(crate) struct RobustNormalizationLifetimeReceiptV2 {
    scratch: RobustNormalizationScratchLifetimeV2,
    fit: RobustNormalizationFitLifetimeV2,
    event: RobustNormalizationEventLifetimeV2,
}

/// Data-owned semantic/allocation authority. This remains unbound until it is
/// consumed beside the one-shot gpu-cuda run carrier.
#[derive(Debug)]
pub(crate) struct SealedRobustNormalizationComponentReceiptV2 {
    authority: &'static str,
    capability: ResidentProducerCapabilityV3,
    allocation: RobustNormalizationAllocationReceiptV2,
    lifetime: RobustNormalizationLifetimeReceiptV2,
    runtime_plan: ResidentRobustNormalizationPlanV2,
    component_identity_sha256: [u8; 32],
}

/// Same component after binding to the exact admission, primary context and
/// producer stream. There is no constructor from hashes or raw handles.
#[derive(Debug)]
struct BoundRobustNormalizationComponentReceiptV2 {
    sealed: SealedRobustNormalizationComponentReceiptV2,
    admission_identity_sha256: [u8; 32],
    primary_context_process_token: [u8; 32],
    producer_stream_process_token: [u8; 32],
    binding_identity_sha256: [u8; 32],
}

impl SealedRobustNormalizationComponentReceiptV2 {
    fn validate_working_set(
        &self,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        let expected_lifetimes = if self.allocation.enabled {
            (
                RobustNormalizationScratchLifetimeV2::ThroughReadyEventAndBoundedDigestReadback,
                RobustNormalizationFitLifetimeV2::AlwaysResidentThroughSearchConsumerCompletion,
                RobustNormalizationEventLifetimeV2::AlwaysResidentThroughSearchConsumerCompletion,
            )
        } else {
            (
                RobustNormalizationScratchLifetimeV2::DisabledZeroExtent,
                RobustNormalizationFitLifetimeV2::DisabledZeroExtent,
                RobustNormalizationEventLifetimeV2::DisabledNoEvent,
            )
        };
        if self.authority != ROBUST_NORMALIZATION_COMPONENT_AUTHORITY_V2
            || self.capability.producer() != ResidentFeatureProducerV3::RobustNormalization
            || self.component_identity_sha256 == [0; 32]
            || self.lifetime.scratch != expected_lifetimes.0
            || self.lifetime.fit != expected_lifetimes.1
            || self.lifetime.event != expected_lifetimes.2
            || working_set.row_count() != self.allocation.row_count
            || working_set.column_count() != self.allocation.feature_column_count
            || working_set.packed_validity_logical_bytes()
                != self.allocation.packed_validity_logical_bytes
            || working_set.packed_validity_allocated_bytes()
                != self.allocation.packed_validity_allocated_bytes
            || working_set.packed_validity_allocated_bytes() % 4 != 0
            || working_set.normalization_scratch_bytes()
                != self.allocation.normalization_scratch_device_bytes
            || working_set.fit_metadata_bytes() != self.allocation.fit_metadata_device_bytes
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }

    fn bind_run_device_v2(
        self,
        run_device: &GpuOnlyRunDeviceAdmissionV3,
    ) -> std::result::Result<
        BoundRobustNormalizationComponentReceiptV2,
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        let admission_identity_sha256 = run_device.admission_identity_sha256();
        let primary_context_process_token =
            run_device.device_identity().primary_context_process_token();
        let producer_stream_process_token = run_device.run_stream_process_token_v3();
        if admission_identity_sha256 == [0; 32]
            || primary_context_process_token == [0; 32]
            || producer_stream_process_token == [0; 32]
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        let mut binding = Sha256::new();
        binding.update(b"neoethos.data.robust-normalization-run-binding.semantic-v2");
        binding.update(self.component_identity_sha256);
        binding.update(admission_identity_sha256);
        binding.update(primary_context_process_token);
        binding.update(producer_stream_process_token);
        let binding_identity_sha256 = binding.finalize().into();
        Ok(BoundRobustNormalizationComponentReceiptV2 {
            sealed: self,
            admission_identity_sha256,
            primary_context_process_token,
            producer_stream_process_token,
            binding_identity_sha256,
        })
    }
}

impl BoundRobustNormalizationComponentReceiptV2 {
    fn validate_working_set(
        &self,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.binding_identity_sha256 == [0; 32] {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        self.sealed.validate_working_set(working_set)
    }

    fn validate_runtime_receipt(
        &self,
        runtime: &neoethos_gpu_cuda::resident_robust_normalization_v2::ResidentRobustNormalizationRuntimeReceiptV2,
    ) -> std::result::Result<[u8; 32], GpuOnlyFeatureMaterializationErrorV3> {
        let allocation = &self.sealed.allocation;
        let runtime_training_rows = runtime.training_rows();
        let fit_metadata_sha256 = runtime.fit_metadata_sha256();
        let event_token_matches_mode = if allocation.enabled {
            runtime.ready_event_process_token() != [0; 32] && fit_metadata_sha256 != [0; 32]
        } else {
            runtime.ready_event_process_token() == [0; 32]
                && fit_metadata_sha256 == resident_robust_normalization_disabled_fit_sha256_v2()
        };
        if runtime.semantic_version() != RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2
            || checked_u64(
                runtime.semantic_version() as usize,
                "normalization semantic version",
            )? != allocation.semantic_version
            || runtime.enabled() != allocation.enabled
            || checked_u64(runtime.row_count(), "normalization runtime rows")?
                != allocation.row_count
            || checked_u64(
                runtime.feature_column_count(),
                "normalization runtime feature columns",
            )? != allocation.feature_column_count
            || checked_u64(runtime_training_rows.start, "normalization training start")?
                != allocation.training_start
            || checked_u64(runtime_training_rows.end, "normalization training end")?
                != allocation.training_end
            || checked_u64(
                runtime.padded_training_rows(),
                "normalization padded training rows",
            )? != allocation.padded_training_rows
            || checked_u64(
                runtime.packed_validity_logical_bytes(),
                "normalization packed validity logical bytes",
            )? != allocation.packed_validity_logical_bytes
            || checked_u64(
                runtime.packed_validity_allocated_bytes(),
                "normalization packed validity allocated bytes",
            )? != allocation.packed_validity_allocated_bytes
            || allocation.packed_validity_allocated_bytes % 4 != 0
            || checked_u64(
                runtime.normalization_scratch_bytes(),
                "normalization runtime scratch bytes",
            )? != allocation.normalization_scratch_device_bytes
            || checked_u64(
                runtime.fit_metadata_bytes(),
                "normalization runtime fit metadata bytes",
            )? != allocation.fit_metadata_device_bytes
            || checked_u64(runtime.batch_count(), "normalization runtime batches")?
                != allocation.batch_count
            || checked_u64(
                runtime.native_launch_count(),
                "normalization runtime launches",
            )? != allocation.native_launch_count
            || checked_u64(
                runtime.producer_ready_event_count(),
                "normalization ready-event records",
            )? != allocation.producer_ready_event_count
            || checked_u64(
                runtime.producer_ready_event_synchronize_count(),
                "normalization ready-event synchronizations",
            )? != allocation.producer_ready_event_synchronize_count
            || checked_u64(
                runtime.control_error_device_bytes(),
                "normalization control-error device bytes",
            )? != allocation.control_error_device_bytes
            || checked_u64(
                runtime.control_error_readback_count(),
                "normalization control-error readbacks",
            )? != allocation.control_error_readback_count
            || checked_u64(
                runtime.control_error_d2h_bytes(),
                "normalization control-error D2H bytes",
            )? != allocation.control_error_d2h_bytes
            || checked_u64(
                runtime.fit_digest_readback_count(),
                "normalization fit-digest readbacks",
            )? != allocation.fit_digest_readback_count
            || checked_u64(
                runtime.fit_digest_d2h_bytes(),
                "normalization fit-digest D2H bytes",
            )? != allocation.fit_digest_d2h_bytes
            || checked_u64(
                runtime.parent_input_h2d_bytes(),
                "normalization parent input H2D bytes",
            )? != allocation.parent_input_h2d_bytes
            || checked_u64(
                runtime.feature_value_d2h_bytes(),
                "normalization feature-value D2H bytes",
            )? != allocation.feature_value_d2h_bytes
            || runtime.admission_identity_sha256() != self.admission_identity_sha256
            || runtime.primary_context_process_token() != self.primary_context_process_token
            || runtime.producer_stream_process_token() != self.producer_stream_process_token
            || !event_token_matches_mode
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(fit_metadata_sha256)
    }

    fn validate_runtime_evidence(
        &self,
        evidence: &ResidentFeatureLayoutEvidenceV3,
    ) -> std::result::Result<[u8; 32], GpuOnlyFeatureMaterializationErrorV3> {
        let runtime = evidence
            .robust_normalization_runtime_receipt_v2
            .as_ref()
            .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
        let exact_compact_d2h_bytes = evidence
            .validity_error_d2h_bytes
            .checked_add(evidence.canonical_root_d2h_bytes)
            .and_then(|bytes| bytes.checked_add(runtime.fit_digest_d2h_bytes()))
            .ok_or_else(|| {
                GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                    "resident compact control-plane D2H accounting overflowed"
                ))
            })?;
        if evidence.validity_error_readback_count != 1
            || evidence.validity_error_d2h_bytes != std::mem::size_of::<u32>()
            || evidence.compact_control_plane_d2h_bytes != exact_compact_d2h_bytes
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        self.validate_runtime_receipt(runtime)
    }

    fn apply(
        &self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        let runtime =
            assembler.apply_resident_robust_normalization_v2(&self.sealed.runtime_plan)?;
        self.validate_runtime_receipt(&runtime)?;
        Ok(())
    }
}

fn seal_robust_normalization_component_receipt_v2(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
    prepared: PreparedResidentRobustNormalizationInputV2,
) -> std::result::Result<
    SealedRobustNormalizationComponentReceiptV2,
    GpuOnlyFeatureMaterializationErrorV3,
> {
    if prepared.semantic_version() != RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2
        || prepared.row_count() != plan.row_count
        || prepared.feature_column_count() != plan.planned_routes.len()
        || plan.dataset_recipe_sha256 == [0; 32]
        || plan.feature_plan_schema_sha256 == [0; 32]
        || plan.route_plan_sha256 == [0; 32]
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let runtime_plan = ResidentRobustNormalizationPlanV2::preflight(
        prepared.row_count(),
        prepared.feature_column_count(),
        prepared.training_rows(),
        prepared.enabled(),
    )?;
    if runtime_plan.padded_training_rows() != prepared.padded_training_rows()
        || runtime_plan.normalization_scratch_bytes() != prepared.normalization_scratch_bytes()
        || runtime_plan.fit_metadata_bytes() != prepared.fit_metadata_bytes()
        || checked_u64(
            prepared.normalization_scratch_bytes(),
            "normalization scratch bytes",
        )? != plan.normalization_scratch_bytes
        || checked_u64(
            prepared.fit_metadata_bytes(),
            "normalization fit metadata bytes",
        )? != plan.fit_metadata_bytes
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let enabled = runtime_plan.enabled();
    let training_rows = runtime_plan.training_rows();
    let one_if_enabled = u64::from(enabled);
    let allocation = RobustNormalizationAllocationReceiptV2 {
        semantic_version: checked_u64(
            RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2 as usize,
            "normalization semantic version",
        )?,
        enabled,
        row_count: checked_u64(runtime_plan.rows(), "normalization rows")?,
        feature_column_count: checked_u64(runtime_plan.columns(), "normalization columns")?,
        training_start: checked_u64(training_rows.start, "normalization training start")?,
        training_end: checked_u64(training_rows.end, "normalization training end")?,
        padded_training_rows: checked_u64(
            runtime_plan.padded_training_rows(),
            "normalization padded training rows",
        )?,
        packed_validity_logical_bytes: checked_u64(
            runtime_plan.packed_validity_logical_bytes(),
            "normalization packed validity logical bytes",
        )?,
        packed_validity_allocated_bytes: checked_u64(
            runtime_plan.packed_validity_allocated_bytes(),
            "normalization packed validity allocated bytes",
        )?,
        normalization_scratch_device_bytes: checked_u64(
            runtime_plan.normalization_scratch_bytes(),
            "normalization scratch bytes",
        )?,
        fit_metadata_device_bytes: checked_u64(
            runtime_plan.fit_metadata_bytes(),
            "normalization fit metadata bytes",
        )?,
        batch_count: checked_u64(runtime_plan.batch_count(), "normalization batches")?,
        native_launch_count: checked_u64(
            runtime_plan.native_launch_count(),
            "normalization launches",
        )?,
        producer_ready_event_count: one_if_enabled,
        producer_ready_event_synchronize_count: one_if_enabled,
        control_error_device_bytes: one_if_enabled * std::mem::size_of::<u32>() as u64,
        control_error_readback_count: one_if_enabled,
        control_error_d2h_bytes: one_if_enabled * std::mem::size_of::<u32>() as u64,
        fit_digest_readback_count: one_if_enabled,
        fit_digest_d2h_bytes: one_if_enabled * 32,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
    };
    if allocation.packed_validity_allocated_bytes % 4 != 0 {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let lifetime = if enabled {
        RobustNormalizationLifetimeReceiptV2 {
            scratch:
                RobustNormalizationScratchLifetimeV2::ThroughReadyEventAndBoundedDigestReadback,
            fit: RobustNormalizationFitLifetimeV2::AlwaysResidentThroughSearchConsumerCompletion,
            event:
                RobustNormalizationEventLifetimeV2::AlwaysResidentThroughSearchConsumerCompletion,
        }
    } else {
        RobustNormalizationLifetimeReceiptV2 {
            scratch: RobustNormalizationScratchLifetimeV2::DisabledZeroExtent,
            fit: RobustNormalizationFitLifetimeV2::DisabledZeroExtent,
            event: RobustNormalizationEventLifetimeV2::DisabledNoEvent,
        }
    };
    let capability = resident_robust_normalization_capability_v2()?;
    if capability.producer() != ResidentFeatureProducerV3::RobustNormalization {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let mut identity = Sha256::new();
    identity.update(ROBUST_NORMALIZATION_COMPONENT_AUTHORITY_V2.as_bytes());
    identity.update(plan.dataset_recipe_sha256);
    identity.update(plan.feature_plan_schema_sha256);
    identity.update(plan.route_plan_sha256);
    identity.update(capability.implementation_sha256());
    identity.update(capability.exact_math_authority().as_bytes());
    identity.update(allocation.semantic_version.to_le_bytes());
    identity.update([u8::from(allocation.enabled)]);
    identity.update(allocation.row_count.to_le_bytes());
    identity.update(allocation.feature_column_count.to_le_bytes());
    identity.update(allocation.training_start.to_le_bytes());
    identity.update(allocation.training_end.to_le_bytes());
    identity.update(allocation.padded_training_rows.to_le_bytes());
    identity.update(allocation.packed_validity_logical_bytes.to_le_bytes());
    identity.update(allocation.packed_validity_allocated_bytes.to_le_bytes());
    identity.update(allocation.normalization_scratch_device_bytes.to_le_bytes());
    identity.update(allocation.fit_metadata_device_bytes.to_le_bytes());
    identity.update(allocation.batch_count.to_le_bytes());
    identity.update(allocation.native_launch_count.to_le_bytes());
    identity.update(allocation.producer_ready_event_count.to_le_bytes());
    identity.update(
        allocation
            .producer_ready_event_synchronize_count
            .to_le_bytes(),
    );
    identity.update(allocation.control_error_device_bytes.to_le_bytes());
    identity.update(allocation.control_error_readback_count.to_le_bytes());
    identity.update(allocation.control_error_d2h_bytes.to_le_bytes());
    identity.update(allocation.fit_digest_readback_count.to_le_bytes());
    identity.update(allocation.fit_digest_d2h_bytes.to_le_bytes());
    identity.update(allocation.parent_input_h2d_bytes.to_le_bytes());
    identity.update(allocation.feature_value_d2h_bytes.to_le_bytes());
    let component_identity_sha256 = identity.finalize().into();
    Ok(SealedRobustNormalizationComponentReceiptV2 {
        authority: ROBUST_NORMALIZATION_COMPONENT_AUTHORITY_V2,
        capability,
        allocation,
        lifetime,
        runtime_plan,
        component_identity_sha256,
    })
}

#[derive(Debug, PartialEq, Eq)]
enum FootprintScratchLifetimeV2 {
    ThroughProducerPackReadyEvent,
}

#[derive(Debug, PartialEq, Eq)]
enum FootprintOutputLifetimeV2 {
    ThroughProducerPackReadyEvent,
}

/// Exact output and prefix-scratch allocation frozen from the CPU semantic-v2
/// oracle and the crate-owned ordered producer batch ledger.
#[derive(Debug)]
pub(crate) struct FootprintAllocationReceiptV2 {
    row_count: u64,
    feature_column_count: u64,
    retained_feature_device_bytes: u64,
    prefix_scratch_device_bytes: u64,
    parent_input_h2d_bytes: u64,
    feature_value_d2h_bytes: u64,
    producer_ready_event_count: u64,
    native_launch_count: u64,
}

/// Move-only lifetime facts. Both allocations remain live until the generic
/// assembler's pack event proves the source batch can be retired.
#[derive(Debug)]
pub(crate) struct FootprintLifetimeReceiptV2 {
    scratch: FootprintScratchLifetimeV2,
    output: FootprintOutputLifetimeV2,
}

/// Data-owned Footprint component authority. Runtime evidence is accepted only
/// from the opaque gpu-cuda owner after its private same-carrier launch.
#[derive(Debug)]
pub(crate) struct SealedFootprintComponentReceiptV2 {
    authority: &'static str,
    capability: ResidentProducerCapabilityV3,
    allocation: FootprintAllocationReceiptV2,
    lifetime: FootprintLifetimeReceiptV2,
    component_identity_sha256: [u8; 32],
}

impl SealedFootprintComponentReceiptV2 {
    fn validate_working_set(
        &self,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.authority != FOOTPRINT_COMPONENT_AUTHORITY_V2
            || self.capability.producer() != ResidentFeatureProducerV3::Footprint
            || self.component_identity_sha256 == [0; 32]
            || self.lifetime.scratch != FootprintScratchLifetimeV2::ThroughProducerPackReadyEvent
            || self.lifetime.output != FootprintOutputLifetimeV2::ThroughProducerPackReadyEvent
            || working_set.row_count() != self.allocation.row_count
            || working_set.max_live_producer_bytes() < self.allocation.retained_feature_device_bytes
            || working_set.max_live_producer_scratch_bytes()
                < self.allocation.prefix_scratch_device_bytes
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }

    fn validate_runtime_evidence(
        &self,
        evidence: &ResidentFeatureLayoutEvidenceV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        let runtime: &neoethos_gpu_cuda::resident_footprint_v2::ResidentFootprintRuntimeReceiptV2 =
            evidence
                .footprint_runtime_receipt_v2
                .as_ref()
                .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
        if checked_u64(
            runtime.semantic_version() as usize,
            "Footprint semantic version",
        )? != FOOTPRINT_SEMANTIC_VERSION_V2 as u64
            || checked_u64(runtime.row_count(), "Footprint runtime rows")?
                != self.allocation.row_count
            || checked_u64(
                runtime.feature_column_count(),
                "Footprint runtime feature columns",
            )? != self.allocation.feature_column_count
            || checked_u64(
                runtime.retained_feature_device_bytes(),
                "Footprint runtime retained bytes",
            )? != self.allocation.retained_feature_device_bytes
            || checked_u64(
                runtime.prefix_scratch_device_bytes(),
                "Footprint runtime prefix scratch",
            )? != self.allocation.prefix_scratch_device_bytes
            || checked_u64(
                runtime.parent_input_h2d_bytes(),
                "Footprint runtime parent H2D bytes",
            )? != self.allocation.parent_input_h2d_bytes
            || checked_u64(
                runtime.feature_value_d2h_bytes(),
                "Footprint runtime feature D2H bytes",
            )? != self.allocation.feature_value_d2h_bytes
            || checked_u64(
                runtime.producer_ready_event_count(),
                "Footprint runtime ready events",
            )? != self.allocation.producer_ready_event_count
            || checked_u64(
                runtime.native_launch_count(),
                "Footprint runtime native launches",
            )? != self.allocation.native_launch_count
            || runtime.logical_validity_schema()
                != "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3"
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }
}

fn seal_footprint_component_receipt_v2(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
) -> std::result::Result<SealedFootprintComponentReceiptV2, GpuOnlyFeatureMaterializationErrorV3> {
    let mut batches = plan
        .producer_batches
        .iter()
        .filter(|batch| batch.producer == ResidentFeatureProducerV3::Footprint);
    let batch = batches.next().ok_or(
        GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
            component: "exact resident Footprint batch ledger",
        },
    )?;
    if batches.next().is_some()
        || batch.column_count != RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len()
        || batch.additional_retained_bytes != 0
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let route_end = batch
        .first_column
        .checked_add(batch.column_count)
        .ok_or_else(|| {
            GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                "Footprint route extent overflowed"
            ))
        })?;
    let routes = plan
        .planned_routes
        .get(batch.first_column..route_end)
        .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
    if routes
        .iter()
        .zip(RESIDENT_FOOTPRINT_COLUMN_NAMES_V2)
        .any(|(route, expected_name)| {
            route.producer() != ResidentFeatureProducerV3::Footprint
                || route.feature_name() != expected_name
        })
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }

    let row_count = checked_u64(plan.row_count, "Footprint rows")?;
    let feature_column_count = checked_u64(
        RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len(),
        "Footprint columns",
    )?;
    let feature_cells = row_count.checked_mul(feature_column_count).ok_or_else(|| {
        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
            "Footprint feature cells overflowed"
        ))
    })?;
    let retained_feature_device_bytes = feature_cells.checked_mul(9).ok_or_else(|| {
        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
            "Footprint retained bytes overflowed"
        ))
    })?;
    let prefix_scratch_device_bytes = row_count
        .checked_add(1)
        .and_then(|extent| extent.checked_mul(8))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| {
            GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                "Footprint prefix scratch bytes overflowed"
            ))
        })?;
    if batch.scratch_bytes != prefix_scratch_device_bytes {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let capability = resident_footprint_capability_v2()?;
    if capability.producer() != ResidentFeatureProducerV3::Footprint {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let allocation = FootprintAllocationReceiptV2 {
        row_count,
        feature_column_count,
        retained_feature_device_bytes,
        prefix_scratch_device_bytes,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        producer_ready_event_count: 1,
        native_launch_count: 2,
    };
    let lifetime = FootprintLifetimeReceiptV2 {
        scratch: FootprintScratchLifetimeV2::ThroughProducerPackReadyEvent,
        output: FootprintOutputLifetimeV2::ThroughProducerPackReadyEvent,
    };
    let mut identity = Sha256::new();
    identity.update(FOOTPRINT_COMPONENT_AUTHORITY_V2.as_bytes());
    identity.update(plan.dataset_recipe_sha256);
    identity.update(plan.feature_plan_schema_sha256);
    identity.update(plan.route_plan_sha256);
    identity.update(capability.implementation_sha256());
    identity.update(capability.exact_math_authority().as_bytes());
    identity.update(allocation.row_count.to_le_bytes());
    identity.update(allocation.feature_column_count.to_le_bytes());
    identity.update(allocation.retained_feature_device_bytes.to_le_bytes());
    identity.update(allocation.prefix_scratch_device_bytes.to_le_bytes());
    identity.update(allocation.parent_input_h2d_bytes.to_le_bytes());
    identity.update(allocation.feature_value_d2h_bytes.to_le_bytes());
    identity.update(allocation.producer_ready_event_count.to_le_bytes());
    identity.update(allocation.native_launch_count.to_le_bytes());
    let component_identity_sha256 = identity.finalize().into();
    Ok(SealedFootprintComponentReceiptV2 {
        authority: FOOTPRINT_COMPONENT_AUTHORITY_V2,
        capability,
        allocation,
        lifetime,
        component_identity_sha256,
    })
}

#[derive(Debug, PartialEq, Eq)]
enum RegimeOutputLifetimeV3 {
    ThroughProducerPackReadyEvent,
}

/// Exact 126N output and zero-scratch authority for one Regime-v3 batch.
#[derive(Debug)]
pub(crate) struct RegimeAllocationReceiptV3 {
    row_count: u64,
    feature_column_count: u64,
    scale_anchor_bits: u64,
    retained_feature_device_bytes: u64,
    additional_retained_device_bytes: u64,
    scratch_device_bytes: u64,
    pointer_table_device_bytes: u64,
    isolated_pointer_schema_metadata_bytes: u64,
    parent_input_h2d_bytes: u64,
    feature_value_d2h_bytes: u64,
    producer_ready_event_count: u64,
    native_launch_count: u64,
}

/// Data-owned, non-caller-mintable and move-only Regime component receipt.
#[derive(Debug)]
pub(crate) struct SealedRegimeComponentReceiptV3 {
    authority: &'static str,
    capability: ResidentProducerCapabilityV3,
    allocation: RegimeAllocationReceiptV3,
    lifetime: RegimeOutputLifetimeV3,
    input_identity_sha256: [u8; 32],
    component_identity_sha256: [u8; 32],
}

impl SealedRegimeComponentReceiptV3 {
    fn validate_working_set(
        &self,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.authority != REGIME_COMPONENT_AUTHORITY_V3
            || self.capability.producer() != ResidentFeatureProducerV3::Regime
            || self.input_identity_sha256 == [0; 32]
            || self.component_identity_sha256 == [0; 32]
            || self.lifetime != RegimeOutputLifetimeV3::ThroughProducerPackReadyEvent
            || working_set.row_count() != self.allocation.row_count
            || working_set.max_live_producer_bytes() < self.allocation.retained_feature_device_bytes
            || working_set.pointer_and_schema_metadata_bytes()
                < self.allocation.isolated_pointer_schema_metadata_bytes
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }

    fn validate_runtime_evidence(
        &self,
        evidence: &ResidentFeatureLayoutEvidenceV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        let runtime: &neoethos_gpu_cuda::resident_regime_v3::ResidentRegimeRuntimeReceiptV3 =
            evidence
                .regime_runtime_receipt_v3
                .as_ref()
                .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
        if checked_u64(
            runtime.semantic_version() as usize,
            "Regime semantic version",
        )? != REGIME_SEMANTIC_VERSION as u64
            || runtime.semantic_version() != RESIDENT_REGIME_SEMANTIC_VERSION_V3
            || checked_u64(runtime.row_count(), "Regime runtime rows")? != self.allocation.row_count
            || checked_u64(runtime.feature_column_count(), "Regime runtime columns")?
                != self.allocation.feature_column_count
            || runtime.scale_anchor_bits() != self.allocation.scale_anchor_bits
            || checked_u64(
                runtime.retained_feature_device_bytes(),
                "Regime runtime retained bytes",
            )? != self.allocation.retained_feature_device_bytes
            || checked_u64(
                runtime.additional_retained_device_bytes(),
                "Regime runtime additional retained bytes",
            )? != self.allocation.additional_retained_device_bytes
            || checked_u64(
                runtime.scratch_device_bytes(),
                "Regime runtime scratch bytes",
            )? != self.allocation.scratch_device_bytes
            || checked_u64(
                runtime.pointer_table_device_bytes(),
                "Regime runtime pointer table bytes",
            )? != self.allocation.pointer_table_device_bytes
            || checked_u64(
                runtime.isolated_pointer_schema_metadata_bytes(),
                "Regime runtime isolated pointer/schema bytes",
            )? != self.allocation.isolated_pointer_schema_metadata_bytes
            || checked_u64(
                runtime.parent_input_h2d_bytes(),
                "Regime runtime parent H2D bytes",
            )? != self.allocation.parent_input_h2d_bytes
            || checked_u64(
                runtime.feature_value_d2h_bytes(),
                "Regime runtime feature D2H bytes",
            )? != self.allocation.feature_value_d2h_bytes
            || checked_u64(
                runtime.producer_ready_event_count(),
                "Regime runtime ready events",
            )? != self.allocation.producer_ready_event_count
            || checked_u64(
                runtime.native_launch_count(),
                "Regime runtime native launches",
            )? != self.allocation.native_launch_count
            || runtime.logical_validity_schema()
                != "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3"
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }
}

fn seal_regime_component_receipt_v3(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
) -> std::result::Result<SealedRegimeComponentReceiptV3, GpuOnlyFeatureMaterializationErrorV3> {
    let mut batches = plan
        .producer_batches
        .iter()
        .filter(|batch| batch.producer == ResidentFeatureProducerV3::Regime);
    let batch = batches.next().ok_or(
        GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
            component: "exact resident Regime batch ledger",
        },
    )?;
    if batches.next().is_some()
        || batch.column_count != RESIDENT_REGIME_COLUMN_NAMES_V3.len()
        || batch.additional_retained_bytes != 0
        || batch.scratch_bytes != 0
        || plan.regime_scale_anchor_bits == 0
        || plan.regime_input_identity_sha256 == [0; 32]
        || REGIME_SEMANTIC_VERSION != 3
        || !REGIME_V2_ARTIFACT_MIGRATION_POLICY.contains("refuse semantic-v2 Regime artifacts")
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let route_end = batch
        .first_column
        .checked_add(batch.column_count)
        .ok_or_else(|| {
            GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                "Regime route extent overflowed"
            ))
        })?;
    let routes = plan
        .planned_routes
        .get(batch.first_column..route_end)
        .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
    if routes
        .iter()
        .zip(RESIDENT_REGIME_COLUMN_NAMES_V3)
        .any(|(route, expected_name)| {
            route.producer() != ResidentFeatureProducerV3::Regime
                || route.feature_name() != expected_name
        })
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }

    let row_count = checked_u64(plan.row_count, "Regime rows")?;
    let feature_column_count =
        checked_u64(RESIDENT_REGIME_COLUMN_NAMES_V3.len(), "Regime columns")?;
    let retained_feature_device_bytes = row_count.checked_mul(126).ok_or_else(|| {
        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
            "Regime 126N retained bytes overflowed"
        ))
    })?;
    let allocation = RegimeAllocationReceiptV3 {
        row_count,
        feature_column_count,
        scale_anchor_bits: plan.regime_scale_anchor_bits,
        retained_feature_device_bytes,
        additional_retained_device_bytes: 0,
        scratch_device_bytes: 0,
        pointer_table_device_bytes: 448,
        isolated_pointer_schema_metadata_bytes: 1_235,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        producer_ready_event_count: 1,
        native_launch_count: 2,
    };
    let capability = resident_regime_capability_v3()?;
    if capability.producer() != ResidentFeatureProducerV3::Regime {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let mut identity = Sha256::new();
    identity.update(REGIME_COMPONENT_AUTHORITY_V3.as_bytes());
    identity.update(REGIME_V2_ARTIFACT_MIGRATION_POLICY.as_bytes());
    identity.update(plan.dataset_recipe_sha256);
    identity.update(plan.feature_plan_schema_sha256);
    identity.update(plan.route_plan_sha256);
    identity.update(plan.regime_input_identity_sha256);
    identity.update(capability.implementation_sha256());
    identity.update(capability.exact_math_authority().as_bytes());
    identity.update(allocation.row_count.to_le_bytes());
    identity.update(allocation.feature_column_count.to_le_bytes());
    identity.update(allocation.scale_anchor_bits.to_le_bytes());
    identity.update(allocation.retained_feature_device_bytes.to_le_bytes());
    identity.update(allocation.additional_retained_device_bytes.to_le_bytes());
    identity.update(allocation.scratch_device_bytes.to_le_bytes());
    identity.update(allocation.pointer_table_device_bytes.to_le_bytes());
    identity.update(
        allocation
            .isolated_pointer_schema_metadata_bytes
            .to_le_bytes(),
    );
    identity.update(allocation.parent_input_h2d_bytes.to_le_bytes());
    identity.update(allocation.feature_value_d2h_bytes.to_le_bytes());
    identity.update(allocation.producer_ready_event_count.to_le_bytes());
    identity.update(allocation.native_launch_count.to_le_bytes());
    Ok(SealedRegimeComponentReceiptV3 {
        authority: REGIME_COMPONENT_AUTHORITY_V3,
        capability,
        allocation,
        lifetime: RegimeOutputLifetimeV3::ThroughProducerPackReadyEvent,
        input_identity_sha256: plan.regime_input_identity_sha256,
        component_identity_sha256: identity.finalize().into(),
    })
}

#[derive(Debug, PartialEq, Eq)]
enum FeatureMajorToBarMajorLifetimeV3 {
    AlwaysResidentThroughSearchConsumerCompletion,
}

/// Exact allocation contribution of the in-tree tiled f64/u4 layout producer.
/// Fields remain private so byte counts cannot be supplied as component
/// authority by App, CLI, Search, tests, or serialized DTOs.
#[derive(Debug)]
pub(crate) struct FeatureMajorToBarMajorAllocationReceiptV3 {
    row_count: u64,
    column_count: u64,
    final_bar_major_value_bytes: u64,
    packed_validity_logical_bytes: u64,
    packed_validity_allocated_bytes: u64,
    expected_pack_launch_count: u64,
    full_feature_major_staging_bytes: u64,
    always_resident_device_bytes: u64,
}

/// Typed lifetime fact paired with the exact allocation receipt. Runtime must
/// still prove the final ready-event semantics before the component can seal.
#[derive(Debug)]
pub(crate) struct FeatureMajorToBarMajorLifetimeReceiptV3 {
    lifetime: FeatureMajorToBarMajorLifetimeV3,
    final_ready_event_record_count: u64,
}

/// Move-only Data receipt for exactly one already-real resident producer. It
/// is not the complete resident-feature-store workspace component: the other
/// producer and Data identity receipts remain mandatory.
#[derive(Debug)]
pub(crate) struct SealedFeatureMajorToBarMajorComponentReceiptV3 {
    authority: &'static str,
    capability: ResidentProducerCapabilityV3,
    allocation: FeatureMajorToBarMajorAllocationReceiptV3,
    lifetime: FeatureMajorToBarMajorLifetimeReceiptV3,
    component_identity_sha256: [u8; 32],
}

impl SealedFeatureMajorToBarMajorComponentReceiptV3 {
    fn validate_working_set(
        &self,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.authority != FEATURE_MAJOR_TO_BAR_MAJOR_COMPONENT_AUTHORITY_V3
            || self.capability.producer() != ResidentFeatureProducerV3::FeatureMajorToBarMajor
            || self.component_identity_sha256 == [0; 32]
            || self.lifetime.lifetime
                != FeatureMajorToBarMajorLifetimeV3::AlwaysResidentThroughSearchConsumerCompletion
            || self.lifetime.final_ready_event_record_count != 1
            || working_set.row_count() != self.allocation.row_count
            || working_set.column_count() != self.allocation.column_count
            || working_set.final_bar_major_value_bytes()
                != self.allocation.final_bar_major_value_bytes
            || working_set.packed_validity_logical_bytes()
                != self.allocation.packed_validity_logical_bytes
            || working_set.packed_validity_allocated_bytes()
                != self.allocation.packed_validity_allocated_bytes
            || working_set.full_feature_major_staging_bytes()
                != self.allocation.full_feature_major_staging_bytes
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }

    fn validate_runtime_evidence(
        &self,
        evidence: &ResidentFeatureLayoutEvidenceV3,
        ready_event: &ResidentReadyEventV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        let rows = checked_u64(evidence.rows, "layout receipt rows")?;
        let columns = checked_u64(evidence.columns, "layout receipt columns")?;
        let producer_batch_count = checked_u64(
            evidence.producer_batch_count,
            "layout receipt producer batches",
        )?;
        let value_layout_launch_count = checked_u64(
            evidence.value_layout_launch_count,
            "layout receipt value launches",
        )?;
        let validity_boundary_launch_count = checked_u64(
            evidence.validity_boundary_launch_count,
            "layout receipt validity launches",
        )?;
        let value_bytes = checked_u64(
            evidence.layout_transform_value_bytes,
            "layout receipt value bytes",
        )?;
        let logical_validity_bytes = checked_u64(
            evidence.layout_transform_logical_validity_bytes,
            "layout receipt logical validity bytes",
        )?;
        let packed_validity_logical_bytes = checked_u64(
            evidence.packed_validity_logical_bytes,
            "layout receipt packed logical validity bytes",
        )?;
        let packed_validity_allocated_bytes = checked_u64(
            evidence.packed_validity_allocated_bytes,
            "layout receipt packed validity allocation",
        )?;
        let full_feature_major_staging_bytes = checked_u64(
            evidence.full_feature_major_staging_bytes,
            "layout receipt feature-major staging bytes",
        )?;
        if rows != self.allocation.row_count
            || columns != self.allocation.column_count
            || evidence.source_column_count != evidence.columns
            || producer_batch_count != self.allocation.expected_pack_launch_count
            || value_layout_launch_count != self.allocation.expected_pack_launch_count
            || validity_boundary_launch_count != self.allocation.expected_pack_launch_count
            || value_bytes != self.allocation.final_bar_major_value_bytes
            || logical_validity_bytes
                != self
                    .allocation
                    .row_count
                    .checked_mul(self.allocation.column_count)
                    .ok_or_else(|| {
                        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                            "layout receipt logical validity extent overflowed"
                        ))
                    })?
            || packed_validity_logical_bytes != self.allocation.packed_validity_logical_bytes
            || packed_validity_allocated_bytes != self.allocation.packed_validity_allocated_bytes
            || full_feature_major_staging_bytes != self.allocation.full_feature_major_staging_bytes
            || !ready_event.recorded_after_final_incremental_layout_normalization_and_merkle()
            || !ready_event.consumer_must_wait_before_first_read()
            || !ready_event.retains_store_until_consumer_completion()
            || ready_event.host_synchronize_count() != 0
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }
}

fn seal_feature_major_to_bar_major_component_receipt_v3(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
) -> std::result::Result<
    SealedFeatureMajorToBarMajorComponentReceiptV3,
    GpuOnlyFeatureMaterializationErrorV3,
> {
    if plan.row_count == 0
        || plan.planned_routes.is_empty()
        || plan.producer_batches.is_empty()
        || plan.dataset_recipe_sha256 == [0; 32]
        || plan.feature_plan_schema_sha256 == [0; 32]
        || plan.route_plan_sha256 == [0; 32]
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!("feature-major layout component requires one exact resolved Data plan"),
        ));
    }
    let mut next_column = 0_usize;
    for batch in &plan.producer_batches {
        if batch.first_column != next_column || batch.column_count == 0 || batch.column_count > 64 {
            return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
                anyhow::anyhow!(
                    "feature-major layout batches must be exact monotonic ranges of at most 64 columns"
                ),
            ));
        }
        let end = batch
            .first_column
            .checked_add(batch.column_count)
            .ok_or_else(|| {
                GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                    "feature-major layout batch range overflowed"
                ))
            })?;
        if end > plan.planned_routes.len()
            || plan.planned_routes[batch.first_column..end]
                .iter()
                .any(|route| route.producer() != batch.producer)
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
                anyhow::anyhow!(
                    "feature-major layout batch ledger differs from the resolved route plan"
                ),
            ));
        }
        next_column = end;
    }
    if next_column != plan.planned_routes.len() {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!(
                "feature-major layout batch ledger does not cover every resolved route"
            ),
        ));
    }

    let row_count = checked_u64(plan.row_count, "feature-major layout rows")?;
    let column_count = checked_u64(plan.planned_routes.len(), "feature-major layout columns")?;
    let cells = row_count.checked_mul(column_count).ok_or_else(|| {
        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
            "feature-major layout cell count overflowed"
        ))
    })?;
    let final_bar_major_value_bytes = cells.checked_mul(8).ok_or_else(|| {
        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
            "feature-major layout value bytes overflowed"
        ))
    })?;
    let packed_validity_logical_bytes = cells.div_ceil(2);
    let packed_validity_allocated_bytes =
        packed_validity_logical_bytes
            .checked_add(3)
            .ok_or_else(|| {
                GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                    "feature-major packed validity alignment overflowed"
                ))
            })?
            / 4
            * 4;
    let always_resident_device_bytes = final_bar_major_value_bytes
        .checked_add(packed_validity_allocated_bytes)
        .ok_or_else(|| {
            GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                "feature-major always-resident bytes overflowed"
            ))
        })?;
    let expected_pack_launch_count = checked_u64(
        plan.producer_batches.len(),
        "feature-major layout pack launches",
    )?;
    let capability = resident_feature_major_to_bar_major_capability_v3()?;
    if capability.producer() != ResidentFeatureProducerV3::FeatureMajorToBarMajor {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let allocation = FeatureMajorToBarMajorAllocationReceiptV3 {
        row_count,
        column_count,
        final_bar_major_value_bytes,
        packed_validity_logical_bytes,
        packed_validity_allocated_bytes,
        expected_pack_launch_count,
        full_feature_major_staging_bytes: 0,
        always_resident_device_bytes,
    };
    let lifetime = FeatureMajorToBarMajorLifetimeReceiptV3 {
        lifetime: FeatureMajorToBarMajorLifetimeV3::AlwaysResidentThroughSearchConsumerCompletion,
        final_ready_event_record_count: 1,
    };
    let mut identity = Sha256::new();
    identity.update(FEATURE_MAJOR_TO_BAR_MAJOR_COMPONENT_AUTHORITY_V3.as_bytes());
    identity.update(plan.dataset_recipe_sha256);
    identity.update(plan.feature_plan_schema_sha256);
    identity.update(plan.route_plan_sha256);
    identity.update(capability.implementation_sha256());
    identity.update(capability.exact_math_authority().as_bytes());
    identity.update(allocation.row_count.to_le_bytes());
    identity.update(allocation.column_count.to_le_bytes());
    identity.update(allocation.final_bar_major_value_bytes.to_le_bytes());
    identity.update(allocation.packed_validity_logical_bytes.to_le_bytes());
    identity.update(allocation.packed_validity_allocated_bytes.to_le_bytes());
    identity.update(allocation.expected_pack_launch_count.to_le_bytes());
    identity.update(allocation.full_feature_major_staging_bytes.to_le_bytes());
    identity.update(allocation.always_resident_device_bytes.to_le_bytes());
    identity.update(lifetime.final_ready_event_record_count.to_le_bytes());
    identity.update(b"always-resident-through-search-consumer-completion");
    let component_identity_sha256 = identity.finalize().into();
    Ok(SealedFeatureMajorToBarMajorComponentReceiptV3 {
        authority: FEATURE_MAJOR_TO_BAR_MAJOR_COMPONENT_AUTHORITY_V3,
        capability,
        allocation,
        lifetime,
        component_identity_sha256,
    })
}

#[derive(Debug, PartialEq, Eq)]
enum CanonicalContentSha256ScratchLifetimeV3 {
    ThroughFinalReadyAndCompactRootReadback,
}

#[derive(Debug, PartialEq, Eq)]
enum CanonicalContentSha256RootLifetimeV3 {
    AlwaysResidentThroughSearchConsumerCompletion,
}

/// Exact root and two-level scratch allocation contribution of the portable
/// in-tree CUDA Merkle producer. Compact host readback bytes are recorded
/// separately and never authorize a device allocation.
#[derive(Debug)]
pub(crate) struct CanonicalContentSha256AllocationReceiptV3 {
    row_count: u64,
    column_count: u64,
    merkle_leaf_count: u64,
    merkle_scratch_bytes: u64,
    canonical_root_device_bytes: u64,
    canonical_root_readback_count: u64,
    canonical_root_d2h_bytes: u64,
}

/// Typed lifetime policy for the transient scratch/name tables and retained
/// canonical root. Runtime still must prove the final event and one compact
/// root readback before the scratch owner can retire.
#[derive(Debug)]
pub(crate) struct CanonicalContentSha256LifetimeReceiptV3 {
    scratch: CanonicalContentSha256ScratchLifetimeV3,
    root: CanonicalContentSha256RootLifetimeV3,
    final_ready_event_record_count: u64,
}

/// Move-only Data receipt for the real canonical CUDA hash producer. This is
/// one constituent of the future complete feature workspace component, never
/// a substitute for the remaining producer or Data identity receipts.
#[derive(Debug)]
pub(crate) struct SealedCanonicalContentSha256ComponentReceiptV3 {
    authority: &'static str,
    capability: ResidentProducerCapabilityV3,
    allocation: CanonicalContentSha256AllocationReceiptV3,
    lifetime: CanonicalContentSha256LifetimeReceiptV3,
    component_identity_sha256: [u8; 32],
}

impl SealedCanonicalContentSha256ComponentReceiptV3 {
    fn validate_working_set(
        &self,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.authority != CANONICAL_CONTENT_SHA256_COMPONENT_AUTHORITY_V3
            || self.capability.producer() != ResidentFeatureProducerV3::CanonicalContentSha256
            || self.component_identity_sha256 == [0; 32]
            || self.lifetime.scratch
                != CanonicalContentSha256ScratchLifetimeV3::ThroughFinalReadyAndCompactRootReadback
            || self.lifetime.root
                != CanonicalContentSha256RootLifetimeV3::AlwaysResidentThroughSearchConsumerCompletion
            || self.lifetime.final_ready_event_record_count != 1
            || working_set.row_count() != self.allocation.row_count
            || working_set.column_count() != self.allocation.column_count
            || working_set.merkle_leaf_count() != self.allocation.merkle_leaf_count
            || working_set.merkle_scratch_bytes() != self.allocation.merkle_scratch_bytes
            || working_set.canonical_root_bytes() != self.allocation.canonical_root_device_bytes
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }

    fn validate_runtime_evidence(
        &self,
        evidence: &ResidentFeatureLayoutEvidenceV3,
        ready_event: &ResidentReadyEventV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if checked_u64(evidence.rows, "canonical receipt rows")? != self.allocation.row_count
            || checked_u64(evidence.columns, "canonical receipt columns")?
                != self.allocation.column_count
            || checked_u64(
                evidence.merkle_leaf_count,
                "canonical receipt Merkle leaves",
            )? != self.allocation.merkle_leaf_count
            || checked_u64(
                evidence.merkle_scratch_bytes,
                "canonical receipt Merkle scratch bytes",
            )? != self.allocation.merkle_scratch_bytes
            || checked_u64(
                evidence.canonical_root_device_bytes,
                "canonical receipt root device bytes",
            )? != self.allocation.canonical_root_device_bytes
            || checked_u64(
                evidence.canonical_root_readback_count,
                "canonical receipt root readback count",
            )? != self.allocation.canonical_root_readback_count
            || checked_u64(
                evidence.canonical_root_d2h_bytes,
                "canonical receipt root D2H bytes",
            )? != self.allocation.canonical_root_d2h_bytes
            || evidence.canonical_content_merkle == [0; 32]
            || !ready_event.recorded_after_final_incremental_layout_normalization_and_merkle()
            || !ready_event.consumer_must_wait_before_first_read()
            || !ready_event.retains_store_until_consumer_completion()
            || ready_event.host_synchronize_count() != 0
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }
}

fn seal_canonical_content_sha256_component_receipt_v3(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
) -> std::result::Result<
    SealedCanonicalContentSha256ComponentReceiptV3,
    GpuOnlyFeatureMaterializationErrorV3,
> {
    if plan.row_count == 0
        || plan.planned_routes.is_empty()
        || plan.dataset_recipe_sha256 == [0; 32]
        || plan.feature_plan_schema_sha256 == [0; 32]
        || plan.route_plan_sha256 == [0; 32]
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!("canonical hash component requires one exact resolved Data plan"),
        ));
    }
    let row_count = checked_u64(plan.row_count, "canonical hash rows")?;
    let column_count = checked_u64(plan.planned_routes.len(), "canonical hash columns")?;
    let chunk_rows = checked_u64(
        CANONICAL_MERKLE_CHUNK_ROWS_V3,
        "canonical Merkle chunk rows",
    )?;
    let timestamp_chunk_count = row_count.div_ceil(chunk_rows);
    let producer_count = column_count.checked_add(1).ok_or_else(|| {
        GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
            "canonical Merkle producer count overflowed"
        ))
    })?;
    let merkle_leaf_count = timestamp_chunk_count
        .checked_mul(producer_count)
        .ok_or_else(|| {
            GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                "canonical Merkle leaf count overflowed"
            ))
        })?;
    let merkle_scratch_bytes = merkle_leaf_count
        .checked_mul(32)
        .and_then(|one_level| one_level.checked_mul(2))
        .ok_or_else(|| {
            GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                "canonical Merkle scratch bytes overflowed"
            ))
        })?;
    let capability = resident_canonical_content_sha256_capability_v3()?;
    if capability.producer() != ResidentFeatureProducerV3::CanonicalContentSha256 {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let allocation = CanonicalContentSha256AllocationReceiptV3 {
        row_count,
        column_count,
        merkle_leaf_count,
        merkle_scratch_bytes,
        canonical_root_device_bytes: 32,
        canonical_root_readback_count: 1,
        canonical_root_d2h_bytes: 32,
    };
    let lifetime = CanonicalContentSha256LifetimeReceiptV3 {
        scratch: CanonicalContentSha256ScratchLifetimeV3::ThroughFinalReadyAndCompactRootReadback,
        root: CanonicalContentSha256RootLifetimeV3::AlwaysResidentThroughSearchConsumerCompletion,
        final_ready_event_record_count: 1,
    };
    let mut identity = Sha256::new();
    identity.update(CANONICAL_CONTENT_SHA256_COMPONENT_AUTHORITY_V3.as_bytes());
    identity.update(plan.dataset_recipe_sha256);
    identity.update(plan.feature_plan_schema_sha256);
    identity.update(plan.route_plan_sha256);
    identity.update(capability.implementation_sha256());
    identity.update(capability.exact_math_authority().as_bytes());
    identity.update(allocation.row_count.to_le_bytes());
    identity.update(allocation.column_count.to_le_bytes());
    identity.update(allocation.merkle_leaf_count.to_le_bytes());
    identity.update(allocation.merkle_scratch_bytes.to_le_bytes());
    identity.update(allocation.canonical_root_device_bytes.to_le_bytes());
    identity.update(allocation.canonical_root_readback_count.to_le_bytes());
    identity.update(allocation.canonical_root_d2h_bytes.to_le_bytes());
    identity.update(lifetime.final_ready_event_record_count.to_le_bytes());
    identity.update(b"scratch-through-final-ready-and-compact-root-readback");
    identity.update(b"root-always-resident-through-search-consumer-completion");
    let component_identity_sha256 = identity.finalize().into();
    Ok(SealedCanonicalContentSha256ComponentReceiptV3 {
        authority: CANONICAL_CONTENT_SHA256_COMPONENT_AUTHORITY_V3,
        capability,
        allocation,
        lifetime,
        component_identity_sha256,
    })
}

fn checked_u64(value: usize, field: &'static str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{field} exceeds the V3 receipt width"))
}

fn exact_resident_working_set_extent_request_v3(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
) -> Result<ResidentWorkingSetExtentRequestV3> {
    if plan.row_count == 0 || plan.planned_routes.is_empty() {
        bail!("strict resident recipe rows and routes must be nonempty")
    }
    let rows = checked_u64(plan.row_count, "resident row count")?;
    let mut next_column = 0_usize;
    let mut max_live_producer_bytes = 0_u64;
    let mut max_live_producer_scratch_bytes = 0_u64;
    let mut max_pointer_table_bytes = 0_u64;
    for batch in &plan.producer_batches {
        if batch.first_column != next_column || batch.column_count == 0 || batch.column_count > 64 {
            bail!("producer batches must cover monotonic nonempty ranges of at most 64 columns")
        }
        let end = batch
            .first_column
            .checked_add(batch.column_count)
            .context("producer batch column range overflow")?;
        if end > plan.planned_routes.len()
            || plan.planned_routes[batch.first_column..end]
                .iter()
                .any(|route| route.producer() != batch.producer)
        {
            bail!("producer batch schema does not match the exact admitted route range")
        }
        let columns = checked_u64(batch.column_count, "producer batch columns")?;
        let cells = rows
            .checked_mul(columns)
            .context("producer batch cell count overflow")?;
        let value_and_logical_validity_bytes = cells
            .checked_mul(9)
            .context("producer batch value/validity byte count overflow")?;
        let exact_retained_bytes = value_and_logical_validity_bytes
            .checked_add(batch.additional_retained_bytes)
            .context("producer batch retained byte count overflow")?;
        max_live_producer_bytes = max_live_producer_bytes.max(exact_retained_bytes);
        max_live_producer_scratch_bytes = max_live_producer_scratch_bytes.max(batch.scratch_bytes);
        max_pointer_table_bytes = max_pointer_table_bytes.max(
            columns
                .checked_mul(4 * u64::BITS as u64 / 8)
                .context("producer pointer-table byte count overflow")?,
        );
        next_column = end;
    }
    if next_column != plan.planned_routes.len() {
        bail!("producer batches leave an admitted feature gap")
    }
    let name_offsets_bytes = checked_u64(plan.planned_routes.len() + 1, "feature name offsets")?
        .checked_mul(u64::BITS as u64 / 8)
        .context("feature name-offset byte count overflow")?;
    let name_bytes = plan.planned_routes.iter().try_fold(0_u64, |sum, route| {
        sum.checked_add(checked_u64(
            route.feature_name().len(),
            "feature name bytes",
        )?)
        .context("ordered feature-name byte count overflow")
    })?;
    let pointer_and_schema_metadata_bytes = max_pointer_table_bytes
        .checked_add(name_offsets_bytes)
        .and_then(|bytes| bytes.checked_add(name_bytes))
        .context("pointer/schema metadata byte count overflow")?;

    Ok(ResidentWorkingSetExtentRequestV3 {
        row_count: plan.row_count,
        column_count: plan.planned_routes.len(),
        max_live_producer_bytes,
        max_live_producer_scratch_bytes,
        normalization_scratch_bytes: plan.normalization_scratch_bytes,
        fit_metadata_bytes: plan.fit_metadata_bytes,
        pointer_and_schema_metadata_bytes,
    })
}

fn derive_exact_resident_working_set_v3(
    plan: &ResolvedGpuOnlyFeatureMaterializationPlanV3,
    phase_one_free_bytes_snapshot: u64,
    allocator_context_reserve_bytes: u64,
) -> Result<neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetBoundV3> {
    let extent = exact_resident_working_set_extent_request_v3(plan)?;
    ResidentWorkingSetRequestV3 {
        row_count: extent.row_count,
        column_count: extent.column_count,
        max_live_producer_bytes: extent.max_live_producer_bytes,
        max_live_producer_scratch_bytes: extent.max_live_producer_scratch_bytes,
        normalization_scratch_bytes: extent.normalization_scratch_bytes,
        fit_metadata_bytes: extent.fit_metadata_bytes,
        pointer_and_schema_metadata_bytes: extent.pointer_and_schema_metadata_bytes,
        device_free_bytes_snapshot: phase_one_free_bytes_snapshot,
        allocator_context_reserve_bytes,
        reserve_policy_id: EXACT_ALLOCATOR_RESERVE_POLICY_V3.into(),
    }
    .seal()
    .map_err(Into::into)
}

/// Opaque phase-one authority. App/CLI/Search cannot construct this from a
/// low-level DTO, a caller-provided ordinal, or a caller capability boolean.
#[derive(Debug)]
pub struct GpuOnlyFeatureMaterializationAdmissionV3 {
    authority: &'static str,
    contract: GpuOnlyResidentAdmissionV3,
    run_device: GpuOnlyRunDeviceAdmissionV3,
    footprint: SealedFootprintComponentReceiptV2,
    regime: SealedRegimeComponentReceiptV3,
    robust_normalization: BoundRobustNormalizationComponentReceiptV2,
    feature_major_to_bar_major: SealedFeatureMajorToBarMajorComponentReceiptV3,
    canonical_content_sha256: SealedCanonicalContentSha256ComponentReceiptV3,
    feature_identity: ResidentFeatureIdentityTemplateV4,
}

impl GpuOnlyFeatureMaterializationAdmissionV3 {
    pub(crate) fn begin_materialization(
        self,
        smc_materialization: ResidentSmcMaterializationV3,
    ) -> std::result::Result<
        (
            GpuOnlyFeatureMaterializationSealTokenV3,
            ResidentFeatureStoreAssemblerV3,
            PendingResidentSmcBatchV3,
        ),
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        if self.authority != DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3 {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        let expected_column_bindings = self
            .contract
            .planned_routes()
            .iter()
            .map(ResidentFeatureColumnBindingV3::from_admitted_route)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let (assembler, pending_smc_batch) = begin_resident_smc_store_v3(
            self.run_device,
            expected_column_bindings,
            self.contract.working_set(),
            smc_materialization,
        )?;
        Ok((
            GpuOnlyFeatureMaterializationSealTokenV3 {
                authority: self.authority,
                contract: self.contract,
                footprint: self.footprint,
                regime: self.regime,
                robust_normalization: self.robust_normalization,
                feature_major_to_bar_major: self.feature_major_to_bar_major,
                canonical_content_sha256: self.canonical_content_sha256,
                feature_identity: self.feature_identity,
            },
            assembler,
            pending_smc_batch,
        ))
    }
}

/// Crate-private continuation token. It contains the sealed admission contract
/// after the unique runtime carrier has moved into the assembler; it cannot be
/// reconstructed from the public low-level DTO.
#[derive(Debug)]
pub(crate) struct GpuOnlyFeatureMaterializationSealTokenV3 {
    authority: &'static str,
    contract: GpuOnlyResidentAdmissionV3,
    footprint: SealedFootprintComponentReceiptV2,
    regime: SealedRegimeComponentReceiptV3,
    robust_normalization: BoundRobustNormalizationComponentReceiptV2,
    feature_major_to_bar_major: SealedFeatureMajorToBarMajorComponentReceiptV3,
    canonical_content_sha256: SealedCanonicalContentSha256ComponentReceiptV3,
    feature_identity: ResidentFeatureIdentityTemplateV4,
}

/// Stable semantic marker exposed to Search only after Data has revalidated
/// the concrete resident Classic TA implementation against the exact device
/// build and the final FeaturePlan. It intentionally carries no raw authority
/// string, ordinal, or free-memory fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalGpuResidentFeatureExecutionSemanticV1 {
    GpuCudaF64Strict,
}

/// Non-forgeable projection minted from a sealed resident store. This is
/// receipt evidence, not a CUDA admission or allocation capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedGpuResidentFeatureExecutionAuthorityV1 {
    semantic: CanonicalGpuResidentFeatureExecutionSemanticV1,
    final_feature_plan_v3_sha256: [u8; 32],
    classic_ta_implementation_sha256: [u8; 32],
    vector_ta_build_sha256: [u8; 32],
    identity_sha256: [u8; 32],
}

impl ValidatedGpuResidentFeatureExecutionAuthorityV1 {
    pub const fn semantic(&self) -> CanonicalGpuResidentFeatureExecutionSemanticV1 {
        self.semantic
    }

    pub const fn final_feature_plan_v3_sha256(&self) -> [u8; 32] {
        self.final_feature_plan_v3_sha256
    }

    pub const fn classic_ta_implementation_sha256(&self) -> [u8; 32] {
        self.classic_ta_implementation_sha256
    }

    pub const fn vector_ta_build_sha256(&self) -> [u8; 32] {
        self.vector_ta_build_sha256
    }

    pub const fn identity_sha256(&self) -> [u8; 32] {
        self.identity_sha256
    }
}

fn seal_gpu_resident_feature_execution_authority_v1(
    final_feature_plan_v3_sha256: [u8; 32],
    classic_ta_capability: &ResidentProducerCapabilityV3,
    device: &CudaPrimaryContextBuildIdentityV3,
) -> std::result::Result<
    ValidatedGpuResidentFeatureExecutionAuthorityV1,
    GpuOnlyFeatureMaterializationErrorV3,
> {
    let classic_ta_implementation_sha256 = classic_ta_capability.implementation_sha256();
    let vector_ta_build_sha256 = device.vector_ta_build_sha256();
    if final_feature_plan_v3_sha256 == [0; 32]
        || classic_ta_capability.producer() != ResidentFeatureProducerV3::ClassicTa
        || classic_ta_implementation_sha256 == [0; 32]
        || vector_ta_build_sha256 != classic_ta_implementation_sha256
        || classic_ta_capability.exact_math_authority() != F64_EXACT_MATH_AUTHORITY_V3
        || device.exact_math_authority() != F64_EXACT_MATH_AUTHORITY_V3
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let semantic = CanonicalGpuResidentFeatureExecutionSemanticV1::GpuCudaF64Strict;
    let mut identity = Sha256::new();
    identity.update(b"neoethos.data.gpu-resident-feature-execution-authority.v1\0");
    identity.update(b"gpu-cuda-f64-strict");
    identity.update(final_feature_plan_v3_sha256);
    identity.update(classic_ta_implementation_sha256);
    identity.update(vector_ta_build_sha256);
    identity.update(F64_EXACT_MATH_AUTHORITY_V3.as_bytes());
    let identity_sha256: [u8; 32] = identity.finalize().into();
    if identity_sha256 == [0; 32] {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    Ok(ValidatedGpuResidentFeatureExecutionAuthorityV1 {
        semantic,
        final_feature_plan_v3_sha256,
        classic_ta_implementation_sha256,
        vector_ta_build_sha256,
        identity_sha256,
    })
}

impl GpuOnlyFeatureMaterializationSealTokenV3 {
    fn apply_resident_robust_normalization_v2(
        &self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.authority != DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3 {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        self.robust_normalization.apply(assembler)
    }
}

/// Runtime store authority. It owns the primary context, final allocations,
/// parent arrays and final ready event through the Search consumer lease.
#[derive(Debug)]
pub struct SealedGpuResidentFeatureStoreV3 {
    authority: &'static str,
    admission: GpuOnlyResidentAdmissionV3,
    contract: SealedResidentFeatureStoreV3,
    final_feature_plan_v3_sha256: [u8; 32],
    normalization_fit_sha256: [u8; 32],
    source_provenance_sha256: [u8; 32],
    footprint: SealedFootprintComponentReceiptV2,
    regime: SealedRegimeComponentReceiptV3,
    robust_normalization: BoundRobustNormalizationComponentReceiptV2,
    feature_major_to_bar_major: SealedFeatureMajorToBarMajorComponentReceiptV3,
    canonical_content_sha256: SealedCanonicalContentSha256ComponentReceiptV3,
    feature_plan: FeaturePlanV1,
    source_provenance: DatasetFeatureArtifactProvenanceV1,
    pinned_source_projection_v1: CanonicalPinnedSourceProjectionV1,
    resident_sources: MaterializedPinnedResidentCanonicalSourcesV1,
    owner: Arc<ResidentFeatureStoreOwnerV3>,
}

impl SealedGpuResidentFeatureStoreV3 {
    pub const fn authority_id(&self) -> &'static str {
        self.authority
    }

    pub fn contract(&self) -> &SealedResidentFeatureStoreV3 {
        &self.contract
    }

    pub fn device_ordinal(&self) -> u32 {
        let owner_ordinal = self.owner.device_ordinal();
        debug_assert_eq!(owner_ordinal, self.admission.device().ordinal());
        owner_ordinal
    }

    pub const fn admission_identity_sha256(&self) -> [u8; 32] {
        self.admission.admission_identity_sha256()
    }

    pub const fn final_feature_plan_v3_sha256(&self) -> [u8; 32] {
        self.final_feature_plan_v3_sha256
    }

    pub const fn normalization_fit_sha256(&self) -> [u8; 32] {
        self.normalization_fit_sha256
    }

    pub const fn source_provenance_sha256(&self) -> [u8; 32] {
        self.source_provenance_sha256
    }

    pub const fn feature_plan(&self) -> &FeaturePlanV1 {
        &self.feature_plan
    }

    pub const fn source_provenance(&self) -> &DatasetFeatureArtifactProvenanceV1 {
        &self.source_provenance
    }

    /// Node-name-independent source generation identity rederived from the
    /// same leases that survived native materialization. This does not alias
    /// the resident V3 feature receipt to a CPU V2 feature receipt.
    pub const fn pinned_source_projection_v1(&self) -> &CanonicalPinnedSourceProjectionV1 {
        &self.pinned_source_projection_v1
    }

    pub fn ordered_feature_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.admission
            .planned_routes()
            .iter()
            .map(|route| route.feature_name())
    }

    pub const fn device_identity(&self) -> &CudaPrimaryContextBuildIdentityV3 {
        self.admission.device()
    }

    /// Revalidate and project the concrete plan-bound Classic TA execution
    /// authority into the canonical strict GPU semantic used by Search. The
    /// concrete V3 authority is never compared to, or replaced by, an abstract
    /// legacy string.
    pub fn validated_gpu_resident_feature_execution_authority_v1(
        &self,
    ) -> std::result::Result<
        ValidatedGpuResidentFeatureExecutionAuthorityV1,
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        self.validate_resident_feature_store_import_v3()?;
        let classic_ta_capability = self
            .admission
            .capabilities()
            .capabilities()
            .iter()
            .find(|capability| capability.producer() == ResidentFeatureProducerV3::ClassicTa)
            .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
        let classic_routes = self
            .admission
            .planned_routes()
            .iter()
            .filter(|route| route.producer() == ResidentFeatureProducerV3::ClassicTa)
            .collect::<Vec<_>>();
        if classic_routes.is_empty() {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        for route in classic_routes {
            let expected_semantic_source_sha256 = derive_route_semantic_source_sha256_v4(
                RESIDENT_CLASSIC_TA_LOCAL_ROUTE_DOMAIN_V4,
                classic_ta_capability.exact_math_authority(),
                route.route_receipt_sha256(),
            )
            .map_err(GpuOnlyFeatureMaterializationErrorV3::Other)?;
            let node = self
                .feature_plan
                .nodes()
                .iter()
                .find(|node| node.id() == route.route_id())
                .ok_or(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch)?;
            if node.operation() != FeatureOperationTagV1::Indicator
                || node.formula_manifest_hash() != classic_ta_capability.implementation_sha256()
                || node.semantic_source_hash() != expected_semantic_source_sha256
            {
                return Err(
                    GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch,
                );
            }
        }
        seal_gpu_resident_feature_execution_authority_v1(
            self.final_feature_plan_v3_sha256,
            classic_ta_capability,
            self.admission.device(),
        )
    }

    pub const fn ready_event(&self) -> &ResidentReadyEventV3 {
        self.contract.ready_event()
    }

    fn validate_resident_feature_store_import_v3(
        &self,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        let runtime_pinned_source_projection_v1 =
            derive_pinned_source_projection_v1(&self.resident_sources)
                .map_err(|error| GpuOnlyFeatureMaterializationErrorV3::Other(error.into()))?;
        let compact_hashes = self.owner.compact_hashes_if_ready()?;
        let owner_ready_event = self.owner.ready_event_contract()?;
        let runtime_layout_evidence = self.owner.layout_evidence(&compact_hashes);
        self.feature_major_to_bar_major
            .validate_runtime_evidence(&runtime_layout_evidence, &owner_ready_event)?;
        self.canonical_content_sha256
            .validate_runtime_evidence(&runtime_layout_evidence, &owner_ready_event)?;
        self.footprint
            .validate_runtime_evidence(&runtime_layout_evidence)?;
        self.regime
            .validate_runtime_evidence(&runtime_layout_evidence)?;
        let runtime_normalization_fit_sha256 = self
            .robust_normalization
            .validate_runtime_evidence(&runtime_layout_evidence)?;
        let htf_route_count = self
            .admission
            .planned_routes()
            .iter()
            .filter(|route| route.producer() == ResidentFeatureProducerV3::HigherTimeframeAlignment)
            .count();
        let htf_runtime_evidence_matches = runtime_layout_evidence
            .higher_timeframe_runtime_receipt_v3
            .as_ref()
            .is_some_and(|receipt| {
                receipt.semantic_version() == HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3
                    && receipt.base_row_count() == self.owner.rows()
                    && receipt.parent_count() == self.resident_sources.direct_parents().len()
                    && receipt.parent_feature_column_count() == htf_route_count
                    && receipt.feature_value_d2h_bytes() == 0
                    && receipt.feature_validity_d2h_bytes() == 0
                    && receipt.producer_ready_event_synchronize_count() == 0
                    && receipt.host_synchronize_count() == 0
            });
        let owner_bindings = self.owner.column_bindings();
        let admitted_routes = self.admission.planned_routes();
        let layout = self.contract.layout();
        let row_count = u64::try_from(self.owner.rows()).map_err(|_| {
            GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch
        })?;
        let column_count = u64::try_from(self.owner.columns()).map_err(|_| {
            GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch
        })?;
        let owner_steady_device_bytes = checked_u64(
            self.owner.sealed_steady_device_bytes(),
            "sealed resident steady device bytes",
        )?;
        let ordered_bindings_match = owner_bindings.len() == admitted_routes.len()
            && owner_bindings.iter().zip(admitted_routes).enumerate().all(
                |(ordinal, (binding, route))| {
                    binding.ordinal == ordinal
                        && u64::try_from(ordinal).is_ok_and(|value| route.ordinal() == value)
                        && binding.feature_name == route.feature_name()
                        && binding.canonical_parameter_tuple_sha256
                            == route.canonical_parameter_tuple_sha256()
                        && binding.route_receipt_sha256 == route.route_receipt_sha256()
                },
            );
        if self.authority != DATA_GPU_ONLY_SEALED_STORE_AUTHORITY_V3
            || self.admission_identity_sha256() == [0; 32]
            || self.final_feature_plan_v3_sha256 == [0; 32]
            || self.final_feature_plan_v3_sha256 != *self.feature_plan.identity().as_bytes()
            || self.normalization_fit_sha256 == [0; 32]
            || self.normalization_fit_sha256 != runtime_normalization_fit_sha256
            || self.source_provenance_sha256 == [0; 32]
            || self.source_provenance_sha256 != *self.source_provenance.identity().as_bytes()
            || self.pinned_source_projection_v1 != runtime_pinned_source_projection_v1
            || !htf_runtime_evidence_matches
            || self.resident_sources.base().frame().len() != usize::try_from(row_count).unwrap_or(0)
            || self.owner.admission_identity_sha256() != self.admission_identity_sha256()
            || self.owner.device_identity() != self.admission.device()
            || self.owner.device_ordinal() != self.admission.device().ordinal()
            || row_count != self.admission.working_set().row_count()
            || column_count != self.admission.working_set().column_count()
            || owner_steady_device_bytes != self.admission.working_set().steady_device_bytes()
            || row_count != layout.row_count()
            || column_count != layout.column_count()
            || compact_hashes.canonical_content_merkle
                != self.contract.canonical_feature_content_merkle_sha256()
            || owner_ready_event != *self.contract.ready_event()
            || !ordered_bindings_match
        {
            return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
        }
        Ok(())
    }

    pub fn into_resident_feature_store_import_v3(
        self,
    ) -> std::result::Result<ResidentFeatureStoreImportV3, GpuOnlyFeatureMaterializationErrorV3>
    {
        self.validate_resident_feature_store_import_v3()?;
        let Self { owner, .. } = self;
        owner.import_on_admitted_run_stream_v3().map_err(Into::into)
    }

    #[cfg(feature = "gpu-cuda-device-fixtures")]
    pub fn copy_bar_major_for_device_fixture_v3(
        &self,
    ) -> std::result::Result<
        neoethos_gpu_cuda::resident_feature_store_v3_device_fixture::ResidentFeatureStoreDeviceReadbackV3,
        GpuOnlyFeatureMaterializationErrorV3,
    >{
        self.validate_resident_feature_store_import_v3()?;
        self.owner
            .copy_bar_major_for_device_fixture_v3()
            .map_err(Into::into)
    }
}

fn require_complete_resident_producer_manifest_v3(
    capabilities: Vec<ResidentProducerCapabilityV3>,
) -> std::result::Result<ResidentProducerCapabilityManifestV3, GpuOnlyFeatureMaterializationErrorV3>
{
    match ResidentProducerCapabilityManifestV3::seal(capabilities) {
        Ok(manifest) => Ok(manifest),
        Err(ResidentFeatureContractErrorV3::MissingProducerCapabilities { missing }) => {
            Err(GpuOnlyFeatureMaterializationErrorV3::MissingProducerCapabilities { missing })
        }
        Err(error) => Err(error.into()),
    }
}

/// Seal only Data's crate-owned production producer census. The caller cannot
/// supply capability DTOs, hashes, or booleans through this boundary.
pub(crate) fn seal_current_resident_producer_capability_manifest_v3()
-> std::result::Result<ResidentProducerCapabilityManifestV3, GpuOnlyFeatureMaterializationErrorV3> {
    require_complete_resident_producer_manifest_v3(current_resident_producer_capabilities_v3()?)
}

#[derive(Debug)]
pub(crate) struct GpuOnlyFeatureRecipePreflightV3 {
    plan: ResolvedGpuOnlyFeatureMaterializationPlanV3,
    footprint: SealedFootprintComponentReceiptV2,
    regime: SealedRegimeComponentReceiptV3,
    robust_normalization: SealedRobustNormalizationComponentReceiptV2,
    feature_major_to_bar_major: SealedFeatureMajorToBarMajorComponentReceiptV3,
    canonical_content_sha256: SealedCanonicalContentSha256ComponentReceiptV3,
}

impl GpuOnlyFeatureRecipePreflightV3 {
    fn resident_sources(&self) -> &MaterializedPinnedResidentCanonicalSourcesV1 {
        self.plan.feature_identity.resident_sources()
    }

    fn exact_bindings_for(
        &self,
        producer: ResidentFeatureProducerV3,
    ) -> std::result::Result<
        Vec<ResidentFeatureColumnBindingV3>,
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        let bindings = self
            .plan
            .planned_routes
            .iter()
            .filter(|route| route.producer() == producer)
            .map(ResidentFeatureColumnBindingV3::from_admitted_route)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if bindings.is_empty() {
            return Err(
                GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                    component: "ordered resident SMC route bindings",
                },
            );
        }
        Ok(bindings)
    }
}

pub(crate) fn preflight_gpu_only_feature_recipe_v3(
    mut plan: ResolvedGpuOnlyFeatureMaterializationPlanV3,
) -> std::result::Result<GpuOnlyFeatureRecipePreflightV3, GpuOnlyFeatureMaterializationErrorV3> {
    // This is the fail-before-device/materialization contract. Missing exact
    // Quant, Session or HigherTimeframeAlignment authority returns the complete
    // ordered frontier before orchestration may acquire one run-device carrier.
    if plan.row_count == 0 || plan.planned_routes.is_empty() || plan.producer_batches.is_empty() {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!(
                "strict resident feature recipe must contain rows, routes and producer batches"
            ),
        ));
    }
    let feature_major_to_bar_major = seal_feature_major_to_bar_major_component_receipt_v3(&plan)?;
    let canonical_content_sha256 = seal_canonical_content_sha256_component_receipt_v3(&plan)?;
    let footprint = seal_footprint_component_receipt_v2(&plan)?;
    let regime = seal_regime_component_receipt_v3(&plan)?;
    let robust_normalization_input = plan.robust_normalization_input.take().ok_or(
        GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
            component: "move-only resident robust-normalization input",
        },
    )?;
    let robust_normalization =
        seal_robust_normalization_component_receipt_v2(&plan, robust_normalization_input)?;
    Ok(GpuOnlyFeatureRecipePreflightV3 {
        plan,
        footprint,
        regime,
        robust_normalization,
        feature_major_to_bar_major,
        canonical_content_sha256,
    })
}

pub(crate) fn bind_gpu_only_run_device_v3(
    preflight: GpuOnlyFeatureRecipePreflightV3,
    run_device: GpuOnlyRunDeviceAdmissionV3,
) -> std::result::Result<
    GpuOnlyFeatureMaterializationAdmissionV3,
    GpuOnlyFeatureMaterializationErrorV3,
> {
    let GpuOnlyFeatureRecipePreflightV3 {
        plan,
        footprint,
        regime,
        robust_normalization,
        feature_major_to_bar_major,
        canonical_content_sha256,
    } = preflight;
    if run_device.reserve_policy_id() != EXACT_ALLOCATOR_RESERVE_POLICY_V3 {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let working_set = derive_exact_resident_working_set_v3(
        &plan,
        run_device.phase_one_free_bytes_snapshot(),
        run_device.allocator_context_reserve_bytes(),
    )?;
    feature_major_to_bar_major.validate_working_set(&working_set)?;
    canonical_content_sha256.validate_working_set(&working_set)?;
    footprint.validate_working_set(&working_set)?;
    regime.validate_working_set(&working_set)?;
    let robust_normalization = robust_normalization.bind_run_device_v2(&run_device)?;
    robust_normalization.validate_working_set(&working_set)?;
    let ResolvedGpuOnlyFeatureMaterializationPlanV3 {
        dataset_recipe_sha256,
        feature_plan_schema_sha256,
        route_plan_sha256,
        planned_routes,
        producer_capabilities,
        feature_identity,
        ..
    } = plan;
    let contract = GpuOnlyResidentAdmissionV3::seal(GpuOnlyResidentAdmissionRequestV3 {
        dataset_recipe_sha256,
        feature_plan_schema_sha256,
        route_plan_sha256,
        admission_identity_sha256: run_device.admission_identity_sha256(),
        planned_routes,
        capabilities: producer_capabilities,
        device: run_device.device_identity().clone(),
        working_set,
    })?;
    Ok(GpuOnlyFeatureMaterializationAdmissionV3 {
        authority: DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3,
        contract,
        run_device,
        footprint,
        regime,
        robust_normalization,
        feature_major_to_bar_major,
        canonical_content_sha256,
        feature_identity,
    })
}

pub(crate) fn seal_gpu_resident_feature_store_v3(
    seal_token: GpuOnlyFeatureMaterializationSealTokenV3,
    owner: Arc<ResidentFeatureStoreOwnerV3>,
) -> std::result::Result<SealedGpuResidentFeatureStoreV3, GpuOnlyFeatureMaterializationErrorV3> {
    if seal_token.authority != DATA_GPU_ONLY_ADMISSION_AUTHORITY_V3 {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let hashes: ResidentFeatureCompactHashesV3 = loop {
        match owner.compact_hashes_if_ready() {
            Ok(hashes) => break hashes,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => std::thread::yield_now(),
            Err(error) => return Err(error.into()),
        }
    };
    let evidence: ResidentFeatureLayoutEvidenceV3 = owner.layout_evidence(&hashes);
    let ready_event = owner.ready_event_contract()?;
    seal_token
        .feature_major_to_bar_major
        .validate_runtime_evidence(&evidence, &ready_event)?;
    seal_token
        .canonical_content_sha256
        .validate_runtime_evidence(&evidence, &ready_event)?;
    seal_token.footprint.validate_runtime_evidence(&evidence)?;
    seal_token.regime.validate_runtime_evidence(&evidence)?;
    let normalization_fit_sha256 = seal_token
        .robust_normalization
        .validate_runtime_evidence(&evidence)?;
    let finalized_identity = seal_token
        .feature_identity
        .finalize_after_normalization_v4(normalization_fit_sha256)
        .map_err(GpuOnlyFeatureMaterializationErrorV3::Other)?;
    let (feature_plan, source_provenance, resident_sources) = finalized_identity.into_parts();
    let pinned_source_projection_v1 = derive_pinned_source_projection_v1(&resident_sources)
        .map_err(|error| GpuOnlyFeatureMaterializationErrorV3::Other(error.into()))?;
    let final_feature_plan_v3_sha256 = *feature_plan.identity().as_bytes();
    let source_provenance_sha256 = *source_provenance.identity().as_bytes();
    let owner_steady_device_bytes = checked_u64(
        owner.sealed_steady_device_bytes(),
        "sealed resident steady device bytes",
    )?;
    if owner.admission_identity_sha256() != seal_token.contract.admission_identity_sha256()
        || owner.device_identity() != seal_token.contract.device()
        || owner.device_ordinal() != seal_token.contract.device().ordinal()
        || owner_steady_device_bytes != seal_token.contract.working_set().steady_device_bytes()
        || evidence.retained_parent_dataset_bytes
            != seal_token.contract.working_set().parent_dataset_bytes()
        || evidence.pre_materialization_free_bytes_snapshot
            != seal_token
                .contract
                .working_set()
                .device_free_bytes_snapshot()
        || evidence.reserve_policy_id != seal_token.contract.working_set().reserve_policy_id()
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::PrimaryContextBuildIdentityMismatch);
    }
    let layout =
        neoethos_gpu_contracts::resident_feature_store_v3::ResidentFeatureLayoutRequestV3 {
            row_count: evidence.rows,
            column_count: evidence.columns,
            canonical_content_merkle_sha256: evidence.canonical_content_merkle,
            source_column_count: checked_u64(evidence.source_column_count, "source columns")?,
            producer_batch_count: checked_u64(evidence.producer_batch_count, "producer batches")?,
            validity_initialization_count: checked_u64(
                evidence.validity_initialization_count,
                "validity initializations",
            )?,
            value_layout_launch_count: checked_u64(
                evidence.value_layout_launch_count,
                "value layout launches",
            )?,
            validity_boundary_launch_count: checked_u64(
                evidence.validity_boundary_launch_count,
                "validity boundary launches",
            )?,
            layout_transform_value_bytes: checked_u64(
                evidence.layout_transform_value_bytes,
                "layout transform value bytes",
            )?,
            layout_transform_logical_validity_bytes: checked_u64(
                evidence.layout_transform_logical_validity_bytes,
                "layout transform logical validity bytes",
            )?,
            full_feature_major_staging_bytes: checked_u64(
                evidence.full_feature_major_staging_bytes,
                "full feature-major staging bytes",
            )?,
            max_live_producer_bytes: checked_u64(
                evidence.max_live_producer_bytes,
                "max live producer bytes",
            )?,
            max_live_producer_scratch_bytes: checked_u64(
                evidence.max_live_producer_scratch_bytes,
                "max live producer scratch bytes",
            )?,
            pre_materialization_free_bytes_snapshot: evidence
                .pre_materialization_free_bytes_snapshot,
            post_parent_free_bytes_snapshot: evidence.post_parent_free_bytes_snapshot,
            retained_parent_dataset_bytes: evidence.retained_parent_dataset_bytes,
            remaining_peak_after_parent_bytes: evidence.remaining_peak_after_parent_bytes,
            allocator_context_reserve_bytes: evidence.allocator_context_reserve_bytes,
            reserve_policy_id: evidence.reserve_policy_id,
        }
        .seal()?;
    let ordered_feature_names = owner
        .column_bindings()
        .iter()
        .map(|binding| binding.feature_name.clone())
        .collect();
    let parent_dataset = owner.parent_dataset_layout().clone();
    let contract = SealedResidentFeatureStoreV3::seal(
        &seal_token.contract,
        SealedResidentFeatureStoreRequestV3 {
            admission_identity_sha256: seal_token.contract.admission_identity_sha256(),
            final_feature_plan_v3_sha256,
            normalization_fit_sha256,
            source_provenance_sha256,
            ordered_feature_names,
            layout,
            parent_dataset,
            ready_event,
            sha256_authority: CanonicalCudaSha256AuthorityV3::portable_in_tree(),
        },
    )?;
    Ok(SealedGpuResidentFeatureStoreV3 {
        authority: DATA_GPU_ONLY_SEALED_STORE_AUTHORITY_V3,
        admission: seal_token.contract,
        contract,
        final_feature_plan_v3_sha256,
        normalization_fit_sha256,
        source_provenance_sha256,
        footprint: seal_token.footprint,
        regime: seal_token.regime,
        robust_normalization: seal_token.robust_normalization,
        feature_major_to_bar_major: seal_token.feature_major_to_bar_major,
        canonical_content_sha256: seal_token.canonical_content_sha256,
        feature_plan,
        source_provenance,
        pinned_source_projection_v1,
        resident_sources,
        owner,
    })
}

fn resident_recipe_parameter_v4(
    name: &'static str,
    value: ResidentCanonicalParameterValueV4,
) -> Result<ResidentCanonicalParameterV4> {
    ResidentCanonicalParameterV4::from_typed_value(name, value).map_err(Into::into)
}

fn resident_fixed_output_route_v4(
    feature_name: &str,
    indicator_id: &'static str,
    route_domain: &'static str,
    mut parameters: Vec<ResidentCanonicalParameterV4>,
) -> Result<ResidentRouteDraftV4> {
    parameters.push(resident_recipe_parameter_v4(
        "output_semantic_tag",
        ResidentCanonicalParameterValueV4::Text(feature_name.to_owned()),
    )?);
    ResidentRouteDraftV4::from_typed_parts(
        feature_name,
        Some(indicator_id),
        Some(feature_name),
        ResidentFeatureStageV3::Derived,
        None,
        parameters,
        route_domain,
    )
    .map_err(Into::into)
}

fn resident_smc_producer_draft_v4(row_count: usize) -> Result<ResidentProducerDraftV4> {
    let memory = preflight_resident_smc_memory_v4(row_count)?;
    if memory.row_count() != row_count
        || memory.feature_column_count() != RESIDENT_SMC_COLUMN_NAMES_V3.len()
    {
        bail!("resident SMC pre-device memory disagrees with the schema authority")
    }
    let routes = RESIDENT_SMC_COLUMN_NAMES_V3
        .iter()
        .copied()
        .map(|name| {
            resident_fixed_output_route_v4(
                name,
                "neoethos_smc_semantic_v3",
                "neoethos.data.resident-smc-route.semantic-v3",
                vec![resident_recipe_parameter_v4(
                    "smc_semantic_version",
                    ResidentCanonicalParameterValueV4::U64(u64::from(SMC_SEMANTIC_VERSION)),
                )?],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::Smc,
        SMC_SEMANTIC_VERSION,
        routes,
        vec![ResidentProducerBatchDraftV4::from_owner_preflight(
            0,
            RESIDENT_SMC_COLUMN_NAMES_V3.len(),
            u64::try_from(memory.additional_retained_bytes())
                .context("resident SMC retained bytes do not fit recipe-v4")?,
            u64::try_from(memory.scratch_bytes())
                .context("resident SMC scratch bytes do not fit recipe-v4")?,
        )],
        resident_smc_capability_v3()?,
    )
    .map_err(Into::into)
}

fn resident_regime_producer_draft_v4(
    row_count: usize,
    scale_anchor_bits: u64,
) -> Result<ResidentProducerDraftV4> {
    if REGIME_FEATURE_NAMES_V3 != RESIDENT_REGIME_COLUMN_NAMES_V3
        || REGIME_SEMANTIC_VERSION != RESIDENT_REGIME_SEMANTIC_VERSION_V3
    {
        bail!("resident Regime CPU and CUDA schema authorities disagree")
    }
    let memory = preflight_resident_regime_memory_v4(row_count)?;
    let scale_anchor = f64::from_bits(scale_anchor_bits);
    if !scale_anchor.is_finite() || scale_anchor <= 0.0 {
        bail!("resident Regime scale anchor is invalid")
    }
    let routes = REGIME_FEATURE_NAMES_V3
        .iter()
        .copied()
        .map(|name| {
            resident_fixed_output_route_v4(
                name,
                "neoethos_regime_semantic_v3",
                "neoethos.data.resident-regime-route.semantic-v3",
                vec![
                    resident_recipe_parameter_v4(
                        "scale_anchor",
                        ResidentCanonicalParameterValueV4::F64Bits(scale_anchor_bits),
                    )?,
                    resident_recipe_parameter_v4(
                        "operation_schedule",
                        ResidentCanonicalParameterValueV4::Text(
                            REGIME_OPERATION_SCHEDULE_V1.to_owned(),
                        ),
                    )?,
                    resident_recipe_parameter_v4(
                        "semantic_fixture_sha256",
                        ResidentCanonicalParameterValueV4::Text(
                            REGIME_SEMANTIC_V3_FIXTURE_SHA256.to_owned(),
                        ),
                    )?,
                ],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::Regime,
        REGIME_SEMANTIC_VERSION,
        routes,
        vec![ResidentProducerBatchDraftV4::from_owner_preflight(
            0,
            REGIME_FEATURE_NAMES_V3.len(),
            u64::try_from(memory.additional_retained_bytes())
                .context("resident Regime retained bytes do not fit recipe-v4")?,
            u64::try_from(memory.scratch_bytes())
                .context("resident Regime scratch bytes do not fit recipe-v4")?,
        )],
        resident_regime_capability_v3()?,
    )
    .map_err(Into::into)
}

fn resident_footprint_producer_draft_v4(row_count: usize) -> Result<ResidentProducerDraftV4> {
    if FOOTPRINT_FEATURE_NAMES != RESIDENT_FOOTPRINT_COLUMN_NAMES_V2
        || FOOTPRINT_SEMANTIC_VERSION != FOOTPRINT_SEMANTIC_VERSION_V2
    {
        bail!("resident Footprint CPU and CUDA schema authorities disagree")
    }
    let memory = preflight_resident_footprint_memory_v4(row_count)?;
    let routes = FOOTPRINT_FEATURE_NAMES
        .iter()
        .copied()
        .map(|name| {
            resident_fixed_output_route_v4(
                name,
                "neoethos_footprint_semantic_v2",
                "neoethos.data.resident-footprint-route.semantic-v2",
                vec![resident_recipe_parameter_v4(
                    "footprint_semantic_version",
                    ResidentCanonicalParameterValueV4::U64(u64::from(FOOTPRINT_SEMANTIC_VERSION)),
                )?],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::Footprint,
        FOOTPRINT_SEMANTIC_VERSION,
        routes,
        vec![ResidentProducerBatchDraftV4::from_owner_preflight(
            0,
            FOOTPRINT_FEATURE_NAMES.len(),
            u64::try_from(memory.additional_retained_bytes())
                .context("resident Footprint retained bytes do not fit recipe-v4")?,
            u64::try_from(memory.scratch_bytes())
                .context("resident Footprint scratch bytes do not fit recipe-v4")?,
        )],
        resident_footprint_capability_v2()?,
    )
    .map_err(Into::into)
}

/// Crate-owned resident producer authority. It accepts only canonical Data
/// inputs and never exposes the low-level producer traits across the crate
/// boundary. Every A2 producer must contribute one exact draft, retained
/// runtime continuation and release-bound native capability before the
/// admitted run-device carrier is consumed.
#[derive(Debug)]
struct CrateOwnedResidentProducerFactoryV3 {
    materialization: CrateOwnedResidentMaterializationV3,
}

#[derive(Debug)]
struct CrateOwnedResidentMaterializationV3 {
    regime_input: Option<PreparedResidentRegimeInputV3>,
    smc_materialization: Option<ResidentSmcMaterializationV3>,
    classic_ta_recipe: Option<ResidentClassicTaRecipeV3>,
    classic_ta_memory: Option<ResidentClassicTaPreDeviceMemoryReceiptV4>,
    quant_runtime: Option<PreparedResidentQuantRuntimeV3>,
    session_runtime: Option<PreparedResidentSessionRuntimeV2>,
    htf_runtime: Option<PendingResidentHigherTimeframeRuntimeV3>,
    htf_capture_templates:
        Option<Vec<PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3>>,
}

impl CrateOwnedResidentProducerFactoryV3 {
    fn resolve(
        workspace_preflight: PreparedGpuOnlyFeatureWorkspacePreflightV3,
    ) -> std::result::Result<
        (
            ResolvedGpuOnlyFeatureMaterializationPlanV3,
            CrateOwnedResidentProducerFactoryV3,
        ),
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        let mut recipe_assembly = workspace_preflight.into_resident_feature_recipe_assembly_v4()?;
        let row_count = recipe_assembly.row_count();
        let budget_rows = recipe_assembly.budget_rows();
        let base_timeframe = recipe_assembly.base_timeframe();
        let profile = recipe_assembly.profile();
        let base_source = recipe_assembly.resident_sources().base().frame();
        let regime_input = preflight_resident_regime_v3(base_source.ohlcv())?;
        let classic_run_plan = match profile {
            FeatureProfile::Full => {
                prepare_classic_ta_run_plan(budget_rows, IndicatorComputePolicy::GpuOnly)?
            }
            FeatureProfile::Standard | FeatureProfile::HPC | FeatureProfile::Adaptive => {
                prepare_classic_ta_gpu_exact_parity_run_plan_v3(budget_rows)?
            }
        };
        let ResidentClassicTaPlanV3 {
            recipe: classic_ta_recipe,
            local_draft: classic_local_draft,
        } = preflight_resident_classic_ta_v3(&classic_run_plan, row_count)?;
        let classic_ta_memory = preflight_resident_classic_ta_memory_v4(&classic_ta_recipe)?;
        let classic_draft = classic_local_draft
            .into_resident_feature_recipe_draft_v4(&classic_ta_recipe, &classic_ta_memory)?;
        let (quant_draft, quant_runtime) =
            preflight_current_native_resident_quant_v3(base_source.ohlcv(), base_timeframe)?
                .into_recipe_parts();
        let (session_draft, session_runtime) =
            preflight_current_native_resident_session_v2(base_source.ohlcv())?.into_recipe_parts();
        let mut htf_host_parents =
            Vec::with_capacity(recipe_assembly.resident_sources().direct_parents().len());
        let mut htf_capture_templates =
            Vec::with_capacity(recipe_assembly.resident_sources().direct_parents().len());
        for direct_parent in recipe_assembly.resident_sources().direct_parents() {
            let parent_rows = direct_parent.frame().len();
            let smc_memory = preflight_resident_smc_memory_v4(parent_rows)?;
            let smc_draft = resident_smc_producer_draft_v4(parent_rows)?;
            let ResidentClassicTaPlanV3 {
                recipe: parent_classic_recipe,
                local_draft: parent_classic_local_draft,
            } = preflight_resident_classic_ta_v3(&classic_run_plan, parent_rows)?;
            let parent_classic_memory =
                preflight_resident_classic_ta_memory_v4(&parent_classic_recipe)?;
            let parent_classic_draft = parent_classic_local_draft
                .into_resident_feature_recipe_draft_v4(
                    &parent_classic_recipe,
                    &parent_classic_memory,
                )?;
            let (parent_quant_draft, parent_quant_runtime) =
                preflight_current_native_resident_quant_v3(
                    direct_parent.frame().ohlcv(),
                    direct_parent.timeframe(),
                )?
                .into_recipe_parts();
            let (parent_session_draft, parent_session_runtime) =
                preflight_current_native_resident_session_v2(direct_parent.frame().ohlcv())?
                    .into_recipe_parts();
            let parent_regime_input = preflight_resident_regime_v3(direct_parent.frame().ohlcv())?;
            let parent_regime_memory = preflight_resident_regime_memory_v4(parent_rows)?;
            let parent_regime_draft =
                resident_regime_producer_draft_v4(parent_rows, parent_regime_input.evidence().1)?;
            let parent_footprint_memory = preflight_resident_footprint_memory_v4(parent_rows)?;
            let parent_footprint_draft = resident_footprint_producer_draft_v4(parent_rows)?;
            let (host_parent, capture_template) =
                prepare_resident_higher_timeframe_direct_parent_owner_v3(
                    direct_parent,
                    smc_draft,
                    smc_memory,
                    parent_classic_draft,
                    parent_classic_recipe,
                    parent_classic_memory,
                    parent_quant_draft,
                    parent_quant_runtime,
                    parent_session_draft,
                    parent_session_runtime,
                    parent_regime_draft,
                    parent_regime_input,
                    parent_regime_memory,
                    parent_footprint_draft,
                    parent_footprint_memory,
                )?;
            htf_host_parents.push(host_parent);
            htf_capture_templates.push(capture_template);
        }
        let base_open_ms = base_source
            .ohlcv()
            .timestamp
            .as_deref()
            .context("canonical resident base source is missing timestamp_ms")?;
        let (htf_draft, htf_runtime) = preflight_resident_higher_timeframe_alignment_v3(
            base_timeframe,
            base_open_ms,
            htf_host_parents,
            resident_higher_timeframe_capability_v3()?,
        )?
        .into_recipe_and_runtime();

        recipe_assembly.append_owner_draft(resident_smc_producer_draft_v4(row_count)?)?;
        recipe_assembly.append_owner_draft(classic_draft)?;
        recipe_assembly.append_owner_draft(quant_draft)?;
        recipe_assembly.append_owner_draft(session_draft)?;
        recipe_assembly.append_owner_draft(resident_regime_producer_draft_v4(
            row_count,
            regime_input.evidence().1,
        )?)?;
        recipe_assembly.append_owner_draft(resident_footprint_producer_draft_v4(row_count)?)?;
        recipe_assembly.append_owner_draft(htf_draft)?;
        let transform_capabilities = ResidentTransformCapabilityDraftV4::from_owner_capabilities(
            resident_robust_normalization_capability_v2()?,
            resident_canonical_content_sha256_capability_v3()?,
            resident_feature_major_to_bar_major_capability_v3()?,
        )?;
        let prepared_recipe = recipe_assembly.seal(transform_capabilities)?;
        let prepared_materialization = prepared_recipe.into_materialization_v4()?;
        let (
            feature_identity,
            robust_normalization_split,
            planned_routes,
            producer_batches_v4,
            producer_capabilities,
        ) = prepared_materialization.into_parts();
        let robust_normalization_input = prepare_resident_robust_normalization_input_v2(
            robust_normalization_split,
            planned_routes.len(),
        )
        .map_err(GpuOnlyFeatureMaterializationErrorV3::Other)?;
        let normalization_scratch_bytes = checked_u64(
            robust_normalization_input.normalization_scratch_bytes(),
            "normalization scratch bytes",
        )?;
        let fit_metadata_bytes = checked_u64(
            robust_normalization_input.fit_metadata_bytes(),
            "normalization fit metadata bytes",
        )?;
        let producer_batches = producer_batches_v4
            .into_iter()
            .map(|batch| ResolvedResidentProducerBatchMemoryV3 {
                producer: batch.producer(),
                first_column: batch.first_column(),
                column_count: batch.column_count(),
                additional_retained_bytes: batch.additional_retained_bytes(),
                scratch_bytes: batch.scratch_bytes(),
            })
            .collect();
        let (_, regime_scale_anchor_bits, regime_input_identity_sha256) = regime_input.evidence();
        let plan = ResolvedGpuOnlyFeatureMaterializationPlanV3 {
            dataset_recipe_sha256: feature_identity.dataset_recipe_sha256(),
            feature_plan_schema_sha256: feature_identity.feature_plan_schema_sha256(),
            route_plan_sha256: feature_identity.route_plan_sha256(),
            row_count,
            planned_routes,
            producer_capabilities,
            producer_batches,
            regime_scale_anchor_bits,
            regime_input_identity_sha256,
            normalization_scratch_bytes,
            fit_metadata_bytes,
            robust_normalization_input: Some(robust_normalization_input),
            feature_identity,
        };
        let factory = Self {
            materialization: CrateOwnedResidentMaterializationV3 {
                regime_input: Some(regime_input),
                smc_materialization: None,
                classic_ta_recipe: Some(classic_ta_recipe),
                classic_ta_memory: Some(classic_ta_memory),
                quant_runtime: Some(quant_runtime),
                session_runtime: Some(session_runtime),
                htf_runtime: Some(htf_runtime),
                htf_capture_templates: Some(htf_capture_templates),
            },
        };
        Ok((plan, factory))
    }

    fn prepare_smc(
        &mut self,
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        source: &CanonicalOhlcvFrame,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        self.materialization
            .prepare_smc(run_device, source, bindings)
    }

    fn take_smc_materialization(
        &mut self,
    ) -> std::result::Result<ResidentSmcMaterializationV3, GpuOnlyFeatureMaterializationErrorV3>
    {
        self.materialization.take_smc_materialization()
    }

    fn take_classic_ta_runtime(
        &mut self,
    ) -> std::result::Result<
        (
            ResidentClassicTaRecipeV3,
            ResidentClassicTaPreDeviceMemoryReceiptV4,
        ),
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        if self.materialization.classic_ta_recipe.is_none()
            || self.materialization.classic_ta_memory.is_none()
        {
            return Err(
                GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                    component: "frozen resident Classic TA recipe and memory receipt",
                },
            );
        }
        Ok((
            self.materialization
                .classic_ta_recipe
                .take()
                .expect("checked resident Classic TA recipe"),
            self.materialization
                .classic_ta_memory
                .take()
                .expect("checked resident Classic TA memory receipt"),
        ))
    }

    fn take_quant_runtime(
        &mut self,
    ) -> std::result::Result<PreparedResidentQuantRuntimeV3, GpuOnlyFeatureMaterializationErrorV3>
    {
        self.materialization.quant_runtime.take().ok_or(
            GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                component: "prepared resident Quant-v3 runtime continuation",
            },
        )
    }

    fn take_session_runtime(
        &mut self,
    ) -> std::result::Result<PreparedResidentSessionRuntimeV2, GpuOnlyFeatureMaterializationErrorV3>
    {
        self.materialization.session_runtime.take().ok_or(
            GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                component: "prepared resident Session-v2 runtime continuation",
            },
        )
    }

    fn take_higher_timeframe_runtime(
        &mut self,
    ) -> std::result::Result<
        (
            PendingResidentHigherTimeframeRuntimeV3,
            Vec<PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3>,
        ),
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        let runtime = self.materialization.htf_runtime.take().ok_or(
            GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                component: "prepared resident HTF-v3 runtime continuation",
            },
        )?;
        let capture_templates = self.materialization.htf_capture_templates.take().ok_or(
            GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                component: "prepared resident HTF-v3 direct-parent capture templates",
            },
        )?;
        Ok((runtime, capture_templates))
    }

    fn take_regime_input(
        &mut self,
    ) -> std::result::Result<PreparedResidentRegimeInputV3, GpuOnlyFeatureMaterializationErrorV3>
    {
        self.materialization.regime_input.take().ok_or(
            GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                component: "sealed resident Regime input admission",
            },
        )
    }
}

impl CrateOwnedResidentMaterializationV3 {
    fn prepare_smc(
        &mut self,
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        source: &CanonicalOhlcvFrame,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> std::result::Result<(), GpuOnlyFeatureMaterializationErrorV3> {
        if self.smc_materialization.is_some() {
            return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
                anyhow::anyhow!("resident SMC materialization is one-shot"),
            ));
        }
        self.smc_materialization = Some(prepare_smc_materialization_v3(
            run_device, source, bindings,
        )?);
        Ok(())
    }

    fn take_smc_materialization(
        &mut self,
    ) -> std::result::Result<ResidentSmcMaterializationV3, GpuOnlyFeatureMaterializationErrorV3>
    {
        self.smc_materialization.take().ok_or(
            GpuOnlyFeatureMaterializationErrorV3::A2ProducerFactoryNotIntegrated {
                component: "resident parent OHLCV/clock/SMC owner",
            },
        )
    }
}

fn prepare_smc_materialization_v3(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    source: &CanonicalOhlcvFrame,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
) -> Result<ResidentSmcMaterializationV3> {
    let source = source.ohlcv();
    let timestamps = source
        .timestamp
        .as_deref()
        .context("canonical resident SMC input is missing timestamp_ms")?;
    let volume = source
        .volume
        .as_deref()
        .context("canonical resident Classic/SMC parent input is missing volume")?;
    prepare_resident_smc_parent_v3(
        run_device,
        &source.open,
        &source.high,
        &source.low,
        &source.close,
        volume,
        timestamps,
        bindings,
    )
    .map_err(Into::into)
}

/// Exact current capability census. SMC is the first real A2 producer: it owns
/// the canonical parent arrays and one exact 46-column resident batch;
/// Classic TA projects and executes the already-resolved run graph. Regime-v3
/// is the exact fourteen-column, zero-scratch two-launch family; Footprint is
/// one complete seven-column semantic-v2 CUDA family over the retained parent.
/// Quant-v3 and Session-v2 are release-receipt-bound native CUDA owners. The
/// in-tree parallel Merkle SHA-256 and f64/u4 layout kernels supply the
/// final two ordered capabilities.
/// HTF-v3 owns the retained direct-parent graph and causal alignment span.
/// RobustNormalization is the real post-pack semantic-v2 transform.
fn current_resident_producer_capabilities_v3()
-> std::result::Result<Vec<ResidentProducerCapabilityV3>, GpuOnlyFeatureMaterializationErrorV3> {
    let _required_order = ResidentFeatureProducerV3::ALL;
    Ok(vec![
        resident_classic_ta_capability_v3()?,
        resident_smc_capability_v3()?,
        resident_quant_capability_v3()?,
        resident_session_capability_v2()?,
        resident_regime_capability_v3()?,
        resident_footprint_capability_v2()?,
        resident_higher_timeframe_capability_v3()?,
        resident_robust_normalization_capability_v2()?,
        resident_canonical_content_sha256_capability_v3()?,
        resident_feature_major_to_bar_major_capability_v3()?,
    ])
}

/// Move-only continuation produced after every Data-owned recipe, route,
/// producer-memory and normalization preflight has sealed, but before a CUDA
/// run carrier is consumed. This is the staging boundary required by the
/// application workspace planner: the complete exact recipe is frozen once
/// and later materialized on the same admitted run without re-resolution.
#[must_use = "the prepared resident feature materialization must consume one admitted CUDA run"]
#[derive(Debug)]
pub struct PreparedGpuOnlyFeatureMaterializationV3 {
    workspace_extent: ResidentWorkingSetExtentV3,
    pinned_source_projection_v1: CanonicalPinnedSourceProjectionV1,
    preflight: GpuOnlyFeatureRecipePreflightV3,
    producers: CrateOwnedResidentProducerFactoryV3,
}

impl PreparedGpuOnlyFeatureMaterializationV3 {
    /// Exact hardware-independent Data allocation extent for the immutable
    /// recipe carried by this token. Full-run workspace admission may account
    /// for these bytes before a CUDA ordinal or free-memory snapshot is bound.
    pub const fn workspace_extent(&self) -> &ResidentWorkingSetExtentV3 {
        &self.workspace_extent
    }

    /// Exact pinned input identity available before native workspace sealing
    /// or Data allocation. It projects immutable source facts only; feature
    /// plan/content identities remain versioned separately for CPU and GPU.
    pub const fn pinned_source_projection_v1(&self) -> &CanonicalPinnedSourceProjectionV1 {
        &self.pinned_source_projection_v1
    }

    /// Seal the exact current-stage Data+population workspace using Data's
    /// already-resolved recipe and crate-owned Classic TA implementation
    /// capability. Search supplies checked population plans resolved from the
    /// exact native admission facts. The stage sealer mints the sizing
    /// authority; legacy population-sizing receipts are not accepted here.
    pub fn seal_data_population_workspace_plan_v1(
        &self,
        native_admission_facts: SealedNativeCudaDataPopulationPreflightFactsV1,
        max_ordered_index_count: usize,
        max_adaptive_row_count: usize,
        gene_plan: PopulationGeneStorePlanV1,
        metrics_plan: PopulationMetricsOnlyPlanV1,
    ) -> std::result::Result<
        SealedDataPopulationGpuWorkspacePlanV1,
        GpuOnlyFeatureMaterializationErrorV3,
    > {
        let classic_ta_capability = self
            .preflight
            .plan
            .producer_capabilities
            .capabilities()
            .iter()
            .find(|capability| capability.producer() == ResidentFeatureProducerV3::ClassicTa)
            .cloned()
            .ok_or_else(|| {
                GpuOnlyFeatureMaterializationErrorV3::Other(anyhow::anyhow!(
                    "prepared resident recipe lacks its Classic TA capability"
                ))
            })?;
        seal_data_population_gpu_workspace_plan_v1(DataPopulationWorkspacePreflightRequestV1 {
            native_admission_facts,
            data_extent: self.workspace_extent.clone(),
            max_ordered_index_count,
            max_adaptive_row_count,
            gene_plan,
            metrics_plan,
            classic_ta_capability,
        })
        .map_err(Into::into)
    }
}

pub fn prepare_gpu_only_feature_materialization_v3(
    workspace_preflight: PreparedGpuOnlyFeatureWorkspacePreflightV3,
) -> std::result::Result<
    PreparedGpuOnlyFeatureMaterializationV3,
    GpuOnlyFeatureMaterializationErrorV3,
> {
    let (plan, producers) = CrateOwnedResidentProducerFactoryV3::resolve(workspace_preflight)?;
    let workspace_extent = exact_resident_working_set_extent_request_v3(&plan)?.seal()?;
    let preflight = preflight_gpu_only_feature_recipe_v3(plan)?;
    let pinned_source_projection_v1 =
        derive_pinned_source_projection_v1(preflight.resident_sources())
            .map_err(|error| GpuOnlyFeatureMaterializationErrorV3::Other(error.into()))?;
    Ok(PreparedGpuOnlyFeatureMaterializationV3 {
        workspace_extent,
        pinned_source_projection_v1,
        preflight,
        producers,
    })
}

/// Compatibility entrypoint for callers that already own the admitted full
/// run. New orchestration should call `prepare_*` before workspace binding and
/// then consume the continuation through `materialize_prepared_*`.
pub fn materialize_gpu_only_feature_store_v3(
    workspace_preflight: PreparedGpuOnlyFeatureWorkspacePreflightV3,
    admitted_run: AdmittedNativeCudaFullDiscoveryRunV1,
) -> std::result::Result<SealedGpuResidentFeatureStoreV3, GpuOnlyFeatureMaterializationErrorV3> {
    let prepared = prepare_gpu_only_feature_materialization_v3(workspace_preflight)?;
    materialize_prepared_gpu_only_feature_store_v3(prepared, admitted_run)
}

/// The only Data entrypoint that may consume a staged exact recipe and
/// materialize a strict GPU-only resident feature store. No public producer
/// trait object, raw device pointer, host feature matrix, or fallback lane can
/// enter this sequence.
pub fn materialize_prepared_gpu_only_feature_store_v3(
    prepared: PreparedGpuOnlyFeatureMaterializationV3,
    admitted_run: AdmittedNativeCudaFullDiscoveryRunV1,
) -> std::result::Result<SealedGpuResidentFeatureStoreV3, GpuOnlyFeatureMaterializationErrorV3> {
    let run_device = admitted_run.into_gpu_only_run_device_admission_v3()?;
    materialize_prepared_gpu_only_feature_store_on_run_device_v3(prepared, run_device)
}

/// Current production Search stage: materialize the prepared Data recipe on
/// the exact Data+population carrier whose limits will remain attached to the
/// resident population session.
pub fn materialize_prepared_gpu_only_feature_store_for_data_population_v3(
    prepared: PreparedGpuOnlyFeatureMaterializationV3,
    admitted_run: AdmittedNativeCudaDataPopulationRunV1,
) -> std::result::Result<SealedGpuResidentFeatureStoreV3, GpuOnlyFeatureMaterializationErrorV3> {
    materialize_prepared_gpu_only_feature_store_on_run_device_v3(
        prepared,
        admitted_run.into_gpu_only_run_device_admission_v3(),
    )
}

fn materialize_prepared_gpu_only_feature_store_on_run_device_v3(
    prepared: PreparedGpuOnlyFeatureMaterializationV3,
    run_device: GpuOnlyRunDeviceAdmissionV3,
) -> std::result::Result<SealedGpuResidentFeatureStoreV3, GpuOnlyFeatureMaterializationErrorV3> {
    if let Some(limits) = run_device.data_population_limits()
        && (prepared.workspace_extent.identity_sha256() != limits.data_extent_identity_sha256()
            || prepared.workspace_extent.row_count() != limits.parent_row_count()
            || prepared.workspace_extent.column_count() != limits.feature_count())
    {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!(
                "prepared Data recipe extent does not match the exact Data+population stage plan"
            ),
        ));
    }
    let materialized_pinned_source_projection_v1 =
        derive_pinned_source_projection_v1(prepared.preflight.resident_sources())
            .map_err(|error| GpuOnlyFeatureMaterializationErrorV3::Other(error.into()))?;
    if prepared.pinned_source_projection_v1 != materialized_pinned_source_projection_v1 {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!(
                "prepared pinned-source projection does not match the immutable resident source leases"
            ),
        ));
    }
    let PreparedGpuOnlyFeatureMaterializationV3 {
        workspace_extent: _,
        pinned_source_projection_v1: _,
        preflight,
        mut producers,
    } = prepared;
    let smc_bindings = preflight.exact_bindings_for(ResidentFeatureProducerV3::Smc)?;
    let classic_bindings = preflight.exact_bindings_for(ResidentFeatureProducerV3::ClassicTa)?;
    let quant_bindings = preflight.exact_bindings_for(ResidentFeatureProducerV3::Quant)?;
    let session_bindings = preflight.exact_bindings_for(ResidentFeatureProducerV3::Session)?;
    let regime_bindings = preflight.exact_bindings_for(ResidentFeatureProducerV3::Regime)?;
    let footprint_bindings = preflight.exact_bindings_for(ResidentFeatureProducerV3::Footprint)?;
    let htf_bindings =
        preflight.exact_bindings_for(ResidentFeatureProducerV3::HigherTimeframeAlignment)?;
    let (classic_ta_recipe, classic_ta_memory) = producers.take_classic_ta_runtime()?;
    let quant_runtime = producers.take_quant_runtime()?;
    let session_runtime = producers.take_session_runtime()?;
    let (htf_runtime, htf_capture_templates) = producers.take_higher_timeframe_runtime()?;
    let regime_input = producers.take_regime_input()?;
    producers.prepare_smc(
        &run_device,
        preflight.resident_sources().base().frame(),
        smc_bindings,
    )?;
    let direct_parents = preflight.resident_sources().direct_parents();
    if direct_parents.len() != htf_capture_templates.len() || direct_parents.is_empty() {
        return Err(GpuOnlyFeatureMaterializationErrorV3::Other(
            anyhow::anyhow!(
                "resident HTF direct-parent source and capture-template censuses disagree"
            ),
        ));
    }
    let mut htf_captures: Vec<ValidatedResidentHigherTimeframeDirectParentCaptureV3> =
        Vec::with_capacity(direct_parents.len());
    for (direct_parent, capture_template) in direct_parents.iter().zip(htf_capture_templates) {
        let (parent_smc_bindings, pending_capture) =
            capture_template.into_smc_preparation_parts_v3();
        let parent_smc = prepare_smc_materialization_v3(
            &run_device,
            direct_parent.frame(),
            parent_smc_bindings,
        )?;
        htf_captures.push(pending_capture.capture_direct_parent_v3(&run_device, parent_smc)?);
    }
    let prepared_htf_append = htf_runtime
        .bind_captured_parents_v3(&run_device, htf_captures)?
        .bind_current_native_v3(&run_device, htf_bindings)?;
    let admission = bind_gpu_only_run_device_v3(preflight, run_device)?;
    let smc_materialization = producers.take_smc_materialization()?;
    let (seal_token, mut assembler, pending_smc_batch) =
        admission.begin_materialization(smc_materialization)?;
    pending_smc_batch.append_to(&mut assembler)?;
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    assembler.append_resident_classic_ta_recipe_v4(
        classic_ta_recipe,
        classic_bindings,
        classic_ta_memory,
    )?;
    let _quant_receipt = quant_runtime.append_to(&mut assembler, quant_bindings)?;
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    let (_session_admission, _session_receipt) =
        session_runtime.append_to(&mut assembler, session_bindings)?;
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    let _regime_receipt = regime_input.append_to(&mut assembler, regime_bindings)?;
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    let _footprint_receipt = assembler.append_resident_footprint_v2(footprint_bindings)?;
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    let (_htf_admission, _htf_receipt) = prepared_htf_append.append_to(&mut assembler)?;
    seal_token.apply_resident_robust_normalization_v2(&mut assembler)?;
    let owner = assembler.seal()?;
    seal_gpu_resident_feature_store_v3(seal_token, owner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_a2_capability_census_is_complete_and_canonical() {
        let manifest = require_complete_resident_producer_manifest_v3(
            current_resident_producer_capabilities_v3()
                .expect("resident producer capability construction must be exact"),
        )
        .expect("all ten resident capabilities must seal");
        assert_eq!(
            manifest.capabilities().len(),
            ResidentFeatureProducerV3::ALL.len()
        );
        assert!(
            manifest
                .capabilities()
                .iter()
                .map(ResidentProducerCapabilityV3::producer)
                .eq(ResidentFeatureProducerV3::ALL)
        );
    }

    #[test]
    fn resident_execution_projection_rejects_vector_build_and_math_mutation() {
        let classic = resident_classic_ta_capability_v3()
            .expect("current resident Classic TA capability must seal");
        let exact_math = vector_ta::cuda::F64_EXACT_MATH_AUTHORITY_V3;
        let device = CudaPrimaryContextBuildIdentityV3::new(
            0,
            [1; 16],
            8,
            6,
            [2; 32],
            "fixture-driver",
            "fixture-runtime",
            "fixture-nvcc",
            "sm_86",
            classic.implementation_sha256(),
            [3; 32],
            exact_math,
        )
        .expect("matching fixture identity must seal");
        let plan_sha256 = [4; 32];
        let projection =
            seal_gpu_resident_feature_execution_authority_v1(plan_sha256, &classic, &device)
                .expect("matching plan/capability/device authority must seal");
        assert_eq!(
            projection.semantic(),
            CanonicalGpuResidentFeatureExecutionSemanticV1::GpuCudaF64Strict
        );
        assert_eq!(projection.final_feature_plan_v3_sha256(), plan_sha256);
        assert_ne!(projection.identity_sha256(), [0; 32]);

        let wrong_build = CudaPrimaryContextBuildIdentityV3::new(
            0,
            [1; 16],
            8,
            6,
            [2; 32],
            "fixture-driver",
            "fixture-runtime",
            "fixture-nvcc",
            "sm_86",
            [5; 32],
            [3; 32],
            exact_math,
        )
        .expect("non-zero mismatched build is structurally valid");
        assert!(
            seal_gpu_resident_feature_execution_authority_v1(plan_sha256, &classic, &wrong_build,)
                .is_err(),
            "device vector build drift must fail before Search receipt mint",
        );

        let wrong_math = ResidentProducerCapabilityV3::new(
            ResidentFeatureProducerV3::ClassicTa,
            classic.implementation_id(),
            classic.implementation_sha256(),
            "neoethos.fixture.wrong-exact-math",
        )
        .expect("non-empty mismatched authority is structurally valid");
        assert!(
            seal_gpu_resident_feature_execution_authority_v1(plan_sha256, &wrong_math, &device,)
                .is_err(),
            "abstract or mutated math text must not authorize concrete resident execution",
        );

        let wrong_device_math = CudaPrimaryContextBuildIdentityV3::new(
            0,
            [1; 16],
            8,
            6,
            [2; 32],
            "fixture-driver",
            "fixture-runtime",
            "fixture-nvcc",
            "sm_86",
            classic.implementation_sha256(),
            [3; 32],
            "neoethos.fixture.wrong-device-exact-math",
        )
        .expect("non-empty mismatched device authority is structurally valid");
        assert!(
            seal_gpu_resident_feature_execution_authority_v1(
                plan_sha256,
                &classic,
                &wrong_device_math,
            )
            .is_err(),
            "device exact-math drift must fail before Search receipt mint",
        );
        assert!(
            seal_gpu_resident_feature_execution_authority_v1([0; 32], &classic, &device).is_err(),
            "a zero final FeaturePlan identity must never mint execution authority",
        );
    }
}
