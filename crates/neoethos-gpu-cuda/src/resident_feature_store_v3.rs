//! Incremental primary-context CUDA owner for one sealed V3 feature store.
//!
//! The assembler allocates the final Search layout once, appends one exact
//! producer batch directly into its monotonic destination range, and refuses a
//! second append until a real pack event proves the prior batch can be retired
//! without host synchronization. It never owns all producer feature columns at
//! once and never creates a full feature-major staging allocation.

use crate::data_population_workspace_plan_v1::SealedDataPopulationExecutionLimitsV1;
use crate::population::{
    CudaPopulationError, PopulationEvaluationViewV1, PopulationGeneView,
    PopulationResidencyCountersV1, PopulationSession, RawResidentFeatureStoreBindV3,
    ResidentAdaptiveBaseRequestV1, ResidentAdaptiveBaseViewTokenIdentityV1,
    ResidentAdaptiveBaseViewTokenV1, ResidentPopulationMetricsV1,
};
use crate::resident_classic_ta_v3::{
    ResidentClassicTaExecutorErrorV3, ResidentClassicTaExecutorV3,
    ResidentClassicTaPreDeviceMemoryReceiptV4, ResidentClassicTaRecipeV3,
};
use crate::resident_footprint_v2::{
    ResidentFootprintRuntimeReceiptV2, launch_resident_footprint_v2,
};
use crate::resident_generation_v1::SealedResidentGenerationPlanV1;
use crate::resident_higher_timeframe_alignment_v3::{
    ResidentHigherTimeframeDirectParentV3, ResidentHigherTimeframeExecutorV3,
    ResidentHigherTimeframeLaunchAuthorityV3, ResidentHigherTimeframeRuntimeReceiptV3,
};
use crate::resident_quant_v3::{
    ResidentQuantLaunchAuthorityV3, ResidentQuantRuntimeReceiptV3, launch_resident_quant_v3,
};
use crate::resident_regime_v3::{ResidentRegimeRuntimeReceiptV3, launch_resident_regime_v3};
use crate::resident_robust_normalization_v2::{
    ResidentRobustNormalizationPlanV2, ResidentRobustNormalizationRuntimeReceiptV2,
    disabled_resident_robust_normalization_receipt_v2, launch_resident_robust_normalization_v2,
};
use crate::resident_search_v2::{ResidentSearchRunV2, ResidentSearchV2Error};
use crate::resident_session_v2::{
    ResidentSessionLaunchAuthorityV2, ResidentSessionRuntimeReceiptV2, launch_resident_session_v2,
};
use crate::resident_trim_prefilter_v1::{
    RESIDENT_TRIM_PREFILTER_CUDA_MATH_FLAGS_V1, ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    ResidentTrimPrefilterImportIdentityV1, ResidentTrimPrefilterInputsV1,
    ResidentTrimPrefilterParentImportV1, SealedResidentColumnClassificationV1,
};
use crate::{NeoPopulationSettings, ScenarioDescriptor};
use cust::context::{Context, CurrentContext};
use cust::memory::{
    AsyncCopyDestination, CopyDestination, DeviceBuffer, DeviceCopy, GpuBuffer, LockedBuffer,
    mem_get_info,
};
use cust::stream::Stream;
use cust::sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING;
use cust::sys::cudaError_enum::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS};
use cust::sys::{
    CUcontext, CUevent, CUresult, CUstream, cuEventCreate, cuEventDestroy_v2, cuEventQuery,
    cuEventRecord, cuEventSynchronize, cuStreamGetCtx, cuStreamSynchronize, cuStreamWaitEvent,
};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    CudaPrimaryContextBuildIdentityV3, PORTABLE_CUDA_SHA256_AUTHORITY_V3,
    ResidentFeatureProducerV3, ResidentFeatureRouteV3, ResidentParentDatasetLayoutV4,
    ResidentProducerCapabilityV3, ResidentReadyEventV3, ResidentWorkingSetBoundV3,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[cfg(all(test, feature = "cuda"))]
#[path = "resident_population_session_v3_device_tests.rs"]
mod resident_population_session_v3_device_tests;

const SHA256_BYTES: usize = 32;
const SMC_SLOTS_V3: usize = 11;
const CANONICAL_MERKLE_CHUNK_ROWS_V3: usize = 4096;
const VALIDITY_ATOMIC_ALIGNMENT_BYTES: usize = 4;
pub const RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3: &str =
    "neoethos.cuda.exact-allocator-context-reserve.v3";
pub const FEATURE_MAJOR_TO_BAR_MAJOR_EXACT_AUTHORITY_V3: &str = "f64-bit-pattern-preserving-tiled-transpose;logical-validity-u8-codes-0-through-9;physical-validity-u4-low-nibble-first;zero-full-feature-major-staging";

/// Bind the ordered producer census to the real in-tree CUDA transpose/pack
/// implementation. The returned DTO is descriptive capability evidence only;
/// gpu-cuda's resident owner remains the allocation and lifetime authority.
pub fn resident_feature_major_to_bar_major_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation
        .update(b"neoethos.gpu-cuda.resident-feature-major-to-bar-major.f64-u4.semantic-v3");
    implementation.update(include_bytes!("resident_feature_store_v3.rs"));
    implementation.update(include_bytes!("../native/resident_feature_store_v3.cu"));
    implementation.update(FEATURE_MAJOR_TO_BAR_MAJOR_EXACT_AUTHORITY_V3.as_bytes());
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::FeatureMajorToBarMajor,
        "neoethos.gpu-cuda.resident-feature-major-to-bar-major.f64-u4.semantic-v3",
        implementation_sha256,
        FEATURE_MAJOR_TO_BAR_MAJOR_EXACT_AUTHORITY_V3,
    )
    .map_err(Into::into)
}

/// Bind the ordered producer census to the real portable in-tree parallel
/// Merkle SHA-256 kernels. The returned value cannot provide allocation,
/// context, stream, event, or compact-root readback authority.
pub fn resident_canonical_content_sha256_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-canonical-content-sha256.semantic-v3");
    implementation.update(include_bytes!("resident_feature_store_v3.rs"));
    implementation.update(include_bytes!("../native/resident_feature_store_v3.cu"));
    implementation.update(PORTABLE_CUDA_SHA256_AUTHORITY_V3.as_bytes());
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::CanonicalContentSha256,
        "neoethos.gpu-cuda.resident-canonical-content-sha256.semantic-v3",
        implementation_sha256,
        PORTABLE_CUDA_SHA256_AUTHORITY_V3,
    )
    .map_err(Into::into)
}

unsafe extern "C" {
    fn neoethos_resident_initialize_validity_u4_v3(
        search_bar_major_validity_u4: *mut u8,
        logical_bytes: usize,
        allocated_bytes: usize,
        validity_code_error: *mut u32,
        stream: CUstream,
    ) -> i32;
    fn neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(
        source_addresses: *const u64,
        source_offsets: *const u64,
        source_validity_addresses: *const u64,
        source_validity_offsets: *const u64,
        rows: usize,
        source_columns: usize,
        destination_columns: usize,
        destination_column_start: usize,
        search_bar_major_values: *mut f64,
        search_bar_major_validity_u4: *mut u8,
        validity_code_error: *mut u32,
        stream: CUstream,
    ) -> i32;
    fn neoethos_resident_canonical_merkle_sha256_v3(
        timestamps: *const i64,
        rows: usize,
        columns: usize,
        name_offsets: *const u64,
        name_bytes: *const u8,
        search_bar_major_values: *const f64,
        search_bar_major_validity_u4: *const u8,
        merkle_scratch_a: *mut u8,
        merkle_scratch_b: *mut u8,
        merkle_scratch_digest_capacity: usize,
        digest: *mut u8,
        stream: CUstream,
    ) -> i32;
}

#[derive(Debug, Error)]
pub enum ResidentFeatureStoreCudaErrorV3 {
    #[error("invalid resident feature store input: {0}")]
    InvalidInput(String),
    #[error("resident feature store arithmetic overflowed at {0}")]
    ArithmeticOverflow(&'static str),
    #[error("CUDA driver call `{operation}` failed with status {status}")]
    Driver {
        operation: &'static str,
        status: i32,
    },
    #[error("resident CUDA operation `{operation}` failed with status {status}")]
    Native {
        operation: &'static str,
        status: i32,
    },
    #[error("resident store event has not completed")]
    NotReady,
    #[error("a producer batch is still awaiting event-proven retirement")]
    ProducerBatchPending,
    #[error("consumer or producer device ordinal differs from the resident store")]
    DeviceMismatch,
    #[error("consumer or producer CUDA context is not the resident store primary context")]
    PrimaryContextMismatch,
    #[error("producer stream is not the admitted resident-store stream")]
    ProducerStreamMismatch,
    #[error("producer emitted a validity code outside the sealed 0..=9 schema")]
    InvalidProducerValidityCode,
    #[error(
        "same-context post-parent free memory cannot cover the remaining admitted peak: required {required_bytes} bytes, observed available {observed_available_bytes} bytes"
    )]
    RuntimeFreeMemoryChanged {
        required_bytes: u64,
        observed_available_bytes: u64,
    },
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error(transparent)]
    Contract(
        #[from] neoethos_gpu_contracts::resident_feature_store_v3::ResidentFeatureContractErrorV3,
    ),
    #[error(transparent)]
    Population(#[from] CudaPopulationError),
    #[error(transparent)]
    ResidentSearch(#[from] ResidentSearchV2Error),
}

/// Opaque full-workspace slice reserved for the resident trim/prefilter. Its
/// constructor is crate-private; public consumers can inspect only bounded
/// byte and identity facts, never CUDA handles.
#[derive(Debug)]
pub struct SealedFullDiscoveryTrimAdmissionV1 {
    workspace_plan_identity_sha256: [u8; SHA256_BYTES],
    required_workspace_bytes: u64,
    trim_prefilter_reserved_bytes: u64,
    full_discovery_reserve_bytes: u64,
}

impl SealedFullDiscoveryTrimAdmissionV1 {
    pub(crate) const fn new(
        workspace_plan_identity_sha256: [u8; SHA256_BYTES],
        required_workspace_bytes: u64,
        trim_prefilter_reserved_bytes: u64,
        full_discovery_reserve_bytes: u64,
    ) -> Self {
        Self {
            workspace_plan_identity_sha256,
            required_workspace_bytes,
            trim_prefilter_reserved_bytes,
            full_discovery_reserve_bytes,
        }
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.workspace_plan_identity_sha256
    }

    pub const fn required_workspace_bytes(&self) -> u64 {
        self.required_workspace_bytes
    }

    pub const fn trim_prefilter_reserved_bytes(&self) -> u64 {
        self.trim_prefilter_reserved_bytes
    }

    pub const fn full_discovery_reserve_bytes(&self) -> u64 {
        self.full_discovery_reserve_bytes
    }
}

/// Host-metadata-only classification sealed by Search's single CPU/schema
/// authority. gpu-cuda validates its exact shape and uploads it once on the
/// admitted stream; it never recomputes classification from a second rule set.
#[derive(Debug)]
pub struct ResidentTrimPrefilterSchemaUploadV1 {
    canonical_search_input_receipt_sha256: [u8; SHA256_BYTES],
    canonical_content_merkle_sha256: [u8; SHA256_BYTES],
    normalization_fit_sha256: [u8; SHA256_BYTES],
    feature_plan_sha256: [u8; SHA256_BYTES],
    source_provenance_sha256: [u8; SHA256_BYTES],
    ordered_feature_schema_sha256: [u8; SHA256_BYTES],
    column_classification_content_sha256: [u8; SHA256_BYTES],
    column_class_flags: Vec<u8>,
    timeframe_group_ids: Vec<u32>,
    template_force_keep_flags: Vec<u8>,
    timeframe_group_count: u64,
}

impl ResidentTrimPrefilterSchemaUploadV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        canonical_search_input_receipt_sha256: [u8; SHA256_BYTES],
        canonical_content_merkle_sha256: [u8; SHA256_BYTES],
        normalization_fit_sha256: [u8; SHA256_BYTES],
        feature_plan_sha256: [u8; SHA256_BYTES],
        source_provenance_sha256: [u8; SHA256_BYTES],
        ordered_feature_schema_sha256: [u8; SHA256_BYTES],
        column_classification_content_sha256: [u8; SHA256_BYTES],
        column_class_flags: Vec<u8>,
        timeframe_group_ids: Vec<u32>,
        template_force_keep_flags: Vec<u8>,
        timeframe_group_count: u64,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let column_count = column_class_flags.len();
        let hashes = [
            canonical_search_input_receipt_sha256,
            canonical_content_merkle_sha256,
            normalization_fit_sha256,
            feature_plan_sha256,
            source_provenance_sha256,
            ordered_feature_schema_sha256,
            column_classification_content_sha256,
        ];
        let group_ids = timeframe_group_ids
            .iter()
            .copied()
            .filter(|group| *group != 0)
            .collect::<BTreeSet<_>>();
        let exact_group_ids = u64::try_from(group_ids.len())
            .ok()
            .is_some_and(|count| count == timeframe_group_count)
            && group_ids
                .iter()
                .copied()
                .eq(1..=u32::try_from(timeframe_group_count).unwrap_or(0));
        let exact_classes = column_class_flags
            .iter()
            .zip(&template_force_keep_flags)
            .all(|(class, force_keep)| {
                *class & !0b11 == 0
                    && *force_keep <= 1
                    && (*force_keep == 1) == (*class & 0b10 != 0)
            });
        if column_count == 0
            || timeframe_group_ids.len() != column_count
            || template_force_keep_flags.len() != column_count
            || timeframe_group_count == 0
            || timeframe_group_ids
                .iter()
                .any(|group| u64::from(*group) > timeframe_group_count)
            || !exact_group_ids
            || !exact_classes
            || hashes.iter().any(|hash| *hash == [0; SHA256_BYTES])
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim schema upload is not an exact sealed classification".into(),
            ));
        }
        Ok(Self {
            canonical_search_input_receipt_sha256,
            canonical_content_merkle_sha256,
            normalization_fit_sha256,
            feature_plan_sha256,
            source_provenance_sha256,
            ordered_feature_schema_sha256,
            column_classification_content_sha256,
            column_class_flags,
            timeframe_group_ids,
            template_force_keep_flags,
            timeframe_group_count,
        })
    }
}

#[derive(Debug)]
struct PendingResidentTrimSchemaUploadV1 {
    host_column_class_flags: Option<LockedBuffer<u8>>,
    host_timeframe_group_ids: Option<LockedBuffer<u32>>,
    host_template_force_keep_flags: Option<LockedBuffer<u8>>,
    column_class_flags: Option<StreamOrderedDeviceBufferV3<u8>>,
    timeframe_group_ids: Option<StreamOrderedDeviceBufferV3<u32>>,
    template_force_keep_flags: Option<StreamOrderedDeviceBufferV3<u8>>,
    ready_event: Option<OwnedCudaEventV3>,
}

impl PendingResidentTrimSchemaUploadV1 {
    fn new(ready_event: OwnedCudaEventV3) -> Self {
        Self {
            host_column_class_flags: None,
            host_timeframe_group_ids: None,
            host_template_force_keep_flags: None,
            column_class_flags: None,
            timeframe_group_ids: None,
            template_force_keep_flags: None,
            ready_event: Some(ready_event),
        }
    }

    fn into_lifetime(mut self) -> ResidentTrimSchemaLifetimeV1 {
        ResidentTrimSchemaLifetimeV1 {
            _host_column_class_flags: self
                .host_column_class_flags
                .take()
                .expect("sealed trim schema retains class upload"),
            _host_timeframe_group_ids: self
                .host_timeframe_group_ids
                .take()
                .expect("sealed trim schema retains timeframe upload"),
            _host_template_force_keep_flags: self
                .host_template_force_keep_flags
                .take()
                .expect("sealed trim schema retains template upload"),
            column_class_flags: self
                .column_class_flags
                .take()
                .expect("sealed trim schema retains class device buffer"),
            timeframe_group_ids: self
                .timeframe_group_ids
                .take()
                .expect("sealed trim schema retains timeframe device buffer"),
            template_force_keep_flags: self
                .template_force_keep_flags
                .take()
                .expect("sealed trim schema retains template device buffer"),
            ready_event: self
                .ready_event
                .take()
                .expect("sealed trim schema retains ready event"),
        }
    }
}

impl Drop for PendingResidentTrimSchemaUploadV1 {
    fn drop(&mut self) {
        // Any failed async allocation/copy/record may have retained every host
        // and device address. Without a terminal event there is no safe host
        // wait or retry boundary, so retire all identities by deliberate leak.
        if let Some(owner) = self.host_column_class_flags.take() {
            std::mem::forget(owner);
        }
        if let Some(owner) = self.host_timeframe_group_ids.take() {
            std::mem::forget(owner);
        }
        if let Some(owner) = self.host_template_force_keep_flags.take() {
            std::mem::forget(owner);
        }
        if let Some(owner) = self.column_class_flags.take() {
            std::mem::forget(owner);
        }
        if let Some(owner) = self.timeframe_group_ids.take() {
            std::mem::forget(owner);
        }
        if let Some(owner) = self.template_force_keep_flags.take() {
            std::mem::forget(owner);
        }
        if let Some(owner) = self.ready_event.take() {
            std::mem::forget(owner);
        }
    }
}

#[derive(Debug)]
struct ResidentTrimSchemaLifetimeV1 {
    _host_column_class_flags: LockedBuffer<u8>,
    _host_timeframe_group_ids: LockedBuffer<u32>,
    _host_template_force_keep_flags: LockedBuffer<u8>,
    column_class_flags: StreamOrderedDeviceBufferV3<u8>,
    timeframe_group_ids: StreamOrderedDeviceBufferV3<u32>,
    template_force_keep_flags: StreamOrderedDeviceBufferV3<u8>,
    ready_event: OwnedCudaEventV3,
}

#[derive(Debug)]
struct ResidentTrimAdmissionLifetimeV1 {
    _owner: Arc<ResidentFeatureStoreOwnerV3>,
    ready_event: OwnedCudaEventV3,
}

/// One process-local, gpu-cuda-owned device admission carried from the single
/// pre-materialization probe through Data materialization and consumed by the
/// strict Search session. It has no public constructor and cannot be rebuilt
/// from an ordinal, low-level receipt, or caller capability flag.
#[derive(Debug)]
pub struct GpuOnlyRunDeviceAdmissionV3 {
    admission_identity_sha256: [u8; SHA256_BYTES],
    workspace_plan_identity_sha256: [u8; SHA256_BYTES],
    device_identity: CudaPrimaryContextBuildIdentityV3,
    device_uuid: [u8; 16],
    compute_capability_major: u16,
    compute_capability_minor: u16,
    run_stream: Arc<Stream>,
    primary_context: Arc<Context>,
    phase_one_free_bytes_snapshot: u64,
    allocator_context_reserve_bytes: u64,
    reserve_policy_id: &'static str,
    data_population_limits: Option<SealedDataPopulationExecutionLimitsV1>,
    full_discovery_trim_admission: Option<SealedFullDiscoveryTrimAdmissionV1>,
}

#[derive(Debug)]
pub(crate) struct GpuOnlyRunDeviceAdmissionRequestV3 {
    pub(crate) source_admission_identity_sha256: [u8; SHA256_BYTES],
    pub(crate) workspace_plan_identity_sha256: [u8; SHA256_BYTES],
    pub(crate) selected_device_ordinal: u32,
    pub(crate) device_uuid: [u8; 16],
    pub(crate) compute_capability_major: u16,
    pub(crate) compute_capability_minor: u16,
    pub(crate) run_stream: Arc<Stream>,
    pub(crate) primary_context: Arc<Context>,
    pub(crate) driver_version: String,
    pub(crate) context_api_version: String,
    pub(crate) nvcc_version: String,
    pub(crate) native_sass_target: String,
    pub(crate) vector_ta_build_sha256: [u8; SHA256_BYTES],
    pub(crate) gpu_cuda_build_sha256: [u8; SHA256_BYTES],
    pub(crate) exact_math_authority: String,
    pub(crate) phase_one_free_bytes_snapshot: u64,
    pub(crate) allocator_context_reserve_bytes: u64,
    pub(crate) data_population_limits: Option<SealedDataPopulationExecutionLimitsV1>,
    pub(crate) full_discovery_trim_admission: Option<SealedFullDiscoveryTrimAdmissionV1>,
}

impl GpuOnlyRunDeviceAdmissionV3 {
    pub const fn admission_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.admission_identity_sha256
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.workspace_plan_identity_sha256
    }

    pub const fn data_population_limits(&self) -> Option<&SealedDataPopulationExecutionLimitsV1> {
        self.data_population_limits.as_ref()
    }

    pub const fn full_discovery_trim_admission(
        &self,
    ) -> Option<&SealedFullDiscoveryTrimAdmissionV1> {
        self.full_discovery_trim_admission.as_ref()
    }

    pub fn device_identity(&self) -> &CudaPrimaryContextBuildIdentityV3 {
        &self.device_identity
    }

    pub(crate) fn primary_context_for_resident_producer_v3(&self) -> &Arc<Context> {
        &self.primary_context
    }

    pub(crate) fn run_stream_for_resident_producer_v3(&self) -> &Arc<Stream> {
        &self.run_stream
    }

    pub const fn phase_one_free_bytes_snapshot(&self) -> u64 {
        self.phase_one_free_bytes_snapshot
    }

    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
    }

    pub const fn reserve_policy_id(&self) -> &'static str {
        self.reserve_policy_id
    }

    pub fn run_stream_process_token_v3(&self) -> [u8; SHA256_BYTES] {
        process_handle_token_v3(
            b"neoethos.cuda.run-stream-process-token.v3",
            self.admission_identity_sha256,
            self.run_stream.as_inner() as usize,
            0,
        )
    }
}

pub(crate) fn seal_gpu_only_run_device_admission_v3(
    request: GpuOnlyRunDeviceAdmissionRequestV3,
) -> Result<GpuOnlyRunDeviceAdmissionV3, ResidentFeatureStoreCudaErrorV3> {
    if request.source_admission_identity_sha256 == [0; SHA256_BYTES]
        || request.workspace_plan_identity_sha256 == [0; SHA256_BYTES]
        || request.phase_one_free_bytes_snapshot == 0
        || request.allocator_context_reserve_bytes == 0
        || request.driver_version.trim().is_empty()
        || request.context_api_version.trim().is_empty()
        || request.nvcc_version.trim().is_empty()
        || request.exact_math_authority.trim().is_empty()
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "GPU-only run-device authority is incomplete".into(),
        ));
    }
    if request.data_population_limits.is_some() == request.full_discovery_trim_admission.is_some() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "GPU-only admission must carry exactly one Data+population or full-Discovery workspace authority"
                .into(),
        ));
    }
    if request
        .data_population_limits
        .as_ref()
        .is_some_and(|limits| {
            limits.workspace_plan_identity_sha256() != request.workspace_plan_identity_sha256
        })
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "Data+population execution limits do not match the workspace plan".into(),
        ));
    }
    if let Some(trim) = request.full_discovery_trim_admission.as_ref() {
        let reserve_fits = request
            .phase_one_free_bytes_snapshot
            .checked_sub(request.allocator_context_reserve_bytes)
            .is_some_and(|available| available >= trim.full_discovery_reserve_bytes);
        if trim.workspace_plan_identity_sha256 != request.workspace_plan_identity_sha256
            || trim.required_workspace_bytes == 0
            || trim.trim_prefilter_reserved_bytes == 0
            || trim.full_discovery_reserve_bytes == 0
            || trim.trim_prefilter_reserved_bytes > trim.required_workspace_bytes
            || trim.full_discovery_reserve_bytes != trim.required_workspace_bytes
            || !reserve_fits
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "full-Discovery trim admission does not match the sealed workspace or admitted reserve"
                    .into(),
            ));
        }
    }
    let admission_identity_sha256 = hash_gpu_only_run_device_admission_v3(&request);
    let primary_context_process_token = process_handle_token_v3(
        b"neoethos.cuda.primary-context-process-token.v3",
        admission_identity_sha256,
        request.primary_context.as_raw() as usize,
        0,
    );
    let device_identity = CudaPrimaryContextBuildIdentityV3::new(
        request.selected_device_ordinal,
        request.device_uuid,
        request.compute_capability_major,
        request.compute_capability_minor,
        primary_context_process_token,
        request.driver_version,
        request.context_api_version,
        request.nvcc_version,
        request.native_sass_target,
        request.vector_ta_build_sha256,
        request.gpu_cuda_build_sha256,
        request.exact_math_authority,
    )?;
    Ok(GpuOnlyRunDeviceAdmissionV3 {
        admission_identity_sha256,
        workspace_plan_identity_sha256: request.workspace_plan_identity_sha256,
        device_identity,
        device_uuid: request.device_uuid,
        compute_capability_major: request.compute_capability_major,
        compute_capability_minor: request.compute_capability_minor,
        run_stream: request.run_stream,
        primary_context: request.primary_context,
        phase_one_free_bytes_snapshot: request.phase_one_free_bytes_snapshot,
        allocator_context_reserve_bytes: request.allocator_context_reserve_bytes,
        reserve_policy_id: RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3,
        data_population_limits: request.data_population_limits,
        full_discovery_trim_admission: request.full_discovery_trim_admission,
    })
}

fn hash_gpu_only_run_device_admission_v3(
    request: &GpuOnlyRunDeviceAdmissionRequestV3,
) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.gpu-only-run-device-admission.v3");
    hasher.update(request.source_admission_identity_sha256);
    hasher.update(request.workspace_plan_identity_sha256);
    hasher.update(request.selected_device_ordinal.to_le_bytes());
    hasher.update(request.device_uuid);
    hasher.update(request.compute_capability_major.to_le_bytes());
    hasher.update(request.compute_capability_minor.to_le_bytes());
    hasher.update((request.primary_context.as_raw() as usize as u64).to_le_bytes());
    hasher.update((request.run_stream.as_inner() as usize as u64).to_le_bytes());
    hasher.update(request.driver_version.as_bytes());
    hasher.update(request.context_api_version.as_bytes());
    hasher.update(request.nvcc_version.as_bytes());
    hasher.update(request.native_sass_target.as_bytes());
    hasher.update(request.vector_ta_build_sha256);
    hasher.update(request.gpu_cuda_build_sha256);
    hasher.update(request.exact_math_authority.as_bytes());
    hasher.update(request.phase_one_free_bytes_snapshot.to_le_bytes());
    hasher.update(request.allocator_context_reserve_bytes.to_le_bytes());
    hasher.update(RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3.as_bytes());
    if let Some(limits) = request.data_population_limits.as_ref() {
        hasher.update(b"data-population-stage-v1");
        hasher.update(limits.population_sizing_authority_sha256());
        hasher.update(limits.data_extent_identity_sha256());
        hasher.update(limits.parent_row_count().to_le_bytes());
        hasher.update(limits.feature_count().to_le_bytes());
        hasher.update(limits.max_ordered_index_count().to_le_bytes());
        hasher.update(limits.max_adaptive_row_count().to_le_bytes());
        hasher.update(limits.max_candidate_count().to_le_bytes());
        hasher.update(limits.max_gene_term_count().to_le_bytes());
        hasher.update(limits.max_concurrent_scenario_count().to_le_bytes());
        hasher.update(limits.month_capacity().to_le_bytes());
        hasher.update(limits.bounded_host_metric_readback_bytes().to_le_bytes());
    }
    if let Some(trim) = request.full_discovery_trim_admission.as_ref() {
        hasher.update(b"full-discovery-trim-stage-v1");
        hasher.update(trim.workspace_plan_identity_sha256);
        hasher.update(trim.required_workspace_bytes.to_le_bytes());
        hasher.update(trim.trim_prefilter_reserved_bytes.to_le_bytes());
        hasher.update(trim.full_discovery_reserve_bytes.to_le_bytes());
    }
    hasher.finalize().into()
}

fn process_handle_token_v3(
    domain: &[u8],
    admission_identity_sha256: [u8; SHA256_BYTES],
    handle: usize,
    sequence: u64,
) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(admission_identity_sha256);
    hasher.update((handle as u64).to_le_bytes());
    hasher.update(sequence.to_le_bytes());
    hasher.finalize().into()
}

fn trim_identity_sha256_v1(domain: &[u8], parts: &[&[u8]]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn ordered_feature_schema_sha256_v1(
    bindings: &[ResidentFeatureColumnBindingV3],
) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.ordered-prefilter-feature-schema.v1");
    hasher.update((bindings.len() as u64).to_le_bytes());
    for binding in bindings {
        hasher.update((binding.feature_name.len() as u64).to_le_bytes());
        hasher.update(binding.feature_name.as_bytes());
    }
    hasher.finalize().into()
}

fn driver_result(
    operation: &'static str,
    status: CUresult,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(ResidentFeatureStoreCudaErrorV3::Driver {
            operation,
            status: status as i32,
        })
    }
}

fn native_result(
    operation: &'static str,
    status: i32,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    if status == 0 {
        Ok(())
    } else {
        Err(ResidentFeatureStoreCudaErrorV3::Native { operation, status })
    }
}

fn compact_device_buffer_from_slice_async<T: DeviceCopy>(
    source: &[T],
    context: &Arc<Context>,
    stream: &Arc<Stream>,
) -> Result<(LockedBuffer<T>, StreamOrderedDeviceBufferV3<T>), ResidentFeatureStoreCudaErrorV3> {
    let locked_source = LockedBuffer::from_slice(source)?;
    let mut destination = StreamOrderedDeviceBufferV3::<T>::uninitialized_async(
        source.len(),
        Arc::clone(context),
        Arc::clone(stream),
    )?;
    // SAFETY: destination has exactly source.len() elements and remains owned
    // through all subsequently queued native work.
    if let Err(error) = unsafe { destination.async_copy_from(&locked_source, stream) } {
        // An async-copy error does not prove that the Driver did not retain the
        // host pointer. Never run either ordinary destructor on this path.
        std::mem::forget(locked_source);
        std::mem::forget(destination);
        return Err(error.into());
    }
    Ok((locked_source, destination))
}

#[derive(Debug)]
struct StreamOrderedDeviceBufferV3<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedDeviceBufferV3<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: the returned owner retains the exact context and stream and
        // only releases this allocation with stream-ordered drop_async.
        let buffer = unsafe { DeviceBuffer::<T>::uninitialized_async(len, &stream)? };
        Ok(Self {
            buffer: Some(buffer),
            context,
            stream,
        })
    }

    fn is_owned_by_stream(&self, stream: &Stream) -> bool {
        !stream.as_inner().is_null() && self.stream.as_inner() == stream.as_inner()
    }

    fn release_async(mut self, stream: &Stream) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self.is_owned_by_stream(stream) {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        CurrentContext::set_current(self.context.as_ref())?;
        let buffer = self.buffer.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "stream-ordered Data transient was already released".into(),
            )
        })?;
        buffer.drop_async(stream)?;
        Ok(())
    }
}

impl<T: DeviceCopy> Deref for StreamOrderedDeviceBufferV3<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live stream-ordered owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedDeviceBufferV3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live stream-ordered owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedDeviceBufferV3<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if CurrentContext::set_current(self.context.as_ref()).is_ok() {
            let _ = buffer.drop_async(&self.stream);
        } else {
            // Never fall back to DeviceBuffer's legacy destructor when the
            // exact primary context cannot be restored.
            std::mem::forget(buffer);
        }
    }
}

fn stream_context(stream: &Stream) -> Result<CUcontext, ResidentFeatureStoreCudaErrorV3> {
    let mut context = MaybeUninit::<CUcontext>::uninit();
    // SAFETY: the Driver writes one context handle on success.
    driver_result("cuStreamGetCtx", unsafe {
        cuStreamGetCtx(stream.as_inner(), context.as_mut_ptr())
    })?;
    // SAFETY: successful cuStreamGetCtx initialized the output.
    Ok(unsafe { context.assume_init() })
}

#[derive(Debug)]
struct OwnedCudaEventV3 {
    raw: CUevent,
}

unsafe impl Send for OwnedCudaEventV3 {}
unsafe impl Sync for OwnedCudaEventV3 {}

impl OwnedCudaEventV3 {
    fn new() -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let mut raw = MaybeUninit::<CUevent>::uninit();
        // SAFETY: CUDA initializes one opaque event on success.
        driver_result("cuEventCreate", unsafe {
            cuEventCreate(
                raw.as_mut_ptr(),
                CU_EVENT_DISABLE_TIMING as std::os::raw::c_uint,
            )
        })?;
        Ok(Self {
            // SAFETY: successful cuEventCreate initialized the handle.
            raw: unsafe { raw.assume_init() },
        })
    }

    fn record(&self, stream: &Stream) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: both owned handles remain live through the recorded work.
        driver_result("cuEventRecord", unsafe {
            cuEventRecord(self.raw, stream.as_inner())
        })
    }

    fn query(&self) -> Result<bool, ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: this owner retains the event.
        let status = unsafe { cuEventQuery(self.raw) };
        if status == CUDA_SUCCESS {
            Ok(true)
        } else if status == CUDA_ERROR_NOT_READY {
            Ok(false)
        } else {
            driver_result("cuEventQuery", status)?;
            unreachable!("driver_result accepted a non-success status")
        }
    }

    fn synchronize(&self) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: this owner retains the event until the explicit wait has
        // completed. This is used only for the bounded normalization verdict,
        // never for feature-value materialization.
        driver_result("cuEventSynchronize", unsafe {
            cuEventSynchronize(self.raw)
        })
    }

    fn enqueue_wait(&self, stream: &Stream) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: the event and stream are retained; callers first prove the
        // exact same primary context.
        driver_result("cuStreamWaitEvent", unsafe {
            cuStreamWaitEvent(stream.as_inner(), self.raw, 0)
        })
    }

    fn process_token(
        &self,
        admission_identity_sha256: [u8; SHA256_BYTES],
        sequence: u64,
    ) -> [u8; SHA256_BYTES] {
        process_handle_token_v3(
            b"neoethos.cuda.ready-event-process-token.v3",
            admission_identity_sha256,
            self.raw as usize,
            sequence,
        )
    }

    fn raw(&self) -> CUevent {
        self.raw
    }
}

impl Drop for OwnedCudaEventV3 {
    fn drop(&mut self) {
        // SAFETY: this type is the sole handle owner. Queued record/wait work
        // remains valid after event destruction under the Driver ABI.
        let _ = unsafe { cuEventDestroy_v2(self.raw) };
    }
}

/// Opaque proof that a producer recorded completion in the exact primary
/// context/stream it exposes. There is no raw-handle constructor.
#[derive(Debug)]
pub struct ResidentProducerReadyEventV3 {
    event: OwnedCudaEventV3,
    primary_context: CUcontext,
    producer_stream: CUstream,
    device_ordinal: u32,
}

unsafe impl Send for ResidentProducerReadyEventV3 {}
unsafe impl Sync for ResidentProducerReadyEventV3 {}

impl ResidentProducerReadyEventV3 {
    pub fn record(
        context: &Context,
        stream: &Stream,
        device_ordinal: u32,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        CurrentContext::set_current(context)?;
        let expected_ordinal = i32::try_from(device_ordinal).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("producer device ordinal ABI")
        })?;
        if CurrentContext::get_device()?.as_raw() != expected_ordinal
            || stream_context(stream)? != context.as_raw()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        let event = OwnedCudaEventV3::new()?;
        event.record(stream)?;
        Ok(Self {
            event,
            primary_context: context.as_raw(),
            producer_stream: stream.as_inner(),
            device_ordinal,
        })
    }

    pub(crate) fn wait_before_read(
        &self,
        context: &Context,
        stream: &Stream,
        device_ordinal: u32,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if self.device_ordinal != device_ordinal {
            return Err(ResidentFeatureStoreCudaErrorV3::DeviceMismatch);
        }
        if self.primary_context != context.as_raw() || stream_context(stream)? != context.as_raw() {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        if self.producer_stream.is_null() || stream.as_inner().is_null() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "default/null CUDA streams are not admitted".into(),
            ));
        }
        self.event.enqueue_wait(stream)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentFeatureColumnBindingV3 {
    pub ordinal: usize,
    pub feature_name: String,
    pub canonical_parameter_tuple_sha256: [u8; SHA256_BYTES],
    pub route_receipt_sha256: [u8; SHA256_BYTES],
}

impl ResidentFeatureColumnBindingV3 {
    pub fn from_admitted_route(
        route: &ResidentFeatureRouteV3,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        Ok(Self {
            ordinal: usize::try_from(route.ordinal()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("route ordinal")
            })?,
            feature_name: route.feature_name().to_owned(),
            canonical_parameter_tuple_sha256: route.canonical_parameter_tuple_sha256(),
            route_receipt_sha256: route.route_receipt_sha256(),
        })
    }
}

/// One already-resident producer batch. Implementors are internal ownership
/// authorities, not caller-provided capability booleans.
///
/// # Safety
///
/// Every returned buffer/context/stream must remain owned and valid until
/// `enqueue_nonblocking_release` consumes the object. That method must enqueue
/// stream-ordered releases and must never synchronize the host.
pub unsafe trait ResidentF64FeatureBatchV3: Send + std::fmt::Debug {
    fn column_bindings(&self) -> &[ResidentFeatureColumnBindingV3];
    fn value_buffer(&self, column: usize) -> &DeviceBuffer<f64>;
    fn validity_buffer(&self, column: usize) -> &DeviceBuffer<u8>;
    fn value_offset(&self, column: usize) -> usize;
    fn validity_offset(&self, column: usize) -> usize;
    fn rows(&self) -> usize;
    fn device_ordinal(&self) -> u32;
    fn producer_context(&self) -> &Context;
    fn producer_stream(&self) -> &Stream;
    fn producer_ready_event(&self) -> &ResidentProducerReadyEventV3;
    fn retained_device_bytes(&self) -> usize;
    fn retained_scratch_bytes(&self) -> usize;
    fn enqueue_nonblocking_release(
        self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3>;
}

/// Immutable native-evaluator parent arrays already resident in the admitted
/// primary context. Unlike feature batches, this owner is retained for the
/// entire Search store lifetime.
///
/// # Safety
///
/// The nine buffers must have the exact declared row extents and must share
/// the returned primary context, stream and ordinal. Destruction must not
/// insert a host synchronization while queued Search work is live.
pub unsafe trait ResidentParentDatasetSourceV3: Send + Sync + std::fmt::Debug {
    fn open(&self) -> &DeviceBuffer<f64>;
    fn close(&self) -> &DeviceBuffer<f64>;
    fn high(&self) -> &DeviceBuffer<f64>;
    fn low(&self) -> &DeviceBuffer<f64>;
    fn volume(&self) -> &DeviceBuffer<f64>;
    fn timestamps(&self) -> &DeviceBuffer<i64>;
    fn months(&self) -> &DeviceBuffer<i64>;
    fn days(&self) -> &DeviceBuffer<i64>;
    fn smc_rows(&self) -> &DeviceBuffer<i8>;
    fn rows(&self) -> usize;
    fn device_ordinal(&self) -> u32;
    fn producer_context(&self) -> &Context;
    fn producer_stream(&self) -> &Stream;
    fn producer_ready_event(&self) -> &ResidentProducerReadyEventV3;
    fn retained_device_bytes(&self) -> usize;
    fn parent_dataset_layout(&self) -> &ResidentParentDatasetLayoutV4;
    fn enqueue_nonblocking_release(
        self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentFeatureCompactHashesV3 {
    pub canonical_content_merkle: [u8; SHA256_BYTES],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentFeatureLayoutEvidenceV3 {
    pub rows: usize,
    pub columns: usize,
    pub canonical_content_merkle: [u8; SHA256_BYTES],
    pub source_column_count: usize,
    pub producer_batch_count: usize,
    pub validity_initialization_count: usize,
    pub value_layout_launch_count: usize,
    pub validity_boundary_launch_count: usize,
    pub layout_transform_value_bytes: usize,
    pub layout_transform_logical_validity_bytes: usize,
    pub packed_validity_logical_bytes: usize,
    pub packed_validity_allocated_bytes: usize,
    pub max_live_producer_bytes: usize,
    pub max_live_producer_scratch_bytes: usize,
    pub max_live_runtime_metadata_bytes: usize,
    pub footprint_runtime_receipt_v2: Option<ResidentFootprintRuntimeReceiptV2>,
    pub regime_runtime_receipt_v3: Option<ResidentRegimeRuntimeReceiptV3>,
    pub session_runtime_receipt_v2: Option<ResidentSessionRuntimeReceiptV2>,
    pub higher_timeframe_runtime_receipt_v3: Option<ResidentHigherTimeframeRuntimeReceiptV3>,
    pub robust_normalization_runtime_receipt_v2:
        Option<ResidentRobustNormalizationRuntimeReceiptV2>,
    pub full_feature_major_staging_bytes: usize,
    pub merkle_leaf_count: usize,
    pub merkle_scratch_bytes: usize,
    pub canonical_root_device_bytes: usize,
    pub validity_error_readback_count: usize,
    pub validity_error_d2h_bytes: usize,
    pub canonical_root_readback_count: usize,
    pub canonical_root_d2h_bytes: usize,
    pub compact_control_plane_d2h_bytes: usize,
    pub pre_materialization_free_bytes_snapshot: u64,
    pub post_parent_free_bytes_snapshot: u64,
    pub retained_parent_dataset_bytes: u64,
    pub remaining_peak_after_parent_bytes: u64,
    pub allocator_context_reserve_bytes: u64,
    pub reserve_policy_id: String,
}

#[derive(Debug)]
struct PendingResidentFeatureBatchV3 {
    batch: Box<dyn ResidentF64FeatureBatchV3>,
    host_pointer_tables: LockedBuffer<u64>,
    pointer_tables: StreamOrderedDeviceBufferV3<u64>,
    batch_ready_event: OwnedCudaEventV3,
}

#[derive(Debug)]
struct ResidentAppendTransactionV3 {
    batch: Option<Box<dyn ResidentF64FeatureBatchV3>>,
    host_pointer_tables: Option<LockedBuffer<u64>>,
    pointer_tables: Option<StreamOrderedDeviceBufferV3<u64>>,
    batch_ready_event: Option<OwnedCudaEventV3>,
}

impl ResidentAppendTransactionV3 {
    fn new(batch: Box<dyn ResidentF64FeatureBatchV3>) -> Self {
        Self {
            batch: Some(batch),
            host_pointer_tables: None,
            pointer_tables: None,
            batch_ready_event: None,
        }
    }

    fn batch(&self) -> &dyn ResidentF64FeatureBatchV3 {
        self.batch
            .as_deref()
            .expect("armed append transaction must own its producer batch")
    }

    fn install_pointer_tables(
        &mut self,
        host_pointer_tables: LockedBuffer<u64>,
        pointer_tables: StreamOrderedDeviceBufferV3<u64>,
    ) {
        self.host_pointer_tables = Some(host_pointer_tables);
        self.pointer_tables = Some(pointer_tables);
    }

    fn pointer_tables(&self) -> &StreamOrderedDeviceBufferV3<u64> {
        self.pointer_tables
            .as_ref()
            .expect("queued pointer tables must remain owned by append transaction")
    }

    fn install_ready_event(&mut self, event: OwnedCudaEventV3) {
        self.batch_ready_event = Some(event);
    }

    fn ready_event(&self) -> &OwnedCudaEventV3 {
        self.batch_ready_event
            .as_ref()
            .expect("append transaction must own its pack-ready event")
    }

    fn disarm(mut self) -> PendingResidentFeatureBatchV3 {
        PendingResidentFeatureBatchV3 {
            batch: self
                .batch
                .take()
                .expect("successful append retains producer batch"),
            host_pointer_tables: self
                .host_pointer_tables
                .take()
                .expect("successful append retains pinned pointer tables"),
            pointer_tables: self
                .pointer_tables
                .take()
                .expect("successful append retains device pointer tables"),
            batch_ready_event: self
                .batch_ready_event
                .take()
                .expect("successful append retains ready event"),
        }
    }
}

impl Drop for ResidentAppendTransactionV3 {
    fn drop(&mut self) {
        // This guard is armed at function entry. Any validation, allocation,
        // native-launch or event error therefore cannot invoke an ordinary
        // destructor for a producer owner or queued compact metadata.
        if let Some(batch) = self.batch.take() {
            std::mem::forget(batch);
        }
        if let Some(host) = self.host_pointer_tables.take() {
            std::mem::forget(host);
        }
        if let Some(device) = self.pointer_tables.take() {
            std::mem::forget(device);
        }
        if let Some(event) = self.batch_ready_event.take() {
            std::mem::forget(event);
        }
    }
}

impl PendingResidentFeatureBatchV3 {
    fn release(
        self,
        stream: &Stream,
        host_copy_is_complete: bool,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self.pointer_tables.is_owned_by_stream(stream) {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        let Self {
            batch,
            host_pointer_tables,
            pointer_tables,
            batch_ready_event: _,
        } = self;
        if !host_copy_is_complete {
            // Fail-closed early unwind: the Driver may still read this pinned
            // source. Leaking one tiny combined table is safer than UAF or a
            // forbidden host wait; normal event-proven retirement drops it.
            std::mem::forget(host_pointer_tables);
        }
        let batch_release = batch.enqueue_nonblocking_release(stream);
        drop(pointer_tables);
        batch_release?;
        Ok(())
    }
}

#[derive(Debug)]
struct ResidentAssemblerConstructionGuardV3 {
    run_device: Option<GpuOnlyRunDeviceAdmissionV3>,
    context: Arc<Context>,
    producer_stream: Arc<Stream>,
    device_ordinal: u32,
    parent_source: Option<Box<dyn ResidentParentDatasetSourceV3>>,
    search_bar_major_values: Option<StreamOrderedDeviceBufferV3<f64>>,
    search_bar_major_validity_u4: Option<StreamOrderedDeviceBufferV3<u8>>,
    validity_code_error: Option<StreamOrderedDeviceBufferV3<u32>>,
}

type ResidentAssemblerPartsV3 = (
    GpuOnlyRunDeviceAdmissionV3,
    Box<dyn ResidentParentDatasetSourceV3>,
    StreamOrderedDeviceBufferV3<f64>,
    StreamOrderedDeviceBufferV3<u8>,
    StreamOrderedDeviceBufferV3<u32>,
);

impl ResidentAssemblerConstructionGuardV3 {
    fn new(
        run_device: GpuOnlyRunDeviceAdmissionV3,
        parent_source: Box<dyn ResidentParentDatasetSourceV3>,
    ) -> Self {
        let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
        let producer_stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
        let device_ordinal = run_device.device_identity().ordinal();
        Self {
            run_device: Some(run_device),
            context,
            producer_stream,
            device_ordinal,
            parent_source: Some(parent_source),
            search_bar_major_values: None,
            search_bar_major_validity_u4: None,
            validity_code_error: None,
        }
    }

    fn parent_source(&self) -> &dyn ResidentParentDatasetSourceV3 {
        self.parent_source
            .as_deref()
            .expect("armed constructor guard retains parent source")
    }

    fn disarm(mut self) -> ResidentAssemblerPartsV3 {
        (
            self.run_device
                .take()
                .expect("successful constructor retains one-shot run-device admission"),
            self.parent_source
                .take()
                .expect("successful constructor retains parent source"),
            self.search_bar_major_values
                .take()
                .expect("successful constructor retains final values"),
            self.search_bar_major_validity_u4
                .take()
                .expect("successful constructor retains packed validity"),
            self.validity_code_error
                .take()
                .expect("successful constructor retains validity error flag"),
        )
    }
}

impl Drop for ResidentAssemblerConstructionGuardV3 {
    fn drop(&mut self) {
        let context_is_current = CurrentContext::set_current(self.context.as_ref()).is_ok();
        if let Some(parent) = self.parent_source.take() {
            let same_authority = context_is_current
                && parent.device_ordinal() == self.device_ordinal
                && parent.producer_context().as_raw() == self.context.as_raw()
                && parent.producer_stream().as_inner() == self.producer_stream.as_inner();
            if same_authority
                && parent
                    .producer_ready_event()
                    .wait_before_read(
                        self.context.as_ref(),
                        self.producer_stream.as_ref(),
                        self.device_ordinal,
                    )
                    .is_ok()
            {
                let _ = parent.enqueue_nonblocking_release(&self.producer_stream);
            } else {
                std::mem::forget(parent);
            }
        }
        drop(self.search_bar_major_values.take());
        drop(self.search_bar_major_validity_u4.take());
        drop(self.validity_code_error.take());
        if let Some(run_device) = self.run_device.take() {
            // A constructor error can occur after a producer-event wait or
            // async allocation was queued. Retain the one-shot context/stream
            // authority instead of invoking an implicit stream/context wait.
            std::mem::forget(run_device);
        }
    }
}

#[derive(Debug)]
struct ResidentSealTransactionV3 {
    host_name_offsets: Option<LockedBuffer<u64>>,
    host_name_bytes: Option<LockedBuffer<u8>>,
    name_offsets: Option<StreamOrderedDeviceBufferV3<u64>>,
    name_bytes: Option<StreamOrderedDeviceBufferV3<u8>>,
    merkle_scratch_a: Option<StreamOrderedDeviceBufferV3<u8>>,
    merkle_scratch_b: Option<StreamOrderedDeviceBufferV3<u8>>,
    canonical_content_merkle: Option<StreamOrderedDeviceBufferV3<u8>>,
    ready_event: Option<OwnedCudaEventV3>,
}

impl ResidentSealTransactionV3 {
    fn new() -> Self {
        Self {
            host_name_offsets: None,
            host_name_bytes: None,
            name_offsets: None,
            name_bytes: None,
            merkle_scratch_a: None,
            merkle_scratch_b: None,
            canonical_content_merkle: None,
            ready_event: None,
        }
    }

    fn disarm(
        mut self,
    ) -> (
        OwnedCudaEventV3,
        StreamOrderedDeviceBufferV3<u8>,
        ResidentHashTransientV3,
    ) {
        (
            self.ready_event
                .take()
                .expect("successful seal retains final ready event"),
            self.canonical_content_merkle
                .take()
                .expect("successful seal retains canonical root"),
            ResidentHashTransientV3 {
                host_name_offsets: self
                    .host_name_offsets
                    .take()
                    .expect("successful seal retains pinned name offsets"),
                host_name_bytes: self
                    .host_name_bytes
                    .take()
                    .expect("successful seal retains pinned name bytes"),
                name_offsets: self
                    .name_offsets
                    .take()
                    .expect("successful seal retains device name offsets"),
                name_bytes: self
                    .name_bytes
                    .take()
                    .expect("successful seal retains device name bytes"),
                merkle_scratch_a: self
                    .merkle_scratch_a
                    .take()
                    .expect("successful seal retains first Merkle scratch level"),
                merkle_scratch_b: self
                    .merkle_scratch_b
                    .take()
                    .expect("successful seal retains second Merkle scratch level"),
            },
        )
    }
}

impl Drop for ResidentSealTransactionV3 {
    fn drop(&mut self) {
        // Name-table H2D copies may still retain the pinned host pointers. A
        // failed seal has no completion event authority, so leak only those
        // compact host tables and queue every device release on the exact
        // producer stream.
        if let Some(host) = self.host_name_offsets.take() {
            std::mem::forget(host);
        }
        if let Some(host) = self.host_name_bytes.take() {
            std::mem::forget(host);
        }
        if let Some(event) = self.ready_event.take() {
            std::mem::forget(event);
        }
        drop(self.name_offsets.take());
        drop(self.name_bytes.take());
        drop(self.merkle_scratch_a.take());
        drop(self.merkle_scratch_b.take());
        drop(self.canonical_content_merkle.take());
    }
}

#[derive(Debug)]
pub struct ResidentFeatureStoreAssemblerV3 {
    run_device: Option<GpuOnlyRunDeviceAdmissionV3>,
    context: Arc<Context>,
    producer_stream: Arc<Stream>,
    device_ordinal: u32,
    expected_column_bindings: Vec<ResidentFeatureColumnBindingV3>,
    parent_source: Option<Box<dyn ResidentParentDatasetSourceV3>>,
    search_bar_major_values: Option<StreamOrderedDeviceBufferV3<f64>>,
    search_bar_major_validity_u4: Option<StreamOrderedDeviceBufferV3<u8>>,
    validity_code_error: Option<StreamOrderedDeviceBufferV3<u32>>,
    rows: usize,
    total_columns: usize,
    cells: usize,
    packed_validity_logical_bytes: usize,
    packed_validity_allocated_bytes: usize,
    next_destination_column: usize,
    pending_batch: Option<PendingResidentFeatureBatchV3>,
    producer_batch_count: usize,
    value_layout_launch_count: usize,
    validity_boundary_launch_count: usize,
    max_live_producer_bytes: usize,
    max_live_producer_scratch_bytes: usize,
    max_live_pointer_table_bytes: usize,
    max_live_runtime_metadata_bytes: usize,
    footprint_runtime_receipt_v2: Option<ResidentFootprintRuntimeReceiptV2>,
    regime_runtime_receipt_v3: Option<ResidentRegimeRuntimeReceiptV3>,
    session_runtime_receipt_v2: Option<ResidentSessionRuntimeReceiptV2>,
    higher_timeframe_runtime_receipt_v3: Option<ResidentHigherTimeframeRuntimeReceiptV3>,
    robust_normalization_fit_metadata_v2: Option<StreamOrderedDeviceBufferV3<u64>>,
    robust_normalization_ready_event_v2: Option<OwnedCudaEventV3>,
    robust_normalization_runtime_receipt_v2: Option<ResidentRobustNormalizationRuntimeReceiptV2>,
    validity_error_readback_count: usize,
    validity_error_d2h_bytes: usize,
    schema_name_offset_count: usize,
    schema_name_bytes: usize,
    admitted_max_live_producer_bytes: usize,
    admitted_max_live_producer_scratch_bytes: usize,
    admitted_pointer_and_schema_metadata_bytes: usize,
    admitted_normalization_scratch_bytes: usize,
    admitted_fit_metadata_bytes: usize,
    pre_materialization_free_bytes_snapshot: u64,
    post_parent_free_bytes_snapshot: u64,
    retained_parent_dataset_bytes: u64,
    remaining_peak_after_parent_bytes: u64,
    allocator_context_reserve_bytes: u64,
    reserve_policy_id: String,
}

impl ResidentFeatureStoreAssemblerV3 {
    pub fn new(
        run_device: GpuOnlyRunDeviceAdmissionV3,
        expected_column_bindings: Vec<ResidentFeatureColumnBindingV3>,
        parent_source: Box<dyn ResidentParentDatasetSourceV3>,
        working_set: &ResidentWorkingSetBoundV3,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        if run_device.phase_one_free_bytes_snapshot() != working_set.device_free_bytes_snapshot()
            || run_device.allocator_context_reserve_bytes()
                != working_set.allocator_context_reserve_bytes()
            || run_device.reserve_policy_id() != working_set.reserve_policy_id()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "working-set capability evidence does not belong to the moved run-device admission"
                    .into(),
            ));
        }
        let mut construction = ResidentAssemblerConstructionGuardV3::new(run_device, parent_source);
        let context = Arc::clone(&construction.context);
        let producer_stream = Arc::clone(&construction.producer_stream);
        let device_ordinal = construction.device_ordinal;
        CurrentContext::set_current(context.as_ref())?;
        let expected_ordinal = i32::try_from(device_ordinal).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("CUDA device ordinal ABI")
        })?;
        if CurrentContext::get_device()?.as_raw() != expected_ordinal {
            return Err(ResidentFeatureStoreCudaErrorV3::DeviceMismatch);
        }
        if working_set.reserve_policy_id() != RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3 {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "allocator/context reserve policy is not the runtime-owned exact V3 authority"
                    .into(),
            ));
        }
        let parent_source = construction.parent_source();
        if stream_context(&producer_stream)? != context.as_raw()
            || parent_source.producer_context().as_raw() != context.as_raw()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        if parent_source.producer_stream().as_inner() != producer_stream.as_inner() {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        parent_source.producer_ready_event().wait_before_read(
            context.as_ref(),
            producer_stream.as_ref(),
            device_ordinal,
        )?;
        let rows = parent_source.rows();
        let total_columns = expected_column_bindings.len();
        if rows == 0 || total_columns == 0 {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident rows and schema must be nonempty".into(),
            ));
        }
        if parent_source.device_ordinal() != device_ordinal {
            return Err(ResidentFeatureStoreCudaErrorV3::DeviceMismatch);
        }
        validate_parent_extents(parent_source, rows)?;
        validate_expected_bindings(&expected_column_bindings)?;
        let retained_parent_dataset_bytes = u64::try_from(parent_source.retained_device_bytes())
            .map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("retained parent dataset bytes")
            })?;
        let parent_layout = parent_source.parent_dataset_layout();
        let parent_layout_bytes = parent_layout
            .ohlcv_bytes()
            .checked_add(parent_layout.clock_bytes())
            .and_then(|bytes| bytes.checked_add(parent_layout.smc_bytes()))
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident parent layout bytes",
            ))?;
        if u64::try_from(rows).ok() != Some(working_set.row_count())
            || u64::try_from(total_columns).ok() != Some(working_set.column_count())
            || parent_layout.row_count() != working_set.row_count()
            || parent_layout_bytes != working_set.parent_dataset_bytes()
            || retained_parent_dataset_bytes != working_set.parent_dataset_bytes()
            || working_set.full_feature_major_staging_bytes() != 0
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "runtime extents do not match the sealed pre-materialization working set".into(),
            ));
        }
        // This second runtime-owned snapshot is deliberately taken only after
        // the exact parent allocation/retained-byte ledger has been verified,
        // yet before any final feature store, hash scratch, or producer batch
        // allocation. It is independent evidence, never an equality check
        // against the earlier phase-one snapshot.
        let (observed_free_bytes, _) = mem_get_info()?;
        let observed_free_bytes = u64::try_from(observed_free_bytes).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("same-context free-memory snapshot")
        })?;
        let observed_available = observed_free_bytes
            .checked_sub(working_set.allocator_context_reserve_bytes())
            .ok_or(ResidentFeatureStoreCudaErrorV3::RuntimeFreeMemoryChanged {
                required_bytes: working_set.remaining_peak_after_parent_bytes(),
                observed_available_bytes: 0,
            })?;
        let remaining_peak_after_parent_bytes = working_set.remaining_peak_after_parent_bytes();
        if remaining_peak_after_parent_bytes > observed_available {
            return Err(ResidentFeatureStoreCudaErrorV3::RuntimeFreeMemoryChanged {
                required_bytes: remaining_peak_after_parent_bytes,
                observed_available_bytes: observed_available,
            });
        }
        let cells = rows.checked_mul(total_columns).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident feature cells"),
        )?;
        let packed_validity_logical_bytes = cells.div_ceil(2);
        let packed_validity_allocated_bytes = packed_validity_logical_bytes
            .checked_add(VALIDITY_ATOMIC_ALIGNMENT_BYTES - 1)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "packed validity alignment",
            ))?
            / VALIDITY_ATOMIC_ALIGNMENT_BYTES
            * VALIDITY_ATOMIC_ALIGNMENT_BYTES;
        if u64::try_from(packed_validity_logical_bytes).ok()
            != Some(working_set.packed_validity_logical_bytes())
            || u64::try_from(packed_validity_allocated_bytes).ok()
                != Some(working_set.packed_validity_allocated_bytes())
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "runtime packed-validity extent drifted from admission".into(),
            ));
        }
        // SAFETY: every use is enqueued later on this same stream. There is one
        // and only one full final f64 allocation; no feature-major staging.
        let search_bar_major_values = StreamOrderedDeviceBufferV3::<f64>::uninitialized_async(
            cells,
            Arc::clone(&context),
            Arc::clone(&producer_stream),
        )?;
        construction.search_bar_major_values = Some(search_bar_major_values);
        // SAFETY: native initialization is the sole zero of the word-padded u4
        // allocation and precedes every pack on the same stream.
        let search_bar_major_validity_u4 = StreamOrderedDeviceBufferV3::<u8>::uninitialized_async(
            packed_validity_allocated_bytes,
            Arc::clone(&context),
            Arc::clone(&producer_stream),
        )?;
        construction.search_bar_major_validity_u4 = Some(search_bar_major_validity_u4);
        let validity_code_error = StreamOrderedDeviceBufferV3::<u32>::uninitialized_async(
            1,
            Arc::clone(&context),
            Arc::clone(&producer_stream),
        )?;
        construction.validity_code_error = Some(validity_code_error);
        native_result(
            "neoethos_resident_initialize_validity_u4_v3",
            // SAFETY: both destinations have the exact checked extents and the
            // stream owns their async allocation order.
            unsafe {
                neoethos_resident_initialize_validity_u4_v3(
                    construction
                        .search_bar_major_validity_u4
                        .as_mut()
                        .expect("constructor retains packed validity")
                        .as_device_ptr()
                        .as_mut_ptr(),
                    packed_validity_logical_bytes,
                    packed_validity_allocated_bytes,
                    construction
                        .validity_code_error
                        .as_mut()
                        .expect("constructor retains validity error flag")
                        .as_device_ptr()
                        .as_mut_ptr(),
                    producer_stream.as_inner(),
                )
            },
        )?;
        let admitted_max_live_producer_bytes =
            usize::try_from(working_set.max_live_producer_bytes()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("max live producer bytes")
            })?;
        let admitted_max_live_producer_scratch_bytes =
            usize::try_from(working_set.max_live_producer_scratch_bytes()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "max live producer scratch bytes",
                )
            })?;
        let admitted_pointer_and_schema_metadata_bytes =
            usize::try_from(working_set.pointer_and_schema_metadata_bytes()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "pointer and schema metadata bytes",
                )
            })?;
        let admitted_normalization_scratch_bytes =
            usize::try_from(working_set.normalization_scratch_bytes()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("normalization scratch bytes")
            })?;
        let admitted_fit_metadata_bytes = usize::try_from(working_set.fit_metadata_bytes())
            .map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "normalization fit metadata bytes",
                )
            })?;
        let (schema_name_offsets, schema_name_bytes) = encode_names(&expected_column_bindings)?;
        let (
            run_device,
            parent_source,
            search_bar_major_values,
            search_bar_major_validity_u4,
            validity_code_error,
        ) = construction.disarm();
        Ok(Self {
            run_device: Some(run_device),
            context,
            producer_stream,
            device_ordinal,
            expected_column_bindings,
            parent_source: Some(parent_source),
            search_bar_major_values: Some(search_bar_major_values),
            search_bar_major_validity_u4: Some(search_bar_major_validity_u4),
            validity_code_error: Some(validity_code_error),
            rows,
            total_columns,
            cells,
            packed_validity_logical_bytes,
            packed_validity_allocated_bytes,
            next_destination_column: 0,
            pending_batch: None,
            producer_batch_count: 0,
            value_layout_launch_count: 0,
            validity_boundary_launch_count: 0,
            max_live_producer_bytes: 0,
            max_live_producer_scratch_bytes: 0,
            max_live_pointer_table_bytes: 0,
            max_live_runtime_metadata_bytes: 0,
            footprint_runtime_receipt_v2: None,
            regime_runtime_receipt_v3: None,
            session_runtime_receipt_v2: None,
            higher_timeframe_runtime_receipt_v3: None,
            robust_normalization_fit_metadata_v2: None,
            robust_normalization_ready_event_v2: None,
            robust_normalization_runtime_receipt_v2: None,
            validity_error_readback_count: 0,
            validity_error_d2h_bytes: 0,
            schema_name_offset_count: schema_name_offsets.len(),
            schema_name_bytes: schema_name_bytes.len(),
            admitted_max_live_producer_bytes,
            admitted_max_live_producer_scratch_bytes,
            admitted_pointer_and_schema_metadata_bytes,
            admitted_normalization_scratch_bytes,
            admitted_fit_metadata_bytes,
            pre_materialization_free_bytes_snapshot: working_set.device_free_bytes_snapshot(),
            post_parent_free_bytes_snapshot: observed_free_bytes,
            retained_parent_dataset_bytes,
            remaining_peak_after_parent_bytes,
            allocator_context_reserve_bytes: working_set.allocator_context_reserve_bytes(),
            reserve_policy_id: working_set.reserve_policy_id().to_owned(),
        })
    }

    pub fn append_batch(
        &mut self,
        batch: Box<dyn ResidentF64FeatureBatchV3>,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        self.append_batch_with_external_live_bytes(batch, 0)
    }

    /// Append one producer batch while accounting for device allocations that
    /// remain live in the producer executor rather than in the emitted batch.
    /// This is private so only crate-owned composite executors can contribute
    /// external peak-memory evidence.
    fn append_batch_with_external_live_bytes(
        &mut self,
        batch: Box<dyn ResidentF64FeatureBatchV3>,
        external_live_device_bytes: usize,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let mut transaction = ResidentAppendTransactionV3::new(batch);
        if self.pending_batch.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerBatchPending);
        }
        CurrentContext::set_current(self.context.as_ref())?;
        let batch = transaction.batch();
        if batch.device_ordinal() != self.device_ordinal || batch.rows() != self.rows {
            return Err(ResidentFeatureStoreCudaErrorV3::DeviceMismatch);
        }
        if batch.producer_context().as_raw() != self.context.as_raw() {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        if batch.producer_stream().as_inner() != self.producer_stream.as_inner() {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        let batch_columns = batch.column_bindings().len();
        if batch_columns == 0 {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident producer batch has no columns".into(),
            ));
        }
        let destination_end = self
            .next_destination_column
            .checked_add(batch_columns)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "monotonic destination column end",
            ))?;
        let expected_column_bindings = self
            .expected_column_bindings
            .get(self.next_destination_column..destination_end)
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "producer batch exceeds the admitted ordered schema".into(),
                )
            })?;
        if batch.column_bindings() != expected_column_bindings {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "producer batch schema/name/route receipt differs from admission".into(),
            ));
        }
        let live_device_bytes = batch
            .retained_device_bytes()
            .checked_add(external_live_device_bytes)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "producer batch plus external live device bytes",
            ))?;
        if live_device_bytes > self.admitted_max_live_producer_bytes
            || batch.retained_scratch_bytes() > self.admitted_max_live_producer_scratch_bytes
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "producer batch exceeds the preflight max-live working set".into(),
            ));
        }
        validate_batch_extents(batch, self.rows)?;
        let mut pointer_tables = Vec::with_capacity(batch_columns.checked_mul(4).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("combined pointer-table entries"),
        )?);
        pointer_tables.extend(
            batch
                .column_bindings()
                .iter()
                .enumerate()
                .map(|(column, _)| batch.value_buffer(column).as_device_ptr().as_raw()),
        );
        let source_offsets = (0..batch_columns)
            .map(|column| {
                u64::try_from(batch.value_offset(column)).map_err(|_| {
                    ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("source value offset ABI")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        pointer_tables.extend(source_offsets);
        pointer_tables.extend(
            batch
                .column_bindings()
                .iter()
                .enumerate()
                .map(|(column, _)| batch.validity_buffer(column).as_device_ptr().as_raw()),
        );
        let source_validity_offsets = (0..batch_columns)
            .map(|column| {
                u64::try_from(batch.validity_offset(column)).map_err(|_| {
                    ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                        "source validity offset ABI",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        pointer_tables.extend(source_validity_offsets);
        let retained_scratch_bytes = batch.retained_scratch_bytes();
        let runtime_pointer_table_bytes = pointer_tables
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "runtime pointer-table bytes",
            ))?;
        let runtime_metadata_bytes = runtime_pointer_and_schema_metadata_bytes_v3(
            runtime_pointer_table_bytes,
            self.schema_name_offset_count,
            self.schema_name_bytes,
        )?;
        if runtime_metadata_bytes > self.admitted_pointer_and_schema_metadata_bytes {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "runtime pointer/name/schema metadata exceeds admission".into(),
            ));
        }
        let pointer_table_offset = |multiple: usize| {
            let entries = batch_columns.checked_mul(multiple).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("pointer-table offset"),
            )?;
            isize::try_from(entries).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("pointer-table offset ABI")
            })
        };
        let source_offset_entries = pointer_table_offset(1)?;
        let validity_address_entries = pointer_table_offset(2)?;
        let validity_offset_entries = pointer_table_offset(3)?;
        // All non-enqueue validation is complete. Only now order the exact
        // producer-ready event before the first metadata copy/read.
        batch.producer_ready_event().wait_before_read(
            self.context.as_ref(),
            self.producer_stream.as_ref(),
            self.device_ordinal,
        )?;
        // One combined page-locked table and one compact H2D copy retain all
        // four logical u64 tables for the complete pack-event lifetime.
        let (host_pointer_tables, pointer_tables) = compact_device_buffer_from_slice_async(
            &pointer_tables,
            &self.context,
            &self.producer_stream,
        )?;
        transaction.install_pointer_tables(host_pointer_tables, pointer_tables);
        transaction.install_ready_event(OwnedCudaEventV3::new()?);
        let pointer_base = transaction.pointer_tables().as_device_ptr();
        // SAFETY: the allocation contains exactly four contiguous tables of
        // batch_columns u64 entries, proven by the checked capacity above.
        let source_offsets = unsafe { pointer_base.offset(source_offset_entries) };
        let source_validity_addresses = unsafe { pointer_base.offset(validity_address_entries) };
        let source_validity_offsets = unsafe { pointer_base.offset(validity_offset_entries) };
        let values = self.search_bar_major_values.as_mut().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("final value store was moved".into())
        })?;
        let validity = self.search_bar_major_validity_u4.as_mut().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("final validity store was moved".into())
        })?;
        let validity_error = self.validity_code_error.as_mut().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "validity error authority was moved".into(),
            )
        })?;
        native_result(
            "neoethos_resident_pack_batch_to_bar_major_f64_u4_v3",
            // SAFETY: all source extents, destination range, pointer-table
            // lengths and exact same context/stream were checked above.
            unsafe {
                neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(
                    pointer_base.as_ptr(),
                    source_offsets.as_ptr(),
                    source_validity_addresses.as_ptr(),
                    source_validity_offsets.as_ptr(),
                    self.rows,
                    batch_columns,
                    self.total_columns,
                    self.next_destination_column,
                    values.as_device_ptr().as_mut_ptr(),
                    validity.as_device_ptr().as_mut_ptr(),
                    validity_error.as_device_ptr().as_mut_ptr(),
                    self.producer_stream.as_inner(),
                )
            },
        )?;
        transaction.ready_event().record(&self.producer_stream)?;
        self.next_destination_column = destination_end;
        self.producer_batch_count += 1;
        self.value_layout_launch_count += 1;
        self.validity_boundary_launch_count += 1;
        self.max_live_producer_bytes = self.max_live_producer_bytes.max(live_device_bytes);
        self.max_live_producer_scratch_bytes = self
            .max_live_producer_scratch_bytes
            .max(retained_scratch_bytes);
        self.max_live_pointer_table_bytes = self
            .max_live_pointer_table_bytes
            .max(runtime_pointer_table_bytes);
        self.max_live_runtime_metadata_bytes = self
            .max_live_runtime_metadata_bytes
            .max(runtime_metadata_bytes);
        self.pending_batch = Some(transaction.disarm());
        Ok(())
    }

    /// Fail-closed migration boundary for the superseded zero-based V3 append.
    /// Production must supply the globally admitted bindings and the move-only
    /// owner-sized receipt through [`Self::append_resident_classic_ta_recipe_v4`].
    pub fn append_resident_classic_ta_recipe_v3(
        &mut self,
        _recipe: ResidentClassicTaRecipeV3,
    ) -> Result<(), ResidentClassicTaExecutorErrorV3> {
        Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch)
    }

    /// Consume Data's globally bound Classic span and the matching owner-sized
    /// pre-device memory receipt. Both authorities are revalidated inside the
    /// executor before it creates the first retained derived-input allocation.
    pub fn append_resident_classic_ta_recipe_v4(
        &mut self,
        recipe: ResidentClassicTaRecipeV3,
        admitted_global_bindings: Vec<ResidentFeatureColumnBindingV3>,
        pre_device_memory_receipt_v4: ResidentClassicTaPreDeviceMemoryReceiptV4,
    ) -> Result<(), ResidentClassicTaExecutorErrorV3> {
        let mut executor = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "run-device admission was moved before Classic TA execution".into(),
                )
            })?;
            let parent = self.parent_source.as_deref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent was moved before Classic TA execution".into(),
                )
            })?;
            ResidentClassicTaExecutorV3::new_v4(
                run_device,
                parent,
                recipe,
                admitted_global_bindings,
                pre_device_memory_receipt_v4,
            )?
        };
        while let Some(batch) = executor.next_pending_batch_v3()? {
            self.append_batch(Box::new(batch))?;
            while !self.try_retire_completed_batch()? {
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    /// Launch and append the complete seven-column Footprint-v2 family from
    /// the retained parent on this assembler's unique run carrier. The only
    /// returned object is descriptive runtime evidence; values and validity
    /// remain device-resident and the opaque batch is retired by pack event.
    pub fn append_resident_footprint_v2(
        &mut self,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> Result<ResidentFootprintRuntimeReceiptV2, ResidentFeatureStoreCudaErrorV3> {
        if self.footprint_runtime_receipt_v2.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Footprint materialization is one-shot".into(),
            ));
        }
        let batch = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "run-device admission was moved before Footprint execution".into(),
                )
            })?;
            let parent = self.parent_source.as_deref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent was moved before Footprint execution".into(),
                )
            })?;
            launch_resident_footprint_v2(run_device, parent, bindings)?
        };
        let receipt = batch.receipt().clone();
        self.append_batch(Box::new(batch))?;
        self.footprint_runtime_receipt_v2 = Some(receipt.clone());
        Ok(receipt)
    }

    /// Launch and append the complete fourteen-column Regime-v3 family from
    /// the retained parent. The exact power-of-two anchor was sealed from the
    /// canonical CPU admission before any resident output allocation.
    pub fn append_resident_regime_v3(
        &mut self,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
        scale_anchor: f64,
    ) -> Result<ResidentRegimeRuntimeReceiptV3, ResidentFeatureStoreCudaErrorV3> {
        if self.regime_runtime_receipt_v3.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Regime materialization is one-shot".into(),
            ));
        }
        let batch = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "run-device admission was moved before Regime execution".into(),
                )
            })?;
            let parent = self.parent_source.as_deref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent was moved before Regime execution".into(),
                )
            })?;
            launch_resident_regime_v3(run_device, parent, bindings, scale_anchor)?
        };
        let receipt = batch.receipt().clone();
        self.append_batch(Box::new(batch))?;
        self.regime_runtime_receipt_v3 = Some(receipt.clone());
        Ok(receipt)
    }

    pub fn append_resident_quant_v3(
        &mut self,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
        launch_authority: ResidentQuantLaunchAuthorityV3,
    ) -> Result<ResidentQuantRuntimeReceiptV3, ResidentFeatureStoreCudaErrorV3> {
        let batch = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "run-device admission was moved before Quant-v3 execution".into(),
                )
            })?;
            let parent = self.parent_source.as_deref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent was moved before Quant-v3 execution".into(),
                )
            })?;
            launch_resident_quant_v3(run_device, parent, bindings, launch_authority)?
        };
        let receipt = batch.receipt().clone();
        self.append_batch(Box::new(batch))?;
        Ok(receipt)
    }

    /// Append the complete variable-width, direct-parent HTF span in exact
    /// resolved recipe-v4 order. Each emitted batch is retired before the
    /// executor may release its moved parent carriers.
    pub fn append_resident_higher_timeframe_alignment_v3(
        &mut self,
        parents: Vec<ResidentHigherTimeframeDirectParentV3>,
        admitted_global_bindings: Vec<ResidentFeatureColumnBindingV3>,
        launch_authority: ResidentHigherTimeframeLaunchAuthorityV3,
    ) -> Result<ResidentHigherTimeframeRuntimeReceiptV3, ResidentFeatureStoreCudaErrorV3> {
        if self.higher_timeframe_runtime_receipt_v3.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident HTF-v3 runtime was already appended".into(),
            ));
        }
        let mut executor = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "run-device admission was moved before HTF-v3 execution".into(),
                )
            })?;
            let base_parent = self.parent_source.as_deref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident base parent was moved before HTF-v3 execution".into(),
                )
            })?;
            ResidentHigherTimeframeExecutorV3::new(
                run_device,
                base_parent,
                parents,
                admitted_global_bindings,
                launch_authority,
            )?
        };
        let external_live_device_bytes = executor.retained_parent_device_bytes();
        while let Some(batch) = executor.next_pending_batch_v3()? {
            self.append_batch_with_external_live_bytes(
                Box::new(batch),
                external_live_device_bytes,
            )?;
            while !self.try_retire_completed_batch()? {
                std::thread::yield_now();
            }
        }
        let receipt = executor.finish_v3()?;
        self.higher_timeframe_runtime_receipt_v3 = Some(receipt.clone());
        Ok(receipt)
    }

    /// Launch and append the atomic twenty-three-column Session-v2 state
    /// machine from the retained parent. Data's move-only launch authority
    /// binds these outputs to one exact canonical millisecond input.
    pub fn append_resident_session_v2(
        &mut self,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
        launch_authority: ResidentSessionLaunchAuthorityV2,
    ) -> Result<ResidentSessionRuntimeReceiptV2, ResidentFeatureStoreCudaErrorV3> {
        if self.session_runtime_receipt_v2.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Session-v2 materialization is one-shot".into(),
            ));
        }
        let batch = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "run-device admission was moved before Session-v2 execution".into(),
                )
            })?;
            let parent = self.parent_source.as_deref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent was moved before Session-v2 execution".into(),
                )
            })?;
            launch_resident_session_v2(run_device, parent, bindings, launch_authority)?
        };
        let receipt = batch.receipt().clone();
        self.append_batch(Box::new(batch))?;
        self.session_runtime_receipt_v2 = Some(receipt.clone());
        Ok(receipt)
    }

    pub fn try_retire_completed_batch(&mut self) -> Result<bool, ResidentFeatureStoreCudaErrorV3> {
        CurrentContext::set_current(self.context.as_ref())?;
        let Some(pending) = self.pending_batch.as_ref() else {
            return Ok(false);
        };
        if !pending.batch_ready_event.query()? {
            return Ok(false);
        }
        let pending = self
            .pending_batch
            .take()
            .expect("event-proven pending batch must still exist");
        pending.release(&self.producer_stream, true)?;
        Ok(true)
    }

    /// Post-pack/pre-SHA runtime seam for semantic-v2 robust normalization.
    /// The low-level plan is descriptive only: Data's move-only component
    /// receipt remains the sole authority that can seal the resulting store.
    pub fn apply_resident_robust_normalization_v2(
        &mut self,
        plan: &ResidentRobustNormalizationPlanV2,
    ) -> Result<ResidentRobustNormalizationRuntimeReceiptV2, ResidentFeatureStoreCudaErrorV3> {
        if self.pending_batch.is_some()
            || self.next_destination_column != self.total_columns
            || self.robust_normalization_runtime_receipt_v2.is_some()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerBatchPending);
        }
        if plan.rows() != self.rows
            || plan.columns() != self.total_columns
            || plan.packed_validity_logical_bytes() != self.packed_validity_logical_bytes
            || plan.packed_validity_allocated_bytes() != self.packed_validity_allocated_bytes
            || self.packed_validity_allocated_bytes % VALIDITY_ATOMIC_ALIGNMENT_BYTES != 0
            || plan.normalization_scratch_bytes() != self.admitted_normalization_scratch_bytes
            || plan.fit_metadata_bytes() != self.admitted_fit_metadata_bytes
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident robust-normalization extents differ from exact workspace admission"
                    .into(),
            ));
        }
        CurrentContext::set_current(self.context.as_ref())?;
        if stream_context(&self.producer_stream)? != self.context.as_raw() {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        let (
            admission_identity_sha256,
            primary_context_process_token,
            producer_stream_process_token,
        ) = {
            let run_device = self.run_device.as_ref().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident normalization lost the one-shot run admission".into(),
                )
            })?;
            (
                run_device.admission_identity_sha256(),
                run_device.device_identity().primary_context_process_token(),
                run_device.run_stream_process_token_v3(),
            )
        };

        if !plan.enabled() {
            let receipt = disabled_resident_robust_normalization_receipt_v2(
                plan,
                admission_identity_sha256,
                primary_context_process_token,
                producer_stream_process_token,
            )?;
            self.robust_normalization_runtime_receipt_v2 = Some(receipt.clone());
            return Ok(receipt);
        }

        let mut sort_scratch_bits = StreamOrderedDeviceBufferV3::<u64>::uninitialized_async(
            plan.normalization_scratch_slots(),
            Arc::clone(&self.context),
            Arc::clone(&self.producer_stream),
        )?;
        let mut fit_metadata_words = StreamOrderedDeviceBufferV3::<u64>::uninitialized_async(
            plan.fit_metadata_words(),
            Arc::clone(&self.context),
            Arc::clone(&self.producer_stream),
        )?;
        let pending_receipt = launch_resident_robust_normalization_v2(
            plan,
            self.search_bar_major_values.as_mut().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "final value store was moved before normalization".into(),
                )
            })?,
            self.search_bar_major_validity_u4.as_mut().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "final validity store was moved before normalization".into(),
                )
            })?,
            &mut sort_scratch_bits,
            &mut fit_metadata_words,
            self.validity_code_error.as_mut().ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "bounded normalization control word was moved".into(),
                )
            })?,
            &self.producer_stream,
        )?;
        let ready_event = OwnedCudaEventV3::new()?;
        ready_event.record(&self.producer_stream)?;
        ready_event.synchronize()?;

        let mut validity_code_error = [0_u32; 1];
        self.validity_code_error
            .as_ref()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "bounded normalization verdict was moved".into(),
                )
            })?
            .copy_to(&mut validity_code_error)?;
        self.validity_error_readback_count = 1;
        self.validity_error_d2h_bytes = std::mem::size_of::<u32>();
        if validity_code_error[0] != 0 {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidProducerValidityCode);
        }

        let mut fit_digest_words = [0_u64; SHA256_BYTES / std::mem::size_of::<u64>()];
        sort_scratch_bits
            .index(0..fit_digest_words.len())
            .copy_to(&mut fit_digest_words)?;
        let mut fit_metadata_sha256 = [0_u8; SHA256_BYTES];
        for (destination, word) in fit_metadata_sha256
            .chunks_exact_mut(std::mem::size_of::<u64>())
            .zip(fit_digest_words)
        {
            destination.copy_from_slice(&word.to_ne_bytes());
        }
        const ROBUST_NORMALIZATION_READY_RECORD_SEQUENCE_V2: u64 = 1;
        let receipt = pending_receipt.seal_after_ready_event_v2(
            fit_metadata_sha256,
            admission_identity_sha256,
            primary_context_process_token,
            producer_stream_process_token,
            ready_event.process_token(
                admission_identity_sha256,
                ROBUST_NORMALIZATION_READY_RECORD_SEQUENCE_V2,
            ),
        )?;
        // The digest prefix has been copied only after the explicit event
        // synchronization. Fit metadata and the event remain resident with the
        // final store owner; scratch can now retire on this same stream.
        drop(sort_scratch_bits);
        self.robust_normalization_fit_metadata_v2 = Some(fit_metadata_words);
        self.robust_normalization_ready_event_v2 = Some(ready_event);
        self.robust_normalization_runtime_receipt_v2 = Some(receipt.clone());
        Ok(receipt)
    }

    pub fn seal(
        mut self,
    ) -> Result<Arc<ResidentFeatureStoreOwnerV3>, ResidentFeatureStoreCudaErrorV3> {
        let all_columns_filled = self.next_destination_column == self.total_columns;
        let all_producer_events_retired = self.pending_batch.is_none();
        if !all_columns_filled || !all_producer_events_retired {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerBatchPending);
        }
        if self.producer_batch_count == 0
            || self.max_live_producer_bytes != self.admitted_max_live_producer_bytes
            || self.max_live_producer_scratch_bytes != self.admitted_max_live_producer_scratch_bytes
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "measured producer ledger differs from the exact preflight peak: live={} admitted_live={} scratch={} admitted_scratch={}",
                self.max_live_producer_bytes,
                self.admitted_max_live_producer_bytes,
                self.max_live_producer_scratch_bytes,
                self.admitted_max_live_producer_scratch_bytes,
            )));
        }
        CurrentContext::set_current(self.context.as_ref())?;
        if self.validity_error_readback_count == 0 {
            let mut validity_code_error = [0_u32; 1];
            self.validity_code_error
                .as_ref()
                .ok_or_else(|| {
                    ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "validity error authority was moved".into(),
                    )
                })?
                .copy_to(&mut validity_code_error)?;
            self.validity_error_readback_count = 1;
            self.validity_error_d2h_bytes = std::mem::size_of::<u32>();
            if validity_code_error[0] != 0 {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidProducerValidityCode);
            }
        } else if self.validity_error_readback_count != 1
            || self.validity_error_d2h_bytes != std::mem::size_of::<u32>()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "validity verdict readback accounting is not exactly once/four bytes".into(),
            ));
        }
        if let Some(error_flag) = self.validity_code_error.take() {
            drop(error_flag);
        }

        let mut seal_transaction = ResidentSealTransactionV3::new();
        let (name_offsets, name_bytes) = encode_names(&self.expected_column_bindings)?;
        let runtime_pointer_and_schema_metadata_bytes =
            runtime_pointer_and_schema_metadata_bytes_v3(
                self.max_live_pointer_table_bytes,
                name_offsets.len(),
                name_bytes.len(),
            )?;
        if runtime_pointer_and_schema_metadata_bytes
            != self.admitted_pointer_and_schema_metadata_bytes
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "runtime pointer/name/schema metadata differs from admission".into(),
            ));
        }
        let (host_name_offsets, name_offsets) = compact_device_buffer_from_slice_async(
            &name_offsets,
            &self.context,
            &self.producer_stream,
        )?;
        seal_transaction.host_name_offsets = Some(host_name_offsets);
        seal_transaction.name_offsets = Some(name_offsets);
        let (host_name_bytes, name_bytes) = compact_device_buffer_from_slice_async(
            &name_bytes,
            &self.context,
            &self.producer_stream,
        )?;
        seal_transaction.host_name_bytes = Some(host_name_bytes);
        seal_transaction.name_bytes = Some(name_bytes);
        let timestamp_chunk_count = self.rows.div_ceil(CANONICAL_MERKLE_CHUNK_ROWS_V3);
        let merkle_leaf_count = timestamp_chunk_count
            .checked_mul(self.total_columns.checked_add(1).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("Merkle producer count"),
            )?)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "Merkle leaf count",
            ))?;
        let merkle_scratch_level_bytes = merkle_leaf_count.checked_mul(SHA256_BYTES).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("Merkle scratch bytes"),
        )?;
        let merkle_scratch_bytes = merkle_scratch_level_bytes.checked_mul(2).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("two Merkle scratch levels"),
        )?;
        // SAFETY: the two scratch levels and root are used later on this exact
        // stream and remain owned through the final ready event.
        let merkle_scratch_a = StreamOrderedDeviceBufferV3::<u8>::uninitialized_async(
            merkle_scratch_level_bytes,
            Arc::clone(&self.context),
            Arc::clone(&self.producer_stream),
        )?;
        seal_transaction.merkle_scratch_a = Some(merkle_scratch_a);
        let merkle_scratch_b = StreamOrderedDeviceBufferV3::<u8>::uninitialized_async(
            merkle_scratch_level_bytes,
            Arc::clone(&self.context),
            Arc::clone(&self.producer_stream),
        )?;
        seal_transaction.merkle_scratch_b = Some(merkle_scratch_b);
        let canonical_content_merkle = StreamOrderedDeviceBufferV3::<u8>::uninitialized_async(
            SHA256_BYTES,
            Arc::clone(&self.context),
            Arc::clone(&self.producer_stream),
        )?;
        seal_transaction.canonical_content_merkle = Some(canonical_content_merkle);
        let parent_source = self.parent_source.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident parent dataset authority was moved".into(),
            )
        })?;
        let values = self.search_bar_major_values.as_mut().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("final value store was moved".into())
        })?;
        let validity = self.search_bar_major_validity_u4.as_mut().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("final validity store was moved".into())
        })?;
        native_result(
            "neoethos_resident_canonical_merkle_sha256_v3",
            // SAFETY: every batch has event-proven completion, every final
            // cell is initialized, exact u4 validity was checked, and all
            // compact metadata/scratch extents are checked above.
            unsafe {
                neoethos_resident_canonical_merkle_sha256_v3(
                    parent_source.timestamps().as_device_ptr().as_ptr(),
                    self.rows,
                    self.total_columns,
                    seal_transaction
                        .name_offsets
                        .as_ref()
                        .expect("seal transaction retains device name offsets")
                        .as_device_ptr()
                        .as_ptr(),
                    seal_transaction
                        .name_bytes
                        .as_ref()
                        .expect("seal transaction retains device name bytes")
                        .as_device_ptr()
                        .as_ptr(),
                    values.as_device_ptr().as_ptr(),
                    validity.as_device_ptr().as_ptr(),
                    seal_transaction
                        .merkle_scratch_a
                        .as_mut()
                        .expect("seal transaction retains first Merkle scratch level")
                        .as_device_ptr()
                        .as_mut_ptr(),
                    seal_transaction
                        .merkle_scratch_b
                        .as_mut()
                        .expect("seal transaction retains second Merkle scratch level")
                        .as_device_ptr()
                        .as_mut_ptr(),
                    merkle_leaf_count,
                    seal_transaction
                        .canonical_content_merkle
                        .as_mut()
                        .expect("seal transaction retains canonical root")
                        .as_device_ptr()
                        .as_mut_ptr(),
                    self.producer_stream.as_inner(),
                )
            },
        )?;
        seal_transaction.ready_event = Some(OwnedCudaEventV3::new()?);
        seal_transaction
            .ready_event
            .as_ref()
            .expect("seal transaction retains final ready event")
            .record(&self.producer_stream)?;

        let layout_transform_value_bytes =
            self.cells.checked_mul(std::mem::size_of::<f64>()).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("layout-transform value bytes"),
            )?;
        let (ready_event, canonical_content_merkle, hash_transient) = seal_transaction.disarm();
        let owner = Arc::new(ResidentFeatureStoreOwnerV3 {
            run_device: self.run_device.take(),
            ready_event,
            canonical_content_merkle: Some(canonical_content_merkle),
            compact_hashes: Mutex::new(None),
            search_bar_major_values: self.search_bar_major_values.take(),
            search_bar_major_validity_u4: self.search_bar_major_validity_u4.take(),
            parent_source: self.parent_source.take(),
            hash_transient: Mutex::new(Some(hash_transient)),
            hash_transient_retirement_event: Mutex::new(None),
            producer_stream: Arc::clone(&self.producer_stream),
            context: Arc::clone(&self.context),
            device_ordinal: self.device_ordinal,
            rows: self.rows,
            columns: self.total_columns,
            column_bindings: self.expected_column_bindings.clone(),
            source_column_count: self.total_columns,
            producer_batch_count: self.producer_batch_count,
            validity_initialization_count: 1,
            value_layout_launch_count: self.value_layout_launch_count,
            validity_boundary_launch_count: self.validity_boundary_launch_count,
            layout_transform_value_bytes,
            layout_transform_logical_validity_bytes: self.cells,
            packed_validity_logical_bytes: self.packed_validity_logical_bytes,
            packed_validity_allocated_bytes: self.packed_validity_allocated_bytes,
            max_live_producer_bytes: self.max_live_producer_bytes,
            max_live_producer_scratch_bytes: self.max_live_producer_scratch_bytes,
            max_live_runtime_metadata_bytes: self.max_live_runtime_metadata_bytes,
            footprint_runtime_receipt_v2: self.footprint_runtime_receipt_v2.take(),
            regime_runtime_receipt_v3: self.regime_runtime_receipt_v3.take(),
            session_runtime_receipt_v2: self.session_runtime_receipt_v2.take(),
            higher_timeframe_runtime_receipt_v3: self.higher_timeframe_runtime_receipt_v3.take(),
            robust_normalization_fit_metadata_v2: self.robust_normalization_fit_metadata_v2.take(),
            robust_normalization_ready_event_v2: self.robust_normalization_ready_event_v2.take(),
            robust_normalization_runtime_receipt_v2: self
                .robust_normalization_runtime_receipt_v2
                .take(),
            validity_error_readback_count: self.validity_error_readback_count,
            validity_error_d2h_bytes: self.validity_error_d2h_bytes,
            full_feature_major_staging_bytes: 0,
            merkle_leaf_count,
            merkle_scratch_bytes,
            pre_materialization_free_bytes_snapshot: self.pre_materialization_free_bytes_snapshot,
            post_parent_free_bytes_snapshot: self.post_parent_free_bytes_snapshot,
            retained_parent_dataset_bytes: self.retained_parent_dataset_bytes,
            remaining_peak_after_parent_bytes: self.remaining_peak_after_parent_bytes,
            allocator_context_reserve_bytes: self.allocator_context_reserve_bytes,
            reserve_policy_id: self.reserve_policy_id.clone(),
        });
        Ok(owner)
    }
}

fn validate_expected_bindings(
    bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    let mut names = BTreeSet::new();
    for (index, binding) in bindings.iter().enumerate() {
        if binding.ordinal != index
            || binding.feature_name.is_empty()
            || !names.insert(binding.feature_name.as_str())
            || binding
                .canonical_parameter_tuple_sha256
                .iter()
                .all(|byte| *byte == 0)
            || binding.route_receipt_sha256.iter().all(|byte| *byte == 0)
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "admitted feature bindings must be exact, monotonic, unique and hashed".into(),
            ));
        }
    }
    Ok(())
}

fn validate_parent_extents(
    parent: &dyn ResidentParentDatasetSourceV3,
    rows: usize,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    for (field, actual) in [
        ("open", parent.open().len()),
        ("close", parent.close().len()),
        ("high", parent.high().len()),
        ("low", parent.low().len()),
        ("volume", parent.volume().len()),
        ("timestamps", parent.timestamps().len()),
        ("months", parent.months().len()),
        ("days", parent.days().len()),
    ] {
        if actual != rows {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident parent {field} has {actual} rows, expected {rows}"
            )));
        }
    }
    let smc_cells = rows.checked_mul(SMC_SLOTS_V3).ok_or(
        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident parent SMC cells"),
    )?;
    if parent.smc_rows().len() != smc_cells {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
            "resident parent SMC has {} cells, expected {smc_cells}",
            parent.smc_rows().len()
        )));
    }
    Ok(())
}

fn validate_batch_extents(
    batch: &dyn ResidentF64FeatureBatchV3,
    rows: usize,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    for column in 0..batch.column_bindings().len() {
        let value_end = batch.value_offset(column).checked_add(rows).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("producer value extent"),
        )?;
        let validity_end = batch.validity_offset(column).checked_add(rows).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("producer validity extent"),
        )?;
        if value_end > batch.value_buffer(column).len()
            || validity_end > batch.validity_buffer(column).len()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident producer column {column} range exceeds its owned buffer"
            )));
        }
    }
    Ok(())
}

fn encode_names(
    bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(Vec<u64>, Vec<u8>), ResidentFeatureStoreCudaErrorV3> {
    let mut offsets = Vec::with_capacity(bindings.len() + 1);
    let mut bytes = Vec::new();
    offsets.push(0);
    for binding in bindings {
        bytes.extend_from_slice(binding.feature_name.as_bytes());
        offsets.push(u64::try_from(bytes.len()).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("feature name bytes")
        })?);
    }
    Ok((offsets, bytes))
}

fn runtime_pointer_and_schema_metadata_bytes_v3(
    max_live_pointer_table_bytes: usize,
    name_offset_count: usize,
    name_bytes: usize,
) -> Result<usize, ResidentFeatureStoreCudaErrorV3> {
    let name_offset_bytes = name_offset_count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "runtime feature-name offset bytes",
        ))?;
    max_live_pointer_table_bytes
        .checked_add(name_offset_bytes)
        .and_then(|bytes| bytes.checked_add(name_bytes))
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "runtime pointer and schema metadata bytes",
        ))
}

#[derive(Debug)]
struct ResidentHashTransientV3 {
    host_name_offsets: LockedBuffer<u64>,
    host_name_bytes: LockedBuffer<u8>,
    name_offsets: StreamOrderedDeviceBufferV3<u64>,
    name_bytes: StreamOrderedDeviceBufferV3<u8>,
    merkle_scratch_a: StreamOrderedDeviceBufferV3<u8>,
    merkle_scratch_b: StreamOrderedDeviceBufferV3<u8>,
}

impl ResidentHashTransientV3 {
    fn release(
        self,
        stream: &Stream,
        host_copy_is_complete: bool,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if [
            self.name_offsets.is_owned_by_stream(stream),
            self.name_bytes.is_owned_by_stream(stream),
            self.merkle_scratch_a.is_owned_by_stream(stream),
            self.merkle_scratch_b.is_owned_by_stream(stream),
        ]
        .contains(&false)
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        let Self {
            host_name_offsets,
            host_name_bytes,
            name_offsets,
            name_bytes,
            merkle_scratch_a,
            merkle_scratch_b,
        } = self;
        if !host_copy_is_complete {
            std::mem::forget(host_name_offsets);
            std::mem::forget(host_name_bytes);
        }
        name_offsets.release_async(stream)?;
        name_bytes.release_async(stream)?;
        merkle_scratch_a.release_async(stream)?;
        merkle_scratch_b.release_async(stream)?;
        Ok(())
    }
}

/// One-shot proof that every async Data transient free was queued before this
/// event on the exact admitted producer stream. It is intentionally pending:
/// population allocations remain forbidden until the proof is synchronized.
#[derive(Debug)]
#[must_use = "synchronize the exact producer stream before any population allocation"]
struct PendingDataTransientRetirementV1 {
    event: OwnedCudaEventV3,
    primary_context: CUcontext,
    producer_stream: CUstream,
    device_ordinal: u32,
    admission_identity_sha256: [u8; SHA256_BYTES],
    workspace_plan_identity_sha256: [u8; SHA256_BYTES],
}

#[derive(Debug)]
#[must_use = "consume this synchronized retirement authority into population binding"]
struct SealedDataTransientRetirementV1 {
    admission_identity_sha256: [u8; SHA256_BYTES],
    workspace_plan_identity_sha256: [u8; SHA256_BYTES],
    retirement_event_process_token: [u8; SHA256_BYTES],
}

impl PendingDataTransientRetirementV1 {
    fn synchronize_before_population_allocation(
        self,
        context: &Context,
        stream: &Stream,
        expected_device_ordinal: u32,
        expected_admission_identity_sha256: [u8; SHA256_BYTES],
        expected_workspace_plan_identity_sha256: [u8; SHA256_BYTES],
    ) -> Result<SealedDataTransientRetirementV1, ResidentFeatureStoreCudaErrorV3> {
        if self.primary_context != context.as_raw()
            || self.producer_stream != stream.as_inner()
            || self.device_ordinal != expected_device_ordinal
            || self.admission_identity_sha256 != expected_admission_identity_sha256
            || self.workspace_plan_identity_sha256 != expected_workspace_plan_identity_sha256
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "Data transient-retirement proof does not match the admitted population run".into(),
            ));
        }
        CurrentContext::set_current(context)?;
        let expected_ordinal = i32::try_from(expected_device_ordinal).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "Data transient-retirement device ordinal ABI",
            )
        })?;
        if CurrentContext::get_device()?.as_raw() != expected_ordinal
            || stream_context(stream)? != context.as_raw()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        // This is deliberately a full synchronization of the exact stream,
        // not merely a wait on the earlier Data-ready event. `drop_async`
        // returns bytes through CUDA's default async pool; the legacy
        // population allocator may begin only after those frees have retired.
        driver_result("cuStreamSynchronize(Data transient retirement)", unsafe {
            cuStreamSynchronize(stream.as_inner())
        })?;
        if !self.event.query()? {
            return Err(ResidentFeatureStoreCudaErrorV3::NotReady);
        }
        let retirement_event_process_token =
            self.event.process_token(self.admission_identity_sha256, 2);
        Ok(SealedDataTransientRetirementV1 {
            admission_identity_sha256: self.admission_identity_sha256,
            workspace_plan_identity_sha256: self.workspace_plan_identity_sha256,
            retirement_event_process_token,
        })
    }
}

fn bind_population_after_data_transient_retirement_v1(
    retirement: SealedDataTransientRetirementV1,
    expected_admission_identity_sha256: [u8; SHA256_BYTES],
    expected_workspace_plan_identity_sha256: [u8; SHA256_BYTES],
    raw: RawResidentFeatureStoreBindV3,
) -> Result<(PopulationSession, SealedDataTransientRetirementV1), ResidentFeatureStoreCudaErrorV3> {
    if retirement.admission_identity_sha256 != expected_admission_identity_sha256
        || retirement.workspace_plan_identity_sha256 != expected_workspace_plan_identity_sha256
        || retirement.retirement_event_process_token == [0; SHA256_BYTES]
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "synchronized Data transient-retirement authority drifted before population binding"
                .into(),
        ));
    }
    let session = PopulationSession::bind_resident_feature_store_v3(raw)?;
    Ok((session, retirement))
}

#[derive(Debug)]
struct ResidentConsumerLifetimeV3 {
    population_session: Option<PopulationSession>,
    owner: Arc<ResidentFeatureStoreOwnerV3>,
    consumer_stream: Arc<Stream>,
    consumer_context: Arc<Context>,
}

/// Owns the one-copy feature layout, exact parent arrays, primary context,
/// stream and final ready event. There is no public raw-pointer constructor.
#[derive(Debug)]
pub struct ResidentFeatureStoreOwnerV3 {
    run_device: Option<GpuOnlyRunDeviceAdmissionV3>,
    ready_event: OwnedCudaEventV3,
    canonical_content_merkle: Option<StreamOrderedDeviceBufferV3<u8>>,
    compact_hashes: Mutex<Option<ResidentFeatureCompactHashesV3>>,
    search_bar_major_values: Option<StreamOrderedDeviceBufferV3<f64>>,
    search_bar_major_validity_u4: Option<StreamOrderedDeviceBufferV3<u8>>,
    parent_source: Option<Box<dyn ResidentParentDatasetSourceV3>>,
    hash_transient: Mutex<Option<ResidentHashTransientV3>>,
    hash_transient_retirement_event: Mutex<Option<OwnedCudaEventV3>>,
    producer_stream: Arc<Stream>,
    context: Arc<Context>,
    device_ordinal: u32,
    rows: usize,
    columns: usize,
    column_bindings: Vec<ResidentFeatureColumnBindingV3>,
    source_column_count: usize,
    producer_batch_count: usize,
    validity_initialization_count: usize,
    value_layout_launch_count: usize,
    validity_boundary_launch_count: usize,
    layout_transform_value_bytes: usize,
    layout_transform_logical_validity_bytes: usize,
    packed_validity_logical_bytes: usize,
    packed_validity_allocated_bytes: usize,
    max_live_producer_bytes: usize,
    max_live_producer_scratch_bytes: usize,
    max_live_runtime_metadata_bytes: usize,
    footprint_runtime_receipt_v2: Option<ResidentFootprintRuntimeReceiptV2>,
    regime_runtime_receipt_v3: Option<ResidentRegimeRuntimeReceiptV3>,
    session_runtime_receipt_v2: Option<ResidentSessionRuntimeReceiptV2>,
    higher_timeframe_runtime_receipt_v3: Option<ResidentHigherTimeframeRuntimeReceiptV3>,
    robust_normalization_fit_metadata_v2: Option<StreamOrderedDeviceBufferV3<u64>>,
    robust_normalization_ready_event_v2: Option<OwnedCudaEventV3>,
    robust_normalization_runtime_receipt_v2: Option<ResidentRobustNormalizationRuntimeReceiptV2>,
    validity_error_readback_count: usize,
    validity_error_d2h_bytes: usize,
    full_feature_major_staging_bytes: usize,
    merkle_leaf_count: usize,
    merkle_scratch_bytes: usize,
    pre_materialization_free_bytes_snapshot: u64,
    post_parent_free_bytes_snapshot: u64,
    retained_parent_dataset_bytes: u64,
    remaining_peak_after_parent_bytes: u64,
    allocator_context_reserve_bytes: u64,
    reserve_policy_id: String,
}

impl ResidentFeatureStoreOwnerV3 {
    pub fn admission_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.run_device
            .as_ref()
            .expect("sealed store retains one-shot run-device admission")
            .admission_identity_sha256()
    }

    pub fn device_identity(&self) -> &CudaPrimaryContextBuildIdentityV3 {
        self.run_device
            .as_ref()
            .expect("sealed store retains one-shot run-device admission")
            .device_identity()
    }

    pub fn ready_event_contract(
        &self,
    ) -> Result<ResidentReadyEventV3, ResidentFeatureStoreCudaErrorV3> {
        const FINAL_READY_RECORD_SEQUENCE_V3: u64 = 1;
        let run_device = self.run_device.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed store lost its one-shot run-device admission".into(),
            )
        })?;
        ResidentReadyEventV3::new(
            self.device_ordinal,
            run_device.device_identity().primary_context_process_token(),
            run_device.run_stream_process_token_v3(),
            self.ready_event.process_token(
                run_device.admission_identity_sha256(),
                FINAL_READY_RECORD_SEQUENCE_V3,
            ),
            FINAL_READY_RECORD_SEQUENCE_V3,
        )
        .map_err(Into::into)
    }

    pub fn compact_hashes_if_ready(
        &self,
    ) -> Result<ResidentFeatureCompactHashesV3, ResidentFeatureStoreCudaErrorV3> {
        let mut compact_hashes = self.compact_hashes.lock().map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident compact-hash state was poisoned".into(),
            )
        })?;
        if let Some(hashes) = compact_hashes.as_ref() {
            return Ok(hashes.clone());
        }
        CurrentContext::set_current(self.context.as_ref())?;
        if !self.ready_event.query()? {
            return Err(ResidentFeatureStoreCudaErrorV3::NotReady);
        }
        let mut canonical_content_merkle = [0_u8; SHA256_BYTES];
        self.canonical_content_merkle
            .as_ref()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "canonical root allocation was already released".into(),
                )
            })?
            .copy_to(canonical_content_merkle.as_mut_slice())?;
        let hashes = ResidentFeatureCompactHashesV3 {
            canonical_content_merkle,
        };
        // Retirement records a new event only after every async free is
        // queued. Never cache success without that later lifetime proof.
        self.retire_hash_transient_after_ready()?;
        *compact_hashes = Some(hashes.clone());
        Ok(hashes)
    }

    fn retire_hash_transient_after_ready(&self) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let transient = self
            .hash_transient
            .lock()
            .map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident hash lifetime state was poisoned".into(),
                )
            })?
            .take();
        if let Some(transient) = transient {
            transient.release(&self.producer_stream, true)?;
            let retirement_event = OwnedCudaEventV3::new()?;
            retirement_event.record(&self.producer_stream)?;
            let mut slot = self.hash_transient_retirement_event.lock().map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident transient-retirement event state was poisoned".into(),
                )
            })?;
            if slot.is_some() {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident Data transients were retired more than once".into(),
                ));
            }
            *slot = Some(retirement_event);
        }
        Ok(())
    }

    fn take_data_transient_retirement_proof_v1(
        &self,
    ) -> Result<PendingDataTransientRetirementV1, ResidentFeatureStoreCudaErrorV3> {
        let event = self
            .hash_transient_retirement_event
            .lock()
            .map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident transient-retirement event state was poisoned".into(),
                )
            })?
            .take()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident Data transients have no post-free retirement event".into(),
                )
            })?;
        let run_device = self.run_device.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident store lost its admitted run-device authority".into(),
            )
        })?;
        Ok(PendingDataTransientRetirementV1 {
            event,
            primary_context: self.context.as_raw(),
            producer_stream: self.producer_stream.as_inner(),
            device_ordinal: self.device_ordinal,
            admission_identity_sha256: run_device.admission_identity_sha256(),
            workspace_plan_identity_sha256: run_device.workspace_plan_identity_sha256(),
        })
    }

    pub fn layout_evidence(
        &self,
        hashes: &ResidentFeatureCompactHashesV3,
    ) -> ResidentFeatureLayoutEvidenceV3 {
        ResidentFeatureLayoutEvidenceV3 {
            rows: self.rows,
            columns: self.columns,
            canonical_content_merkle: hashes.canonical_content_merkle,
            source_column_count: self.source_column_count,
            producer_batch_count: self.producer_batch_count,
            validity_initialization_count: self.validity_initialization_count,
            value_layout_launch_count: self.value_layout_launch_count,
            validity_boundary_launch_count: self.validity_boundary_launch_count,
            layout_transform_value_bytes: self.layout_transform_value_bytes,
            layout_transform_logical_validity_bytes: self.layout_transform_logical_validity_bytes,
            packed_validity_logical_bytes: self.packed_validity_logical_bytes,
            packed_validity_allocated_bytes: self.packed_validity_allocated_bytes,
            max_live_producer_bytes: self.max_live_producer_bytes,
            max_live_producer_scratch_bytes: self.max_live_producer_scratch_bytes,
            max_live_runtime_metadata_bytes: self.max_live_runtime_metadata_bytes,
            footprint_runtime_receipt_v2: self.footprint_runtime_receipt_v2.clone(),
            regime_runtime_receipt_v3: self.regime_runtime_receipt_v3.clone(),
            session_runtime_receipt_v2: self.session_runtime_receipt_v2.clone(),
            higher_timeframe_runtime_receipt_v3: self.higher_timeframe_runtime_receipt_v3.clone(),
            robust_normalization_runtime_receipt_v2: self
                .robust_normalization_runtime_receipt_v2
                .clone(),
            full_feature_major_staging_bytes: self.full_feature_major_staging_bytes,
            merkle_leaf_count: self.merkle_leaf_count,
            merkle_scratch_bytes: self.merkle_scratch_bytes,
            canonical_root_device_bytes: self
                .canonical_content_merkle
                .as_ref()
                .map(|root| root.len())
                .unwrap_or(0),
            validity_error_readback_count: self.validity_error_readback_count,
            validity_error_d2h_bytes: self.validity_error_d2h_bytes,
            canonical_root_readback_count: 1,
            canonical_root_d2h_bytes: SHA256_BYTES,
            compact_control_plane_d2h_bytes: self.validity_error_d2h_bytes
                + SHA256_BYTES
                + self
                    .robust_normalization_runtime_receipt_v2
                    .as_ref()
                    .map_or(0, |receipt| receipt.fit_digest_d2h_bytes()),
            pre_materialization_free_bytes_snapshot: self.pre_materialization_free_bytes_snapshot,
            post_parent_free_bytes_snapshot: self.post_parent_free_bytes_snapshot,
            retained_parent_dataset_bytes: self.retained_parent_dataset_bytes,
            remaining_peak_after_parent_bytes: self.remaining_peak_after_parent_bytes,
            allocator_context_reserve_bytes: self.allocator_context_reserve_bytes,
            reserve_policy_id: self.reserve_policy_id.clone(),
        }
    }

    pub(crate) fn import_on_consumer_stream(
        self: &Arc<Self>,
        consumer_context: Arc<Context>,
        consumer_stream: Arc<Stream>,
        consumer_device_ordinal: u32,
    ) -> Result<ResidentFeatureStoreImportV3, ResidentFeatureStoreCudaErrorV3> {
        if consumer_device_ordinal != self.device_ordinal {
            return Err(ResidentFeatureStoreCudaErrorV3::DeviceMismatch);
        }
        if consumer_context.as_raw() != self.context.as_raw() {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        CurrentContext::set_current(consumer_context.as_ref())?;
        if stream_context(&consumer_stream)? != consumer_context.as_raw()
            || consumer_stream.as_inner().is_null()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        self.ready_event.enqueue_wait(&consumer_stream)?;
        Ok(ResidentFeatureStoreImportV3 {
            owner: Some(Arc::clone(self)),
            consumer_context: Some(consumer_context),
            consumer_stream: Some(consumer_stream),
        })
    }

    /// Moves the store toward its sole Search consumer using the exact
    /// context, stream and ordinal retained by the one-shot run admission.
    /// Callers cannot inject or reconstruct any of those handles.
    pub fn import_on_admitted_run_stream_v3(
        self: &Arc<Self>,
    ) -> Result<ResidentFeatureStoreImportV3, ResidentFeatureStoreCudaErrorV3> {
        let admitted = self.run_device.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed store lost its one-shot run-device admission".into(),
            )
        })?;
        let consumer_context = Arc::clone(admitted.primary_context_for_resident_producer_v3());
        let consumer_stream = Arc::clone(admitted.run_stream_for_resident_producer_v3());
        let consumer_device_ordinal = admitted.device_identity().ordinal();
        self.import_on_consumer_stream(consumer_context, consumer_stream, consumer_device_ordinal)
    }

    pub fn column_bindings(&self) -> &[ResidentFeatureColumnBindingV3] {
        &self.column_bindings
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn columns(&self) -> usize {
        self.columns
    }
    pub const fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }
    pub fn producer_stream(&self) -> &Arc<Stream> {
        &self.producer_stream
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub fn copy_bar_major_for_device_fixture_v3(
        &self,
    ) -> Result<
        crate::resident_feature_store_v3_device_fixture::ResidentFeatureStoreDeviceReadbackV3,
        ResidentFeatureStoreCudaErrorV3,
    > {
        CurrentContext::set_current(self.context.as_ref())?;
        self.ready_event.synchronize()?;
        let values = self.search_bar_major_values.as_deref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "device-fixture resident values were already released".into(),
            )
        })?;
        let validity_u4 = self
            .search_bar_major_validity_u4
            .as_deref()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "device-fixture resident validity was already released".into(),
                )
            })?;
        crate::resident_feature_store_v3_device_fixture::copy_bar_major_for_device_fixture_v3(
            values,
            validity_u4,
            self.rows,
            self.columns,
        )
    }

    pub fn sealed_steady_device_bytes(&self) -> usize {
        let values = self
            .search_bar_major_values
            .as_ref()
            .map_or(0, |buffer| buffer.len() * std::mem::size_of::<f64>());
        let validity = self
            .search_bar_major_validity_u4
            .as_ref()
            .map_or(0, |buffer| buffer.len());
        let root = self
            .canonical_content_merkle
            .as_ref()
            .map_or(0, |buffer| buffer.len());
        let normalization_fit = self
            .robust_normalization_fit_metadata_v2
            .as_ref()
            .map_or(0, |buffer| buffer.len() * std::mem::size_of::<u64>());
        let parent = self
            .parent_source
            .as_ref()
            .map_or(0, |source| source.retained_device_bytes());
        [values, validity, root, normalization_fit, parent]
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .expect("admission validated steady resident extent")
    }

    pub(crate) fn parent_source(&self) -> &dyn ResidentParentDatasetSourceV3 {
        self.parent_source
            .as_deref()
            .expect("sealed store retains native parent arrays")
    }
    pub fn parent_dataset_layout(&self) -> &ResidentParentDatasetLayoutV4 {
        self.parent_source().parent_dataset_layout()
    }
}

impl Drop for ResidentFeatureStoreOwnerV3 {
    fn drop(&mut self) {
        let context_is_current = CurrentContext::set_current(self.context.as_ref()).is_ok();
        let ready = context_is_current && matches!(self.ready_event.query(), Ok(true));
        if context_is_current && !ready {
            let _ = self.ready_event.enqueue_wait(&self.producer_stream);
        }
        if let Ok(transient) = self.hash_transient.get_mut()
            && let Some(transient) = transient.take()
        {
            if context_is_current {
                let _ = transient.release(&self.producer_stream, ready);
            } else {
                std::mem::forget(transient);
            }
        }
        if let Some(parent) = self.parent_source.take() {
            if context_is_current {
                let _ = parent.enqueue_nonblocking_release(&self.producer_stream);
            } else {
                std::mem::forget(parent);
            }
        }
        drop(self.robust_normalization_fit_metadata_v2.take());
        drop(self.robust_normalization_ready_event_v2.take());
        drop(self.search_bar_major_values.take());
        drop(self.search_bar_major_validity_u4.take());
        drop(self.canonical_content_merkle.take());
    }
}

#[derive(Debug)]
#[must_use = "Search must record a consumer completion event before releasing this import"]
pub struct ResidentFeatureStoreImportV3 {
    owner: Option<Arc<ResidentFeatureStoreOwnerV3>>,
    consumer_context: Option<Arc<Context>>,
    consumer_stream: Option<Arc<Stream>>,
}

impl ResidentFeatureStoreImportV3 {
    pub fn rows(&self) -> usize {
        self.owner.as_ref().map_or(0, |owner| owner.rows)
    }
    pub fn columns(&self) -> usize {
        self.owner.as_ref().map_or(0, |owner| owner.columns)
    }

    pub fn device_ordinal(&self) -> Result<u32, ResidentFeatureStoreCudaErrorV3> {
        self.owner
            .as_ref()
            .map(|owner| owner.device_ordinal)
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident import has no admitted device ordinal".into(),
                )
            })
    }

    pub fn admission_identity_sha256(
        &self,
    ) -> Result<[u8; SHA256_BYTES], ResidentFeatureStoreCudaErrorV3> {
        self.owner
            .as_ref()
            .map(|owner| owner.admission_identity_sha256())
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident import has no admitted identity".into(),
                )
            })
    }

    /// Move the complete admitted V3 lifetime into the resident trim stage.
    /// The only host data copied is the compact schema classification; feature
    /// values, validity, parent prices and selected-column results stay on the
    /// admitted device and stream.
    pub fn consume_into_resident_trim_prefilter_v1(
        mut self,
        schema: ResidentTrimPrefilterSchemaUploadV1,
    ) -> Result<ResidentTrimPrefilterInputsV1, ResidentFeatureStoreCudaErrorV3> {
        let owner = self.owner.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("resident import was consumed".into())
        })?;
        let consumer_context = self.consumer_context.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("consumer context was consumed".into())
        })?;
        let consumer_stream = self.consumer_stream.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("consumer stream was consumed".into())
        })?;
        let admitted = owner.run_device.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident store lost its admitted run-device authority".into(),
            )
        })?;
        let full_trim = admitted.full_discovery_trim_admission().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident store has no sealed full-Discovery trim admission".into(),
            )
        })?;
        if owner.device_ordinal != admitted.device_identity.ordinal()
            || owner.device_ordinal != owner.parent_source().device_ordinal()
            || !Arc::ptr_eq(
                consumer_context,
                admitted.primary_context_for_resident_producer_v3(),
            )
            || !Arc::ptr_eq(consumer_context, &owner.context)
            || !Arc::ptr_eq(
                consumer_stream,
                admitted.run_stream_for_resident_producer_v3(),
            )
            || !Arc::ptr_eq(consumer_stream, &owner.producer_stream)
            || consumer_context.as_raw() != owner.context.as_raw()
            || consumer_stream.as_inner() != owner.producer_stream.as_inner()
            || owner.parent_source().producer_context().as_raw() != consumer_context.as_raw()
            || owner.parent_source().producer_stream().as_inner() != consumer_stream.as_inner()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        CurrentContext::set_current(consumer_context.as_ref())?;
        let expected_ordinal = i32::try_from(owner.device_ordinal).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident trim device ordinal ABI")
        })?;
        if CurrentContext::get_device()?.as_raw() != expected_ordinal
            || stream_context(consumer_stream)? != consumer_context.as_raw()
            || consumer_stream.as_inner().is_null()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }

        let parent_row_count = u64::try_from(owner.rows).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident trim parent rows")
        })?;
        let parent_column_count = u64::try_from(owner.columns).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident trim parent columns")
        })?;
        let cells = owner.rows.checked_mul(owner.columns).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident trim parent cells"),
        )?;
        let logical_validity_bytes = cells / 2 + cells % 2;
        let values = owner.search_bar_major_values.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed resident bar-major values were released".into(),
            )
        })?;
        let validity = owner.search_bar_major_validity_u4.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed resident packed validity was released".into(),
            )
        })?;
        let parent = owner.parent_source();
        if parent_row_count == 0
            || parent_column_count == 0
            || values.len() != cells
            || validity.len() != owner.packed_validity_allocated_bytes
            || validity.len() < logical_validity_bytes
            || schema.column_class_flags.len() != owner.columns
            || schema.timeframe_group_ids.len() != owner.columns
            || schema.template_force_keep_flags.len() != owner.columns
            || ordered_feature_schema_sha256_v1(&owner.column_bindings)
                != schema.ordered_feature_schema_sha256
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim parent or schema extent drifted".into(),
            ));
        }
        let compact_hashes = owner.compact_hashes_if_ready()?;
        let normalization_fit_sha256 = owner
            .robust_normalization_runtime_receipt_v2
            .as_ref()
            .map(ResidentRobustNormalizationRuntimeReceiptV2::fit_metadata_sha256)
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident store has no sealed normalization-fit identity".into(),
                )
            })?;
        if compact_hashes.canonical_content_merkle != schema.canonical_content_merkle_sha256
            || normalization_fit_sha256 != schema.normalization_fit_sha256
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim identities do not match the sealed V3 store".into(),
            ));
        }

        let packed_validity_bytes = u64::try_from(validity.len()).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident trim packed-validity bytes",
            )
        })?;
        let retained_schema_bytes = schema
            .column_class_flags
            .len()
            .checked_add(
                schema
                    .timeframe_group_ids
                    .len()
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                        "resident trim timeframe metadata bytes",
                    ))?,
            )
            .and_then(|bytes| bytes.checked_add(schema.template_force_keep_flags.len()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident trim schema metadata bytes",
            ))?;
        if retained_schema_bytes > full_trim.trim_prefilter_reserved_bytes() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim schema metadata exceeds its sealed workspace slice".into(),
            ));
        }

        let admitted_run_stream = NonNull::new(consumer_stream.as_inner().cast::<c_void>())
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident trim admitted stream is null".into(),
                )
            })?;
        let parent_ready_event = NonNull::new(owner.ready_event.raw().cast::<c_void>())
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident trim parent-ready event is null".into(),
                )
            })?;
        let indicators_bar_major =
            NonNull::new(values.as_device_ptr().as_ptr()).ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident trim feature pointer is null".into(),
                )
            })?;
        let indicators_validity_u4 =
            NonNull::new(validity.as_device_ptr().as_ptr()).ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident trim validity pointer is null".into(),
                )
            })?;
        let close = NonNull::new(parent.close().as_device_ptr().as_ptr()).ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim close pointer is null".into(),
            )
        })?;
        let high = NonNull::new(parent.high().as_device_ptr().as_ptr()).ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim high pointer is null".into(),
            )
        })?;
        let low = NonNull::new(parent.low().as_device_ptr().as_ptr()).ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident trim low pointer is null".into(),
            )
        })?;

        let admission_identity_sha256 = admitted.admission_identity_sha256();
        let workspace_plan_identity_sha256 = admitted.workspace_plan_identity_sha256();
        let selected_cuda_ordinal = owner.device_ordinal;
        let phase_one_free_bytes_snapshot = admitted.phase_one_free_bytes_snapshot();
        let allocator_context_reserve_bytes = admitted.allocator_context_reserve_bytes();
        let required_workspace_bytes = full_trim.required_workspace_bytes();
        let trim_prefilter_reserved_bytes = full_trim.trim_prefilter_reserved_bytes();
        let full_discovery_reserve_bytes = full_trim.full_discovery_reserve_bytes();
        let ordinal_bytes = selected_cuda_ordinal.to_le_bytes();
        let compute_major_bytes = admitted.compute_capability_major.to_le_bytes();
        let compute_minor_bytes = admitted.compute_capability_minor.to_le_bytes();
        let cuda_device_identity_sha256 = trim_identity_sha256_v1(
            b"neoethos.resident-trim-device-identity.v1",
            &[
                &admission_identity_sha256,
                &admitted.device_uuid,
                &ordinal_bytes,
                &compute_major_bytes,
                &compute_minor_bytes,
            ],
        );
        let primary_context_identity_sha256 =
            admitted.device_identity().primary_context_process_token();
        let run_stream_identity_sha256 = admitted.run_stream_process_token_v3();
        let vector_ta_build_sha256 = admitted.device_identity().vector_ta_build_sha256();
        let cuda_build_manifest_sha256 = trim_identity_sha256_v1(
            b"neoethos.resident-trim-build-manifest.v1",
            &[
                &admission_identity_sha256,
                &workspace_plan_identity_sha256,
                &vector_ta_build_sha256,
                admitted.device_identity().native_sass_target().as_bytes(),
                admitted.device_identity().nvcc_version().as_bytes(),
            ],
        );
        let mut math_hasher = Sha256::new();
        math_hasher.update(b"neoethos.resident-trim-cuda-math-flags.v1");
        for flag in RESIDENT_TRIM_PREFILTER_CUDA_MATH_FLAGS_V1 {
            math_hasher.update((flag.len() as u64).to_le_bytes());
            math_hasher.update(flag.as_bytes());
        }
        let cuda_math_flags_sha256 = math_hasher.finalize().into();

        // Both events are created before the first schema H2D. Once a copy is
        // attempted, every failure path below deliberately retires its pointer
        // identities through PendingResidentTrimSchemaUploadV1::drop.
        let trim_prefilter_ready_event = OwnedCudaEventV3::new()?;
        let schema_ready_event = OwnedCudaEventV3::new()?;
        let mut upload = PendingResidentTrimSchemaUploadV1::new(schema_ready_event);
        let (host_column_class_flags, column_class_flags) = compact_device_buffer_from_slice_async(
            &schema.column_class_flags,
            consumer_context,
            consumer_stream,
        )?;
        upload.host_column_class_flags = Some(host_column_class_flags);
        upload.column_class_flags = Some(column_class_flags);
        let (host_timeframe_group_ids, timeframe_group_ids) =
            compact_device_buffer_from_slice_async(
                &schema.timeframe_group_ids,
                consumer_context,
                consumer_stream,
            )?;
        upload.host_timeframe_group_ids = Some(host_timeframe_group_ids);
        upload.timeframe_group_ids = Some(timeframe_group_ids);
        let (host_template_force_keep_flags, template_force_keep_flags) =
            compact_device_buffer_from_slice_async(
                &schema.template_force_keep_flags,
                consumer_context,
                consumer_stream,
            )?;
        upload.host_template_force_keep_flags = Some(host_template_force_keep_flags);
        upload.template_force_keep_flags = Some(template_force_keep_flags);
        upload
            .ready_event
            .as_ref()
            .expect("armed trim schema upload retains its ready event")
            .record(consumer_stream)?;
        let schema_lifetime = upload.into_lifetime();
        let schema_ready_event = NonNull::new(schema_lifetime.ready_event.raw().cast::<c_void>())
            .expect("owned CUDA event is non-null");
        let column_class_flags_device =
            NonNull::new(schema_lifetime.column_class_flags.as_device_ptr().as_ptr())
                .expect("non-empty trim class allocation is non-null");
        let timeframe_group_ids_device =
            NonNull::new(schema_lifetime.timeframe_group_ids.as_device_ptr().as_ptr())
                .expect("non-empty trim timeframe allocation is non-null");
        let template_force_keep_flags_device = NonNull::new(
            schema_lifetime
                .template_force_keep_flags
                .as_device_ptr()
                .as_ptr(),
        )
        .expect("non-empty trim template allocation is non-null");
        let trim_prefilter_ready_event_raw =
            NonNull::new(trim_prefilter_ready_event.raw().cast::<c_void>())
                .expect("owned CUDA event is non-null");
        let admission_lifetime = ResidentTrimAdmissionLifetimeV1 {
            _owner: Arc::clone(owner),
            ready_event: trim_prefilter_ready_event,
        };
        debug_assert_eq!(
            admission_lifetime.ready_event.raw().cast::<c_void>(),
            trim_prefilter_ready_event_raw.as_ptr()
        );

        // No fallible operation follows these three takes. The complete V3
        // lifetime is moved into the parent token atomically; Drop can no
        // longer observe a half-consumed import.
        let retained_owner = self.owner.take().expect("validated V3 owner");
        let retained_context = self.consumer_context.take().expect("validated context");
        let retained_stream = self.consumer_stream.take().expect("validated stream");
        let retained_import = ResidentFeatureStoreImportV3 {
            owner: Some(retained_owner),
            consumer_context: Some(retained_context),
            consumer_stream: Some(retained_stream),
        };
        let parent_import = ResidentTrimPrefilterParentImportV1 {
            owner: Some(Box::new(retained_import)),
            selected_cuda_ordinal,
            parent_row_count,
            parent_column_count,
            packed_validity_bytes,
            admitted_run_stream,
            parent_ready_event,
            indicators_bar_major,
            indicators_validity_u4,
            close,
            high,
            low,
            canonical_search_input_receipt_sha256: schema.canonical_search_input_receipt_sha256,
            canonical_content_merkle_sha256: schema.canonical_content_merkle_sha256,
            normalization_fit_sha256: schema.normalization_fit_sha256,
            feature_plan_sha256: schema.feature_plan_sha256,
            source_provenance_sha256: schema.source_provenance_sha256,
            cuda_device_identity_sha256,
            primary_context_identity_sha256,
            run_stream_identity_sha256,
            cuda_build_manifest_sha256,
            cuda_math_flags_sha256,
        };
        let sealed_schema = SealedResidentColumnClassificationV1 {
            owner: Some(Box::new(schema_lifetime)),
            selected_cuda_ordinal,
            parent_column_count,
            retained_device_bytes: retained_schema_bytes,
            timeframe_group_count: schema.timeframe_group_count,
            schema_ready_event,
            column_class_flags_device,
            timeframe_group_ids_device,
            template_force_keep_flags_device,
            ordered_feature_schema_sha256: schema.ordered_feature_schema_sha256,
            column_classification_content_sha256: schema.column_classification_content_sha256,
            primary_context_identity_sha256,
            run_stream_identity_sha256,
            cuda_build_manifest_sha256,
        };
        let full_admission = ResidentTrimPrefilterFullDiscoveryAdmissionV1 {
            owner: Some(Box::new(admission_lifetime)),
            selected_cuda_ordinal,
            trim_prefilter_ready_event: trim_prefilter_ready_event_raw,
            trim_prefilter_reserved_bytes,
            full_discovery_reserve_bytes,
            primary_context_identity_sha256,
            run_stream_identity_sha256,
            cuda_build_manifest_sha256,
        };
        let identity = ResidentTrimPrefilterImportIdentityV1 {
            admission_identity_sha256,
            workspace_plan_identity_sha256,
            canonical_search_input_receipt_sha256: schema.canonical_search_input_receipt_sha256,
            canonical_content_merkle_sha256: schema.canonical_content_merkle_sha256,
            normalization_fit_sha256: schema.normalization_fit_sha256,
            feature_plan_sha256: schema.feature_plan_sha256,
            source_provenance_sha256: schema.source_provenance_sha256,
            ordered_feature_schema_sha256: schema.ordered_feature_schema_sha256,
            column_classification_content_sha256: schema.column_classification_content_sha256,
            selected_cuda_ordinal,
            parent_row_count,
            parent_column_count,
            cuda_device_identity_sha256,
            primary_context_identity_sha256,
            run_stream_identity_sha256,
            cuda_build_manifest_sha256,
            cuda_math_flags_sha256,
            phase_one_free_bytes_snapshot,
            allocator_context_reserve_bytes,
            required_workspace_bytes,
            trim_prefilter_reserved_bytes,
            full_discovery_reserve_bytes,
        };
        Ok(ResidentTrimPrefilterInputsV1 {
            parent_import,
            sealed_schema,
            full_admission,
            identity,
        })
    }

    pub fn consume_into_population_session_v3(
        self,
    ) -> Result<ResidentPopulationSessionV3, ResidentFeatureStoreCudaErrorV3> {
        let owner = self.owner.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("resident import was consumed".into())
        })?;
        let consumer_context = self.consumer_context.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("consumer context was consumed".into())
        })?;
        let consumer_stream = self.consumer_stream.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("consumer stream was consumed".into())
        })?;
        let admitted = owner.run_device.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident store lost its admitted run-device authority".into(),
            )
        })?;
        let admitted_primary_context = admitted.primary_context_for_resident_producer_v3();
        let admitted_run_stream = admitted.run_stream_for_resident_producer_v3();
        if owner.device_ordinal != admitted.device_identity.ordinal()
            || owner.device_ordinal != owner.parent_source().device_ordinal()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::DeviceMismatch);
        }
        if !Arc::ptr_eq(consumer_context, admitted_primary_context)
            || !Arc::ptr_eq(consumer_context, &owner.context)
            || consumer_context.as_raw() != admitted_primary_context.as_raw()
            || consumer_context.as_raw() != owner.context.as_raw()
            || owner.parent_source().producer_context().as_raw() != consumer_context.as_raw()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        if !Arc::ptr_eq(consumer_stream, admitted_run_stream)
            || !Arc::ptr_eq(consumer_stream, &owner.producer_stream)
            || consumer_stream.as_inner() != admitted_run_stream.as_inner()
            || consumer_stream.as_inner() != owner.producer_stream.as_inner()
            || owner.parent_source().producer_stream().as_inner() != consumer_stream.as_inner()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        CurrentContext::set_current(consumer_context.as_ref())?;
        let expected_ordinal = i32::try_from(owner.device_ordinal).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident device ordinal ABI")
        })?;
        if CurrentContext::get_device()?.as_raw() != expected_ordinal
            || stream_context(consumer_stream)? != consumer_context.as_raw()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }

        let rows = owner.rows;
        let columns = owner.columns;
        let parent = owner.parent_source();
        let parent_layout = parent.parent_dataset_layout();
        let row_count = u64::try_from(rows).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident population rows ABI")
        })?;
        let feature_count = u32::try_from(columns).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident population feature count ABI",
            )
        })?;
        let cells = rows.checked_mul(columns).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident population cells"),
        )?;
        let logical_validity_bytes = cells / 2 + cells % 2;
        let expected_smc_cells = rows.checked_mul(SMC_SLOTS_V3).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident population SMC cells"),
        )?;
        let values = owner.search_bar_major_values.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed resident bar-major values were released".into(),
            )
        })?;
        let validity = owner.search_bar_major_validity_u4.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed resident packed validity was released".into(),
            )
        })?;
        if rows == 0
            || columns == 0
            || parent.rows() != rows
            || parent_layout.row_count() != row_count
            || values.len() != cells
            || owner.packed_validity_logical_bytes != logical_validity_bytes
            || validity.len() != owner.packed_validity_allocated_bytes
            || validity.len() < logical_validity_bytes
            || parent.close().len() != rows
            || parent.high().len() != rows
            || parent.low().len() != rows
            || parent.timestamps().len() != rows
            || parent.months().len() != rows
            || parent.days().len() != rows
            || parent.smc_rows().len() != expected_smc_cells
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "sealed resident population scope or shape drifted".into(),
            ));
        }
        let compact_hashes = owner.compact_hashes_if_ready()?;
        let transient_retirement = owner
            .take_data_transient_retirement_proof_v1()?
            .synchronize_before_population_allocation(
                consumer_context,
                consumer_stream,
                owner.device_ordinal,
                admitted.admission_identity_sha256(),
                admitted.workspace_plan_identity_sha256(),
            )?;
        let packed_validity_bytes = u64::try_from(validity.len()).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident packed-validity ABI bytes",
            )
        })?;
        let raw = RawResidentFeatureStoreBindV3 {
            abi_version: neoethos_gpu_contracts::ABI_VERSION,
            selected_device_ordinal: owner.device_ordinal,
            row_count,
            feature_count,
            smc_slots: SMC_SLOTS_V3 as u32,
            compute_capability_major: admitted.compute_capability_major,
            compute_capability_minor: admitted.compute_capability_minor,
            reserved: 0,
            packed_validity_bytes,
            close: parent.close().as_device_ptr().as_ptr(),
            high: parent.high().as_device_ptr().as_ptr(),
            low: parent.low().as_device_ptr().as_ptr(),
            indicators_bar_major: values.as_device_ptr().as_ptr(),
            indicators_validity_u4: validity.as_device_ptr().as_ptr(),
            months: parent.months().as_device_ptr().as_ptr(),
            days: parent.days().as_device_ptr().as_ptr(),
            timestamps: parent.timestamps().as_device_ptr().as_ptr(),
            smc_rows: parent.smc_rows().as_device_ptr().as_ptr(),
            admitted_primary_context: consumer_context.as_raw().cast(),
            admitted_run_stream: consumer_stream.as_inner().cast(),
            ready_event: owner.ready_event.raw().cast(),
            device_uuid: admitted.device_uuid,
            admission_identity_sha256: admitted.admission_identity_sha256(),
            canonical_content_merkle: compact_hashes.canonical_content_merkle,
            allocator_context_reserve_bytes: admitted.allocator_context_reserve_bytes(),
            run_stream_process_token_v3: admitted.run_stream_process_token_v3(),
        };
        let (population_session, transient_retirement) =
            bind_population_after_data_transient_retirement_v1(
                transient_retirement,
                admitted.admission_identity_sha256(),
                admitted.workspace_plan_identity_sha256(),
                raw,
            )?;
        let device_identity = admitted.device_identity().clone();
        let data_population_limits = admitted.data_population_limits().copied();
        let parent_dataset_layout = parent_layout.clone();
        let admission_identity_sha256 = admitted.admission_identity_sha256();
        let pre_materialization_free_bytes_snapshot = owner.pre_materialization_free_bytes_snapshot;
        Ok(ResidentPopulationSessionV3 {
            population_session,
            resident_import: Some(self),
            consumer_lease: None,
            slice2_population_detached: false,
            admission_identity_sha256,
            canonical_content_merkle: compact_hashes.canonical_content_merkle,
            device_identity,
            parent_dataset_layout,
            pre_materialization_free_bytes_snapshot,
            data_transient_retirement_process_token: transient_retirement
                .retirement_event_process_token,
            data_population_limits,
            rows,
            columns,
        })
    }

    pub fn record_consumer_completion(
        mut self,
    ) -> Result<ResidentFeatureStoreConsumerLeaseV3, ResidentFeatureStoreCudaErrorV3> {
        let owner = self.owner.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("resident import was consumed".into())
        })?;
        let consumer_context = self.consumer_context.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("consumer context was consumed".into())
        })?;
        let consumer_stream = self.consumer_stream.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput("consumer stream was consumed".into())
        })?;
        CurrentContext::set_current(consumer_context.as_ref())?;
        let consumer_completion_event = OwnedCudaEventV3::new()?;
        consumer_completion_event.record(consumer_stream)?;
        // The producer stream becomes the stream-ordered deallocation queue.
        // Its wait ensures final async frees cannot overtake Search reads.
        consumer_completion_event.enqueue_wait(&owner.producer_stream)?;
        // Move the complete lifetime out only after the completion event was
        // recorded and the producer-stream wait was accepted. On every prior
        // error `self` remains armed and its Drop leaks rather than frees live
        // consumer allocations.
        let lifetime = ResidentConsumerLifetimeV3 {
            population_session: None,
            owner: self
                .owner
                .take()
                .expect("validated import retains resident owner"),
            consumer_stream: self
                .consumer_stream
                .take()
                .expect("validated import retains consumer stream"),
            consumer_context: self
                .consumer_context
                .take()
                .expect("validated import retains consumer context"),
        };
        Ok(ResidentFeatureStoreConsumerLeaseV3 {
            consumer_completion_event,
            lifetime: Some(lifetime),
        })
    }
}

/// One move-only native population session armed to one sealed V3 store. The
/// imported device buffers and native session cannot be detached or cloned.
#[derive(Debug)]
#[must_use = "record the resident consumer completion event and retain its lease"]
pub struct ResidentPopulationSessionV3 {
    population_session: PopulationSession,
    resident_import: Option<ResidentFeatureStoreImportV3>,
    consumer_lease: Option<ResidentFeatureStoreConsumerLeaseV3>,
    slice2_population_detached: bool,
    admission_identity_sha256: [u8; SHA256_BYTES],
    canonical_content_merkle: [u8; SHA256_BYTES],
    device_identity: CudaPrimaryContextBuildIdentityV3,
    parent_dataset_layout: ResidentParentDatasetLayoutV4,
    pre_materialization_free_bytes_snapshot: u64,
    data_transient_retirement_process_token: [u8; SHA256_BYTES],
    data_population_limits: Option<SealedDataPopulationExecutionLimitsV1>,
    rows: usize,
    columns: usize,
}

#[must_use = "record Search completion and retain the resident consumer lease"]
pub struct ResidentFeatureStoreSearchRunV2 {
    search_run: Option<ResidentSearchRunV2>,
    resident_import: Option<ResidentFeatureStoreImportV3>,
}

/// Move-only start failure that preserves the V3 owner until an event recorded
/// after every attempted Search enqueue reaches the admitted stream boundary.
#[derive(Debug, Error)]
pub(crate) enum ResidentFeatureStoreSearchStartErrorV2 {
    #[error(transparent)]
    Precondition(#[from] ResidentFeatureStoreCudaErrorV3),
    #[error("resident Search start failed; a completion lease retains the V3 owner: {source}")]
    Search {
        #[source]
        source: ResidentSearchV2Error,
        cleanup_lease: ResidentFeatureStoreConsumerLeaseV3,
    },
    #[error(
        "resident Search start failed ({search}); recording its V3 cleanup event also failed: {cleanup}"
    )]
    CleanupEvent {
        search: ResidentSearchV2Error,
        #[source]
        cleanup: ResidentFeatureStoreCudaErrorV3,
    },
}

impl ResidentFeatureStoreSearchStartErrorV2 {
    #[allow(dead_code)] // The next private Search coordinator retains this lease explicitly.
    pub(crate) fn into_cleanup_lease(self) -> Option<ResidentFeatureStoreConsumerLeaseV3> {
        match self {
            Self::Search { cleanup_lease, .. } => Some(cleanup_lease),
            Self::Precondition(_) | Self::CleanupEvent { .. } => None,
        }
    }
}

impl ResidentFeatureStoreSearchRunV2 {
    #[allow(dead_code)] // The next Search chunk consumes this private enqueue seam.
    pub(crate) fn upload_resident_scenarios_v2(
        &mut self,
        scenarios: &[ScenarioDescriptor],
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        self.search_run
            .as_mut()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident Search run was already consumed".into(),
                )
            })?
            .upload_resident_scenarios_v2(scenarios)?;
        Ok(())
    }

    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) fn enqueue_resident_gene_metrics_fixture_v2(
        &mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<ResidentPopulationMetricsV1<'_>, ResidentFeatureStoreCudaErrorV3> {
        self.search_run
            .as_mut()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident Search run was already consumed".into(),
                )
            })?
            .enqueue_resident_gene_metrics_fixture_v2(settings)
            .map_err(Into::into)
    }

    pub fn record_consumer_completion(
        mut self,
    ) -> Result<ResidentFeatureStoreConsumerLeaseV3, ResidentFeatureStoreCudaErrorV3> {
        let search_run = self.search_run.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Search run was already consumed".into(),
            )
        })?;
        let population_session = search_run.close_v2()?;
        let resident_import = self.resident_import.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Search import was already consumed".into(),
            )
        })?;
        let mut consumer_lease = resident_import.record_consumer_completion()?;
        consumer_lease.attach_population_session_v3(population_session)?;
        Ok(consumer_lease)
    }
}

impl ResidentPopulationSessionV3 {
    pub(crate) fn take_population_session_for_slice2_v3(
        &mut self,
    ) -> Result<PopulationSession, ResidentFeatureStoreCudaErrorV3> {
        if self.slice2_population_detached
            || self.resident_import.is_none()
            || self.consumer_lease.is_some()
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Slice2 population owner is not available".into(),
            ));
        }
        self.slice2_population_detached = true;
        Ok(self
            .population_session
            .take_for_resident_consumer_lease_v3())
    }

    pub(crate) fn restore_population_session_from_slice2_v3(
        &mut self,
        session: PopulationSession,
    ) -> Result<(), PopulationSession> {
        if !self.slice2_population_detached {
            return Err(session);
        }
        self.population_session = session;
        self.slice2_population_detached = false;
        Ok(())
    }

    #[allow(dead_code)] // First bounded production owner seam; next chunk calls it.
    pub(crate) fn consume_into_resident_search_run_v2(
        mut self,
        plan: SealedResidentGenerationPlanV1,
        smc_weights: [f64; SMC_SLOTS_V3],
        smc_gate_disabled: bool,
    ) -> Result<ResidentFeatureStoreSearchRunV2, ResidentFeatureStoreSearchStartErrorV2> {
        if self.consumer_lease.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident population session already owns a completion lease".into(),
            )
            .into());
        }
        let resident_import = self.resident_import.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident population import was already consumed".into(),
            )
        })?;
        let population_session = self
            .population_session
            .take_for_resident_consumer_lease_v3();
        match population_session.begin_resident_search_from_plan_v2(
            plan,
            smc_weights,
            smc_gate_disabled,
        ) {
            Ok(search_run) => Ok(ResidentFeatureStoreSearchRunV2 {
                search_run: Some(search_run),
                resident_import: Some(resident_import),
            }),
            Err(source) => match resident_import.record_consumer_completion() {
                Ok(cleanup_lease) => Err(ResidentFeatureStoreSearchStartErrorV2::Search {
                    source,
                    cleanup_lease,
                }),
                Err(cleanup) => Err(ResidentFeatureStoreSearchStartErrorV2::CleanupEvent {
                    search: source,
                    cleanup,
                }),
            },
        }
    }

    pub const fn admission_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.admission_identity_sha256
    }

    pub const fn canonical_content_merkle(&self) -> [u8; SHA256_BYTES] {
        self.canonical_content_merkle
    }

    pub fn device_identity(&self) -> &CudaPrimaryContextBuildIdentityV3 {
        &self.device_identity
    }

    pub fn parent_dataset_layout(&self) -> &ResidentParentDatasetLayoutV4 {
        &self.parent_dataset_layout
    }

    /// Exact free-memory snapshot captured on the admitted primary context
    /// before the resident parent was materialized. Search may use this sealed
    /// value for deterministic population sizing; it must not probe CUDA again.
    pub const fn pre_materialization_free_bytes_snapshot(&self) -> u64 {
        self.pre_materialization_free_bytes_snapshot
    }

    pub const fn data_transient_retirement_process_token(&self) -> [u8; SHA256_BYTES] {
        self.data_transient_retirement_process_token
    }

    pub const fn data_population_limits(&self) -> Option<&SealedDataPopulationExecutionLimitsV1> {
        self.data_population_limits.as_ref()
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub fn bind_evaluation_view_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if let Some(limits) = self.data_population_limits {
            let parent_rows = u64::try_from(self.rows).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent rows do not fit the stage authority".into(),
                )
            })?;
            let feature_count = u64::try_from(self.columns).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident feature count does not fit the stage authority".into(),
                )
            })?;
            let view_rows = u64::try_from(view.row_count()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population view rows do not fit the stage authority".into(),
                )
            })?;
            let ordered_rows = u64::try_from(view.ordered_index_values().map_or(0, <[u64]>::len))
                .map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population ordered-view rows do not fit the stage authority".into(),
                )
            })?;
            let adaptive_rows = u64::try_from(view.adaptive_base_pips().map_or(0, <[f64]>::len))
                .map_err(|_| {
                    ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "population adaptive-view rows do not fit the stage authority".into(),
                    )
                })?;
            if limits.parent_row_count() != parent_rows
                || limits.feature_count() != feature_count
                || view_rows > limits.parent_row_count()
                || ordered_rows > limits.max_ordered_index_count()
                || adaptive_rows > limits.max_adaptive_row_count()
            {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population evaluation view exceeds the sealed Data+population workspace"
                        .into(),
                ));
            }
        }
        self.population_session
            .bind_evaluation_view_v1(view)
            .map_err(Into::into)
    }

    /// Bind a full/contiguous view and produce its canonical adaptive-stop base
    /// directly from the resident parent on the admitted stream. The host view
    /// must contain no adaptive slice; its exact row extent is charged against
    /// the sealed adaptive capacity as if all output rows were already live.
    pub(crate) fn bind_evaluation_view_with_resident_adaptive_base_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
        request: ResidentAdaptiveBaseRequestV1,
    ) -> Result<&ResidentAdaptiveBaseViewTokenV1, ResidentFeatureStoreCudaErrorV3> {
        if let Some(limits) = self.data_population_limits {
            let parent_rows = u64::try_from(self.rows).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident parent rows do not fit the adaptive stage authority".into(),
                )
            })?;
            let feature_count = u64::try_from(self.columns).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident feature count does not fit the adaptive stage authority".into(),
                )
            })?;
            let view_rows = u64::try_from(view.row_count()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident adaptive view rows do not fit the stage authority".into(),
                )
            })?;
            if limits.parent_row_count() != parent_rows
                || limits.feature_count() != feature_count
                || view_rows > limits.parent_row_count()
                || view_rows > limits.max_adaptive_row_count()
                || request.parent_row_count() != parent_rows
                || request.view_row_count() != view_rows
            {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident adaptive view exceeds or drifts from the sealed Data+population workspace"
                        .into(),
                ));
            }
        }
        self.population_session
            .bind_evaluation_view_with_resident_adaptive_base_v1(view, request)
            .map_err(Into::into)
    }

    /// Bind the resident adaptive view and immediately give the exact
    /// non-Clone token borrowed from this session to the caller's receipt
    /// validator. The public API accepts no caller-supplied token, so a token
    /// from another or earlier session cannot be substituted. Validator
    /// rejection poisons and clears the bound state before any upload can run.
    /// Copyable evidence is returned only after validation and is not itself
    /// an authorization capability.
    pub fn bind_evaluation_view_with_resident_adaptive_base_checked_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
        request: ResidentAdaptiveBaseRequestV1,
        validator: impl FnOnce(
            &ResidentAdaptiveBaseViewTokenV1,
        ) -> Result<(), ResidentFeatureStoreCudaErrorV3>,
    ) -> Result<ResidentAdaptiveBaseViewTokenIdentityV1, ResidentFeatureStoreCudaErrorV3> {
        self.bind_evaluation_view_with_resident_adaptive_base_v1(view, request)?;
        let (facts, validation) = {
            let current = self
                .population_session
                .arm_resident_adaptive_validator_guard_v1()?;
            let facts = current.identity_facts_v1();
            (facts, validator(current))
        };
        if let Err(error) = validation {
            self.population_session
                .poison_after_resident_adaptive_validator_rejection_v1();
            return Err(error);
        }
        self.population_session
            .accept_resident_adaptive_validator_guard_v1()?;
        Ok(facts)
    }

    #[cfg(feature = "cuda-device-fixtures")]
    #[cfg_attr(not(test), allow(dead_code))] // Used only by the feature-gated device oracle.
    pub(crate) fn copy_resident_adaptive_base_fixture_v1(
        &mut self,
    ) -> Result<Vec<f64>, ResidentFeatureStoreCudaErrorV3> {
        self.population_session
            .copy_resident_adaptive_base_fixture_v1()
            .map_err(Into::into)
    }

    pub fn upload_genes(
        &mut self,
        genes: PopulationGeneView<'_>,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if let Some(limits) = self.data_population_limits {
            let candidate_count = u64::try_from(genes.descriptors.len()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population candidate count does not fit the stage authority".into(),
                )
            })?;
            let term_count = u64::try_from(genes.indices.len()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population gene-term count does not fit the stage authority".into(),
                )
            })?;
            if candidate_count > limits.max_candidate_count()
                || term_count > limits.max_gene_term_count()
            {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population gene upload exceeds the sealed Data+population workspace".into(),
                ));
            }
        }
        self.population_session
            .upload_genes(genes)
            .map_err(Into::into)
    }

    pub fn upload_scenarios(
        &mut self,
        scenarios: &[ScenarioDescriptor],
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if let Some(limits) = self.data_population_limits {
            let scenario_count = u64::try_from(scenarios.len()).map_err(|_| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population scenario count does not fit the stage authority".into(),
                )
            })?;
            if scenario_count > limits.max_concurrent_scenario_count() {
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "population scenario upload exceeds the sealed concurrent chunk cap".into(),
                ));
            }
        }
        self.population_session
            .upload_scenarios(scenarios)
            .map_err(Into::into)
    }

    pub fn enqueue_metrics_only_v1(
        &mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<ResidentPopulationMetricsV1<'_>, ResidentFeatureStoreCudaErrorV3> {
        if self
            .data_population_limits
            .is_some_and(|limits| u64::from(settings.month_capacity) != limits.month_capacity())
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "population month capacity differs from the sealed workspace authority".into(),
            ));
        }
        self.population_session
            .enqueue_metrics_only_v1(settings)
            .map_err(Into::into)
    }

    pub fn read_residency_counters_v1(
        &self,
    ) -> Result<PopulationResidencyCountersV1, ResidentFeatureStoreCudaErrorV3> {
        self.population_session
            .read_residency_counters_v1()
            .map_err(Into::into)
    }

    pub fn record_consumer_completion(
        mut self,
    ) -> Result<ResidentFeatureStoreConsumerLeaseV3, ResidentFeatureStoreCudaErrorV3> {
        if self.slice2_population_detached {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Slice2 population owner was not restored".into(),
            ));
        }
        let resident_import = self.resident_import.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident population import was already consumed".into(),
            )
        })?;
        let population_session = self
            .population_session
            .take_for_resident_consumer_lease_v3();
        let mut consumer_lease = resident_import.record_consumer_completion()?;
        consumer_lease.attach_population_session_v3(population_session)?;
        self.consumer_lease = Some(consumer_lease);
        Ok(self
            .consumer_lease
            .take()
            .expect("fresh resident completion retains its consumer lease"))
    }
}

impl Drop for ResidentPopulationSessionV3 {
    fn drop(&mut self) {
        if self.resident_import.is_some() {
            self.population_session.arm_resident_session_leak_only_v3();
        }
    }
}

impl Drop for ResidentFeatureStoreImportV3 {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            // An unsealed import may have queued native reads but has no
            // completion event. Leak its complete lifetime rather than free
            // early or synchronize the host.
            std::mem::forget(owner);
        }
        if let Some(context) = self.consumer_context.take() {
            std::mem::forget(context);
        }
        if let Some(stream) = self.consumer_stream.take() {
            std::mem::forget(stream);
        }
    }
}

#[derive(Debug)]
#[must_use = "the resident store lease must outlive every queued consumer read"]
pub struct ResidentFeatureStoreConsumerLeaseV3 {
    consumer_completion_event: OwnedCudaEventV3,
    lifetime: Option<ResidentConsumerLifetimeV3>,
}

impl ResidentFeatureStoreConsumerLeaseV3 {
    fn attach_population_session_v3(
        &mut self,
        mut population_session: PopulationSession,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let lifetime = self.lifetime.as_mut().ok_or_else(|| {
            population_session.arm_resident_session_leak_only_v3();
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident consumer lifetime was already released".into(),
            )
        })?;
        if lifetime.population_session.is_some() {
            population_session.arm_resident_session_leak_only_v3();
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident consumer lease already owns a population session".into(),
            ));
        }
        lifetime.population_session = Some(population_session);
        Ok(())
    }

    pub fn completion_is_ready(&self) -> Result<bool, ResidentFeatureStoreCudaErrorV3> {
        let lifetime = self.lifetime.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident consumer lifetime was already released".into(),
            )
        })?;
        CurrentContext::set_current(lifetime.consumer_context.as_ref())?;
        if stream_context(&lifetime.consumer_stream)? != lifetime.consumer_context.as_raw() {
            return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
        }
        self.consumer_completion_event.query()
    }

    pub fn rows(&self) -> usize {
        self.lifetime
            .as_ref()
            .map_or(0, |lifetime| lifetime.owner.rows)
    }

    pub fn columns(&self) -> usize {
        self.lifetime
            .as_ref()
            .map_or(0, |lifetime| lifetime.owner.columns)
    }

    fn leak_owner_if_consumer_is_still_running(&mut self) {
        if let Some(lifetime) = self.lifetime.take() {
            std::mem::forget(lifetime);
        }
    }
}

impl Drop for ResidentFeatureStoreConsumerLeaseV3 {
    fn drop(&mut self) {
        if self.lifetime.is_some() && matches!(self.consumer_completion_event.query(), Ok(true)) {
            if let Some(population_session) = self
                .lifetime
                .as_mut()
                .and_then(|lifetime| lifetime.population_session.as_mut())
            {
                population_session.authorize_resident_session_destroy_v3();
            }
        } else if self.lifetime.is_some() {
            self.leak_owner_if_consumer_is_still_running();
        }
    }
}

impl Drop for ResidentFeatureStoreAssemblerV3 {
    fn drop(&mut self) {
        let context_is_current = CurrentContext::set_current(self.context.as_ref()).is_ok();
        if let Some(pending) = self.pending_batch.take() {
            if context_is_current {
                let host_copy_is_complete = matches!(pending.batch_ready_event.query(), Ok(true));
                if !host_copy_is_complete {
                    let _ = pending
                        .batch_ready_event
                        .enqueue_wait(&self.producer_stream);
                }
                let _ = pending.release(&self.producer_stream, host_copy_is_complete);
            } else {
                std::mem::forget(pending);
            }
        }
        if let Some(parent) = self.parent_source.take() {
            if context_is_current {
                let _ = parent.enqueue_nonblocking_release(&self.producer_stream);
            } else {
                std::mem::forget(parent);
            }
        }
        drop(self.search_bar_major_values.take());
        drop(self.search_bar_major_validity_u4.take());
        drop(self.validity_code_error.take());
        drop(self.robust_normalization_fit_metadata_v2.take());
        drop(self.robust_normalization_ready_event_v2.take());
        if let Some(run_device) = self.run_device.take() {
            // An unsealed assembler may have queued producer waits, compact
            // metadata transfers, or pack work. Do not destroy its one-shot
            // stream/context and implicitly wait on an early/error path.
            std::mem::forget(run_device);
        }
    }
}
