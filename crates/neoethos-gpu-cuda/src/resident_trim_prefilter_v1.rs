//! Opaque same-run CUDA row-trim, correlation-prefilter and view authority.
//!
//! This module is additive and deliberately unexported. Its one-shot inputs
//! can only be minted by the future resident-store/session bridge. No pointer,
//! event, selected count or selected-column list is exposed outside gpu-cuda.

use crate::data_population_workspace_plan_v1::SealedNativeCudaDataPopulationPreflightFactsV1;
use crate::resident_feature_store_v3::{
    ResidentFeatureStoreCudaErrorV3, ResidentFeatureStoreImportV3, ResidentPopulationSessionV3,
};
use cust::error::CudaError;
use cust::memory::LockedBuffer;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::ffi::c_void;
use std::mem;
use std::ptr::NonNull;

const ABI_VERSION_V1: u32 = 1;
pub(crate) const SCREENING_IMPORT_ABI_VERSION_V2: u32 = 2;
const SCORE_BATCH_ABI_VERSION_V2: u32 = 2;
const SELECTED_MAP_READ_ABI_VERSION_V2: u32 = 2;
const STATUS_OK_V1: i32 = 0;
const STAGE_LABELS_V1: u32 = 1;
const STAGE_LABEL_GUARD_V1: u32 = 2;
const STAGE_FOLDS_V1: u32 = 3;
const STAGE_CORRELATIONS_V1: u32 = 4;
const STAGE_RANK_V1: u32 = 5;
const STAGE_QUOTAS_V1: u32 = 6;
const STAGE_ASCENDING_MAP_V1: u32 = 7;
const STAGE_DEVICE_SEAL_V1: u32 = 8;
const MAX_GRID_X_V1: u64 = i32::MAX as u64;
const LAUNCH_THREADS_V1: u64 = 256;
const LABEL_CENSUS_COUNTER_COUNT_V1: u64 = 12;
const MAXIMUM_REFIT_FOLDS_V1: u64 = 8;
const FOLD_DESCRIPTOR_BYTES_V1: u64 = 80;
const DEVICE_SEAL_BYTES_V1: u64 = 88;

pub const RESIDENT_TRIM_PREFILTER_CUDA_MATH_FLAGS_V1: [&str; 4] = [
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
];

#[derive(Debug)]
pub enum ResidentTrimPrefilterDeviceErrorV1 {
    InvalidPlan(&'static str),
    IdentityMismatch(&'static str),
    ArithmeticOverflow(&'static str),
    AllocationReceiptMismatch,
    MissingExactSelectedIndexDeviceParity,
    RunStateViolation,
    Native {
        operation: &'static str,
        status: i32,
    },
    Population(ResidentFeatureStoreCudaErrorV3),
}

impl From<ResidentFeatureStoreCudaErrorV3> for ResidentTrimPrefilterDeviceErrorV1 {
    fn from(error: ResidentFeatureStoreCudaErrorV3) -> Self {
        Self::Population(error)
    }
}

impl From<CudaError> for ResidentTrimPrefilterDeviceErrorV1 {
    fn from(error: CudaError) -> Self {
        Self::Population(error.into())
    }
}

/// Number of label-safe rows in the single configured selection-prefix fit.
///
/// This is planning geometry, not device evaluation evidence. `selection_rows`
/// already excludes the outer holdout and incorporates the configured suffix
/// cap. A result below three means keep every column, as in CPU Discovery.
pub fn resident_trim_prefilter_prefix_fit_rows_v1(
    selection_rows: u64,
    insample_fraction: f64,
    max_hold_bars: u64,
) -> Result<u64, ResidentTrimPrefilterDeviceErrorV1> {
    if selection_rows > (1_u64 << 53)
        || !insample_fraction.is_finite()
        || insample_fraction <= 0.0
        || insample_fraction > 1.0
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
            "selection-prefix fit geometry",
        ));
    }
    let requested_end = (insample_fraction * selection_rows as f64).floor() as u64;
    Ok(requested_end
        .min(selection_rows)
        .saturating_sub(max_hold_bars.max(1)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTrimPrefilterRunStateV1 {
    StrictIdle,
    InFlight,
    Sealed,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTrimPrefilterArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTrimPrefilterPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawResidentTrimPrefilterImportV1 {
    abi_version: u32,
    selected_cuda_ordinal: u32,
    parent_row_count: u64,
    parent_column_count: u64,
    packed_validity_bytes: u64,
    schema_metadata_bytes: u64,
    timeframe_group_count: u64,
    full_discovery_reserve_bytes: u64,
    trim_prefilter_reserved_bytes: u64,
    admitted_run_stream: *mut c_void,
    parent_ready_event: *mut c_void,
    schema_ready_event: *mut c_void,
    trim_prefilter_ready_event: *mut c_void,
    parent_lifetime_owner: *mut c_void,
    schema_lifetime_owner: *mut c_void,
    indicators_bar_major: *const f64,
    indicators_validity_u4: *const u8,
    close: *const f64,
    high: *const f64,
    low: *const f64,
    column_class_flags_device: *const u8,
    timeframe_group_ids_device: *const u32,
    template_force_keep_flags_device: *const u8,
    canonical_search_input_receipt_sha256: [u8; 32],
    canonical_content_merkle_sha256: [u8; 32],
    normalization_fit_sha256: [u8; 32],
    feature_plan_sha256: [u8; 32],
    source_provenance_sha256: [u8; 32],
    ordered_feature_schema_sha256: [u8; 32],
    column_classification_content_sha256: [u8; 32],
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawResidentTrimPrefilterPlanV1 {
    abi_version: u32,
    atr_period: u32,
    parent_row_count: u64,
    parent_column_count: u64,
    global_row_cap: u64,
    timeframe_row_cap: u64,
    outer_split_at: u64,
    selection_row_start: u64,
    selection_row_end: u64,
    holdout_row_start: u64,
    holdout_row_end: u64,
    configured_top_k: u64,
    resolved_top_k: u64,
    minimum_per_timeframe: u64,
    max_hold_bars: u64,
    minimum_pairwise_samples: u64,
    minimum_decided_labels: u64,
    maximum_refit_folds: u64,
    cpcv_split_count: u64,
    cpcv_test_group_count: u64,
    cpcv_max_rows: u64,
    insample_fraction: f64,
    stop_atr_multiplier: f64,
    reward_risk_ratio: f64,
    round_trip_cost_price: f64,
    cpcv_embargo_fraction: f64,
    cpcv_purge_fraction: f64,
    charged_peak_device_bytes: u64,
    full_discovery_reserve_bytes: u64,
    semantics_sha256: [u8; 32],
    state_family_semantics_sha256: [u8; 32],
    timeframe_group_semantics_sha256: [u8; 32],
    template_force_keep_semantics_sha256: [u8; 32],
    score_order_semantics_sha256: [u8; 32],
    plan_identity_sha256: [u8; 32],
    allocation_plan_sha256: [u8; 32],
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawResidentTrimPrefilterAllocationReceiptV1 {
    abi_version: u32,
    allocation_count: u32,
    long_labels_bytes: u64,
    short_labels_bytes: u64,
    label_census_bytes: u64,
    fold_descriptor_bytes: u64,
    column_score_bytes: u64,
    column_instability_bytes: u64,
    column_rankability_bytes: u64,
    state_template_timeframe_metadata_bytes: u64,
    radix_key_ping_pong_bytes: u64,
    radix_index_ping_pong_bytes: u64,
    timeframe_group_counter_bytes: u64,
    selected_column_map_bytes: u64,
    selected_column_count_bytes: u64,
    cub_select_scratch_bytes: u64,
    cub_radix_sort_scratch_bytes: u64,
    device_seal_bytes: u64,
    retained_device_bytes: u64,
    peak_device_bytes: u64,
    same_context_free_bytes: u64,
    full_discovery_reserve_bytes: u64,
    allocation_plan_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawResidentTrimPrefilterReadyEventV1 {
    abi_version: u32,
    reserved: u32,
    same_stream_enqueue_count: u64,
    intermediate_host_wait_count: u64,
    intermediate_readback_count: u64,
    host_to_device_transfer_count: u64,
    device_to_host_transfer_count: u64,
    explicit_synchronization_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawResidentTrimPrefilterViewsV1 {
    abi_version: u32,
    same_selected_column_map_for_holdout: u32,
    selected_compact_to_parent_columns_device: *const u32,
    selected_column_count_device: *const u64,
    device_seal: *const c_void,
    trim_prefilter_ready_event: *mut c_void,
    parent_row_count: u64,
    parent_column_count: u64,
    selection_row_start: u64,
    selection_row_end: u64,
    holdout_row_start: u64,
    holdout_row_end: u64,
    plan_identity_sha256: [u8; 32],
    view_semantics_sha256: [u8; 32],
    canonical_content_merkle_sha256: [u8; 32],
    ordered_feature_schema_sha256: [u8; 32],
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawResidentTrimPrefilterScoreBatchV2 {
    abi_version: u32,
    reserved: u32,
    batch_values_bar_major: *const f64,
    batch_validity_u4: *const u8,
    batch_row_count: u64,
    batch_column_count: u64,
    local_batch_stride: u64,
    global_parent_column_start: u64,
    global_parent_ordinals_device: *const u32,
    batch_ready_event: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawResidentTrimPrefilterSelectedMapReadV2 {
    abi_version: u32,
    reserved: u32,
    selected_capacity: u64,
    selected_count: u64,
    selected_map_readback_bytes: u64,
    selected_global_parent_ordinals_host: *mut u32,
    selected_map_readback_ready_event: *mut c_void,
}

const _: [(); 560] = [(); mem::size_of::<RawResidentTrimPrefilterImportV1>()];
const _: [(); 608] = [(); mem::size_of::<RawResidentTrimPrefilterPlanV1>()];
const _: [(); 200] = [(); mem::size_of::<RawResidentTrimPrefilterAllocationReceiptV1>()];
const _: [(); 56] = [(); mem::size_of::<RawResidentTrimPrefilterReadyEventV1>()];
const _: [(); 344] = [(); mem::size_of::<RawResidentTrimPrefilterViewsV1>()];
const _: [(); 72] = [(); mem::size_of::<RawResidentTrimPrefilterScoreBatchV2>()];
const _: [(); 48] = [(); mem::size_of::<RawResidentTrimPrefilterSelectedMapReadV2>()];
const _: [(); 0] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, abi_version)];
const _: [(); 4] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, reserved)];
const _: [(); 8] =
    [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, batch_values_bar_major)];
const _: [(); 16] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, batch_validity_u4)];
const _: [(); 24] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, batch_row_count)];
const _: [(); 32] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, batch_column_count)];
const _: [(); 40] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, local_batch_stride)];
const _: [(); 48] = [(); mem::offset_of!(
    RawResidentTrimPrefilterScoreBatchV2,
    global_parent_column_start
)];
const _: [(); 56] = [(); mem::offset_of!(
    RawResidentTrimPrefilterScoreBatchV2,
    global_parent_ordinals_device
)];
const _: [(); 64] = [(); mem::offset_of!(RawResidentTrimPrefilterScoreBatchV2, batch_ready_event)];
const _: [(); 0] = [(); mem::offset_of!(RawResidentTrimPrefilterSelectedMapReadV2, abi_version)];
const _: [(); 4] = [(); mem::offset_of!(RawResidentTrimPrefilterSelectedMapReadV2, reserved)];
const _: [(); 8] =
    [(); mem::offset_of!(RawResidentTrimPrefilterSelectedMapReadV2, selected_capacity)];
const _: [(); 16] =
    [(); mem::offset_of!(RawResidentTrimPrefilterSelectedMapReadV2, selected_count)];
const _: [(); 24] = [(); mem::offset_of!(
    RawResidentTrimPrefilterSelectedMapReadV2,
    selected_map_readback_bytes
)];
const _: [(); 32] = [(); mem::offset_of!(
    RawResidentTrimPrefilterSelectedMapReadV2,
    selected_global_parent_ordinals_host
)];
const _: [(); 40] = [(); mem::offset_of!(
    RawResidentTrimPrefilterSelectedMapReadV2,
    selected_map_readback_ready_event
)];

impl Default for RawResidentTrimPrefilterViewsV1 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            same_selected_column_map_for_holdout: 0,
            selected_compact_to_parent_columns_device: std::ptr::null(),
            selected_column_count_device: std::ptr::null(),
            device_seal: std::ptr::null(),
            trim_prefilter_ready_event: std::ptr::null_mut(),
            parent_row_count: 0,
            parent_column_count: 0,
            selection_row_start: 0,
            selection_row_end: 0,
            holdout_row_start: 0,
            holdout_row_end: 0,
            plan_identity_sha256: [0; 32],
            view_semantics_sha256: [0; 32],
            canonical_content_merkle_sha256: [0; 32],
            ordered_feature_schema_sha256: [0; 32],
            cuda_device_identity_sha256: [0; 32],
            primary_context_identity_sha256: [0; 32],
            run_stream_identity_sha256: [0; 32],
            cuda_build_manifest_sha256: [0; 32],
        }
    }
}

#[derive(Clone, Copy)]
struct ResidentTrimPrefilterExpectedViewsV1 {
    trim_prefilter_ready_event: NonNull<c_void>,
    parent_row_count: u64,
    parent_column_count: u64,
    selection_row_start: u64,
    selection_row_end: u64,
    holdout_row_start: u64,
    holdout_row_end: u64,
    plan_identity_sha256: [u8; 32],
    view_semantics_sha256: [u8; 32],
    canonical_content_merkle_sha256: [u8; 32],
    ordered_feature_schema_sha256: [u8; 32],
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
}

#[repr(C)]
struct NativeResidentTrimPrefilterRunV1 {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn query_resident_trim_prefilter_scratch_v1(
        admitted_run_stream: *mut c_void,
        selected_cuda_ordinal: u32,
        parent_column_count: u64,
        prefilter_active: u32,
        cub_select_scratch_bytes: *mut u64,
        cub_radix_sort_scratch_bytes: *mut u64,
    ) -> i32;
    fn query_resident_trim_prefilter_allocation_v1(
        import: *const RawResidentTrimPrefilterImportV1,
        plan: *const RawResidentTrimPrefilterPlanV1,
        receipt: *mut RawResidentTrimPrefilterAllocationReceiptV1,
    ) -> i32;
    fn create_resident_trim_prefilter_run_v1(
        import: *const RawResidentTrimPrefilterImportV1,
        plan: *const RawResidentTrimPrefilterPlanV1,
        receipt: *const RawResidentTrimPrefilterAllocationReceiptV1,
        run: *mut *mut NativeResidentTrimPrefilterRunV1,
    ) -> i32;
    fn enqueue_resident_trim_prefilter_stage_v1(
        run: *mut NativeResidentTrimPrefilterRunV1,
        stage: u32,
    ) -> i32;
    fn enqueue_resident_trim_prefilter_score_batch_v2(
        run: *mut NativeResidentTrimPrefilterRunV1,
        batch: *const RawResidentTrimPrefilterScoreBatchV2,
    ) -> i32;
    fn seal_resident_trim_prefilter_selected_map_v2(
        run: *mut NativeResidentTrimPrefilterRunV1,
    ) -> i32;
    fn read_resident_trim_prefilter_selected_map_v2(
        run: *mut NativeResidentTrimPrefilterRunV1,
        read: *mut RawResidentTrimPrefilterSelectedMapReadV2,
    ) -> i32;
    fn seal_resident_trim_prefilter_views_v1(
        run: *mut NativeResidentTrimPrefilterRunV1,
        views: *mut RawResidentTrimPrefilterViewsV1,
        ready: *mut RawResidentTrimPrefilterReadyEventV1,
    ) -> i32;
    fn enqueue_resident_trim_prefilter_release_v1(
        run: *mut NativeResidentTrimPrefilterRunV1,
    ) -> i32;
}

#[derive(Clone, Debug)]
pub struct ResidentTrimPrefilterNativePlanFieldsV1 {
    pub parent_row_count: u64,
    pub parent_column_count: u64,
    pub global_row_cap: u64,
    pub timeframe_row_cap: u64,
    pub outer_split_at: u64,
    pub selection_row_start: u64,
    pub selection_row_end: u64,
    pub holdout_row_start: u64,
    pub holdout_row_end: u64,
    pub configured_top_k: u64,
    pub resolved_top_k: u64,
    pub minimum_per_timeframe: u64,
    pub max_hold_bars: u64,
    pub atr_period: u32,
    pub insample_fraction: f64,
    pub stop_atr_multiplier: f64,
    pub reward_risk_ratio: f64,
    pub round_trip_cost_price: f64,
    pub cpcv_split_count: u64,
    pub cpcv_test_group_count: u64,
    pub cpcv_embargo_fraction: f64,
    pub cpcv_purge_fraction: f64,
    pub cpcv_max_rows: u64,
    pub semantics_sha256: [u8; 32],
    pub plan_identity_sha256: [u8; 32],
    pub cuda_device_identity_sha256: [u8; 32],
    pub primary_context_identity_sha256: [u8; 32],
    pub run_stream_identity_sha256: [u8; 32],
    pub cuda_build_manifest_sha256: [u8; 32],
    pub cuda_math_flags_sha256: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct ResidentTrimPrefilterNativeMemoryFieldsV1 {
    pub long_labels_bytes: u64,
    pub short_labels_bytes: u64,
    pub label_census_bytes: u64,
    pub fold_descriptor_bytes: u64,
    pub column_score_bytes: u64,
    pub column_instability_bytes: u64,
    pub column_rankability_bytes: u64,
    pub state_template_timeframe_metadata_bytes: u64,
    pub radix_key_ping_pong_bytes: u64,
    pub radix_index_ping_pong_bytes: u64,
    pub timeframe_group_counter_bytes: u64,
    pub selected_column_map_bytes: u64,
    pub selected_column_count_bytes: u64,
    pub cub_select_scratch_bytes: u64,
    pub cub_radix_sort_scratch_bytes: u64,
    pub device_seal_bytes: u64,
    pub retained_device_bytes: u64,
    pub peak_device_bytes: u64,
    pub full_discovery_reserve_bytes: u64,
    pub allocation_plan_sha256: [u8; 32],
}

pub trait ResidentTrimPrefilterSearchPlanV1 {
    fn resident_trim_prefilter_native_plan_fields_v1(
        &self,
    ) -> ResidentTrimPrefilterNativePlanFieldsV1;
}

pub trait ResidentTrimPrefilterSearchMemoryReceiptV1 {
    fn resident_trim_prefilter_native_memory_fields_v1(
        &self,
    ) -> ResidentTrimPrefilterNativeMemoryFieldsV1;
}

pub struct ResidentTrimPrefilterSemanticBindingsV1<'a> {
    pub state_family_semantics: &'a str,
    pub timeframe_group_semantics: &'a str,
    pub template_force_keep_semantics: &'a str,
    pub score_order_semantics: &'a str,
    pub minimum_pairwise_samples: u64,
    pub minimum_decided_labels: u64,
    pub maximum_refit_folds: u64,
}

#[derive(Clone, Debug)]
pub struct ResidentTrimPrefilterNativePlanV1 {
    raw: RawResidentTrimPrefilterPlanV1,
    expected_memory: ResidentTrimPrefilterNativeMemoryFieldsV1,
}

impl ResidentTrimPrefilterNativePlanV1 {
    pub fn from_search_authority<P, M>(
        plan: &P,
        memory: &M,
        bindings: ResidentTrimPrefilterSemanticBindingsV1<'_>,
    ) -> Result<Self, ResidentTrimPrefilterDeviceErrorV1>
    where
        P: ResidentTrimPrefilterSearchPlanV1,
        M: ResidentTrimPrefilterSearchMemoryReceiptV1,
    {
        let fields = plan.resident_trim_prefilter_native_plan_fields_v1();
        let expected_memory = memory.resident_trim_prefilter_native_memory_fields_v1();
        let selection_rows = fields
            .selection_row_end
            .checked_sub(fields.selection_row_start)
            .ok_or(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
                "selection row range",
            ))?;
        if fields.parent_row_count == 0
            || fields.parent_column_count == 0
            || fields.parent_column_count > MAX_GRID_X_V1
            || fields.selection_row_start >= fields.selection_row_end
            || selection_rows > MAX_GRID_X_V1 * LAUNCH_THREADS_V1
            || fields.selection_row_end != fields.outer_split_at
            || fields.holdout_row_start != fields.outer_split_at
            || fields.holdout_row_end != fields.parent_row_count
            || fields.atr_period != 14
            || fields.max_hold_bars == 0
            || bindings.minimum_pairwise_samples != 30
            || bindings.minimum_decided_labels != 100
            || bindings.maximum_refit_folds != 8
            || !fields.insample_fraction.is_finite()
            || fields.insample_fraction <= 0.0
            || fields.insample_fraction > 1.0
            || !fields.stop_atr_multiplier.is_finite()
            || fields.stop_atr_multiplier <= 0.0
            || !fields.reward_risk_ratio.is_finite()
            || fields.reward_risk_ratio <= 0.0
            || !fields.round_trip_cost_price.is_finite()
            || fields.round_trip_cost_price < 0.0
            || expected_memory.peak_device_bytes == 0
            || expected_memory.peak_device_bytes > expected_memory.full_discovery_reserve_bytes
        {
            return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
                "native trim/prefilter fields",
            ));
        }
        for (hash, field) in [
            (&fields.semantics_sha256, "semantics"),
            (&fields.plan_identity_sha256, "plan identity"),
            (&fields.cuda_device_identity_sha256, "device identity"),
            (&fields.primary_context_identity_sha256, "context identity"),
            (&fields.run_stream_identity_sha256, "stream identity"),
            (&fields.cuda_build_manifest_sha256, "build manifest"),
            (&fields.cuda_math_flags_sha256, "math flags"),
            (&expected_memory.allocation_plan_sha256, "allocation plan"),
        ] {
            require_hash_v1(hash, field)?;
        }
        let raw = RawResidentTrimPrefilterPlanV1 {
            abi_version: ABI_VERSION_V1,
            atr_period: fields.atr_period,
            parent_row_count: fields.parent_row_count,
            parent_column_count: fields.parent_column_count,
            global_row_cap: fields.global_row_cap,
            timeframe_row_cap: fields.timeframe_row_cap,
            outer_split_at: fields.outer_split_at,
            selection_row_start: fields.selection_row_start,
            selection_row_end: fields.selection_row_end,
            holdout_row_start: fields.holdout_row_start,
            holdout_row_end: fields.holdout_row_end,
            configured_top_k: fields.configured_top_k,
            resolved_top_k: fields.resolved_top_k,
            minimum_per_timeframe: fields.minimum_per_timeframe,
            max_hold_bars: fields.max_hold_bars,
            minimum_pairwise_samples: bindings.minimum_pairwise_samples,
            minimum_decided_labels: bindings.minimum_decided_labels,
            maximum_refit_folds: bindings.maximum_refit_folds,
            cpcv_split_count: fields.cpcv_split_count,
            cpcv_test_group_count: fields.cpcv_test_group_count,
            cpcv_max_rows: fields.cpcv_max_rows,
            insample_fraction: fields.insample_fraction,
            stop_atr_multiplier: fields.stop_atr_multiplier,
            reward_risk_ratio: fields.reward_risk_ratio,
            round_trip_cost_price: fields.round_trip_cost_price,
            cpcv_embargo_fraction: fields.cpcv_embargo_fraction,
            cpcv_purge_fraction: fields.cpcv_purge_fraction,
            charged_peak_device_bytes: expected_memory.peak_device_bytes,
            full_discovery_reserve_bytes: expected_memory.full_discovery_reserve_bytes,
            semantics_sha256: fields.semantics_sha256,
            state_family_semantics_sha256: sha256_v1(bindings.state_family_semantics.as_bytes()),
            timeframe_group_semantics_sha256: sha256_v1(
                bindings.timeframe_group_semantics.as_bytes(),
            ),
            template_force_keep_semantics_sha256: sha256_v1(
                bindings.template_force_keep_semantics.as_bytes(),
            ),
            score_order_semantics_sha256: sha256_v1(bindings.score_order_semantics.as_bytes()),
            plan_identity_sha256: fields.plan_identity_sha256,
            allocation_plan_sha256: expected_memory.allocation_plan_sha256,
            cuda_device_identity_sha256: fields.cuda_device_identity_sha256,
            primary_context_identity_sha256: fields.primary_context_identity_sha256,
            run_stream_identity_sha256: fields.run_stream_identity_sha256,
            cuda_build_manifest_sha256: fields.cuda_build_manifest_sha256,
            cuda_math_flags_sha256: fields.cuda_math_flags_sha256,
        };
        Ok(Self {
            raw,
            expected_memory,
        })
    }
}

fn sha256_v1(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn require_hash_v1(
    hash: &[u8; 32],
    field: &'static str,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    if *hash == [0; 32] {
        return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(field));
    }
    Ok(())
}

/// Opaque parent import. Its constructor stays gpu-cuda-private so Search can
/// only receive it by consuming the already-admitted resident session.
pub struct ResidentTrimPrefilterParentImportV1 {
    pub(crate) owner: Option<Box<dyn Any + Send>>,
    pub(crate) import_abi_version: u32,
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) parent_row_count: u64,
    pub(crate) parent_column_count: u64,
    pub(crate) packed_validity_bytes: u64,
    pub(crate) admitted_run_stream: NonNull<c_void>,
    pub(crate) parent_ready_event: NonNull<c_void>,
    pub(crate) indicators_bar_major: NonNull<f64>,
    pub(crate) indicators_validity_u4: NonNull<u8>,
    pub(crate) close: NonNull<f64>,
    pub(crate) high: NonNull<f64>,
    pub(crate) low: NonNull<f64>,
    pub(crate) canonical_search_input_receipt_sha256: [u8; 32],
    pub(crate) canonical_content_merkle_sha256: [u8; 32],
    pub(crate) normalization_fit_sha256: [u8; 32],
    pub(crate) feature_plan_sha256: [u8; 32],
    pub(crate) source_provenance_sha256: [u8; 32],
    pub(crate) cuda_device_identity_sha256: [u8; 32],
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
    pub(crate) cuda_math_flags_sha256: [u8; 32],
}

impl ResidentTrimPrefilterParentImportV1 {
    pub const fn selected_cuda_ordinal(&self) -> u32 {
        self.selected_cuda_ordinal
    }

    pub const fn parent_row_count(&self) -> u64 {
        self.parent_row_count
    }

    pub const fn parent_column_count(&self) -> u64 {
        self.parent_column_count
    }

    pub const fn canonical_content_merkle_sha256(&self) -> [u8; 32] {
        self.canonical_content_merkle_sha256
    }

    pub const fn primary_context_identity_sha256(&self) -> [u8; 32] {
        self.primary_context_identity_sha256
    }

    pub const fn run_stream_identity_sha256(&self) -> [u8; 32] {
        self.run_stream_identity_sha256
    }

    pub const fn cuda_build_manifest_sha256(&self) -> [u8; 32] {
        self.cuda_build_manifest_sha256
    }
}

/// Device-resident classification authority sealed during Data materialization.
pub struct SealedResidentColumnClassificationV1 {
    pub(crate) owner: Option<Box<dyn Any + Send>>,
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) parent_column_count: u64,
    pub(crate) retained_device_bytes: u64,
    pub(crate) timeframe_group_count: u64,
    pub(crate) schema_ready_event: NonNull<c_void>,
    pub(crate) column_class_flags_device: NonNull<u8>,
    pub(crate) timeframe_group_ids_device: NonNull<u32>,
    pub(crate) template_force_keep_flags_device: NonNull<u8>,
    pub(crate) ordered_feature_schema_sha256: [u8; 32],
    pub(crate) column_classification_content_sha256: [u8; 32],
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
}

impl SealedResidentColumnClassificationV1 {
    pub const fn selected_cuda_ordinal(&self) -> u32 {
        self.selected_cuda_ordinal
    }

    pub const fn parent_column_count(&self) -> u64 {
        self.parent_column_count
    }

    pub const fn retained_device_bytes(&self) -> u64 {
        self.retained_device_bytes
    }

    pub const fn timeframe_group_count(&self) -> u64 {
        self.timeframe_group_count
    }

    pub const fn column_class_flags_device(&self) -> bool {
        true
    }

    pub const fn timeframe_group_ids_device(&self) -> bool {
        true
    }

    pub const fn template_force_keep_flags_device(&self) -> bool {
        true
    }

    pub const fn ordered_feature_schema_sha256(&self) -> [u8; 32] {
        self.ordered_feature_schema_sha256
    }

    pub const fn column_classification_content_sha256(&self) -> [u8; 32] {
        self.column_classification_content_sha256
    }

    pub const fn primary_context_identity_sha256(&self) -> [u8; 32] {
        self.primary_context_identity_sha256
    }

    pub const fn run_stream_identity_sha256(&self) -> [u8; 32] {
        self.run_stream_identity_sha256
    }

    pub const fn cuda_build_manifest_sha256(&self) -> [u8; 32] {
        self.cuda_build_manifest_sha256
    }
}

/// Opaque slice of the already-sealed full-discovery workspace authority.
pub struct ResidentTrimPrefilterFullDiscoveryAdmissionV1 {
    pub(crate) owner: Option<Box<dyn Any + Send>>,
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) trim_prefilter_ready_event: NonNull<c_void>,
    pub(crate) trim_prefilter_reserved_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
}

impl ResidentTrimPrefilterFullDiscoveryAdmissionV1 {
    pub const fn selected_cuda_ordinal(&self) -> u32 {
        self.selected_cuda_ordinal
    }

    pub const fn trim_prefilter_reserved_bytes(&self) -> u64 {
        self.trim_prefilter_reserved_bytes
    }

    pub const fn full_discovery_reserve_bytes(&self) -> u64 {
        self.full_discovery_reserve_bytes
    }

    pub const fn primary_context_identity_sha256(&self) -> [u8; 32] {
        self.primary_context_identity_sha256
    }

    pub const fn run_stream_identity_sha256(&self) -> [u8; 32] {
        self.run_stream_identity_sha256
    }

    pub const fn cuda_build_manifest_sha256(&self) -> [u8; 32] {
        self.cuda_build_manifest_sha256
    }
}

/// Process-local identity receipt for the three one-shot trim inputs. It binds
/// only immutable hashes and the selected ordinal; raw CUDA representations
/// remain private to this crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentTrimPrefilterImportIdentityV1 {
    pub(crate) admission_identity_sha256: [u8; 32],
    pub(crate) workspace_plan_identity_sha256: [u8; 32],
    pub(crate) canonical_search_input_receipt_sha256: [u8; 32],
    pub(crate) canonical_content_merkle_sha256: [u8; 32],
    pub(crate) normalization_fit_sha256: [u8; 32],
    pub(crate) feature_plan_sha256: [u8; 32],
    pub(crate) source_provenance_sha256: [u8; 32],
    pub(crate) ordered_feature_schema_sha256: [u8; 32],
    pub(crate) column_classification_content_sha256: [u8; 32],
    pub(crate) selected_cuda_ordinal: u32,
    pub(crate) parent_row_count: u64,
    pub(crate) parent_column_count: u64,
    pub(crate) cuda_device_identity_sha256: [u8; 32],
    pub(crate) primary_context_identity_sha256: [u8; 32],
    pub(crate) run_stream_identity_sha256: [u8; 32],
    pub(crate) cuda_build_manifest_sha256: [u8; 32],
    pub(crate) cuda_math_flags_sha256: [u8; 32],
    pub(crate) phase_one_free_bytes_snapshot: u64,
    pub(crate) allocator_context_reserve_bytes: u64,
    pub(crate) required_workspace_bytes: u64,
    pub(crate) trim_prefilter_reserved_bytes: u64,
    pub(crate) full_discovery_reserve_bytes: u64,
}

impl ResidentTrimPrefilterImportIdentityV1 {
    pub const fn admission_identity_sha256(&self) -> [u8; 32] {
        self.admission_identity_sha256
    }

    pub const fn workspace_plan_identity_sha256(&self) -> [u8; 32] {
        self.workspace_plan_identity_sha256
    }

    pub const fn canonical_search_input_receipt_sha256(&self) -> [u8; 32] {
        self.canonical_search_input_receipt_sha256
    }

    pub const fn canonical_content_merkle_sha256(&self) -> [u8; 32] {
        self.canonical_content_merkle_sha256
    }

    pub const fn normalization_fit_sha256(&self) -> [u8; 32] {
        self.normalization_fit_sha256
    }

    pub const fn feature_plan_sha256(&self) -> [u8; 32] {
        self.feature_plan_sha256
    }

    pub const fn source_provenance_sha256(&self) -> [u8; 32] {
        self.source_provenance_sha256
    }

    pub const fn ordered_feature_schema_sha256(&self) -> [u8; 32] {
        self.ordered_feature_schema_sha256
    }

    pub const fn column_classification_content_sha256(&self) -> [u8; 32] {
        self.column_classification_content_sha256
    }

    pub const fn selected_cuda_ordinal(&self) -> u32 {
        self.selected_cuda_ordinal
    }

    pub const fn parent_row_count(&self) -> u64 {
        self.parent_row_count
    }

    pub const fn parent_column_count(&self) -> u64 {
        self.parent_column_count
    }

    pub const fn cuda_device_identity_sha256(&self) -> [u8; 32] {
        self.cuda_device_identity_sha256
    }

    pub const fn primary_context_identity_sha256(&self) -> [u8; 32] {
        self.primary_context_identity_sha256
    }

    pub const fn run_stream_identity_sha256(&self) -> [u8; 32] {
        self.run_stream_identity_sha256
    }

    pub const fn cuda_build_manifest_sha256(&self) -> [u8; 32] {
        self.cuda_build_manifest_sha256
    }

    pub const fn cuda_math_flags_sha256(&self) -> [u8; 32] {
        self.cuda_math_flags_sha256
    }

    pub const fn phase_one_free_bytes_snapshot(&self) -> u64 {
        self.phase_one_free_bytes_snapshot
    }

    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
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

/// Move-only result of consuming a sealed V3 feature-store import. The three
/// native inputs cannot be reconstructed independently or cloned.
#[must_use = "resident trim inputs must be consumed by the same admitted run"]
pub struct ResidentTrimPrefilterInputsV1 {
    pub(crate) parent_import: ResidentTrimPrefilterParentImportV1,
    pub(crate) sealed_schema: SealedResidentColumnClassificationV1,
    pub(crate) full_admission: ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    pub(crate) identity: ResidentTrimPrefilterImportIdentityV1,
}

impl ResidentTrimPrefilterInputsV1 {
    pub const fn identity(&self) -> &ResidentTrimPrefilterImportIdentityV1 {
        &self.identity
    }

    pub fn into_parts(
        self,
    ) -> (
        ResidentTrimPrefilterParentImportV1,
        SealedResidentColumnClassificationV1,
        ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    ) {
        (self.parent_import, self.sealed_schema, self.full_admission)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentTrimPrefilterNativeScratchBytesV1 {
    cub_select_scratch_bytes: u64,
    cub_radix_sort_scratch_bytes: u64,
}

impl ResidentTrimPrefilterNativeScratchBytesV1 {
    pub fn query_from_same_run(
        parent: &ResidentTrimPrefilterParentImportV1,
        parent_column_count: u64,
        prefilter_active: bool,
    ) -> Result<Self, ResidentTrimPrefilterDeviceErrorV1> {
        if parent.parent_column_count != parent_column_count || parent_column_count == 0 {
            return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
                "scratch-query column count",
            ));
        }
        Self::query_from_raw_admission_v2(
            parent.admitted_run_stream,
            parent.selected_cuda_ordinal,
            parent_column_count,
            prefilter_active,
        )
    }

    fn query_from_raw_admission_v2(
        admitted_run_stream: NonNull<c_void>,
        selected_cuda_ordinal: u32,
        parent_column_count: u64,
        prefilter_active: bool,
    ) -> Result<Self, ResidentTrimPrefilterDeviceErrorV1> {
        if parent_column_count == 0 {
            return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
                "scratch-query column count",
            ));
        }
        let mut cub_select_scratch_bytes = 0_u64;
        let mut cub_radix_sort_scratch_bytes = 0_u64;
        // SAFETY: the handle is retained by the move-only native admission;
        // the query allocates nothing, enqueues no work and synchronizes
        // nothing. The native side also verifies the selected ordinal.
        let status = unsafe {
            query_resident_trim_prefilter_scratch_v1(
                admitted_run_stream.as_ptr(),
                selected_cuda_ordinal,
                parent_column_count,
                u32::from(prefilter_active),
                &mut cub_select_scratch_bytes,
                &mut cub_radix_sort_scratch_bytes,
            )
        };
        require_native_ok_v1("query_resident_trim_prefilter_scratch_v1", status)?;
        Ok(Self {
            cub_select_scratch_bytes,
            cub_radix_sort_scratch_bytes,
        })
    }

    pub const fn cub_select_scratch_bytes(self) -> u64 {
        self.cub_select_scratch_bytes
    }

    pub const fn cub_radix_sort_scratch_bytes(self) -> u64 {
        self.cub_radix_sort_scratch_bytes
    }
}

/// Allocation-free inputs for the trim/prefilter device-memory preflight.
/// Search supplies only resolved semantic extents; gpu-cuda remains the one
/// authority for native scratch and byte arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentTrimPrefilterWorkspacePreflightRequestV2 {
    pub selection_row_count: u64,
    pub parent_column_count: u64,
    pub schema_metadata_bytes: u64,
    pub timeframe_group_count: u64,
    pub prefilter_active: bool,
}

/// Native trim bytes before they are bound to the complete screening-stage
/// reserve. This split removes the former circular dependency: native scratch
/// is queried first, the complete Data screening peak is then calculated, and
/// only that exact reserve can seal the native allocation receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnboundResidentTrimPrefilterWorkspacePreflightV2 {
    selection_row_count: u64,
    parent_column_count: u64,
    schema_metadata_bytes: u64,
    timeframe_group_count: u64,
    prefilter_active: bool,
    long_labels_bytes: u64,
    short_labels_bytes: u64,
    label_census_bytes: u64,
    fold_descriptor_bytes: u64,
    column_score_bytes: u64,
    column_instability_bytes: u64,
    column_rankability_bytes: u64,
    radix_key_ping_pong_bytes: u64,
    radix_index_ping_pong_bytes: u64,
    timeframe_group_counter_bytes: u64,
    selected_column_map_bytes: u64,
    selected_column_count_bytes: u64,
    cub_select_scratch_bytes: u64,
    cub_radix_sort_scratch_bytes: u64,
    device_seal_bytes: u64,
    retained_device_bytes: u64,
    peak_device_bytes: u64,
    unbound_preflight_identity_sha256: [u8; 32],
}

impl UnboundResidentTrimPrefilterWorkspacePreflightV2 {
    pub const fn parent_column_count(&self) -> u64 {
        self.parent_column_count
    }

    pub const fn schema_metadata_bytes(&self) -> u64 {
        self.schema_metadata_bytes
    }

    pub const fn peak_device_bytes(&self) -> u64 {
        self.peak_device_bytes
    }

    pub const fn retained_device_bytes(&self) -> u64 {
        self.retained_device_bytes
    }

    pub const fn unbound_preflight_identity_sha256(&self) -> [u8; 32] {
        self.unbound_preflight_identity_sha256
    }

    pub fn bind_screening_workspace_reserve_v2(
        self,
        screening_workspace_reserve_bytes: u64,
    ) -> Result<SealedResidentTrimPrefilterWorkspacePreflightV2, ResidentTrimPrefilterDeviceErrorV1>
    {
        if screening_workspace_reserve_bytes == 0
            || self.peak_device_bytes > screening_workspace_reserve_bytes
        {
            return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
                "screening workspace undercharges native trim/prefilter",
            ));
        }
        let allocation_plan_sha256 = trim_workspace_identity_v2(
            b"neoethos.resident-trim-prefilter-memory.v2",
            &[
                self.selection_row_count,
                self.parent_column_count,
                self.long_labels_bytes,
                self.short_labels_bytes,
                self.label_census_bytes,
                self.fold_descriptor_bytes,
                self.column_score_bytes,
                self.column_instability_bytes,
                self.column_rankability_bytes,
                self.schema_metadata_bytes,
                self.radix_key_ping_pong_bytes,
                self.radix_index_ping_pong_bytes,
                self.timeframe_group_counter_bytes,
                self.selected_column_map_bytes,
                self.selected_column_count_bytes,
                self.cub_select_scratch_bytes,
                self.cub_radix_sort_scratch_bytes,
                self.device_seal_bytes,
                self.retained_device_bytes,
                self.peak_device_bytes,
                screening_workspace_reserve_bytes,
            ],
        );
        Ok(SealedResidentTrimPrefilterWorkspacePreflightV2 {
            fields: ResidentTrimPrefilterNativeMemoryFieldsV1 {
                long_labels_bytes: self.long_labels_bytes,
                short_labels_bytes: self.short_labels_bytes,
                label_census_bytes: self.label_census_bytes,
                fold_descriptor_bytes: self.fold_descriptor_bytes,
                column_score_bytes: self.column_score_bytes,
                column_instability_bytes: self.column_instability_bytes,
                column_rankability_bytes: self.column_rankability_bytes,
                state_template_timeframe_metadata_bytes: self.schema_metadata_bytes,
                radix_key_ping_pong_bytes: self.radix_key_ping_pong_bytes,
                radix_index_ping_pong_bytes: self.radix_index_ping_pong_bytes,
                timeframe_group_counter_bytes: self.timeframe_group_counter_bytes,
                selected_column_map_bytes: self.selected_column_map_bytes,
                selected_column_count_bytes: self.selected_column_count_bytes,
                cub_select_scratch_bytes: self.cub_select_scratch_bytes,
                cub_radix_sort_scratch_bytes: self.cub_radix_sort_scratch_bytes,
                device_seal_bytes: self.device_seal_bytes,
                retained_device_bytes: self.retained_device_bytes,
                peak_device_bytes: self.peak_device_bytes,
                full_discovery_reserve_bytes: screening_workspace_reserve_bytes,
                allocation_plan_sha256,
            },
            unbound_preflight_identity_sha256: self.unbound_preflight_identity_sha256,
        })
    }
}

#[derive(Clone, Debug)]
pub struct SealedResidentTrimPrefilterWorkspacePreflightV2 {
    fields: ResidentTrimPrefilterNativeMemoryFieldsV1,
    unbound_preflight_identity_sha256: [u8; 32],
}

impl SealedResidentTrimPrefilterWorkspacePreflightV2 {
    pub const fn peak_device_bytes(&self) -> u64 {
        self.fields.peak_device_bytes
    }

    pub const fn screening_workspace_reserve_bytes(&self) -> u64 {
        self.fields.full_discovery_reserve_bytes
    }

    pub const fn allocation_plan_sha256(&self) -> [u8; 32] {
        self.fields.allocation_plan_sha256
    }

    pub const fn unbound_preflight_identity_sha256(&self) -> [u8; 32] {
        self.unbound_preflight_identity_sha256
    }
}

impl ResidentTrimPrefilterSearchMemoryReceiptV1
    for SealedResidentTrimPrefilterWorkspacePreflightV2
{
    fn resident_trim_prefilter_native_memory_fields_v1(
        &self,
    ) -> ResidentTrimPrefilterNativeMemoryFieldsV1 {
        self.fields.clone()
    }
}

pub fn preflight_resident_trim_prefilter_workspace_v2(
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
    request: ResidentTrimPrefilterWorkspacePreflightRequestV2,
) -> Result<UnboundResidentTrimPrefilterWorkspacePreflightV2, ResidentTrimPrefilterDeviceErrorV1> {
    if request.selection_row_count < 2
        || request.parent_column_count == 0
        || request.schema_metadata_bytes == 0
        || request.timeframe_group_count == 0
        || request.timeframe_group_count > request.parent_column_count
        || native_facts.selected_device_ordinal() == u32::MAX
        || native_facts.run_stream_handle_v2() == 0
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
            "trim/prefilter workspace preflight inputs",
        ));
    }
    let admitted_run_stream = NonNull::new(native_facts.run_stream_handle_v2() as *mut c_void)
        .ok_or(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
            "trim/prefilter admitted stream",
        ))?;
    let scratch = ResidentTrimPrefilterNativeScratchBytesV1::query_from_raw_admission_v2(
        admitted_run_stream,
        native_facts.selected_device_ordinal(),
        request.parent_column_count,
        request.prefilter_active,
    )?;
    let active_bytes = |unit: u64, field: &'static str| {
        if request.prefilter_active {
            request.parent_column_count.checked_mul(unit).ok_or(
                ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow(field),
            )
        } else {
            Ok(0)
        }
    };
    let long_labels_bytes = if request.prefilter_active {
        request.selection_row_count.checked_mul(8).ok_or(
            ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("long labels"),
        )?
    } else {
        0
    };
    let short_labels_bytes = long_labels_bytes;
    let label_census_bytes = if request.prefilter_active {
        LABEL_CENSUS_COUNTER_COUNT_V1 * 8
    } else {
        0
    };
    let fold_descriptor_bytes = if request.prefilter_active {
        MAXIMUM_REFIT_FOLDS_V1 * FOLD_DESCRIPTOR_BYTES_V1
    } else {
        0
    };
    let column_score_bytes = active_bytes(8, "column scores")?;
    let column_instability_bytes = column_score_bytes;
    let column_rankability_bytes = active_bytes(1, "column rankability")?;
    let radix_key_ping_pong_bytes = active_bytes(16, "radix key ping-pong")?;
    let radix_index_ping_pong_bytes = active_bytes(8, "radix index ping-pong")?;
    let timeframe_group_counter_bytes = if request.prefilter_active {
        request.timeframe_group_count.checked_mul(4).ok_or(
            ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("timeframe group counters"),
        )?
    } else {
        0
    };
    let selected_column_map_bytes = request.parent_column_count.checked_mul(4).ok_or(
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("selected-column map"),
    )?;
    let selected_column_count_bytes = 8;
    let cub_select_scratch_bytes = scratch.cub_select_scratch_bytes();
    let cub_radix_sort_scratch_bytes = scratch.cub_radix_sort_scratch_bytes();
    let retained_device_bytes = checked_sum_trim_workspace_v2(
        &[
            selected_column_map_bytes,
            selected_column_count_bytes,
            DEVICE_SEAL_BYTES_V1,
        ],
        "trim/prefilter retained bytes",
    )?;
    let peak_device_bytes = checked_sum_trim_workspace_v2(
        &[
            long_labels_bytes,
            short_labels_bytes,
            label_census_bytes,
            fold_descriptor_bytes,
            column_score_bytes,
            column_instability_bytes,
            column_rankability_bytes,
            radix_key_ping_pong_bytes,
            radix_index_ping_pong_bytes,
            timeframe_group_counter_bytes,
            selected_column_map_bytes,
            selected_column_count_bytes,
            cub_select_scratch_bytes,
            cub_radix_sort_scratch_bytes,
            DEVICE_SEAL_BYTES_V1,
        ],
        "trim/prefilter peak bytes",
    )?;
    let unbound_preflight_identity_sha256 = trim_workspace_identity_v2(
        b"neoethos.resident-trim-prefilter-unbound-preflight.v2",
        &[
            request.selection_row_count,
            request.parent_column_count,
            request.schema_metadata_bytes,
            request.timeframe_group_count,
            u64::from(request.prefilter_active),
            cub_select_scratch_bytes,
            cub_radix_sort_scratch_bytes,
            retained_device_bytes,
            peak_device_bytes,
        ],
    );
    Ok(UnboundResidentTrimPrefilterWorkspacePreflightV2 {
        selection_row_count: request.selection_row_count,
        parent_column_count: request.parent_column_count,
        schema_metadata_bytes: request.schema_metadata_bytes,
        timeframe_group_count: request.timeframe_group_count,
        prefilter_active: request.prefilter_active,
        long_labels_bytes,
        short_labels_bytes,
        label_census_bytes,
        fold_descriptor_bytes,
        column_score_bytes,
        column_instability_bytes,
        column_rankability_bytes,
        radix_key_ping_pong_bytes,
        radix_index_ping_pong_bytes,
        timeframe_group_counter_bytes,
        selected_column_map_bytes,
        selected_column_count_bytes,
        cub_select_scratch_bytes,
        cub_radix_sort_scratch_bytes,
        device_seal_bytes: DEVICE_SEAL_BYTES_V1,
        retained_device_bytes,
        peak_device_bytes,
        unbound_preflight_identity_sha256,
    })
}

fn checked_sum_trim_workspace_v2(
    values: &[u64],
    field: &'static str,
) -> Result<u64, ResidentTrimPrefilterDeviceErrorV1> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow(
                field,
            ))
    })
}

fn trim_workspace_identity_v2(domain: &[u8], values: &[u64]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentTrimPrefilterScoreBatchAddressV2 {
    local_offset: u64,
    global_parent_ordinal: u32,
}

impl ResidentTrimPrefilterScoreBatchAddressV2 {
    pub const fn local_offset(&self) -> u64 {
        self.local_offset
    }

    pub const fn global_parent_ordinal(&self) -> u32 {
        self.global_parent_ordinal
    }
}

pub fn checked_resident_trim_prefilter_score_batch_address_v2(
    row: u64,
    local_column: u64,
    batch_column_count: u64,
    local_batch_stride: u64,
    global_parent_column_start: u64,
    global_parent_ordinals: &[u32],
) -> Result<ResidentTrimPrefilterScoreBatchAddressV2, ResidentTrimPrefilterDeviceErrorV1> {
    if batch_column_count == 0
        || local_column >= batch_column_count
        || local_batch_stride < batch_column_count
        || usize::try_from(batch_column_count)
            .ok()
            .is_none_or(|column_count| global_parent_ordinals.len() < column_count)
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::InvalidPlan(
            "V2 score-batch shape",
        ));
    }
    let local_offset = row
        .checked_mul(local_batch_stride)
        .and_then(|base| base.checked_add(local_column))
        .ok_or(ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow(
            "V2 score-batch local offset",
        ))?;
    let _global_parent_column_end = global_parent_column_start
        .checked_add(batch_column_count)
        .ok_or(ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow(
            "V2 score-batch global extent",
        ))?;
    let local_column = usize::try_from(local_column).map_err(|_| {
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("V2 score-batch local column")
    })?;
    Ok(ResidentTrimPrefilterScoreBatchAddressV2 {
        local_offset,
        global_parent_ordinal: global_parent_ordinals[local_column],
    })
}

pub(crate) struct ResidentTrimPrefilterScoreBatchDeviceViewV2 {
    pub(crate) batch_values_bar_major: NonNull<f64>,
    pub(crate) batch_validity_u4: NonNull<u8>,
    pub(crate) batch_row_count: u64,
    pub(crate) batch_column_count: u64,
    pub(crate) local_batch_stride: u64,
    pub(crate) global_parent_column_start: u64,
    pub(crate) global_parent_ordinals_device: NonNull<u32>,
    pub(crate) batch_ready_event: NonNull<c_void>,
}

/// Opaque native owner. Every stage is enqueued on the imported run stream.
#[must_use = "resident trim/prefilter work must be consumed by the same GPU run"]
pub struct ResidentTrimPrefilterDeviceRunV1 {
    native: NonNull<NativeResidentTrimPrefilterRunV1>,
    parent_import: Option<ResidentTrimPrefilterParentImportV1>,
    sealed_schema: Option<SealedResidentColumnClassificationV1>,
    full_admission: Option<ResidentTrimPrefilterFullDiscoveryAdmissionV1>,
    state: ResidentTrimPrefilterRunStateV1,
    next_stage: u32,
    selected_cuda_ordinal: u32,
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    expected_views: ResidentTrimPrefilterExpectedViewsV1,
    same_stream_enqueue_count: u64,
    intermediate_host_wait_count: u64,
    intermediate_readback_count: u64,
    host_to_device_transfer_count: u64,
    device_to_host_transfer_count: u64,
    explicit_synchronization_count: u64,
}

impl ResidentTrimPrefilterDeviceRunV1 {
    pub const fn selected_cuda_ordinal(&self) -> u32 {
        self.selected_cuda_ordinal
    }

    pub const fn primary_context_identity_sha256(&self) -> [u8; 32] {
        self.primary_context_identity_sha256
    }

    pub const fn run_stream_identity_sha256(&self) -> [u8; 32] {
        self.run_stream_identity_sha256
    }

    pub const fn cuda_build_manifest_sha256(&self) -> [u8; 32] {
        self.cuda_build_manifest_sha256
    }

    pub const fn same_stream_enqueue_count(&self) -> u64 {
        self.same_stream_enqueue_count
    }

    pub const fn intermediate_host_wait_count(&self) -> u64 {
        self.intermediate_host_wait_count
    }

    pub const fn intermediate_readback_count(&self) -> u64 {
        self.intermediate_readback_count
    }

    pub const fn host_to_device_transfer_count(&self) -> u64 {
        self.host_to_device_transfer_count
    }

    pub const fn device_to_host_transfer_count(&self) -> u64 {
        self.device_to_host_transfer_count
    }

    pub const fn explicit_synchronization_count(&self) -> u64 {
        self.explicit_synchronization_count
    }
}

pub fn begin_resident_trim_prefilter_device_run_v1(
    mut parent_import: ResidentTrimPrefilterParentImportV1,
    mut sealed_schema: SealedResidentColumnClassificationV1,
    full_admission: ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    plan: ResidentTrimPrefilterNativePlanV1,
) -> Result<ResidentTrimPrefilterDeviceRunV1, ResidentTrimPrefilterDeviceErrorV1> {
    validate_one_shot_identities_v1(&parent_import, &sealed_schema, &full_admission, &plan)?;
    let raw_import = RawResidentTrimPrefilterImportV1 {
        abi_version: parent_import.import_abi_version,
        selected_cuda_ordinal: parent_import.selected_cuda_ordinal,
        parent_row_count: parent_import.parent_row_count,
        parent_column_count: parent_import.parent_column_count,
        packed_validity_bytes: parent_import.packed_validity_bytes,
        schema_metadata_bytes: sealed_schema.retained_device_bytes,
        timeframe_group_count: sealed_schema.timeframe_group_count,
        full_discovery_reserve_bytes: full_admission.full_discovery_reserve_bytes,
        trim_prefilter_reserved_bytes: full_admission.trim_prefilter_reserved_bytes,
        admitted_run_stream: parent_import.admitted_run_stream.as_ptr(),
        parent_ready_event: parent_import.parent_ready_event.as_ptr(),
        schema_ready_event: sealed_schema.schema_ready_event.as_ptr(),
        trim_prefilter_ready_event: full_admission.trim_prefilter_ready_event.as_ptr(),
        parent_lifetime_owner: parent_import
            .owner
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), |owner| {
                owner as *mut dyn Any as *mut c_void
            }),
        schema_lifetime_owner: sealed_schema
            .owner
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), |owner| {
                owner as *mut dyn Any as *mut c_void
            }),
        indicators_bar_major: parent_import.indicators_bar_major.as_ptr(),
        indicators_validity_u4: parent_import.indicators_validity_u4.as_ptr(),
        close: parent_import.close.as_ptr(),
        high: parent_import.high.as_ptr(),
        low: parent_import.low.as_ptr(),
        column_class_flags_device: sealed_schema.column_class_flags_device.as_ptr(),
        timeframe_group_ids_device: sealed_schema.timeframe_group_ids_device.as_ptr(),
        template_force_keep_flags_device: sealed_schema.template_force_keep_flags_device.as_ptr(),
        canonical_search_input_receipt_sha256: parent_import.canonical_search_input_receipt_sha256,
        canonical_content_merkle_sha256: parent_import.canonical_content_merkle_sha256,
        normalization_fit_sha256: parent_import.normalization_fit_sha256,
        feature_plan_sha256: parent_import.feature_plan_sha256,
        source_provenance_sha256: parent_import.source_provenance_sha256,
        ordered_feature_schema_sha256: sealed_schema.ordered_feature_schema_sha256,
        column_classification_content_sha256: sealed_schema.column_classification_content_sha256,
        cuda_device_identity_sha256: parent_import.cuda_device_identity_sha256,
        primary_context_identity_sha256: parent_import.primary_context_identity_sha256,
        run_stream_identity_sha256: parent_import.run_stream_identity_sha256,
        cuda_build_manifest_sha256: parent_import.cuda_build_manifest_sha256,
        cuda_math_flags_sha256: parent_import.cuda_math_flags_sha256,
    };
    if raw_import.parent_lifetime_owner.is_null() || raw_import.schema_lifetime_owner.is_null() {
        return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(
            "resident lifetime owner",
        ));
    }
    let mut receipt = RawResidentTrimPrefilterAllocationReceiptV1::default();
    // SAFETY: raw_import borrows the three move-only owners retained below.
    let query_status = unsafe {
        query_resident_trim_prefilter_allocation_v1(&raw_import, &plan.raw, &mut receipt)
    };
    require_native_ok_v1("query_resident_trim_prefilter_allocation_v1", query_status)?;
    validate_allocation_receipt_v1(&receipt, &plan.expected_memory)?;
    let mut native = std::ptr::null_mut();
    // SAFETY: native validates the same import, plan and exact receipt again
    // before allocating on the admitted stream.
    let create_status = unsafe {
        create_resident_trim_prefilter_run_v1(&raw_import, &plan.raw, &receipt, &mut native)
    };
    require_native_ok_v1("create_resident_trim_prefilter_run_v1", create_status)?;
    let native = NonNull::new(native).ok_or(ResidentTrimPrefilterDeviceErrorV1::Native {
        operation: "create_resident_trim_prefilter_run_v1",
        status: create_status,
    })?;
    let expected_views = ResidentTrimPrefilterExpectedViewsV1 {
        trim_prefilter_ready_event: full_admission.trim_prefilter_ready_event,
        parent_row_count: plan.raw.parent_row_count,
        parent_column_count: plan.raw.parent_column_count,
        selection_row_start: plan.raw.selection_row_start,
        selection_row_end: plan.raw.selection_row_end,
        holdout_row_start: plan.raw.holdout_row_start,
        holdout_row_end: plan.raw.holdout_row_end,
        plan_identity_sha256: plan.raw.plan_identity_sha256,
        view_semantics_sha256: plan.raw.semantics_sha256,
        canonical_content_merkle_sha256: raw_import.canonical_content_merkle_sha256,
        ordered_feature_schema_sha256: raw_import.ordered_feature_schema_sha256,
        cuda_device_identity_sha256: raw_import.cuda_device_identity_sha256,
        primary_context_identity_sha256: raw_import.primary_context_identity_sha256,
        run_stream_identity_sha256: raw_import.run_stream_identity_sha256,
        cuda_build_manifest_sha256: raw_import.cuda_build_manifest_sha256,
    };
    Ok(ResidentTrimPrefilterDeviceRunV1 {
        native,
        parent_import: Some(parent_import),
        sealed_schema: Some(sealed_schema),
        full_admission: Some(full_admission),
        state: ResidentTrimPrefilterRunStateV1::StrictIdle,
        next_stage: STAGE_LABELS_V1,
        selected_cuda_ordinal: raw_import.selected_cuda_ordinal,
        primary_context_identity_sha256: raw_import.primary_context_identity_sha256,
        run_stream_identity_sha256: raw_import.run_stream_identity_sha256,
        cuda_build_manifest_sha256: raw_import.cuda_build_manifest_sha256,
        expected_views,
        same_stream_enqueue_count: 0,
        intermediate_host_wait_count: 0,
        intermediate_readback_count: 0,
        host_to_device_transfer_count: 0,
        device_to_host_transfer_count: 0,
        explicit_synchronization_count: 0,
    })
}

fn enqueue_stage_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
    expected_stage: u32,
    operation: &'static str,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    if run.next_stage != expected_stage
        || matches!(
            run.state,
            ResidentTrimPrefilterRunStateV1::Sealed | ResidentTrimPrefilterRunStateV1::Poisoned
        )
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::RunStateViolation);
    }
    // SAFETY: the native pointer and all borrowed owners remain retained by run.
    let status =
        unsafe { enqueue_resident_trim_prefilter_stage_v1(run.native.as_ptr(), expected_stage) };
    run.state = if status == STATUS_OK_V1 {
        ResidentTrimPrefilterRunStateV1::InFlight
    } else {
        ResidentTrimPrefilterRunStateV1::Poisoned
    };
    if status != STATUS_OK_V1 {
        return Err(ResidentTrimPrefilterDeviceErrorV1::Native { operation, status });
    }
    run.next_stage = run.next_stage.checked_add(1).ok_or(
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("same-stream stage index"),
    )?;
    run.same_stream_enqueue_count = run.same_stream_enqueue_count.checked_add(1).ok_or(
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("same-stream enqueue count"),
    )?;
    Ok(())
}

pub fn enqueue_first_passage_labels_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_LABELS_V1, "enqueue first-passage labels")
}

pub fn enqueue_invalidate_device_seal_if_insufficient_decisions_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_LABEL_GUARD_V1, "enqueue label decision guard")
}

pub fn enqueue_exact_cpcv_fold_descriptors_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_FOLDS_V1, "enqueue CPCV fold descriptors")
}

pub fn enqueue_pairwise_two_pass_f64_correlations_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_CORRELATIONS_V1, "enqueue f64 correlations")
}

pub fn enqueue_stable_score_index_rank_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_RANK_V1, "enqueue stable score rank")
}

pub fn enqueue_state_template_timeframe_quota_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_QUOTAS_V1, "enqueue schema quotas")
}

pub fn enqueue_ascending_parent_column_map_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_ASCENDING_MAP_V1, "enqueue ascending column map")
}

pub fn enqueue_trim_prefilter_device_seal_v1(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    enqueue_stage_v1(run, STAGE_DEVICE_SEAL_V1, "enqueue device seal")
}

pub(crate) fn enqueue_score_batch_v2(
    run: &mut ResidentTrimPrefilterDeviceRunV1,
    batch: ResidentTrimPrefilterScoreBatchDeviceViewV2,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    if run.next_stage != STAGE_CORRELATIONS_V1
        || run.state != ResidentTrimPrefilterRunStateV1::InFlight
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::RunStateViolation);
    }
    let raw_batch = RawResidentTrimPrefilterScoreBatchV2 {
        abi_version: SCORE_BATCH_ABI_VERSION_V2,
        reserved: 0,
        batch_values_bar_major: batch.batch_values_bar_major.as_ptr(),
        batch_validity_u4: batch.batch_validity_u4.as_ptr(),
        batch_row_count: batch.batch_row_count,
        batch_column_count: batch.batch_column_count,
        local_batch_stride: batch.local_batch_stride,
        global_parent_column_start: batch.global_parent_column_start,
        global_parent_ordinals_device: batch.global_parent_ordinals_device.as_ptr(),
        batch_ready_event: batch.batch_ready_event.as_ptr(),
    };
    // SAFETY: the batch owner retains every device pointer and its ready event
    // until this same-stream enqueue has accepted the dependency.
    let status =
        unsafe { enqueue_resident_trim_prefilter_score_batch_v2(run.native.as_ptr(), &raw_batch) };
    if status != STATUS_OK_V1 {
        run.state = ResidentTrimPrefilterRunStateV1::Poisoned;
        return Err(ResidentTrimPrefilterDeviceErrorV1::Native {
            operation: "enqueue_resident_trim_prefilter_score_batch_v2",
            status,
        });
    }
    run.same_stream_enqueue_count = run.same_stream_enqueue_count.checked_add(1).ok_or(
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("V2 score-batch enqueue count"),
    )?;
    Ok(())
}

#[must_use = "the sealed selected map must be read exactly once before compact materialization"]
pub struct SealedResidentTrimPrefilterSelectedMapV2 {
    native: NonNull<NativeResidentTrimPrefilterRunV1>,
    parent_import: Option<ResidentTrimPrefilterParentImportV1>,
    sealed_schema: Option<SealedResidentColumnClassificationV1>,
    full_admission: Option<ResidentTrimPrefilterFullDiscoveryAdmissionV1>,
    selected_capacity: u64,
    parent_column_count: u64,
    selection_row_start: u64,
    selection_row_end: u64,
    holdout_row_start: u64,
    holdout_row_end: u64,
    plan_identity_sha256: [u8; 32],
    selected_map_readback_ready_event: NonNull<c_void>,
    armed: bool,
}

impl SealedResidentTrimPrefilterSelectedMapV2 {
    pub fn selected_capacity(&self) -> u64 {
        self.selected_capacity
    }
}

#[must_use = "the bounded selected map must size the compact resident store"]
pub struct BoundedResidentTrimPrefilterSelectedMapReadV2 {
    selected_capacity: u64,
    selected_count: u64,
    selected_map_sha256: [u8; 32],
    selected_map_readback_bytes: u64,
    selected_global_parent_ordinals: Vec<u32>,
}

impl BoundedResidentTrimPrefilterSelectedMapReadV2 {
    pub fn selected_capacity(&self) -> u64 {
        self.selected_capacity
    }

    pub fn selected_count(&self) -> u64 {
        self.selected_count
    }

    pub fn capacity_covers_actual_v2(&self) -> bool {
        self.selected_count <= self.selected_capacity
    }

    pub const fn selected_map_sha256(&self) -> [u8; 32] {
        self.selected_map_sha256
    }

    pub const fn selected_map_readback_bytes(&self) -> u64 {
        self.selected_map_readback_bytes
    }

    pub fn selected_global_parent_ordinals(&self) -> &[u32] {
        &self.selected_global_parent_ordinals
    }
}

pub fn seal_selected_map_v2(
    mut run: ResidentTrimPrefilterDeviceRunV1,
) -> Result<SealedResidentTrimPrefilterSelectedMapV2, ResidentTrimPrefilterDeviceErrorV1> {
    if run.next_stage != STAGE_CORRELATIONS_V1
        || run.state != ResidentTrimPrefilterRunStateV1::InFlight
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::RunStateViolation);
    }
    // SAFETY: the native run uniquely owns all screening buffers and retains
    // the imported parent/schema owners through the final same-stream seal.
    let status = unsafe { seal_resident_trim_prefilter_selected_map_v2(run.native.as_ptr()) };
    if status != STATUS_OK_V1 {
        run.state = ResidentTrimPrefilterRunStateV1::Poisoned;
        return Err(ResidentTrimPrefilterDeviceErrorV1::Native {
            operation: "seal_resident_trim_prefilter_selected_map_v2",
            status,
        });
    }
    run.state = ResidentTrimPrefilterRunStateV1::Sealed;
    let output = SealedResidentTrimPrefilterSelectedMapV2 {
        native: run.native,
        parent_import: run.parent_import.take(),
        sealed_schema: run.sealed_schema.take(),
        full_admission: run.full_admission.take(),
        selected_capacity: run.expected_views.parent_column_count,
        parent_column_count: run.expected_views.parent_column_count,
        selection_row_start: run.expected_views.selection_row_start,
        selection_row_end: run.expected_views.selection_row_end,
        holdout_row_start: run.expected_views.holdout_row_start,
        holdout_row_end: run.expected_views.holdout_row_end,
        plan_identity_sha256: run.expected_views.plan_identity_sha256,
        selected_map_readback_ready_event: run.expected_views.trim_prefilter_ready_event,
        armed: true,
    };
    mem::forget(run);
    Ok(output)
}

pub fn read_bounded_selected_map_v2(
    mut sealed: SealedResidentTrimPrefilterSelectedMapV2,
) -> Result<BoundedResidentTrimPrefilterSelectedMapReadV2, ResidentTrimPrefilterDeviceErrorV1> {
    let selected_capacity = sealed.selected_capacity;
    let selected_capacity_usize = usize::try_from(selected_capacity).map_err(|_| {
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("V2 selected-map capacity")
    })?;
    let initial_host_map = vec![0_u32; selected_capacity_usize];
    let mut selected_host = LockedBuffer::from_slice(&initial_host_map)
        .map_err(ResidentFeatureStoreCudaErrorV3::from)?;
    let mut raw_read = RawResidentTrimPrefilterSelectedMapReadV2 {
        abi_version: SELECTED_MAP_READ_ABI_VERSION_V2,
        reserved: 0,
        selected_capacity,
        selected_count: 0,
        selected_map_readback_bytes: 0,
        selected_global_parent_ordinals_host: selected_host.as_mut_ptr(),
        selected_map_readback_ready_event: sealed.selected_map_readback_ready_event.as_ptr(),
    };
    // SAFETY: the sealed owner retains the native run and event, while the
    // page-locked destination remains live through the bounded native wait.
    let status = unsafe {
        read_resident_trim_prefilter_selected_map_v2(sealed.native.as_ptr(), &mut raw_read)
    };
    if status != STATUS_OK_V1 {
        // An error may follow a queued D2H copy but precede confirmed event
        // completion. Retain its pinned destination, just as the armed sealed
        // owner retains the native run; freeing it here has no completion proof.
        mem::forget(selected_host);
        return Err(ResidentTrimPrefilterDeviceErrorV1::Native {
            operation: "read_resident_trim_prefilter_selected_map_v2",
            status,
        });
    }
    // The native bounded read completed its ready event, so release is now
    // ordered after the only permitted device-to-host control transfer.
    let release_status =
        unsafe { enqueue_resident_trim_prefilter_release_v1(sealed.native.as_ptr()) };
    require_native_ok_v1("enqueue_resident_trim_prefilter_release_v1", release_status)?;
    sealed.armed = false;

    let selected_count_usize = usize::try_from(raw_read.selected_count).map_err(|_| {
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("V2 selected-map count")
    })?;
    let expected_readback_bytes = raw_read.selected_count.checked_mul(4).ok_or(
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("V2 selected-map bytes"),
    )?;
    if raw_read.abi_version != SELECTED_MAP_READ_ABI_VERSION_V2
        || raw_read.selected_capacity != selected_capacity
        || raw_read.selected_count == 0
        || raw_read.selected_count > selected_capacity
        || raw_read.selected_map_readback_bytes != expected_readback_bytes
        || selected_count_usize > selected_host.len()
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(
            "bounded V2 selected-map receipt",
        ));
    }
    let selected_global_parent_ordinals = selected_host[..selected_count_usize].to_vec();
    if selected_global_parent_ordinals
        .iter()
        .any(|&parent| u64::from(parent) >= sealed.parent_column_count)
        || selected_global_parent_ordinals
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(
            "ascending V2 selected-map ordinals",
        ));
    }
    let mut selected_map_hasher = Sha256::new();
    selected_map_hasher.update(b"neoethos.resident-trim-prefilter-selected-map.v1\0");
    selected_map_hasher.update(sealed.plan_identity_sha256);
    selected_map_hasher.update(raw_read.selected_count.to_le_bytes());
    selected_map_hasher.update(sealed.selection_row_start.to_le_bytes());
    selected_map_hasher.update(sealed.selection_row_end.to_le_bytes());
    selected_map_hasher.update(sealed.holdout_row_start.to_le_bytes());
    selected_map_hasher.update(sealed.holdout_row_end.to_le_bytes());
    for parent in &selected_global_parent_ordinals {
        selected_map_hasher.update(parent.to_le_bytes());
    }
    let selected_map_sha256 = selected_map_hasher.finalize().into();
    Ok(BoundedResidentTrimPrefilterSelectedMapReadV2 {
        selected_capacity,
        selected_count: raw_read.selected_count,
        selected_map_sha256,
        selected_map_readback_bytes: raw_read.selected_map_readback_bytes,
        selected_global_parent_ordinals,
    })
}

/// Opaque device handoff. No selected count or pointer accessor is public.
#[must_use = "resident trim/prefilter views must be consumed by the next same-run stage"]
pub struct SealedResidentTrimPrefilterDeviceViewsV1 {
    native: NonNull<NativeResidentTrimPrefilterRunV1>,
    parent_import: Option<ResidentTrimPrefilterParentImportV1>,
    sealed_schema: Option<SealedResidentColumnClassificationV1>,
    full_admission: Option<ResidentTrimPrefilterFullDiscoveryAdmissionV1>,
    views: RawResidentTrimPrefilterViewsV1,
    ready: RawResidentTrimPrefilterReadyEventV1,
    artifact_class: ResidentTrimPrefilterArtifactClassV1,
    promotion_eligibility: ResidentTrimPrefilterPromotionEligibilityV1,
    armed: bool,
}

/// Move-only ownership carrier joining the native population session to the
/// exact compact-column map that selected it. It deliberately exposes neither
/// owner as an executable population API. Repeated Search instead consumes
/// the separately materialized compact Data owner.
#[must_use = "the trimmed population carrier retains in-flight GPU lifetimes"]
pub struct ResidentTrimmedPopulationSessionV1 {
    population_session: Option<ResidentPopulationSessionV3>,
    trim_native: NonNull<NativeResidentTrimPrefilterRunV1>,
    parent_import: Option<ResidentTrimPrefilterParentImportV1>,
    sealed_schema: Option<SealedResidentColumnClassificationV1>,
    full_admission: Option<ResidentTrimPrefilterFullDiscoveryAdmissionV1>,
    views: RawResidentTrimPrefilterViewsV1,
    ready: RawResidentTrimPrefilterReadyEventV1,
    armed: bool,
}

impl ResidentTrimmedPopulationSessionV1 {
    pub const fn selected_compact_to_parent_columns_device(&self) -> bool {
        !self
            .views
            .selected_compact_to_parent_columns_device
            .is_null()
    }

    pub const fn selected_column_count_device(&self) -> bool {
        !self.views.selected_column_count_device.is_null()
    }

    pub const fn same_selected_column_map_for_holdout(&self) -> bool {
        self.views.same_selected_column_map_for_holdout == 1
    }

    pub const fn same_stream_enqueue_count(&self) -> u64 {
        self.ready.same_stream_enqueue_count
    }

    pub const fn has_zero_trim_host_boundary(&self) -> bool {
        self.ready.intermediate_host_wait_count == 0
            && self.ready.intermediate_readback_count == 0
            && self.ready.host_to_device_transfer_count == 0
            && self.ready.device_to_host_transfer_count == 0
            && self.ready.explicit_synchronization_count == 0
    }

    pub fn population_rows(&self) -> usize {
        self.population_session
            .as_ref()
            .map_or(0, |session| session.rows())
    }

    pub fn parent_columns(&self) -> usize {
        self.population_session
            .as_ref()
            .map_or(0, |session| session.columns())
    }

    pub const fn plan_identity_sha256(&self) -> [u8; 32] {
        self.views.plan_identity_sha256
    }

    pub const fn canonical_content_merkle_sha256(&self) -> [u8; 32] {
        self.views.canonical_content_merkle_sha256
    }
}

impl SealedResidentTrimPrefilterDeviceViewsV1 {
    pub const fn selected_compact_to_parent_columns_device(&self) -> bool {
        !self
            .views
            .selected_compact_to_parent_columns_device
            .is_null()
    }

    pub const fn selected_column_count_device(&self) -> bool {
        !self.views.selected_column_count_device.is_null()
    }

    pub const fn same_selected_column_map_for_holdout(&self) -> bool {
        self.views.same_selected_column_map_for_holdout == 1
    }

    pub const fn same_stream_enqueue_count(&self) -> u64 {
        self.ready.same_stream_enqueue_count
    }

    pub const fn has_zero_intermediate_host_boundary(&self) -> bool {
        self.ready.intermediate_host_wait_count == 0
            && self.ready.intermediate_readback_count == 0
            && self.ready.host_to_device_transfer_count == 0
            && self.ready.device_to_host_transfer_count == 0
            && self.ready.explicit_synchronization_count == 0
    }

    pub const fn is_research_only(&self) -> bool {
        matches!(
            self.artifact_class,
            ResidentTrimPrefilterArtifactClassV1::ResearchOnly
        ) && matches!(
            self.promotion_eligibility,
            ResidentTrimPrefilterPromotionEligibilityV1::NotPromotionEligible
        )
    }

    /// Consume the sealed map and its three retained lifetimes into the one
    /// population owner created from the original V3 import. The compact map,
    /// selected-count scalar and trim-ready event remain private and retained;
    /// no selected result is read back. Population creation reuses the V3
    /// store's existing one-time Data-transient retirement boundary; it does
    /// not add a Search-generation host boundary.
    pub fn consume_into_population_session_v3(
        mut self,
    ) -> Result<ResidentTrimmedPopulationSessionV1, ResidentTrimPrefilterDeviceErrorV1> {
        if !self.armed
            || self
                .parent_import
                .as_ref()
                .is_none_or(|parent| parent.owner.is_none())
            || self.sealed_schema.is_none()
            || self.full_admission.is_none()
            || self
                .views
                .selected_compact_to_parent_columns_device
                .is_null()
            || self.views.selected_column_count_device.is_null()
            || self.views.trim_prefilter_ready_event.is_null()
        {
            return Err(ResidentTrimPrefilterDeviceErrorV1::RunStateViolation);
        }

        // These are the only ownership moves out of the sealed trim handoff.
        // Every fallible path below either returns the joined carrier or leaks
        // the still-in-flight owners fail-closed.
        let mut parent_import = self.parent_import.take().expect("sealed parent import");
        let sealed_schema = self.sealed_schema.take().expect("sealed schema owner");
        let full_admission = self.full_admission.take().expect("sealed admission owner");
        let resident_import_owner = parent_import
            .owner
            .take()
            .expect("sealed parent retains the typed V3 import");
        let resident_import = match resident_import_owner.downcast::<ResidentFeatureStoreImportV3>()
        {
            Ok(resident_import) => *resident_import,
            Err(owner) => {
                parent_import.owner = Some(owner);
                mem::forget(parent_import);
                mem::forget(sealed_schema);
                mem::forget(full_admission);
                return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(
                    "materialized resident parent import",
                ));
            }
        };
        let population_session = match resident_import.consume_into_population_session_v3() {
            Ok(session) => session,
            Err(error) => {
                mem::forget(parent_import);
                mem::forget(sealed_schema);
                mem::forget(full_admission);
                return Err(error.into());
            }
        };

        let output = ResidentTrimmedPopulationSessionV1 {
            population_session: Some(population_session),
            trim_native: self.native,
            parent_import: Some(parent_import),
            sealed_schema: Some(sealed_schema),
            full_admission: Some(full_admission),
            views: self.views,
            ready: self.ready,
            armed: true,
        };
        self.armed = false;
        mem::forget(self);
        Ok(output)
    }

    /// Queue release of this stage's owned buffers on the admitted stream.
    /// Borrowed parent/schema owners remain leaked until the future top-level
    /// run completion authority exists; this method never waits on host.
    pub fn enqueue_research_only_release_v1(
        mut self,
    ) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
        if !self.armed {
            return Err(ResidentTrimPrefilterDeviceErrorV1::RunStateViolation);
        }
        // SAFETY: native ownership is unique and its release is ordered after
        // the pre-owned ready event on the same admitted stream.
        let status = unsafe { enqueue_resident_trim_prefilter_release_v1(self.native.as_ptr()) };
        require_native_ok_v1("enqueue_resident_trim_prefilter_release_v1", status)?;
        self.armed = false;
        if let Some(owner) = self.parent_import.take() {
            mem::forget(owner);
        }
        if let Some(owner) = self.sealed_schema.take() {
            mem::forget(owner);
        }
        if let Some(owner) = self.full_admission.take() {
            mem::forget(owner);
        }
        Ok(())
    }
}

pub fn seal_resident_trim_prefilter_device_views_v1(
    mut run: ResidentTrimPrefilterDeviceRunV1,
) -> Result<SealedResidentTrimPrefilterDeviceViewsV1, ResidentTrimPrefilterDeviceErrorV1> {
    if run.next_stage != STAGE_DEVICE_SEAL_V1 + 1
        || run.state != ResidentTrimPrefilterRunStateV1::InFlight
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::RunStateViolation);
    }
    let expected_ready_enqueue_count = run.same_stream_enqueue_count.checked_add(1).ok_or(
        ResidentTrimPrefilterDeviceErrorV1::ArithmeticOverflow("sealed ready enqueue count"),
    )?;
    let mut views = RawResidentTrimPrefilterViewsV1::default();
    let mut ready = RawResidentTrimPrefilterReadyEventV1::default();
    // SAFETY: the native run and all owners are retained until the opaque
    // handoff is consumed by the next same-stream stage.
    let status = unsafe {
        seal_resident_trim_prefilter_views_v1(run.native.as_ptr(), &mut views, &mut ready)
    };
    if status != STATUS_OK_V1 {
        run.state = ResidentTrimPrefilterRunStateV1::Poisoned;
        return Err(ResidentTrimPrefilterDeviceErrorV1::Native {
            operation: "seal_resident_trim_prefilter_views_v1",
            status,
        });
    }
    if views.abi_version != ABI_VERSION_V1
        || views.same_selected_column_map_for_holdout != 1
        || views.selected_compact_to_parent_columns_device.is_null()
        || views.selected_column_count_device.is_null()
        || views.device_seal.is_null()
        || views.trim_prefilter_ready_event.is_null()
        || views.trim_prefilter_ready_event
            != run.expected_views.trim_prefilter_ready_event.as_ptr()
        || views.parent_row_count != run.expected_views.parent_row_count
        || views.parent_column_count != run.expected_views.parent_column_count
        || views.selection_row_start != run.expected_views.selection_row_start
        || views.selection_row_end != run.expected_views.selection_row_end
        || views.holdout_row_start != run.expected_views.holdout_row_start
        || views.holdout_row_end != run.expected_views.holdout_row_end
        || views.plan_identity_sha256 != run.expected_views.plan_identity_sha256
        || views.view_semantics_sha256 != run.expected_views.view_semantics_sha256
        || views.canonical_content_merkle_sha256
            != run.expected_views.canonical_content_merkle_sha256
        || views.ordered_feature_schema_sha256 != run.expected_views.ordered_feature_schema_sha256
        || views.cuda_device_identity_sha256 != run.expected_views.cuda_device_identity_sha256
        || views.primary_context_identity_sha256
            != run.expected_views.primary_context_identity_sha256
        || views.run_stream_identity_sha256 != run.expected_views.run_stream_identity_sha256
        || views.cuda_build_manifest_sha256 != run.expected_views.cuda_build_manifest_sha256
        || ready.abi_version != ABI_VERSION_V1
        || ready.same_stream_enqueue_count != expected_ready_enqueue_count
        || ready.intermediate_host_wait_count != 0
        || ready.intermediate_readback_count != 0
        || ready.host_to_device_transfer_count != 0
        || ready.device_to_host_transfer_count != 0
        || ready.explicit_synchronization_count != 0
    {
        run.state = ResidentTrimPrefilterRunStateV1::Poisoned;
        return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(
            "sealed resident views",
        ));
    }
    run.state = ResidentTrimPrefilterRunStateV1::Sealed;
    let output = SealedResidentTrimPrefilterDeviceViewsV1 {
        native: run.native,
        parent_import: run.parent_import.take(),
        sealed_schema: run.sealed_schema.take(),
        full_admission: run.full_admission.take(),
        views,
        ready,
        artifact_class: ResidentTrimPrefilterArtifactClassV1::ResearchOnly,
        promotion_eligibility: ResidentTrimPrefilterPromotionEligibilityV1::NotPromotionEligible,
        armed: true,
    };
    mem::forget(run);
    Ok(output)
}

fn validate_one_shot_identities_v1(
    parent: &ResidentTrimPrefilterParentImportV1,
    schema: &SealedResidentColumnClassificationV1,
    admission: &ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    plan: &ResidentTrimPrefilterNativePlanV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    if parent.owner.is_none()
        || !matches!(
            parent.import_abi_version,
            ABI_VERSION_V1 | SCREENING_IMPORT_ABI_VERSION_V2
        )
        || schema.owner.is_none()
        || admission.owner.is_none()
        || parent.selected_cuda_ordinal != schema.selected_cuda_ordinal
        || parent.selected_cuda_ordinal != admission.selected_cuda_ordinal
        || parent.parent_column_count != schema.parent_column_count
        || parent.primary_context_identity_sha256 != schema.primary_context_identity_sha256
        || parent.primary_context_identity_sha256 != admission.primary_context_identity_sha256
        || parent.run_stream_identity_sha256 != schema.run_stream_identity_sha256
        || parent.run_stream_identity_sha256 != admission.run_stream_identity_sha256
        || parent.cuda_build_manifest_sha256 != schema.cuda_build_manifest_sha256
        || parent.cuda_build_manifest_sha256 != admission.cuda_build_manifest_sha256
        || parent.parent_row_count != plan.raw.parent_row_count
        || parent.parent_column_count != plan.raw.parent_column_count
        || parent.cuda_device_identity_sha256 != plan.raw.cuda_device_identity_sha256
        || parent.primary_context_identity_sha256 != plan.raw.primary_context_identity_sha256
        || parent.run_stream_identity_sha256 != plan.raw.run_stream_identity_sha256
        || parent.cuda_build_manifest_sha256 != plan.raw.cuda_build_manifest_sha256
        || parent.cuda_math_flags_sha256 != plan.raw.cuda_math_flags_sha256
        || schema.retained_device_bytes
            != plan.expected_memory.state_template_timeframe_metadata_bytes
        || admission.trim_prefilter_reserved_bytes < plan.expected_memory.peak_device_bytes
        || admission.full_discovery_reserve_bytes
            != plan.expected_memory.full_discovery_reserve_bytes
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::IdentityMismatch(
            "one-shot parent/schema/admission",
        ));
    }
    Ok(())
}

fn validate_allocation_receipt_v1(
    actual: &RawResidentTrimPrefilterAllocationReceiptV1,
    expected: &ResidentTrimPrefilterNativeMemoryFieldsV1,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    if actual.abi_version != ABI_VERSION_V1
        || actual.long_labels_bytes != expected.long_labels_bytes
        || actual.short_labels_bytes != expected.short_labels_bytes
        || actual.label_census_bytes != expected.label_census_bytes
        || actual.fold_descriptor_bytes != expected.fold_descriptor_bytes
        || actual.column_score_bytes != expected.column_score_bytes
        || actual.column_instability_bytes != expected.column_instability_bytes
        || actual.column_rankability_bytes != expected.column_rankability_bytes
        || actual.state_template_timeframe_metadata_bytes
            != expected.state_template_timeframe_metadata_bytes
        || actual.radix_key_ping_pong_bytes != expected.radix_key_ping_pong_bytes
        || actual.radix_index_ping_pong_bytes != expected.radix_index_ping_pong_bytes
        || actual.timeframe_group_counter_bytes != expected.timeframe_group_counter_bytes
        || actual.selected_column_map_bytes != expected.selected_column_map_bytes
        || actual.selected_column_count_bytes != expected.selected_column_count_bytes
        || actual.cub_select_scratch_bytes != expected.cub_select_scratch_bytes
        || actual.cub_radix_sort_scratch_bytes != expected.cub_radix_sort_scratch_bytes
        || actual.device_seal_bytes != expected.device_seal_bytes
        || actual.retained_device_bytes != expected.retained_device_bytes
        || actual.peak_device_bytes != expected.peak_device_bytes
        || actual.full_discovery_reserve_bytes != expected.full_discovery_reserve_bytes
        || actual.allocation_plan_sha256 != expected.allocation_plan_sha256
        || actual.same_context_free_bytes < actual.peak_device_bytes
    {
        return Err(ResidentTrimPrefilterDeviceErrorV1::AllocationReceiptMismatch);
    }
    Ok(())
}

fn require_native_ok_v1(
    operation: &'static str,
    status: i32,
) -> Result<(), ResidentTrimPrefilterDeviceErrorV1> {
    if status != STATUS_OK_V1 {
        return Err(ResidentTrimPrefilterDeviceErrorV1::Native { operation, status });
    }
    Ok(())
}

fn leak_ambiguous_resident_trim_prefilter_run_v1(run: &mut ResidentTrimPrefilterDeviceRunV1) {
    if let Some(owner) = run.parent_import.take() {
        mem::forget(owner);
    }
    if let Some(owner) = run.sealed_schema.take() {
        mem::forget(owner);
    }
    if let Some(owner) = run.full_admission.take() {
        mem::forget(owner);
    }
}

impl Drop for ResidentTrimPrefilterDeviceRunV1 {
    fn drop(&mut self) {
        // No host wait is permitted to discover whether a launch happened.
        // Until a same-stream consumer exists, every armed state leaks rather
        // than freeing a live borrowed parent or creating an implicit sync.
        leak_ambiguous_resident_trim_prefilter_run_v1(self);
    }
}

impl Drop for SealedResidentTrimPrefilterSelectedMapV2 {
    fn drop(&mut self) {
        if self.armed {
            if let Some(owner) = self.parent_import.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.sealed_schema.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.full_admission.take() {
                mem::forget(owner);
            }
            // An abandoned or failed bounded read leaves native completion
            // ambiguous. Retain the run and borrowed owners rather than free
            // buffers that may still be referenced by queued CUDA work.
        }
    }
}

impl Drop for SealedResidentTrimPrefilterDeviceViewsV1 {
    fn drop(&mut self) {
        if self.armed {
            if let Some(owner) = self.parent_import.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.sealed_schema.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.full_admission.take() {
                mem::forget(owner);
            }
            // Native selected-map ownership is also deliberately retained.
            // A future same-stream consumer will disarm this handoff and call
            // `enqueue_resident_trim_prefilter_release_v1` only after its own
            // completion event has been recorded.
        }
    }
}

impl Drop for ResidentTrimmedPopulationSessionV1 {
    fn drop(&mut self) {
        if self.armed {
            if let Some(owner) = self.population_session.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.parent_import.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.sealed_schema.take() {
                mem::forget(owner);
            }
            if let Some(owner) = self.full_admission.take() {
                mem::forget(owner);
            }
            // `trim_native` owns the selected map and ready event dependency.
            // Until the next same-stream Search consumer exists, an abandoned
            // carrier deliberately leaks rather than freeing in-flight state.
            let _ = self.trim_native;
        }
    }
}
