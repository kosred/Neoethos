//! Opaque same-run CUDA row-trim, correlation-prefilter and view authority.
//!
//! This module is additive and deliberately unexported. Its one-shot inputs
//! can only be minted by the future resident-store/session bridge. No pointer,
//! event, selected count or selected-column list is exposed outside gpu-cuda.

use crate::resident_feature_store_v3::{
    ResidentFeatureStoreCudaErrorV3, ResidentFeatureStoreImportV3, ResidentPopulationSessionV3,
};
use sha2::{Digest, Sha256};
use std::any::Any;
use std::ffi::c_void;
use std::mem;
use std::ptr::NonNull;

const ABI_VERSION_V1: u32 = 1;
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

const _: [(); 560] = [(); mem::size_of::<RawResidentTrimPrefilterImportV1>()];
const _: [(); 608] = [(); mem::size_of::<RawResidentTrimPrefilterPlanV1>()];
const _: [(); 200] = [(); mem::size_of::<RawResidentTrimPrefilterAllocationReceiptV1>()];
const _: [(); 56] = [(); mem::size_of::<RawResidentTrimPrefilterReadyEventV1>()];
const _: [(); 344] = [(); mem::size_of::<RawResidentTrimPrefilterViewsV1>()];

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
            || fields.insample_fraction >= 1.0
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
    pub(crate) owner: Option<Box<ResidentFeatureStoreImportV3>>,
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
        let mut cub_select_scratch_bytes = 0_u64;
        let mut cub_radix_sort_scratch_bytes = 0_u64;
        // SAFETY: the opaque parent retains the exact admitted stream and
        // context. The query allocates nothing and performs no synchronization.
        let status = unsafe {
            query_resident_trim_prefilter_scratch_v1(
                parent.admitted_run_stream.as_ptr(),
                parent.selected_cuda_ordinal,
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
        abi_version: ABI_VERSION_V1,
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
                owner as *mut ResidentFeatureStoreImportV3 as *mut c_void
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
/// owner as an executable population API: the next resident Search slice must
/// consume both together and bind the device map before numerical evaluation.
#[must_use = "the trimmed population carrier must be consumed by resident Search"]
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
        let resident_import = *parent_import
            .owner
            .take()
            .expect("sealed parent retains the typed V3 import");
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
