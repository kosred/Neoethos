//! Strict-GPU row trim, correlation prefilter and resident-view authority.
//!
//! This additive V1 is feature-compiled as a public opaque boundary but remains
//! explicitly unwired. Its inputs cannot be minted outside this crate: the Data
//! resident store must first retain sealed, device-side column classifications,
//! and full-discovery admission must charge this stage's exact native scratch.
//! Device parity for selected parent indices is also still required before the
//! prepared NativeCuda route may use it.

use neoethos_gpu_cuda::resident_trim_prefilter_v1::{
    ResidentTrimPrefilterDeviceRunV1, ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    ResidentTrimPrefilterNativeMemoryFieldsV1, ResidentTrimPrefilterNativePlanFieldsV1,
    ResidentTrimPrefilterNativePlanV1, ResidentTrimPrefilterNativeScratchBytesV1,
    ResidentTrimPrefilterParentImportV1, ResidentTrimPrefilterSearchMemoryReceiptV1,
    ResidentTrimPrefilterSearchPlanV1, ResidentTrimPrefilterSemanticBindingsV1,
    ResidentTrimmedPopulationSessionV1, SealedResidentColumnClassificationV1,
    SealedResidentTrimPrefilterDeviceViewsV1, begin_resident_trim_prefilter_device_run_v1,
    enqueue_ascending_parent_column_map_v1, enqueue_exact_cpcv_fold_descriptors_v1,
    enqueue_first_passage_labels_v1, enqueue_invalidate_device_seal_if_insufficient_decisions_v1,
    enqueue_pairwise_two_pass_f64_correlations_v1, enqueue_stable_score_index_rank_v1,
    enqueue_state_template_timeframe_quota_v1, enqueue_trim_prefilter_device_seal_v1,
    seal_resident_trim_prefilter_device_views_v1,
};
use sha2::{Digest, Sha256};

pub(crate) const RESIDENT_TRIM_CORRELATION_PREFILTER_SEMANTICS_V1: &str = concat!(
    "neoethos.resident-trim-correlation-prefilter.v1;",
    "outer-split-floor-0.8;minimum-selection-64;suffix-trim-after-split;",
    "atr14-simple-finite-mean;directional-first-passage-cost-barriers;",
    "same-bar-dual-hit-ambiguous-zero;vertical-zero;undefined-nan;",
    "insufficient-decided-labels-invalidates-device-seal;",
    "cpcv-tail-cap-lexicographic-combinations-purge-embargo-max8;",
    "prefix-fit-floor-clamp-exclusive-last-row-when-cpcv-unavailable;",
    "pairwise-complete-two-pass-f64-ascending-row-min30;",
    "max-long-short-absolute-correlation;worst-fold-score;",
    "state-families-additive;timeframe-quota;template-force-keep;",
    "stable-descending-score-ascending-parent-index;",
    "selected-parent-indices-final-ascending;compact-gene-map-preserved;",
    "same-admitted-stream;zero-host-transfer-or-sync;",
    "strict-math-build-bound;research-only;not-promotion-eligible"
);

const PREFILTER_STATE_FAMILY_SEMANTICS_V1: &str =
    "base-only-prefixes:regime_,smc_,session_,fp_;additive-to-top-k";
const TIMEFRAME_GROUP_SEMANTICS_V1: &str =
    "head:M|H|D|W|MN-plus-digits;head-length-2-or-3;underscore-delimited";
const TEMPLATE_FORCE_KEEP_SEMANTICS_V1: &str =
    "seed-template-role-resolution-over-full-prefilter-schema-v1";
const SCORE_ORDER_SEMANTICS_V1: &str =
    "finite-nonnegative-f64-monotone-u64;descending;stable-parent-index-tie";

const DEFAULT_OOS_HOLDOUT_FRACTION_BITS_V1: u64 = 0.2_f64.to_bits();
const MINIMUM_IN_SAMPLE_ROWS_V1: u64 = 64;
const MINIMUM_PAIRWISE_SAMPLES_V1: u64 = 30;
const MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1: u64 = 100;
const MAXIMUM_REFIT_FOLDS_V1: u64 = 8;
const DEFAULT_ATR_PERIOD_V1: u64 = 14;
const F64_BYTES_V1: u64 = 8;
const U64_BYTES_V1: u64 = 8;
const U32_BYTES_V1: u64 = 4;
const U8_BYTES_V1: u64 = 1;
const LABEL_CENSUS_COUNTER_COUNT_V1: u64 = 12;
const FOLD_DESCRIPTOR_BYTES_V1: u64 = 80;
const DEVICE_SEAL_BYTES_V1: u64 = 88;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTrimPrefilterArtifactClassV1 {
    ResearchOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentTrimPrefilterPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentTrimPrefilterIntegrationStateV1 {
    Unwired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StrictMathFlagsV1 {
    fmad_disabled: bool,
    ftz_disabled: bool,
    precise_division: bool,
    precise_square_root: bool,
}

impl StrictMathFlagsV1 {
    const REQUIRED: Self = Self {
        fmad_disabled: true,
        ftz_disabled: true,
        precise_division: true,
        precise_square_root: true,
    };
}

#[derive(Debug)]
pub enum ResidentTrimPrefilterErrorV1 {
    InvalidParentShape(&'static str),
    InvalidResolvedPlan(&'static str),
    ArithmeticOverflow(&'static str),
    IdentityMismatch(&'static str),
    MissingFullDiscoveryWorkspaceAdmission,
    MissingResidentSchemaClassificationAuthority,
    MissingPopulationCompactColumnMapConsumer,
    MissingExactSelectedIndexDeviceParity,
    InsufficientDecidedFirstPassageLabels,
    SchemaClassificationIdentityMismatch,
    MemoryPlanArithmeticOverflow,
    FullDiscoveryAdmissionUndercharged,
    Device(neoethos_gpu_cuda::resident_trim_prefilter_v1::ResidentTrimPrefilterDeviceErrorV1),
}

impl From<neoethos_gpu_cuda::resident_trim_prefilter_v1::ResidentTrimPrefilterDeviceErrorV1>
    for ResidentTrimPrefilterErrorV1
{
    fn from(
        error: neoethos_gpu_cuda::resident_trim_prefilter_v1::ResidentTrimPrefilterDeviceErrorV1,
    ) -> Self {
        Self::Device(error)
    }
}

/// Immutable, config-derived semantics. Every numeric field is bound into the
/// plan identity before native allocation; no ambient settings are consulted.
#[derive(Clone, Debug)]
pub(crate) struct ResidentTrimCorrelationPrefilterSemanticsV1 {
    parent_row_count: u64,
    parent_column_count: u64,
    global_row_cap: u64,
    timeframe_row_cap: u64,
    configured_top_k: u64,
    resolved_top_k: u64,
    minimum_per_timeframe: u64,
    insample_fraction_bits: u64,
    max_hold_bars: u64,
    atr_period: u64,
    stop_atr_multiplier_bits: u64,
    reward_risk_ratio_bits: u64,
    round_trip_cost_price_bits: u64,
    cpcv_split_count: u64,
    cpcv_test_group_count: u64,
    cpcv_embargo_fraction_bits: u64,
    cpcv_purge_fraction_bits: u64,
    cpcv_max_rows: u64,
    semantics_sha256: [u8; 32],
    strict_math_flags: StrictMathFlagsV1,
}

/// Absolute parent-row ranges. No values are copied to construct these views.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentTrimPrefilterAbsoluteScopesV1 {
    parent_row_count: u64,
    outer_split_at: u64,
    selection_row_start: u64,
    selection_row_end: u64,
    holdout_row_start: u64,
    holdout_row_end: u64,
    retained_selection_rows: u64,
}

/// Exact, checked allocation evidence that the future full-run workspace must
/// charge before Data materialization begins.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResidentTrimPrefilterMemoryReceiptV1 {
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
    full_discovery_reserve_bytes: u64,
    allocation_plan_sha256: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct ResidentTrimPrefilterResolvedPlanV1 {
    semantics: ResidentTrimCorrelationPrefilterSemanticsV1,
    scopes: ResidentTrimPrefilterAbsoluteScopesV1,
    canonical_search_input_receipt_sha256: [u8; 32],
    canonical_content_merkle_sha256: [u8; 32],
    normalization_fit_sha256: [u8; 32],
    feature_plan_sha256: [u8; 32],
    source_provenance_sha256: [u8; 32],
    ordered_feature_schema_sha256: [u8; 32],
    column_classification_content_sha256: [u8; 32],
    selected_cuda_ordinal: u32,
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
    current_config_plan_identity_sha256: [u8; 32],
    plan_identity_sha256: [u8; 32],
}

/// Search owns only the opaque gpu-cuda run. The native pointer, resident
/// parent, schema metadata and pre-owned events never cross the crate boundary.
#[must_use = "resident trim/prefilter work must be sealed into same-run views"]
pub struct ResidentTrimPrefilterRunV1 {
    device_run: ResidentTrimPrefilterDeviceRunV1,
    resolved_plan: ResidentTrimPrefilterResolvedPlanV1,
    memory_receipt: ResidentTrimPrefilterMemoryReceiptV1,
    selected_cuda_ordinal: u32,
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
}

/// Opaque handoff to resident population generation and final holdout replay.
/// The selected count and map remain device data until the final bounded seal.
#[must_use = "resident trim/prefilter views must be consumed by the same GPU run"]
pub struct SealedResidentTrimPrefilterViewsV1 {
    device_views: SealedResidentTrimPrefilterDeviceViewsV1,
    resolved_plan: ResidentTrimPrefilterResolvedPlanV1,
    memory_receipt: ResidentTrimPrefilterMemoryReceiptV1,
    artifact_class: ResidentTrimPrefilterArtifactClassV1,
    promotion_eligibility: ResidentTrimPrefilterPromotionEligibilityV1,
}

impl SealedResidentTrimPrefilterViewsV1 {
    pub const fn is_research_only(&self) -> bool {
        matches!(
            self.artifact_class,
            ResidentTrimPrefilterArtifactClassV1::ResearchOnly
        ) && self.device_views.is_research_only()
    }

    pub const fn is_not_promotion_eligible(&self) -> bool {
        matches!(
            self.promotion_eligibility,
            ResidentTrimPrefilterPromotionEligibilityV1::NotPromotionEligible
        ) && self.device_views.is_research_only()
    }

    pub const fn plan_identity_sha256(&self) -> [u8; 32] {
        self.resolved_plan.plan_identity_sha256
    }

    pub const fn allocation_plan_sha256(&self) -> [u8; 32] {
        self.memory_receipt.allocation_plan_sha256
    }

    pub const fn has_zero_intermediate_host_boundary(&self) -> bool {
        self.device_views.has_zero_intermediate_host_boundary()
    }

    pub fn consume_into_population_session_v3(
        self,
    ) -> Result<ResidentTrimmedPopulationSessionV1, ResidentTrimPrefilterErrorV1> {
        let expected_plan_identity = self.resolved_plan.plan_identity_sha256;
        let expected_content_identity = self.resolved_plan.canonical_content_merkle_sha256;
        let population = self.device_views.consume_into_population_session_v3()?;
        if population.plan_identity_sha256() != expected_plan_identity
            || population.canonical_content_merkle_sha256() != expected_content_identity
            || !population.selected_compact_to_parent_columns_device()
            || !population.selected_column_count_device()
            || !population.same_selected_column_map_for_holdout()
            || !population.has_zero_trim_host_boundary()
        {
            return Err(ResidentTrimPrefilterErrorV1::IdentityMismatch(
                "trimmed population carrier",
            ));
        }
        Ok(population)
    }
}

impl ResidentTrimPrefilterSearchPlanV1 for ResidentTrimPrefilterResolvedPlanV1 {
    fn resident_trim_prefilter_native_plan_fields_v1(
        &self,
    ) -> ResidentTrimPrefilterNativePlanFieldsV1 {
        ResidentTrimPrefilterNativePlanFieldsV1 {
            parent_row_count: self.semantics.parent_row_count,
            parent_column_count: self.semantics.parent_column_count,
            global_row_cap: self.semantics.global_row_cap,
            timeframe_row_cap: self.semantics.timeframe_row_cap,
            outer_split_at: self.scopes.outer_split_at,
            selection_row_start: self.scopes.selection_row_start,
            selection_row_end: self.scopes.selection_row_end,
            holdout_row_start: self.scopes.holdout_row_start,
            holdout_row_end: self.scopes.holdout_row_end,
            configured_top_k: self.semantics.configured_top_k,
            resolved_top_k: self.semantics.resolved_top_k,
            minimum_per_timeframe: self.semantics.minimum_per_timeframe,
            max_hold_bars: self.semantics.max_hold_bars,
            atr_period: self.semantics.atr_period as u32,
            insample_fraction: f64::from_bits(self.semantics.insample_fraction_bits),
            stop_atr_multiplier: f64::from_bits(self.semantics.stop_atr_multiplier_bits),
            reward_risk_ratio: f64::from_bits(self.semantics.reward_risk_ratio_bits),
            round_trip_cost_price: f64::from_bits(self.semantics.round_trip_cost_price_bits),
            cpcv_split_count: self.semantics.cpcv_split_count,
            cpcv_test_group_count: self.semantics.cpcv_test_group_count,
            cpcv_embargo_fraction: f64::from_bits(self.semantics.cpcv_embargo_fraction_bits),
            cpcv_purge_fraction: f64::from_bits(self.semantics.cpcv_purge_fraction_bits),
            cpcv_max_rows: self.semantics.cpcv_max_rows,
            semantics_sha256: self.semantics.semantics_sha256,
            plan_identity_sha256: self.plan_identity_sha256,
            cuda_device_identity_sha256: self.cuda_device_identity_sha256,
            primary_context_identity_sha256: self.primary_context_identity_sha256,
            run_stream_identity_sha256: self.run_stream_identity_sha256,
            cuda_build_manifest_sha256: self.cuda_build_manifest_sha256,
            cuda_math_flags_sha256: self.cuda_math_flags_sha256,
        }
    }
}

impl ResidentTrimPrefilterSearchMemoryReceiptV1 for ResidentTrimPrefilterMemoryReceiptV1 {
    fn resident_trim_prefilter_native_memory_fields_v1(
        &self,
    ) -> ResidentTrimPrefilterNativeMemoryFieldsV1 {
        ResidentTrimPrefilterNativeMemoryFieldsV1 {
            long_labels_bytes: self.long_labels_bytes,
            short_labels_bytes: self.short_labels_bytes,
            label_census_bytes: self.label_census_bytes,
            fold_descriptor_bytes: self.fold_descriptor_bytes,
            column_score_bytes: self.column_score_bytes,
            column_instability_bytes: self.column_instability_bytes,
            column_rankability_bytes: self.column_rankability_bytes,
            state_template_timeframe_metadata_bytes: self.state_template_timeframe_metadata_bytes,
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
            full_discovery_reserve_bytes: self.full_discovery_reserve_bytes,
            allocation_plan_sha256: self.allocation_plan_sha256,
        }
    }
}

pub const fn resident_trim_prefilter_integration_state_v1()
-> ResidentTrimPrefilterIntegrationStateV1 {
    ResidentTrimPrefilterIntegrationStateV1::Unwired
}

fn checked_add_v1(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, ResidentTrimPrefilterErrorV1> {
    left.checked_add(right)
        .ok_or(ResidentTrimPrefilterErrorV1::ArithmeticOverflow(field))
}

fn checked_mul_v1(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, ResidentTrimPrefilterErrorV1> {
    left.checked_mul(right)
        .ok_or(ResidentTrimPrefilterErrorV1::ArithmeticOverflow(field))
}

fn checked_sum_v1(
    values: &[u64],
    field: &'static str,
) -> Result<u64, ResidentTrimPrefilterErrorV1> {
    values
        .iter()
        .try_fold(0_u64, |sum, value| checked_add_v1(sum, *value, field))
}

fn require_nonzero_hash_v1(
    hash: &[u8; 32],
    field: &'static str,
) -> Result<(), ResidentTrimPrefilterErrorV1> {
    if *hash == [0; 32] {
        return Err(ResidentTrimPrefilterErrorV1::IdentityMismatch(field));
    }
    Ok(())
}

fn sha256_v1(parts: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((*part).len().to_le_bytes());
        digest.update(part);
    }
    digest.finalize().into()
}

fn resolve_absolute_scopes_v1(
    parent_row_count: u64,
    global_row_cap: u64,
    timeframe_row_cap: u64,
) -> Result<ResidentTrimPrefilterAbsoluteScopesV1, ResidentTrimPrefilterErrorV1> {
    if parent_row_count == 0 {
        return Err(ResidentTrimPrefilterErrorV1::InvalidParentShape(
            "parent row count",
        ));
    }
    let outer_split_at = ((parent_row_count as f64)
        * (1.0 - f64::from_bits(DEFAULT_OOS_HOLDOUT_FRACTION_BITS_V1)))
    .floor() as u64;
    if outer_split_at < MINIMUM_IN_SAMPLE_ROWS_V1 || outer_split_at >= parent_row_count {
        return Err(ResidentTrimPrefilterErrorV1::InvalidResolvedPlan(
            "80/20 selection and holdout extents",
        ));
    }
    let row_cap = match (global_row_cap, timeframe_row_cap) {
        (0, 0) => 0,
        (0, timeframe) => timeframe,
        (global, 0) => global,
        (global, timeframe) => global.min(timeframe),
    };
    let retained_selection_rows = if row_cap > 0 && row_cap < outer_split_at {
        row_cap
    } else {
        outer_split_at
    };
    let selection_row_start = outer_split_at - retained_selection_rows;
    let selection_row_end = outer_split_at;
    let holdout_row_start = outer_split_at;
    let holdout_row_end = parent_row_count;
    Ok(ResidentTrimPrefilterAbsoluteScopesV1 {
        parent_row_count,
        outer_split_at,
        selection_row_start,
        selection_row_end,
        holdout_row_start,
        holdout_row_end,
        retained_selection_rows,
    })
}

fn checked_trim_prefilter_memory_plan_v1(
    selection_rows: u64,
    parent_columns: u64,
    schema_metadata_bytes: u64,
    timeframe_group_count: u64,
    prefilter_active: bool,
    scratch: ResidentTrimPrefilterNativeScratchBytesV1,
    full_discovery_reserve_bytes: u64,
) -> Result<ResidentTrimPrefilterMemoryReceiptV1, ResidentTrimPrefilterErrorV1> {
    if selection_rows < 2 || parent_columns == 0 || schema_metadata_bytes == 0 {
        return Err(ResidentTrimPrefilterErrorV1::MemoryPlanArithmeticOverflow);
    }
    let long_labels_bytes = if prefilter_active {
        checked_mul_v1(selection_rows, F64_BYTES_V1, "long labels")?
    } else {
        0
    };
    let short_labels_bytes = long_labels_bytes;
    let label_census_bytes = if prefilter_active {
        checked_mul_v1(LABEL_CENSUS_COUNTER_COUNT_V1, U64_BYTES_V1, "label census")?
    } else {
        0
    };
    let fold_descriptor_bytes = if prefilter_active {
        checked_mul_v1(
            MAXIMUM_REFIT_FOLDS_V1,
            FOLD_DESCRIPTOR_BYTES_V1,
            "fold descriptors",
        )?
    } else {
        0
    };
    let column_score_bytes = if prefilter_active {
        checked_mul_v1(parent_columns, F64_BYTES_V1, "column scores")?
    } else {
        0
    };
    let column_instability_bytes = column_score_bytes;
    let column_rankability_bytes = if prefilter_active {
        checked_mul_v1(parent_columns, U8_BYTES_V1, "column rankability")?
    } else {
        0
    };
    let state_template_timeframe_metadata_bytes = schema_metadata_bytes;
    let radix_key_ping_pong_bytes = if prefilter_active {
        checked_mul_v1(
            checked_mul_v1(parent_columns, U64_BYTES_V1, "radix keys")?,
            2,
            "radix key ping-pong",
        )?
    } else {
        0
    };
    let radix_index_ping_pong_bytes = if prefilter_active {
        checked_mul_v1(
            checked_mul_v1(parent_columns, U32_BYTES_V1, "radix indices")?,
            2,
            "radix index ping-pong",
        )?
    } else {
        0
    };
    let timeframe_group_counter_bytes = if prefilter_active {
        checked_mul_v1(
            timeframe_group_count,
            U32_BYTES_V1,
            "timeframe group counters",
        )?
    } else {
        0
    };
    let selected_column_map_bytes =
        checked_mul_v1(parent_columns, U32_BYTES_V1, "selected-column map")?;
    let selected_column_count_bytes = U64_BYTES_V1;
    let cub_select_scratch_bytes = scratch.cub_select_scratch_bytes();
    let cub_radix_sort_scratch_bytes = scratch.cub_radix_sort_scratch_bytes();
    let device_seal_bytes = DEVICE_SEAL_BYTES_V1;
    let retained_device_bytes = checked_sum_v1(
        &[
            selected_column_map_bytes,
            selected_column_count_bytes,
            device_seal_bytes,
        ],
        "resident trim/prefilter sealed handoff bytes",
    )?;
    let peak_device_bytes = checked_sum_v1(
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
            device_seal_bytes,
        ],
        "resident trim/prefilter peak owned bytes",
    )?;
    if peak_device_bytes > full_discovery_reserve_bytes {
        return Err(ResidentTrimPrefilterErrorV1::FullDiscoveryAdmissionUndercharged);
    }
    let allocation_plan_sha256 = sha256_v1(&[
        b"neoethos.resident-trim-prefilter-memory.v1\0",
        &selection_rows.to_le_bytes(),
        &parent_columns.to_le_bytes(),
        &long_labels_bytes.to_le_bytes(),
        &short_labels_bytes.to_le_bytes(),
        &label_census_bytes.to_le_bytes(),
        &fold_descriptor_bytes.to_le_bytes(),
        &column_score_bytes.to_le_bytes(),
        &column_instability_bytes.to_le_bytes(),
        &column_rankability_bytes.to_le_bytes(),
        &state_template_timeframe_metadata_bytes.to_le_bytes(),
        &radix_key_ping_pong_bytes.to_le_bytes(),
        &radix_index_ping_pong_bytes.to_le_bytes(),
        &timeframe_group_counter_bytes.to_le_bytes(),
        &selected_column_map_bytes.to_le_bytes(),
        &selected_column_count_bytes.to_le_bytes(),
        &cub_select_scratch_bytes.to_le_bytes(),
        &cub_radix_sort_scratch_bytes.to_le_bytes(),
        &device_seal_bytes.to_le_bytes(),
        &retained_device_bytes.to_le_bytes(),
        &peak_device_bytes.to_le_bytes(),
        &full_discovery_reserve_bytes.to_le_bytes(),
    ]);
    Ok(ResidentTrimPrefilterMemoryReceiptV1 {
        long_labels_bytes,
        short_labels_bytes,
        label_census_bytes,
        fold_descriptor_bytes,
        column_score_bytes,
        column_instability_bytes,
        column_rankability_bytes,
        state_template_timeframe_metadata_bytes,
        radix_key_ping_pong_bytes,
        radix_index_ping_pong_bytes,
        timeframe_group_counter_bytes,
        selected_column_map_bytes,
        selected_column_count_bytes,
        cub_select_scratch_bytes,
        cub_radix_sort_scratch_bytes,
        device_seal_bytes,
        retained_device_bytes,
        peak_device_bytes,
        full_discovery_reserve_bytes,
        allocation_plan_sha256,
    })
}

fn validate_against_full_discovery_admission_v1(
    admission: &ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    plan: &ResidentTrimPrefilterResolvedPlanV1,
    memory: &ResidentTrimPrefilterMemoryReceiptV1,
) -> Result<(), ResidentTrimPrefilterErrorV1> {
    if admission.selected_cuda_ordinal() != plan.selected_cuda_ordinal
        || admission.primary_context_identity_sha256() != plan.primary_context_identity_sha256
        || admission.run_stream_identity_sha256() != plan.run_stream_identity_sha256
        || admission.cuda_build_manifest_sha256() != plan.cuda_build_manifest_sha256
        || admission.trim_prefilter_reserved_bytes() < memory.peak_device_bytes
        || admission.full_discovery_reserve_bytes() != memory.full_discovery_reserve_bytes
    {
        return Err(ResidentTrimPrefilterErrorV1::FullDiscoveryAdmissionUndercharged);
    }
    Ok(())
}

fn consume_same_run_parent_and_schema_v1(
    parent: ResidentTrimPrefilterParentImportV1,
    schema: SealedResidentColumnClassificationV1,
    plan: &ResidentTrimPrefilterResolvedPlanV1,
) -> Result<
    (
        ResidentTrimPrefilterParentImportV1,
        SealedResidentColumnClassificationV1,
    ),
    ResidentTrimPrefilterErrorV1,
> {
    if parent.selected_cuda_ordinal() != plan.selected_cuda_ordinal
        || parent.primary_context_identity_sha256() != plan.primary_context_identity_sha256
        || parent.run_stream_identity_sha256() != plan.run_stream_identity_sha256
        || parent.cuda_build_manifest_sha256() != plan.cuda_build_manifest_sha256
        || parent.canonical_content_merkle_sha256() != plan.canonical_content_merkle_sha256
        || parent.parent_row_count() != plan.semantics.parent_row_count
        || parent.parent_column_count() != plan.semantics.parent_column_count
    {
        return Err(ResidentTrimPrefilterErrorV1::IdentityMismatch(
            "resident parent import",
        ));
    }
    if schema.selected_cuda_ordinal() != plan.selected_cuda_ordinal
        || schema.primary_context_identity_sha256() != plan.primary_context_identity_sha256
        || schema.run_stream_identity_sha256() != plan.run_stream_identity_sha256
        || schema.cuda_build_manifest_sha256() != plan.cuda_build_manifest_sha256
        || schema.parent_column_count() != plan.semantics.parent_column_count
        || schema.ordered_feature_schema_sha256() != plan.ordered_feature_schema_sha256
        || schema.column_classification_content_sha256()
            != plan.column_classification_content_sha256
        || !schema.column_class_flags_device()
        || !schema.timeframe_group_ids_device()
        || !schema.template_force_keep_flags_device()
    {
        return Err(ResidentTrimPrefilterErrorV1::SchemaClassificationIdentityMismatch);
    }
    Ok((parent, schema))
}

pub fn begin_gpu_resident_trim_prefilter_view_v1(
    parent: ResidentTrimPrefilterParentImportV1,
    sealed_schema: SealedResidentColumnClassificationV1,
    admission: ResidentTrimPrefilterFullDiscoveryAdmissionV1,
    mut resolved_plan: ResidentTrimPrefilterResolvedPlanV1,
) -> Result<ResidentTrimPrefilterRunV1, ResidentTrimPrefilterErrorV1> {
    for (hash, field) in [
        (
            &resolved_plan.canonical_search_input_receipt_sha256,
            "canonical Search input receipt",
        ),
        (
            &resolved_plan.canonical_content_merkle_sha256,
            "canonical content Merkle root",
        ),
        (&resolved_plan.normalization_fit_sha256, "normalization fit"),
        (&resolved_plan.feature_plan_sha256, "feature plan"),
        (&resolved_plan.source_provenance_sha256, "source provenance"),
        (
            &resolved_plan.ordered_feature_schema_sha256,
            "ordered feature schema",
        ),
        (
            &resolved_plan.column_classification_content_sha256,
            "column classification",
        ),
        (
            &resolved_plan.cuda_device_identity_sha256,
            "CUDA device identity",
        ),
        (
            &resolved_plan.primary_context_identity_sha256,
            "primary context identity",
        ),
        (
            &resolved_plan.run_stream_identity_sha256,
            "run stream identity",
        ),
        (
            &resolved_plan.cuda_build_manifest_sha256,
            "CUDA build manifest",
        ),
        (&resolved_plan.cuda_math_flags_sha256, "CUDA math flags"),
        (
            &resolved_plan.current_config_plan_identity_sha256,
            "current-config Search plan",
        ),
    ] {
        require_nonzero_hash_v1(hash, field)?;
    }
    if resolved_plan.semantics.atr_period != DEFAULT_ATR_PERIOD_V1
        || resolved_plan.semantics.semantics_sha256
            != sha256_v1(&[RESIDENT_TRIM_CORRELATION_PREFILTER_SEMANTICS_V1.as_bytes()])
        || resolved_plan.semantics.strict_math_flags != StrictMathFlagsV1::REQUIRED
    {
        return Err(ResidentTrimPrefilterErrorV1::InvalidResolvedPlan(
            "resident trim/prefilter semantics",
        ));
    }
    resolved_plan.scopes = resolve_absolute_scopes_v1(
        resolved_plan.semantics.parent_row_count,
        resolved_plan.semantics.global_row_cap,
        resolved_plan.semantics.timeframe_row_cap,
    )?;
    let (parent, sealed_schema) =
        consume_same_run_parent_and_schema_v1(parent, sealed_schema, &resolved_plan)?;
    let prefilter_active = resolved_plan.semantics.resolved_top_k > 0
        && resolved_plan.semantics.parent_column_count > resolved_plan.semantics.resolved_top_k;
    let scratch = ResidentTrimPrefilterNativeScratchBytesV1::query_from_same_run(
        &parent,
        resolved_plan.semantics.parent_column_count,
        prefilter_active,
    )?;
    let memory_receipt = checked_trim_prefilter_memory_plan_v1(
        resolved_plan.scopes.retained_selection_rows,
        resolved_plan.semantics.parent_column_count,
        sealed_schema.retained_device_bytes(),
        sealed_schema.timeframe_group_count(),
        prefilter_active,
        scratch,
        admission.full_discovery_reserve_bytes(),
    )?;
    validate_against_full_discovery_admission_v1(&admission, &resolved_plan, &memory_receipt)?;
    resolved_plan.plan_identity_sha256 = compute_resolved_plan_identity_v1(&resolved_plan);
    let native_plan = ResidentTrimPrefilterNativePlanV1::from_search_authority(
        &resolved_plan,
        &memory_receipt,
        ResidentTrimPrefilterSemanticBindingsV1 {
            state_family_semantics: PREFILTER_STATE_FAMILY_SEMANTICS_V1,
            timeframe_group_semantics: TIMEFRAME_GROUP_SEMANTICS_V1,
            template_force_keep_semantics: TEMPLATE_FORCE_KEEP_SEMANTICS_V1,
            score_order_semantics: SCORE_ORDER_SEMANTICS_V1,
            minimum_pairwise_samples: MINIMUM_PAIRWISE_SAMPLES_V1,
            minimum_decided_labels: MINIMUM_DECIDED_FIRST_PASSAGE_LABELS_V1,
            maximum_refit_folds: MAXIMUM_REFIT_FOLDS_V1,
        },
    )?;
    let selected_cuda_ordinal = resolved_plan.selected_cuda_ordinal;
    let primary_context_identity_sha256 = resolved_plan.primary_context_identity_sha256;
    let run_stream_identity_sha256 = resolved_plan.run_stream_identity_sha256;
    let cuda_build_manifest_sha256 = resolved_plan.cuda_build_manifest_sha256;
    let device_run =
        begin_resident_trim_prefilter_device_run_v1(parent, sealed_schema, admission, native_plan)?;
    Ok(ResidentTrimPrefilterRunV1 {
        device_run,
        resolved_plan,
        memory_receipt,
        selected_cuda_ordinal,
        primary_context_identity_sha256,
        run_stream_identity_sha256,
        cuda_build_manifest_sha256,
    })
}

pub fn execute_gpu_resident_trim_prefilter_view_v1(
    mut run: ResidentTrimPrefilterRunV1,
) -> Result<ResidentTrimPrefilterRunV1, ResidentTrimPrefilterErrorV1> {
    enqueue_first_passage_labels_v1(&mut run.device_run)?;
    enqueue_invalidate_device_seal_if_insufficient_decisions_v1(&mut run.device_run)?;
    enqueue_exact_cpcv_fold_descriptors_v1(&mut run.device_run)?;
    enqueue_pairwise_two_pass_f64_correlations_v1(&mut run.device_run)?;
    enqueue_stable_score_index_rank_v1(&mut run.device_run)?;
    enqueue_state_template_timeframe_quota_v1(&mut run.device_run)?;
    enqueue_ascending_parent_column_map_v1(&mut run.device_run)?;
    enqueue_trim_prefilter_device_seal_v1(&mut run.device_run)?;
    Ok(run)
}

pub fn seal_gpu_resident_trim_prefilter_view_v1(
    run: ResidentTrimPrefilterRunV1,
) -> Result<SealedResidentTrimPrefilterViewsV1, ResidentTrimPrefilterErrorV1> {
    if run.device_run.same_stream_enqueue_count() == 0
        || run.device_run.intermediate_host_wait_count() != 0
        || run.device_run.intermediate_readback_count() != 0
        || run.device_run.host_to_device_transfer_count() != 0
        || run.device_run.device_to_host_transfer_count() != 0
        || run.device_run.explicit_synchronization_count() != 0
    {
        return Err(ResidentTrimPrefilterErrorV1::IdentityMismatch(
            "resident transfer accounting",
        ));
    }
    if run.device_run.selected_cuda_ordinal() != run.selected_cuda_ordinal
        || run.device_run.primary_context_identity_sha256() != run.primary_context_identity_sha256
        || run.device_run.run_stream_identity_sha256() != run.run_stream_identity_sha256
        || run.device_run.cuda_build_manifest_sha256() != run.cuda_build_manifest_sha256
    {
        return Err(ResidentTrimPrefilterErrorV1::IdentityMismatch(
            "resident route at seal",
        ));
    }
    let device_views = seal_resident_trim_prefilter_device_views_v1(run.device_run)?;
    if !device_views.selected_compact_to_parent_columns_device()
        || !device_views.selected_column_count_device()
        || !device_views.same_selected_column_map_for_holdout()
        || device_views.same_stream_enqueue_count() == 0
        || !device_views.has_zero_intermediate_host_boundary()
        || !device_views.is_research_only()
    {
        return Err(ResidentTrimPrefilterErrorV1::IdentityMismatch(
            "resident compact-to-parent column map",
        ));
    }
    Ok(SealedResidentTrimPrefilterViewsV1 {
        device_views,
        resolved_plan: run.resolved_plan,
        memory_receipt: run.memory_receipt,
        artifact_class: ResidentTrimPrefilterArtifactClassV1::ResearchOnly,
        promotion_eligibility: ResidentTrimPrefilterPromotionEligibilityV1::NotPromotionEligible,
    })
}

fn compute_resolved_plan_identity_v1(plan: &ResidentTrimPrefilterResolvedPlanV1) -> [u8; 32] {
    sha256_v1(&[
        b"neoethos.search.resident-trim-prefilter-plan.v1\0",
        RESIDENT_TRIM_CORRELATION_PREFILTER_SEMANTICS_V1.as_bytes(),
        PREFILTER_STATE_FAMILY_SEMANTICS_V1.as_bytes(),
        TIMEFRAME_GROUP_SEMANTICS_V1.as_bytes(),
        TEMPLATE_FORCE_KEEP_SEMANTICS_V1.as_bytes(),
        SCORE_ORDER_SEMANTICS_V1.as_bytes(),
        &plan.semantics.parent_row_count.to_le_bytes(),
        &plan.semantics.parent_column_count.to_le_bytes(),
        &plan.semantics.global_row_cap.to_le_bytes(),
        &plan.semantics.timeframe_row_cap.to_le_bytes(),
        &plan.semantics.configured_top_k.to_le_bytes(),
        &plan.semantics.resolved_top_k.to_le_bytes(),
        &plan.semantics.minimum_per_timeframe.to_le_bytes(),
        &plan.semantics.insample_fraction_bits.to_le_bytes(),
        &plan.semantics.max_hold_bars.to_le_bytes(),
        &plan.semantics.atr_period.to_le_bytes(),
        &plan.semantics.stop_atr_multiplier_bits.to_le_bytes(),
        &plan.semantics.reward_risk_ratio_bits.to_le_bytes(),
        &plan.semantics.round_trip_cost_price_bits.to_le_bytes(),
        &plan.semantics.cpcv_split_count.to_le_bytes(),
        &plan.semantics.cpcv_test_group_count.to_le_bytes(),
        &plan.semantics.cpcv_embargo_fraction_bits.to_le_bytes(),
        &plan.semantics.cpcv_purge_fraction_bits.to_le_bytes(),
        &plan.semantics.cpcv_max_rows.to_le_bytes(),
        &plan.scopes.parent_row_count.to_le_bytes(),
        &plan.scopes.outer_split_at.to_le_bytes(),
        &plan.scopes.selection_row_start.to_le_bytes(),
        &plan.scopes.selection_row_end.to_le_bytes(),
        &plan.scopes.holdout_row_start.to_le_bytes(),
        &plan.scopes.holdout_row_end.to_le_bytes(),
        &plan.canonical_search_input_receipt_sha256,
        &plan.canonical_content_merkle_sha256,
        &plan.normalization_fit_sha256,
        &plan.feature_plan_sha256,
        &plan.source_provenance_sha256,
        &plan.ordered_feature_schema_sha256,
        &plan.column_classification_content_sha256,
        &plan.selected_cuda_ordinal.to_le_bytes(),
        &plan.cuda_device_identity_sha256,
        &plan.primary_context_identity_sha256,
        &plan.run_stream_identity_sha256,
        &plan.cuda_build_manifest_sha256,
        &plan.cuda_math_flags_sha256,
        &plan.current_config_plan_identity_sha256,
    ])
}
