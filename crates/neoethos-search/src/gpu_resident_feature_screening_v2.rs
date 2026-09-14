//! Connected Search authority for bounded resident feature screening.
//!
//! Data owns the immutable recipe, schema upload and producer replay. Search
//! owns only the configured statistical/financial semantics and binds them to
//! the opaque same-run identities supplied by gpu-cuda.

use anyhow::{Context, Result, ensure};
use neoethos_data::PreparedGpuOnlyFeatureMaterializationV3;
use neoethos_gpu_cuda::SealedNativeCudaDataPopulationPreflightFactsV1;
use neoethos_gpu_cuda::resident_feature_store_v3::ResidentTrimPrefilterScreeningSchemaUploadV2;
use neoethos_gpu_cuda::resident_trim_prefilter_v1::{
    ResidentTrimPrefilterDeviceRunV1, ResidentTrimPrefilterInputsV1,
    ResidentTrimPrefilterNativePlanFieldsV1, ResidentTrimPrefilterNativePlanV1,
    ResidentTrimPrefilterSearchPlanV1, ResidentTrimPrefilterSemanticBindingsV1,
    ResidentTrimPrefilterWorkspacePreflightRequestV2,
    SealedResidentTrimPrefilterWorkspacePreflightV2,
    UnboundResidentTrimPrefilterWorkspacePreflightV2, begin_resident_trim_prefilter_device_run_v1,
    preflight_resident_trim_prefilter_workspace_v2, resident_trim_prefilter_prefix_fit_rows_v1,
};
use sha2::{Digest, Sha256};

use crate::DiscoveryConfig;
use crate::discovery::{
    resolve_prefilter_financial_geometry_from_evaluation_v2, resolve_prefilter_top_k,
};
use crate::prefilter_schema_v1::seal_prefilter_column_classification_v1;
use crate::resident_selection_scope_v2::ResidentFeatureScreeningScopeV2;

const RESIDENT_FEATURE_SCREENING_SEMANTICS_V2: &str = concat!(
    "neoethos.resident-feature-screening.v2;",
    "outer-split-floor-four-fifths;minimum-selection-64;suffix-row-cap;",
    "atr14-directional-first-passage-cost-barriers;",
    "same-bar-dual-hit-ambiguous-zero;vertical-zero;undefined-nan;",
    "insufficient-decided-labels-invalidates-device-seal;",
    "one-selection-prefix-fit-floor-n-times-f-minus-max-horizon-1;",
    "fraction-positive-at-most-one;fit-below-three-keeps-all;no-pre-ga-cpcv;",
    "pairwise-complete-two-pass-f64-ascending-row-min30;",
    "state-template-timeframe-quota;stable-score-parent-index-tie;",
    "selected-parent-indices-final-ascending;same-admitted-stream;",
    "bounded-selected-map-readback;compact-replay;research-only"
);
const STATE_FAMILY_SEMANTICS_V2: &str =
    "base-only-prefixes:regime_,smc_,session_,fp_;additive-to-top-k";
const TIMEFRAME_GROUP_SEMANTICS_V2: &str =
    "head:M|H|D|W|MN-plus-digits;head-length-2-or-3;underscore-delimited";
const TEMPLATE_FORCE_KEEP_SEMANTICS_V2: &str =
    "seed-template-role-resolution-over-full-prefilter-schema-v1";
const SCORE_ORDER_SEMANTICS_V2: &str =
    "finite-nonnegative-f64-monotone-u64;descending;stable-parent-index-tie";
const MINIMUM_SELECTION_ROWS_V2: u64 = 64;
const MINIMUM_PAIRWISE_SAMPLES_V2: u64 = 30;
const MINIMUM_DECIDED_LABELS_V2: u64 = 100;
const MAXIMUM_REFIT_FOLDS_V2: u64 = 8;
const ATR_PERIOD_V2: u32 = 14;

#[derive(Clone, Debug)]
pub(crate) struct ResidentFeatureScreeningPlanTemplateV2 {
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
    insample_fraction: f64,
    stop_atr_multiplier: f64,
    reward_risk_ratio: f64,
    round_trip_cost_price: f64,
    cpcv_split_count: u64,
    cpcv_test_group_count: u64,
    cpcv_embargo_fraction: f64,
    cpcv_purge_fraction: f64,
    cpcv_max_rows: u64,
    semantics_sha256: [u8; 32],
}

impl ResidentFeatureScreeningPlanTemplateV2 {
    const fn screening_scope_v2(&self) -> ResidentFeatureScreeningScopeV2 {
        ResidentFeatureScreeningScopeV2::from_screening_plan_v2(
            self.parent_row_count,
            self.selection_row_start,
            self.selection_row_end,
            self.holdout_row_start,
            self.holdout_row_end,
        )
    }
}

#[derive(Clone, Debug)]
struct BoundResidentFeatureScreeningPlanV2 {
    template: ResidentFeatureScreeningPlanTemplateV2,
    plan_identity_sha256: [u8; 32],
    cuda_device_identity_sha256: [u8; 32],
    primary_context_identity_sha256: [u8; 32],
    run_stream_identity_sha256: [u8; 32],
    cuda_build_manifest_sha256: [u8; 32],
    cuda_math_flags_sha256: [u8; 32],
}

impl ResidentTrimPrefilterSearchPlanV1 for BoundResidentFeatureScreeningPlanV2 {
    fn resident_trim_prefilter_native_plan_fields_v1(
        &self,
    ) -> ResidentTrimPrefilterNativePlanFieldsV1 {
        let plan = &self.template;
        ResidentTrimPrefilterNativePlanFieldsV1 {
            parent_row_count: plan.parent_row_count,
            parent_column_count: plan.parent_column_count,
            global_row_cap: plan.global_row_cap,
            timeframe_row_cap: plan.timeframe_row_cap,
            outer_split_at: plan.outer_split_at,
            selection_row_start: plan.selection_row_start,
            selection_row_end: plan.selection_row_end,
            holdout_row_start: plan.holdout_row_start,
            holdout_row_end: plan.holdout_row_end,
            configured_top_k: plan.configured_top_k,
            resolved_top_k: plan.resolved_top_k,
            minimum_per_timeframe: plan.minimum_per_timeframe,
            max_hold_bars: plan.max_hold_bars,
            atr_period: ATR_PERIOD_V2,
            insample_fraction: plan.insample_fraction,
            stop_atr_multiplier: plan.stop_atr_multiplier,
            reward_risk_ratio: plan.reward_risk_ratio,
            round_trip_cost_price: plan.round_trip_cost_price,
            cpcv_split_count: plan.cpcv_split_count,
            cpcv_test_group_count: plan.cpcv_test_group_count,
            cpcv_embargo_fraction: plan.cpcv_embargo_fraction,
            cpcv_purge_fraction: plan.cpcv_purge_fraction,
            cpcv_max_rows: plan.cpcv_max_rows,
            semantics_sha256: plan.semantics_sha256,
            plan_identity_sha256: self.plan_identity_sha256,
            cuda_device_identity_sha256: self.cuda_device_identity_sha256,
            primary_context_identity_sha256: self.primary_context_identity_sha256,
            run_stream_identity_sha256: self.run_stream_identity_sha256,
            cuda_build_manifest_sha256: self.cuda_build_manifest_sha256,
            cuda_math_flags_sha256: self.cuda_math_flags_sha256,
        }
    }
}

#[must_use = "consume the screening preparation into one admitted two-pass run"]
pub(crate) struct PreparedResidentFeatureScreeningV2 {
    schema_upload: ResidentTrimPrefilterScreeningSchemaUploadV2,
    trim_workspace: UnboundResidentTrimPrefilterWorkspacePreflightV2,
    plan: ResidentFeatureScreeningPlanTemplateV2,
}

impl PreparedResidentFeatureScreeningV2 {
    pub(crate) fn into_parts(
        self,
    ) -> (
        ResidentTrimPrefilterScreeningSchemaUploadV2,
        UnboundResidentTrimPrefilterWorkspacePreflightV2,
        ResidentFeatureScreeningScopeV2,
        ResidentFeatureScreeningPlanTemplateV2,
    ) {
        let scope = self.plan.screening_scope_v2();
        (self.schema_upload, self.trim_workspace, scope, self.plan)
    }
}

pub(crate) fn prepare_resident_feature_screening_v2(
    config: &DiscoveryConfig,
    evaluation: &crate::genetic::EvaluationConfig,
    prepared: &PreparedGpuOnlyFeatureMaterializationV3,
    canonical_source_contract_receipt_sha256: [u8; 32],
    native_facts: &SealedNativeCudaDataPopulationPreflightFactsV1,
) -> Result<PreparedResidentFeatureScreeningV2> {
    let parent_row_count = prepared.workspace_extent().row_count();
    let parent_column_count = prepared.workspace_extent().column_count();
    ensure!(
        parent_row_count > 0 && parent_column_count > 0,
        "resident screening requires a nonempty prepared parent"
    );
    let parent_columns = usize::try_from(parent_column_count)
        .context("screening parent column count does not fit this process")?;
    let ordered_feature_names = prepared
        .ordered_feature_names_v2()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        ordered_feature_names.len() == parent_columns,
        "screening ordered feature names drifted from the prepared parent width"
    );
    let classification = seal_prefilter_column_classification_v1(&ordered_feature_names)
        .context("seal canonical resident prefilter schema classification")?;
    let schema_metadata_bytes = parent_column_count
        .checked_mul(6)
        .context("screening schema metadata bytes overflowed")?;
    let schema_upload = prepared
        .seal_feature_screening_schema_upload_v2(
            canonical_source_contract_receipt_sha256,
            classification.ordered_feature_schema_sha256(),
            classification.column_classification_content_sha256(),
            classification.column_class_flags().to_vec(),
            classification.timeframe_group_ids().to_vec(),
            classification.template_force_keep_flags().to_vec(),
            classification.timeframe_group_count(),
        )
        .map_err(anyhow::Error::new)
        .context("bind resident prefilter schema to Data's immutable parent recipe")?;

    let insample_fraction = config.runtime_overrides.resolved_prefilter_insample_frac();
    ensure!(
        insample_fraction.is_finite() && insample_fraction > 0.0 && insample_fraction <= 1.0,
        "resident screening requires a resolved fit fraction in (0, 1]"
    );
    let outer_split_at = parent_row_count
        .checked_mul(4)
        .context("screening outer split overflowed")?
        / 5;
    ensure!(
        outer_split_at >= MINIMUM_SELECTION_ROWS_V2 && outer_split_at < parent_row_count,
        "resident screening requires at least {MINIMUM_SELECTION_ROWS_V2} selection rows and a nonempty holdout"
    );
    let timeframe_row_cap = config
        .max_rows_by_timeframe
        .get(&config.timeframe_label)
        .copied()
        .unwrap_or(0);
    let effective_row_cap = match (config.max_rows, timeframe_row_cap) {
        (0, 0) => 0,
        (0, timeframe) => timeframe,
        (global, 0) => global,
        (global, timeframe) => global.min(timeframe),
    };
    let retained_selection_rows = if effective_row_cap > 0 {
        u64::try_from(effective_row_cap)
            .context("screening row cap does not fit u64")?
            .min(outer_split_at)
    } else {
        outer_split_at
    };
    ensure!(
        retained_selection_rows >= MINIMUM_SELECTION_ROWS_V2,
        "resident screening row cap leaves fewer than {MINIMUM_SELECTION_ROWS_V2} selection rows"
    );
    let selection_row_start = outer_split_at - retained_selection_rows;
    let configured_top_k = config.runtime_overrides.prefilter_top_k;
    let resolved_top_k = resolve_prefilter_top_k(
        configured_top_k,
        parent_columns,
        config.population,
        config.max_indicators,
    );
    let financial = resolve_prefilter_financial_geometry_from_evaluation_v2(evaluation);
    ensure!(
        financial.max_hold_bars > 0
            && financial.stop_atr_multiplier.is_finite()
            && financial.stop_atr_multiplier > 0.0
            && financial.reward_risk_ratio.is_finite()
            && financial.reward_risk_ratio > 0.0
            && financial.round_trip_cost_price.is_finite()
            && financial.round_trip_cost_price >= 0.0,
        "resident screening financial geometry is invalid"
    );
    let max_hold_bars =
        u64::try_from(financial.max_hold_bars).context("prefilter max-hold bars do not fit u64")?;
    let fit_rows = resident_trim_prefilter_prefix_fit_rows_v1(
        retained_selection_rows,
        insample_fraction,
        max_hold_bars,
    )
    .map_err(|error| anyhow::anyhow!("resident prefilter fit geometry: {error:?}"))?;
    // CPU Discovery keeps all columns when this one label-safe fit has fewer
    // than three rows. Charge the identical keep-all native allocation path.
    let prefilter_active = resolved_top_k > 0 && resolved_top_k < parent_columns && fit_rows >= 3;
    let semantics_sha256 =
        Sha256::digest(RESIDENT_FEATURE_SCREENING_SEMANTICS_V2.as_bytes()).into();
    let plan = ResidentFeatureScreeningPlanTemplateV2 {
        parent_row_count,
        parent_column_count,
        global_row_cap: u64::try_from(config.max_rows)
            .context("global row cap does not fit u64")?,
        timeframe_row_cap: u64::try_from(timeframe_row_cap)
            .context("timeframe row cap does not fit u64")?,
        outer_split_at,
        selection_row_start,
        selection_row_end: outer_split_at,
        holdout_row_start: outer_split_at,
        holdout_row_end: parent_row_count,
        configured_top_k: u64::try_from(configured_top_k)
            .context("configured prefilter top-k does not fit u64")?,
        resolved_top_k: u64::try_from(resolved_top_k)
            .context("resolved prefilter top-k does not fit u64")?,
        minimum_per_timeframe: u64::try_from(config.runtime_overrides.prefilter_min_per_timeframe)
            .context("prefilter timeframe minimum does not fit u64")?,
        max_hold_bars,
        insample_fraction,
        stop_atr_multiplier: financial.stop_atr_multiplier,
        reward_risk_ratio: financial.reward_risk_ratio,
        round_trip_cost_price: financial.round_trip_cost_price,
        // These describe only the cheap pre-GA fit. The immutable config and
        // all post-GA CPCV/holdout gates remain unchanged.
        cpcv_split_count: 0,
        cpcv_test_group_count: 0,
        cpcv_embargo_fraction: 0.0,
        cpcv_purge_fraction: 0.0,
        cpcv_max_rows: 0,
        semantics_sha256,
    };
    let trim_workspace = preflight_resident_trim_prefilter_workspace_v2(
        native_facts,
        ResidentTrimPrefilterWorkspacePreflightRequestV2 {
            selection_row_count: retained_selection_rows,
            parent_column_count,
            schema_metadata_bytes,
            timeframe_group_count: classification.timeframe_group_count(),
            prefilter_active,
        },
    )
    .map_err(|error| anyhow::anyhow!("resident trim workspace preflight failed: {error:?}"))?;
    Ok(PreparedResidentFeatureScreeningV2 {
        schema_upload,
        trim_workspace,
        plan,
    })
}

pub(crate) fn begin_resident_feature_screening_run_v2(
    template: ResidentFeatureScreeningPlanTemplateV2,
    memory: &SealedResidentTrimPrefilterWorkspacePreflightV2,
    inputs: ResidentTrimPrefilterInputsV1,
) -> Result<ResidentTrimPrefilterDeviceRunV1> {
    let identity = *inputs.identity();
    ensure!(
        identity.parent_row_count() == template.parent_row_count
            && identity.parent_column_count() == template.parent_column_count,
        "screening native import shape drifted from the resolved Search plan"
    );
    let mut plan_identity = Sha256::new();
    plan_identity.update(RESIDENT_FEATURE_SCREENING_SEMANTICS_V2.as_bytes());
    for value in [
        template.parent_row_count,
        template.parent_column_count,
        template.global_row_cap,
        template.timeframe_row_cap,
        template.outer_split_at,
        template.selection_row_start,
        template.selection_row_end,
        template.holdout_row_start,
        template.holdout_row_end,
        template.configured_top_k,
        template.resolved_top_k,
        template.minimum_per_timeframe,
        template.max_hold_bars,
        template.cpcv_split_count,
        template.cpcv_test_group_count,
        template.cpcv_max_rows,
    ] {
        plan_identity.update(value.to_le_bytes());
    }
    for value in [
        template.insample_fraction,
        template.stop_atr_multiplier,
        template.reward_risk_ratio,
        template.round_trip_cost_price,
        template.cpcv_embargo_fraction,
        template.cpcv_purge_fraction,
    ] {
        plan_identity.update(value.to_bits().to_le_bytes());
    }
    for hash in [
        identity.admission_identity_sha256(),
        identity.workspace_plan_identity_sha256(),
        identity.canonical_search_input_receipt_sha256(),
        identity.canonical_content_merkle_sha256(),
        identity.normalization_fit_sha256(),
        identity.feature_plan_sha256(),
        identity.source_provenance_sha256(),
        identity.ordered_feature_schema_sha256(),
        identity.column_classification_content_sha256(),
        memory.allocation_plan_sha256(),
    ] {
        plan_identity.update(hash);
    }
    let plan = BoundResidentFeatureScreeningPlanV2 {
        template,
        plan_identity_sha256: plan_identity.finalize().into(),
        cuda_device_identity_sha256: identity.cuda_device_identity_sha256(),
        primary_context_identity_sha256: identity.primary_context_identity_sha256(),
        run_stream_identity_sha256: identity.run_stream_identity_sha256(),
        cuda_build_manifest_sha256: identity.cuda_build_manifest_sha256(),
        cuda_math_flags_sha256: identity.cuda_math_flags_sha256(),
    };
    let native_plan = ResidentTrimPrefilterNativePlanV1::from_search_authority(
        &plan,
        memory,
        ResidentTrimPrefilterSemanticBindingsV1 {
            state_family_semantics: STATE_FAMILY_SEMANTICS_V2,
            timeframe_group_semantics: TIMEFRAME_GROUP_SEMANTICS_V2,
            template_force_keep_semantics: TEMPLATE_FORCE_KEEP_SEMANTICS_V2,
            score_order_semantics: SCORE_ORDER_SEMANTICS_V2,
            minimum_pairwise_samples: MINIMUM_PAIRWISE_SAMPLES_V2,
            minimum_decided_labels: MINIMUM_DECIDED_LABELS_V2,
            maximum_refit_folds: MAXIMUM_REFIT_FOLDS_V2,
        },
    )
    .map_err(|error| anyhow::anyhow!("seal native resident screening plan: {error:?}"))?;
    let (parent, schema, admission) = inputs.into_parts();
    begin_resident_trim_prefilter_device_run_v1(parent, schema, admission, native_plan)
        .map_err(|error| anyhow::anyhow!("begin native resident screening run: {error:?}"))
}
