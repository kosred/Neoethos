//! Data-owned resident Quant semantic-v4 producer preflight.
//!
//! Quant-v4 extends the v3 exact-log/temporal migration and corrects the
//! cumulative-delta validity dependency while preserving its value bits. This
//! owner admits the ordered 63-column family only after canonical OHLCV,
//! millisecond timestamps, a fixed M30-or-finer timeframe and the explicit
//! 252-trading-session annualization authority are present.

use anyhow::{Context as _, Result, ensure};
use neoethos_dataset_contracts::CanonicalTimeframe;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentFeatureStageV3, ResidentProducerCapabilityV3,
};
use neoethos_gpu_cuda::resident_feature_store_v3::{
    ResidentFeatureColumnBindingV3, ResidentFeatureStoreAssemblerV3,
    ResidentFeatureStoreCudaErrorV3,
};
use neoethos_gpu_cuda::resident_quant_v3::{
    RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4 as NATIVE_QUANT_EXACT_MATH_AUTHORITY_V4,
    RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4 as NATIVE_QUANT_FEATURE_SEMANTIC_VERSION_V4,
    RESIDENT_QUANT_IMPLEMENTATION_ID_V4 as NATIVE_QUANT_IMPLEMENTATION_ID_V4,
    RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY as NATIVE_QUANT_MIGRATION_POLICY_V3,
    RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY as NATIVE_QUANT_MIGRATION_POLICY_V4,
    ResidentQuantLaunchAuthorityV3, ResidentQuantRuntimeReceiptV3, resident_quant_capability_v3,
    seal_resident_quant_migration_closure_v3,
};
use sha2::{Digest, Sha256};

use crate::Ohlcv;

use super::gpu_resident_feature_recipe_v4::{
    ResidentCanonicalParameterV4, ResidentCanonicalParameterValueV4, ResidentProducerBatchDraftV4,
    ResidentProducerDraftV4, ResidentRouteDraftV4,
};
use super::gpu_resident_temporal_grid_v1::{
    AdmittedFixedIntradayGridV1, TRADING_SESSIONS_PER_YEAR_V3, admit_fixed_intraday_grid_v1,
};
use super::timestamps::validate_canonical_millisecond_timestamps;

pub(crate) const RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4: u32 =
    NATIVE_QUANT_FEATURE_SEMANTIC_VERSION_V4;
pub(crate) const RESIDENT_QUANT_IMPLEMENTATION_ID_V4: &str = NATIVE_QUANT_IMPLEMENTATION_ID_V4;
pub(crate) const RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4: &str =
    NATIVE_QUANT_EXACT_MATH_AUTHORITY_V4;
pub(crate) const RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY: &str = NATIVE_QUANT_MIGRATION_POLICY_V3;
pub(crate) const RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY: &str = NATIVE_QUANT_MIGRATION_POLICY_V4;
const RESIDENT_QUANT_ROUTE_DOMAIN_V4: &str = "neoethos.data.resident-quant-route.semantic-v4";
const RESIDENT_QUANT_INDICATOR_ID_V4: &str = "neoethos_quant_semantic_v4";

pub(crate) const RESIDENT_QUANT_COLUMN_NAMES_V3: [&str; 63] = [
    "quant_close",
    "quant_return_1",
    "quant_return_2",
    "quant_return_3",
    "quant_return_5",
    "quant_return_8",
    "quant_return_13",
    "quant_return_21",
    "quant_log_return",
    "quant_log_volatility",
    "quant_realized_vol_5",
    "quant_realized_vol_10",
    "quant_realized_vol_20",
    "quant_realized_vol_50",
    "quant_gk_vol_10",
    "quant_gk_vol_20",
    "quant_parkinson_vol_10",
    "quant_parkinson_vol_20",
    "quant_vol_ratio",
    "quant_hurst_100",
    "quant_autocorr_1",
    "quant_autocorr_5",
    "quant_autocorr_10",
    "quant_efficiency_ratio_10",
    "quant_efficiency_ratio_20",
    "quant_skewness_30",
    "quant_kurtosis_30",
    "quant_kyle_lambda",
    "quant_vpin",
    "quant_amihud_illiquidity",
    "quant_roll_spread",
    "quant_consec_up",
    "quant_consec_down",
    "quant_inside_bar",
    "quant_outside_bar",
    "quant_body_ratio",
    "quant_upper_shadow",
    "quant_lower_shadow",
    "quant_prev_day_h_dist",
    "quant_prev_day_l_dist",
    "quant_prev_week_h_dist",
    "quant_prev_week_l_dist",
    "quant_orb_4",
    "quant_orb_8",
    "quant_orb_12",
    "quant_amd_phase",
    "quant_wyckoff",
    "quant_engulfing_vol",
    "quant_pivot_dist",
    "quant_r1_dist",
    "quant_r2_dist",
    "quant_s1_dist",
    "quant_s2_dist",
    "quant_cam_r3_dist",
    "quant_cam_s3_dist",
    "quant_zscore_20",
    "quant_zscore_50",
    "quant_fractal_dim",
    "quant_rvol_10",
    "quant_rvol_20",
    "quant_rvol_50",
    "quant_delta_volume",
    "quant_cum_delta_zscore",
];

#[derive(Debug)]
struct QuantRouteSemanticV3 {
    formula_id: &'static str,
    warmup_bars: u64,
    requires_volume: bool,
    annualized_volatility: bool,
    temporal_session_input: bool,
}

const fn q(
    formula_id: &'static str,
    warmup_bars: u64,
    requires_volume: bool,
    annualized_volatility: bool,
    temporal_session_input: bool,
) -> QuantRouteSemanticV3 {
    QuantRouteSemanticV3 {
        formula_id,
        warmup_bars,
        requires_volume,
        annualized_volatility,
        temporal_session_input,
    }
}

const RESIDENT_QUANT_ROUTE_SEMANTICS_V3: [QuantRouteSemanticV3; 63] = [
    q("close", 0, false, false, false),
    q("simple_return:lag=1", 1, false, false, false),
    q("simple_return:lag=2", 2, false, false, false),
    q("simple_return:lag=3", 3, false, false, false),
    q("simple_return:lag=5", 5, false, false, false),
    q("simple_return:lag=8", 8, false, false, false),
    q("simple_return:lag=13", 13, false, false, false),
    q("simple_return:lag=21", 21, false, false, false),
    q("log_return:lag=1", 1, false, false, false),
    q("log_range", 0, false, false, false),
    q("realized_vol:window=5", 5, false, true, false),
    q("realized_vol:window=10", 10, false, true, false),
    q("realized_vol:window=20", 20, false, true, false),
    q("realized_vol:window=50", 50, false, true, false),
    q("garman_klass_vol:window=10", 10, false, true, false),
    q("garman_klass_vol:window=20", 20, false, true, false),
    q("parkinson_vol:window=10", 10, false, true, false),
    q("parkinson_vol:window=20", 20, false, true, false),
    q("volatility_ratio:short=5;long=20", 20, false, false, false),
    q("hurst_rescaled_range:window=100", 100, false, false, false),
    q("autocorrelation:window=50;lag=1", 51, false, false, false),
    q("autocorrelation:window=50;lag=5", 55, false, false, false),
    q("autocorrelation:window=50;lag=10", 60, false, false, false),
    q(
        "kaufman_efficiency_ratio:window=10",
        10,
        false,
        false,
        false,
    ),
    q(
        "kaufman_efficiency_ratio:window=20",
        20,
        false,
        false,
        false,
    ),
    q("return_skewness:window=30", 30, false, false, false),
    q("return_excess_kurtosis:window=30", 30, false, false, false),
    q("kyle_lambda:window=20", 20, true, false, false),
    q("vpin:bucket=50;window=10", 500, true, false, false),
    q("amihud_illiquidity:window=20", 20, true, false, false),
    q("roll_spread:window=20", 21, false, false, false),
    q("consecutive_up_bars", 1, false, false, false),
    q("consecutive_down_bars", 1, false, false, false),
    q("inside_bar", 1, false, false, false),
    q("outside_bar", 1, false, false, false),
    q("body_to_range", 0, false, false, false),
    q("upper_shadow_to_range", 0, false, false, false),
    q("lower_shadow_to_range", 0, false, false, false),
    q("previous_utc_day_high_distance", 0, false, false, true),
    q("previous_utc_day_low_distance", 0, false, false, true),
    q(
        "previous_five_trading_day_high_distance",
        0,
        false,
        false,
        true,
    ),
    q(
        "previous_five_trading_day_low_distance",
        0,
        false,
        false,
        true,
    ),
    q("asian_opening_range_breakout:bars=4", 4, false, false, true),
    q("asian_opening_range_breakout:bars=8", 8, false, false, true),
    q(
        "asian_opening_range_breakout:bars=12",
        12,
        false,
        false,
        true,
    ),
    q(
        "accumulation_manipulation_distribution:window=20",
        20,
        false,
        false,
        false,
    ),
    q("wyckoff_phase:window=30", 30, false, false, false),
    q("engulfing_with_volume", 1, true, false, false),
    q("previous_utc_day_pivot_distance", 0, false, false, true),
    q("previous_utc_day_r1_distance", 0, false, false, true),
    q("previous_utc_day_r2_distance", 0, false, false, true),
    q("previous_utc_day_s1_distance", 0, false, false, true),
    q("previous_utc_day_s2_distance", 0, false, false, true),
    q(
        "previous_utc_day_camarilla_r3_distance",
        0,
        false,
        false,
        true,
    ),
    q(
        "previous_utc_day_camarilla_s3_distance",
        0,
        false,
        false,
        true,
    ),
    q("close_zscore:window=20", 20, false, false, false),
    q("close_zscore:window=50", 50, false, false, false),
    q("fractal_dimension:window=30", 30, false, false, false),
    q("relative_volume:window=10", 10, true, false, false),
    q("relative_volume:window=20", 20, true, false, false),
    q("relative_volume:window=50", 50, true, false, false),
    q("delta_volume", 0, true, false, false),
    q("cumulative_delta_zscore:window=50", 50, true, false, false),
];

#[derive(Debug)]
pub(crate) struct ResidentQuantAllocationReceiptV3 {
    row_count: u64,
    feature_column_count: u64,
    retained_feature_device_bytes: u64,
    additional_retained_device_bytes: u64,
    scratch_device_bytes: u64,
    pointer_table_device_bytes: u64,
    isolated_pointer_schema_metadata_bytes: u64,
    parent_input_h2d_bytes: u64,
    feature_value_d2h_bytes: u64,
    native_launch_count: u64,
    producer_ready_event_count: u64,
    producer_ready_event_synchronize_count: u64,
    host_synchronize_count: u64,
    logical_validity_codes: [u8; 4],
    invalid_nan_bits: u64,
}

#[derive(Debug)]
pub(crate) struct ResidentQuantRuntimeAdmissionV3 {
    input_identity_sha256: [u8; 32],
    semantic_source_sha256: [u8; 32],
    implementation_sha256: [u8; 32],
    timeframe_millis: u64,
    bars_per_asian_session: u64,
    bars_per_utc_day: u64,
    bars_per_trading_week: u64,
    annualization_periods_per_year: u64,
    allocation: ResidentQuantAllocationReceiptV3,
}

/// Owner-derived Quant allocation evidence used by HTF recipe preflight
/// before any run-device carrier exists. Private fields prevent the shared
/// materializer from substituting caller-provided byte counts.
#[derive(Debug)]
pub(crate) struct ResidentQuantHigherTimeframeBatchMemoryV3 {
    row_count: u64,
    feature_column_count: u64,
    retained_feature_device_bytes: u64,
    additional_retained_device_bytes: u64,
    scratch_device_bytes: u64,
}

impl ResidentQuantHigherTimeframeBatchMemoryV3 {
    pub(crate) const fn row_count(&self) -> u64 {
        self.row_count
    }

    pub(crate) const fn feature_column_count(&self) -> u64 {
        self.feature_column_count
    }

    pub(crate) const fn retained_feature_device_bytes(&self) -> u64 {
        self.retained_feature_device_bytes
    }

    pub(crate) const fn additional_retained_device_bytes(&self) -> u64 {
        self.additional_retained_device_bytes
    }

    pub(crate) const fn scratch_device_bytes(&self) -> u64 {
        self.scratch_device_bytes
    }
}

impl ResidentQuantRuntimeAdmissionV3 {
    fn validate_native_receipt(&self, receipt: &ResidentQuantRuntimeReceiptV3) -> Result<()> {
        ensure!(
            receipt.semantic_version() == RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4,
            "resident Quant-v3 native semantic version drifted"
        );
        ensure!(
            u64::try_from(receipt.row_count()).context("Quant native row receipt overflow")?
                == self.allocation.row_count
                && u64::try_from(receipt.feature_column_count())
                    .context("Quant native column receipt overflow")?
                    == self.allocation.feature_column_count,
            "resident Quant-v3 native shape drifted"
        );
        ensure!(
            receipt.timeframe_millis() == self.timeframe_millis
                && receipt.bars_per_asian_session() == self.bars_per_asian_session
                && receipt.bars_per_utc_day() == self.bars_per_utc_day
                && receipt.bars_per_trading_week() == self.bars_per_trading_week
                && receipt.trading_sessions_per_year() == TRADING_SESSIONS_PER_YEAR_V3
                && receipt.annualization_periods_per_year() == self.annualization_periods_per_year,
            "resident Quant-v3 native temporal/252 authority drifted"
        );
        ensure!(
            u64::try_from(receipt.retained_feature_device_bytes())
                .context("Quant native retained-byte receipt overflow")?
                == self.allocation.retained_feature_device_bytes
                && u64::try_from(receipt.additional_retained_device_bytes())
                    .context("Quant native additional-byte receipt overflow")?
                    == self.allocation.additional_retained_device_bytes
                && u64::try_from(receipt.scratch_device_bytes())
                    .context("Quant native scratch receipt overflow")?
                    == self.allocation.scratch_device_bytes
                && u64::try_from(receipt.pointer_table_device_bytes())
                    .context("Quant native pointer-table receipt overflow")?
                    == self.allocation.pointer_table_device_bytes
                && u64::try_from(receipt.isolated_pointer_schema_metadata_bytes())
                    .context("Quant native schema receipt overflow")?
                    == self.allocation.isolated_pointer_schema_metadata_bytes,
            "resident Quant-v3 native allocation receipt drifted"
        );
        ensure!(
            u64::try_from(receipt.parent_input_h2d_bytes())
                .context("Quant native parent-H2D receipt overflow")?
                == self.allocation.parent_input_h2d_bytes
                && u64::try_from(receipt.feature_value_d2h_bytes())
                    .context("Quant native feature-D2H receipt overflow")?
                    == self.allocation.feature_value_d2h_bytes
                && u64::try_from(receipt.native_launch_count())
                    .context("Quant native launch receipt overflow")?
                    == self.allocation.native_launch_count
                && u64::try_from(receipt.producer_ready_event_count())
                    .context("Quant native event receipt overflow")?
                    == self.allocation.producer_ready_event_count
                && u64::try_from(receipt.producer_ready_event_synchronize_count())
                    .context("Quant native event-sync receipt overflow")?
                    == self.allocation.producer_ready_event_synchronize_count
                && u64::try_from(receipt.host_synchronize_count())
                    .context("Quant native host-sync receipt overflow")?
                    == self.allocation.host_synchronize_count,
            "resident Quant-v3 native transfer/launch/event receipt drifted"
        );
        ensure!(
            receipt.logical_validity_codes() == self.allocation.logical_validity_codes
                && receipt.invalid_nan_bits() == self.allocation.invalid_nan_bits
                && receipt.logical_validity_schema()
                    == "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3",
            "resident Quant-v3 native validity authority drifted"
        );
        ensure!(
            receipt.input_identity_sha256() == self.input_identity_sha256
                && receipt.semantic_source_sha256() == self.semantic_source_sha256
                && receipt.implementation_sha256() == self.implementation_sha256,
            "resident Quant-v3 native source/input identity drifted"
        );
        Ok(())
    }
}

#[must_use = "resident Quant-v3 preflight must move into schema and runtime assembly"]
#[derive(Debug)]
pub(crate) struct PreparedResidentQuantProducerV3 {
    draft: ResidentProducerDraftV4,
    runtime_admission: ResidentQuantRuntimeAdmissionV3,
}

#[must_use = "current native Quant-v3 preflight must launch or move into recipe assembly"]
#[derive(Debug)]
pub(crate) struct PreparedCurrentNativeResidentQuantProducerV3 {
    prepared: PreparedResidentQuantProducerV3,
    launch_authority: ResidentQuantLaunchAuthorityV3,
}

#[must_use = "resident Quant-v3 runtime continuation must append its exact admitted batch"]
#[derive(Debug)]
pub(crate) struct PreparedResidentQuantRuntimeV3 {
    runtime_admission: ResidentQuantRuntimeAdmissionV3,
    launch_authority: ResidentQuantLaunchAuthorityV3,
}

#[must_use = "Quant-v3 HTF parent authority and admission must both be consumed"]
#[derive(Debug)]
pub(crate) struct PendingResidentQuantHigherTimeframeParentV3 {
    runtime_admission: Option<ResidentQuantRuntimeAdmissionV3>,
    launch_authority: Option<ResidentQuantLaunchAuthorityV3>,
}

impl PreparedCurrentNativeResidentQuantProducerV3 {
    pub(crate) fn into_recipe_parts(
        self,
    ) -> (ResidentProducerDraftV4, PreparedResidentQuantRuntimeV3) {
        let (draft, runtime_admission) = self.prepared.into_parts();
        (
            draft,
            PreparedResidentQuantRuntimeV3 {
                runtime_admission,
                launch_authority: self.launch_authority,
            },
        )
    }
}

impl PreparedResidentQuantRuntimeV3 {
    pub(crate) fn into_higher_timeframe_parent_v3(
        self,
    ) -> PendingResidentQuantHigherTimeframeParentV3 {
        PendingResidentQuantHigherTimeframeParentV3 {
            runtime_admission: Some(self.runtime_admission),
            launch_authority: Some(self.launch_authority),
        }
    }

    pub(crate) fn append_to(
        self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> std::result::Result<ResidentQuantRuntimeReceiptV3, ResidentFeatureStoreCudaErrorV3> {
        let receipt = assembler.append_resident_quant_v3(bindings, self.launch_authority)?;
        self.runtime_admission
            .validate_native_receipt(&receipt)
            .map_err(|error| ResidentFeatureStoreCudaErrorV3::InvalidInput(error.to_string()))?;
        Ok(receipt)
    }
}

impl PendingResidentQuantHigherTimeframeParentV3 {
    pub(crate) fn higher_timeframe_batch_memory_v3(
        &self,
    ) -> std::result::Result<
        ResidentQuantHigherTimeframeBatchMemoryV3,
        ResidentFeatureStoreCudaErrorV3,
    > {
        let admission = self.runtime_admission.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Quant-v3 HTF admission was already consumed".into(),
            )
        })?;
        Ok(ResidentQuantHigherTimeframeBatchMemoryV3 {
            row_count: admission.allocation.row_count,
            feature_column_count: admission.allocation.feature_column_count,
            retained_feature_device_bytes: admission.allocation.retained_feature_device_bytes,
            additional_retained_device_bytes: admission.allocation.additional_retained_device_bytes,
            scratch_device_bytes: admission.allocation.scratch_device_bytes,
        })
    }

    pub(crate) fn take_launch_authority_v3(
        &mut self,
    ) -> std::result::Result<ResidentQuantLaunchAuthorityV3, ResidentFeatureStoreCudaErrorV3> {
        self.launch_authority.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Quant-v3 HTF launch authority was already consumed".into(),
            )
        })
    }

    pub(crate) fn validate_captured_parent_receipt_v3(
        mut self,
        receipt: &ResidentQuantRuntimeReceiptV3,
    ) -> std::result::Result<(), ResidentFeatureStoreCudaErrorV3> {
        if self.launch_authority.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Quant-v3 HTF receipt cannot be admitted before launch authority moves"
                    .into(),
            ));
        }
        let runtime_admission = self.runtime_admission.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Quant-v3 HTF admission was already consumed".into(),
            )
        })?;
        runtime_admission
            .validate_native_receipt(receipt)
            .map_err(|error| ResidentFeatureStoreCudaErrorV3::InvalidInput(error.to_string()))?;
        Ok(())
    }
}

pub(crate) fn preflight_current_native_resident_quant_v3(
    ohlcv: &Ohlcv,
    timeframe: CanonicalTimeframe,
) -> Result<PreparedCurrentNativeResidentQuantProducerV3> {
    let source_closure = seal_resident_quant_migration_closure_v3();
    let native_capability = resident_quant_capability_v3()?;
    ensure!(
        native_capability.implementation_sha256() == source_closure.implementation_sha256(),
        "resident Quant-v3 native capability/source closure drifted"
    );
    let prepared = preflight_resident_quant_v3(ohlcv, timeframe, native_capability)?;
    let runtime = &prepared.runtime_admission;
    let launch_authority = ResidentQuantLaunchAuthorityV3::seal(
        ohlcv.len(),
        runtime.timeframe_millis,
        runtime.bars_per_asian_session,
        runtime.bars_per_utc_day,
        runtime.bars_per_trading_week,
        TRADING_SESSIONS_PER_YEAR_V3,
        runtime.annualization_periods_per_year,
        runtime.input_identity_sha256,
        runtime.semantic_source_sha256,
        source_closure,
    )?;
    Ok(PreparedCurrentNativeResidentQuantProducerV3 {
        prepared,
        launch_authority,
    })
}

impl PreparedResidentQuantProducerV3 {
    pub(crate) fn into_parts(self) -> (ResidentProducerDraftV4, ResidentQuantRuntimeAdmissionV3) {
        (self.draft, self.runtime_admission)
    }
}

fn preflight_resident_quant_v3(
    ohlcv: &Ohlcv,
    timeframe: CanonicalTimeframe,
    native_capability: ResidentProducerCapabilityV3,
) -> Result<PreparedResidentQuantProducerV3> {
    ensure!(
        native_capability.producer() == ResidentFeatureProducerV3::Quant,
        "resident Quant-v3 native capability has the wrong producer"
    );
    ensure!(
        native_capability.implementation_id() == RESIDENT_QUANT_IMPLEMENTATION_ID_V4,
        "resident Quant-v3 native capability has the wrong implementation id"
    );
    ensure!(
        native_capability.exact_math_authority() == RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4,
        "resident Quant-v3 native capability has the wrong exact-math authority"
    );
    let implementation_sha256 = native_capability.implementation_sha256();
    let (timestamps, volume) = validate_quant_parent_v3(ohlcv)?;
    let timeframe_millis = timeframe.fixed_duration_ms().with_context(|| {
        format!("resident Quant-v3 rejects calendar base timeframe {timeframe}")
    })?;
    let grid = admit_fixed_intraday_grid_v1(timeframe_millis, timestamps)?;
    let semantic_source_sha256 = quant_semantic_source_sha256_v4();
    let input_identity_sha256 =
        quant_input_identity_v4(ohlcv, timeframe, &grid, volume, semantic_source_sha256);
    ensure!(
        input_identity_sha256 != [0; 32],
        "resident Quant-v3 input identity is zero"
    );

    let routes = RESIDENT_QUANT_COLUMN_NAMES_V3
        .iter()
        .copied()
        .zip(RESIDENT_QUANT_ROUTE_SEMANTICS_V3)
        .map(|(name, semantic)| {
            quant_route_v3(
                name,
                semantic,
                &grid,
                input_identity_sha256,
                semantic_source_sha256,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let draft = ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::Quant,
        RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4,
        routes,
        vec![ResidentProducerBatchDraftV4::from_owner_preflight(
            0, 63, 0, 0,
        )],
        native_capability,
    )?;
    let rows = u64::try_from(ohlcv.len()).context("resident Quant-v3 row count overflow")?;
    let retained_feature_device_bytes = rows
        .checked_mul(RESIDENT_QUANT_COLUMN_NAMES_V3.len() as u64)
        .and_then(|cells| cells.checked_mul(9))
        .context("resident Quant-v3 retained output bytes overflow")?;
    let pointer_table_device_bytes = 63_u64
        .checked_mul(4 * u64::BITS as u64 / 8)
        .context("resident Quant-v3 pointer-table bytes overflow")?;
    let isolated_pointer_schema_metadata_bytes = pointer_table_device_bytes
        .checked_add(64 * u64::BITS as u64 / 8)
        .and_then(|bytes| {
            RESIDENT_QUANT_COLUMN_NAMES_V3
                .iter()
                .try_fold(bytes, |sum, name| sum.checked_add(name.len() as u64))
        })
        .context("resident Quant-v3 isolated pointer/schema bytes overflow")?;
    Ok(PreparedResidentQuantProducerV3 {
        draft,
        runtime_admission: ResidentQuantRuntimeAdmissionV3 {
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256,
            timeframe_millis: grid.timeframe_millis(),
            bars_per_asian_session: grid.bars_per_asian_session(),
            bars_per_utc_day: grid.bars_per_utc_day(),
            bars_per_trading_week: grid.bars_per_trading_week(),
            annualization_periods_per_year: grid.annualization_periods_per_year(),
            allocation: ResidentQuantAllocationReceiptV3 {
                row_count: rows,
                feature_column_count: 63,
                retained_feature_device_bytes,
                additional_retained_device_bytes: 0,
                scratch_device_bytes: 0,
                pointer_table_device_bytes,
                isolated_pointer_schema_metadata_bytes,
                parent_input_h2d_bytes: 0,
                feature_value_d2h_bytes: 0,
                native_launch_count: 1,
                producer_ready_event_count: 1,
                producer_ready_event_synchronize_count: 0,
                host_synchronize_count: 0,
                logical_validity_codes: [0, 1, 5, 8],
                invalid_nan_bits: 0x7ff8_0000_0000_0000,
            },
        },
    })
}

fn validate_quant_parent_v3(ohlcv: &Ohlcv) -> Result<(&[i64], &[f64])> {
    let rows = ohlcv.len();
    ensure!(rows > 0, "resident Quant-v3 requires nonempty OHLCV");
    ensure!(
        ohlcv.open.len() == rows && ohlcv.high.len() == rows && ohlcv.low.len() == rows,
        "resident Quant-v3 OHLC shape mismatch"
    );
    let timestamps = ohlcv
        .timestamp
        .as_deref()
        .context("resident Quant-v3 requires canonical millisecond timestamps")?;
    ensure!(
        timestamps.len() == rows,
        "resident Quant-v3 timestamp shape mismatch"
    );
    validate_canonical_millisecond_timestamps(timestamps)?;
    let volume = ohlcv
        .volume
        .as_deref()
        .context("resident Quant-v3 requires volume for its complete 63-column family")?;
    ensure!(
        volume.len() == rows,
        "resident Quant-v3 volume shape mismatch"
    );
    for row in 0..rows {
        let (open, high, low, close, volume) = (
            ohlcv.open[row],
            ohlcv.high[row],
            ohlcv.low[row],
            ohlcv.close[row],
            volume[row],
        );
        ensure!(
            open.is_finite()
                && high.is_finite()
                && low.is_finite()
                && close.is_finite()
                && volume.is_finite(),
            "resident Quant-v3 row {row} contains non-finite OHLCV"
        );
        ensure!(
            open > 0.0
                && high > 0.0
                && low > 0.0
                && close > 0.0
                && volume >= 0.0
                && low <= open.min(close)
                && high >= open.max(close),
            "resident Quant-v3 row {row} violates canonical OHLCV bounds"
        );
    }
    Ok((timestamps, volume))
}

fn quant_route_v3(
    name: &str,
    semantic: QuantRouteSemanticV3,
    grid: &AdmittedFixedIntradayGridV1,
    input_identity_sha256: [u8; 32],
    semantic_source_sha256: [u8; 32],
) -> Result<ResidentRouteDraftV4> {
    let (warmup_bars, warmup_policy) = quant_warmup_authority_v3(name, semantic.warmup_bars, grid);
    let parameters = vec![
        parameter(
            "input_identity_sha256",
            ResidentCanonicalParameterValueV4::Hash(input_identity_sha256),
        )?,
        parameter(
            "semantic_source_sha256",
            ResidentCanonicalParameterValueV4::Hash(semantic_source_sha256),
        )?,
        parameter(
            "formula_id",
            ResidentCanonicalParameterValueV4::Text(semantic.formula_id.to_owned()),
        )?,
        parameter(
            "exact_log_migration",
            ResidentCanonicalParameterValueV4::Bool(quant_exact_log_route_v3(name)),
        )?,
        parameter(
            "quant_v3_migration_policy",
            ResidentCanonicalParameterValueV4::Text(
                RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY.to_owned(),
            ),
        )?,
        parameter(
            "quant_v4_migration_policy",
            ResidentCanonicalParameterValueV4::Text(
                RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY.to_owned(),
            ),
        )?,
        parameter(
            "warmup_bars",
            ResidentCanonicalParameterValueV4::U64(warmup_bars),
        )?,
        parameter(
            "warmup_policy",
            ResidentCanonicalParameterValueV4::Text(warmup_policy.to_owned()),
        )?,
        parameter(
            "requires_volume",
            ResidentCanonicalParameterValueV4::Bool(semantic.requires_volume),
        )?,
        parameter(
            "annualized_volatility",
            ResidentCanonicalParameterValueV4::Bool(semantic.annualized_volatility),
        )?,
        parameter(
            "temporal_session_input",
            ResidentCanonicalParameterValueV4::Bool(semantic.temporal_session_input),
        )?,
        parameter(
            "timeframe_millis",
            ResidentCanonicalParameterValueV4::U64(grid.timeframe_millis()),
        )?,
        parameter(
            "trading_sessions_per_year",
            ResidentCanonicalParameterValueV4::U64(TRADING_SESSIONS_PER_YEAR_V3),
        )?,
        parameter(
            "bars_per_asian_session",
            ResidentCanonicalParameterValueV4::U64(grid.bars_per_asian_session()),
        )?,
        parameter(
            "bars_per_utc_day",
            ResidentCanonicalParameterValueV4::U64(grid.bars_per_utc_day()),
        )?,
        parameter(
            "bars_per_trading_week",
            ResidentCanonicalParameterValueV4::U64(grid.bars_per_trading_week()),
        )?,
        parameter(
            "annualization_periods_per_year",
            ResidentCanonicalParameterValueV4::U64(grid.annualization_periods_per_year()),
        )?,
        parameter(
            "utc_session_contract",
            ResidentCanonicalParameterValueV4::Text(
                "asian-00:00-08:00;day-open-00:00;week=five-utc-trading-days;orb-reset=utc-day-key"
                    .to_owned(),
            ),
        )?,
        parameter(
            "temporal_formula_contract",
            ResidentCanonicalParameterValueV4::Text(
                "previous-day/week=completed-utc-boundary-high-low-close;previous-extreme-distance=(close-extreme)/(previous-high-previous-low);orb=frozen-first-N-asian-bars-then-hold-until-next-utc-day;classic-pivot=pp=(h+l+c)/3,r1=2pp-l,r2=pp+(h-l),s1=2pp-h,s2=pp-(h-l);camarilla-r3/s3=c+-(h-l)*1.1/4;pivot-distance=(close-level)/(current-high-current-low)"
                    .to_owned(),
            ),
        )?,
    ];
    ResidentRouteDraftV4::from_typed_parts(
        name,
        Some(RESIDENT_QUANT_INDICATOR_ID_V4),
        Some(name),
        ResidentFeatureStageV3::Derived,
        None,
        parameters,
        RESIDENT_QUANT_ROUTE_DOMAIN_V4,
    )
    .map_err(Into::into)
}

fn quant_exact_log_route_v3(name: &str) -> bool {
    matches!(
        name,
        "quant_log_return"
            | "quant_log_volatility"
            | "quant_realized_vol_5"
            | "quant_realized_vol_10"
            | "quant_realized_vol_20"
            | "quant_realized_vol_50"
            | "quant_gk_vol_10"
            | "quant_gk_vol_20"
            | "quant_parkinson_vol_10"
            | "quant_parkinson_vol_20"
            | "quant_vol_ratio"
            | "quant_hurst_100"
            | "quant_autocorr_1"
            | "quant_autocorr_5"
            | "quant_autocorr_10"
            | "quant_skewness_30"
            | "quant_kurtosis_30"
            | "quant_fractal_dim"
    )
}

fn quant_warmup_authority_v3(
    name: &str,
    fixed_warmup: u64,
    grid: &AdmittedFixedIntradayGridV1,
) -> (u64, &'static str) {
    match name {
        "quant_prev_day_h_dist"
        | "quant_prev_day_l_dist"
        | "quant_pivot_dist"
        | "quant_r1_dist"
        | "quant_r2_dist"
        | "quant_s1_dist"
        | "quant_s2_dist"
        | "quant_cam_r3_dist"
        | "quant_cam_s3_dist" => (
            grid.bars_per_utc_day(),
            "until-first-complete-utc-day-boundary",
        ),
        "quant_prev_week_h_dist" | "quant_prev_week_l_dist" => (
            grid.bars_per_trading_week(),
            "until-first-five-complete-utc-trading-days",
        ),
        "quant_orb_4" => (4, "reset-each-utc-day;valid-after-four-asian-bars"),
        "quant_orb_8" => (8, "reset-each-utc-day;valid-after-eight-asian-bars"),
        "quant_orb_12" => (12, "reset-each-utc-day;valid-after-twelve-asian-bars"),
        _ => (fixed_warmup, "fixed-leading-bars"),
    }
}

fn parameter(
    name: &'static str,
    value: ResidentCanonicalParameterValueV4,
) -> Result<ResidentCanonicalParameterV4> {
    ResidentCanonicalParameterV4::from_typed_value(name, value).map_err(Into::into)
}

fn quant_input_identity_v4(
    ohlcv: &Ohlcv,
    timeframe: CanonicalTimeframe,
    grid: &AdmittedFixedIntradayGridV1,
    volume: &[f64],
    semantic_source_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-quant-input.semantic-v4\0");
    hash.update(RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4.to_le_bytes());
    hash.update(semantic_source_sha256);
    hash.update([timeframe.identity_tag()]);
    hash.update(grid.timeframe_millis().to_le_bytes());
    hash.update(grid.bars_per_asian_session().to_le_bytes());
    hash.update(grid.bars_per_utc_day().to_le_bytes());
    hash.update(grid.bars_per_trading_week().to_le_bytes());
    hash.update(grid.annualization_periods_per_year().to_le_bytes());
    hash.update(TRADING_SESSIONS_PER_YEAR_V3.to_le_bytes());
    hash.update((ohlcv.len() as u64).to_le_bytes());
    for timestamp in ohlcv.timestamp.as_deref().expect("validated timestamps") {
        hash.update(timestamp.to_le_bytes());
    }
    for lane in [&ohlcv.open, &ohlcv.high, &ohlcv.low, &ohlcv.close] {
        for value in lane {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    for value in volume {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}

fn quant_semantic_source_sha256_v4() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-quant-semantic-source.v4\0");
    hash.update(include_bytes!("quant_features.rs"));
    hash.update(include_bytes!("quant_exact_math_v3.rs"));
    hash.update(include_bytes!("timestamps.rs"));
    hash.update(include_bytes!("gpu_resident_temporal_grid_v1.rs"));
    hash.update(RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4.as_bytes());
    hash.update(RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY.as_bytes());
    hash.update(RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY.as_bytes());
    hash.finalize().into()
}
