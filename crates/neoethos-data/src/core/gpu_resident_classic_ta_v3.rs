//! Crate-owned projection of the frozen Classic TA run plan into the opaque
//! gpu-cuda resident executor.
//!
//! Registry/accounting authority stays in [`super::hpc_ta`]. This module does
//! not probe memory, resolve routes a second time, invent output names, or
//! expose a caller-mintable recipe. It only converts the resolved launches
//! retained inside one [`ClassicTaRunPlan`] into content-addressed immutable
//! route evidence consumed by the same-carrier gpu-cuda assembler.

use anyhow::{Context as _, Result, bail};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentFeatureStageV3, ResidentProducerCapabilityV3,
};
use neoethos_gpu_cuda::resident_classic_ta_v3::{
    ResidentClassicTaFirstValidRuleV3, ResidentClassicTaInputV3, ResidentClassicTaLaunchRecipeV3,
    ResidentClassicTaOutputRouteV3, ResidentClassicTaParameterV3,
    ResidentClassicTaParameterValueV3, ResidentClassicTaPreDeviceMemoryReceiptV4,
    ResidentClassicTaRecipeV3, ResidentClassicTaStageV3, resident_classic_ta_capability_v3,
};
use sha2::{Digest, Sha256};
use vector_ta::indicators::bulls_v_bears::{BullsVBearsCalculationMethod, BullsVBearsMaType};
use vector_ta::indicators::dispatch::{F64FirstValidRule, F64InputKind, f64_kernel_for};

use super::classic_cuda_plan::{
    ClassicCandleStrengthMode, ClassicCudaStage, ResolvedClassicCudaLaunch, ResolvedClassicCudaNode,
};
use super::feature_registry::CLASSIC_VECTOR_TA_SEMANTIC_VERSION_V9;
use super::features::FeatureCellValidity;
use super::gpu_resident_feature_recipe_v4::{
    ResidentCanonicalParameterV4, ResidentCanonicalParameterValueV4, ResidentProducerBatchDraftV4,
    ResidentProducerDraftV4, ResidentRouteDraftV4,
};
use super::hpc_ta::{ClassicTaResidentPlanProjectionV3, ClassicTaRunPlan};

#[expect(
    dead_code,
    reason = "reserved for the unresolved Data-wide route-plan receipt; active recipe identity is sealed by gpu-cuda"
)]
pub(crate) const RESIDENT_CLASSIC_TA_PLAN_AUTHORITY_V3: &str =
    "neoethos.data.resident-classic-ta-plan.v3";
pub(crate) const MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V3: usize = 64;
pub(crate) const RESIDENT_CLASSIC_TA_LOCAL_ROUTE_DOMAIN_V4: &str =
    "neoethos.data.resident-classic-ta-local-route.semantic-v4";

/// Data-owned pre-device route projection. Every destination is producer-local;
/// no global ordinal, route id or globally hashed receipt exists until the V4
/// column-schema assembler has sealed every preceding producer span.
#[derive(Debug)]
pub(crate) struct ResidentClassicTaLocalDraftV4 {
    routes: Vec<ResidentRouteDraftV4>,
    capability: ResidentProducerCapabilityV3,
}

impl ResidentClassicTaLocalDraftV4 {
    pub(crate) fn route_count(&self) -> usize {
        self.routes.len()
    }

    pub(crate) fn into_resident_feature_recipe_draft_v4(
        self,
        recipe: &ResidentClassicTaRecipeV3,
        memory: &ResidentClassicTaPreDeviceMemoryReceiptV4,
    ) -> Result<ResidentProducerDraftV4> {
        if memory.recipe_sha256() != recipe.route_plan_sha256()
            || memory.rows() != recipe.rows()
            || recipe.output_count() != self.routes.len()
            || recipe.launches().len() != memory.launch_plans().len()
        {
            bail!("Classic TA local draft, recipe and pre-device memory authority disagree")
        }
        let mut local_first_column = 0_usize;
        let mut batches = Vec::with_capacity(recipe.launches().len());
        for (launch, launch_memory) in recipe.launches().iter().zip(memory.launch_plans()) {
            let column_count = launch.outputs().len();
            let additional_retained_bytes =
                u64::try_from(launch_memory.additional_retained_bytes())
                    .context("Classic TA additional retained bytes do not fit recipe-v4")?;
            let scratch_bytes = u64::try_from(launch_memory.retained_scratch_bytes())
                .context("Classic TA scratch bytes do not fit recipe-v4")?;
            batches.push(ResidentProducerBatchDraftV4::from_owner_preflight(
                local_first_column,
                column_count,
                additional_retained_bytes,
                scratch_bytes,
            ));
            local_first_column = local_first_column
                .checked_add(column_count)
                .context("Classic TA producer-local batch range overflow")?;
        }
        if local_first_column != self.routes.len() {
            bail!("Classic TA launch batches do not cover the local route draft")
        }
        ResidentProducerDraftV4::from_owner_preflight(
            ResidentFeatureProducerV3::ClassicTa,
            CLASSIC_VECTOR_TA_SEMANTIC_VERSION_V9,
            self.routes,
            batches,
            self.capability,
        )
        .map_err(Into::into)
    }
}

#[derive(Debug)]
pub(crate) struct ResidentClassicTaPlanV3 {
    pub(crate) recipe: ResidentClassicTaRecipeV3,
    pub(crate) local_draft: ResidentClassicTaLocalDraftV4,
}

#[derive(Debug)]
struct LaunchDescriptorV3<'plan> {
    nodes: Vec<&'plan ResolvedClassicCudaNode>,
    entry_point: &'static str,
    input: ResidentClassicTaInputV3,
    first_valid: ResidentClassicTaFirstValidRuleV3,
    parameters: Vec<ResidentClassicTaParameterV3>,
    all_nan_validity_code: u8,
}

fn usize_parameter(key: &'static str, value: usize) -> Result<ResidentClassicTaParameterV3> {
    ResidentClassicTaParameterV3::new(
        key,
        ResidentClassicTaParameterValueV3::Usize(
            u64::try_from(value).with_context(|| format!("{key} exceeds the V3 receipt width"))?,
        ),
    )
    .map_err(Into::into)
}

fn i32_parameter(key: &'static str, value: i32) -> Result<ResidentClassicTaParameterV3> {
    ResidentClassicTaParameterV3::new(key, ResidentClassicTaParameterValueV3::I32(value))
        .map_err(Into::into)
}

fn bool_parameter(key: &'static str, value: bool) -> Result<ResidentClassicTaParameterV3> {
    ResidentClassicTaParameterV3::new(key, ResidentClassicTaParameterValueV3::Bool(value))
        .map_err(Into::into)
}

fn f64_bits_parameter(key: &'static str, value: u64) -> Result<ResidentClassicTaParameterV3> {
    ResidentClassicTaParameterV3::new(key, ResidentClassicTaParameterValueV3::F64Bits(value))
        .map_err(Into::into)
}

fn text_parameter(key: &'static str, value: &'static str) -> Result<ResidentClassicTaParameterV3> {
    ResidentClassicTaParameterV3::new(
        key,
        ResidentClassicTaParameterValueV3::Text(value.to_owned()),
    )
    .map_err(Into::into)
}

fn named_descriptor<'plan, const N: usize>(
    nodes: &'plan [ResolvedClassicCudaNode; N],
    entry_point: &'static str,
    input: ResidentClassicTaInputV3,
    parameters: Vec<ResidentClassicTaParameterV3>,
) -> LaunchDescriptorV3<'plan> {
    LaunchDescriptorV3 {
        nodes: nodes.iter().collect(),
        entry_point,
        input,
        first_valid: ResidentClassicTaFirstValidRuleV3::NamedRouteOwned,
        parameters,
        all_nan_validity_code: FeatureCellValidity::ComputeFailure.code(),
    }
}

fn single_named_descriptor<'plan>(
    node: &'plan ResolvedClassicCudaNode,
    entry_point: &'static str,
    input: ResidentClassicTaInputV3,
    parameters: Vec<ResidentClassicTaParameterV3>,
) -> LaunchDescriptorV3<'plan> {
    LaunchDescriptorV3 {
        nodes: vec![node],
        entry_point,
        input,
        first_valid: ResidentClassicTaFirstValidRuleV3::NamedRouteOwned,
        parameters,
        all_nan_validity_code: FeatureCellValidity::ComputeFailure.code(),
    }
}

fn primary_input(input: F64InputKind) -> ResidentClassicTaInputV3 {
    match input {
        F64InputKind::CloseSlice => ResidentClassicTaInputV3::Close,
        F64InputKind::Hlc => ResidentClassicTaInputV3::Hlc,
        F64InputKind::Hlc3Slice => ResidentClassicTaInputV3::Hlc3,
        F64InputKind::Hlc3Volume => ResidentClassicTaInputV3::Hlc3Volume,
        F64InputKind::CloseVolume => ResidentClassicTaInputV3::CloseVolume,
        F64InputKind::HighLow => ResidentClassicTaInputV3::HighLow,
        F64InputKind::TimestampCloseVolume => ResidentClassicTaInputV3::TimestampCloseVolume,
        F64InputKind::Hl2Slice => ResidentClassicTaInputV3::Hl2,
        F64InputKind::HighLowVolume => ResidentClassicTaInputV3::HighLowVolume,
        F64InputKind::Hlcv => ResidentClassicTaInputV3::Hlcv,
        F64InputKind::Ohlcv5 => ResidentClassicTaInputV3::Ohlcv,
        F64InputKind::Ohlc4 => ResidentClassicTaInputV3::Ohlc,
        F64InputKind::Hlcc4Slice => ResidentClassicTaInputV3::Hlcc4,
        F64InputKind::VolumeSlice => ResidentClassicTaInputV3::Volume,
        F64InputKind::OpenCloseVolume => ResidentClassicTaInputV3::OpenCloseVolume,
        F64InputKind::Hlcc4Volume => ResidentClassicTaInputV3::Hlcc4Volume,
    }
}

fn primary_first_valid(rule: F64FirstValidRule) -> ResidentClassicTaFirstValidRuleV3 {
    match rule {
        F64FirstValidRule::ConsecutiveValidReturnPair => {
            ResidentClassicTaFirstValidRuleV3::CloseReturnPair
        }
        F64FirstValidRule::HighLowFiniteAndPositive => {
            ResidentClassicTaFirstValidRuleV3::HighLowFinitePositive
        }
        F64FirstValidRule::PriceVolumeFinite => {
            ResidentClassicTaFirstValidRuleV3::PriceVolumeFinite
        }
        F64FirstValidRule::HighLowFinite
        | F64FirstValidRule::HighLowMidpointFinite
        | F64FirstValidRule::CloseFinite
        | F64FirstValidRule::VolumeFiniteOnly
        | F64FirstValidRule::Ohlc4AllFinite
        | F64FirstValidRule::OpenCloseFinite => ResidentClassicTaFirstValidRuleV3::AllInputsFinite,
        F64FirstValidRule::Ignored => ResidentClassicTaFirstValidRuleV3::NotApplicable,
        F64FirstValidRule::AllInputsNonNan
        | F64FirstValidRule::HlcMaxOfIndependentFirsts
        | F64FirstValidRule::HlcCloseOnly
        | F64FirstValidRule::MaxOfIndependentFirsts
        | F64FirstValidRule::OpenCloseNonNan
        | F64FirstValidRule::HlcConsecutivePairNonNan
        | F64FirstValidRule::Ohlc4AllNonNan => ResidentClassicTaFirstValidRuleV3::AllInputsNonNan,
    }
}

fn primary_descriptor(node: &ResolvedClassicCudaNode) -> Result<LaunchDescriptorV3<'_>> {
    let spec = f64_kernel_for(node.indicator_id()).with_context(|| {
        format!(
            "resolved primary `{}` lost its vector-ta f64 kernel spec",
            node.indicator_id()
        )
    })?;
    let cuda_period = node.primary_cuda_period().with_context(|| {
        format!(
            "resolved primary `{}` lost its exact CUDA period",
            node.indicator_id()
        )
    })?;
    Ok(LaunchDescriptorV3 {
        nodes: vec![node],
        entry_point: spec.kernel.entry_point(),
        input: primary_input(spec.input),
        first_valid: primary_first_valid(spec.first_valid),
        parameters: vec![usize_parameter("cuda_period", cuda_period)?],
        all_nan_validity_code: FeatureCellValidity::ComputeFailure.code(),
    })
}

fn describe_launch(launch: &ResolvedClassicCudaLaunch) -> Result<LaunchDescriptorV3<'_>> {
    let descriptor = match launch {
        ResolvedClassicCudaLaunch::Primary(node) => return primary_descriptor(node),
        ResolvedClassicCudaLaunch::AbsoluteStrengthIndexOscillator {
            routes,
            ema_length,
            signal_length,
        } => named_descriptor(
            routes,
            "absolute_strength_index_oscillator_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("ema_length", *ema_length)?,
                usize_parameter("signal_length", *signal_length)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AdaptiveBandpassTriggerOscillator {
            routes,
            delta_bits,
            alpha_bits,
        } => named_descriptor(
            routes,
            "adaptive_bandpass_trigger_oscillator_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                f64_bits_parameter("delta", *delta_bits)?,
                f64_bits_parameter("alpha", *alpha_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AdaptiveBoundsRsi {
            routes,
            rsi_length,
            alpha_bits,
        } => named_descriptor(
            routes,
            "adaptive_bounds_rsi_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("rsi_length", *rsi_length)?,
                f64_bits_parameter("alpha", *alpha_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AdaptiveMacd {
            routes,
            length,
            fast_period,
            slow_period,
            signal_period,
        } => named_descriptor(
            routes,
            "adaptive_macd_neo_all_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("length", *length)?,
                usize_parameter("fast_period", *fast_period)?,
                usize_parameter("slow_period", *slow_period)?,
                usize_parameter("signal_period", *signal_period)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AdaptiveMomentumOscillator {
            routes,
            length,
            smoothing_length,
        } => named_descriptor(
            routes,
            "adaptive_momentum_oscillator_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("length", *length)?,
                usize_parameter("smoothing_length", *smoothing_length)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AdaptiveSchaffTrendCycle {
            routes,
            adaptive_length,
            stc_length,
            smoothing_factor_bits,
            fast_length,
            slow_length,
        } => named_descriptor(
            routes,
            "adaptive_schaff_trend_cycle_batch_f64",
            ResidentClassicTaInputV3::Hlc,
            vec![
                usize_parameter("adaptive_length", *adaptive_length)?,
                usize_parameter("stc_length", *stc_length)?,
                f64_bits_parameter("smoothing_factor", *smoothing_factor_bits)?,
                usize_parameter("fast_length", *fast_length)?,
                usize_parameter("slow_length", *slow_length)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AdjustableMaAlternatingExtremities {
            routes,
            length,
            mult_bits,
            alpha_bits,
            beta_bits,
        } => named_descriptor(
            routes,
            "adjustable_ma_alternating_extremities_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("length", *length)?,
                f64_bits_parameter("mult", *mult_bits)?,
                f64_bits_parameter("alpha", *alpha_bits)?,
                f64_bits_parameter("beta", *beta_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Alligator {
            routes,
            jaw_period,
            jaw_offset,
            teeth_period,
            teeth_offset,
            lips_period,
            lips_offset,
        } => named_descriptor(
            routes,
            "alligator_outputs_f64",
            ResidentClassicTaInputV3::Hl2,
            vec![
                usize_parameter("jaw_period", *jaw_period)?,
                usize_parameter("jaw_offset", *jaw_offset)?,
                usize_parameter("teeth_period", *teeth_period)?,
                usize_parameter("teeth_offset", *teeth_offset)?,
                usize_parameter("lips_period", *lips_period)?,
                usize_parameter("lips_offset", *lips_offset)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AlphaTrend {
            routes,
            coeff_bits,
            period,
            no_volume,
        } => named_descriptor(
            routes,
            "alphatrend_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                f64_bits_parameter("coeff", *coeff_bits)?,
                usize_parameter("period", *period)?,
                bool_parameter("no_volume", *no_volume)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Acosc { routes } => named_descriptor(
            routes,
            "acosc_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            Vec::new(),
        ),
        ResolvedClassicCudaLaunch::AndeanOscillator {
            routes,
            length,
            signal_length,
        } => named_descriptor(
            routes,
            "andean_oscillator_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("length", *length)?,
                usize_parameter("signal_length", *signal_length)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Aroon { routes, length } => named_descriptor(
            routes,
            "aroon_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![usize_parameter("length", *length)?],
        ),
        ResolvedClassicCudaLaunch::Aso {
            routes,
            period,
            mode,
        } => named_descriptor(
            routes,
            "neoethos_aso_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("period", *period)?,
                usize_parameter("mode", *mode)?,
            ],
        ),
        ResolvedClassicCudaLaunch::AutocorrelationIndicator {
            routes,
            length,
            lag,
            use_test_signal,
        } => named_descriptor(
            routes,
            "autocorrelation_indicator_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("length", *length)?,
                usize_parameter("lag", *lag)?,
                bool_parameter("use_test_signal", *use_test_signal)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Avsl {
            routes,
            fast_period,
            slow_period,
            multiplier_bits,
        } => named_descriptor(
            routes,
            "avsl_production_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("fast_period", *fast_period)?,
                usize_parameter("slow_period", *slow_period)?,
                f64_bits_parameter("multiplier", *multiplier_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Bandpass {
            routes,
            period,
            bandwidth_bits,
        } => named_descriptor(
            routes,
            "bandpass_production_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("period", *period)?,
                f64_bits_parameter("bandwidth", *bandwidth_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::BollingerBands {
            routes,
            period,
            devup_bits,
            devdn_bits,
        } => named_descriptor(
            routes,
            "bollinger_bands_production_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("period", *period)?,
                f64_bits_parameter("devup", *devup_bits)?,
                f64_bits_parameter("devdn", *devdn_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::BuffAverages {
            routes,
            fast_period,
            slow_period,
        } => named_descriptor(
            routes,
            "buff_averages_production_f64",
            ResidentClassicTaInputV3::CloseVolume,
            vec![
                usize_parameter("fast_period", *fast_period)?,
                usize_parameter("slow_period", *slow_period)?,
            ],
        ),
        ResolvedClassicCudaLaunch::CandleStrengthOscillator {
            routes,
            period,
            atr_enabled,
            atr_length,
            mode,
        } => named_descriptor(
            routes,
            "candle_strength_oscillator_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("period", *period)?,
                bool_parameter("atr_enabled", *atr_enabled)?,
                usize_parameter("atr_length", *atr_length)?,
                text_parameter(
                    "mode",
                    match mode {
                        ClassicCandleStrengthMode::Bollinger => "bollinger",
                    },
                )?,
            ],
        ),
        ResolvedClassicCudaLaunch::ChandelierExit {
            routes,
            period,
            mult_bits,
            use_close,
        } => named_descriptor(
            routes,
            "chandelier_exit_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("period", *period)?,
                f64_bits_parameter("mult", *mult_bits)?,
                bool_parameter("use_close", *use_close)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Cksp {
            routes,
            p,
            x_bits,
            q,
        } => named_descriptor(
            routes,
            "cksp_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("p", *p)?,
                f64_bits_parameter("x", *x_bits)?,
                usize_parameter("q", *q)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Coppock {
            route,
            short_roc_period,
            long_roc_period,
            ma_period,
        } => single_named_descriptor(
            route,
            "coppock_production_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("short_roc_period", *short_roc_period)?,
                usize_parameter("long_roc_period", *long_roc_period)?,
                usize_parameter("ma_period", *ma_period)?,
            ],
        ),
        ResolvedClassicCudaLaunch::CorrelationCycle {
            routes,
            period,
            threshold_bits,
        } => named_descriptor(
            routes,
            "correlation_cycle_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("period", *period)?,
                f64_bits_parameter("threshold", *threshold_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Cvi { route, period } => single_named_descriptor(
            route,
            "cvi_batch_f64",
            ResidentClassicTaInputV3::HighLow,
            vec![usize_parameter("period", *period)?],
        ),
        ResolvedClassicCudaLaunch::CyberpunkValueTrendAnalyzer {
            routes,
            entry_level,
            exit_level,
        } => named_descriptor(
            routes,
            "cyberpunk_value_trend_analyzer_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("entry_level", *entry_level)?,
                usize_parameter("exit_level", *exit_level)?,
            ],
        ),
        ResolvedClassicCudaLaunch::CycleChannelOscillator {
            routes,
            short_cycle_length,
            medium_cycle_length,
            short_multiplier_bits,
            medium_multiplier_bits,
        } => named_descriptor(
            routes,
            "cycle_channel_oscillator_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("short_cycle_length", *short_cycle_length)?,
                usize_parameter("medium_cycle_length", *medium_cycle_length)?,
                f64_bits_parameter("short_multiplier", *short_multiplier_bits)?,
                f64_bits_parameter("medium_multiplier", *medium_multiplier_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::DailyFactor {
            routes,
            threshold_level_bits,
        } => named_descriptor(
            routes,
            "daily_factor_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![f64_bits_parameter(
                "threshold_level",
                *threshold_level_bits,
            )?],
        ),
        ResolvedClassicCudaLaunch::DamianiVolatmeter {
            routes,
            vis_atr,
            vis_std,
            sed_atr,
            sed_std,
            threshold_bits,
        } => named_descriptor(
            routes,
            "damiani_volatmeter_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("vis_atr", *vis_atr)?,
                usize_parameter("vis_std", *vis_std)?,
                usize_parameter("sed_atr", *sed_atr)?,
                usize_parameter("sed_std", *sed_std)?,
                f64_bits_parameter("threshold", *threshold_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Di { routes, period } => named_descriptor(
            routes,
            "di_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![usize_parameter("period", *period)?],
        ),
        ResolvedClassicCudaLaunch::DidiIndex {
            routes,
            short_length,
            medium_length,
            long_length,
        } => named_descriptor(
            routes,
            "didi_index_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("short_length", *short_length)?,
                usize_parameter("medium_length", *medium_length)?,
                usize_parameter("long_length", *long_length)?,
            ],
        ),
        ResolvedClassicCudaLaunch::DirectionalImbalanceIndex {
            routes,
            length,
            period,
        } => named_descriptor(
            routes,
            "directional_imbalance_index_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("length", *length)?,
                usize_parameter("period", *period)?,
            ],
        ),
        ResolvedClassicCudaLaunch::DisparityIndex {
            route,
            ema_period,
            lookback_period,
            smoothing_period,
            smoothing_is_sma,
        } => single_named_descriptor(
            route,
            "disparity_index_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("ema_period", *ema_period)?,
                usize_parameter("lookback_period", *lookback_period)?,
                usize_parameter("smoothing_period", *smoothing_period)?,
                bool_parameter("smoothing_is_sma", *smoothing_is_sma)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Dm { routes, period } => named_descriptor(
            routes,
            "dm_batch_f64",
            ResidentClassicTaInputV3::HighLow,
            vec![usize_parameter("period", *period)?],
        ),
        ResolvedClassicCudaLaunch::Donchian { routes, period } => named_descriptor(
            routes,
            "donchian_all_outputs_batch_f64",
            ResidentClassicTaInputV3::HighLow,
            vec![usize_parameter("period", *period)?],
        ),
        ResolvedClassicCudaLaunch::DualUlcerIndex {
            routes,
            period,
            auto_threshold,
            threshold_bits,
        } => named_descriptor(
            routes,
            "dual_ulcer_index_all_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("period", *period)?,
                bool_parameter("auto_threshold", *auto_threshold)?,
                f64_bits_parameter("threshold", *threshold_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Dvdiqqe {
            routes,
            period,
            smoothing_period,
            fast_multiplier_bits,
            slow_multiplier_bits,
            use_tick_only,
            dynamic_center,
            tick_size_bits,
        } => named_descriptor(
            routes,
            "dvdiqqe_all_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("period", *period)?,
                usize_parameter("smoothing_period", *smoothing_period)?,
                f64_bits_parameter("fast_multiplier", *fast_multiplier_bits)?,
                f64_bits_parameter("slow_multiplier", *slow_multiplier_bits)?,
                bool_parameter("use_tick_only", *use_tick_only)?,
                bool_parameter("dynamic_center", *dynamic_center)?,
                f64_bits_parameter("tick_size", *tick_size_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::EhlersAutocorrelationPeriodogram {
            routes,
            min_period,
            max_period,
            avg_length,
            enhance,
        } => named_descriptor(
            routes,
            "ehlers_autocorrelation_periodogram_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("min_period", *min_period)?,
                usize_parameter("max_period", *max_period)?,
                usize_parameter("avg_length", *avg_length)?,
                bool_parameter("enhance", *enhance)?,
            ],
        ),
        ResolvedClassicCudaLaunch::EhlersLinearExtrapolationPredictor {
            routes,
            high_pass_length,
            low_pass_length,
            gain_bits,
            bars_forward,
            signal_mode,
        } => named_descriptor(
            routes,
            "ehlers_linear_extrapolation_predictor_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("high_pass_length", *high_pass_length)?,
                usize_parameter("low_pass_length", *low_pass_length)?,
                f64_bits_parameter("gain", *gain_bits)?,
                usize_parameter("bars_forward", *bars_forward)?,
                i32_parameter("signal_mode", *signal_mode)?,
            ],
        ),
        ResolvedClassicCudaLaunch::EhlersUndersampledDoubleMovingAverage {
            routes,
            fast_length,
            slow_length,
            sample_length,
        } => named_descriptor(
            routes,
            "ehlers_undersampled_double_moving_average_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("fast_length", *fast_length)?,
                usize_parameter("slow_length", *slow_length)?,
                usize_parameter("sample_length", *sample_length)?,
            ],
        ),
        ResolvedClassicCudaLaunch::EmaDeviationCorrectedT3 {
            routes,
            period,
            hot_bits,
            t3_mode,
        } => named_descriptor(
            routes,
            "ema_deviation_corrected_t3_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("period", *period)?,
                f64_bits_parameter("hot", *hot_bits)?,
                usize_parameter("t3_mode", *t3_mode)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Emd {
            routes,
            period,
            delta_bits,
            fraction_bits,
        } => named_descriptor(
            routes,
            "emd_outputs_f64",
            ResidentClassicTaInputV3::HighLow,
            vec![
                usize_parameter("period", *period)?,
                f64_bits_parameter("delta", *delta_bits)?,
                f64_bits_parameter("fraction", *fraction_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::EmdTrend {
            routes,
            length,
            mult_bits,
        } => named_descriptor(
            routes,
            "emd_trend_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("length", *length)?,
                f64_bits_parameter("mult", *mult_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Eri { routes, period } => named_descriptor(
            routes,
            "eri_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![usize_parameter("period", *period)?],
        ),
        ResolvedClassicCudaLaunch::EvasiveSupertrend {
            routes,
            atr_length,
            base_multiplier_bits,
            noise_threshold_bits,
            expansion_alpha_bits,
        } => named_descriptor(
            routes,
            "evasive_supertrend_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("atr_length", *atr_length)?,
                f64_bits_parameter("base_multiplier", *base_multiplier_bits)?,
                f64_bits_parameter("noise_threshold", *noise_threshold_bits)?,
                f64_bits_parameter("expansion_alpha", *expansion_alpha_bits)?,
            ],
        ),
        ResolvedClassicCudaLaunch::FibonacciTrailingStop {
            routes,
            left_bars,
            right_bars,
            level_bits,
            trigger_mode,
        } => named_descriptor(
            routes,
            "fibonacci_trailing_stop_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("left_bars", *left_bars)?,
                usize_parameter("right_bars", *right_bars)?,
                f64_bits_parameter("level", *level_bits)?,
                i32_parameter("trigger_mode", *trigger_mode)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Fisher { routes, period } => named_descriptor(
            routes,
            "fisher_outputs_f64",
            ResidentClassicTaInputV3::HighLow,
            vec![usize_parameter("period", *period)?],
        ),
        ResolvedClassicCudaLaunch::ForwardBackwardExponentialOscillator {
            routes,
            length,
            smooth,
        } => named_descriptor(
            routes,
            "forward_backward_exponential_oscillator_batch_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("length", *length)?,
                usize_parameter("smooth", *smooth)?,
            ],
        ),
        ResolvedClassicCudaLaunch::FvgTrailingStop {
            routes,
            unmitigated_fvg_lookback,
            smoothing_length,
            reset_on_cross,
        } => named_descriptor(
            routes,
            "fvg_trailing_stop_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("unmitigated_fvg_lookback", *unmitigated_fvg_lookback)?,
                usize_parameter("smoothing_length", *smoothing_length)?,
                bool_parameter("reset_on_cross", *reset_on_cross)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Gatorosc {
            routes,
            jaws_length,
            jaws_shift,
            teeth_length,
            teeth_shift,
            lips_length,
            lips_shift,
        } => named_descriptor(
            routes,
            "gatorosc_outputs_f64",
            ResidentClassicTaInputV3::Close,
            vec![
                usize_parameter("jaws_length", *jaws_length)?,
                usize_parameter("jaws_shift", *jaws_shift)?,
                usize_parameter("teeth_length", *teeth_length)?,
                usize_parameter("teeth_shift", *teeth_shift)?,
                usize_parameter("lips_length", *lips_length)?,
                usize_parameter("lips_shift", *lips_shift)?,
            ],
        ),
        ResolvedClassicCudaLaunch::Halftrend {
            routes,
            amplitude,
            channel_deviation_bits,
            atr_period,
        } => named_descriptor(
            routes,
            "halftrend_outputs_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("amplitude", *amplitude)?,
                f64_bits_parameter("channel_deviation", *channel_deviation_bits)?,
                usize_parameter("atr_period", *atr_period)?,
            ],
        ),
        ResolvedClassicCudaLaunch::FibonacciEntryBands { routes, length } => named_descriptor(
            routes,
            "fibonacci_entry_bands_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![usize_parameter("length", *length)?],
        ),
        ResolvedClassicCudaLaunch::EhlersDataSamplingRsi { routes, length } => named_descriptor(
            routes,
            "ehlers_data_sampling_relative_strength_indicator_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![usize_parameter("length", *length)?],
        ),
        ResolvedClassicCudaLaunch::BullsVBears {
            routes,
            period,
            ma_type,
            calculation_method,
            normalized_bars_back,
            raw_rolling_period,
            raw_threshold_percentile_bits,
            threshold_level_bits,
        } => named_descriptor(
            routes,
            "bulls_v_bears_batch_f64",
            ResidentClassicTaInputV3::Ohlcv,
            vec![
                usize_parameter("period", *period)?,
                text_parameter(
                    "ma_type",
                    match ma_type {
                        BullsVBearsMaType::Ema => "ema",
                        BullsVBearsMaType::Sma => "sma",
                        BullsVBearsMaType::Wma => "wma",
                    },
                )?,
                text_parameter(
                    "calculation_method",
                    match calculation_method {
                        BullsVBearsCalculationMethod::Normalized => "normalized",
                        BullsVBearsCalculationMethod::Raw => "raw",
                    },
                )?,
                usize_parameter("normalized_bars_back", *normalized_bars_back)?,
                usize_parameter("raw_rolling_period", *raw_rolling_period)?,
                f64_bits_parameter("raw_threshold_percentile", *raw_threshold_percentile_bits)?,
                f64_bits_parameter("threshold_level", *threshold_level_bits)?,
            ],
        ),
    };
    if descriptor.nodes.is_empty()
        || descriptor.nodes.len() > MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V3
    {
        bail!(
            "resolved Classic TA launch width {} is outside 1..={MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V3}",
            descriptor.nodes.len()
        )
    }
    Ok(descriptor)
}

fn stage_v3(stage: ClassicCudaStage) -> (ResidentClassicTaStageV3, ResidentFeatureStageV3) {
    match stage {
        ClassicCudaStage::Base => (ResidentClassicTaStageV3::Base, ResidentFeatureStageV3::Base),
        ClassicCudaStage::Historical => (
            ResidentClassicTaStageV3::Historical,
            ResidentFeatureStageV3::Historical,
        ),
        ClassicCudaStage::Extended => (
            ResidentClassicTaStageV3::Extended,
            ResidentFeatureStageV3::Extended,
        ),
    }
}

fn update_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    hasher.update(
        u64::try_from(bytes.len())
            .context("Classic TA identity field length exceeds u64")?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

fn admitted_working_set_sha256(projection: &ClassicTaResidentPlanProjectionV3) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.classic-ta-admitted-working-set.v3\0");
    hasher.update(
        u64::try_from(projection.budget_rows)
            .context("Classic TA budget rows exceed u64")?
            .to_le_bytes(),
    );
    hasher.update(projection.available_bytes_at_admission.to_le_bytes());
    update_len_prefixed(&mut hasher, projection.gpu_route_mode.as_bytes())?;
    hasher.update(
        u64::try_from(projection.capability_deferred_output_count)
            .context("Classic TA deferred output count exceeds u64")?
            .to_le_bytes(),
    );
    for indicator_id in &projection.capability_deferred_indicator_ids {
        update_len_prefixed(&mut hasher, indicator_id.as_bytes())?;
    }
    for indicator_id in &projection.admitted_indicator_ids {
        update_len_prefixed(&mut hasher, indicator_id.as_bytes())?;
    }
    for (indicator_id, periods) in &projection.extended_groups {
        update_len_prefixed(&mut hasher, indicator_id.as_bytes())?;
        for period in periods {
            hasher.update(
                u64::try_from(*period)
                    .context("Classic TA extended period exceeds u64")?
                    .to_le_bytes(),
            );
        }
    }
    match projection.working_set.as_deref() {
        Some(working_set) => {
            hasher.update([1]);
            for value in [
                working_set.cursor,
                working_set.next_cursor,
                working_set.planned_columns,
                working_set.space_len,
            ] {
                hasher.update(
                    u64::try_from(value)
                        .context("Classic TA working-set field exceeds u64")?
                        .to_le_bytes(),
                );
            }
            hasher.update([u8::from(working_set.exhausted)]);
            for pair in &working_set.pairs {
                update_len_prefixed(&mut hasher, pair.id.as_bytes())?;
                hasher.update(
                    u64::try_from(pair.period)
                        .context("Classic TA working-set period exceeds u64")?
                        .to_le_bytes(),
                );
            }
        }
        None => hasher.update([0]),
    }
    Ok(hasher.finalize().into())
}

fn local_output_identity_sha256(
    local_destination_column: usize,
    node: &ResolvedClassicCudaNode,
    entry_point: &str,
    parameter_tuple_sha256: [u8; 32],
) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.classic-ta-local-output.v4\0");
    hasher.update(
        u64::try_from(local_destination_column)
            .context("Classic TA local destination exceeds u64")?
            .to_le_bytes(),
    );
    update_len_prefixed(&mut hasher, node.column_name().as_bytes())?;
    update_len_prefixed(&mut hasher, node.indicator_id().as_bytes())?;
    update_len_prefixed(&mut hasher, node.output_id().as_bytes())?;
    update_len_prefixed(&mut hasher, entry_point.as_bytes())?;
    hasher.update([match node.stage() {
        ClassicCudaStage::Base => 0,
        ClassicCudaStage::Historical => 1,
        ClassicCudaStage::Extended => 2,
    }]);
    match node.swept_period() {
        Some(period) => {
            hasher.update([1]);
            hasher.update(
                u64::try_from(period)
                    .context("Classic TA swept period exceeds u64")?
                    .to_le_bytes(),
            );
        }
        None => hasher.update([0]),
    }
    hasher.update(parameter_tuple_sha256);
    Ok(hasher.finalize().into())
}

fn input_semantic_name_v4(input: ResidentClassicTaInputV3) -> &'static str {
    match input {
        ResidentClassicTaInputV3::Close => "close",
        ResidentClassicTaInputV3::Ohlc => "ohlc",
        ResidentClassicTaInputV3::Hlc3 => "hlc3",
        ResidentClassicTaInputV3::Hlc3Volume => "hlc3_volume",
        ResidentClassicTaInputV3::CloseVolume => "close_volume",
        ResidentClassicTaInputV3::HighLow => "high_low",
        ResidentClassicTaInputV3::TimestampCloseVolume => "timestamp_close_volume",
        ResidentClassicTaInputV3::Hl2 => "hl2",
        ResidentClassicTaInputV3::HighLowVolume => "high_low_volume",
        ResidentClassicTaInputV3::Hlcv => "hlcv",
        ResidentClassicTaInputV3::Ohlcv => "ohlcv",
        ResidentClassicTaInputV3::OpenCloseVolume => "open_close_volume",
        ResidentClassicTaInputV3::Hlcc4 => "hlcc4",
        ResidentClassicTaInputV3::Volume => "volume",
        ResidentClassicTaInputV3::Hlcc4Volume => "hlcc4_volume",
        ResidentClassicTaInputV3::Hlc => "hlc",
    }
}

fn first_valid_semantic_name_v4(rule: ResidentClassicTaFirstValidRuleV3) -> &'static str {
    match rule {
        ResidentClassicTaFirstValidRuleV3::AllInputsNonNan => "all_inputs_non_nan",
        ResidentClassicTaFirstValidRuleV3::AllInputsFinite => "all_inputs_finite",
        ResidentClassicTaFirstValidRuleV3::PriceVolumeFinite => "price_volume_finite",
        ResidentClassicTaFirstValidRuleV3::HighLowFinitePositive => "high_low_finite_positive",
        ResidentClassicTaFirstValidRuleV3::CloseReturnPair => "close_return_pair",
        ResidentClassicTaFirstValidRuleV3::NamedRouteOwned => "named_route_owned",
        ResidentClassicTaFirstValidRuleV3::NotApplicable => "not_applicable",
    }
}

fn typed_parameter_v4(
    parameter: &ResidentClassicTaParameterV3,
) -> Result<ResidentCanonicalParameterV4> {
    let value = match parameter.value() {
        ResidentClassicTaParameterValueV3::Usize(value) => {
            ResidentCanonicalParameterValueV4::U64(*value)
        }
        ResidentClassicTaParameterValueV3::I64(value) => {
            ResidentCanonicalParameterValueV4::I64(*value)
        }
        ResidentClassicTaParameterValueV3::I32(value) => {
            ResidentCanonicalParameterValueV4::I64(i64::from(*value))
        }
        ResidentClassicTaParameterValueV3::Bool(value) => {
            ResidentCanonicalParameterValueV4::Bool(*value)
        }
        ResidentClassicTaParameterValueV3::F64Bits(value) => {
            ResidentCanonicalParameterValueV4::F64Bits(*value)
        }
        ResidentClassicTaParameterValueV3::Text(value) => {
            ResidentCanonicalParameterValueV4::Text(value.clone())
        }
    };
    ResidentCanonicalParameterV4::from_typed_value(parameter.key(), value).map_err(Into::into)
}

fn local_route_parameters_v4(
    entry_point: &str,
    input: ResidentClassicTaInputV3,
    first_valid: ResidentClassicTaFirstValidRuleV3,
    all_nan_validity_code: u8,
    parameters: &[ResidentClassicTaParameterV3],
) -> Result<Vec<ResidentCanonicalParameterV4>> {
    let mut typed = parameters
        .iter()
        .map(typed_parameter_v4)
        .collect::<Result<Vec<_>>>()?;
    for (name, value) in [
        ("classic_cuda_entry_point", entry_point),
        ("classic_input_semantics", input_semantic_name_v4(input)),
        (
            "classic_first_valid_semantics",
            first_valid_semantic_name_v4(first_valid),
        ),
    ] {
        typed.push(
            ResidentCanonicalParameterV4::from_typed_value(
                name,
                ResidentCanonicalParameterValueV4::Text(value.to_owned()),
            )
            .map_err(anyhow::Error::from)?,
        );
    }
    typed.push(
        ResidentCanonicalParameterV4::from_typed_value(
            "classic_all_nan_validity_code",
            ResidentCanonicalParameterValueV4::U64(u64::from(all_nan_validity_code)),
        )
        .map_err(anyhow::Error::from)?,
    );
    Ok(typed)
}

/// Project the exact graph retained by `prepare_classic_ta_run_plan`. The
/// `before_full_workspace_admission` marker is intentional: the recipe is
/// sealed before the one-shot native carrier exists. `no_second_budget_probe`
/// is an executable structural invariant, not a caller promise.
pub(crate) fn preflight_resident_classic_ta_v3(
    run_plan: &ClassicTaRunPlan,
    rows: usize,
) -> Result<ResidentClassicTaPlanV3> {
    let before_full_workspace_admission = true;
    let no_second_budget_probe = true;
    if !before_full_workspace_admission || !no_second_budget_probe || rows == 0 {
        bail!("resident Classic TA preflight sequencing is invalid")
    }
    let projection = run_plan.resident_admission_projection_v3()?;
    if projection.budget_rows < rows || projection.launches.is_empty() {
        bail!("resident Classic TA projection has invalid rows or an empty graph")
    }
    let admitted_working_set_sha256 = admitted_working_set_sha256(&projection)?;
    let mut next_destination_column = 0_usize;
    let mut launches = Vec::with_capacity(projection.launches.len());
    let mut local_routes = Vec::new();
    for resolved in &projection.launches {
        let LaunchDescriptorV3 {
            nodes,
            entry_point,
            input,
            first_valid,
            parameters,
            all_nan_validity_code: base_all_nan_validity_code,
        } = describe_launch(resolved)?;
        let indicator_id = nodes[0].indicator_id();
        let stage = nodes[0].stage();
        if nodes.iter().any(|node| {
            node.indicator_id() != indicator_id
                || node.stage() != stage
                || node.swept_period() != nodes[0].swept_period()
        }) {
            bail!("resolved Classic TA launch mixed indicator/stage/period identities")
        }
        let all_nan_validity_code = if nodes[0]
            .swept_period()
            .is_some_and(|period| (period as f64) * 1.25 >= rows as f64)
        {
            FeatureCellValidity::Warmup.code()
        } else {
            base_all_nan_validity_code
        };
        let mut launch_outputs = Vec::with_capacity(nodes.len());
        for node in nodes {
            let local_destination_column = next_destination_column;
            let local_route = ResidentRouteDraftV4::from_typed_parts(
                node.column_name(),
                Some(node.indicator_id()),
                Some(node.output_id()),
                stage_v3(node.stage()).1,
                node.swept_period()
                    .map(u64::try_from)
                    .transpose()
                    .context("Classic TA swept period exceeds u64")?,
                local_route_parameters_v4(
                    entry_point,
                    input,
                    first_valid,
                    all_nan_validity_code,
                    &parameters,
                )?,
                RESIDENT_CLASSIC_TA_LOCAL_ROUTE_DOMAIN_V4,
            )?;
            let canonical_parameter_tuple_sha256 =
                local_route.canonical_parameter_tuple_sha256_v4()?;
            let local_output_identity_sha256 = local_output_identity_sha256(
                local_destination_column,
                node,
                entry_point,
                canonical_parameter_tuple_sha256,
            )?;
            let swept_period = node
                .swept_period()
                .map(u64::try_from)
                .transpose()
                .context("Classic TA swept period exceeds u64")?;
            let (classic_stage, _) = stage_v3(node.stage());
            launch_outputs.push(ResidentClassicTaOutputRouteV3::new(
                local_destination_column,
                node.column_name(),
                node.output_id(),
                classic_stage,
                swept_period,
                canonical_parameter_tuple_sha256,
                local_output_identity_sha256,
            )?);
            local_routes.push(local_route);
            next_destination_column = next_destination_column
                .checked_add(1)
                .context("Classic TA destination column overflow")?;
        }
        launches.push(ResidentClassicTaLaunchRecipeV3::new(
            indicator_id,
            entry_point,
            input,
            first_valid,
            parameters,
            launch_outputs,
            all_nan_validity_code,
        )?);
    }
    let recipe = ResidentClassicTaRecipeV3::seal(
        rows,
        projection.budget_rows,
        projection.available_bytes_at_admission,
        admitted_working_set_sha256,
        launches,
    )?;
    let capability = resident_classic_ta_capability_v3()?;
    if capability.producer() != ResidentFeatureProducerV3::ClassicTa {
        bail!("Classic TA runtime capability has the wrong producer identity")
    }
    let local_draft = ResidentClassicTaLocalDraftV4 {
        routes: local_routes,
        capability,
    };
    if recipe.output_count() != local_draft.route_count() {
        bail!("Classic TA resident recipe and local route schema widths differ")
    }
    Ok(ResidentClassicTaPlanV3 {
        recipe,
        local_draft,
    })
}
