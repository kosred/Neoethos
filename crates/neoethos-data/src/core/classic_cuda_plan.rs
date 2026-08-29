//! Exact, fail-closed CUDA plan for the canonical Classic/vector-ta producer.
//!
//! This module owns no alternate vocabulary.  Its input is the admission
//! decision already made by `hpc_ta`; every admitted base, historical, and
//! installed/budget-prefix extended output becomes one ordered typed node.
//! Preflight inventories the whole request before `GpuIndicatorEngine::new`, so
//! an incomplete graph cannot leave a half-CUDA/half-CPU feature frame behind.

use super::super::Ohlcv;
use super::gpu_indicators::{GpuIndicatorEngine, f64_primary_device_route_for};
use super::hpc_ta::{
    ClassicTaAdmissionPlan, ClassicTaComputation, MIN_BASE_VOCABULARY_COLUMNS,
    MIN_PRODUCING_INDICATOR_IDS, VOCABULARY_FLOOR_MIN_ROWS, classic_cuda_base_has_no_window,
    classic_cuda_period_anchor, classic_cuda_sweep_params, sweep_point_exclusion,
};
use super::indicator_ledger::{
    DropReason, expected_non_producing, has_finite_variation, output_ids_for, planned_output_count,
    series_fingerprint,
};
use anyhow::{Result, bail, ensure};
use std::collections::HashSet;
use vector_ta::cuda::F64NamedOutputsResult;
use vector_ta::cuda::neoethos_f64_wrapper::{ALPHATREND_MAX_PERIOD, CVI_MAX_PERIOD};
use vector_ta::indicators::adaptive_bounds_rsi::AdaptiveBoundsRsiParams;
use vector_ta::indicators::adaptive_schaff_trend_cycle::AdaptiveSchaffTrendCycleParams;
use vector_ta::indicators::adjustable_ma_alternating_extremities::AdjustableMaAlternatingExtremitiesParams;
use vector_ta::indicators::alligator::AlligatorParams;
use vector_ta::indicators::alphatrend::AlphaTrendParams;
use vector_ta::indicators::bulls_v_bears::{
    BullsVBearsCalculationMethod, BullsVBearsMaType, BullsVBearsParams,
};
use vector_ta::indicators::candle_strength_oscillator::CandleStrengthOscillatorParams;
use vector_ta::indicators::chandelier_exit::ChandelierExitParams;
use vector_ta::indicators::cksp::CkspParams;
use vector_ta::indicators::coppock::CoppockParams;
use vector_ta::indicators::dispatch::cuda_f64::is_ma_dispatcher;
use vector_ta::indicators::dispatch::{IndicatorCudaOutputF64, has_f64_resident_output_route};
use vector_ta::indicators::fibonacci_entry_bands::FibonacciEntryBandsBatchRange;
use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic, get_indicator};

const ASI_ID: &str = "absolute_strength_index_oscillator";
const ASI_OUTPUT_IDS: [&str; 3] = ["oscillator", "signal", "histogram"];
const ASI_PARAMETER_KEYS: [&str; 2] = ["ema_length", "signal_length"];
const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID: &str = "adaptive_bandpass_trigger_oscillator";
const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS: [&str; 2] = ["in_phase", "lead"];
const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_PARAMETER_KEYS: [&str; 2] = ["delta", "alpha"];
const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_DEFAULT_BITS: [u64; 2] =
    [0.1_f64.to_bits(), 0.07_f64.to_bits()];
const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_MIN_BITS: [u64; 2] =
    [0.0000001_f64.to_bits(), 0.0000001_f64.to_bits()];
const ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_MAX_BITS: [u64; 2] =
    [0.9999999_f64.to_bits(), 0.9999999_f64.to_bits()];
const ADAPTIVE_BOUNDS_RSI_ID: &str = "adaptive_bounds_rsi";
const ADAPTIVE_BOUNDS_RSI_KERNEL_OUTPUT_IDS: [&str; 10] = [
    "rsi",
    "lower",
    "lower_mid",
    "middle",
    "upper_mid",
    "upper",
    "regime",
    "regime_flip",
    "lower_signal",
    "upper_signal",
];
const ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS: [&str; 9] = [
    "lower",
    "lower_mid",
    "middle",
    "upper_mid",
    "upper",
    "regime",
    "regime_flip",
    "lower_signal",
    "upper_signal",
];
const ADAPTIVE_BOUNDS_RSI_PARAMETER_KEYS: [&str; 2] = ["rsi_length", "alpha"];
const ADAPTIVE_MACD_ID: &str = "adaptive_macd";
const ADAPTIVE_MACD_OUTPUT_IDS: [&str; 3] = ["macd", "signal", "hist"];
const ADAPTIVE_MACD_PARAMETER_KEYS: [&str; 4] =
    ["length", "fast_period", "slow_period", "signal_period"];
const ADAPTIVE_MOMENTUM_OSCILLATOR_ID: &str = "adaptive_momentum_oscillator";
const ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS: [&str; 2] = ["amo", "ama"];
const ADAPTIVE_MOMENTUM_OSCILLATOR_PARAMETER_KEYS: [&str; 3] =
    ["length", "smoothing_length", "output"];
const ADAPTIVE_SCHAFF_TREND_CYCLE_ID: &str = "adaptive_schaff_trend_cycle";
const ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS: [&str; 2] = ["stc", "histogram"];
const ADAPTIVE_SCHAFF_TREND_CYCLE_PARAMETER_KEYS: [&str; 5] = [
    "adaptive_length",
    "stc_length",
    "smoothing_factor",
    "fast_length",
    "slow_length",
];
const ADAPTIVE_SCHAFF_TREND_CYCLE_WINDOW_KEYS: [&str; 4] = [
    "adaptive_length",
    "stc_length",
    "fast_length",
    "slow_length",
];
const ADJUSTABLE_MA_ID: &str = "adjustable_ma_alternating_extremities";
const ADJUSTABLE_MA_FULL_OUTPUT_IDS: [&str; 10] = [
    "ma",
    "upper",
    "lower",
    "extremity",
    "state",
    "changed",
    "smoothed_open",
    "smoothed_high",
    "smoothed_low",
    "smoothed_close",
];
const ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS: [&str; 9] = [
    "ma",
    "upper",
    "lower",
    "extremity",
    "state",
    "changed",
    "smoothed_open",
    "smoothed_high",
    "smoothed_low",
];
const ADJUSTABLE_MA_PARAMETER_KEYS: [&str; 4] = ["length", "mult", "alpha", "beta"];
const ALLIGATOR_ID: &str = "alligator";
const ALLIGATOR_OUTPUT_IDS: [&str; 3] = ["jaw", "teeth", "lips"];
const ALLIGATOR_PARAMETER_KEYS: [&str; 6] = [
    "jaw_period",
    "jaw_offset",
    "teeth_period",
    "teeth_offset",
    "lips_period",
    "lips_offset",
];
const ALLIGATOR_SWEEP_PARAMETER_KEYS: [&str; 3] = ["jaw_period", "teeth_period", "lips_period"];
const ALPHATREND_ID: &str = "alphatrend";
const ALPHATREND_OUTPUT_IDS: [&str; 2] = ["k1", "k2"];
const ALPHATREND_PARAMETER_KEYS: [&str; 3] = ["coeff", "period", "no_volume"];
const ALPHATREND_SWEEP_PARAMETER_KEYS: [&str; 1] = ["period"];
const ACOSC_ID: &str = "acosc";
const ACOSC_OUTPUT_IDS: [&str; 2] = ["osc", "change"];
const ANDEAN_OSCILLATOR_ID: &str = "andean_oscillator";
const ANDEAN_OSCILLATOR_OUTPUT_IDS: [&str; 3] = ["bull", "bear", "signal"];
const ANDEAN_OSCILLATOR_PARAMETER_KEYS: [&str; 2] = ["length", "signal_length"];
const AROON_ID: &str = "aroon";
const AROON_OUTPUT_IDS: [&str; 2] = ["up", "down"];
const AROON_PARAMETER_KEYS: [&str; 1] = ["length"];
const ASO_ID: &str = "aso";
const ASO_OUTPUT_IDS: [&str; 2] = ["bulls", "bears"];
const ASO_PARAMETER_KEYS: [&str; 2] = ["period", "mode"];
const AUTOCORRELATION_INDICATOR_ID: &str = "autocorrelation_indicator";
const AUTOCORRELATION_INDICATOR_OUTPUT_IDS: [&str; 2] = ["filtered", "correlation"];
const AUTOCORRELATION_INDICATOR_PARAMETER_KEYS: [&str; 3] = ["length", "lag", "use_test_signal"];
const AVSL_ID: &str = "avsl";
const AVSL_OUTPUT_IDS: [&str; 1] = ["value"];
// Registered single-output families enter the CPU dispatcher through its
// default-output request. Preflight resolves that receipt to `value`; changing
// the request itself to `Some("value")` would also change the canonical column
// name from `avsl`/`avsl_{period}` to a non-CPU `_value` vocabulary.
const AVSL_REQUESTED_OUTPUT_IDS: [Option<&str>; 1] = [None];
const AVSL_PARAMETER_KEYS: [&str; 3] = ["fast_period", "slow_period", "multiplier"];
const BANDPASS_ID: &str = "bandpass";
const BANDPASS_OUTPUT_IDS: [&str; 4] = ["bp", "bp_normalized", "signal", "trigger"];
const BANDPASS_PARAMETER_KEYS: [&str; 2] = ["period", "bandwidth"];
const BOLLINGER_BANDS_ID: &str = "bollinger_bands";
const BOLLINGER_BANDS_OUTPUT_IDS: [&str; 3] = ["upper", "middle", "lower"];
const BOLLINGER_BANDS_PARAMETER_KEYS: [&str; 3] = ["period", "devup", "devdn"];
const BUFF_AVERAGES_ID: &str = "buff_averages";
const BUFF_AVERAGES_OUTPUT_IDS: [&str; 2] = ["fast", "slow"];
const BUFF_AVERAGES_PARAMETER_KEYS: [&str; 3] = ["fast_period", "slow_period", "output"];
const CANDLE_STRENGTH_OSCILLATOR_ID: &str = "candle_strength_oscillator";
const CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS: [&str; 6] = [
    "strength",
    "highs",
    "lows",
    "mid",
    "long_signal",
    "short_signal",
];
const CANDLE_STRENGTH_OSCILLATOR_PARAMETER_KEYS: [&str; 4] =
    ["period", "atr_enabled", "atr_length", "mode"];
const CHANDELIER_EXIT_ID: &str = "chandelier_exit";
const CHANDELIER_EXIT_OUTPUT_IDS: [&str; 2] = ["long_stop", "short_stop"];
const CHANDELIER_EXIT_PARAMETER_KEYS: [&str; 3] = ["period", "mult", "use_close"];
const CKSP_ID: &str = "cksp";
const CKSP_OUTPUT_IDS: [&str; 2] = ["long_values", "short_values"];
const CKSP_PARAMETER_KEYS: [&str; 3] = ["p", "x", "q"];
const COPPOCK_ID: &str = "coppock";
const COPPOCK_OUTPUT_ID: &str = "value";
const COPPOCK_PARAMETER_KEYS: [&str; 3] = ["short_roc_period", "long_roc_period", "ma_period"];
const CORRELATION_CYCLE_ID: &str = "correlation_cycle";
const CORRELATION_CYCLE_OUTPUT_IDS: [&str; 4] = ["real", "imag", "angle", "state"];
const CORRELATION_CYCLE_PARAMETER_KEYS: [&str; 2] = ["period", "threshold"];
const CVI_ID: &str = "cvi";
const CVI_OUTPUT_ID: &str = "value";
const CVI_PARAMETER_KEYS: [&str; 1] = ["period"];
const CYBERPUNK_VALUE_TREND_ANALYZER_ID: &str = "cyberpunk_value_trend_analyzer";
const CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS: [&str; 6] = [
    "value_trend",
    "value_trend_lag",
    "deviation_index",
    "overbought_signal",
    "buy_signal",
    "sell_signal",
];
const CYBERPUNK_VALUE_TREND_ANALYZER_PARAMETER_KEYS: [&str; 2] = ["entry_level", "exit_level"];
const CYCLE_CHANNEL_OSCILLATOR_ID: &str = "cycle_channel_oscillator";
const CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS: [&str; 2] = ["fast", "slow"];
const CYCLE_CHANNEL_OSCILLATOR_PARAMETER_KEYS: [&str; 5] = [
    "source",
    "short_cycle_length",
    "medium_cycle_length",
    "short_multiplier",
    "medium_multiplier",
];
const CYCLE_CHANNEL_OSCILLATOR_WINDOW_KEYS: [&str; 2] =
    ["short_cycle_length", "medium_cycle_length"];
const CYCLE_CHANNEL_OSCILLATOR_SOURCE_VALUES: [&str; 8] = [
    "open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4",
];
const DAILY_FACTOR_ID: &str = "daily_factor";
const DAILY_FACTOR_FULL_OUTPUT_IDS: [&str; 3] = ["value", "ema", "signal"];
const DAILY_FACTOR_PRODUCTION_OUTPUT_IDS: [&str; 2] = ["value", "signal"];
const DAILY_FACTOR_PARAMETER_KEYS: [&str; 1] = ["threshold_level"];
const DAMIANI_VOLATMETER_ID: &str = "damiani_volatmeter";
const DAMIANI_VOLATMETER_OUTPUT_IDS: [&str; 2] = ["vol", "anti"];
const DAMIANI_VOLATMETER_PARAMETER_KEYS: [&str; 5] =
    ["vis_atr", "vis_std", "sed_atr", "sed_std", "threshold"];
const DAMIANI_VOLATMETER_WINDOW_KEYS: [&str; 4] = ["vis_atr", "vis_std", "sed_atr", "sed_std"];
const DI_ID: &str = "di";
const DI_OUTPUT_IDS: [&str; 2] = ["plus", "minus"];
const DI_PARAMETER_KEYS: [&str; 1] = ["period"];
const DIDI_INDEX_ID: &str = "didi_index";
const DIDI_INDEX_OUTPUT_IDS: [&str; 4] = ["short", "long", "crossover", "crossunder"];
const DIDI_INDEX_PARAMETER_KEYS: [&str; 3] = ["short_length", "medium_length", "long_length"];
const DIRECTIONAL_IMBALANCE_INDEX_ID: &str = "directional_imbalance_index";
const DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS: [&str; 6] =
    ["up", "down", "bulls", "bears", "upper", "lower"];
const DIRECTIONAL_IMBALANCE_INDEX_PARAMETER_KEYS: [&str; 2] = ["length", "period"];
const DISPARITY_INDEX_ID: &str = "disparity_index";
const DISPARITY_INDEX_OUTPUT_ID: &str = "value";
const DISPARITY_INDEX_PARAMETER_KEYS: [&str; 4] = [
    "ema_period",
    "lookback_period",
    "smoothing_period",
    "smoothing_type",
];
const DM_ID: &str = "dm";
const DM_OUTPUT_IDS: [&str; 2] = ["plus", "minus"];
const DM_PARAMETER_KEYS: [&str; 1] = ["period"];
const DONCHIAN_ID: &str = "donchian";
const DONCHIAN_OUTPUT_IDS: [&str; 3] = ["upper", "middle", "lower"];
const DONCHIAN_PARAMETER_KEYS: [&str; 1] = ["period"];
const DUAL_ULCER_INDEX_ID: &str = "dual_ulcer_index";
const DUAL_ULCER_INDEX_OUTPUT_IDS: [&str; 3] = ["long_ulcer", "short_ulcer", "threshold"];
const DUAL_ULCER_INDEX_PARAMETER_KEYS: [&str; 3] = ["period", "auto_threshold", "threshold"];
const DVDIQQE_ID: &str = "dvdiqqe";
const DVDIQQE_OUTPUT_IDS: [&str; 4] = ["dvdi", "fast_tl", "slow_tl", "center_line"];
const DVDIQQE_PARAMETER_KEYS: [&str; 7] = [
    "period",
    "smoothing_period",
    "fast_multiplier",
    "slow_multiplier",
    "volume_type",
    "center_type",
    "tick_size",
];
const EHLERS_AUTOCORRELATION_PERIODOGRAM_ID: &str = "ehlers_autocorrelation_periodogram";
const EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS: [&str; 2] =
    ["dominant_cycle", "normalized_power"];
const EHLERS_AUTOCORRELATION_PERIODOGRAM_PARAMETER_KEYS: [&str; 4] =
    ["min_period", "max_period", "avg_length", "enhance"];
const EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID: &str = "ehlers_linear_extrapolation_predictor";
const EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS: [&str; 5] =
    ["prediction", "filter", "state", "go_long", "go_short"];
const EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_PARAMETER_KEYS: [&str; 5] = [
    "high_pass_length",
    "low_pass_length",
    "gain",
    "bars_forward",
    "signal_mode",
];
const EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID: &str =
    "ehlers_undersampled_double_moving_average";
const EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS: [&str; 2] = ["fast", "slow"];
const EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_PARAMETER_KEYS: [&str; 4] =
    ["fast_length", "slow_length", "sample_length", "output"];
const EMA_DEVIATION_CORRECTED_T3_ID: &str = "ema_deviation_corrected_t3";
const EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS: [&str; 2] = ["corrected", "t3"];
const EMA_DEVIATION_CORRECTED_T3_PARAMETER_KEYS: [&str; 4] = ["period", "hot", "t3_mode", "output"];
const EMD_ID: &str = "emd";
const EMD_OUTPUT_IDS: [&str; 3] = ["upperband", "middleband", "lowerband"];
const EMD_PARAMETER_KEYS: [&str; 3] = ["period", "delta", "fraction"];
const EMD_TREND_ID: &str = "emd_trend";
const EMD_TREND_OUTPUT_IDS: [&str; 4] = ["direction", "average", "upper", "lower"];
const EMD_TREND_PARAMETER_KEYS: [&str; 4] = ["source", "avg_type", "length", "mult"];
const ERI_ID: &str = "eri";
const ERI_OUTPUT_IDS: [&str; 2] = ["bull", "bear"];
const ERI_PARAMETER_KEYS: [&str; 2] = ["period", "ma_type"];
const EVASIVE_SUPERTREND_ID: &str = "evasive_supertrend";
const EVASIVE_SUPERTREND_OUTPUT_IDS: [&str; 4] = ["band", "state", "noisy", "changed"];
const EVASIVE_SUPERTREND_PARAMETER_KEYS: [&str; 4] = [
    "atr_length",
    "base_multiplier",
    "noise_threshold",
    "expansion_alpha",
];
const FIBONACCI_TRAILING_STOP_ID: &str = "fibonacci_trailing_stop";
const FIBONACCI_TRAILING_STOP_OUTPUT_IDS: [&str; 4] =
    ["trailing_stop", "long_stop", "short_stop", "direction"];
const FIBONACCI_TRAILING_STOP_PARAMETER_KEYS: [&str; 4] =
    ["left_bars", "right_bars", "level", "trigger"];
const FISHER_ID: &str = "fisher";
const FISHER_OUTPUT_IDS: [&str; 2] = ["fisher", "signal"];
const FISHER_PARAMETER_KEYS: [&str; 1] = ["period"];
const FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID: &str = "forward_backward_exponential_oscillator";
const FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS: [&str; 3] =
    ["forward_backward", "backward", "histogram"];
const FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_PARAMETER_KEYS: [&str; 2] = ["length", "smooth"];
const FVG_TRAILING_STOP_ID: &str = "fvg_trailing_stop";
const FVG_TRAILING_STOP_OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];
const FVG_TRAILING_STOP_PARAMETER_KEYS: [&str; 3] = [
    "unmitigated_fvg_lookback",
    "smoothing_length",
    "reset_on_cross",
];
const GATOROSC_ID: &str = "gatorosc";
const GATOROSC_OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_change", "lower_change"];
const GATOROSC_PARAMETER_KEYS: [&str; 6] = [
    "jaws_length",
    "jaws_shift",
    "teeth_length",
    "teeth_shift",
    "lips_length",
    "lips_shift",
];
const GATOROSC_SWEEP_PARAMETER_KEYS: [&str; 3] = ["jaws_length", "teeth_length", "lips_length"];
const HALFTREND_ID: &str = "halftrend";
const HALFTREND_OUTPUT_IDS: [&str; 6] = [
    "halftrend",
    "trend",
    "atr_high",
    "atr_low",
    "buy_signal",
    "sell_signal",
];
const HALFTREND_PARAMETER_KEYS: [&str; 3] = ["amplitude", "channel_deviation", "atr_period"];
const FIBONACCI_ENTRY_BANDS_ID: &str = "fibonacci_entry_bands";
const FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS: [&str; 18] = [
    "middle",
    "trend",
    "upper_0618",
    "upper_1000",
    "upper_1618",
    "upper_2618",
    "lower_0618",
    "lower_1000",
    "lower_1618",
    "lower_2618",
    "tp_long_band",
    "tp_short_band",
    "go_long",
    "go_short",
    "rejection_long",
    "rejection_short",
    "long_bounce",
    "short_bounce",
];
const FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS: [&str; 16] = [
    "middle",
    "trend",
    "upper_0618",
    "upper_1000",
    "upper_1618",
    "upper_2618",
    "lower_0618",
    "lower_1000",
    "lower_1618",
    "lower_2618",
    "go_long",
    "go_short",
    "rejection_long",
    "rejection_short",
    "long_bounce",
    "short_bounce",
];
const FIBONACCI_ENTRY_BANDS_PARAMETER_KEYS: [&str; 5] = [
    "source",
    "length",
    "atr_length",
    "use_atr",
    "tp_aggressiveness",
];
const EHLERS_DATA_SAMPLING_RSI_ID: &str = "ehlers_data_sampling_relative_strength_indicator";
const EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS: [&str; 3] = ["ds_rsi", "original_rsi", "signal"];
const EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS: [&str; 2] = ["ds_rsi", "signal"];
const EHLERS_DATA_SAMPLING_RSI_PARAMETER_KEYS: [&str; 1] = ["length"];
const BULLS_V_BEARS_ID: &str = "bulls_v_bears";
const BULLS_V_BEARS_FULL_OUTPUT_IDS: [&str; 10] = [
    "value",
    "bull",
    "bear",
    "ma",
    "upper",
    "lower",
    "bullish_signal",
    "bearish_signal",
    "zero_cross_up",
    "zero_cross_down",
];
const BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS: [&str; 7] = [
    "value",
    "bull",
    "bear",
    "bullish_signal",
    "bearish_signal",
    "zero_cross_up",
    "zero_cross_down",
];
const BULLS_V_BEARS_PARAMETER_KEYS: [&str; 7] = [
    "period",
    "ma_type",
    "calculation_method",
    "normalized_bars_back",
    "raw_rolling_period",
    "raw_threshold_percentile",
    "threshold_level",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassicCandleStrengthMode {
    Bollinger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassicCudaStage {
    Base,
    Historical,
    Extended,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassicCudaValueKind {
    F64,
    /// One signed candlestick-pattern row.  It stays typed separately until a
    /// real discrete CUDA matrix route exists.
    PatternI32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ClassicCudaAnchor {
    Resolved(usize),
    Missing(String),
    NotApplicable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ClassicCudaParameters {
    Defaults {
        anchor: ClassicCudaAnchor,
        require_period_invariant_kernel: bool,
    },
    Swept {
        period: usize,
        overrides: Vec<(&'static str, i64)>,
        anchor: ClassicCudaAnchor,
    },
    DiscreteDefaults,
}

impl ClassicCudaParameters {
    fn anchor(&self) -> &ClassicCudaAnchor {
        match self {
            Self::Defaults { anchor, .. } | Self::Swept { anchor, .. } => anchor,
            Self::DiscreteDefaults => &ClassicCudaAnchor::NotApplicable,
        }
    }

    fn swept_period(&self) -> Option<usize> {
        match self {
            Self::Swept { period, .. } => Some(*period),
            Self::Defaults { .. } | Self::DiscreteDefaults => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClassicCudaNode {
    pub(crate) stage: ClassicCudaStage,
    pub(crate) indicator_id: &'static str,
    /// `None` is the CPU dispatcher's canonical default-output request.  It is
    /// resolved to the registered primary identity during preflight.
    pub(crate) requested_output_id: Option<&'static str>,
    pub(crate) column_name: String,
    pub(crate) value_kind: ClassicCudaValueKind,
    pub(crate) parameters: ClassicCudaParameters,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClassicCudaPlan {
    pub(crate) rows: usize,
    pub(crate) nodes: Vec<ClassicCudaNode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ClassicCudaResolvedRoute {
    Primary {
        cuda_period: usize,
    },
    AbsoluteStrengthIndexOscillator {
        ema_length: usize,
        signal_length: usize,
    },
    AdaptiveBandpassTriggerOscillator {
        delta_bits: u64,
        alpha_bits: u64,
    },
    AdaptiveBoundsRsi {
        rsi_length: usize,
        alpha_bits: u64,
    },
    AdaptiveMacd {
        length: usize,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    },
    AdaptiveMomentumOscillator {
        length: usize,
        smoothing_length: usize,
    },
    AdaptiveSchaffTrendCycle {
        adaptive_length: usize,
        stc_length: usize,
        smoothing_factor_bits: u64,
        fast_length: usize,
        slow_length: usize,
    },
    AdjustableMaAlternatingExtremities {
        length: usize,
        mult_bits: u64,
        alpha_bits: u64,
        beta_bits: u64,
    },
    Alligator {
        jaw_period: usize,
        jaw_offset: usize,
        teeth_period: usize,
        teeth_offset: usize,
        lips_period: usize,
        lips_offset: usize,
    },
    AlphaTrend {
        coeff_bits: u64,
        period: usize,
        no_volume: bool,
    },
    Acosc,
    AndeanOscillator {
        length: usize,
        signal_length: usize,
    },
    Aroon {
        length: usize,
    },
    Aso {
        period: usize,
        mode: usize,
    },
    AutocorrelationIndicator {
        length: usize,
        lag: usize,
        use_test_signal: bool,
    },
    Avsl {
        fast_period: usize,
        slow_period: usize,
        multiplier_bits: u64,
    },
    Bandpass {
        period: usize,
        bandwidth_bits: u64,
    },
    BollingerBands {
        period: usize,
        devup_bits: u64,
        devdn_bits: u64,
    },
    BuffAverages {
        fast_period: usize,
        slow_period: usize,
    },
    CandleStrengthOscillator {
        period: usize,
        atr_enabled: bool,
        atr_length: usize,
        mode: ClassicCandleStrengthMode,
    },
    ChandelierExit {
        period: usize,
        mult_bits: u64,
        use_close: bool,
    },
    Cksp {
        p: usize,
        x_bits: u64,
        q: usize,
    },
    Coppock {
        short_roc_period: usize,
        long_roc_period: usize,
        ma_period: usize,
    },
    CorrelationCycle {
        period: usize,
        threshold_bits: u64,
    },
    Cvi {
        period: usize,
    },
    CyberpunkValueTrendAnalyzer {
        entry_level: usize,
        exit_level: usize,
    },
    CycleChannelOscillator {
        short_cycle_length: usize,
        medium_cycle_length: usize,
        short_multiplier_bits: u64,
        medium_multiplier_bits: u64,
    },
    DailyFactor {
        threshold_level_bits: u64,
    },
    DamianiVolatmeter {
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
        threshold_bits: u64,
    },
    Di {
        period: usize,
    },
    DidiIndex {
        short_length: usize,
        medium_length: usize,
        long_length: usize,
    },
    DirectionalImbalanceIndex {
        length: usize,
        period: usize,
    },
    DisparityIndex {
        ema_period: usize,
        lookback_period: usize,
        smoothing_period: usize,
        smoothing_is_sma: bool,
    },
    Dm {
        period: usize,
    },
    Donchian {
        period: usize,
    },
    DualUlcerIndex {
        period: usize,
        auto_threshold: bool,
        threshold_bits: u64,
    },
    Dvdiqqe {
        period: usize,
        smoothing_period: usize,
        fast_multiplier_bits: u64,
        slow_multiplier_bits: u64,
        use_tick_only: bool,
        dynamic_center: bool,
        tick_size_bits: u64,
    },
    EhlersAutocorrelationPeriodogram {
        min_period: usize,
        max_period: usize,
        avg_length: usize,
        enhance: bool,
    },
    EhlersLinearExtrapolationPredictor {
        high_pass_length: usize,
        low_pass_length: usize,
        gain_bits: u64,
        bars_forward: usize,
        signal_mode: i32,
    },
    EhlersUndersampledDoubleMovingAverage {
        fast_length: usize,
        slow_length: usize,
        sample_length: usize,
    },
    EmaDeviationCorrectedT3 {
        period: usize,
        hot_bits: u64,
        t3_mode: usize,
    },
    Emd {
        period: usize,
        delta_bits: u64,
        fraction_bits: u64,
    },
    EmdTrend {
        length: usize,
        mult_bits: u64,
    },
    Eri {
        period: usize,
    },
    EvasiveSupertrend {
        atr_length: usize,
        base_multiplier_bits: u64,
        noise_threshold_bits: u64,
        expansion_alpha_bits: u64,
    },
    FibonacciTrailingStop {
        left_bars: usize,
        right_bars: usize,
        level_bits: u64,
        trigger_mode: i32,
    },
    Fisher {
        period: usize,
    },
    ForwardBackwardExponentialOscillator {
        length: usize,
        smooth: usize,
    },
    FvgTrailingStop {
        unmitigated_fvg_lookback: usize,
        smoothing_length: usize,
        reset_on_cross: bool,
    },
    Gatorosc {
        jaws_length: usize,
        jaws_shift: usize,
        teeth_length: usize,
        teeth_shift: usize,
        lips_length: usize,
        lips_shift: usize,
    },
    Halftrend {
        amplitude: usize,
        channel_deviation_bits: u64,
        atr_period: usize,
    },
    FibonacciEntryBands {
        length: usize,
    },
    EhlersDataSamplingRsi {
        length: usize,
    },
    BullsVBears {
        period: usize,
        ma_type: BullsVBearsMaType,
        calculation_method: BullsVBearsCalculationMethod,
        normalized_bars_back: usize,
        raw_rolling_period: usize,
        raw_threshold_percentile_bits: u64,
        threshold_level_bits: u64,
    },
}

fn positive_usize_parameter(
    indicator_id: &str,
    key: &str,
    value: i64,
) -> std::result::Result<usize, String> {
    if value <= 0 {
        return Err(format!(
            "{indicator_id}.{key}: expected a positive integer, found {value}"
        ));
    }
    usize::try_from(value).map_err(|_| format!("{indicator_id}.{key}: value {value} exceeds usize"))
}

/// Resolve the same two-dimensional parameter point the canonical CPU batch
/// arm receives.  The all-output CUDA entry point has no generic `period`
/// argument: admitting it from the one-dimensional primary ABI would silently
/// change ASI's formula, so both keys and the registry/default contract are
/// proved here before a CUDA engine exists.
fn resolve_absolute_strength_index_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(ASI_ID)
        .ok_or_else(|| format!("{ASI_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ASI_PARAMETER_KEYS {
        return Err(format!(
            "{ASI_ID}: CUDA all-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            ASI_PARAMETER_KEYS
        ));
    }

    let values = match parameters {
        ClassicCudaParameters::Defaults { .. } => {
            let mut values = Vec::with_capacity(ASI_PARAMETER_KEYS.len());
            for key in ASI_PARAMETER_KEYS {
                let declared = info
                    .params
                    .iter()
                    .find(|param| param.key == key)
                    .expect("declared key equality proved above");
                let value = match declared.default {
                    Some(ParamValueStatic::Int(value)) => value,
                    other => {
                        return Err(format!(
                            "{ASI_ID}.{key}: expected an integer registry default, found {other:?}"
                        ));
                    }
                };
                values.push(positive_usize_parameter(ASI_ID, key, value)?);
            }
            values
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ASI_PARAMETER_KEYS {
                return Err(format!(
                    "{ASI_ID}: canonical sweep must override exactly {:?}, found {override_keys:?}",
                    ASI_PARAMETER_KEYS
                ));
            }
            overrides
                .iter()
                .map(|(key, value)| positive_usize_parameter(ASI_ID, key, *value))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ASI_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    let [ema_length, signal_length] = values.as_slice() else {
        return Err(format!(
            "{ASI_ID}: expected two resolved integer parameters, found {}",
            values.len()
        ));
    };
    let expected_anchor = (*ema_length).max(*signal_length);
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == expected_anchor => {}
        other => {
            return Err(format!(
                "{ASI_ID}: resolved tuple ({ema_length}, {signal_length}) requires anchor \
                 {expected_anchor}, found {other:?}"
            ));
        }
    }
    Ok((*ema_length, *signal_length))
}

/// Resolve the one exact no-window f64 tuple consumed by both canonical
/// Adaptive Bandpass outputs. The registry values are compared by bits so a
/// schema/default/bound drift cannot silently change the production formula.
fn resolve_adaptive_bandpass_trigger_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(u64, u64), String> {
    let info = get_indicator(ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID).ok_or_else(|| {
        format!("{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_PARAMETER_KEYS {
        return Err(format!(
            "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: CUDA all-output ABI requires exact \
             registry parameters {:?}, found {declared_keys:?}",
            ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_PARAMETER_KEYS
        ));
    }

    let mut value_bits = [0_u64; ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_PARAMETER_KEYS.len()];
    for (index, key) in ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_PARAMETER_KEYS
        .iter()
        .copied()
        .enumerate()
    {
        let declared = &info.params[index];
        if declared.kind != IndicatorParamKind::Float || declared.required {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}.{key}: expected an optional f64, \
                 found kind={:?} required={}",
                declared.kind, declared.required
            ));
        }
        let value = match declared.default {
            Some(ParamValueStatic::Float(value)) if value.is_finite() => value,
            other => {
                return Err(format!(
                    "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}.{key}: expected a finite f64 \
                     registry default, found {other:?}"
                ));
            }
        };
        if value.to_bits() != ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_DEFAULT_BITS[index] {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}.{key}: registry default bits \
                 {:#018x} != reviewed CUDA tuple bits {:#018x}",
                value.to_bits(),
                ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_DEFAULT_BITS[index]
            ));
        }
        let Some(minimum) = declared.min else {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}.{key}: missing exact lower bound"
            ));
        };
        let Some(maximum) = declared.max else {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}.{key}: missing exact upper bound"
            ));
        };
        if minimum.to_bits() != ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_MIN_BITS[index]
            || maximum.to_bits() != ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_MAX_BITS[index]
            || !(minimum > 0.0 && minimum <= value && value <= maximum && maximum < 1.0)
        {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}.{key}: registry default/bounds \
                 ({value}, {minimum}..={maximum}) differ from the reviewed open-unit-interval \
                 contract"
            ));
        }
        value_bits[index] = value.to_bits();
    }

    match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => {}
        ClassicCudaParameters::Defaults { anchor, .. } => {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: no-window defaults require anchor 1 \
                 and a period-invariant primary contract, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::Swept { .. } => {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: delta/alpha are formula coefficients, \
                 not a canonical integer period sweep"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: discrete parameters cannot enter an \
                 f64 all-output route"
            ));
        }
    }

    Ok((value_bits[0], value_bits[1]))
}

/// Resolve the exact mixed integer/f64 point used by the canonical CPU batch
/// arm. Extended search changes only `rsi_length`; `alpha` remains the exact
/// registry default and is stored by bits so typed preflight equality cannot
/// blur a float parameter.
fn resolve_adaptive_bounds_rsi_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64), String> {
    let info = get_indicator(ADAPTIVE_BOUNDS_RSI_ID)
        .ok_or_else(|| format!("{ADAPTIVE_BOUNDS_RSI_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ADAPTIVE_BOUNDS_RSI_PARAMETER_KEYS {
        return Err(format!(
            "{ADAPTIVE_BOUNDS_RSI_ID}: CUDA all-output ABI requires exact registry parameters \
             {:?}, found {declared_keys:?}",
            ADAPTIVE_BOUNDS_RSI_PARAMETER_KEYS
        ));
    }

    let rsi_default = match info.params[0].default {
        Some(ParamValueStatic::Int(value)) => {
            positive_usize_parameter(ADAPTIVE_BOUNDS_RSI_ID, "rsi_length", value)?
        }
        other => {
            return Err(format!(
                "{ADAPTIVE_BOUNDS_RSI_ID}.rsi_length: expected an integer registry default, \
                 found {other:?}"
            ));
        }
    };
    let alpha = match info.params[1].default {
        Some(ParamValueStatic::Float(value)) if value.is_finite() => value,
        other => {
            return Err(format!(
                "{ADAPTIVE_BOUNDS_RSI_ID}.alpha: expected a finite f64 registry default, found \
                 {other:?}"
            ));
        }
    };
    if info.params[1].min.is_some_and(|minimum| alpha < minimum)
        || info.params[1].max.is_some_and(|maximum| alpha > maximum)
    {
        return Err(format!(
            "{ADAPTIVE_BOUNDS_RSI_ID}.alpha: registry default {alpha} is outside declared bounds \
             {:?}..={:?}",
            info.params[1].min, info.params[1].max
        ));
    }

    let rsi_length = match parameters {
        ClassicCudaParameters::Defaults { .. } => rsi_default,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let [(key, value)] = overrides.as_slice() else {
                return Err(format!(
                    "{ADAPTIVE_BOUNDS_RSI_ID}: canonical sweep must override exactly \
                     [`rsi_length`], found {overrides:?}"
                ));
            };
            if *key != "rsi_length" {
                return Err(format!(
                    "{ADAPTIVE_BOUNDS_RSI_ID}: canonical sweep expected `rsi_length`, found `{key}`"
                ));
            }
            positive_usize_parameter(ADAPTIVE_BOUNDS_RSI_ID, key, *value)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ADAPTIVE_BOUNDS_RSI_ID}: discrete parameters cannot enter an f64 all-output \
                 route"
            ));
        }
    };
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == rsi_length => {}
        other => {
            return Err(format!(
                "{ADAPTIVE_BOUNDS_RSI_ID}: resolved rsi_length {rsi_length} requires the same \
                 anchor, found {other:?}"
            ));
        }
    }
    Ok((rsi_length, alpha.to_bits()))
}

/// Resolve the canonical four-dimensional Adaptive MACD point. The extended
/// feature sweep changes only `length`; the other three integer windows stay
/// at their exact registry defaults, matching the CPU batch request.
fn resolve_adaptive_macd_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize, usize), String> {
    let info = get_indicator(ADAPTIVE_MACD_ID)
        .ok_or_else(|| format!("{ADAPTIVE_MACD_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ADAPTIVE_MACD_PARAMETER_KEYS {
        return Err(format!(
            "{ADAPTIVE_MACD_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            ADAPTIVE_MACD_PARAMETER_KEYS
        ));
    }

    let mut defaults = Vec::with_capacity(ADAPTIVE_MACD_PARAMETER_KEYS.len());
    for (index, key) in ADAPTIVE_MACD_PARAMETER_KEYS.iter().copied().enumerate() {
        let declared = &info.params[index];
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) => value,
            other => {
                return Err(format!(
                    "{ADAPTIVE_MACD_ID}.{key}: expected an integer registry default, found \
                     {other:?}"
                ));
            }
        };
        if declared.min.is_some_and(|minimum| (raw as f64) < minimum)
            || declared.max.is_some_and(|maximum| (raw as f64) > maximum)
        {
            return Err(format!(
                "{ADAPTIVE_MACD_ID}.{key}: registry default {raw} is outside declared bounds \
                 {:?}..={:?}",
                declared.min, declared.max
            ));
        }
        let value = positive_usize_parameter(ADAPTIVE_MACD_ID, key, raw)?;
        i32::try_from(value).map_err(|_| {
            format!("{ADAPTIVE_MACD_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
        defaults.push(value);
    }

    let values = match parameters {
        ClassicCudaParameters::Defaults { .. } => defaults,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let [(key, raw)] = overrides.as_slice() else {
                return Err(format!(
                    "{ADAPTIVE_MACD_ID}: canonical sweep must override exactly [`length`], \
                     found {overrides:?}"
                ));
            };
            if *key != "length" {
                return Err(format!(
                    "{ADAPTIVE_MACD_ID}: canonical sweep expected `length`, found `{key}`"
                ));
            }
            let length = positive_usize_parameter(ADAPTIVE_MACD_ID, key, *raw)?;
            let declared = &info.params[0];
            if declared
                .min
                .is_some_and(|minimum| (length as f64) < minimum)
                || declared
                    .max
                    .is_some_and(|maximum| (length as f64) > maximum)
            {
                return Err(format!(
                    "{ADAPTIVE_MACD_ID}.length: swept value {length} is outside declared bounds \
                     {:?}..={:?}",
                    declared.min, declared.max
                ));
            }
            i32::try_from(length).map_err(|_| {
                format!("{ADAPTIVE_MACD_ID}.length: value {length} exceeds the CUDA i32 ABI")
            })?;
            let mut swept = defaults;
            swept[0] = length;
            swept
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ADAPTIVE_MACD_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    let [length, fast_period, slow_period, signal_period] = values.as_slice() else {
        return Err(format!(
            "{ADAPTIVE_MACD_ID}: expected four resolved integer parameters, found {}",
            values.len()
        ));
    };
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == *length => {}
        other => {
            return Err(format!(
                "{ADAPTIVE_MACD_ID}: resolved length {length} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((*length, *fast_period, *slow_period, *signal_period))
}

/// Resolve Adaptive Momentum Oscillator's two computational windows while
/// proving the registry's redundant output selector still names the same two
/// canonical outputs handled by the typed all-output launch.
fn resolve_adaptive_momentum_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(ADAPTIVE_MOMENTUM_OSCILLATOR_ID).ok_or_else(|| {
        format!("{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ADAPTIVE_MOMENTUM_OSCILLATOR_PARAMETER_KEYS {
        return Err(format!(
            "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: CUDA all-output ABI requires exact registry \
             parameters {:?}, found {declared_keys:?}",
            ADAPTIVE_MOMENTUM_OSCILLATOR_PARAMETER_KEYS
        ));
    }

    let mut defaults = Vec::with_capacity(2);
    for (index, key) in ADAPTIVE_MOMENTUM_OSCILLATOR_PARAMETER_KEYS[..2]
        .iter()
        .copied()
        .enumerate()
    {
        let declared = &info.params[index];
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) => value,
            other => {
                return Err(format!(
                    "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.{key}: expected an integer registry \
                     default, found {other:?}"
                ));
            }
        };
        if declared.min.is_some_and(|minimum| (raw as f64) < minimum)
            || declared.max.is_some_and(|maximum| (raw as f64) > maximum)
        {
            return Err(format!(
                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.{key}: registry default {raw} is outside \
                 declared bounds {:?}..={:?}",
                declared.min, declared.max
            ));
        }
        let value = positive_usize_parameter(ADAPTIVE_MOMENTUM_OSCILLATOR_ID, key, raw)?;
        i32::try_from(value).map_err(|_| {
            format!(
                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI"
            )
        })?;
        defaults.push(value);
    }

    let output_selector = &info.params[2];
    match output_selector.default {
        Some(ParamValueStatic::EnumString(value))
            if value == ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS[0] => {}
        other => {
            return Err(format!(
                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.output: expected canonical primary default \
                 `{}`, found {other:?}",
                ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS[0]
            ));
        }
    }
    if output_selector.enum_values != ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS.as_slice() {
        return Err(format!(
            "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.output: exact selector vocabulary {:?} != {:?}",
            output_selector.enum_values, ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS
        ));
    }

    let values = match parameters {
        ClassicCudaParameters::Defaults { .. } => defaults,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let [(key, raw)] = overrides.as_slice() else {
                return Err(format!(
                    "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: canonical sweep must override exactly \
                     [`length`], found {overrides:?}"
                ));
            };
            if *key != "length" {
                return Err(format!(
                    "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: canonical sweep expected `length`, found \
                     `{key}`"
                ));
            }
            let length = positive_usize_parameter(ADAPTIVE_MOMENTUM_OSCILLATOR_ID, key, *raw)?;
            let declared = &info.params[0];
            if declared
                .min
                .is_some_and(|minimum| (length as f64) < minimum)
                || declared
                    .max
                    .is_some_and(|maximum| (length as f64) > maximum)
            {
                return Err(format!(
                    "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.length: swept value {length} is outside \
                     declared bounds {:?}..={:?}",
                    declared.min, declared.max
                ));
            }
            i32::try_from(length).map_err(|_| {
                format!(
                    "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}.length: value {length} exceeds the CUDA \
                     i32 ABI"
                )
            })?;
            let mut swept = defaults;
            swept[0] = length;
            swept
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: discrete parameters cannot enter an f64 \
                 all-output route"
            ));
        }
    };
    let [length, smoothing_length] = values.as_slice() else {
        return Err(format!(
            "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: expected two resolved integer parameters, found \
             {}",
            values.len()
        ));
    };
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == *length => {}
        other => {
            return Err(format!(
                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: resolved length {length} requires the same \
                 anchor, found {other:?}"
            ));
        }
    }
    Ok((*length, *smoothing_length))
}

/// Resolve ASTC's exact mixed four-window/f64 parameter point. The canonical
/// extended sweep scales all four integer windows as one registry ratio while
/// preserving the smoothing factor by exact bits.
fn resolve_adaptive_schaff_trend_cycle_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, u64, usize, usize), String> {
    let info = get_indicator(ADAPTIVE_SCHAFF_TREND_CYCLE_ID).ok_or_else(|| {
        format!("{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ADAPTIVE_SCHAFF_TREND_CYCLE_PARAMETER_KEYS {
        return Err(format!(
            "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: CUDA all-output ABI requires exact registry \
             parameters {:?}, found {declared_keys:?}",
            ADAPTIVE_SCHAFF_TREND_CYCLE_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS {
        return Err(format!(
            "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: canonical output schema {:?} != \
             {declared_outputs:?}",
            ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS
        ));
    }

    let mut default_windows = [0usize; ADAPTIVE_SCHAFF_TREND_CYCLE_WINDOW_KEYS.len()];
    for (slot, key) in ADAPTIVE_SCHAFF_TREND_CYCLE_WINDOW_KEYS
        .iter()
        .copied()
        .enumerate()
    {
        let declared = info
            .params
            .iter()
            .find(|param| param.key == key)
            .expect("exact registry key equality proved above");
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) => value,
            other => {
                return Err(format!(
                    "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.{key}: expected an integer registry \
                     default, found {other:?}"
                ));
            }
        };
        if declared.min.is_some_and(|minimum| (raw as f64) < minimum)
            || declared.max.is_some_and(|maximum| (raw as f64) > maximum)
        {
            return Err(format!(
                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.{key}: registry default {raw} is outside \
                 declared bounds {:?}..={:?}",
                declared.min, declared.max
            ));
        }
        let value = positive_usize_parameter(ADAPTIVE_SCHAFF_TREND_CYCLE_ID, key, raw)?;
        i32::try_from(value).map_err(|_| {
            format!(
                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.{key}: value {value} exceeds the CUDA i32 ABI"
            )
        })?;
        default_windows[slot] = value;
    }

    let smoothing = &info.params[2];
    let smoothing_factor = match smoothing.default {
        Some(ParamValueStatic::Float(value))
            if value.is_finite() && value > 0.0 && value <= 1.0 =>
        {
            value
        }
        other => {
            return Err(format!(
                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.smoothing_factor: expected a finite registry \
                 default in (0, 1], found {other:?}"
            ));
        }
    };
    if smoothing
        .min
        .is_some_and(|minimum| smoothing_factor < minimum)
        || smoothing
            .max
            .is_some_and(|maximum| smoothing_factor > maximum)
    {
        return Err(format!(
            "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.smoothing_factor: registry default \
             {smoothing_factor} is outside declared bounds {:?}..={:?}",
            smoothing.min, smoothing.max
        ));
    }

    let windows = match parameters {
        ClassicCudaParameters::Defaults { .. } => default_windows,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ADAPTIVE_SCHAFF_TREND_CYCLE_WINDOW_KEYS {
                return Err(format!(
                    "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: canonical sweep must override exactly \
                     {:?}, found {override_keys:?}",
                    ADAPTIVE_SCHAFF_TREND_CYCLE_WINDOW_KEYS
                ));
            }
            let mut swept = default_windows;
            for (slot, (key, raw)) in overrides.iter().enumerate() {
                let value = positive_usize_parameter(ADAPTIVE_SCHAFF_TREND_CYCLE_ID, key, *raw)?;
                let declared = info
                    .params
                    .iter()
                    .find(|param| param.key == *key)
                    .expect("exact sweep key equality proved above");
                if declared.min.is_some_and(|minimum| (value as f64) < minimum)
                    || declared.max.is_some_and(|maximum| (value as f64) > maximum)
                {
                    return Err(format!(
                        "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.{key}: swept value {value} is outside \
                         declared bounds {:?}..={:?}",
                        declared.min, declared.max
                    ));
                }
                i32::try_from(value).map_err(|_| {
                    format!(
                        "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}.{key}: value {value} exceeds the CUDA \
                         i32 ABI"
                    )
                })?;
                swept[slot] = value;
            }
            swept
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: discrete parameters cannot enter an f64 \
                 all-output route"
            ));
        }
    };
    let [adaptive_length, stc_length, fast_length, slow_length] = windows;
    let expected_anchor = windows.into_iter().max().expect("four ASTC windows");
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == expected_anchor => {}
        other => {
            return Err(format!(
                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: resolved windows {windows:?} require anchor \
                 {expected_anchor}, found {other:?}"
            ));
        }
    }
    Ok((
        adaptive_length,
        stc_length,
        smoothing_factor.to_bits(),
        fast_length,
        slow_length,
    ))
}

/// Resolve the exact Adjustable MA parameter point while keeping the reviewed
/// `smoothed_close == ma` production exclusion separate from the full kernel
/// ABI. Only `length` is sweepable; the three continuous shape parameters are
/// the registry defaults and cross the typed boundary by exact bits.
fn resolve_adjustable_ma_alternating_extremities_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, u64, u64), String> {
    let info = get_indicator(ADJUSTABLE_MA_ID)
        .ok_or_else(|| format!("{ADJUSTABLE_MA_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ADJUSTABLE_MA_PARAMETER_KEYS {
        return Err(format!(
            "{ADJUSTABLE_MA_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            ADJUSTABLE_MA_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ADJUSTABLE_MA_FULL_OUTPUT_IDS {
        return Err(format!(
            "{ADJUSTABLE_MA_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ADJUSTABLE_MA_FULL_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(ADJUSTABLE_MA_ID);
    let expected_planned_outputs = ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ADJUSTABLE_MA_ID}: reviewed admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let length_param = &info.params[0];
    if !matches!(&length_param.kind, IndicatorParamKind::Int) || length_param.required {
        return Err(format!(
            "{ADJUSTABLE_MA_ID}.length: expected an optional integer registry parameter, found \
             kind={:?} required={}",
            length_param.kind, length_param.required
        ));
    }
    let default_length_raw = match length_param.default {
        Some(ParamValueStatic::Int(value)) => value,
        other => {
            return Err(format!(
                "{ADJUSTABLE_MA_ID}.length: expected an integer registry default, found \
                 {other:?}"
            ));
        }
    };
    let default_length = positive_usize_parameter(ADJUSTABLE_MA_ID, "length", default_length_raw)?;
    if default_length < 2
        || length_param
            .min
            .is_some_and(|minimum| (default_length as f64) < minimum)
        || length_param
            .max
            .is_some_and(|maximum| (default_length as f64) > maximum)
    {
        return Err(format!(
            "{ADJUSTABLE_MA_ID}.length: registry default {default_length} violates formula or \
             declared bounds {:?}..={:?}",
            length_param.min, length_param.max
        ));
    }
    i32::try_from(default_length).map_err(|_| {
        format!("{ADJUSTABLE_MA_ID}.length: value {default_length} exceeds the CUDA i32 ABI")
    })?;

    let resolve_float_default = |index: usize,
                                 key: &'static str,
                                 formula_minimum: f64|
     -> std::result::Result<f64, String> {
        let declared = &info.params[index];
        if declared.key != key
            || !matches!(&declared.kind, IndicatorParamKind::Float)
            || declared.required
        {
            return Err(format!(
                "{ADJUSTABLE_MA_ID}.{key}: expected an optional float at registry slot \
                     {index}, found key={} kind={:?} required={}",
                declared.key, declared.kind, declared.required
            ));
        }
        let value = match declared.default {
            Some(ParamValueStatic::Float(value)) if value.is_finite() => value,
            other => {
                return Err(format!(
                    "{ADJUSTABLE_MA_ID}.{key}: expected a finite float registry default, \
                         found {other:?}"
                ));
            }
        };
        if value < formula_minimum
            || declared.min.is_some_and(|minimum| value < minimum)
            || declared.max.is_some_and(|maximum| value > maximum)
        {
            return Err(format!(
                "{ADJUSTABLE_MA_ID}.{key}: registry default {value} violates formula \
                     minimum {formula_minimum} or declared bounds {:?}..={:?}",
                declared.min, declared.max
            ));
        }
        Ok(value)
    };
    let mult = resolve_float_default(1, "mult", 1.0)?;
    let alpha = resolve_float_default(2, "alpha", 0.0)?;
    let beta = resolve_float_default(3, "beta", 0.0)?;

    let length = match parameters {
        ClassicCudaParameters::Defaults { .. } => default_length,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["length"] {
                return Err(format!(
                    "{ADJUSTABLE_MA_ID}: canonical sweep must override exactly [\"length\"], \
                     found {override_keys:?}"
                ));
            }
            let swept = positive_usize_parameter(ADJUSTABLE_MA_ID, "length", overrides[0].1)?;
            if swept < 2
                || length_param
                    .min
                    .is_some_and(|minimum| (swept as f64) < minimum)
                || length_param
                    .max
                    .is_some_and(|maximum| (swept as f64) > maximum)
            {
                return Err(format!(
                    "{ADJUSTABLE_MA_ID}.length: swept value {swept} violates formula or \
                     declared bounds {:?}..={:?}",
                    length_param.min, length_param.max
                ));
            }
            i32::try_from(swept).map_err(|_| {
                format!("{ADJUSTABLE_MA_ID}.length: value {swept} exceeds the CUDA i32 ABI")
            })?;
            swept
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ADJUSTABLE_MA_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == length => {}
        other => {
            return Err(format!(
                "{ADJUSTABLE_MA_ID}: resolved length {length} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((length, mult.to_bits(), alpha.to_bits(), beta.to_bits()))
}

/// Resolve Alligator's exact six-integer tuple. The extended sweep scales the
/// three period members as one registry ratio while every write offset remains
/// at its canonical default; changing an offset would shift the feature rather
/// than merely change its timescale.
fn resolve_alligator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize, usize, usize, usize), String> {
    let info = get_indicator(ALLIGATOR_ID)
        .ok_or_else(|| format!("{ALLIGATOR_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ALLIGATOR_PARAMETER_KEYS {
        return Err(format!(
            "{ALLIGATOR_ID}: CUDA all-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            ALLIGATOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ALLIGATOR_OUTPUT_IDS {
        return Err(format!(
            "{ALLIGATOR_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ALLIGATOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(ALLIGATOR_ID);
    let expected_planned_outputs = ALLIGATOR_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ALLIGATOR_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let mut defaults = [0usize; ALLIGATOR_PARAMETER_KEYS.len()];
    for (index, key) in ALLIGATOR_PARAMETER_KEYS.iter().copied().enumerate() {
        let declared = &info.params[index];
        if !matches!(&declared.kind, IndicatorParamKind::Int) || declared.required {
            return Err(format!(
                "{ALLIGATOR_ID}.{key}: expected an optional integer at registry slot {index}, \
                 found kind={:?} required={}",
                declared.kind, declared.required
            ));
        }
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) => value,
            other => {
                return Err(format!(
                    "{ALLIGATOR_ID}.{key}: expected an integer registry default, found {other:?}"
                ));
            }
        };
        let value = usize::try_from(raw).map_err(|_| {
            format!("{ALLIGATOR_ID}.{key}: expected a non-negative integer, found {raw}")
        })?;
        if key.ends_with("_period") && value == 0 {
            return Err(format!(
                "{ALLIGATOR_ID}.{key}: period must be positive, found {value}"
            ));
        }
        if declared.min.is_some_and(|minimum| (value as f64) < minimum)
            || declared.max.is_some_and(|maximum| (value as f64) > maximum)
        {
            return Err(format!(
                "{ALLIGATOR_ID}.{key}: registry default {value} is outside declared bounds \
                 {:?}..={:?}",
                declared.min, declared.max
            ));
        }
        i32::try_from(value)
            .map_err(|_| format!("{ALLIGATOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI"))?;
        defaults[index] = value;
    }

    let values = match parameters {
        ClassicCudaParameters::Defaults { .. } => defaults,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ALLIGATOR_SWEEP_PARAMETER_KEYS {
                return Err(format!(
                    "{ALLIGATOR_ID}: canonical ratio sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    ALLIGATOR_SWEEP_PARAMETER_KEYS
                ));
            }
            let mut swept = defaults;
            for (key, raw) in overrides {
                let index = ALLIGATOR_PARAMETER_KEYS
                    .iter()
                    .position(|candidate| candidate == key)
                    .expect("exact Alligator sweep-key equality proved above");
                let value = positive_usize_parameter(ALLIGATOR_ID, key, *raw)?;
                let declared = &info.params[index];
                if declared.min.is_some_and(|minimum| (value as f64) < minimum)
                    || declared.max.is_some_and(|maximum| (value as f64) > maximum)
                {
                    return Err(format!(
                        "{ALLIGATOR_ID}.{key}: swept value {value} is outside declared bounds \
                         {:?}..={:?}",
                        declared.min, declared.max
                    ));
                }
                i32::try_from(value).map_err(|_| {
                    format!("{ALLIGATOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
                })?;
                swept[index] = value;
            }
            swept
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ALLIGATOR_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    let [
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
    ] = values;
    let expected_anchor = jaw_period.max(teeth_period).max(lips_period);
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == expected_anchor => {}
        other => {
            return Err(format!(
                "{ALLIGATOR_ID}: resolved periods [{jaw_period}, {teeth_period}, {lips_period}] \
                 require anchor {expected_anchor}, found {other:?}"
            ));
        }
    }
    Ok((
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
    ))
}

/// Resolve AlphaTrend's exact production tuple. The f64 kernel implements the
/// canonical volume-backed MFI branch only: coeff and no_volume stay pinned to
/// their registry defaults while the extended plan may override period alone.
fn resolve_alphatrend_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(u64, usize, bool), String> {
    let info = get_indicator(ALPHATREND_ID)
        .ok_or_else(|| format!("{ALPHATREND_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != ALPHATREND_PARAMETER_KEYS {
        return Err(format!(
            "{ALPHATREND_ID}: CUDA all-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            ALPHATREND_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ALPHATREND_OUTPUT_IDS {
        return Err(format!(
            "{ALPHATREND_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ALPHATREND_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(ALPHATREND_ID);
    let expected_planned_outputs = ALPHATREND_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ALPHATREND_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let coeff_param = &info.params[0];
    if !matches!(&coeff_param.kind, IndicatorParamKind::Float) || coeff_param.required {
        return Err(format!(
            "{ALPHATREND_ID}.coeff: expected an optional float, found kind={:?} required={}",
            coeff_param.kind, coeff_param.required
        ));
    }
    let coeff = match coeff_param.default {
        Some(ParamValueStatic::Float(value)) if value.is_finite() && value > 0.0 => value,
        other => {
            return Err(format!(
                "{ALPHATREND_ID}.coeff: expected a finite positive float default, found \
                 {other:?}"
            ));
        }
    };
    if coeff.to_bits() != 1.0_f64.to_bits() {
        return Err(format!(
            "{ALPHATREND_ID}.coeff: f64 kernel contract requires exact default 1.0, found \
             {coeff:?}"
        ));
    }
    if coeff_param.min.is_some_and(|minimum| coeff < minimum)
        || coeff_param.max.is_some_and(|maximum| coeff > maximum)
    {
        return Err(format!(
            "{ALPHATREND_ID}.coeff: registry default {coeff} is outside declared bounds \
             {:?}..={:?}",
            coeff_param.min, coeff_param.max
        ));
    }

    let period_param = &info.params[1];
    if !matches!(&period_param.kind, IndicatorParamKind::Int) || period_param.required {
        return Err(format!(
            "{ALPHATREND_ID}.period: expected an optional integer, found kind={:?} required={}",
            period_param.kind, period_param.required
        ));
    }
    let default_period = match period_param.default {
        Some(ParamValueStatic::Int(value)) => {
            positive_usize_parameter(ALPHATREND_ID, "period", value)?
        }
        other => {
            return Err(format!(
                "{ALPHATREND_ID}.period: expected an integer registry default, found {other:?}"
            ));
        }
    };
    if default_period != 14 {
        return Err(format!(
            "{ALPHATREND_ID}.period: canonical default must remain 14, found {default_period}"
        ));
    }

    let no_volume_param = &info.params[2];
    if !matches!(&no_volume_param.kind, IndicatorParamKind::Bool) || no_volume_param.required {
        return Err(format!(
            "{ALPHATREND_ID}.no_volume: expected an optional bool, found kind={:?} required={}",
            no_volume_param.kind, no_volume_param.required
        ));
    }
    let no_volume = match no_volume_param.default {
        Some(ParamValueStatic::Bool(value)) => value,
        other => {
            return Err(format!(
                "{ALPHATREND_ID}.no_volume: expected a bool registry default, found {other:?}"
            ));
        }
    };
    if no_volume {
        return Err(format!(
            "{ALPHATREND_ID}.no_volume: f64 production route requires the canonical false/MFI \
             branch"
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults { .. } => default_period,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ALPHATREND_SWEEP_PARAMETER_KEYS {
                return Err(format!(
                    "{ALPHATREND_ID}: canonical sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    ALPHATREND_SWEEP_PARAMETER_KEYS
                ));
            }
            positive_usize_parameter(ALPHATREND_ID, "period", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ALPHATREND_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    if period_param
        .min
        .is_some_and(|minimum| (period as f64) < minimum)
        || period_param
            .max
            .is_some_and(|maximum| (period as f64) > maximum)
        || period > ALPHATREND_MAX_PERIOD
    {
        return Err(format!(
            "{ALPHATREND_ID}.period: resolved value {period} violates declared bounds \
             {:?}..={:?} or compiled max {ALPHATREND_MAX_PERIOD}",
            period_param.min, period_param.max
        ));
    }
    i32::try_from(period)
        .map_err(|_| format!("{ALPHATREND_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{ALPHATREND_ID}: resolved period {period} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }

    Ok((coeff.to_bits(), period, no_volume))
}

/// Resolve ACOSC's single canonical no-parameter point. The inert primary-ABI
/// anchor is admitted only because the registered kernel is period-invariant;
/// neither a period sweep nor an undisclosed parameter may enter this route.
fn resolve_acosc_parameters(parameters: &ClassicCudaParameters) -> std::result::Result<(), String> {
    let info = get_indicator(ACOSC_ID)
        .ok_or_else(|| format!("{ACOSC_ID}: absent from the vector-ta registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ACOSC_OUTPUT_IDS {
        return Err(format!(
            "{ACOSC_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ACOSC_OUTPUT_IDS
        ));
    }
    if !info.params.is_empty() {
        let declared_keys = info
            .params
            .iter()
            .map(|parameter| parameter.key)
            .collect::<Vec<_>>();
        return Err(format!(
            "{ACOSC_ID}: CUDA route requires an empty parameter tuple, found {declared_keys:?}"
        ));
    }
    let planned_outputs = output_ids_for(ACOSC_ID);
    let expected_planned_outputs = ACOSC_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ACOSC_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => Ok(()),
        ClassicCudaParameters::Defaults { .. } => Err(format!(
            "{ACOSC_ID}: no-parameter route requires inert anchor 1 and a period-invariant kernel, \
             found {parameters:?}"
        )),
        ClassicCudaParameters::Swept { .. } => Err(format!(
            "{ACOSC_ID}: canonical registry declares no period sweep"
        )),
        ClassicCudaParameters::DiscreteDefaults => Err(format!(
            "{ACOSC_ID}: discrete parameters cannot enter an f64 all-output route"
        )),
    }
}

/// Resolve the canonical two-window Andean Oscillator tuple. Production's
/// one-dimensional extended sweep changes only `length`; `signal_length`
/// remains the exact registry default and is never inferred from the anchor.
fn resolve_andean_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(ANDEAN_OSCILLATOR_ID)
        .ok_or_else(|| format!("{ANDEAN_OSCILLATOR_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != ANDEAN_OSCILLATOR_PARAMETER_KEYS {
        return Err(format!(
            "{ANDEAN_OSCILLATOR_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            ANDEAN_OSCILLATOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ANDEAN_OSCILLATOR_OUTPUT_IDS {
        return Err(format!(
            "{ANDEAN_OSCILLATOR_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ANDEAN_OSCILLATOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(ANDEAN_OSCILLATOR_ID);
    let expected_planned_outputs = ANDEAN_OSCILLATOR_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ANDEAN_OSCILLATOR_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let expected_defaults = [50_i64, 9_i64];
    let mut defaults = [0usize; ANDEAN_OSCILLATOR_PARAMETER_KEYS.len()];
    for (index, (key, expected_default)) in ANDEAN_OSCILLATOR_PARAMETER_KEYS
        .iter()
        .copied()
        .zip(expected_defaults)
        .enumerate()
    {
        let declared = &info.params[index];
        if !matches!(&declared.kind, IndicatorParamKind::Int)
            || declared.required
            || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || declared.max.is_some()
        {
            return Err(format!(
                "{ANDEAN_OSCILLATOR_ID}.{key}: expected optional integer bounds 1..=unbounded at \
                 registry slot {index}, found kind={:?} required={} bounds={:?}..={:?}",
                declared.kind, declared.required, declared.min, declared.max
            ));
        }
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) if value == expected_default => value,
            other => {
                return Err(format!(
                    "{ANDEAN_OSCILLATOR_ID}.{key}: expected exact registry default \
                     {expected_default}, found {other:?}"
                ));
            }
        };
        let value = positive_usize_parameter(ANDEAN_OSCILLATOR_ID, key, raw)?;
        i32::try_from(value).map_err(|_| {
            format!("{ANDEAN_OSCILLATOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
        defaults[index] = value;
    }

    let [length, signal_length] = match parameters {
        ClassicCudaParameters::Defaults { .. } => defaults,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["length"] {
                return Err(format!(
                    "{ANDEAN_OSCILLATOR_ID}: canonical sweep must override exactly [\"length\"], \
                     found {override_keys:?}"
                ));
            }
            let length = positive_usize_parameter(ANDEAN_OSCILLATOR_ID, "length", overrides[0].1)?;
            i32::try_from(length).map_err(|_| {
                format!("{ANDEAN_OSCILLATOR_ID}.length: value {length} exceeds the CUDA i32 ABI")
            })?;
            [length, defaults[1]]
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ANDEAN_OSCILLATOR_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == length => {}
        other => {
            return Err(format!(
                "{ANDEAN_OSCILLATOR_ID}: resolved length {length} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((length, signal_length))
}

/// Resolve Aroon's sole canonical length. Both published extrema consume the
/// same length+1 window, so a default or swept tuple is admitted only when the
/// registry identity, default, bounds, and planning anchor all agree exactly.
fn resolve_aroon_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(AROON_ID)
        .ok_or_else(|| format!("{AROON_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != AROON_PARAMETER_KEYS {
        return Err(format!(
            "{AROON_ID}: CUDA all-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            AROON_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != AROON_OUTPUT_IDS {
        return Err(format!(
            "{AROON_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            AROON_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(AROON_ID);
    let expected_planned_outputs = AROON_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{AROON_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let declared = &info.params[0];
    if !matches!(&declared.kind, IndicatorParamKind::Int)
        || declared.required
        || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || declared.max.is_some()
    {
        return Err(format!(
            "{AROON_ID}.length: expected an optional integer bounded 1..=unbounded, found \
             kind={:?} required={} bounds={:?}..={:?}",
            declared.kind, declared.required, declared.min, declared.max
        ));
    }
    let default_length = match declared.default {
        Some(ParamValueStatic::Int(14)) => 14usize,
        other => {
            return Err(format!(
                "{AROON_ID}.length: expected exact registry default 14, found {other:?}"
            ));
        }
    };

    let length = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => default_length,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{AROON_ID}: length-bearing default route cannot require a period-invariant \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["length"] {
                return Err(format!(
                    "{AROON_ID}: canonical sweep must override exactly [\"length\"], found \
                     {override_keys:?}"
                ));
            }
            positive_usize_parameter(AROON_ID, "length", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{AROON_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    i32::try_from(length)
        .map_err(|_| format!("{AROON_ID}.length: value {length} exceeds the CUDA i32 ABI"))?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == length => {}
        other => {
            return Err(format!(
                "{AROON_ID}: resolved length {length} requires the same anchor, found {other:?}"
            ));
        }
    }
    Ok(length)
}

/// Resolve ASO's exact `(period, mode)` tuple. Production's one-dimensional
/// extended sweep changes only `period`; `mode` remains the canonical zero
/// default, while the registry's complete 0..=2 domain is validated so schema
/// drift cannot silently change the full-output CUDA ABI.
fn resolve_aso_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(ASO_ID)
        .ok_or_else(|| format!("{ASO_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != ASO_PARAMETER_KEYS {
        return Err(format!(
            "{ASO_ID}: CUDA all-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            ASO_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ASO_OUTPUT_IDS {
        return Err(format!(
            "{ASO_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ASO_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(ASO_ID);
    let expected_planned_outputs = ASO_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ASO_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
    {
        return Err(format!(
            "{ASO_ID}.period: expected an optional integer bounded 1..=unbounded with step 1, \
             found kind={:?} required={} bounds={:?}..={:?} step={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step
        ));
    }
    let default_period = match period_parameter.default {
        Some(ParamValueStatic::Int(10)) => 10usize,
        other => {
            return Err(format!(
                "{ASO_ID}.period: expected exact registry default 10, found {other:?}"
            ));
        }
    };

    let mode_parameter = &info.params[1];
    if !matches!(&mode_parameter.kind, IndicatorParamKind::Int)
        || mode_parameter.required
        || mode_parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || mode_parameter.max.map(f64::to_bits) != Some(2.0_f64.to_bits())
        || mode_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
    {
        return Err(format!(
            "{ASO_ID}.mode: expected an optional integer bounded 0..=2 with step 1, found \
             kind={:?} required={} bounds={:?}..={:?} step={:?}",
            mode_parameter.kind,
            mode_parameter.required,
            mode_parameter.min,
            mode_parameter.max,
            mode_parameter.step
        ));
    }
    let default_mode = match mode_parameter.default {
        Some(ParamValueStatic::Int(0)) => 0usize,
        other => {
            return Err(format!(
                "{ASO_ID}.mode: expected exact registry default 0, found {other:?}"
            ));
        }
    };

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{ASO_ID}: period-bearing default route cannot require a period-invariant \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys.as_slice() != ["period"] {
                return Err(format!(
                    "{ASO_ID}: canonical sweep must override exactly [\"period\"], found \
                     {override_keys:?}"
                ));
            }
            positive_usize_parameter(ASO_ID, "period", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ASO_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{ASO_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    i32::try_from(default_mode)
        .map_err(|_| format!("{ASO_ID}.mode: value {default_mode} exceeds the CUDA i32 ABI"))?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{ASO_ID}: resolved period {period} requires the same anchor, found {other:?}"
            ));
        }
    }
    Ok((period, default_mode))
}

/// Resolve Autocorrelation Indicator's exact selected-output tuple. The
/// production sweep changes only `length`; canonical lag 1 and the real-input
/// (`use_test_signal=false`) mode remain fixed so the two published matrices
/// come from one auditable f64 launch rather than the standalone all-lag API.
fn resolve_autocorrelation_indicator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, bool), String> {
    let info = get_indicator(AUTOCORRELATION_INDICATOR_ID).ok_or_else(|| {
        format!("{AUTOCORRELATION_INDICATOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != AUTOCORRELATION_INDICATOR_PARAMETER_KEYS {
        return Err(format!(
            "{AUTOCORRELATION_INDICATOR_ID}: CUDA all-output ABI requires exact registry \
             parameters {:?}, found {declared_keys:?}",
            AUTOCORRELATION_INDICATOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != AUTOCORRELATION_INDICATOR_OUTPUT_IDS {
        return Err(format!(
            "{AUTOCORRELATION_INDICATOR_ID}: selected-output CUDA ABI {:?} != registry \
             {declared_outputs:?}",
            AUTOCORRELATION_INDICATOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(AUTOCORRELATION_INDICATOR_ID);
    let expected_planned_outputs = AUTOCORRELATION_INDICATOR_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{AUTOCORRELATION_INDICATOR_ID}: admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let expected_integer_defaults = [20_i64, 1_i64];
    let mut integer_defaults = [0usize; 2];
    for (index, (key, expected_default)) in AUTOCORRELATION_INDICATOR_PARAMETER_KEYS[..2]
        .iter()
        .copied()
        .zip(expected_integer_defaults)
        .enumerate()
    {
        let declared = &info.params[index];
        if !matches!(&declared.kind, IndicatorParamKind::Int)
            || declared.required
            || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || declared.max.is_some()
            || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !declared.enum_values.is_empty()
        {
            return Err(format!(
                "{AUTOCORRELATION_INDICATOR_ID}.{key}: expected optional integer bounds \
                 1..=unbounded with step 1 and no enum values at registry slot {index}, found \
                 kind={:?} required={} bounds={:?}..={:?} step={:?} enum={:?}",
                declared.kind,
                declared.required,
                declared.min,
                declared.max,
                declared.step,
                declared.enum_values
            ));
        }
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) if value == expected_default => value,
            other => {
                return Err(format!(
                    "{AUTOCORRELATION_INDICATOR_ID}.{key}: expected exact registry default \
                     {expected_default}, found {other:?}"
                ));
            }
        };
        let value = positive_usize_parameter(AUTOCORRELATION_INDICATOR_ID, key, raw)?;
        i32::try_from(value).map_err(|_| {
            format!("{AUTOCORRELATION_INDICATOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
        integer_defaults[index] = value;
    }

    let test_signal = &info.params[2];
    if !matches!(&test_signal.kind, IndicatorParamKind::Bool)
        || test_signal.required
        || test_signal.min.is_some()
        || test_signal.max.is_some()
        || test_signal.step.is_some()
        || test_signal.enum_values != ["true", "false"]
    {
        return Err(format!(
            "{AUTOCORRELATION_INDICATOR_ID}.use_test_signal: expected optional bool with no \
             numeric bounds and exact true/false vocabulary, found kind={:?} required={} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            test_signal.kind,
            test_signal.required,
            test_signal.min,
            test_signal.max,
            test_signal.step,
            test_signal.enum_values
        ));
    }
    let use_test_signal = match test_signal.default {
        Some(ParamValueStatic::Bool(false)) => false,
        other => {
            return Err(format!(
                "{AUTOCORRELATION_INDICATOR_ID}.use_test_signal: expected exact registry \
                 default false, found {other:?}"
            ));
        }
    };

    let length = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => integer_defaults[0],
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{AUTOCORRELATION_INDICATOR_ID}: length-bearing default route cannot require a \
                 period-invariant kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["length"] {
                return Err(format!(
                    "{AUTOCORRELATION_INDICATOR_ID}: canonical sweep must override exactly \
                     [\"length\"], found {override_keys:?}"
                ));
            }
            positive_usize_parameter(AUTOCORRELATION_INDICATOR_ID, "length", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{AUTOCORRELATION_INDICATOR_ID}: discrete parameters cannot enter an f64 \
                 selected-output route"
            ));
        }
    };
    i32::try_from(length).map_err(|_| {
        format!("{AUTOCORRELATION_INDICATOR_ID}.length: value {length} exceeds the CUDA i32 ABI")
    })?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == length => {}
        other => {
            return Err(format!(
                "{AUTOCORRELATION_INDICATOR_ID}: resolved length {length} requires the same \
                 anchor, found {other:?}"
            ));
        }
    }
    Ok((length, integer_defaults[1], use_test_signal))
}

/// Resolve AVSL's canonical `(fast_period, slow_period, multiplier)` point.
/// The production sweep names `slow_period` as its timescale anchor and scales
/// `fast_period` with the exact 12:26 registry ratio using the same positive
/// half-up integer arithmetic as `hpc_ta`. Any schema, bound, tuple-order, or
/// ratio drift is rejected before the resident engine is constructed.
fn resolve_avsl_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, u64), String> {
    let info =
        get_indicator(AVSL_ID).ok_or_else(|| format!("{AVSL_ID}: absent from the registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != AVSL_PARAMETER_KEYS {
        return Err(format!(
            "{AVSL_ID}: CUDA production ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            AVSL_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != AVSL_OUTPUT_IDS {
        return Err(format!(
            "{AVSL_ID}: CUDA production output ABI {:?} != registry {declared_outputs:?}",
            AVSL_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(AVSL_ID);
    let expected_planned_outputs = AVSL_REQUESTED_OUTPUT_IDS.to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{AVSL_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let expected_integer_defaults = [12_i64, 26_i64];
    let mut integer_defaults = [0usize; 2];
    for (index, (key, expected_default)) in AVSL_PARAMETER_KEYS[..2]
        .iter()
        .copied()
        .zip(expected_integer_defaults)
        .enumerate()
    {
        let declared = &info.params[index];
        if !matches!(&declared.kind, IndicatorParamKind::Int)
            || declared.required
            || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || declared.max.is_some()
            || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !declared.enum_values.is_empty()
        {
            return Err(format!(
                "{AVSL_ID}.{key}: expected optional integer bounds 1..=unbounded with step 1 \
                 and no enum values at registry slot {index}, found kind={:?} required={} \
                 bounds={:?}..={:?} step={:?} enum={:?}",
                declared.kind,
                declared.required,
                declared.min,
                declared.max,
                declared.step,
                declared.enum_values
            ));
        }
        let raw = match declared.default {
            Some(ParamValueStatic::Int(value)) if value == expected_default => value,
            other => {
                return Err(format!(
                    "{AVSL_ID}.{key}: expected exact registry default {expected_default}, found \
                     {other:?}"
                ));
            }
        };
        let value = positive_usize_parameter(AVSL_ID, key, raw)?;
        i32::try_from(value)
            .map_err(|_| format!("{AVSL_ID}.{key}: value {value} exceeds the CUDA i32 ABI"))?;
        integer_defaults[index] = value;
    }

    let multiplier = &info.params[2];
    if !matches!(&multiplier.kind, IndicatorParamKind::Float)
        || multiplier.required
        || multiplier.min.map(f64::to_bits) != Some(0.1_f64.to_bits())
        || multiplier.max.is_some()
        || multiplier.step.map(f64::to_bits) != Some(0.1_f64.to_bits())
        || !multiplier.enum_values.is_empty()
    {
        return Err(format!(
            "{AVSL_ID}.multiplier: expected optional positive float bounds 0.1..=unbounded \
             with step 0.1 and no enum values, found kind={:?} required={} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            multiplier.kind,
            multiplier.required,
            multiplier.min,
            multiplier.max,
            multiplier.step,
            multiplier.enum_values
        ));
    }
    let multiplier_bits = match multiplier.default {
        Some(ParamValueStatic::Float(value)) if value.to_bits() == 2.0_f64.to_bits() => {
            value.to_bits()
        }
        other => {
            return Err(format!(
                "{AVSL_ID}.multiplier: expected exact registry default 2.0, found {other:?}"
            ));
        }
    };

    let (fast_period, slow_period) = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => (integer_defaults[0], integer_defaults[1]),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{AVSL_ID}: window-bearing default route cannot require a period-invariant \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["fast_period", "slow_period"] {
                return Err(format!(
                    "{AVSL_ID}: canonical sweep must override exactly \
                     [\"fast_period\", \"slow_period\"], found {override_keys:?}"
                ));
            }
            (
                positive_usize_parameter(AVSL_ID, "fast_period", overrides[0].1)?,
                positive_usize_parameter(AVSL_ID, "slow_period", overrides[1].1)?,
            )
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{AVSL_ID}: discrete parameters cannot enter an f64 production route"
            ));
        }
    };
    let expected_fast = slow_period
        .checked_mul(integer_defaults[0])
        .and_then(|scaled| scaled.checked_add(integer_defaults[1] / 2))
        .map(|scaled| scaled / integer_defaults[1])
        .ok_or_else(|| format!("{AVSL_ID}: fast/slow ratio scaling overflow"))?
        .max(1);
    if fast_period != expected_fast {
        return Err(format!(
            "{AVSL_ID}: slow anchor {slow_period} requires exact half-up 12:26 fast period \
             {expected_fast}, found {fast_period}"
        ));
    }
    i32::try_from(fast_period).map_err(|_| {
        format!("{AVSL_ID}.fast_period: value {fast_period} exceeds the CUDA i32 ABI")
    })?;
    i32::try_from(slow_period).map_err(|_| {
        format!("{AVSL_ID}.slow_period: value {slow_period} exceeds the CUDA i32 ABI")
    })?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == slow_period => {}
        other => {
            return Err(format!(
                "{AVSL_ID}: resolved tuple ({fast_period}, {slow_period}) requires slow anchor \
                 {slow_period}, found {other:?}"
            ));
        }
    }
    Ok((fast_period, slow_period, multiplier_bits))
}

/// Resolve Bandpass's exact `(period, bandwidth)` production point. The
/// canonical extended sweep changes only `period`; bandwidth remains the
/// registry's exact 0.3 f64 default and the full positive finite `(0, 1]`
/// domain is proved before a CUDA session exists.
fn resolve_bandpass_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64), String> {
    let info = get_indicator(BANDPASS_ID)
        .ok_or_else(|| format!("{BANDPASS_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != BANDPASS_PARAMETER_KEYS {
        return Err(format!(
            "{BANDPASS_ID}: CUDA all-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            BANDPASS_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != BANDPASS_OUTPUT_IDS {
        return Err(format!(
            "{BANDPASS_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            BANDPASS_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(BANDPASS_ID);
    let expected_planned_outputs = BANDPASS_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{BANDPASS_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{BANDPASS_ID}.period: expected an optional integer bounded 1..=unbounded with \
             step 1 and no enum values, found kind={:?} required={} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }
    let default_period = match period_parameter.default {
        Some(ParamValueStatic::Int(20)) => 20usize,
        other => {
            return Err(format!(
                "{BANDPASS_ID}.period: expected exact registry default 20, found {other:?}"
            ));
        }
    };

    let bandwidth_parameter = &info.params[1];
    if !matches!(&bandwidth_parameter.kind, IndicatorParamKind::Float)
        || bandwidth_parameter.required
        || bandwidth_parameter.min.map(f64::to_bits) != Some(f64::from_bits(1).to_bits())
        || bandwidth_parameter.max.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || bandwidth_parameter.step.is_some()
        || !bandwidth_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{BANDPASS_ID}.bandwidth: expected an optional finite float bounded by the exact \
             representable domain [f64::from_bits(1), 1.0] with no step/enum, found kind={:?} \
             required={} bounds={:?}..={:?} step={:?} enum={:?}",
            bandwidth_parameter.kind,
            bandwidth_parameter.required,
            bandwidth_parameter.min,
            bandwidth_parameter.max,
            bandwidth_parameter.step,
            bandwidth_parameter.enum_values
        ));
    }
    let bandwidth_bits = match bandwidth_parameter.default {
        Some(ParamValueStatic::Float(value)) if value.to_bits() == 0.3_f64.to_bits() => {
            value.to_bits()
        }
        other => {
            return Err(format!(
                "{BANDPASS_ID}.bandwidth: expected exact registry default 0.3, found {other:?}"
            ));
        }
    };

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{BANDPASS_ID}: period-bearing default route cannot require a period-invariant \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys.as_slice() != ["period"] {
                return Err(format!(
                    "{BANDPASS_ID}: canonical sweep must override exactly [\"period\"], found \
                     {override_keys:?}"
                ));
            }
            positive_usize_parameter(BANDPASS_ID, "period", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{BANDPASS_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{BANDPASS_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{BANDPASS_ID}: resolved period {period} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((period, bandwidth_bits))
}

/// Resolve the exact canonical Bollinger Bands point admitted by the feature
/// registry. `matype=sma` and `devtype=0` are the CPU dispatcher's existing
/// hidden defaults rather than admitted search parameters; this typed route
/// carries every registered value and fails closed if that schema changes.
fn resolve_bollinger_bands_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, u64), String> {
    let info = get_indicator(BOLLINGER_BANDS_ID)
        .ok_or_else(|| format!("{BOLLINGER_BANDS_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != BOLLINGER_BANDS_PARAMETER_KEYS {
        return Err(format!(
            "{BOLLINGER_BANDS_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            BOLLINGER_BANDS_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != BOLLINGER_BANDS_OUTPUT_IDS {
        return Err(format!(
            "{BOLLINGER_BANDS_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            BOLLINGER_BANDS_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(BOLLINGER_BANDS_ID);
    let expected_planned_outputs = BOLLINGER_BANDS_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{BOLLINGER_BANDS_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{BOLLINGER_BANDS_ID}.period: expected an optional integer bounded \
             1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }
    let default_period = match period_parameter.default {
        Some(ParamValueStatic::Int(20)) => 20usize,
        other => {
            return Err(format!(
                "{BOLLINGER_BANDS_ID}.period: expected exact registry default 20, found \
                 {other:?}"
            ));
        }
    };

    let resolve_deviation_default =
        |index: usize, key: &'static str| -> std::result::Result<u64, String> {
            let parameter = &info.params[index];
            if parameter.key != key
                || !matches!(&parameter.kind, IndicatorParamKind::Float)
                || parameter.required
                || parameter.min.is_some()
                || parameter.max.is_some()
                || parameter.step.is_some()
                || !parameter.enum_values.is_empty()
            {
                return Err(format!(
                    "{BOLLINGER_BANDS_ID}.{key}: expected an optional unbounded float with no \
                     step/enum at slot {index}, found key={} kind={:?} required={} \
                     bounds={:?}..={:?} step={:?} enum={:?}",
                    parameter.key,
                    parameter.kind,
                    parameter.required,
                    parameter.min,
                    parameter.max,
                    parameter.step,
                    parameter.enum_values
                ));
            }
            match parameter.default {
                Some(ParamValueStatic::Float(value)) if value.to_bits() == 2.0_f64.to_bits() => {
                    Ok(value.to_bits())
                }
                other => Err(format!(
                    "{BOLLINGER_BANDS_ID}.{key}: expected exact registry default 2.0, found \
                     {other:?}"
                )),
            }
        };
    let devup_bits = resolve_deviation_default(1, "devup")?;
    let devdn_bits = resolve_deviation_default(2, "devdn")?;

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{BOLLINGER_BANDS_ID}: period-bearing default route cannot require a \
                 period-invariant kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys.as_slice() != ["period"] {
                return Err(format!(
                    "{BOLLINGER_BANDS_ID}: canonical sweep must override exactly [\"period\"], \
                     found {override_keys:?}"
                ));
            }
            positive_usize_parameter(BOLLINGER_BANDS_ID, "period", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{BOLLINGER_BANDS_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    i32::try_from(period).map_err(|_| {
        format!("{BOLLINGER_BANDS_ID}.period: value {period} exceeds the CUDA i32 ABI")
    })?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{BOLLINGER_BANDS_ID}: resolved period {period} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((period, devup_bits, devdn_bits))
}

/// Resolve the canonical Buff Averages fast/slow tuple. The output selector is
/// part of the CPU registry contract because CPU dispatch materializes one
/// named row at a time; the production CUDA route emits both registered rows
/// in one launch and therefore validates, but never forwards, that selector.
fn resolve_buff_averages_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(BUFF_AVERAGES_ID)
        .ok_or_else(|| format!("{BUFF_AVERAGES_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != BUFF_AVERAGES_PARAMETER_KEYS {
        return Err(format!(
            "{BUFF_AVERAGES_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            BUFF_AVERAGES_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != BUFF_AVERAGES_OUTPUT_IDS {
        return Err(format!(
            "{BUFF_AVERAGES_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            BUFF_AVERAGES_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(BUFF_AVERAGES_ID);
    let expected_planned_outputs = BUFF_AVERAGES_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{BUFF_AVERAGES_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let expected_defaults = [5_i64, 20_i64];
    for (index, (key, expected_default)) in BUFF_AVERAGES_PARAMETER_KEYS[..2]
        .iter()
        .copied()
        .zip(expected_defaults)
        .enumerate()
    {
        let parameter = &info.params[index];
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{BUFF_AVERAGES_ID}.{key}: expected an optional integer bounded \
                 1..=unbounded with step 1 and no enum values at slot {index}, found key={} \
                 kind={:?} required={} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
        if parameter.default != Some(ParamValueStatic::Int(expected_default)) {
            return Err(format!(
                "{BUFF_AVERAGES_ID}.{key}: expected exact registry default {expected_default}, \
                 found {:?}",
                parameter.default
            ));
        }
    }

    let output_parameter = &info.params[2];
    if output_parameter.key != "output"
        || !matches!(&output_parameter.kind, IndicatorParamKind::EnumString)
        || output_parameter.required
        || output_parameter.default != Some(ParamValueStatic::EnumString("fast"))
        || output_parameter.min.is_some()
        || output_parameter.max.is_some()
        || output_parameter.step.is_some()
        || output_parameter.enum_values != ["fast", "slow"]
    {
        return Err(format!(
            "{BUFF_AVERAGES_ID}.output: expected optional canonical fast/slow selector with \
             default fast, found kind={:?} required={} default={:?} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            output_parameter.kind,
            output_parameter.required,
            output_parameter.default,
            output_parameter.min,
            output_parameter.max,
            output_parameter.step,
            output_parameter.enum_values
        ));
    }

    let (fast_period, slow_period) = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => (5, 20),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{BUFF_AVERAGES_ID}: window-bearing default route cannot require a \
                 period-invariant kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["fast_period", "slow_period"] {
                return Err(format!(
                    "{BUFF_AVERAGES_ID}: canonical ratio sweep must override exactly \
                     [\"fast_period\", \"slow_period\"], found {override_keys:?}"
                ));
            }
            (
                positive_usize_parameter(BUFF_AVERAGES_ID, "fast_period", overrides[0].1)?,
                positive_usize_parameter(BUFF_AVERAGES_ID, "slow_period", overrides[1].1)?,
            )
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{BUFF_AVERAGES_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    let expected_fast_period = (5usize
        .checked_mul(slow_period)
        .and_then(|value| value.checked_add(10))
        .ok_or_else(|| format!("{BUFF_AVERAGES_ID}: ratio scaling overflow"))?
        / 20)
        .max(1);
    if fast_period != expected_fast_period {
        return Err(format!(
            "{BUFF_AVERAGES_ID}: slow anchor {slow_period} requires exact half-up 5:20 fast \
             period {expected_fast_period}, found {fast_period}"
        ));
    }
    i32::try_from(fast_period).map_err(|_| {
        format!("{BUFF_AVERAGES_ID}.fast_period: value {fast_period} exceeds the CUDA i32 ABI")
    })?;
    i32::try_from(slow_period).map_err(|_| {
        format!("{BUFF_AVERAGES_ID}.slow_period: value {slow_period} exceeds the CUDA i32 ABI")
    })?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == slow_period => {}
        other => {
            return Err(format!(
                "{BUFF_AVERAGES_ID}: resolved tuple ({fast_period}, {slow_period}) requires slow \
                 anchor {slow_period}, found {other:?}"
            ));
        }
    }
    Ok((fast_period, slow_period))
}

/// Resolve the exact canonical Candle Strength Oscillator point admitted by
/// the feature plan. The current one-dimensional sweep changes only `period`;
/// ATR remains disabled at length 50 and mode remains Bollinger, while the
/// full registered boolean and mode domains are still validated before CUDA
/// allocation so schema drift cannot silently change the formula.
fn resolve_candle_strength_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, bool, usize, ClassicCandleStrengthMode), String> {
    let info = get_indicator(CANDLE_STRENGTH_OSCILLATOR_ID).ok_or_else(|| {
        format!("{CANDLE_STRENGTH_OSCILLATOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CANDLE_STRENGTH_OSCILLATOR_PARAMETER_KEYS {
        return Err(format!(
            "{CANDLE_STRENGTH_OSCILLATOR_ID}: CUDA all-output ABI requires exact registry \
             parameters {:?}, found {declared_keys:?}",
            CANDLE_STRENGTH_OSCILLATOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS {
        return Err(format!(
            "{CANDLE_STRENGTH_OSCILLATOR_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(CANDLE_STRENGTH_OSCILLATOR_ID);
    let expected_planned_outputs = CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{CANDLE_STRENGTH_OSCILLATOR_ID}: admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    for (index, key) in [(0usize, "period"), (2usize, "atr_length")] {
        let parameter = &info.params[index];
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(50))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{CANDLE_STRENGTH_OSCILLATOR_ID}.{key}: expected an optional integer default \
                 50 bounded 1..=unbounded with step 1 and no enum values at slot {index}, \
                 found key={} kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} \
                 enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let atr_enabled_parameter = &info.params[1];
    if atr_enabled_parameter.key != "atr_enabled"
        || !matches!(&atr_enabled_parameter.kind, IndicatorParamKind::Bool)
        || atr_enabled_parameter.required
        || atr_enabled_parameter.default != Some(ParamValueStatic::Bool(false))
        || atr_enabled_parameter.min.is_some()
        || atr_enabled_parameter.max.is_some()
        || atr_enabled_parameter.step.is_some()
        || !atr_enabled_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CANDLE_STRENGTH_OSCILLATOR_ID}.atr_enabled: expected optional boolean default \
             false with no bounds/step/enum, found kind={:?} required={} default={:?} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            atr_enabled_parameter.kind,
            atr_enabled_parameter.required,
            atr_enabled_parameter.default,
            atr_enabled_parameter.min,
            atr_enabled_parameter.max,
            atr_enabled_parameter.step,
            atr_enabled_parameter.enum_values
        ));
    }

    let mode_parameter = &info.params[3];
    if mode_parameter.key != "mode"
        || !matches!(&mode_parameter.kind, IndicatorParamKind::EnumString)
        || mode_parameter.required
        || mode_parameter.default != Some(ParamValueStatic::EnumString("bollinger"))
        || mode_parameter.min.is_some()
        || mode_parameter.max.is_some()
        || mode_parameter.step.is_some()
        || mode_parameter.enum_values != ["bollinger", "donchian"]
    {
        return Err(format!(
            "{CANDLE_STRENGTH_OSCILLATOR_ID}.mode: expected optional canonical \
             bollinger/donchian selector with default bollinger, found kind={:?} required={} \
             default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            mode_parameter.kind,
            mode_parameter.required,
            mode_parameter.default,
            mode_parameter.min,
            mode_parameter.max,
            mode_parameter.step,
            mode_parameter.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => 50usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{CANDLE_STRENGTH_OSCILLATOR_ID}: period-bearing default route cannot require \
                 a period-invariant kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{CANDLE_STRENGTH_OSCILLATOR_ID}: canonical sweep must override exactly \
                     [\"period\"], found {override_keys:?}"
                ));
            }
            positive_usize_parameter(CANDLE_STRENGTH_OSCILLATOR_ID, "period", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{CANDLE_STRENGTH_OSCILLATOR_ID}: discrete parameters cannot enter an f64 \
                 all-output route"
            ));
        }
    };
    i32::try_from(period).map_err(|_| {
        format!("{CANDLE_STRENGTH_OSCILLATOR_ID}.period: value {period} exceeds the CUDA i32 ABI")
    })?;
    let atr_length = 50usize;
    i32::try_from(atr_length).map_err(|_| {
        format!(
            "{CANDLE_STRENGTH_OSCILLATOR_ID}.atr_length: value {atr_length} exceeds the CUDA \
             i32 ABI"
        )
    })?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{CANDLE_STRENGTH_OSCILLATOR_ID}: resolved period {period} requires the same \
                 anchor, found {other:?}"
            ));
        }
    }
    Ok((
        period,
        false,
        atr_length,
        ClassicCandleStrengthMode::Bollinger,
    ))
}

/// Resolve the exact canonical Chandelier Exit point admitted by the feature
/// plan. The extended sweep changes only `period`; multiplier 3.0 and
/// close-based extrema remain the canonical production defaults.
fn resolve_chandelier_exit_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, bool), String> {
    let info = get_indicator(CHANDELIER_EXIT_ID)
        .ok_or_else(|| format!("{CHANDELIER_EXIT_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CHANDELIER_EXIT_PARAMETER_KEYS {
        return Err(format!(
            "{CHANDELIER_EXIT_ID}: CUDA pair ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            CHANDELIER_EXIT_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != CHANDELIER_EXIT_OUTPUT_IDS {
        return Err(format!(
            "{CHANDELIER_EXIT_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            CHANDELIER_EXIT_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(CHANDELIER_EXIT_ID);
    let expected_planned_outputs = CHANDELIER_EXIT_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{CHANDELIER_EXIT_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let period_parameter = &info.params[0];
    if period_parameter.key != "period"
        || !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.default != Some(ParamValueStatic::Int(22))
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CHANDELIER_EXIT_ID}.period: expected optional integer default 22 bounded \
             1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
             default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.default,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }

    let mult_parameter = &info.params[1];
    if mult_parameter.key != "mult"
        || !matches!(&mult_parameter.kind, IndicatorParamKind::Float)
        || mult_parameter.required
        || mult_parameter.default != Some(ParamValueStatic::Float(3.0))
        || mult_parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || mult_parameter.max.is_some()
        || mult_parameter.step.is_some()
        || !mult_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CHANDELIER_EXIT_ID}.mult: expected optional f64 default 3.0 bounded \
             0..=unbounded with no step/enum, found kind={:?} required={} default={:?} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            mult_parameter.kind,
            mult_parameter.required,
            mult_parameter.default,
            mult_parameter.min,
            mult_parameter.max,
            mult_parameter.step,
            mult_parameter.enum_values
        ));
    }

    let use_close_parameter = &info.params[2];
    if use_close_parameter.key != "use_close"
        || !matches!(&use_close_parameter.kind, IndicatorParamKind::Bool)
        || use_close_parameter.required
        || use_close_parameter.default != Some(ParamValueStatic::Bool(true))
        || use_close_parameter.min.is_some()
        || use_close_parameter.max.is_some()
        || use_close_parameter.step.is_some()
        || use_close_parameter.enum_values != ["true", "false"]
    {
        return Err(format!(
            "{CHANDELIER_EXIT_ID}.use_close: expected optional boolean default true with exact \
             [true,false] vocabulary and no bounds/step, found kind={:?} required={} default={:?} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            use_close_parameter.kind,
            use_close_parameter.required,
            use_close_parameter.default,
            use_close_parameter.min,
            use_close_parameter.max,
            use_close_parameter.step,
            use_close_parameter.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            require_period_invariant_kernel: false,
            ..
        } => 22usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{CHANDELIER_EXIT_ID}: period-bearing default route cannot require a \
                 period-invariant kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{CHANDELIER_EXIT_ID}: canonical sweep must override exactly [\"period\"], \
                     found {override_keys:?}"
                ));
            }
            positive_usize_parameter(CHANDELIER_EXIT_ID, "period", overrides[0].1)?
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{CHANDELIER_EXIT_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    i32::try_from(period).map_err(|_| {
        format!("{CHANDELIER_EXIT_ID}.period: value {period} exceeds the CUDA i32 ABI")
    })?;
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{CHANDELIER_EXIT_ID}: resolved period {period} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((period, 3.0_f64.to_bits(), true))
}

/// Resolve CKSP's exact currently admitted default tuple. The canonical
/// registry owns p/x/q, but the production period planner deliberately emits
/// no synthetic override for those keys in this bounded slice. Any swept node
/// is therefore a fail-closed schema error rather than an inert duplicate.
fn resolve_cksp_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, usize), String> {
    let info =
        get_indicator(CKSP_ID).ok_or_else(|| format!("{CKSP_ID}: absent from the registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CKSP_PARAMETER_KEYS {
        return Err(format!(
            "{CKSP_ID}: CUDA pair ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            CKSP_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != CKSP_OUTPUT_IDS {
        return Err(format!(
            "{CKSP_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            CKSP_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(CKSP_ID);
    let expected_planned_outputs = CKSP_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{CKSP_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let p_parameter = &info.params[0];
    if p_parameter.key != "p"
        || !matches!(&p_parameter.kind, IndicatorParamKind::Int)
        || p_parameter.required
        || p_parameter.default != Some(ParamValueStatic::Int(10))
        || p_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || p_parameter.max.is_some()
        || p_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !p_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CKSP_ID}.p: expected optional integer default 10 bounded 1..=unbounded with step 1 \
             and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            p_parameter.kind,
            p_parameter.required,
            p_parameter.default,
            p_parameter.min,
            p_parameter.max,
            p_parameter.step,
            p_parameter.enum_values
        ));
    }

    let x_parameter = &info.params[1];
    if x_parameter.key != "x"
        || !matches!(&x_parameter.kind, IndicatorParamKind::Float)
        || x_parameter.required
        || x_parameter.default != Some(ParamValueStatic::Float(1.0))
        || x_parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || x_parameter.max.is_some()
        || x_parameter.step.is_some()
        || !x_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CKSP_ID}.x: expected optional f64 default 1.0 bounded 0..=unbounded with no \
             step/enum, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} \
             enum={:?}",
            x_parameter.kind,
            x_parameter.required,
            x_parameter.default,
            x_parameter.min,
            x_parameter.max,
            x_parameter.step,
            x_parameter.enum_values
        ));
    }

    let q_parameter = &info.params[2];
    if q_parameter.key != "q"
        || !matches!(&q_parameter.kind, IndicatorParamKind::Int)
        || q_parameter.required
        || q_parameter.default != Some(ParamValueStatic::Int(9))
        || q_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || q_parameter.max.is_some()
        || q_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !q_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CKSP_ID}.q: expected optional integer default 9 bounded 1..=unbounded with step 1 \
             and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            q_parameter.kind,
            q_parameter.required,
            q_parameter.default,
            q_parameter.min,
            q_parameter.max,
            q_parameter.step,
            q_parameter.enum_values
        ));
    }

    match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => {}
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{CKSP_ID}: default-only route requires inert anchor 1 and the preserved \
                 period-invariant primary classification, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { .. } => {
            return Err(format!(
                "{CKSP_ID}: currently admitted CUDA vocabulary is default-only; a synthetic \
                 period sweep cannot stand in for explicit p/q schema expansion"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{CKSP_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    }
    Ok((10, 1.0_f64.to_bits(), 9))
}

/// Resolve Coppock's exact three-window registry tuple. The long ROC default
/// is the RegistryRatio anchor, so every admitted sweep must carry all three
/// half-up-scaled overrides in canonical registry order. A stale generic
/// period-invariant route or an arbitrary tuple fails before a CUDA session
/// exists.
fn resolve_coppock_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize), String> {
    let info = get_indicator(COPPOCK_ID)
        .ok_or_else(|| format!("{COPPOCK_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != COPPOCK_PARAMETER_KEYS {
        return Err(format!(
            "{COPPOCK_ID}: CUDA value ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            COPPOCK_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != [COPPOCK_OUTPUT_ID] {
        return Err(format!(
            "{COPPOCK_ID}: canonical output ABI [{COPPOCK_OUTPUT_ID:?}] != registry \
             {declared_outputs:?}"
        ));
    }
    let planned_outputs = output_ids_for(COPPOCK_ID);
    if planned_outputs != [None] {
        return Err(format!(
            "{COPPOCK_ID}: canonical admitted receipt must be the sole unsuffixed value output, \
             found {planned_outputs:?}"
        ));
    }

    for (parameter, (key, default)) in info.params.iter().zip([
        ("short_roc_period", ParamValueStatic::Int(11)),
        ("long_roc_period", ParamValueStatic::Int(14)),
        ("ma_period", ParamValueStatic::Int(10)),
    ]) {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(default)
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{COPPOCK_ID}.{key}: expected optional integer default {default:?} bounded \
                 1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
                 default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let (short_roc_period, long_roc_period, ma_period) = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(14),
            require_period_invariant_kernel: false,
        } => (11, 14, 10),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{COPPOCK_ID}: default route requires ratio anchor 14 and a window-consuming \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{COPPOCK_ID}: swept period {period} requires the same resolved anchor, \
                     found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != COPPOCK_PARAMETER_KEYS {
                return Err(format!(
                    "{COPPOCK_ID}: canonical sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    COPPOCK_PARAMETER_KEYS
                ));
            }
            let short_roc_period =
                positive_usize_parameter(COPPOCK_ID, "short_roc_period", overrides[0].1)?;
            let long_roc_period =
                positive_usize_parameter(COPPOCK_ID, "long_roc_period", overrides[1].1)?;
            let ma_period = positive_usize_parameter(COPPOCK_ID, "ma_period", overrides[2].1)?;
            let scale = |default: usize| {
                default
                    .checked_mul(*period)
                    .and_then(|scaled| scaled.checked_add(7))
                    .map(|scaled| (scaled / 14).max(1))
                    .ok_or_else(|| {
                        format!(
                            "{COPPOCK_ID}: {default}:14 ratio scaling overflow at period \
                             {period}"
                        )
                    })
            };
            let expected = (scale(11)?, *period, scale(10)?);
            let actual = (short_roc_period, long_roc_period, ma_period);
            if actual != expected {
                return Err(format!(
                    "{COPPOCK_ID}: period {period} requires exact half-up 11:14:10 tuple \
                     {expected:?}, found {actual:?}"
                ));
            }
            actual
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{COPPOCK_ID}: swept tuple requires a resolved long-window anchor, found \
                 {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{COPPOCK_ID}: discrete parameters cannot enter an f64 value route"
            ));
        }
    };

    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == long_roc_period => {}
        other => {
            return Err(format!(
                "{COPPOCK_ID}: long ROC {long_roc_period} must equal the resolved ratio anchor, \
                 found {other:?}"
            ));
        }
    }
    for (key, value) in [
        ("short_roc_period", short_roc_period),
        ("long_roc_period", long_roc_period),
        ("ma_period", ma_period),
    ] {
        i32::try_from(value)
            .map_err(|_| format!("{COPPOCK_ID}.{key}: value {value} exceeds the CUDA i32 ABI"))?;
    }
    Ok((short_roc_period, long_roc_period, ma_period))
}

/// Resolve Correlation Cycle's exact `(period, threshold)` production point.
/// The admitted Classic sweep changes only `period`; `threshold` remains the
/// registry's exact 9.0 f64 default while all four canonical outputs share one
/// sequential CUDA row.
fn resolve_correlation_cycle_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64), String> {
    let info = get_indicator(CORRELATION_CYCLE_ID)
        .ok_or_else(|| format!("{CORRELATION_CYCLE_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CORRELATION_CYCLE_PARAMETER_KEYS {
        return Err(format!(
            "{CORRELATION_CYCLE_ID}: CUDA all-output ABI requires exact registry parameters \
             {:?}, found {declared_keys:?}",
            CORRELATION_CYCLE_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != CORRELATION_CYCLE_OUTPUT_IDS {
        return Err(format!(
            "{CORRELATION_CYCLE_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            CORRELATION_CYCLE_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(CORRELATION_CYCLE_ID);
    let expected_planned_outputs = CORRELATION_CYCLE_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{CORRELATION_CYCLE_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.default != Some(ParamValueStatic::Int(20))
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CORRELATION_CYCLE_ID}.period: expected optional integer default 20 bounded \
             1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
             default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.default,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }

    let threshold_parameter = &info.params[1];
    let threshold_bits = match threshold_parameter.default {
        Some(ParamValueStatic::Float(value)) if value.to_bits() == 9.0_f64.to_bits() => {
            value.to_bits()
        }
        other => {
            return Err(format!(
                "{CORRELATION_CYCLE_ID}.threshold: expected exact registry default 9.0, found \
                 {other:?}"
            ));
        }
    };
    if !matches!(&threshold_parameter.kind, IndicatorParamKind::Float)
        || threshold_parameter.required
        || threshold_parameter.min.is_some()
        || threshold_parameter.max.is_some()
        || threshold_parameter.step.is_some()
        || !threshold_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CORRELATION_CYCLE_ID}.threshold: expected optional unbounded float default 9.0 \
             with no step/enum, found kind={:?} required={} bounds={:?}..={:?} step={:?} \
             enum={:?}",
            threshold_parameter.kind,
            threshold_parameter.required,
            threshold_parameter.min,
            threshold_parameter.max,
            threshold_parameter.step,
            threshold_parameter.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(20),
            require_period_invariant_kernel: false,
        } => 20,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{CORRELATION_CYCLE_ID}: default route requires period anchor 20 and a \
                 window-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{CORRELATION_CYCLE_ID}: swept period {period} requires the same resolved \
                     anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{CORRELATION_CYCLE_ID}: canonical sweep must override exactly \
                     [\"period\"], found {override_keys:?}"
                ));
            }
            let overridden_period =
                positive_usize_parameter(CORRELATION_CYCLE_ID, "period", overrides[0].1)?;
            if overridden_period != *period {
                return Err(format!(
                    "{CORRELATION_CYCLE_ID}: swept period {period} != exact override \
                     {overridden_period}"
                ));
            }
            *period
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{CORRELATION_CYCLE_ID}: swept tuple requires a resolved period anchor, found \
                 {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{CORRELATION_CYCLE_ID}: discrete parameters cannot enter an f64 all-output \
                 route"
            ));
        }
    };
    i32::try_from(period).map_err(|_| {
        format!("{CORRELATION_CYCLE_ID}.period: value {period} exceeds the CUDA i32 ABI")
    })?;
    Ok((period, threshold_bits))
}

/// Resolve CVI's sole canonical `(value, period)` production point. The
/// registry is the CPU/schema authority; the additional 512 ceiling is the
/// compiled CUDA ring bound and is checked here before a CUDA session exists.
fn resolve_cvi_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(CVI_ID)
        .ok_or_else(|| format!("{CVI_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CVI_PARAMETER_KEYS {
        return Err(format!(
            "{CVI_ID}: CUDA primary ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            CVI_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != [CVI_OUTPUT_ID] {
        return Err(format!(
            "{CVI_ID}: CUDA output ABI [{CVI_OUTPUT_ID:?}] != registry {declared_outputs:?}"
        ));
    }
    let planned_outputs = output_ids_for(CVI_ID);
    if planned_outputs != [None] {
        return Err(format!(
            "{CVI_ID}: admitted single-output receipt [None] != {planned_outputs:?}"
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.default != Some(ParamValueStatic::Int(10))
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{CVI_ID}.period: expected optional integer default 10 bounded 1..=unbounded with \
             step 1 and no enum values, found kind={:?} required={} default={:?} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.default,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(10),
            require_period_invariant_kernel: false,
        } => 10,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{CVI_ID}: default route requires period anchor 10 and a period-consuming \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{CVI_ID}: swept period {period} requires the same resolved anchor, found \
                     {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != CVI_PARAMETER_KEYS {
                return Err(format!(
                    "{CVI_ID}: canonical sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    CVI_PARAMETER_KEYS
                ));
            }
            let resolved = positive_usize_parameter(CVI_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{CVI_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{CVI_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{CVI_ID}: discrete parameters cannot enter an f64 value route"
            ));
        }
    };
    if period > CVI_MAX_PERIOD {
        return Err(format!(
            "{CVI_ID}.period: value {period} exceeds compiled CUDA ring bound \
             {CVI_MAX_PERIOD}"
        ));
    }
    i32::try_from(period)
        .map_err(|_| format!("{CVI_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok(period)
}

/// Resolve Cyberpunk Value Trend Analyzer's exact default threshold tuple.
/// The formula has no period parameter, so the primary ABI's inert anchor is
/// legal only for the period-invariant default route and every synthetic sweep
/// is rejected before a CUDA session exists.
fn resolve_cyberpunk_value_trend_analyzer_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(CYBERPUNK_VALUE_TREND_ANALYZER_ID).ok_or_else(|| {
        format!("{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CYBERPUNK_VALUE_TREND_ANALYZER_PARAMETER_KEYS {
        return Err(format!(
            "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: CUDA all-output ABI requires exact registry \
             parameters {:?}, found {declared_keys:?}",
            CYBERPUNK_VALUE_TREND_ANALYZER_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS {
        return Err(format!(
            "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(CYBERPUNK_VALUE_TREND_ANALYZER_ID);
    let expected_planned_outputs = CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let expected_defaults = [ParamValueStatic::Int(30), ParamValueStatic::Int(75)];
    let mut resolved = [0usize; CYBERPUNK_VALUE_TREND_ANALYZER_PARAMETER_KEYS.len()];
    for (index, key) in CYBERPUNK_VALUE_TREND_ANALYZER_PARAMETER_KEYS
        .iter()
        .copied()
        .enumerate()
    {
        let parameter = &info.params[index];
        if !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(expected_defaults[index])
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.map(f64::to_bits) != Some(100.0_f64.to_bits())
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}.{key}: expected optional integer default \
                 {:?} bounded 1..=100 with step 1 and no enum values, found kind={:?} \
                 required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                expected_defaults[index],
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
        let ParamValueStatic::Int(default) = expected_defaults[index] else {
            unreachable!("Cyberpunk Value Trend Analyzer defaults are typed integers")
        };
        resolved[index] =
            positive_usize_parameter(CYBERPUNK_VALUE_TREND_ANALYZER_ID, key, default)?;
        i32::try_from(resolved[index]).map_err(|_| {
            format!(
                "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}.{key}: value {} exceeds the CUDA i32 ABI",
                resolved[index]
            )
        })?;
    }

    match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => Ok((resolved[0], resolved[1])),
        ClassicCudaParameters::Defaults { .. } => Err(format!(
            "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: default route requires inert anchor 1 and a \
             period-invariant kernel, found {parameters:?}"
        )),
        ClassicCudaParameters::Swept { .. } => Err(format!(
            "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: formula has no canonical period sweep"
        )),
        ClassicCudaParameters::DiscreteDefaults => Err(format!(
            "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: discrete parameters cannot enter an f64 \
             all-output route"
        )),
    }
}

/// Resolve Cycle Channel Oscillator's canonical default source, coupled
/// short/medium length ratio and multiplier tuple. The preserved primary ABI
/// remains the default-only `fast` lane; this typed full-kernel route carries
/// both outputs and every admitted ratio point explicitly.
fn resolve_cycle_channel_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, u64, u64), String> {
    let info = get_indicator(CYCLE_CHANNEL_OSCILLATOR_ID).ok_or_else(|| {
        format!("{CYCLE_CHANNEL_OSCILLATOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != CYCLE_CHANNEL_OSCILLATOR_PARAMETER_KEYS {
        return Err(format!(
            "{CYCLE_CHANNEL_OSCILLATOR_ID}: CUDA pair ABI requires exact registry parameters \
             {:?}, found {declared_keys:?}",
            CYCLE_CHANNEL_OSCILLATOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS {
        return Err(format!(
            "{CYCLE_CHANNEL_OSCILLATOR_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(CYCLE_CHANNEL_OSCILLATOR_ID);
    let expected_planned_outputs = CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{CYCLE_CHANNEL_OSCILLATOR_ID}: admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let source = &info.params[0];
    if !matches!(&source.kind, IndicatorParamKind::EnumString)
        || source.required
        || source.default != Some(ParamValueStatic::EnumString("close"))
        || source.min.is_some()
        || source.max.is_some()
        || source.step.is_some()
        || source.enum_values != CYCLE_CHANNEL_OSCILLATOR_SOURCE_VALUES.as_slice()
    {
        return Err(format!(
            "{CYCLE_CHANNEL_OSCILLATOR_ID}.source: expected optional exact default `close`, \
             canonical price-source enum {:?}, and no numeric bounds/step; found kind={:?} \
             required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            CYCLE_CHANNEL_OSCILLATOR_SOURCE_VALUES,
            source.kind,
            source.required,
            source.default,
            source.min,
            source.max,
            source.step,
            source.enum_values
        ));
    }

    let default_windows = [10usize, 30usize];
    for (index, (key, default)) in CYCLE_CHANNEL_OSCILLATOR_WINDOW_KEYS
        .iter()
        .copied()
        .zip(default_windows)
        .enumerate()
    {
        let parameter = &info.params[index + 1];
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default as i64))
            || parameter.min.map(f64::to_bits) != Some(2.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{CYCLE_CHANNEL_OSCILLATOR_ID}.{key}: expected optional integer default \
                 {default} bounded 2..=unbounded with step 1 and no enum values, found \
                 kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let expected_multiplier_defaults = [1.0_f64, 3.0_f64];
    let mut multiplier_bits = [0_u64; 2];
    for (offset, (key, expected)) in ["short_multiplier", "medium_multiplier"]
        .into_iter()
        .zip(expected_multiplier_defaults)
        .enumerate()
    {
        let parameter = &info.params[offset + 3];
        let default = match parameter.default {
            Some(ParamValueStatic::Float(value)) if value.to_bits() == expected.to_bits() => value,
            other => {
                return Err(format!(
                    "{CYCLE_CHANNEL_OSCILLATOR_ID}.{key}: expected exact float default \
                     {expected}, found {other:?}"
                ));
            }
        };
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Float)
            || parameter.required
            || parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(0.1_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{CYCLE_CHANNEL_OSCILLATOR_ID}.{key}: expected optional finite float default \
                 {expected} bounded 0..=unbounded with step 0.1 and no enum values, found \
                 kind={:?} required={} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
        multiplier_bits[offset] = default.to_bits();
    }

    let (short_cycle_length, medium_cycle_length) = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(30),
            require_period_invariant_kernel: false,
        } => (10, 30),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{CYCLE_CHANNEL_OSCILLATOR_ID}: default route requires ratio anchor 30 and the \
                 typed window-consuming full kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{CYCLE_CHANNEL_OSCILLATOR_ID}: swept period {period} requires the same \
                     resolved anchor, found {anchor}"
                ));
            }
            if let Some(reason) = sweep_point_exclusion(CYCLE_CHANNEL_OSCILLATOR_ID, *period) {
                return Err(format!(
                    "{CYCLE_CHANNEL_OSCILLATOR_ID}@{period}: formula-level sweep exclusion: \
                     {reason}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != CYCLE_CHANNEL_OSCILLATOR_WINDOW_KEYS {
                return Err(format!(
                    "{CYCLE_CHANNEL_OSCILLATOR_ID}: canonical sweep must override exactly {:?}, \
                     found {override_keys:?}",
                    CYCLE_CHANNEL_OSCILLATOR_WINDOW_KEYS
                ));
            }
            let short_cycle_length = positive_usize_parameter(
                CYCLE_CHANNEL_OSCILLATOR_ID,
                "short_cycle_length",
                overrides[0].1,
            )?;
            let medium_cycle_length = positive_usize_parameter(
                CYCLE_CHANNEL_OSCILLATOR_ID,
                "medium_cycle_length",
                overrides[1].1,
            )?;
            if short_cycle_length < 2 || medium_cycle_length < 2 {
                return Err(format!(
                    "{CYCLE_CHANNEL_OSCILLATOR_ID}: swept cycle lengths must both be >= 2, \
                     found ({short_cycle_length}, {medium_cycle_length})"
                ));
            }
            let expected_short = (*period)
                .checked_mul(10)
                .and_then(|scaled| scaled.checked_add(15))
                .map(|scaled| (scaled / 30).max(1))
                .ok_or_else(|| {
                    format!(
                        "{CYCLE_CHANNEL_OSCILLATOR_ID}: 10:30 ratio scaling overflow at period \
                         {period}"
                    )
                })?;
            let expected = (expected_short, *period);
            let actual = (short_cycle_length, medium_cycle_length);
            if actual != expected {
                return Err(format!(
                    "{CYCLE_CHANNEL_OSCILLATOR_ID}: period {period} requires exact half-up \
                     10:30 tuple {expected:?}, found {actual:?}"
                ));
            }
            actual
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{CYCLE_CHANNEL_OSCILLATOR_ID}: swept tuple requires a resolved medium-window \
                 anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{CYCLE_CHANNEL_OSCILLATOR_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };

    for (key, value) in [
        ("short_cycle_length", short_cycle_length),
        ("medium_cycle_length", medium_cycle_length),
    ] {
        i32::try_from(value).map_err(|_| {
            format!("{CYCLE_CHANNEL_OSCILLATOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
    }
    Ok((
        short_cycle_length,
        medium_cycle_length,
        multiplier_bits[0],
        multiplier_bits[1],
    ))
}

/// Resolve Daily Factor's sole canonical threshold. The indicator has no
/// integer timescale, so only the period-invariant base receipt may enter this
/// typed route; any synthetic period sweep fails before CUDA session
/// construction. The kernel retains its complete three-output ABI, while the
/// reviewed fixed EMA(14) auxiliary is absent from the admitted feature plan.
fn resolve_daily_factor_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<u64, String> {
    let info = get_indicator(DAILY_FACTOR_ID)
        .ok_or_else(|| format!("{DAILY_FACTOR_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DAILY_FACTOR_PARAMETER_KEYS {
        return Err(format!(
            "{DAILY_FACTOR_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            DAILY_FACTOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DAILY_FACTOR_FULL_OUTPUT_IDS {
        return Err(format!(
            "{DAILY_FACTOR_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            DAILY_FACTOR_FULL_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DAILY_FACTOR_ID);
    let expected_planned_outputs = DAILY_FACTOR_PRODUCTION_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DAILY_FACTOR_ID}: reviewed admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let parameter = &info.params[0];
    if !matches!(&parameter.kind, IndicatorParamKind::Float)
        || parameter.required
        || parameter.default != Some(ParamValueStatic::Float(0.35))
        || parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || parameter.max.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || parameter.step.map(f64::to_bits) != Some(0.01_f64.to_bits())
        || !parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{DAILY_FACTOR_ID}.threshold_level: expected optional float default 0.35 bounded \
             0..=1 with step 0.01 and no enum values, found kind={:?} required={} default={:?} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            parameter.kind,
            parameter.required,
            parameter.default,
            parameter.min,
            parameter.max,
            parameter.step,
            parameter.enum_values
        ));
    }

    match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => Ok(0.35_f64.to_bits()),
        ClassicCudaParameters::Defaults { .. } => Err(format!(
            "{DAILY_FACTOR_ID}: default route requires inert anchor 1 and a period-invariant \
             kernel, found {parameters:?}"
        )),
        ClassicCudaParameters::Swept { .. } => Err(format!(
            "{DAILY_FACTOR_ID}: formula has no canonical period sweep"
        )),
        ClassicCudaParameters::DiscreteDefaults => Err(format!(
            "{DAILY_FACTOR_ID}: discrete parameters cannot enter an f64 all-output route"
        )),
    }
}

/// Resolve Damiani Volatmeter's complete default four-window and threshold
/// tuple. None of its registry keys belongs to the canonical production period
/// vocabulary, so the preserved primary ABI and this typed pair route are both
/// default-only; any manually injected synthetic sweep fails before a session.
fn resolve_damiani_volatmeter_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize, usize, u64), String> {
    let info = get_indicator(DAMIANI_VOLATMETER_ID)
        .ok_or_else(|| format!("{DAMIANI_VOLATMETER_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DAMIANI_VOLATMETER_PARAMETER_KEYS {
        return Err(format!(
            "{DAMIANI_VOLATMETER_ID}: CUDA pair ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            DAMIANI_VOLATMETER_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DAMIANI_VOLATMETER_OUTPUT_IDS {
        return Err(format!(
            "{DAMIANI_VOLATMETER_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            DAMIANI_VOLATMETER_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DAMIANI_VOLATMETER_ID);
    let expected_planned_outputs = DAMIANI_VOLATMETER_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DAMIANI_VOLATMETER_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let default_windows = [13_usize, 20, 40, 100];
    for (index, ((key, default), parameter)) in DAMIANI_VOLATMETER_WINDOW_KEYS
        .into_iter()
        .zip(default_windows)
        .zip(&info.params[..4])
        .enumerate()
    {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default as i64))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{DAMIANI_VOLATMETER_ID}.{key}: expected optional integer default {default} \
                 bounded 1..=unbounded with step 1 and no enum values at registry offset \
                 {index}, found kind={:?} required={} default={:?} bounds={:?}..={:?} \
                 step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let threshold_parameter = &info.params[4];
    let threshold = match threshold_parameter.default {
        Some(ParamValueStatic::Float(value)) if value.to_bits() == 1.4_f64.to_bits() => value,
        other => {
            return Err(format!(
                "{DAMIANI_VOLATMETER_ID}.threshold: expected exact float default 1.4, found \
                 {other:?}"
            ));
        }
    };
    if threshold_parameter.key != "threshold"
        || !matches!(&threshold_parameter.kind, IndicatorParamKind::Float)
        || threshold_parameter.required
        || threshold_parameter.min.is_some()
        || threshold_parameter.max.is_some()
        || threshold_parameter.step.is_some()
        || !threshold_parameter.enum_values.is_empty()
        || !threshold.is_finite()
    {
        return Err(format!(
            "{DAMIANI_VOLATMETER_ID}.threshold: expected optional finite float default 1.4 with \
             no bounds, step, or enum values; found kind={:?} required={} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            threshold_parameter.kind,
            threshold_parameter.required,
            threshold_parameter.min,
            threshold_parameter.max,
            threshold_parameter.step,
            threshold_parameter.enum_values
        ));
    }

    let windows = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => default_windows,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DAMIANI_VOLATMETER_ID}: default route requires inert anchor 1 and the \
                 period-invariant primary receipt, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept { .. } => {
            return Err(format!(
                "{DAMIANI_VOLATMETER_ID}: formula has no canonical period sweep"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DAMIANI_VOLATMETER_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };

    for (key, value) in DAMIANI_VOLATMETER_WINDOW_KEYS.into_iter().zip(windows) {
        i32::try_from(value).map_err(|_| {
            format!("{DAMIANI_VOLATMETER_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
    }
    Ok((
        windows[0],
        windows[1],
        windows[2],
        windows[3],
        threshold.to_bits(),
    ))
}

/// Resolve DI's sole canonical period for its plus/minus pair. The registry,
/// admitted schema, planning anchor, and explicit sweep override must agree
/// before either output can enter the shared CUDA session.
fn resolve_di_parameters(parameters: &ClassicCudaParameters) -> std::result::Result<usize, String> {
    let info = get_indicator(DI_ID)
        .ok_or_else(|| format!("{DI_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DI_PARAMETER_KEYS {
        return Err(format!(
            "{DI_ID}: CUDA pair ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            DI_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DI_OUTPUT_IDS {
        return Err(format!(
            "{DI_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            DI_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DI_ID);
    let expected_planned_outputs = DI_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DI_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let declared = &info.params[0];
    if declared.key != "period"
        || !matches!(&declared.kind, IndicatorParamKind::Int)
        || declared.required
        || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || declared.max.is_some()
        || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !declared.enum_values.is_empty()
    {
        return Err(format!(
            "{DI_ID}.period: expected optional integer default 14 bounded 1..=unbounded with \
             step 1 and no enum values, found key={} kind={:?} required={} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            declared.key,
            declared.kind,
            declared.required,
            declared.min,
            declared.max,
            declared.step,
            declared.enum_values
        ));
    }
    let default_period = match declared.default {
        Some(ParamValueStatic::Int(14)) => 14usize,
        other => {
            return Err(format!(
                "{DI_ID}.period: expected exact registry default 14, found {other:?}"
            ));
        }
    };

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(14),
            require_period_invariant_kernel: false,
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DI_ID}: default route requires period anchor 14 and a period-consuming kernel, \
                 found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DI_ID}: swept period {period} requires the same resolved anchor, found \
                     {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != DI_PARAMETER_KEYS {
                return Err(format!(
                    "{DI_ID}: canonical sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    DI_PARAMETER_KEYS
                ));
            }
            let resolved = positive_usize_parameter(DI_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{DI_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DI_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DI_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{DI_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok(period)
}

/// Resolve Didi Index's exact registry-ratio tuple. The preserved generic f64
/// primary is fixed at 3:8:20, while the typed four-output ABI consumes the
/// complete half-up-scaled tuple carried by the admitted production graph.
fn resolve_didi_index_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize), String> {
    let info = get_indicator(DIDI_INDEX_ID)
        .ok_or_else(|| format!("{DIDI_INDEX_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DIDI_INDEX_PARAMETER_KEYS {
        return Err(format!(
            "{DIDI_INDEX_ID}: CUDA four-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            DIDI_INDEX_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DIDI_INDEX_OUTPUT_IDS {
        return Err(format!(
            "{DIDI_INDEX_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            DIDI_INDEX_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DIDI_INDEX_ID);
    let expected_planned_outputs = DIDI_INDEX_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DIDI_INDEX_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    const DEFAULTS: [i64; 3] = [3, 8, 20];
    for ((declared, key), default) in info
        .params
        .iter()
        .zip(DIDI_INDEX_PARAMETER_KEYS)
        .zip(DEFAULTS)
    {
        if declared.key != key
            || !matches!(&declared.kind, IndicatorParamKind::Int)
            || declared.required
            || declared.default != Some(ParamValueStatic::Int(default))
            || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || declared.max.is_some()
            || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !declared.enum_values.is_empty()
        {
            return Err(format!(
                "{DIDI_INDEX_ID}.{key}: expected optional integer default {default} bounded \
                 1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
                 default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                declared.kind,
                declared.required,
                declared.default,
                declared.min,
                declared.max,
                declared.step,
                declared.enum_values
            ));
        }
    }

    let tuple = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(20),
            require_period_invariant_kernel: false,
        } => [3usize, 8, 20],
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DIDI_INDEX_ID}: default route requires RegistryRatio anchor 20 without a \
                 no-window classification, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DIDI_INDEX_ID}: swept period {period} requires the same resolved anchor, \
                     found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != DIDI_INDEX_PARAMETER_KEYS {
                return Err(format!(
                    "{DIDI_INDEX_ID}: canonical RegistryRatio sweep must override exactly {:?}, \
                     found {override_keys:?}",
                    DIDI_INDEX_PARAMETER_KEYS
                ));
            }
            let actual = [
                positive_usize_parameter(DIDI_INDEX_ID, "short_length", overrides[0].1)?,
                positive_usize_parameter(DIDI_INDEX_ID, "medium_length", overrides[1].1)?,
                positive_usize_parameter(DIDI_INDEX_ID, "long_length", overrides[2].1)?,
            ];
            let target = i128::try_from(*period)
                .map_err(|_| format!("{DIDI_INDEX_ID}: period {period} exceeds i128"))?;
            let expected = DEFAULTS.map(|default| {
                usize::try_from(((i128::from(default) * target + 10) / 20).max(1))
                    .expect("positive RegistryRatio value must fit usize after period conversion")
            });
            if actual != expected {
                return Err(format!(
                    "{DIDI_INDEX_ID}: period {period} requires exact half-up 3:8:20 tuple \
                     {expected:?}, found {actual:?}"
                ));
            }
            actual
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DIDI_INDEX_ID}: swept tuple requires a resolved RegistryRatio anchor, found \
                 {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DIDI_INDEX_ID}: discrete parameters cannot enter an f64 four-output route"
            ));
        }
    };
    for (key, value) in DIDI_INDEX_PARAMETER_KEYS.into_iter().zip(tuple) {
        i32::try_from(value).map_err(|_| {
            format!("{DIDI_INDEX_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
    }
    Ok((tuple[0], tuple[1], tuple[2]))
}

/// Resolve Directional Imbalance Index's canonical fixed length and exact
/// admitted period. The registry is the sole output/parameter authority; the
/// typed CUDA launch consumes every base/sweep tuple rather than replaying one
/// primary output under six labels.
fn resolve_directional_imbalance_index_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(DIRECTIONAL_IMBALANCE_INDEX_ID).ok_or_else(|| {
        format!("{DIRECTIONAL_IMBALANCE_INDEX_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DIRECTIONAL_IMBALANCE_INDEX_PARAMETER_KEYS {
        return Err(format!(
            "{DIRECTIONAL_IMBALANCE_INDEX_ID}: CUDA six-output ABI requires exact registry \
             parameters {:?}, found {declared_keys:?}",
            DIRECTIONAL_IMBALANCE_INDEX_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS {
        return Err(format!(
            "{DIRECTIONAL_IMBALANCE_INDEX_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DIRECTIONAL_IMBALANCE_INDEX_ID);
    let expected_planned_outputs = DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DIRECTIONAL_IMBALANCE_INDEX_ID}: admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    const DEFAULTS: [i64; 2] = [10, 70];
    for ((declared, key), default) in info
        .params
        .iter()
        .zip(DIRECTIONAL_IMBALANCE_INDEX_PARAMETER_KEYS)
        .zip(DEFAULTS)
    {
        if declared.key != key
            || !matches!(&declared.kind, IndicatorParamKind::Int)
            || declared.required
            || declared.default != Some(ParamValueStatic::Int(default))
            || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || declared.max.is_some()
            || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !declared.enum_values.is_empty()
        {
            return Err(format!(
                "{DIRECTIONAL_IMBALANCE_INDEX_ID}.{key}: expected optional integer default \
                 {default} bounded 1..=unbounded with step 1 and no enum values, found \
                 kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                declared.kind,
                declared.required,
                declared.default,
                declared.min,
                declared.max,
                declared.step,
                declared.enum_values
            ));
        }
    }

    let tuple = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(70),
            require_period_invariant_kernel: false,
        } => [10usize, 70],
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DIRECTIONAL_IMBALANCE_INDEX_ID}: default route requires period anchor 70 \
                 and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DIRECTIONAL_IMBALANCE_INDEX_ID}: swept period {period} requires the same \
                     resolved anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{DIRECTIONAL_IMBALANCE_INDEX_ID}: canonical sweep must override exactly \
                     [\"period\"], found {override_keys:?}"
                ));
            }
            let resolved_period =
                positive_usize_parameter(DIRECTIONAL_IMBALANCE_INDEX_ID, "period", overrides[0].1)?;
            if resolved_period != *period {
                return Err(format!(
                    "{DIRECTIONAL_IMBALANCE_INDEX_ID}: swept period {period} != exact override \
                     {resolved_period}"
                ));
            }
            [10, resolved_period]
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DIRECTIONAL_IMBALANCE_INDEX_ID}: swept tuple requires a resolved period \
                 anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DIRECTIONAL_IMBALANCE_INDEX_ID}: discrete parameters cannot enter an f64 \
                 six-output route"
            ));
        }
    };
    for (key, value) in DIRECTIONAL_IMBALANCE_INDEX_PARAMETER_KEYS
        .into_iter()
        .zip(tuple)
    {
        i32::try_from(value).map_err(|_| {
            format!(
                "{DIRECTIONAL_IMBALANCE_INDEX_ID}.{key}: value {value} exceeds the CUDA i32 ABI"
            )
        })?;
    }
    Ok((tuple[0], tuple[1]))
}

/// Resolve Disparity Index's exact registered four-parameter point. The
/// canonical feature plan varies only `lookback_period`; the EMA period,
/// smoothing period, and EMA/SMA selector remain their registered defaults.
/// The dynamic CUDA ABI still carries all four values so no fixed-primary
/// replay can masquerade as a RegistryRatio column.
fn resolve_disparity_index_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize, bool), String> {
    let info = get_indicator(DISPARITY_INDEX_ID)
        .ok_or_else(|| format!("{DISPARITY_INDEX_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DISPARITY_INDEX_PARAMETER_KEYS {
        return Err(format!(
            "{DISPARITY_INDEX_ID}: CUDA full ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            DISPARITY_INDEX_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != [DISPARITY_INDEX_OUTPUT_ID] {
        return Err(format!(
            "{DISPARITY_INDEX_ID}: CUDA output ABI [{DISPARITY_INDEX_OUTPUT_ID:?}] != registry \
             {declared_outputs:?}"
        ));
    }
    let planned_outputs = output_ids_for(DISPARITY_INDEX_ID);
    if planned_outputs != [None] {
        return Err(format!(
            "{DISPARITY_INDEX_ID}: admitted single-output receipt [None] != {planned_outputs:?}"
        ));
    }

    const INTEGER_DEFAULTS: [i64; 3] = [14, 14, 9];
    for ((parameter, key), default) in info
        .params
        .iter()
        .take(3)
        .zip(DISPARITY_INDEX_PARAMETER_KEYS)
        .zip(INTEGER_DEFAULTS)
    {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{DISPARITY_INDEX_ID}.{key}: expected optional integer default {default} bounded \
                 1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
                 default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }
    let smoothing_type = &info.params[3];
    if smoothing_type.key != "smoothing_type"
        || !matches!(&smoothing_type.kind, IndicatorParamKind::EnumString)
        || smoothing_type.required
        || smoothing_type.default != Some(ParamValueStatic::EnumString("ema"))
        || smoothing_type.min.is_some()
        || smoothing_type.max.is_some()
        || smoothing_type.step.is_some()
        || smoothing_type.enum_values != ["ema", "sma"]
    {
        return Err(format!(
            "{DISPARITY_INDEX_ID}.smoothing_type: expected optional canonical ema/sma selector \
             with default ema, found kind={:?} required={} default={:?} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            smoothing_type.kind,
            smoothing_type.required,
            smoothing_type.default,
            smoothing_type.min,
            smoothing_type.max,
            smoothing_type.step,
            smoothing_type.enum_values
        ));
    }

    let tuple = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(14),
            require_period_invariant_kernel: false,
        } => (14usize, 14usize, 9usize, false),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DISPARITY_INDEX_ID}: default route requires lookback anchor 14 and a \
                 parameter-consuming typed kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DISPARITY_INDEX_ID}: swept lookback {period} requires the same resolved \
                     anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["lookback_period"] {
                return Err(format!(
                    "{DISPARITY_INDEX_ID}: canonical sweep must override exactly \
                     [\"lookback_period\"], found {override_keys:?}"
                ));
            }
            let lookback_period =
                positive_usize_parameter(DISPARITY_INDEX_ID, "lookback_period", overrides[0].1)?;
            if lookback_period != *period {
                return Err(format!(
                    "{DISPARITY_INDEX_ID}: swept lookback {period} != exact override \
                     {lookback_period}"
                ));
            }
            (14, lookback_period, 9, false)
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DISPARITY_INDEX_ID}: swept tuple requires a resolved lookback anchor, found \
                 {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DISPARITY_INDEX_ID}: discrete parameters cannot enter an f64 value route"
            ));
        }
    };
    for (key, value) in DISPARITY_INDEX_PARAMETER_KEYS
        .into_iter()
        .take(3)
        .zip([tuple.0, tuple.1, tuple.2])
    {
        i32::try_from(value).map_err(|_| {
            format!("{DISPARITY_INDEX_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
    }
    Ok(tuple)
}

/// Resolve DM's sole canonical period for its plus/minus pair. Registry,
/// admitted schema, planning anchor, and explicit sweep override must agree
/// before either output can enter the shared CUDA session.
fn resolve_dm_parameters(parameters: &ClassicCudaParameters) -> std::result::Result<usize, String> {
    let info = get_indicator(DM_ID)
        .ok_or_else(|| format!("{DM_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DM_PARAMETER_KEYS {
        return Err(format!(
            "{DM_ID}: CUDA pair ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            DM_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DM_OUTPUT_IDS {
        return Err(format!(
            "{DM_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            DM_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DM_ID);
    let expected_planned_outputs = DM_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DM_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let declared = &info.params[0];
    if declared.key != "period"
        || !matches!(&declared.kind, IndicatorParamKind::Int)
        || declared.required
        || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || declared.max.is_some()
        || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !declared.enum_values.is_empty()
    {
        return Err(format!(
            "{DM_ID}.period: expected optional integer default 14 bounded 1..=unbounded with \
             step 1 and no enum values, found key={} kind={:?} required={} bounds={:?}..={:?} \
             step={:?} enum={:?}",
            declared.key,
            declared.kind,
            declared.required,
            declared.min,
            declared.max,
            declared.step,
            declared.enum_values
        ));
    }
    let default_period = match declared.default {
        Some(ParamValueStatic::Int(14)) => 14usize,
        other => {
            return Err(format!(
                "{DM_ID}.period: expected exact registry default 14, found {other:?}"
            ));
        }
    };

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(14),
            require_period_invariant_kernel: false,
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DM_ID}: default route requires period anchor 14 and a period-consuming kernel, \
                 found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DM_ID}: swept period {period} requires the same resolved anchor, found \
                     {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != DM_PARAMETER_KEYS {
                return Err(format!(
                    "{DM_ID}: canonical sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    DM_PARAMETER_KEYS
                ));
            }
            let resolved = positive_usize_parameter(DM_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{DM_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DM_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DM_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{DM_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok(period)
}

/// Resolve Donchian's sole canonical period for its upper/middle/lower triple.
/// Registry, admitted schema, planning anchor, and explicit sweep override must
/// agree before any output can enter the shared CUDA session.
fn resolve_donchian_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(DONCHIAN_ID)
        .ok_or_else(|| format!("{DONCHIAN_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DONCHIAN_PARAMETER_KEYS {
        return Err(format!(
            "{DONCHIAN_ID}: CUDA triple ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            DONCHIAN_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DONCHIAN_OUTPUT_IDS {
        return Err(format!(
            "{DONCHIAN_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            DONCHIAN_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(DONCHIAN_ID);
    let expected_planned_outputs = DONCHIAN_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DONCHIAN_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let declared = &info.params[0];
    if declared.key != "period"
        || !matches!(&declared.kind, IndicatorParamKind::Int)
        || declared.required
        || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || declared.max.is_some()
        || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !declared.enum_values.is_empty()
    {
        return Err(format!(
            "{DONCHIAN_ID}.period: expected optional integer default 20 bounded 1..=unbounded \
             with step 1 and no enum values, found key={} kind={:?} required={} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            declared.key,
            declared.kind,
            declared.required,
            declared.min,
            declared.max,
            declared.step,
            declared.enum_values
        ));
    }
    let default_period = match declared.default {
        Some(ParamValueStatic::Int(20)) => 20usize,
        other => {
            return Err(format!(
                "{DONCHIAN_ID}.period: expected exact registry default 20, found {other:?}"
            ));
        }
    };

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(20),
            require_period_invariant_kernel: false,
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DONCHIAN_ID}: default route requires period anchor 20 and a \
                 period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DONCHIAN_ID}: swept period {period} requires the same resolved anchor, \
                     found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != DONCHIAN_PARAMETER_KEYS {
                return Err(format!(
                    "{DONCHIAN_ID}: canonical sweep must override exactly {:?}, found \
                     {override_keys:?}",
                    DONCHIAN_PARAMETER_KEYS
                ));
            }
            let resolved = positive_usize_parameter(DONCHIAN_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{DONCHIAN_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DONCHIAN_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DONCHIAN_ID}: discrete parameters cannot enter an f64 triple route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{DONCHIAN_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok(period)
}

/// Resolve Dual Ulcer Index's complete canonical tuple. Only `period` is an
/// admitted sweep axis; `auto_threshold=true` and `threshold=0.1` remain the
/// exact registry defaults carried into every one-launch triple-output row.
fn resolve_dual_ulcer_index_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, bool, u64), String> {
    let info = get_indicator(DUAL_ULCER_INDEX_ID)
        .ok_or_else(|| format!("{DUAL_ULCER_INDEX_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DUAL_ULCER_INDEX_PARAMETER_KEYS {
        return Err(format!(
            "{DUAL_ULCER_INDEX_ID}: CUDA triple ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            DUAL_ULCER_INDEX_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DUAL_ULCER_INDEX_OUTPUT_IDS {
        return Err(format!(
            "{DUAL_ULCER_INDEX_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            DUAL_ULCER_INDEX_OUTPUT_IDS
        ));
    }
    let expected_planned_outputs = DUAL_ULCER_INDEX_OUTPUT_IDS.map(Some).to_vec();
    let planned_outputs = output_ids_for(DUAL_ULCER_INDEX_ID);
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DUAL_ULCER_INDEX_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let period_metadata = &info.params[0];
    if !matches!(&period_metadata.kind, IndicatorParamKind::Int)
        || period_metadata.required
        || period_metadata.default != Some(ParamValueStatic::Int(5))
        || period_metadata.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_metadata.max.is_some()
        || period_metadata.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_metadata.enum_values.is_empty()
    {
        return Err(format!(
            "{DUAL_ULCER_INDEX_ID}.period: expected optional integer default 5 bounded \
             1..=unbounded with step 1 and no enum values, found kind={:?} required={} \
             default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            period_metadata.kind,
            period_metadata.required,
            period_metadata.default,
            period_metadata.min,
            period_metadata.max,
            period_metadata.step,
            period_metadata.enum_values
        ));
    }
    let auto_threshold_metadata = &info.params[1];
    if !matches!(&auto_threshold_metadata.kind, IndicatorParamKind::Bool)
        || auto_threshold_metadata.required
        || auto_threshold_metadata.default != Some(ParamValueStatic::Bool(true))
        || auto_threshold_metadata.min.is_some()
        || auto_threshold_metadata.max.is_some()
        || auto_threshold_metadata.step.is_some()
        || !auto_threshold_metadata.enum_values.is_empty()
    {
        return Err(format!(
            "{DUAL_ULCER_INDEX_ID}.auto_threshold: expected optional boolean default true \
             without numeric bounds/step or enum values, found kind={:?} required={} \
             default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            auto_threshold_metadata.kind,
            auto_threshold_metadata.required,
            auto_threshold_metadata.default,
            auto_threshold_metadata.min,
            auto_threshold_metadata.max,
            auto_threshold_metadata.step,
            auto_threshold_metadata.enum_values
        ));
    }
    let threshold_metadata = &info.params[2];
    let threshold = match threshold_metadata.default {
        Some(ParamValueStatic::Float(value)) if value.to_bits() == 0.1_f64.to_bits() => value,
        other => {
            return Err(format!(
                "{DUAL_ULCER_INDEX_ID}.threshold: expected exact f64 default 0.1, found \
                 {other:?}"
            ));
        }
    };
    if !matches!(&threshold_metadata.kind, IndicatorParamKind::Float)
        || threshold_metadata.required
        || threshold_metadata.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || threshold_metadata.max.is_some()
        || threshold_metadata.step.map(f64::to_bits) != Some(0.1_f64.to_bits())
        || !threshold_metadata.enum_values.is_empty()
    {
        return Err(format!(
            "{DUAL_ULCER_INDEX_ID}.threshold: expected optional finite f64 default 0.1 bounded \
             0..=unbounded with step 0.1 and no enum values, found kind={:?} required={} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            threshold_metadata.kind,
            threshold_metadata.required,
            threshold_metadata.min,
            threshold_metadata.max,
            threshold_metadata.step,
            threshold_metadata.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(5),
            require_period_invariant_kernel: false,
        } => 5,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DUAL_ULCER_INDEX_ID}: default route requires period anchor 5 and a \
                 period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DUAL_ULCER_INDEX_ID}: swept period {period} requires the same resolved \
                     anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{DUAL_ULCER_INDEX_ID}: canonical period-only sweep must override exactly \
                     [\"period\"], found {override_keys:?}"
                ));
            }
            let resolved = positive_usize_parameter(DUAL_ULCER_INDEX_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{DUAL_ULCER_INDEX_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DUAL_ULCER_INDEX_ID}: swept tuple requires a resolved period anchor, found \
                 {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DUAL_ULCER_INDEX_ID}: discrete parameters cannot enter an f64 triple route"
            ));
        }
    };
    i32::try_from(period).map_err(|_| {
        format!("{DUAL_ULCER_INDEX_ID}.period: value {period} exceeds the CUDA i32 ABI")
    })?;
    Ok((period, true, threshold.to_bits()))
}

/// Resolve DVDIQQE's complete canonical seven-parameter tuple. The current
/// admitted vocabulary varies only `period`, but every default remains part of
/// the explicit full-output ABI so a registry drift fails before allocation.
fn resolve_dvdiqqe_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, u64, u64, bool, bool, u64), String> {
    let info = get_indicator(DVDIQQE_ID)
        .ok_or_else(|| format!("{DVDIQQE_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != DVDIQQE_PARAMETER_KEYS {
        return Err(format!(
            "{DVDIQQE_ID}: CUDA four-output ABI requires exact registry parameters {:?}, found \
             {declared_keys:?}",
            DVDIQQE_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != DVDIQQE_OUTPUT_IDS {
        return Err(format!(
            "{DVDIQQE_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            DVDIQQE_OUTPUT_IDS
        ));
    }
    let expected_planned_outputs = DVDIQQE_OUTPUT_IDS.map(Some).to_vec();
    let planned_outputs = output_ids_for(DVDIQQE_ID);
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{DVDIQQE_ID}: admitted output schema {expected_planned_outputs:?} != \
             {planned_outputs:?}"
        ));
    }

    let exact_positive_int =
        |index: usize, key: &'static str, expected: i64| -> std::result::Result<usize, String> {
            let parameter = &info.params[index];
            if parameter.key != key
                || !matches!(&parameter.kind, IndicatorParamKind::Int)
                || parameter.required
                || parameter.default != Some(ParamValueStatic::Int(expected))
                || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
                || parameter.max.is_some()
                || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
                || !parameter.enum_values.is_empty()
            {
                return Err(format!(
                    "{DVDIQQE_ID}.{key}: expected optional integer default {expected} bounded \
                 1..=unbounded with step 1 and no enum values, found key={} kind={:?} \
                 required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                    parameter.key,
                    parameter.kind,
                    parameter.required,
                    parameter.default,
                    parameter.min,
                    parameter.max,
                    parameter.step,
                    parameter.enum_values
                ));
            }
            let value = positive_usize_parameter(DVDIQQE_ID, key, expected)?;
            i32::try_from(value).map_err(|_| {
                format!("{DVDIQQE_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
            })?;
            Ok(value)
        };
    let default_period = exact_positive_int(0, "period", 13)?;
    let smoothing_period = exact_positive_int(1, "smoothing_period", 6)?;

    let exact_positive_float =
        |index: usize, key: &'static str, expected: f64| -> std::result::Result<f64, String> {
            let parameter = &info.params[index];
            let value = match parameter.default {
                Some(ParamValueStatic::Float(value))
                    if value.is_finite() && value.to_bits() == expected.to_bits() =>
                {
                    value
                }
                other => {
                    return Err(format!(
                        "{DVDIQQE_ID}.{key}: expected exact finite f64 default {expected}, found \
                     {other:?}"
                    ));
                }
            };
            if parameter.key != key
                || !matches!(&parameter.kind, IndicatorParamKind::Float)
                || parameter.required
                || parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
                || parameter.max.is_some()
                || parameter.step.is_some()
                || !parameter.enum_values.is_empty()
                || value <= 0.0
            {
                return Err(format!(
                    "{DVDIQQE_ID}.{key}: expected optional positive f64 default {expected} with \
                 declared minimum 0, no maximum/step/enum, found key={} kind={:?} required={} \
                 bounds={:?}..={:?} step={:?} enum={:?}",
                    parameter.key,
                    parameter.kind,
                    parameter.required,
                    parameter.min,
                    parameter.max,
                    parameter.step,
                    parameter.enum_values
                ));
            }
            Ok(value)
        };
    let fast_multiplier = exact_positive_float(2, "fast_multiplier", 2.618)?;
    let slow_multiplier = exact_positive_float(3, "slow_multiplier", 4.236)?;

    let exact_enum_default = |index: usize,
                              key: &'static str,
                              expected: &'static str|
     -> std::result::Result<&'static str, String> {
        let parameter = &info.params[index];
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::EnumString)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::EnumString(expected))
            || parameter.min.is_some()
            || parameter.max.is_some()
            || parameter.step.is_some()
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{DVDIQQE_ID}.{key}: expected optional enum default `{expected}` without \
                 numeric bounds/step or alternate values, found key={} kind={:?} required={} \
                 default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
        Ok(expected)
    };
    let volume_type = exact_enum_default(4, "volume_type", "default")?;
    let center_type = exact_enum_default(5, "center_type", "dynamic")?;
    let tick_size = exact_positive_float(6, "tick_size", 0.01)?;

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(13),
            require_period_invariant_kernel: false,
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{DVDIQQE_ID}: default route requires period anchor 13 and a period-consuming \
                 kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{DVDIQQE_ID}: swept period {period} requires the same resolved anchor, \
                     found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{DVDIQQE_ID}: canonical period-only sweep must override exactly \
                     [\"period\"], found {override_keys:?}"
                ));
            }
            let resolved = positive_usize_parameter(DVDIQQE_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{DVDIQQE_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{DVDIQQE_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{DVDIQQE_ID}: discrete parameters cannot enter an f64 four-output route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{DVDIQQE_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    let wper = period
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| format!("{DVDIQQE_ID}.period: {period} double-period overflow"))?;
    i32::try_from(wper).map_err(|_| {
        format!("{DVDIQQE_ID}.period: derived double period {wper} exceeds the CUDA i32 ABI")
    })?;

    Ok((
        period,
        smoothing_period,
        fast_multiplier.to_bits(),
        slow_multiplier.to_bits(),
        volume_type == "tick",
        center_type == "dynamic",
        tick_size.to_bits(),
    ))
}

/// Resolve the exact registered spectrum tuple for the canonical Ehlers
/// Autocorrelation Periodogram pair. The preserved generic primary ABI remains
/// fixed at 8:48:3:true; this typed production route consumes the complete
/// RegistryRatio tuple carried by every admitted base/sweep receipt.
fn resolve_ehlers_autocorrelation_periodogram_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize, bool), String> {
    let info = get_indicator(EHLERS_AUTOCORRELATION_PERIODOGRAM_ID).ok_or_else(|| {
        format!("{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: absent from the vector-ta registry")
    })?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS {
        return Err(format!(
            "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: full CUDA output ABI {:?} != registry \
             {declared_outputs:?}",
            EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EHLERS_AUTOCORRELATION_PERIODOGRAM_ID);
    let expected_planned_outputs = EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
        .map(Some)
        .to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EHLERS_AUTOCORRELATION_PERIODOGRAM_PARAMETER_KEYS {
        return Err(format!(
            "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: exact two-output ABI requires registry \
             parameters {:?}, found {declared_keys:?}",
            EHLERS_AUTOCORRELATION_PERIODOGRAM_PARAMETER_KEYS
        ));
    }
    for (parameter, key, default, minimum) in [
        (&info.params[0], "min_period", 8_i64, 3.0_f64),
        (&info.params[1], "max_period", 48_i64, 4.0_f64),
        (&info.params[2], "avg_length", 3_i64, 0.0_f64),
    ] {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default))
            || parameter.min.map(f64::to_bits) != Some(minimum.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}.{key}: expected optional integer \
                 default {default} bounded {minimum}..=unbounded with step 1 and no enum values, \
                 found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }
    let enhance = &info.params[3];
    if enhance.key != "enhance"
        || !matches!(&enhance.kind, IndicatorParamKind::Bool)
        || enhance.required
        || enhance.default != Some(ParamValueStatic::Bool(true))
        || enhance.min.is_some()
        || enhance.max.is_some()
        || enhance.step.is_some()
        || enhance.enum_values != ["true", "false"]
    {
        return Err(format!(
            "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}.enhance: expected optional bool default \
             true with canonical true/false enum, found kind={:?} required={} default={:?} \
             bounds={:?}..={:?} step={:?} enum={:?}",
            enhance.kind,
            enhance.required,
            enhance.default,
            enhance.min,
            enhance.max,
            enhance.step,
            enhance.enum_values
        ));
    }

    let tuple = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(48),
            require_period_invariant_kernel: false,
        } => (8usize, 48usize, 3usize),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: default route requires \
                 RegistryRatio anchor 48 without a no-window classification, found \
                 {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: swept period {period} requires \
                     the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides =
                classic_cuda_sweep_params(EHLERS_AUTOCORRELATION_PERIODOGRAM_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: period {period} requires exact \
                     RegistryRatio overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let mut values = [0usize; 3];
            for (index, (key, value)) in overrides.iter().enumerate() {
                values[index] = usize::try_from(*value).map_err(|_| {
                    format!(
                        "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}.{key}: expected a non-negative \
                         integer, found {value}"
                    )
                })?;
            }
            (values[0], values[1], values[2])
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: swept tuple requires a resolved \
                 RegistryRatio anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: discrete parameters cannot enter an \
                 f64 pair route"
            ));
        }
    };
    if tuple.0 < 3 || tuple.1 < 4 || tuple.1 <= tuple.0 {
        return Err(format!(
            "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: invalid resolved spectrum tuple \
             min_period={}, max_period={}, avg_length={}",
            tuple.0, tuple.1, tuple.2
        ));
    }
    for (key, value) in EHLERS_AUTOCORRELATION_PERIODOGRAM_PARAMETER_KEYS
        .into_iter()
        .take(3)
        .zip([tuple.0, tuple.1, tuple.2])
    {
        i32::try_from(value).map_err(|_| {
            format!(
                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}.{key}: value {value} exceeds the CUDA \
                 i32 ABI"
            )
        })?;
    }
    Ok((tuple.0, tuple.1, tuple.2, true))
}

/// Resolve ELEP's complete canonical five-output tuple. The preserved generic
/// primary ABI remains fixed at defaults; this typed production route carries
/// the exact RegistryRatio high/low-pass windows and bypasses that ABI for all
/// five admitted receipts, including `prediction`.
fn resolve_ehlers_linear_extrapolation_predictor_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, u64, usize, i32), String> {
    let info = get_indicator(EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID).ok_or_else(|| {
        format!("{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS {
        return Err(format!(
            "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID);
    let expected_planned_outputs = EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
        .map(Some)
        .to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_PARAMETER_KEYS {
        return Err(format!(
            "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: exact five-output ABI requires registry parameters {:?}, found {declared_keys:?}",
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_PARAMETER_KEYS
        ));
    }
    for (parameter, key, default) in [
        (&info.params[0], "high_pass_length", 125_i64),
        (&info.params[1], "low_pass_length", 12_i64),
    ] {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}.{key}: expected optional positive integer default {default} with step 1, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }
    let gain = &info.params[2];
    if gain.key != "gain"
        || !matches!(&gain.kind, IndicatorParamKind::Float)
        || gain.required
        || gain.default != Some(ParamValueStatic::Float(0.7))
        || gain.min.is_some()
        || gain.max.is_some()
        || gain.step.is_some()
        || !gain.enum_values.is_empty()
    {
        return Err(format!(
            "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}.gain: expected optional exact f64 default 0.7 without bounds/step/enum, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            gain.kind, gain.required, gain.default, gain.min, gain.max, gain.step, gain.enum_values
        ));
    }
    let bars_forward = &info.params[3];
    if bars_forward.key != "bars_forward"
        || !matches!(&bars_forward.kind, IndicatorParamKind::Int)
        || bars_forward.required
        || bars_forward.default != Some(ParamValueStatic::Int(5))
        || bars_forward.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || bars_forward.max.map(f64::to_bits) != Some(10.0_f64.to_bits())
        || bars_forward.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !bars_forward.enum_values.is_empty()
    {
        return Err(format!(
            "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}.bars_forward: expected optional integer default 5 bounded 0..=10 with step 1, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            bars_forward.kind,
            bars_forward.required,
            bars_forward.default,
            bars_forward.min,
            bars_forward.max,
            bars_forward.step,
            bars_forward.enum_values
        ));
    }
    let signal_mode = &info.params[4];
    let expected_modes = [
        "predict_filter_crosses",
        "predict_middle_crosses",
        "filter_middle_crosses",
    ];
    if signal_mode.key != "signal_mode"
        || !matches!(&signal_mode.kind, IndicatorParamKind::EnumString)
        || signal_mode.required
        || signal_mode.default != Some(ParamValueStatic::EnumString("predict_filter_crosses"))
        || signal_mode.min.is_some()
        || signal_mode.max.is_some()
        || signal_mode.step.is_some()
        || signal_mode.enum_values != expected_modes
    {
        return Err(format!(
            "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}.signal_mode: expected optional canonical enum {expected_modes:?} default predict_filter_crosses, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            signal_mode.kind,
            signal_mode.required,
            signal_mode.default,
            signal_mode.min,
            signal_mode.max,
            signal_mode.step,
            signal_mode.enum_values
        ));
    }

    let (high_pass_length, low_pass_length) = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(125),
            require_period_invariant_kernel: false,
        } => (125usize, 12usize),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: default route requires RegistryRatio anchor 125 without a no-window classification, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: swept period {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides =
                classic_cuda_sweep_params(EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: period {period} requires exact RegistryRatio overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [(_, high_pass), (_, low_pass)] = overrides.as_slice() else {
                return Err(format!(
                    "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: expected two exact window overrides, found {overrides:?}"
                ));
            };
            (
                positive_usize_parameter(
                    EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID,
                    "high_pass_length",
                    *high_pass,
                )?,
                positive_usize_parameter(
                    EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID,
                    "low_pass_length",
                    *low_pass,
                )?,
            )
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: swept tuple requires a resolved RegistryRatio anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: discrete parameters cannot enter an f64 quint route"
            ));
        }
    };
    for (key, value) in [
        ("high_pass_length", high_pass_length),
        ("low_pass_length", low_pass_length),
        ("bars_forward", 5usize),
    ] {
        i32::try_from(value).map_err(|_| {
            format!(
                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}.{key}: value {value} exceeds the CUDA i32 ABI"
            )
        })?;
    }
    Ok((high_pass_length, low_pass_length, 0.7_f64.to_bits(), 5, 0))
}

/// Resolve EUDMA's exact coupled Hann tuple. The generic primary ABI remains
/// fixed at defaults for compatibility, so the typed pair route must carry all
/// three RegistryRatio windows and must own both canonical outputs.
fn resolve_ehlers_undersampled_double_moving_average_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize), String> {
    let info = get_indicator(EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID).ok_or_else(|| {
        format!(
            "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: absent from the vector-ta registry"
        )
    })?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS {
        return Err(format!(
            "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID);
    let expected_planned_outputs = EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
        .map(Some)
        .to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_PARAMETER_KEYS {
        return Err(format!(
            "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: exact pair ABI requires registry parameters {:?}, found {declared_keys:?}",
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_PARAMETER_KEYS
        ));
    }
    for (parameter, key, default) in [
        (&info.params[0], "fast_length", 6_i64),
        (&info.params[1], "slow_length", 12_i64),
        (&info.params[2], "sample_length", 5_i64),
    ] {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.map(f64::to_bits) != Some(4096.0_f64.to_bits())
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}.{key}: expected optional integer default {default} bounded 1..=4096 with step 1, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }
    let output = &info.params[3];
    if output.key != "output"
        || !matches!(&output.kind, IndicatorParamKind::EnumString)
        || output.required
        || output.default != Some(ParamValueStatic::EnumString("fast"))
        || output.min.is_some()
        || output.max.is_some()
        || output.step.is_some()
        || output.enum_values != ["fast", "slow"]
    {
        return Err(format!(
            "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}.output: expected optional canonical [fast, slow] enum default fast, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            output.kind,
            output.required,
            output.default,
            output.min,
            output.max,
            output.step,
            output.enum_values
        ));
    }

    let tuple = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(12),
            require_period_invariant_kernel: false,
        } => (6usize, 12usize, 5usize),
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: default route requires RegistryRatio anchor 12 without a no-window classification, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: swept period {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides =
                classic_cuda_sweep_params(EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: period {period} requires exact RegistryRatio overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [(_, fast), (_, slow), (_, sample)] = overrides.as_slice() else {
                return Err(format!(
                    "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: expected three exact window overrides, found {overrides:?}"
                ));
            };
            (
                positive_usize_parameter(
                    EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID,
                    "fast_length",
                    *fast,
                )?,
                positive_usize_parameter(
                    EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID,
                    "slow_length",
                    *slow,
                )?,
                positive_usize_parameter(
                    EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID,
                    "sample_length",
                    *sample,
                )?,
            )
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: swept tuple requires a resolved RegistryRatio anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    for (key, value) in [
        ("fast_length", tuple.0),
        ("slow_length", tuple.1),
        ("sample_length", tuple.2),
    ] {
        if value > 4096 {
            return Err(format!(
                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}.{key}: value {value} exceeds the canonical 4096 bound"
            ));
        }
        i32::try_from(value).map_err(|_| {
            format!(
                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}.{key}: value {value} exceeds the CUDA i32 ABI"
            )
        })?;
    }
    Ok(tuple)
}

/// Resolve EDCT3's exact scalar/batch defaults and its one admitted period
/// dimension. The old generic moving-average period-14 default changed the
/// unsuffixed feature under the same identity, so only the canonical period-10
/// registry receipt may enter this typed pair route.
fn resolve_ema_deviation_corrected_t3_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, usize), String> {
    let info = get_indicator(EMA_DEVIATION_CORRECTED_T3_ID)
        .ok_or_else(|| format!("{EMA_DEVIATION_CORRECTED_T3_ID}: absent from the registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EMA_DEVIATION_CORRECTED_T3_ID);
    let expected_planned_outputs = EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EMA_DEVIATION_CORRECTED_T3_PARAMETER_KEYS {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}: exact pair ABI requires registry parameters {:?}, found {declared_keys:?}",
            EMA_DEVIATION_CORRECTED_T3_PARAMETER_KEYS
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.default != Some(ParamValueStatic::Int(10))
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}.period: expected optional integer default 10 bounded 1..=unbounded with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.default,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }
    let hot_parameter = &info.params[1];
    if !matches!(&hot_parameter.kind, IndicatorParamKind::Float)
        || hot_parameter.required
        || hot_parameter.default != Some(ParamValueStatic::Float(0.7))
        || hot_parameter.min.map(f64::to_bits) != Some((-16.0_f64).to_bits())
        || hot_parameter.max.map(f64::to_bits) != Some(16.0_f64.to_bits())
        || hot_parameter.step.map(f64::to_bits) != Some(0.01_f64.to_bits())
        || !hot_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}.hot: expected optional finite default 0.7 bounded -16..=16 with step 0.01 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            hot_parameter.kind,
            hot_parameter.required,
            hot_parameter.default,
            hot_parameter.min,
            hot_parameter.max,
            hot_parameter.step,
            hot_parameter.enum_values
        ));
    }
    let mode_parameter = &info.params[2];
    if !matches!(&mode_parameter.kind, IndicatorParamKind::Int)
        || mode_parameter.required
        || mode_parameter.default != Some(ParamValueStatic::Int(0))
        || mode_parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || mode_parameter.max.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || mode_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !mode_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}.t3_mode: expected optional integer default 0 bounded 0..=1 with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            mode_parameter.kind,
            mode_parameter.required,
            mode_parameter.default,
            mode_parameter.min,
            mode_parameter.max,
            mode_parameter.step,
            mode_parameter.enum_values
        ));
    }
    let output_parameter = &info.params[3];
    if !matches!(&output_parameter.kind, IndicatorParamKind::EnumString)
        || output_parameter.required
        || output_parameter.default != Some(ParamValueStatic::EnumString("corrected"))
        || output_parameter.min.is_some()
        || output_parameter.max.is_some()
        || output_parameter.step.is_some()
        || output_parameter.enum_values != EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
    {
        return Err(format!(
            "{EMA_DEVIATION_CORRECTED_T3_ID}.output: expected optional canonical {:?} enum default corrected, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS,
            output_parameter.kind,
            output_parameter.required,
            output_parameter.default,
            output_parameter.min,
            output_parameter.max,
            output_parameter.step,
            output_parameter.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(10),
            require_period_invariant_kernel: false,
        } => 10usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EMA_DEVIATION_CORRECTED_T3_ID}: default route requires the canonical period-10 anchor and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EMA_DEVIATION_CORRECTED_T3_ID}: swept period {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides =
                classic_cuda_sweep_params(EMA_DEVIATION_CORRECTED_T3_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EMA_DEVIATION_CORRECTED_T3_ID}: period {period} requires exact RegistryRatio overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [("period", resolved)] = overrides.as_slice() else {
                return Err(format!(
                    "{EMA_DEVIATION_CORRECTED_T3_ID}: expected one exact period override, found {overrides:?}"
                ));
            };
            let resolved =
                positive_usize_parameter(EMA_DEVIATION_CORRECTED_T3_ID, "period", *resolved)?;
            if resolved != *period {
                return Err(format!(
                    "{EMA_DEVIATION_CORRECTED_T3_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EMA_DEVIATION_CORRECTED_T3_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EMA_DEVIATION_CORRECTED_T3_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    i32::try_from(period).map_err(|_| {
        format!("{EMA_DEVIATION_CORRECTED_T3_ID}.period: value {period} exceeds the CUDA i32 ABI")
    })?;
    Ok((period, 0.7_f64.to_bits(), 0))
}

/// Resolve EMD's one admitted period dimension while pinning the exact scalar
/// CPU delta/fraction defaults. Registry identity, admitted receipts, anchor,
/// and RegistryRatio overrides must agree before the resident triple route is
/// allowed to allocate scratch or launch CUDA.
fn resolve_emd_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, u64), String> {
    let info = get_indicator(EMD_ID)
        .ok_or_else(|| format!("{EMD_ID}: absent from the vector-ta registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EMD_OUTPUT_IDS {
        return Err(format!(
            "{EMD_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            EMD_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EMD_ID);
    let expected_planned_outputs = EMD_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EMD_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EMD_PARAMETER_KEYS {
        return Err(format!(
            "{EMD_ID}: exact triple ABI requires registry parameters {:?}, found {declared_keys:?}",
            EMD_PARAMETER_KEYS
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.default != Some(ParamValueStatic::Int(20))
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EMD_ID}.period: expected optional integer default 20 bounded 1..=unbounded with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.default,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }
    for (index, key, default) in [(1, "delta", 0.5_f64), (2, "fraction", 0.1_f64)] {
        let parameter = &info.params[index];
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Float)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Float(default))
            || parameter.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.is_some()
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{EMD_ID}.{key}: expected optional finite default {default} bounded 0..=unbounded with no step or enum values, found key={} kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(20),
            require_period_invariant_kernel: false,
        } => 20usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EMD_ID}: default route requires the canonical period-20 anchor and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EMD_ID}: swept period {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides = classic_cuda_sweep_params(EMD_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EMD_ID}: period {period} requires exact RegistryRatio overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [("period", resolved)] = overrides.as_slice() else {
                return Err(format!(
                    "{EMD_ID}: expected one exact period override, found {overrides:?}"
                ));
            };
            let resolved = positive_usize_parameter(EMD_ID, "period", *resolved)?;
            if resolved != *period {
                return Err(format!(
                    "{EMD_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EMD_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EMD_ID}: discrete parameters cannot enter an f64 triple route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{EMD_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok((period, 0.5_f64.to_bits(), 0.1_f64.to_bits()))
}

/// Resolve the newly registered EMD Trend identity without admitting the
/// retired anonymous average-only artifact. Production is intentionally
/// narrower than the public scalar surface: close + SMA are pinned, length is
/// the sole admitted RegistryRatio dimension, and the exact multiplier is
/// carried into the resident four-output launch.
fn resolve_emd_trend_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64), String> {
    const SOURCE_VALUES: [&str; 10] = [
        "open", "high", "low", "close", "oc2", "hl2", "occ3", "hlc3", "ohlc4", "hlcc4",
    ];
    const AVG_TYPE_VALUES: [&str; 7] = ["SMA", "EMA", "HMA", "DEMA", "TEMA", "RMA", "FRAMA"];

    let info = get_indicator(EMD_TREND_ID)
        .ok_or_else(|| format!("{EMD_TREND_ID}: absent from the vector-ta registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EMD_TREND_OUTPUT_IDS {
        return Err(format!(
            "{EMD_TREND_ID}: exact four-output CUDA ABI {:?} != registry {declared_outputs:?}",
            EMD_TREND_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EMD_TREND_ID);
    let expected_planned_outputs = EMD_TREND_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EMD_TREND_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EMD_TREND_PARAMETER_KEYS {
        return Err(format!(
            "{EMD_TREND_ID}: exact production ABI requires registry parameters {:?}, found {declared_keys:?}",
            EMD_TREND_PARAMETER_KEYS
        ));
    }

    let source = &info.params[0];
    if !matches!(&source.kind, IndicatorParamKind::EnumString)
        || source.required
        || source.default != Some(ParamValueStatic::EnumString("close"))
        || source.min.is_some()
        || source.max.is_some()
        || source.step.is_some()
        || source.enum_values != SOURCE_VALUES
    {
        return Err(format!(
            "{EMD_TREND_ID}.source: expected optional exact close default and canonical {SOURCE_VALUES:?} enum, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            source.kind,
            source.required,
            source.default,
            source.min,
            source.max,
            source.step,
            source.enum_values
        ));
    }
    let avg_type = &info.params[1];
    if !matches!(&avg_type.kind, IndicatorParamKind::EnumString)
        || avg_type.required
        || avg_type.default != Some(ParamValueStatic::EnumString("SMA"))
        || avg_type.min.is_some()
        || avg_type.max.is_some()
        || avg_type.step.is_some()
        || avg_type.enum_values != AVG_TYPE_VALUES
    {
        return Err(format!(
            "{EMD_TREND_ID}.avg_type: expected optional exact SMA default and canonical {AVG_TYPE_VALUES:?} enum, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            avg_type.kind,
            avg_type.required,
            avg_type.default,
            avg_type.min,
            avg_type.max,
            avg_type.step,
            avg_type.enum_values
        ));
    }
    let length_parameter = &info.params[2];
    if !matches!(&length_parameter.kind, IndicatorParamKind::Int)
        || length_parameter.required
        || length_parameter.default != Some(ParamValueStatic::Int(28))
        || length_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || length_parameter.max.is_some()
        || length_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !length_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EMD_TREND_ID}.length: expected optional integer default 28 bounded 1..=unbounded with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            length_parameter.kind,
            length_parameter.required,
            length_parameter.default,
            length_parameter.min,
            length_parameter.max,
            length_parameter.step,
            length_parameter.enum_values
        ));
    }
    let mult_parameter = &info.params[3];
    if !matches!(&mult_parameter.kind, IndicatorParamKind::Float)
        || mult_parameter.required
        || mult_parameter.default != Some(ParamValueStatic::Float(1.0))
        || mult_parameter.min.map(f64::to_bits) != Some(0.05_f64.to_bits())
        || mult_parameter.max.is_some()
        || mult_parameter.step.is_some()
        || !mult_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EMD_TREND_ID}.mult: expected optional finite default 1 bounded 0.05..=unbounded with no step or enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            mult_parameter.kind,
            mult_parameter.required,
            mult_parameter.default,
            mult_parameter.min,
            mult_parameter.max,
            mult_parameter.step,
            mult_parameter.enum_values
        ));
    }

    let length = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(28),
            require_period_invariant_kernel: false,
        } => 28usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EMD_TREND_ID}: default route requires the canonical length-28 anchor and a length-consuming typed kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EMD_TREND_ID}: swept length {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides = classic_cuda_sweep_params(EMD_TREND_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EMD_TREND_ID}: length {period} requires exact RegistryRatio overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [("length", resolved)] = overrides.as_slice() else {
                return Err(format!(
                    "{EMD_TREND_ID}: expected one exact length override, found {overrides:?}"
                ));
            };
            let resolved = positive_usize_parameter(EMD_TREND_ID, "length", *resolved)?;
            if resolved != *period {
                return Err(format!(
                    "{EMD_TREND_ID}: swept length {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EMD_TREND_ID}: swept tuple requires a resolved length anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EMD_TREND_ID}: discrete parameters cannot enter an f64 four-output route"
            ));
        }
    };
    i32::try_from(length)
        .map_err(|_| format!("{EMD_TREND_ID}.length: value {length} exceeds the CUDA i32 ABI"))?;
    Ok((length, 1.0_f64.to_bits()))
}

/// Resolve ERI's one admitted period dimension while proving that production
/// stays on the scalar CPU's exact EMA authority. The public indicator can
/// compute other moving-average families, but Classic admits only the registry
/// default plus period-only sweeps and must fail before launch if that contract
/// changes.
fn resolve_eri_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(ERI_ID)
        .ok_or_else(|| format!("{ERI_ID}: absent from the vector-ta registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != ERI_OUTPUT_IDS {
        return Err(format!(
            "{ERI_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            ERI_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(ERI_ID);
    let expected_planned_outputs = ERI_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{ERI_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != ERI_PARAMETER_KEYS {
        return Err(format!(
            "{ERI_ID}: exact pair ABI requires registry parameters {:?}, found {declared_keys:?}",
            ERI_PARAMETER_KEYS
        ));
    }

    let period_parameter = &info.params[0];
    if !matches!(&period_parameter.kind, IndicatorParamKind::Int)
        || period_parameter.required
        || period_parameter.default != Some(ParamValueStatic::Int(13))
        || period_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || period_parameter.max.is_some()
        || period_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !period_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{ERI_ID}.period: expected optional integer default 13 bounded 1..=unbounded with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            period_parameter.kind,
            period_parameter.required,
            period_parameter.default,
            period_parameter.min,
            period_parameter.max,
            period_parameter.step,
            period_parameter.enum_values
        ));
    }
    let ma_type_parameter = &info.params[1];
    if !matches!(&ma_type_parameter.kind, IndicatorParamKind::EnumString)
        || ma_type_parameter.required
        || ma_type_parameter.default != Some(ParamValueStatic::EnumString("ema"))
        || ma_type_parameter.min.is_some()
        || ma_type_parameter.max.is_some()
        || ma_type_parameter.step.is_some()
        || !ma_type_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{ERI_ID}.ma_type: exact Classic CUDA authority requires optional enum-string default `ema` with no bounds, step or aliases, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            ma_type_parameter.kind,
            ma_type_parameter.required,
            ma_type_parameter.default,
            ma_type_parameter.min,
            ma_type_parameter.max,
            ma_type_parameter.step,
            ma_type_parameter.enum_values
        ));
    }

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(13),
            require_period_invariant_kernel: false,
        } => 13usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{ERI_ID}: default route requires the canonical period-13 anchor and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{ERI_ID}: swept period {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let expected_overrides = classic_cuda_sweep_params(ERI_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{ERI_ID}: period {period} requires exact period overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [("period", resolved)] = overrides.as_slice() else {
                return Err(format!(
                    "{ERI_ID}: expected one exact period override, found {overrides:?}"
                ));
            };
            let resolved = positive_usize_parameter(ERI_ID, "period", *resolved)?;
            if resolved != *period {
                return Err(format!(
                    "{ERI_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{ERI_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{ERI_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{ERI_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok(period)
}

/// Resolve Evasive Supertrend's canonical ATR-length dimension while pinning
/// the other three scalar parameters to their exact registry defaults. Classic
/// admits the base plus atr_length-only sweeps; any schema or tuple drift must
/// fail before allocation or the first launch.
fn resolve_evasive_supertrend_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, u64, u64), String> {
    let info = get_indicator(EVASIVE_SUPERTREND_ID)
        .ok_or_else(|| format!("{EVASIVE_SUPERTREND_ID}: absent from the vector-ta registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EVASIVE_SUPERTREND_OUTPUT_IDS {
        return Err(format!(
            "{EVASIVE_SUPERTREND_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            EVASIVE_SUPERTREND_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EVASIVE_SUPERTREND_ID);
    let expected_planned_outputs = EVASIVE_SUPERTREND_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EVASIVE_SUPERTREND_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EVASIVE_SUPERTREND_PARAMETER_KEYS {
        return Err(format!(
            "{EVASIVE_SUPERTREND_ID}: exact quad ABI requires registry parameters {:?}, found {declared_keys:?}",
            EVASIVE_SUPERTREND_PARAMETER_KEYS
        ));
    }

    let atr_length_parameter = &info.params[0];
    if !matches!(&atr_length_parameter.kind, IndicatorParamKind::Int)
        || atr_length_parameter.required
        || atr_length_parameter.default != Some(ParamValueStatic::Int(10))
        || atr_length_parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || atr_length_parameter.max.is_some()
        || atr_length_parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !atr_length_parameter.enum_values.is_empty()
    {
        return Err(format!(
            "{EVASIVE_SUPERTREND_ID}.atr_length: expected optional integer default 10 bounded 1..=unbounded with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            atr_length_parameter.kind,
            atr_length_parameter.required,
            atr_length_parameter.default,
            atr_length_parameter.min,
            atr_length_parameter.max,
            atr_length_parameter.step,
            atr_length_parameter.enum_values
        ));
    }

    let expected_float_parameters = [
        ("base_multiplier", 3.0_f64, 0.1_f64),
        ("noise_threshold", 1.0_f64, 0.1_f64),
        ("expansion_alpha", 0.5_f64, 0.0_f64),
    ];
    for (parameter, (key, default, min)) in info.params[1..].iter().zip(expected_float_parameters) {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Float)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Float(default))
            || parameter.min.map(f64::to_bits) != Some(min.to_bits())
            || parameter.max.is_some()
            || parameter.step.is_some()
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{EVASIVE_SUPERTREND_ID}.{key}: expected optional finite float default {default:?} bounded {min:?}..=unbounded with no step or enum values, found key={} kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let atr_length = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(10),
            require_period_invariant_kernel: false,
        } => 10usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EVASIVE_SUPERTREND_ID}: default route requires the canonical atr_length-10 anchor and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EVASIVE_SUPERTREND_ID}: swept period {period} requires the same resolved atr_length anchor, found {anchor}"
                ));
            }
            let expected_overrides = classic_cuda_sweep_params(EVASIVE_SUPERTREND_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{EVASIVE_SUPERTREND_ID}: period {period} requires exact atr_length overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [("atr_length", resolved)] = overrides.as_slice() else {
                return Err(format!(
                    "{EVASIVE_SUPERTREND_ID}: expected one exact atr_length override, found {overrides:?}"
                ));
            };
            let resolved =
                positive_usize_parameter(EVASIVE_SUPERTREND_ID, "atr_length", *resolved)?;
            if resolved != *period {
                return Err(format!(
                    "{EVASIVE_SUPERTREND_ID}: swept period {period} != exact atr_length override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EVASIVE_SUPERTREND_ID}: swept tuple requires a resolved atr_length anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EVASIVE_SUPERTREND_ID}: discrete parameters cannot enter an f64 quad route"
            ));
        }
    };
    i32::try_from(atr_length).map_err(|_| {
        format!("{EVASIVE_SUPERTREND_ID}.atr_length: value {atr_length} exceeds the CUDA i32 ABI")
    })?;
    Ok((
        atr_length,
        3.0_f64.to_bits(),
        1.0_f64.to_bits(),
        0.5_f64.to_bits(),
    ))
}

/// Resolve Fibonacci Trailing Stop's exact currently admitted default tuple.
/// Its left/right bars are real formula parameters, but neither belongs to the
/// current versioned Classic sweep vocabulary. Synthetic generic periods are
/// rejected rather than silently producing duplicate or renamed columns.
fn resolve_fibonacci_trailing_stop_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, u64, i32), String> {
    let info = get_indicator(FIBONACCI_TRAILING_STOP_ID)
        .ok_or_else(|| format!("{FIBONACCI_TRAILING_STOP_ID}: absent from the registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != FIBONACCI_TRAILING_STOP_OUTPUT_IDS {
        return Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            FIBONACCI_TRAILING_STOP_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(FIBONACCI_TRAILING_STOP_ID);
    let expected_planned_outputs = FIBONACCI_TRAILING_STOP_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != FIBONACCI_TRAILING_STOP_PARAMETER_KEYS {
        return Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}: exact full ABI requires registry parameters {:?}, found {declared_keys:?}",
            FIBONACCI_TRAILING_STOP_PARAMETER_KEYS
        ));
    }

    for (parameter, key, default) in [
        (&info.params[0], "left_bars", 20_i64),
        (&info.params[1], "right_bars", 1_i64),
    ] {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{FIBONACCI_TRAILING_STOP_ID}.{key}: expected optional integer default {default} bounded 1..=unbounded with step 1 and no enum values, found key={} kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let level = &info.params[2];
    if level.key != "level"
        || !matches!(&level.kind, IndicatorParamKind::Float)
        || level.required
        || level.default != Some(ParamValueStatic::Float(-0.382))
        || level.min.is_some()
        || level.max.is_some()
        || level.step.map(f64::to_bits) != Some(0.001_f64.to_bits())
        || !level.enum_values.is_empty()
        || level.notes != Some("Finite Fibonacci extension/retracement factor")
    {
        return Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}.level: expected optional finite float default -0.382, no bounds, step 0.001 and the finite-domain note; found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?} notes={:?}",
            level.kind,
            level.required,
            level.default,
            level.min,
            level.max,
            level.step,
            level.enum_values,
            level.notes
        ));
    }

    let trigger = &info.params[3];
    if trigger.key != "trigger"
        || !matches!(&trigger.kind, IndicatorParamKind::EnumString)
        || trigger.required
        || trigger.default != Some(ParamValueStatic::EnumString("close"))
        || trigger.min.is_some()
        || trigger.max.is_some()
        || trigger.step.is_some()
        || trigger.enum_values != ["close", "wick"]
    {
        return Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}.trigger: expected optional close enum over [close, wick], found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            trigger.kind,
            trigger.required,
            trigger.default,
            trigger.min,
            trigger.max,
            trigger.step,
            trigger.enum_values
        ));
    }

    match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(1),
            require_period_invariant_kernel: true,
        } => Ok((20, 1, (-0.382_f64).to_bits(), 0)),
        ClassicCudaParameters::Defaults { .. } => Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}: default route requires inert anchor 1 and the preserved period-invariant primary classification, found {parameters:?}"
        )),
        ClassicCudaParameters::Swept { .. } => Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}: formula has no canonical period sweep"
        )),
        ClassicCudaParameters::DiscreteDefaults => Err(format!(
            "{FIBONACCI_TRAILING_STOP_ID}: discrete parameters cannot enter an f64 four-output route"
        )),
    }
}

/// Resolve Fisher's sole canonical period for its fisher/signal pair. Registry,
/// admitted schema, planning anchor, and explicit sweep override must agree
/// before either output can enter the shared CUDA session.
fn resolve_fisher_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(FISHER_ID)
        .ok_or_else(|| format!("{FISHER_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != FISHER_PARAMETER_KEYS {
        return Err(format!(
            "{FISHER_ID}: CUDA pair ABI requires exact registry parameters {:?}, found {declared_keys:?}",
            FISHER_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != FISHER_OUTPUT_IDS {
        return Err(format!(
            "{FISHER_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            FISHER_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(FISHER_ID);
    let expected_planned_outputs = FISHER_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{FISHER_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let declared = &info.params[0];
    if declared.key != "period"
        || !matches!(&declared.kind, IndicatorParamKind::Int)
        || declared.required
        || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || declared.max.is_some()
        || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !declared.enum_values.is_empty()
    {
        return Err(format!(
            "{FISHER_ID}.period: expected optional integer default 9 bounded 1..=unbounded with step 1 and no enum values, found key={} kind={:?} required={} bounds={:?}..={:?} step={:?} enum={:?}",
            declared.key,
            declared.kind,
            declared.required,
            declared.min,
            declared.max,
            declared.step,
            declared.enum_values
        ));
    }
    let default_period = match declared.default {
        Some(ParamValueStatic::Int(9)) => 9usize,
        other => {
            return Err(format!(
                "{FISHER_ID}.period: expected exact registry default 9, found {other:?}"
            ));
        }
    };

    let period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(9),
            require_period_invariant_kernel: false,
        } => default_period,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{FISHER_ID}: default route requires period anchor 9 and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{FISHER_ID}: swept period {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != FISHER_PARAMETER_KEYS {
                return Err(format!(
                    "{FISHER_ID}: canonical sweep must override exactly {:?}, found {override_keys:?}",
                    FISHER_PARAMETER_KEYS
                ));
            }
            let resolved = positive_usize_parameter(FISHER_ID, "period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{FISHER_ID}: swept period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{FISHER_ID}: swept tuple requires a resolved period anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{FISHER_ID}: discrete parameters cannot enter an f64 pair route"
            ));
        }
    };
    i32::try_from(period)
        .map_err(|_| format!("{FISHER_ID}.period: value {period} exceeds the CUDA i32 ABI"))?;
    Ok(period)
}

/// Resolve the canonical FBEO (length,smooth) tuple. The frozen hpc authority
/// sweeps only length; smooth remains the exact registry default 10. Every
/// output and parameter must agree before any resident scratch is allocated.
fn resolve_forward_backward_exponential_oscillator_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize), String> {
    let info = get_indicator(FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID).ok_or_else(|| {
        format!("{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: absent from the vector-ta registry")
    })?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_PARAMETER_KEYS {
        return Err(format!(
            "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: CUDA triple ABI requires exact registry parameters {:?}, found {declared_keys:?}",
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS {
        return Err(format!(
            "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID);
    let expected_planned_outputs = FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
        .map(Some)
        .to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    for parameter in &info.params {
        if !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}.{}: expected an optional positive integer with step 1 and no upper/enum bound, found kind={:?} required={} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }
    let default_length = match info.params[0].default {
        Some(ParamValueStatic::Int(20)) => 20usize,
        other => {
            return Err(format!(
                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}.length: expected exact registry default 20, found {other:?}"
            ));
        }
    };
    let default_smooth = match info.params[1].default {
        Some(ParamValueStatic::Int(10)) => 10usize,
        other => {
            return Err(format!(
                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}.smooth: expected exact registry default 10, found {other:?}"
            ));
        }
    };

    let length = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(20),
            require_period_invariant_kernel: false,
        } => default_length,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: default route requires length anchor 20 and a length-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: swept length {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["length"] {
                return Err(format!(
                    "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: canonical sweep must override length only, found {override_keys:?}"
                ));
            }
            let resolved = positive_usize_parameter(
                FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID,
                "length",
                overrides[0].1,
            )?;
            if resolved != *period {
                return Err(format!(
                    "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: swept length {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: swept tuple requires a resolved length anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: discrete parameters cannot enter an f64 triple route"
            ));
        }
    };
    i32::try_from(length).map_err(|_| {
        format!(
            "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}.length: value {length} exceeds the CUDA i32 ABI"
        )
    })?;
    i32::try_from(default_smooth).map_err(|_| {
        format!(
            "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}.smooth: value {default_smooth} exceeds the CUDA i32 ABI"
        )
    })?;
    Ok((length, default_smooth))
}

/// Resolve the exact FVG Trailing Stop tuple admitted by the frozen hpc
/// authority. Only smoothing_length is a production search dimension;
/// unmitigated_fvg_lookback and reset_on_cross stay at their canonical
/// defaults until a versioned vocabulary change admits them explicitly.
fn resolve_fvg_trailing_stop_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, bool), String> {
    let info = get_indicator(FVG_TRAILING_STOP_ID)
        .ok_or_else(|| format!("{FVG_TRAILING_STOP_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != FVG_TRAILING_STOP_PARAMETER_KEYS {
        return Err(format!(
            "{FVG_TRAILING_STOP_ID}: CUDA four-output ABI requires exact registry parameters {:?}, found {declared_keys:?}",
            FVG_TRAILING_STOP_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != FVG_TRAILING_STOP_OUTPUT_IDS {
        return Err(format!(
            "{FVG_TRAILING_STOP_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            FVG_TRAILING_STOP_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(FVG_TRAILING_STOP_ID);
    let expected_planned_outputs = FVG_TRAILING_STOP_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{FVG_TRAILING_STOP_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    for parameter in &info.params[..2] {
        if !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{FVG_TRAILING_STOP_ID}.{}: expected an optional positive integer with step 1 and no upper/enum bound, found kind={:?} required={} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }
    let default_lookback = match info.params[0].default {
        Some(ParamValueStatic::Int(5)) => 5usize,
        other => {
            return Err(format!(
                "{FVG_TRAILING_STOP_ID}.unmitigated_fvg_lookback: expected exact registry default 5, found {other:?}"
            ));
        }
    };
    let default_smoothing = match info.params[1].default {
        Some(ParamValueStatic::Int(9)) => 9usize,
        other => {
            return Err(format!(
                "{FVG_TRAILING_STOP_ID}.smoothing_length: expected exact registry default 9, found {other:?}"
            ));
        }
    };
    let reset = &info.params[2];
    if !matches!(&reset.kind, IndicatorParamKind::Bool)
        || reset.required
        || reset.default != Some(ParamValueStatic::Bool(false))
        || reset.min.is_some()
        || reset.max.is_some()
        || reset.step.is_some()
        || reset.enum_values != ["true", "false"]
    {
        return Err(format!(
            "{FVG_TRAILING_STOP_ID}.reset_on_cross: expected optional false bool with exact [true, false] vocabulary and no numeric bounds, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            reset.kind,
            reset.required,
            reset.default,
            reset.min,
            reset.max,
            reset.step,
            reset.enum_values
        ));
    }

    let smoothing_length = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(9),
            require_period_invariant_kernel: false,
        } => default_smoothing,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{FVG_TRAILING_STOP_ID}: default route requires smoothing anchor 9 and a smoothing-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{FVG_TRAILING_STOP_ID}: swept smoothing_length {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["smoothing_length"] {
                return Err(format!(
                    "{FVG_TRAILING_STOP_ID}: canonical sweep must override smoothing_length only, found {override_keys:?}"
                ));
            }
            let resolved =
                positive_usize_parameter(FVG_TRAILING_STOP_ID, "smoothing_length", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{FVG_TRAILING_STOP_ID}: swept smoothing_length {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{FVG_TRAILING_STOP_ID}: swept tuple requires a resolved smoothing anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{FVG_TRAILING_STOP_ID}: discrete parameters cannot enter an f64 four-output route"
            ));
        }
    };
    for (key, value) in [
        ("unmitigated_fvg_lookback", default_lookback),
        ("smoothing_length", smoothing_length),
    ] {
        i32::try_from(value).map_err(|_| {
            format!("{FVG_TRAILING_STOP_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
        })?;
    }
    Ok((default_lookback, smoothing_length, false))
}

/// Resolve Gator Oscillator's exact six-integer tuple. The hpc registry-ratio
/// sweep scales the three length members from jaws_length=13 while every shift
/// remains at its canonical default, so no synthetic generic period is used.
fn resolve_gatorosc_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, usize, usize, usize, usize, usize), String> {
    let info = get_indicator(GATOROSC_ID)
        .ok_or_else(|| format!("{GATOROSC_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != GATOROSC_PARAMETER_KEYS {
        return Err(format!(
            "{GATOROSC_ID}: CUDA four-output ABI requires exact registry parameters {:?}, found {declared_keys:?}",
            GATOROSC_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != GATOROSC_OUTPUT_IDS {
        return Err(format!(
            "{GATOROSC_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            GATOROSC_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(GATOROSC_ID);
    let expected_planned_outputs = GATOROSC_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{GATOROSC_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let exact_defaults = [13_i64, 8, 8, 5, 5, 3];
    let exact_minimums = [1.0_f64, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut defaults = [0usize; GATOROSC_PARAMETER_KEYS.len()];
    for (index, key) in GATOROSC_PARAMETER_KEYS.iter().copied().enumerate() {
        let declared = &info.params[index];
        if !matches!(&declared.kind, IndicatorParamKind::Int)
            || declared.required
            || declared.default != Some(ParamValueStatic::Int(exact_defaults[index]))
            || declared.min.map(f64::to_bits) != Some(exact_minimums[index].to_bits())
            || declared.max.is_some()
            || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !declared.enum_values.is_empty()
        {
            return Err(format!(
                "{GATOROSC_ID}.{key}: canonical optional Int contract drifted at slot {index}: default={:?} bounds={:?}..={:?} step={:?} enum={:?} required={}",
                declared.default,
                declared.min,
                declared.max,
                declared.step,
                declared.enum_values,
                declared.required
            ));
        }
        defaults[index] = usize::try_from(exact_defaults[index])
            .map_err(|_| format!("{GATOROSC_ID}.{key}: default exceeds usize"))?;
    }

    let values = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(13),
            require_period_invariant_kernel: false,
        } => defaults,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{GATOROSC_ID}: default route requires resolved jaws anchor 13 and a length-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{GATOROSC_ID}: swept ratio anchor {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != GATOROSC_SWEEP_PARAMETER_KEYS {
                return Err(format!(
                    "{GATOROSC_ID}: canonical ratio sweep must override exactly {:?}, found {override_keys:?}",
                    GATOROSC_SWEEP_PARAMETER_KEYS
                ));
            }
            let mut swept = defaults;
            for (key, raw) in overrides {
                let index = GATOROSC_PARAMETER_KEYS
                    .iter()
                    .position(|candidate| candidate == key)
                    .expect("exact Gator sweep-key equality proved above");
                let value = positive_usize_parameter(GATOROSC_ID, key, *raw)?;
                i32::try_from(value).map_err(|_| {
                    format!("{GATOROSC_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
                })?;
                swept[index] = value;
            }
            let resolved_anchor = swept[0].max(swept[2]).max(swept[4]);
            if resolved_anchor != *period {
                return Err(format!(
                    "{GATOROSC_ID}: scaled lengths [{}, {}, {}] require anchor {period}, found {resolved_anchor}",
                    swept[0], swept[2], swept[4]
                ));
            }
            swept
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{GATOROSC_ID}: swept tuple requires a resolved ratio anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{GATOROSC_ID}: discrete parameters cannot enter an f64 four-output route"
            ));
        }
    };
    for (key, value) in GATOROSC_PARAMETER_KEYS.iter().zip(values) {
        i32::try_from(value)
            .map_err(|_| format!("{GATOROSC_ID}.{key}: value {value} exceeds the CUDA i32 ABI"))?;
    }
    let [
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    ] = values;
    Ok((
        jaws_length,
        jaws_shift,
        teeth_length,
        teeth_shift,
        lips_length,
        lips_shift,
    ))
}

/// Resolve HalfTrend's exact registered tuple. The current hpc authority keeps
/// amplitude and channel deviation at their canonical defaults and sweeps only
/// `atr_period`; the default 100 point is emitted by the base pass and therefore
/// never appears again under an extended suffix.
fn resolve_halftrend_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<(usize, u64, usize), String> {
    let info = get_indicator(HALFTREND_ID)
        .ok_or_else(|| format!("{HALFTREND_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != HALFTREND_PARAMETER_KEYS {
        return Err(format!(
            "{HALFTREND_ID}: CUDA six-output ABI requires exact registry parameters {:?}, found {declared_keys:?}",
            HALFTREND_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != HALFTREND_OUTPUT_IDS {
        return Err(format!(
            "{HALFTREND_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            HALFTREND_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(HALFTREND_ID);
    let expected_planned_outputs = HALFTREND_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{HALFTREND_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let amplitude = &info.params[0];
    if !matches!(&amplitude.kind, IndicatorParamKind::Int)
        || amplitude.required
        || amplitude.default != Some(ParamValueStatic::Int(2))
        || amplitude.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || amplitude.max.is_some()
        || amplitude.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !amplitude.enum_values.is_empty()
    {
        return Err(format!(
            "{HALFTREND_ID}.amplitude canonical optional Int contract drifted: default={:?} bounds={:?}..={:?} step={:?} enum={:?} required={}",
            amplitude.default,
            amplitude.min,
            amplitude.max,
            amplitude.step,
            amplitude.enum_values,
            amplitude.required
        ));
    }
    let channel_deviation = &info.params[1];
    if !matches!(&channel_deviation.kind, IndicatorParamKind::Float)
        || channel_deviation.required
        || channel_deviation.default != Some(ParamValueStatic::Float(2.0))
        || channel_deviation.min.map(f64::to_bits) != Some(0.0_f64.to_bits())
        || channel_deviation.max.is_some()
        || channel_deviation.step.is_some()
        || !channel_deviation.enum_values.is_empty()
    {
        return Err(format!(
            "{HALFTREND_ID}.channel_deviation canonical optional Float contract drifted: default={:?} bounds={:?}..={:?} step={:?} enum={:?} required={}",
            channel_deviation.default,
            channel_deviation.min,
            channel_deviation.max,
            channel_deviation.step,
            channel_deviation.enum_values,
            channel_deviation.required
        ));
    }
    let atr = &info.params[2];
    if !matches!(&atr.kind, IndicatorParamKind::Int)
        || atr.required
        || atr.default != Some(ParamValueStatic::Int(100))
        || atr.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || atr.max.is_some()
        || atr.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !atr.enum_values.is_empty()
    {
        return Err(format!(
            "{HALFTREND_ID}.atr_period canonical optional Int contract drifted: default={:?} bounds={:?}..={:?} step={:?} enum={:?} required={}",
            atr.default, atr.min, atr.max, atr.step, atr.enum_values, atr.required
        ));
    }

    let atr_period = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(100),
            require_period_invariant_kernel: false,
        } => 100,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{HALFTREND_ID}: default route requires resolved ATR anchor 100 and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{HALFTREND_ID}: swept ATR anchor {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            if overrides.len() != 1 || overrides[0].0 != "atr_period" {
                return Err(format!(
                    "{HALFTREND_ID}: canonical sweep must override atr_period only, found {overrides:?}"
                ));
            }
            let resolved = positive_usize_parameter(HALFTREND_ID, "atr_period", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{HALFTREND_ID}: swept atr_period {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{HALFTREND_ID}: swept tuple requires a resolved ATR anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{HALFTREND_ID}: discrete parameters cannot enter an f64 six-output route"
            ));
        }
    };
    for (key, value) in [("amplitude", 2usize), ("atr_period", atr_period)] {
        i32::try_from(value)
            .map_err(|_| format!("{HALFTREND_ID}.{key}: value {value} exceeds the CUDA i32 ABI"))?;
    }
    Ok((2, 2.0_f64.to_bits(), atr_period))
}

/// Resolve the one canonical Fibonacci Entry Bands search dimension. The
/// full CUDA ABI retains all eighteen registered outputs, while the production
/// ledger admits sixteen because the two default-low TP bands are exact
/// duplicates of lower_2618/upper_2618. Every other parameter is pinned to the
/// registry default before the first allocation or launch.
fn resolve_fibonacci_entry_bands_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(FIBONACCI_ENTRY_BANDS_ID)
        .ok_or_else(|| format!("{FIBONACCI_ENTRY_BANDS_ID}: absent from the registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS {
        return Err(format!(
            "{FIBONACCI_ENTRY_BANDS_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(FIBONACCI_ENTRY_BANDS_ID);
    let expected_planned_outputs = FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS
        .map(Some)
        .to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{FIBONACCI_ENTRY_BANDS_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != FIBONACCI_ENTRY_BANDS_PARAMETER_KEYS {
        return Err(format!(
            "{FIBONACCI_ENTRY_BANDS_ID}: exact full ABI requires registry parameters {:?}, found {declared_keys:?}",
            FIBONACCI_ENTRY_BANDS_PARAMETER_KEYS
        ));
    }

    let source = &info.params[0];
    let source_values = [
        "open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4",
    ];
    if !matches!(&source.kind, IndicatorParamKind::EnumString)
        || source.required
        || source.default != Some(ParamValueStatic::EnumString("hlc3"))
        || source.min.is_some()
        || source.max.is_some()
        || source.step.is_some()
        || source.enum_values != source_values
    {
        return Err(format!(
            "{FIBONACCI_ENTRY_BANDS_ID}.source: expected optional HLC3 enum over {source_values:?}, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            source.kind,
            source.required,
            source.default,
            source.min,
            source.max,
            source.step,
            source.enum_values
        ));
    }

    for (parameter, key, default) in [
        (&info.params[1], "length", 21_i64),
        (&info.params[2], "atr_length", 14_i64),
    ] {
        if parameter.key != key
            || !matches!(&parameter.kind, IndicatorParamKind::Int)
            || parameter.required
            || parameter.default != Some(ParamValueStatic::Int(default))
            || parameter.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || parameter.max.is_some()
            || parameter.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
            || !parameter.enum_values.is_empty()
        {
            return Err(format!(
                "{FIBONACCI_ENTRY_BANDS_ID}.{key}: expected optional integer default {default} bounded 1..=unbounded with step 1 and no enum values, found key={} kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
                parameter.key,
                parameter.kind,
                parameter.required,
                parameter.default,
                parameter.min,
                parameter.max,
                parameter.step,
                parameter.enum_values
            ));
        }
    }

    let use_atr = &info.params[3];
    if !matches!(&use_atr.kind, IndicatorParamKind::Bool)
        || use_atr.required
        || use_atr.default != Some(ParamValueStatic::Bool(true))
        || use_atr.min.is_some()
        || use_atr.max.is_some()
        || use_atr.step.is_some()
        || use_atr.enum_values != ["true", "false"]
    {
        return Err(format!(
            "{FIBONACCI_ENTRY_BANDS_ID}.use_atr: expected optional true bool, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            use_atr.kind,
            use_atr.required,
            use_atr.default,
            use_atr.min,
            use_atr.max,
            use_atr.step,
            use_atr.enum_values
        ));
    }

    let tp = &info.params[4];
    if !matches!(&tp.kind, IndicatorParamKind::EnumString)
        || tp.required
        || tp.default != Some(ParamValueStatic::EnumString("low"))
        || tp.min.is_some()
        || tp.max.is_some()
        || tp.step.is_some()
        || tp.enum_values != ["low", "medium", "high"]
    {
        return Err(format!(
            "{FIBONACCI_ENTRY_BANDS_ID}.tp_aggressiveness: expected optional low enum over [low, medium, high], found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            tp.kind, tp.required, tp.default, tp.min, tp.max, tp.step, tp.enum_values
        ));
    }

    let length = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(21),
            require_period_invariant_kernel: false,
        } => 21usize,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{FIBONACCI_ENTRY_BANDS_ID}: default route requires the canonical length-21 anchor and a period-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{FIBONACCI_ENTRY_BANDS_ID}: swept period {period} requires the same resolved length anchor, found {anchor}"
                ));
            }
            let expected_overrides = classic_cuda_sweep_params(FIBONACCI_ENTRY_BANDS_ID, *period)?;
            if overrides.as_slice() != expected_overrides.as_slice() {
                return Err(format!(
                    "{FIBONACCI_ENTRY_BANDS_ID}: period {period} requires exact length overrides {expected_overrides:?}, found {overrides:?}"
                ));
            }
            let [("length", resolved)] = overrides.as_slice() else {
                return Err(format!(
                    "{FIBONACCI_ENTRY_BANDS_ID}: expected one exact length override, found {overrides:?}"
                ));
            };
            let resolved = positive_usize_parameter(FIBONACCI_ENTRY_BANDS_ID, "length", *resolved)?;
            if resolved != *period {
                return Err(format!(
                    "{FIBONACCI_ENTRY_BANDS_ID}: swept period {period} != exact length override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{FIBONACCI_ENTRY_BANDS_ID}: swept tuple requires a resolved length anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{FIBONACCI_ENTRY_BANDS_ID}: discrete parameters cannot enter an f64 full-output route"
            ));
        }
    };
    i32::try_from(length).map_err(|_| {
        format!("{FIBONACCI_ENTRY_BANDS_ID}.length: value {length} exceeds the CUDA i32 ABI")
    })?;
    Ok(length)
}

/// Resolve Ehlers Data Sampling RSI's sole canonical length. The full CUDA ABI
/// emits all three registered outputs while the production ledger deliberately
/// admits only ds_rsi/signal because original_rsi duplicates standalone RSI.
fn resolve_ehlers_data_sampling_rsi_length(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<usize, String> {
    let info = get_indicator(EHLERS_DATA_SAMPLING_RSI_ID)
        .ok_or_else(|| format!("{EHLERS_DATA_SAMPLING_RSI_ID}: absent from the registry"))?;
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS {
        return Err(format!(
            "{EHLERS_DATA_SAMPLING_RSI_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(EHLERS_DATA_SAMPLING_RSI_ID);
    let expected_planned_outputs = EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS
        .map(Some)
        .to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{EHLERS_DATA_SAMPLING_RSI_ID}: admitted output schema {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }
    let declared_keys = info
        .params
        .iter()
        .map(|parameter| parameter.key)
        .collect::<Vec<_>>();
    if declared_keys != EHLERS_DATA_SAMPLING_RSI_PARAMETER_KEYS {
        return Err(format!(
            "{EHLERS_DATA_SAMPLING_RSI_ID}: exact full-output ABI requires registry parameters {:?}, found {declared_keys:?}",
            EHLERS_DATA_SAMPLING_RSI_PARAMETER_KEYS
        ));
    }
    let declared = &info.params[0];
    if declared.key != "length"
        || !matches!(&declared.kind, IndicatorParamKind::Int)
        || declared.required
        || declared.default != Some(ParamValueStatic::Int(14))
        || declared.min.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || declared.max.is_some()
        || declared.step.map(f64::to_bits) != Some(1.0_f64.to_bits())
        || !declared.enum_values.is_empty()
    {
        return Err(format!(
            "{EHLERS_DATA_SAMPLING_RSI_ID}.length: expected optional integer default 14 bounded 1..=unbounded with step 1 and no enum values, found kind={:?} required={} default={:?} bounds={:?}..={:?} step={:?} enum={:?}",
            declared.kind,
            declared.required,
            declared.default,
            declared.min,
            declared.max,
            declared.step,
            declared.enum_values
        ));
    }

    let length = match parameters {
        ClassicCudaParameters::Defaults {
            anchor: ClassicCudaAnchor::Resolved(14),
            require_period_invariant_kernel: false,
        } => 14,
        ClassicCudaParameters::Defaults { .. } => {
            return Err(format!(
                "{EHLERS_DATA_SAMPLING_RSI_ID}: default route requires length anchor 14 and a length-consuming kernel, found {parameters:?}"
            ));
        }
        ClassicCudaParameters::Swept {
            period,
            overrides,
            anchor: ClassicCudaAnchor::Resolved(anchor),
        } => {
            if anchor != period {
                return Err(format!(
                    "{EHLERS_DATA_SAMPLING_RSI_ID}: swept length {period} requires the same resolved anchor, found {anchor}"
                ));
            }
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != EHLERS_DATA_SAMPLING_RSI_PARAMETER_KEYS {
                return Err(format!(
                    "{EHLERS_DATA_SAMPLING_RSI_ID}: canonical sweep must override exactly {:?}, found {override_keys:?}",
                    EHLERS_DATA_SAMPLING_RSI_PARAMETER_KEYS
                ));
            }
            let resolved =
                positive_usize_parameter(EHLERS_DATA_SAMPLING_RSI_ID, "length", overrides[0].1)?;
            if resolved != *period {
                return Err(format!(
                    "{EHLERS_DATA_SAMPLING_RSI_ID}: swept length {period} != exact override {resolved}"
                ));
            }
            resolved
        }
        ClassicCudaParameters::Swept { anchor, .. } => {
            return Err(format!(
                "{EHLERS_DATA_SAMPLING_RSI_ID}: swept tuple requires a resolved length anchor, found {anchor:?}"
            ));
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{EHLERS_DATA_SAMPLING_RSI_ID}: discrete parameters cannot enter an f64 full-output route"
            ));
        }
    };
    i32::try_from(length).map_err(|_| {
        format!("{EHLERS_DATA_SAMPLING_RSI_ID}.length: value {length} exceeds the CUDA i32 ABI")
    })?;
    Ok(length)
}

/// Resolve the exact seven-parameter Bulls v Bears production point. The
/// reviewed `ma`, `upper`, and `lower` exclusions depend on the canonical EMA
/// plus Normalized defaults, so any registry mode drift is a fail-closed
/// parameter-contract gap rather than permission to change the feature schema.
fn resolve_bulls_v_bears_parameters(
    parameters: &ClassicCudaParameters,
) -> std::result::Result<
    (
        usize,
        BullsVBearsMaType,
        BullsVBearsCalculationMethod,
        usize,
        usize,
        u64,
        u64,
    ),
    String,
> {
    let info = get_indicator(BULLS_V_BEARS_ID)
        .ok_or_else(|| format!("{BULLS_V_BEARS_ID}: absent from the vector-ta registry"))?;
    let declared_keys = info
        .params
        .iter()
        .map(|param| param.key)
        .collect::<Vec<_>>();
    if declared_keys != BULLS_V_BEARS_PARAMETER_KEYS {
        return Err(format!(
            "{BULLS_V_BEARS_ID}: CUDA all-output ABI requires exact registry parameters {:?}, \
             found {declared_keys:?}",
            BULLS_V_BEARS_PARAMETER_KEYS
        ));
    }
    let declared_outputs = info
        .outputs
        .iter()
        .map(|output| output.id)
        .collect::<Vec<_>>();
    if declared_outputs != BULLS_V_BEARS_FULL_OUTPUT_IDS {
        return Err(format!(
            "{BULLS_V_BEARS_ID}: full CUDA output ABI {:?} != registry {declared_outputs:?}",
            BULLS_V_BEARS_FULL_OUTPUT_IDS
        ));
    }
    let planned_outputs = output_ids_for(BULLS_V_BEARS_ID);
    let expected_planned_outputs = BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS.map(Some).to_vec();
    if planned_outputs != expected_planned_outputs {
        return Err(format!(
            "{BULLS_V_BEARS_ID}: reviewed admitted output schema \
             {expected_planned_outputs:?} != {planned_outputs:?}"
        ));
    }

    let resolve_positive_int_default =
        |index: usize, key: &'static str| -> std::result::Result<usize, String> {
            let declared = &info.params[index];
            if declared.key != key
                || !matches!(&declared.kind, IndicatorParamKind::Int)
                || declared.required
            {
                return Err(format!(
                    "{BULLS_V_BEARS_ID}.{key}: expected an optional integer at registry slot \
                     {index}, found key={} kind={:?} required={}",
                    declared.key, declared.kind, declared.required
                ));
            }
            let raw = match declared.default {
                Some(ParamValueStatic::Int(value)) => value,
                other => {
                    return Err(format!(
                        "{BULLS_V_BEARS_ID}.{key}: expected an integer registry default, found \
                         {other:?}"
                    ));
                }
            };
            let value = positive_usize_parameter(BULLS_V_BEARS_ID, key, raw)?;
            if declared.min.is_some_and(|minimum| (value as f64) < minimum)
                || declared.max.is_some_and(|maximum| (value as f64) > maximum)
            {
                return Err(format!(
                    "{BULLS_V_BEARS_ID}.{key}: registry default {value} is outside declared \
                     bounds {:?}..={:?}",
                    declared.min, declared.max
                ));
            }
            i32::try_from(value).map_err(|_| {
                format!("{BULLS_V_BEARS_ID}.{key}: value {value} exceeds the CUDA i32 ABI")
            })?;
            Ok(value)
        };
    let default_period = resolve_positive_int_default(0, "period")?;
    let normalized_bars_back = resolve_positive_int_default(3, "normalized_bars_back")?;
    let raw_rolling_period = resolve_positive_int_default(4, "raw_rolling_period")?;

    let ma_type_param = &info.params[1];
    if !matches!(&ma_type_param.kind, IndicatorParamKind::EnumString)
        || ma_type_param.required
        || ma_type_param.enum_values != ["ema", "sma", "wma"]
    {
        return Err(format!(
            "{BULLS_V_BEARS_ID}.ma_type: expected optional enum [ema, sma, wma], found \
             kind={:?} required={} values={:?}",
            ma_type_param.kind, ma_type_param.required, ma_type_param.enum_values
        ));
    }
    let ma_type = match ma_type_param.default {
        Some(ParamValueStatic::EnumString("ema")) => BullsVBearsMaType::Ema,
        other => {
            return Err(format!(
                "{BULLS_V_BEARS_ID}.ma_type: reviewed `ma` exclusion requires canonical EMA \
                 default, found {other:?}"
            ));
        }
    };

    let calculation_method_param = &info.params[2];
    if !matches!(
        &calculation_method_param.kind,
        IndicatorParamKind::EnumString
    ) || calculation_method_param.required
        || calculation_method_param.enum_values != ["normalized", "raw"]
    {
        return Err(format!(
            "{BULLS_V_BEARS_ID}.calculation_method: expected optional enum [normalized, raw], \
             found kind={:?} required={} values={:?}",
            calculation_method_param.kind,
            calculation_method_param.required,
            calculation_method_param.enum_values
        ));
    }
    let calculation_method = match calculation_method_param.default {
        Some(ParamValueStatic::EnumString("normalized")) => {
            BullsVBearsCalculationMethod::Normalized
        }
        other => {
            return Err(format!(
                "{BULLS_V_BEARS_ID}.calculation_method: reviewed upper/lower exclusions require \
                 canonical Normalized default, found {other:?}"
            ));
        }
    };

    let resolve_float_default = |index: usize,
                                 key: &'static str,
                                 formula_minimum: f64,
                                 formula_maximum: f64|
     -> std::result::Result<f64, String> {
        let declared = &info.params[index];
        if declared.key != key
            || !matches!(&declared.kind, IndicatorParamKind::Float)
            || declared.required
        {
            return Err(format!(
                "{BULLS_V_BEARS_ID}.{key}: expected an optional float at registry slot {index}, \
                 found key={} kind={:?} required={}",
                declared.key, declared.kind, declared.required
            ));
        }
        let value = match declared.default {
            Some(ParamValueStatic::Float(value)) if value.is_finite() => value,
            other => {
                return Err(format!(
                    "{BULLS_V_BEARS_ID}.{key}: expected a finite float registry default, found \
                     {other:?}"
                ));
            }
        };
        if !(formula_minimum..=formula_maximum).contains(&value)
            || declared.min.is_some_and(|minimum| value < minimum)
            || declared.max.is_some_and(|maximum| value > maximum)
        {
            return Err(format!(
                "{BULLS_V_BEARS_ID}.{key}: registry default {value} violates formula bounds \
                 {formula_minimum}..={formula_maximum} or declared bounds {:?}..={:?}",
                declared.min, declared.max
            ));
        }
        Ok(value)
    };
    let raw_threshold_percentile =
        resolve_float_default(5, "raw_threshold_percentile", 80.0, 99.0)?;
    let threshold_level = resolve_float_default(6, "threshold_level", 0.0, 100.0)?;

    let period = match parameters {
        ClassicCudaParameters::Defaults { .. } => default_period,
        ClassicCudaParameters::Swept { overrides, .. } => {
            let override_keys = overrides.iter().map(|(key, _)| *key).collect::<Vec<_>>();
            if override_keys != ["period"] {
                return Err(format!(
                    "{BULLS_V_BEARS_ID}: canonical sweep must override exactly [\"period\"], \
                     found {override_keys:?}"
                ));
            }
            let swept = positive_usize_parameter(BULLS_V_BEARS_ID, "period", overrides[0].1)?;
            let period_param = &info.params[0];
            if period_param
                .min
                .is_some_and(|minimum| (swept as f64) < minimum)
                || period_param
                    .max
                    .is_some_and(|maximum| (swept as f64) > maximum)
            {
                return Err(format!(
                    "{BULLS_V_BEARS_ID}.period: swept value {swept} is outside declared bounds \
                     {:?}..={:?}",
                    period_param.min, period_param.max
                ));
            }
            i32::try_from(swept).map_err(|_| {
                format!("{BULLS_V_BEARS_ID}.period: value {swept} exceeds the CUDA i32 ABI")
            })?;
            swept
        }
        ClassicCudaParameters::DiscreteDefaults => {
            return Err(format!(
                "{BULLS_V_BEARS_ID}: discrete parameters cannot enter an f64 all-output route"
            ));
        }
    };
    match parameters.anchor() {
        ClassicCudaAnchor::Resolved(anchor) if *anchor == period => {}
        other => {
            return Err(format!(
                "{BULLS_V_BEARS_ID}: resolved period {period} requires the same anchor, found \
                 {other:?}"
            ));
        }
    }
    Ok((
        period,
        ma_type,
        calculation_method,
        normalized_bars_back,
        raw_rolling_period,
        raw_threshold_percentile.to_bits(),
        threshold_level.to_bits(),
    ))
}

fn parameter_request(
    indicator_id: &'static str,
    swept_period: Option<usize>,
) -> ClassicCudaParameters {
    let mut anchor = match classic_cuda_period_anchor(indicator_id, swept_period) {
        Ok(period) => ClassicCudaAnchor::Resolved(period),
        Err(reason) => ClassicCudaAnchor::Missing(reason),
    };
    match swept_period {
        Some(period) => {
            let overrides = match classic_cuda_sweep_params(indicator_id, period) {
                Ok(overrides) => overrides,
                Err(reason) => {
                    anchor = ClassicCudaAnchor::Missing(reason);
                    Vec::new()
                }
            };
            ClassicCudaParameters::Swept {
                period,
                overrides,
                anchor,
            }
        }
        None => ClassicCudaParameters::Defaults {
            anchor,
            require_period_invariant_kernel: classic_cuda_base_has_no_window(indicator_id),
        },
    }
}

fn append_indicator_nodes(
    nodes: &mut Vec<ClassicCudaNode>,
    stage: ClassicCudaStage,
    indicator_id: &'static str,
    swept_period: Option<usize>,
) -> Result<()> {
    let prefix = swept_period
        .map(|period| format!("{indicator_id}_{period}"))
        .unwrap_or_else(|| indicator_id.to_string());
    let first = nodes.len();

    if indicator_id == "pattern_recognition" {
        ensure!(
            swept_period.is_none(),
            "pattern_recognition cannot enter a period sweep"
        );
        for pattern in vector_ta::indicators::pattern_recognition::list_patterns() {
            nodes.push(ClassicCudaNode {
                stage,
                indicator_id,
                requested_output_id: Some(pattern.id),
                column_name: format!("{prefix}_{}", pattern.id),
                value_kind: ClassicCudaValueKind::PatternI32,
                parameters: ClassicCudaParameters::DiscreteDefaults,
            });
        }
    } else {
        let parameters = parameter_request(indicator_id, swept_period);
        for requested_output_id in output_ids_for(indicator_id) {
            let column_name = requested_output_id
                .map(|output_id| format!("{prefix}_{output_id}"))
                .unwrap_or_else(|| prefix.clone());
            nodes.push(ClassicCudaNode {
                stage,
                indicator_id,
                requested_output_id,
                column_name,
                value_kind: ClassicCudaValueKind::F64,
                parameters: parameters.clone(),
            });
        }
    }

    let actual = nodes.len() - first;
    ensure!(
        actual == planned_output_count(indicator_id),
        "{indicator_id}: exact CUDA plan expanded {actual} output(s), but canonical budget \
         accounting planned {}",
        planned_output_count(indicator_id)
    );
    Ok(())
}

/// Build the exact ordered graph from the already-finalized admission facts.
/// No registry-wide capability scan and no second RAM probe is allowed here.
pub(crate) fn build_exact_classic_cuda_plan(
    rows: usize,
    admitted_indicator_ids: &[&'static str],
    historical: &[&'static str],
    extended_groups: &[(&'static str, Vec<usize>)],
) -> Result<ClassicCudaPlan> {
    let planned_capacity = admitted_indicator_ids
        .iter()
        .map(|id| planned_output_count(id))
        .sum::<usize>()
        + historical
            .iter()
            .map(|id| planned_output_count(id) * super::hpc_ta::ALT_PERIODS.len())
            .sum::<usize>()
        + extended_groups
            .iter()
            .map(|(id, periods)| planned_output_count(id) * periods.len())
            .sum::<usize>();
    let mut nodes = Vec::with_capacity(planned_capacity);

    for &indicator_id in admitted_indicator_ids {
        append_indicator_nodes(&mut nodes, ClassicCudaStage::Base, indicator_id, None)?;
    }
    for &indicator_id in historical {
        for &period in &super::hpc_ta::ALT_PERIODS {
            append_indicator_nodes(
                &mut nodes,
                ClassicCudaStage::Historical,
                indicator_id,
                Some(period),
            )?;
        }
    }
    for (indicator_id, periods) in extended_groups {
        for &period in periods {
            append_indicator_nodes(
                &mut nodes,
                ClassicCudaStage::Extended,
                *indicator_id,
                Some(period),
            )?;
        }
    }

    ensure!(
        nodes.len() == planned_capacity,
        "exact CUDA plan width {} != canonical planned width {planned_capacity}",
        nodes.len()
    );
    Ok(ClassicCudaPlan { rows, nodes })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClassicCudaGapReason {
    CanonicalCpuOutputUnavailable,
    MissingF64Kernel,
    MissingMovingAverageDispatcherRoute,
    MissingNamedOutputRoute,
    MissingNamedProductionDispatcher,
    NamedFamilyContractMismatch,
    MissingDiscreteMatrixRoute,
    MissingParameterContract,
    NoWindowKernelConsumesAnchor,
}

impl std::fmt::Display for ClassicCudaGapReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CanonicalCpuOutputUnavailable => "canonical_cpu_output_unavailable",
            Self::MissingF64Kernel => "missing_f64_kernel",
            Self::MissingMovingAverageDispatcherRoute => "missing_moving_average_dispatcher_route",
            Self::MissingNamedOutputRoute => "missing_named_output_route",
            Self::MissingNamedProductionDispatcher => "missing_named_production_dispatcher",
            Self::NamedFamilyContractMismatch => "named_family_contract_mismatch",
            Self::MissingDiscreteMatrixRoute => "missing_discrete_matrix_route",
            Self::MissingParameterContract => "missing_parameter_contract",
            Self::NoWindowKernelConsumesAnchor => "no_window_kernel_consumes_anchor",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClassicCudaGap {
    pub(crate) column_name: String,
    pub(crate) indicator_id: &'static str,
    pub(crate) requested_output_id: Option<&'static str>,
    pub(crate) parameters: ClassicCudaParameters,
    pub(crate) reason: ClassicCudaGapReason,
    pub(crate) detail: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedClassicCudaNode {
    node: ClassicCudaNode,
    output_id: &'static str,
    route: ClassicCudaResolvedRoute,
}

impl ResolvedClassicCudaNode {
    pub(crate) const fn stage(&self) -> ClassicCudaStage {
        self.node.stage
    }

    pub(crate) fn indicator_id(&self) -> &'static str {
        self.node.indicator_id
    }

    pub(crate) fn output_id(&self) -> &'static str {
        self.output_id
    }

    pub(crate) fn column_name(&self) -> &str {
        &self.node.column_name
    }

    pub(crate) fn swept_period(&self) -> Option<usize> {
        self.node.parameters.swept_period()
    }

    pub(crate) fn primary_cuda_period(&self) -> Option<usize> {
        match self.route {
            ClassicCudaResolvedRoute::Primary { cuda_period } => Some(cuda_period),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ResolvedClassicCudaLaunch {
    Primary(ResolvedClassicCudaNode),
    AbsoluteStrengthIndexOscillator {
        routes: [ResolvedClassicCudaNode; ASI_OUTPUT_IDS.len()],
        ema_length: usize,
        signal_length: usize,
    },
    AdaptiveBandpassTriggerOscillator {
        routes: [ResolvedClassicCudaNode; ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS.len()],
        delta_bits: u64,
        alpha_bits: u64,
    },
    AdaptiveBoundsRsi {
        routes: [ResolvedClassicCudaNode; ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS.len()],
        rsi_length: usize,
        alpha_bits: u64,
    },
    AdaptiveMacd {
        routes: [ResolvedClassicCudaNode; ADAPTIVE_MACD_OUTPUT_IDS.len()],
        length: usize,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    },
    AdaptiveMomentumOscillator {
        routes: [ResolvedClassicCudaNode; ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS.len()],
        length: usize,
        smoothing_length: usize,
    },
    AdaptiveSchaffTrendCycle {
        routes: [ResolvedClassicCudaNode; ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS.len()],
        adaptive_length: usize,
        stc_length: usize,
        smoothing_factor_bits: u64,
        fast_length: usize,
        slow_length: usize,
    },
    AdjustableMaAlternatingExtremities {
        routes: [ResolvedClassicCudaNode; ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS.len()],
        length: usize,
        mult_bits: u64,
        alpha_bits: u64,
        beta_bits: u64,
    },
    Alligator {
        routes: [ResolvedClassicCudaNode; ALLIGATOR_OUTPUT_IDS.len()],
        jaw_period: usize,
        jaw_offset: usize,
        teeth_period: usize,
        teeth_offset: usize,
        lips_period: usize,
        lips_offset: usize,
    },
    AlphaTrend {
        routes: [ResolvedClassicCudaNode; ALPHATREND_OUTPUT_IDS.len()],
        coeff_bits: u64,
        period: usize,
        no_volume: bool,
    },
    Acosc {
        routes: [ResolvedClassicCudaNode; ACOSC_OUTPUT_IDS.len()],
    },
    AndeanOscillator {
        routes: [ResolvedClassicCudaNode; ANDEAN_OSCILLATOR_OUTPUT_IDS.len()],
        length: usize,
        signal_length: usize,
    },
    Aroon {
        routes: [ResolvedClassicCudaNode; AROON_OUTPUT_IDS.len()],
        length: usize,
    },
    Aso {
        routes: [ResolvedClassicCudaNode; ASO_OUTPUT_IDS.len()],
        period: usize,
        mode: usize,
    },
    AutocorrelationIndicator {
        routes: [ResolvedClassicCudaNode; AUTOCORRELATION_INDICATOR_OUTPUT_IDS.len()],
        length: usize,
        lag: usize,
        use_test_signal: bool,
    },
    Avsl {
        routes: [ResolvedClassicCudaNode; AVSL_OUTPUT_IDS.len()],
        fast_period: usize,
        slow_period: usize,
        multiplier_bits: u64,
    },
    Bandpass {
        routes: [ResolvedClassicCudaNode; BANDPASS_OUTPUT_IDS.len()],
        period: usize,
        bandwidth_bits: u64,
    },
    BollingerBands {
        routes: [ResolvedClassicCudaNode; BOLLINGER_BANDS_OUTPUT_IDS.len()],
        period: usize,
        devup_bits: u64,
        devdn_bits: u64,
    },
    BuffAverages {
        routes: [ResolvedClassicCudaNode; BUFF_AVERAGES_OUTPUT_IDS.len()],
        fast_period: usize,
        slow_period: usize,
    },
    CandleStrengthOscillator {
        routes: [ResolvedClassicCudaNode; CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS.len()],
        period: usize,
        atr_enabled: bool,
        atr_length: usize,
        mode: ClassicCandleStrengthMode,
    },
    ChandelierExit {
        routes: [ResolvedClassicCudaNode; CHANDELIER_EXIT_OUTPUT_IDS.len()],
        period: usize,
        mult_bits: u64,
        use_close: bool,
    },
    Cksp {
        routes: [ResolvedClassicCudaNode; CKSP_OUTPUT_IDS.len()],
        p: usize,
        x_bits: u64,
        q: usize,
    },
    Coppock {
        route: ResolvedClassicCudaNode,
        short_roc_period: usize,
        long_roc_period: usize,
        ma_period: usize,
    },
    CorrelationCycle {
        routes: [ResolvedClassicCudaNode; CORRELATION_CYCLE_OUTPUT_IDS.len()],
        period: usize,
        threshold_bits: u64,
    },
    Cvi {
        route: ResolvedClassicCudaNode,
        period: usize,
    },
    CyberpunkValueTrendAnalyzer {
        routes: [ResolvedClassicCudaNode; CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS.len()],
        entry_level: usize,
        exit_level: usize,
    },
    CycleChannelOscillator {
        routes: [ResolvedClassicCudaNode; CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS.len()],
        short_cycle_length: usize,
        medium_cycle_length: usize,
        short_multiplier_bits: u64,
        medium_multiplier_bits: u64,
    },
    DailyFactor {
        routes: [ResolvedClassicCudaNode; DAILY_FACTOR_PRODUCTION_OUTPUT_IDS.len()],
        threshold_level_bits: u64,
    },
    DamianiVolatmeter {
        routes: [ResolvedClassicCudaNode; DAMIANI_VOLATMETER_OUTPUT_IDS.len()],
        vis_atr: usize,
        vis_std: usize,
        sed_atr: usize,
        sed_std: usize,
        threshold_bits: u64,
    },
    Di {
        routes: [ResolvedClassicCudaNode; DI_OUTPUT_IDS.len()],
        period: usize,
    },
    DidiIndex {
        routes: [ResolvedClassicCudaNode; DIDI_INDEX_OUTPUT_IDS.len()],
        short_length: usize,
        medium_length: usize,
        long_length: usize,
    },
    DirectionalImbalanceIndex {
        routes: [ResolvedClassicCudaNode; DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS.len()],
        length: usize,
        period: usize,
    },
    DisparityIndex {
        route: ResolvedClassicCudaNode,
        ema_period: usize,
        lookback_period: usize,
        smoothing_period: usize,
        smoothing_is_sma: bool,
    },
    Dm {
        routes: [ResolvedClassicCudaNode; DM_OUTPUT_IDS.len()],
        period: usize,
    },
    Donchian {
        routes: [ResolvedClassicCudaNode; DONCHIAN_OUTPUT_IDS.len()],
        period: usize,
    },
    DualUlcerIndex {
        routes: [ResolvedClassicCudaNode; DUAL_ULCER_INDEX_OUTPUT_IDS.len()],
        period: usize,
        auto_threshold: bool,
        threshold_bits: u64,
    },
    Dvdiqqe {
        routes: [ResolvedClassicCudaNode; DVDIQQE_OUTPUT_IDS.len()],
        period: usize,
        smoothing_period: usize,
        fast_multiplier_bits: u64,
        slow_multiplier_bits: u64,
        use_tick_only: bool,
        dynamic_center: bool,
        tick_size_bits: u64,
    },
    EhlersAutocorrelationPeriodogram {
        routes: [ResolvedClassicCudaNode; EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS.len()],
        min_period: usize,
        max_period: usize,
        avg_length: usize,
        enhance: bool,
    },
    EhlersLinearExtrapolationPredictor {
        routes: [ResolvedClassicCudaNode; EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS.len()],
        high_pass_length: usize,
        low_pass_length: usize,
        gain_bits: u64,
        bars_forward: usize,
        signal_mode: i32,
    },
    EhlersUndersampledDoubleMovingAverage {
        routes:
            [ResolvedClassicCudaNode; EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS.len()],
        fast_length: usize,
        slow_length: usize,
        sample_length: usize,
    },
    EmaDeviationCorrectedT3 {
        routes: [ResolvedClassicCudaNode; EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.len()],
        period: usize,
        hot_bits: u64,
        t3_mode: usize,
    },
    Emd {
        routes: [ResolvedClassicCudaNode; EMD_OUTPUT_IDS.len()],
        period: usize,
        delta_bits: u64,
        fraction_bits: u64,
    },
    EmdTrend {
        routes: [ResolvedClassicCudaNode; EMD_TREND_OUTPUT_IDS.len()],
        length: usize,
        mult_bits: u64,
    },
    Eri {
        routes: [ResolvedClassicCudaNode; ERI_OUTPUT_IDS.len()],
        period: usize,
    },
    EvasiveSupertrend {
        routes: [ResolvedClassicCudaNode; EVASIVE_SUPERTREND_OUTPUT_IDS.len()],
        atr_length: usize,
        base_multiplier_bits: u64,
        noise_threshold_bits: u64,
        expansion_alpha_bits: u64,
    },
    FibonacciTrailingStop {
        routes: [ResolvedClassicCudaNode; FIBONACCI_TRAILING_STOP_OUTPUT_IDS.len()],
        left_bars: usize,
        right_bars: usize,
        level_bits: u64,
        trigger_mode: i32,
    },
    Fisher {
        routes: [ResolvedClassicCudaNode; FISHER_OUTPUT_IDS.len()],
        period: usize,
    },
    ForwardBackwardExponentialOscillator {
        routes: [ResolvedClassicCudaNode; FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS.len()],
        length: usize,
        smooth: usize,
    },
    FvgTrailingStop {
        routes: [ResolvedClassicCudaNode; FVG_TRAILING_STOP_OUTPUT_IDS.len()],
        unmitigated_fvg_lookback: usize,
        smoothing_length: usize,
        reset_on_cross: bool,
    },
    Gatorosc {
        routes: [ResolvedClassicCudaNode; GATOROSC_OUTPUT_IDS.len()],
        jaws_length: usize,
        jaws_shift: usize,
        teeth_length: usize,
        teeth_shift: usize,
        lips_length: usize,
        lips_shift: usize,
    },
    Halftrend {
        routes: [ResolvedClassicCudaNode; HALFTREND_OUTPUT_IDS.len()],
        amplitude: usize,
        channel_deviation_bits: u64,
        atr_period: usize,
    },
    FibonacciEntryBands {
        routes: [ResolvedClassicCudaNode; FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS.len()],
        length: usize,
    },
    EhlersDataSamplingRsi {
        routes: [ResolvedClassicCudaNode; EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS.len()],
        length: usize,
    },
    BullsVBears {
        routes: [ResolvedClassicCudaNode; BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS.len()],
        period: usize,
        ma_type: BullsVBearsMaType,
        calculation_method: BullsVBearsCalculationMethod,
        normalized_bars_back: usize,
        raw_rolling_period: usize,
        raw_threshold_percentile_bits: u64,
        threshold_level_bits: u64,
    },
}

impl ResolvedClassicCudaLaunch {
    pub(crate) fn output_count(&self) -> usize {
        match self {
            Self::Primary(_) => 1,
            Self::AbsoluteStrengthIndexOscillator { routes, .. } => routes.len(),
            Self::AdaptiveBandpassTriggerOscillator { routes, .. } => routes.len(),
            Self::AdaptiveBoundsRsi { routes, .. } => routes.len(),
            Self::AdaptiveMacd { routes, .. } => routes.len(),
            Self::AdaptiveMomentumOscillator { routes, .. } => routes.len(),
            Self::AdaptiveSchaffTrendCycle { routes, .. } => routes.len(),
            Self::AdjustableMaAlternatingExtremities { routes, .. } => routes.len(),
            Self::Alligator { routes, .. } => routes.len(),
            Self::AlphaTrend { routes, .. } => routes.len(),
            Self::Acosc { routes } => routes.len(),
            Self::AndeanOscillator { routes, .. } => routes.len(),
            Self::Aroon { routes, .. } => routes.len(),
            Self::Aso { routes, .. } => routes.len(),
            Self::AutocorrelationIndicator { routes, .. } => routes.len(),
            Self::Avsl { routes, .. } => routes.len(),
            Self::Bandpass { routes, .. } => routes.len(),
            Self::BollingerBands { routes, .. } => routes.len(),
            Self::BuffAverages { routes, .. } => routes.len(),
            Self::CandleStrengthOscillator { routes, .. } => routes.len(),
            Self::ChandelierExit { routes, .. } => routes.len(),
            Self::Cksp { routes, .. } => routes.len(),
            Self::Coppock { .. } => 1,
            Self::CorrelationCycle { routes, .. } => routes.len(),
            Self::Cvi { .. } => 1,
            Self::CyberpunkValueTrendAnalyzer { routes, .. } => routes.len(),
            Self::CycleChannelOscillator { routes, .. } => routes.len(),
            Self::DailyFactor { routes, .. } => routes.len(),
            Self::DamianiVolatmeter { routes, .. } => routes.len(),
            Self::Di { routes, .. } => routes.len(),
            Self::DidiIndex { routes, .. } => routes.len(),
            Self::DirectionalImbalanceIndex { routes, .. } => routes.len(),
            Self::DisparityIndex { .. } => 1,
            Self::Dm { routes, .. } => routes.len(),
            Self::Donchian { routes, .. } => routes.len(),
            Self::DualUlcerIndex { routes, .. } => routes.len(),
            Self::Dvdiqqe { routes, .. } => routes.len(),
            Self::EhlersAutocorrelationPeriodogram { routes, .. } => routes.len(),
            Self::EhlersLinearExtrapolationPredictor { routes, .. } => routes.len(),
            Self::EhlersUndersampledDoubleMovingAverage { routes, .. } => routes.len(),
            Self::EmaDeviationCorrectedT3 { routes, .. } => routes.len(),
            Self::Emd { routes, .. } => routes.len(),
            Self::EmdTrend { routes, .. } => routes.len(),
            Self::Eri { routes, .. } => routes.len(),
            Self::EvasiveSupertrend { routes, .. } => routes.len(),
            Self::FibonacciTrailingStop { routes, .. } => routes.len(),
            Self::Fisher { routes, .. } => routes.len(),
            Self::ForwardBackwardExponentialOscillator { routes, .. } => routes.len(),
            Self::FvgTrailingStop { routes, .. } => routes.len(),
            Self::Gatorosc { routes, .. } => routes.len(),
            Self::Halftrend { routes, .. } => routes.len(),
            Self::FibonacciEntryBands { routes, .. } => routes.len(),
            Self::EhlersDataSamplingRsi { routes, .. } => routes.len(),
            Self::BullsVBears { routes, .. } => routes.len(),
        }
    }
}

fn push_gap(
    gaps: &mut Vec<ClassicCudaGap>,
    node: &ClassicCudaNode,
    reason: ClassicCudaGapReason,
    detail: impl Into<String>,
) {
    gaps.push(ClassicCudaGap {
        column_name: node.column_name.clone(),
        indicator_id: node.indicator_id,
        requested_output_id: node.requested_output_id,
        parameters: node.parameters.clone(),
        reason,
        detail: detail.into(),
    });
}

/// Resolve every admitted node before the first CUDA context/probe/launch.
/// The error is the complete ordered manifest, not the first failure.
pub(crate) fn preflight_exact_classic_cuda_plan(
    plan: &ClassicCudaPlan,
) -> std::result::Result<Vec<ResolvedClassicCudaLaunch>, Vec<ClassicCudaGap>> {
    let mut resolved = Vec::with_capacity(plan.nodes.len());
    let mut gaps = Vec::new();

    for node in &plan.nodes {
        let before = gaps.len();
        if let Some(reason) = expected_non_producing(node.indicator_id) {
            push_gap(
                &mut gaps,
                node,
                ClassicCudaGapReason::CanonicalCpuOutputUnavailable,
                reason,
            );
        }
        if node.value_kind == ClassicCudaValueKind::PatternI32 {
            push_gap(
                &mut gaps,
                node,
                ClassicCudaGapReason::MissingDiscreteMatrixRoute,
                "pattern_recognition has no typed resident signed-I32 CUDA matrix route",
            );
            continue;
        }

        let Some(route) = f64_primary_device_route_for(node.indicator_id) else {
            let (reason, detail) = if is_ma_dispatcher(node.indicator_id) {
                (
                    ClassicCudaGapReason::MissingMovingAverageDispatcherRoute,
                    "moving-average selector needs its exact default ma_type routed to a concrete \
                     f64 family member",
                )
            } else {
                (
                    ClassicCudaGapReason::MissingF64Kernel,
                    "no registered f64 primary kernel exists",
                )
            };
            push_gap(&mut gaps, node, reason, detail);
            continue;
        };

        let requested_output_id = node.requested_output_id.unwrap_or(route.output_id);
        let cuda_period = match node.parameters.anchor() {
            ClassicCudaAnchor::Resolved(period) => *period,
            ClassicCudaAnchor::Missing(detail) => {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    detail.clone(),
                );
                1
            }
            ClassicCudaAnchor::NotApplicable => {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "f64 output has no CUDA period/default anchor",
                );
                1
            }
        };
        if let ClassicCudaParameters::Swept { overrides, .. } = &node.parameters {
            if overrides.is_empty() {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "canonical CPU sweep produced no explicit integer window overrides",
                );
            }
        }

        let resolved_route = if node.indicator_id == ASI_ID {
            if route.output_id != ASI_OUTPUT_IDS[0]
                || !ASI_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ASI_ID}: canonical all-output contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        ASI_OUTPUT_IDS, ASI_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the ASI named output has no resident f64 route",
                );
            }
            match resolve_absolute_strength_index_oscillator_parameters(&node.parameters) {
                Ok((ema_length, signal_length)) => {
                    Some(ClassicCudaResolvedRoute::AbsoluteStrengthIndexOscillator {
                        ema_length,
                        signal_length,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID {
            if route.output_id != ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS[0]
                || !ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: canonical all-output contract \
                         is {:?} with primary `{}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS,
                        ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Adaptive Bandpass named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NoWindowKernelConsumesAnchor,
                    "Adaptive Bandpass has no period parameter, but its primary route consumes \
                     the planning anchor",
                );
            }
            match resolve_adaptive_bandpass_trigger_oscillator_parameters(&node.parameters) {
                Ok((delta_bits, alpha_bits)) => Some(
                    ClassicCudaResolvedRoute::AdaptiveBandpassTriggerOscillator {
                        delta_bits,
                        alpha_bits,
                    },
                ),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ADAPTIVE_BOUNDS_RSI_ID {
            if route.output_id != ADAPTIVE_BOUNDS_RSI_KERNEL_OUTPUT_IDS[0]
                || !ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ADAPTIVE_BOUNDS_RSI_ID}: admitted output contract is {:?} and full \
                         kernel contract is {:?}, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS,
                        ADAPTIVE_BOUNDS_RSI_KERNEL_OUTPUT_IDS,
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Adaptive Bounds RSI named output has no resident f64 route",
                );
            }
            match resolve_adaptive_bounds_rsi_parameters(&node.parameters) {
                Ok((rsi_length, alpha_bits)) => Some(ClassicCudaResolvedRoute::AdaptiveBoundsRsi {
                    rsi_length,
                    alpha_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ADAPTIVE_MACD_ID {
            if route.output_id != ADAPTIVE_MACD_OUTPUT_IDS[0]
                || !ADAPTIVE_MACD_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ADAPTIVE_MACD_ID}: canonical all-output contract is {:?} with primary \
                         `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        ADAPTIVE_MACD_OUTPUT_IDS, ADAPTIVE_MACD_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Adaptive MACD named output has no resident f64 route",
                );
            }
            match resolve_adaptive_macd_parameters(&node.parameters) {
                Ok((length, fast_period, slow_period, signal_period)) => {
                    Some(ClassicCudaResolvedRoute::AdaptiveMacd {
                        length,
                        fast_period,
                        slow_period,
                        signal_period,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ADAPTIVE_MOMENTUM_OSCILLATOR_ID {
            if route.output_id != ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS[0]
                || !ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: canonical all-output contract is {:?} \
                         with primary `{}`, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS,
                        ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Adaptive Momentum Oscillator named output has no resident f64 route",
                );
            }
            match resolve_adaptive_momentum_oscillator_parameters(&node.parameters) {
                Ok((length, smoothing_length)) => {
                    Some(ClassicCudaResolvedRoute::AdaptiveMomentumOscillator {
                        length,
                        smoothing_length,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ADAPTIVE_SCHAFF_TREND_CYCLE_ID {
            if route.output_id != ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS[0]
                || !ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: canonical all-output contract is {:?} \
                         with primary `{}`, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS,
                        ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Adaptive Schaff Trend Cycle named output has no resident f64 route",
                );
            }
            match resolve_adaptive_schaff_trend_cycle_parameters(&node.parameters) {
                Ok((
                    adaptive_length,
                    stc_length,
                    smoothing_factor_bits,
                    fast_length,
                    slow_length,
                )) => Some(ClassicCudaResolvedRoute::AdaptiveSchaffTrendCycle {
                    adaptive_length,
                    stc_length,
                    smoothing_factor_bits,
                    fast_length,
                    slow_length,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ADJUSTABLE_MA_ID {
            if route.output_id != ADJUSTABLE_MA_FULL_OUTPUT_IDS[0]
                || !ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ADJUSTABLE_MA_ID}: admitted output contract is {:?} and full kernel \
                         contract is {:?}, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS,
                        ADJUSTABLE_MA_FULL_OUTPUT_IDS,
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Adjustable MA named output has no resident f64 route",
                );
            }
            match resolve_adjustable_ma_alternating_extremities_parameters(&node.parameters) {
                Ok((length, mult_bits, alpha_bits, beta_bits)) => Some(
                    ClassicCudaResolvedRoute::AdjustableMaAlternatingExtremities {
                        length,
                        mult_bits,
                        alpha_bits,
                        beta_bits,
                    },
                ),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ALLIGATOR_ID {
            if route.output_id != ALLIGATOR_OUTPUT_IDS[0]
                || !ALLIGATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ALLIGATOR_ID}: canonical all-output contract is {:?} with primary `{}`, \
                         but primary `{}` / requested `{requested_output_id}` was planned",
                        ALLIGATOR_OUTPUT_IDS, ALLIGATOR_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Alligator named output has no resident f64 route",
                );
            }
            match resolve_alligator_parameters(&node.parameters) {
                Ok((
                    jaw_period,
                    jaw_offset,
                    teeth_period,
                    teeth_offset,
                    lips_period,
                    lips_offset,
                )) => Some(ClassicCudaResolvedRoute::Alligator {
                    jaw_period,
                    jaw_offset,
                    teeth_period,
                    teeth_offset,
                    lips_period,
                    lips_offset,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ALPHATREND_ID {
            if route.output_id != ALPHATREND_OUTPUT_IDS[0]
                || !ALPHATREND_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ALPHATREND_ID}: canonical all-output contract is {:?} with primary `{}`, \
                         but primary `{}` / requested `{requested_output_id}` was planned",
                        ALPHATREND_OUTPUT_IDS, ALPHATREND_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the AlphaTrend named output has no resident f64 route",
                );
            }
            match resolve_alphatrend_parameters(&node.parameters) {
                Ok((coeff_bits, period, no_volume)) => Some(ClassicCudaResolvedRoute::AlphaTrend {
                    coeff_bits,
                    period,
                    no_volume,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ACOSC_ID {
            if route.output_id != ACOSC_OUTPUT_IDS[0]
                || !ACOSC_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ACOSC_ID}: canonical all-output contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        ACOSC_OUTPUT_IDS, ACOSC_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the ACOSC named output has no resident f64 route",
                );
            }
            match resolve_acosc_parameters(&node.parameters) {
                Ok(()) => Some(ClassicCudaResolvedRoute::Acosc),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ANDEAN_OSCILLATOR_ID {
            if route.output_id != ANDEAN_OSCILLATOR_OUTPUT_IDS[0]
                || !ANDEAN_OSCILLATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ANDEAN_OSCILLATOR_ID}: canonical all-output contract is {:?} with \
                         primary `{}`, but primary `{}` / requested `{requested_output_id}` was \
                         planned",
                        ANDEAN_OSCILLATOR_OUTPUT_IDS,
                        ANDEAN_OSCILLATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Andean Oscillator named output has no resident f64 route",
                );
            }
            match resolve_andean_oscillator_parameters(&node.parameters) {
                Ok((length, signal_length)) => Some(ClassicCudaResolvedRoute::AndeanOscillator {
                    length,
                    signal_length,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == AROON_ID {
            if route.output_id != AROON_OUTPUT_IDS[0]
                || !AROON_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{AROON_ID}: canonical all-output contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        AROON_OUTPUT_IDS, AROON_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Aroon named output has no resident f64 route",
                );
            }
            match resolve_aroon_parameters(&node.parameters) {
                Ok(length) => Some(ClassicCudaResolvedRoute::Aroon { length }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ASO_ID {
            if route.output_id != ASO_OUTPUT_IDS[0]
                || !ASO_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ASO_ID}: canonical all-output contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        ASO_OUTPUT_IDS, ASO_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the ASO named output has no resident f64 route",
                );
            }
            match resolve_aso_parameters(&node.parameters) {
                Ok((period, mode)) => Some(ClassicCudaResolvedRoute::Aso { period, mode }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == AUTOCORRELATION_INDICATOR_ID {
            if route.output_id != AUTOCORRELATION_INDICATOR_OUTPUT_IDS[0]
                || !AUTOCORRELATION_INDICATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{AUTOCORRELATION_INDICATOR_ID}: canonical selected-output contract is \
                         {:?} with primary `{}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        AUTOCORRELATION_INDICATOR_OUTPUT_IDS,
                        AUTOCORRELATION_INDICATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Autocorrelation Indicator named output has no resident f64 route",
                );
            }
            match resolve_autocorrelation_indicator_parameters(&node.parameters) {
                Ok((length, lag, use_test_signal)) => {
                    Some(ClassicCudaResolvedRoute::AutocorrelationIndicator {
                        length,
                        lag,
                        use_test_signal,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == AVSL_ID {
            if route.output_id != AVSL_OUTPUT_IDS[0]
                || !AVSL_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{AVSL_ID}: canonical production contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        AVSL_OUTPUT_IDS, AVSL_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the AVSL value output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "AVSL is still classified period-invariant even though its production route \
                     must consume the admitted slow-period anchor",
                );
            }
            match resolve_avsl_parameters(&node.parameters) {
                Ok((fast_period, slow_period, multiplier_bits)) => {
                    Some(ClassicCudaResolvedRoute::Avsl {
                        fast_period,
                        slow_period,
                        multiplier_bits,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == BANDPASS_ID {
            if route.output_id != BANDPASS_OUTPUT_IDS[0]
                || !BANDPASS_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{BANDPASS_ID}: canonical production contract is {:?} with primary `{}`, \
                         but primary `{}` / requested `{requested_output_id}` was planned",
                        BANDPASS_OUTPUT_IDS, BANDPASS_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Bandpass named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Bandpass is still classified period-invariant even though its production \
                     route must consume the admitted period anchor",
                );
            }
            match resolve_bandpass_parameters(&node.parameters) {
                Ok((period, bandwidth_bits)) => Some(ClassicCudaResolvedRoute::Bandpass {
                    period,
                    bandwidth_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == BOLLINGER_BANDS_ID {
            if route.output_id != BOLLINGER_BANDS_OUTPUT_IDS[0]
                || !BOLLINGER_BANDS_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{BOLLINGER_BANDS_ID}: canonical production contract is {:?} with \
                         primary `{}`, but primary `{}` / requested `{requested_output_id}` was \
                         planned",
                        BOLLINGER_BANDS_OUTPUT_IDS, BOLLINGER_BANDS_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Bollinger Bands named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Bollinger Bands is still classified period-invariant even though its \
                     production route must consume the admitted period anchor",
                );
            }
            match resolve_bollinger_bands_parameters(&node.parameters) {
                Ok((period, devup_bits, devdn_bits)) => {
                    Some(ClassicCudaResolvedRoute::BollingerBands {
                        period,
                        devup_bits,
                        devdn_bits,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == BUFF_AVERAGES_ID {
            if route.output_id != BUFF_AVERAGES_OUTPUT_IDS[0]
                || !BUFF_AVERAGES_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{BUFF_AVERAGES_ID}: canonical production contract is {:?} with primary \
                         `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        BUFF_AVERAGES_OUTPUT_IDS, BUFF_AVERAGES_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Buff Averages named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Buff Averages is still classified period-invariant even though its \
                     production route must consume both admitted window parameters",
                );
            }
            match resolve_buff_averages_parameters(&node.parameters) {
                Ok((fast_period, slow_period)) => Some(ClassicCudaResolvedRoute::BuffAverages {
                    fast_period,
                    slow_period,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CANDLE_STRENGTH_OSCILLATOR_ID {
            if route.output_id != CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS[0]
                || !CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CANDLE_STRENGTH_OSCILLATOR_ID}: canonical production contract is {:?} \
                         with primary `{}`, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS,
                        CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Candle Strength Oscillator named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Candle Strength Oscillator is still classified period-invariant even \
                     though its production route must consume the admitted period anchor",
                );
            }
            match resolve_candle_strength_oscillator_parameters(&node.parameters) {
                Ok((period, atr_enabled, atr_length, mode)) => {
                    Some(ClassicCudaResolvedRoute::CandleStrengthOscillator {
                        period,
                        atr_enabled,
                        atr_length,
                        mode,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CHANDELIER_EXIT_ID {
            if route.output_id != CHANDELIER_EXIT_OUTPUT_IDS[0]
                || !CHANDELIER_EXIT_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CHANDELIER_EXIT_ID}: canonical production contract is {:?} with primary \
                         `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        CHANDELIER_EXIT_OUTPUT_IDS, CHANDELIER_EXIT_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Chandelier Exit named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Chandelier Exit is still classified period-invariant even though its \
                     production route must consume the admitted period anchor",
                );
            }
            match resolve_chandelier_exit_parameters(&node.parameters) {
                Ok((period, mult_bits, use_close)) => {
                    Some(ClassicCudaResolvedRoute::ChandelierExit {
                        period,
                        mult_bits,
                        use_close,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CKSP_ID {
            if route.output_id != CKSP_OUTPUT_IDS[0]
                || !CKSP_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CKSP_ID}: canonical production contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        CKSP_OUTPUT_IDS, CKSP_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the CKSP named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "CKSP's preserved generic primary must remain period-invariant while the \
                     admitted feature vocabulary is default-only",
                );
            }
            match resolve_cksp_parameters(&node.parameters) {
                Ok((p, x_bits, q)) => Some(ClassicCudaResolvedRoute::Cksp { p, x_bits, q }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == COPPOCK_ID {
            if route.output_id != COPPOCK_OUTPUT_ID || requested_output_id != COPPOCK_OUTPUT_ID {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{COPPOCK_ID}: canonical production contract is the sole output \
                         `{COPPOCK_OUTPUT_ID}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Coppock value output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Coppock is still classified period-invariant even though its production \
                     route must consume the admitted 11:14:10 RegistryRatio tuple",
                );
            }
            match resolve_coppock_parameters(&node.parameters) {
                Ok((short_roc_period, long_roc_period, ma_period)) => {
                    Some(ClassicCudaResolvedRoute::Coppock {
                        short_roc_period,
                        long_roc_period,
                        ma_period,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CORRELATION_CYCLE_ID {
            if route.output_id != CORRELATION_CYCLE_OUTPUT_IDS[0]
                || !CORRELATION_CYCLE_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CORRELATION_CYCLE_ID}: canonical production contract is {:?} with \
                         primary `{}`, but primary `{}` / requested `{requested_output_id}` was \
                         planned",
                        CORRELATION_CYCLE_OUTPUT_IDS,
                        CORRELATION_CYCLE_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Correlation Cycle named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Correlation Cycle is still classified period-invariant even though its \
                     production route must consume the admitted period anchor",
                );
            }
            match resolve_correlation_cycle_parameters(&node.parameters) {
                Ok((period, threshold_bits)) => Some(ClassicCudaResolvedRoute::CorrelationCycle {
                    period,
                    threshold_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CVI_ID {
            if route.output_id != CVI_OUTPUT_ID || requested_output_id != CVI_OUTPUT_ID {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CVI_ID}: canonical production contract is the sole output \
                         `{CVI_OUTPUT_ID}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the CVI value output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "CVI is classified period-invariant even though its production route must \
                     consume every admitted period",
                );
            }
            match resolve_cvi_parameters(&node.parameters) {
                Ok(period) => Some(ClassicCudaResolvedRoute::Cvi { period }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CYBERPUNK_VALUE_TREND_ANALYZER_ID {
            if route.output_id != CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS[0]
                || !CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: canonical production contract is \
                         {:?} with primary `{}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS,
                        CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Cyberpunk Value Trend Analyzer named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Cyberpunk Value Trend Analyzer must keep the preserved primary ABI \
                     period-invariant because its formula has no period parameter",
                );
            }
            match resolve_cyberpunk_value_trend_analyzer_parameters(&node.parameters) {
                Ok((entry_level, exit_level)) => {
                    Some(ClassicCudaResolvedRoute::CyberpunkValueTrendAnalyzer {
                        entry_level,
                        exit_level,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == CYCLE_CHANNEL_OSCILLATOR_ID {
            if route.output_id != CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS[0]
                || !CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{CYCLE_CHANNEL_OSCILLATOR_ID}: canonical production contract is {:?} \
                         with primary `{}`, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS,
                        CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Cycle Channel Oscillator named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Cycle Channel Oscillator's preserved primary default-fast ABI must remain \
                     period-invariant; the typed full-kernel route carries ratio tuples",
                );
            }
            match resolve_cycle_channel_oscillator_parameters(&node.parameters) {
                Ok((
                    short_cycle_length,
                    medium_cycle_length,
                    short_multiplier_bits,
                    medium_multiplier_bits,
                )) => Some(ClassicCudaResolvedRoute::CycleChannelOscillator {
                    short_cycle_length,
                    medium_cycle_length,
                    short_multiplier_bits,
                    medium_multiplier_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DAILY_FACTOR_ID {
            if route.output_id != DAILY_FACTOR_FULL_OUTPUT_IDS[0]
                || !DAILY_FACTOR_PRODUCTION_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DAILY_FACTOR_ID}: admitted output contract is {:?} and full kernel \
                         contract is {:?}, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        DAILY_FACTOR_PRODUCTION_OUTPUT_IDS,
                        DAILY_FACTOR_FULL_OUTPUT_IDS,
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Daily Factor named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Daily Factor must keep the preserved primary value ABI period-invariant \
                     because its formula has no period parameter",
                );
            }
            match resolve_daily_factor_parameters(&node.parameters) {
                Ok(threshold_level_bits) => Some(ClassicCudaResolvedRoute::DailyFactor {
                    threshold_level_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DAMIANI_VOLATMETER_ID {
            if route.output_id != DAMIANI_VOLATMETER_OUTPUT_IDS[0]
                || !DAMIANI_VOLATMETER_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DAMIANI_VOLATMETER_ID}: canonical production contract is {:?} with \
                         primary `{}`, but primary `{}` / requested `{requested_output_id}` was \
                         planned",
                        DAMIANI_VOLATMETER_OUTPUT_IDS,
                        DAMIANI_VOLATMETER_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Damiani Volatmeter named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Damiani Volatmeter's preserved primary default-vol ABI must remain \
                     period-invariant; the typed pair route carries the dynamic ratio tuple",
                );
            }
            match resolve_damiani_volatmeter_parameters(&node.parameters) {
                Ok((vis_atr, vis_std, sed_atr, sed_std, threshold_bits)) => {
                    Some(ClassicCudaResolvedRoute::DamianiVolatmeter {
                        vis_atr,
                        vis_std,
                        sed_atr,
                        sed_std,
                        threshold_bits,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DI_ID {
            if route.output_id != DI_OUTPUT_IDS[0] || !DI_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DI_ID}: canonical production contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        DI_OUTPUT_IDS, DI_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the DI named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "DI is classified period-invariant even though its production route must \
                     consume every admitted period",
                );
            }
            match resolve_di_parameters(&node.parameters) {
                Ok(period) => Some(ClassicCudaResolvedRoute::Di { period }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DIDI_INDEX_ID {
            if route.output_id != DIDI_INDEX_OUTPUT_IDS[0]
                || !DIDI_INDEX_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DIDI_INDEX_ID}: canonical production contract is {:?} with primary \
                         `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        DIDI_INDEX_OUTPUT_IDS, DIDI_INDEX_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Didi Index named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Didi Index's preserved generic primary must remain fixed at canonical \
                     3:8:20 while the typed full-output route carries RegistryRatio tuples",
                );
            }
            match resolve_didi_index_parameters(&node.parameters) {
                Ok((short_length, medium_length, long_length)) => {
                    Some(ClassicCudaResolvedRoute::DidiIndex {
                        short_length,
                        medium_length,
                        long_length,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DIRECTIONAL_IMBALANCE_INDEX_ID {
            if route.output_id != DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS[0]
                || !DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DIRECTIONAL_IMBALANCE_INDEX_ID}: canonical production contract is \
                         {:?} with primary `{}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS,
                        DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Directional Imbalance Index named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Directional Imbalance Index is classified period-invariant even though \
                     its production route must consume every admitted period",
                );
            }
            match resolve_directional_imbalance_index_parameters(&node.parameters) {
                Ok((length, period)) => {
                    Some(ClassicCudaResolvedRoute::DirectionalImbalanceIndex { length, period })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DISPARITY_INDEX_ID {
            if route.output_id != DISPARITY_INDEX_OUTPUT_ID
                || requested_output_id != DISPARITY_INDEX_OUTPUT_ID
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DISPARITY_INDEX_ID}: canonical production contract is sole output \
                         `{DISPARITY_INDEX_OUTPUT_ID}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Disparity Index value output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Disparity Index's preserved generic primary must remain fixed at canonical \
                     14:14:9:ema while the typed full route carries each lookback tuple",
                );
            }
            match resolve_disparity_index_parameters(&node.parameters) {
                Ok((ema_period, lookback_period, smoothing_period, smoothing_is_sma)) => {
                    Some(ClassicCudaResolvedRoute::DisparityIndex {
                        ema_period,
                        lookback_period,
                        smoothing_period,
                        smoothing_is_sma,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DM_ID {
            if route.output_id != DM_OUTPUT_IDS[0] || !DM_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DM_ID}: canonical production contract is {:?} with primary `{}`, but \
                         primary `{}` / requested `{requested_output_id}` was planned",
                        DM_OUTPUT_IDS, DM_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the DM named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "DM is classified period-invariant even though its production route must \
                     consume every admitted period",
                );
            }
            match resolve_dm_parameters(&node.parameters) {
                Ok(period) => Some(ClassicCudaResolvedRoute::Dm { period }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DONCHIAN_ID {
            if route.output_id != DONCHIAN_OUTPUT_IDS[0]
                || !DONCHIAN_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DONCHIAN_ID}: canonical production contract is {:?} with primary `{}`, \
                         but primary `{}` / requested `{requested_output_id}` was planned",
                        DONCHIAN_OUTPUT_IDS, DONCHIAN_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Donchian named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Donchian is classified period-invariant even though its production route \
                     must consume every admitted period",
                );
            }
            match resolve_donchian_parameters(&node.parameters) {
                Ok(period) => Some(ClassicCudaResolvedRoute::Donchian { period }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DUAL_ULCER_INDEX_ID {
            if route.output_id != DUAL_ULCER_INDEX_OUTPUT_IDS[0]
                || !DUAL_ULCER_INDEX_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DUAL_ULCER_INDEX_ID}: canonical production contract is {:?} with \
                         primary `{}`, but primary `{}` / requested `{requested_output_id}` was \
                         planned",
                        DUAL_ULCER_INDEX_OUTPUT_IDS,
                        DUAL_ULCER_INDEX_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Dual Ulcer Index named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Dual Ulcer Index is classified period-invariant even though its production \
                     route must consume every admitted period",
                );
            }
            match resolve_dual_ulcer_index_parameters(&node.parameters) {
                Ok((period, auto_threshold, threshold_bits)) => {
                    Some(ClassicCudaResolvedRoute::DualUlcerIndex {
                        period,
                        auto_threshold,
                        threshold_bits,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == DVDIQQE_ID {
            if route.output_id != DVDIQQE_OUTPUT_IDS[0]
                || !DVDIQQE_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{DVDIQQE_ID}: canonical production contract is {:?} with primary `{}`, \
                         but primary `{}` / requested `{requested_output_id}` was planned",
                        DVDIQQE_OUTPUT_IDS, DVDIQQE_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the DVDIQQE named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "DVDIQQE is classified period-invariant even though its production route \
                     must consume every admitted period",
                );
            }
            match resolve_dvdiqqe_parameters(&node.parameters) {
                Ok((
                    period,
                    smoothing_period,
                    fast_multiplier_bits,
                    slow_multiplier_bits,
                    use_tick_only,
                    dynamic_center,
                    tick_size_bits,
                )) => Some(ClassicCudaResolvedRoute::Dvdiqqe {
                    period,
                    smoothing_period,
                    fast_multiplier_bits,
                    slow_multiplier_bits,
                    use_tick_only,
                    dynamic_center,
                    tick_size_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EHLERS_AUTOCORRELATION_PERIODOGRAM_ID {
            if route.output_id != EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS[0]
                || !EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: canonical production contract \
                         is {:?} with primary `{}`, but primary `{}` / requested \
                         `{requested_output_id}` was planned",
                        EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS,
                        EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Ehlers Autocorrelation Periodogram named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "the preserved generic periodogram primary must remain fixed at 8:48:3:true \
                     while the typed full-output route carries RegistryRatio tuples",
                );
            }
            match resolve_ehlers_autocorrelation_periodogram_parameters(&node.parameters) {
                Ok((min_period, max_period, avg_length, enhance)) => {
                    Some(ClassicCudaResolvedRoute::EhlersAutocorrelationPeriodogram {
                        min_period,
                        max_period,
                        avg_length,
                        enhance,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID {
            if route.output_id != EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS[0]
                || !EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS,
                        EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the ELEP named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "the preserved generic ELEP primary must remain fixed at defaults while the typed five-output route carries RegistryRatio tuples",
                );
            }
            match resolve_ehlers_linear_extrapolation_predictor_parameters(&node.parameters) {
                Ok((high_pass_length, low_pass_length, gain_bits, bars_forward, signal_mode)) => {
                    Some(
                        ClassicCudaResolvedRoute::EhlersLinearExtrapolationPredictor {
                            high_pass_length,
                            low_pass_length,
                            gain_bits,
                            bars_forward,
                            signal_mode,
                        },
                    )
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID {
            if route.output_id != EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS[0]
                || !EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
                    .contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS,
                        EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the EUDMA named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "the preserved generic EUDMA primary must remain fixed at defaults while the typed pair route carries RegistryRatio tuples",
                );
            }
            match resolve_ehlers_undersampled_double_moving_average_parameters(&node.parameters) {
                Ok((fast_length, slow_length, sample_length)) => Some(
                    ClassicCudaResolvedRoute::EhlersUndersampledDoubleMovingAverage {
                        fast_length,
                        slow_length,
                        sample_length,
                    },
                ),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EMA_DEVIATION_CORRECTED_T3_ID {
            if route.output_id != EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS[0]
                || !EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EMA_DEVIATION_CORRECTED_T3_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS,
                        EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the EDCT3 named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "EDCT3 is classified period-invariant even though its canonical production route must consume period 10 and every admitted sweep",
                );
            }
            match resolve_ema_deviation_corrected_t3_parameters(&node.parameters) {
                Ok((period, hot_bits, t3_mode)) => {
                    Some(ClassicCudaResolvedRoute::EmaDeviationCorrectedT3 {
                        period,
                        hot_bits,
                        t3_mode,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EMD_ID {
            if route.output_id != EMD_OUTPUT_IDS[0]
                || !EMD_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EMD_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        EMD_OUTPUT_IDS, EMD_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the EMD named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "EMD is classified period-invariant even though its exact triple route must consume period 20 and every admitted sweep",
                );
            }
            match resolve_emd_parameters(&node.parameters) {
                Ok((period, delta_bits, fraction_bits)) => Some(ClassicCudaResolvedRoute::Emd {
                    period,
                    delta_bits,
                    fraction_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EMD_TREND_ID {
            if route.output_id != "average" || !EMD_TREND_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EMD_TREND_ID}: canonical production contract is {:?} while the preserved primary ABI is `average`, but primary `{}` / requested `{requested_output_id}` was planned",
                        EMD_TREND_OUTPUT_IDS, route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the EMD Trend named output has no resident f64 route",
                );
            }
            match resolve_emd_trend_parameters(&node.parameters) {
                Ok((length, mult_bits)) => {
                    Some(ClassicCudaResolvedRoute::EmdTrend { length, mult_bits })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == ERI_ID {
            if route.output_id != ERI_OUTPUT_IDS[0]
                || !ERI_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{ERI_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        ERI_OUTPUT_IDS, ERI_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the ERI named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "ERI is classified period-invariant even though its exact pair route must consume period 13 and every admitted sweep",
                );
            }
            match resolve_eri_parameters(&node.parameters) {
                Ok(period) => Some(ClassicCudaResolvedRoute::Eri { period }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EVASIVE_SUPERTREND_ID {
            if route.output_id != EVASIVE_SUPERTREND_OUTPUT_IDS[0]
                || !EVASIVE_SUPERTREND_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EVASIVE_SUPERTREND_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        EVASIVE_SUPERTREND_OUTPUT_IDS,
                        EVASIVE_SUPERTREND_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Evasive Supertrend named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Evasive Supertrend is classified period-invariant even though its exact quad route must consume atr_length 10 and every admitted sweep",
                );
            }
            match resolve_evasive_supertrend_parameters(&node.parameters) {
                Ok((
                    atr_length,
                    base_multiplier_bits,
                    noise_threshold_bits,
                    expansion_alpha_bits,
                )) => Some(ClassicCudaResolvedRoute::EvasiveSupertrend {
                    atr_length,
                    base_multiplier_bits,
                    noise_threshold_bits,
                    expansion_alpha_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == FIBONACCI_TRAILING_STOP_ID {
            if route.output_id != FIBONACCI_TRAILING_STOP_OUTPUT_IDS[0]
                || !FIBONACCI_TRAILING_STOP_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{FIBONACCI_TRAILING_STOP_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        FIBONACCI_TRAILING_STOP_OUTPUT_IDS,
                        FIBONACCI_TRAILING_STOP_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Fibonacci Trailing Stop named output has no resident f64 route",
                );
            }
            if !route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Fibonacci Trailing Stop's preserved primary ABI must remain period-invariant while the current admitted vocabulary is default-only",
                );
            }
            match resolve_fibonacci_trailing_stop_parameters(&node.parameters) {
                Ok((left_bars, right_bars, level_bits, trigger_mode)) => {
                    Some(ClassicCudaResolvedRoute::FibonacciTrailingStop {
                        left_bars,
                        right_bars,
                        level_bits,
                        trigger_mode,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == FISHER_ID {
            if route.output_id != FISHER_OUTPUT_IDS[0]
                || !FISHER_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{FISHER_ID}: canonical production contract is {:?} with primary `{}`, but primary `{}` / requested `{requested_output_id}` was planned",
                        FISHER_OUTPUT_IDS, FISHER_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Fisher named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Fisher is classified period-invariant even though its production pair route must consume every admitted period",
                );
            }
            match resolve_fisher_parameters(&node.parameters) {
                Ok(period) => Some(ClassicCudaResolvedRoute::Fisher { period }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID {
            if route.output_id != FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS[0]
                || !FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
                    .contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: canonical production contract is {:?} with primary {}, but primary {} / requested {requested_output_id} was planned",
                        FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS,
                        FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the FBEO named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "FBEO is classified period-invariant even though its production triple route must consume every admitted length",
                );
            }
            match resolve_forward_backward_exponential_oscillator_parameters(&node.parameters) {
                Ok((length, smooth)) => Some(
                    ClassicCudaResolvedRoute::ForwardBackwardExponentialOscillator {
                        length,
                        smooth,
                    },
                ),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == FVG_TRAILING_STOP_ID {
            if route.output_id != FVG_TRAILING_STOP_OUTPUT_IDS[0]
                || !FVG_TRAILING_STOP_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{FVG_TRAILING_STOP_ID}: canonical production contract is {:?} with primary {}, but primary {} / requested {requested_output_id} was planned",
                        FVG_TRAILING_STOP_OUTPUT_IDS,
                        FVG_TRAILING_STOP_OUTPUT_IDS[0],
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the FVG Trailing Stop named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "FVG Trailing Stop is classified period-invariant even though its production route must consume every admitted smoothing_length",
                );
            }
            match resolve_fvg_trailing_stop_parameters(&node.parameters) {
                Ok((unmitigated_fvg_lookback, smoothing_length, reset_on_cross)) => {
                    Some(ClassicCudaResolvedRoute::FvgTrailingStop {
                        unmitigated_fvg_lookback,
                        smoothing_length,
                        reset_on_cross,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == GATOROSC_ID {
            if route.output_id != GATOROSC_OUTPUT_IDS[0]
                || !GATOROSC_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{GATOROSC_ID}: canonical production contract is {:?} with primary {}, but primary {} / requested {requested_output_id} was planned",
                        GATOROSC_OUTPUT_IDS, GATOROSC_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Gator Oscillator named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Gator Oscillator is classified period-invariant even though its production route must consume every admitted length ratio",
                );
            }
            match resolve_gatorosc_parameters(&node.parameters) {
                Ok((
                    jaws_length,
                    jaws_shift,
                    teeth_length,
                    teeth_shift,
                    lips_length,
                    lips_shift,
                )) => Some(ClassicCudaResolvedRoute::Gatorosc {
                    jaws_length,
                    jaws_shift,
                    teeth_length,
                    teeth_shift,
                    lips_length,
                    lips_shift,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == HALFTREND_ID {
            if route.output_id != HALFTREND_OUTPUT_IDS[0]
                || !HALFTREND_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{HALFTREND_ID}: canonical production contract is {:?} with primary {}, but primary {} / requested {requested_output_id} was planned",
                        HALFTREND_OUTPUT_IDS, HALFTREND_OUTPUT_IDS[0], route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the HalfTrend named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "HalfTrend is classified period-invariant even though its production route must consume every admitted atr_period",
                );
            }
            match resolve_halftrend_parameters(&node.parameters) {
                Ok((amplitude, channel_deviation_bits, atr_period)) => {
                    Some(ClassicCudaResolvedRoute::Halftrend {
                        amplitude,
                        channel_deviation_bits,
                        atr_period,
                    })
                }
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == FIBONACCI_ENTRY_BANDS_ID {
            if route.output_id != FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS[0]
                || !FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{FIBONACCI_ENTRY_BANDS_ID}: admitted output contract is {:?} and full kernel contract is {:?}, but primary `{}` / requested `{requested_output_id}` was planned",
                        FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS,
                        FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS,
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Fibonacci Entry Bands named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "Fibonacci Entry Bands is classified period-invariant even though its exact full route must consume length 21 and every admitted sweep",
                );
            }
            match resolve_fibonacci_entry_bands_parameters(&node.parameters) {
                Ok(length) => Some(ClassicCudaResolvedRoute::FibonacciEntryBands { length }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == EHLERS_DATA_SAMPLING_RSI_ID {
            if route.output_id != EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS[0]
                || !EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{EHLERS_DATA_SAMPLING_RSI_ID}: admitted output contract is {:?} and full kernel contract is {:?}, but primary `{}` / requested `{requested_output_id}` was planned",
                        EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS,
                        EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS,
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the EDSRSI named output has no resident f64 route",
                );
            }
            if route.period_invariant {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingParameterContract,
                    "EDSRSI is classified period-invariant even though its production route must consume every admitted length",
                );
            }
            match resolve_ehlers_data_sampling_rsi_length(&node.parameters) {
                Ok(length) => Some(ClassicCudaResolvedRoute::EhlersDataSamplingRsi { length }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else if node.indicator_id == BULLS_V_BEARS_ID {
            if route.output_id != BULLS_V_BEARS_FULL_OUTPUT_IDS[0]
                || !BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS.contains(&requested_output_id)
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NamedFamilyContractMismatch,
                    format!(
                        "{BULLS_V_BEARS_ID}: admitted output contract is {:?} and full kernel \
                         contract is {:?}, but primary `{}` / requested `{requested_output_id}` \
                         was planned",
                        BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS,
                        BULLS_V_BEARS_FULL_OUTPUT_IDS,
                        route.output_id
                    ),
                );
            }
            if !has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::MissingNamedOutputRoute,
                    "the Bulls v Bears named output has no resident f64 route",
                );
            }
            match resolve_bulls_v_bears_parameters(&node.parameters) {
                Ok((
                    period,
                    ma_type,
                    calculation_method,
                    normalized_bars_back,
                    raw_rolling_period,
                    raw_threshold_percentile_bits,
                    threshold_level_bits,
                )) => Some(ClassicCudaResolvedRoute::BullsVBears {
                    period,
                    ma_type,
                    calculation_method,
                    normalized_bars_back,
                    raw_rolling_period,
                    raw_threshold_percentile_bits,
                    threshold_level_bits,
                }),
                Err(detail) => {
                    push_gap(
                        &mut gaps,
                        node,
                        ClassicCudaGapReason::MissingParameterContract,
                        detail,
                    );
                    None
                }
            }
        } else {
            if requested_output_id != route.output_id {
                let (reason, detail) =
                    if has_f64_resident_output_route(node.indicator_id, requested_output_id) {
                        (
                            ClassicCudaGapReason::MissingNamedProductionDispatcher,
                            "a low-level resident named-output kernel exists, but the canonical \
                             typed executor has no source-stable dispatcher for it yet",
                        )
                    } else {
                        (
                            ClassicCudaGapReason::MissingNamedOutputRoute,
                            "the named output has no resident f64 route",
                        )
                    };
                push_gap(&mut gaps, node, reason, detail);
            }
            if matches!(
                &node.parameters,
                ClassicCudaParameters::Defaults {
                    require_period_invariant_kernel: true,
                    ..
                }
            ) && !route.period_invariant
            {
                push_gap(
                    &mut gaps,
                    node,
                    ClassicCudaGapReason::NoWindowKernelConsumesAnchor,
                    "CPU declares no window, but the registered f64 kernel consumes the supplied \
                     anchor; passing an invented value would change the formula",
                );
            }
            Some(ClassicCudaResolvedRoute::Primary { cuda_period })
        };

        if gaps.len() == before
            && let Some(route) = resolved_route
        {
            resolved.push(ResolvedClassicCudaNode {
                node: node.clone(),
                output_id: requested_output_id,
                route,
            });
        }
    }

    if gaps.is_empty() {
        group_resolved_classic_cuda_launches(resolved)
    } else {
        Err(gaps)
    }
}

/// Collapse a proven canonical multi-output family into one launch while
/// retaining one receipt-bearing node per emitted column.  This runs inside
/// preflight: a drifted output set/order or parameter point cannot be noticed
/// only after a CUDA context exists.
fn group_resolved_classic_cuda_launches(
    resolved: Vec<ResolvedClassicCudaNode>,
) -> std::result::Result<Vec<ResolvedClassicCudaLaunch>, Vec<ClassicCudaGap>> {
    let mut launches = Vec::with_capacity(resolved.len());
    let mut gaps = Vec::new();
    let mut index = 0usize;

    while index < resolved.len() {
        match &resolved[index].route {
            ClassicCudaResolvedRoute::Primary { .. } => {
                launches.push(ResolvedClassicCudaLaunch::Primary(resolved[index].clone()));
                index += 1;
            }
            ClassicCudaResolvedRoute::AbsoluteStrengthIndexOscillator {
                ema_length,
                signal_length,
            } => {
                let remaining = resolved.len() - index;
                if remaining < ASI_OUTPUT_IDS.len() {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ASI_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                ASI_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + ASI_OUTPUT_IDS.len()];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ASI_ID
                        && candidate.output_id == ASI_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AbsoluteStrengthIndexOscillator {
                                ema_length: *ema_length,
                                signal_length: *signal_length,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ASI_ID}: expected one contiguous {:?} group at tuple \
                                 ({ema_length}, {signal_length})",
                                ASI_OUTPUT_IDS
                            ),
                        );
                    }
                    index += ASI_OUTPUT_IDS.len();
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AbsoluteStrengthIndexOscillator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    ema_length: *ema_length,
                    signal_length: *signal_length,
                });
                index += ASI_OUTPUT_IDS.len();
            }
            ClassicCudaResolvedRoute::AdaptiveBandpassTriggerOscillator {
                delta_bits,
                alpha_bits,
            } => {
                let group_len = ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: expected one \
                                 contiguous {:?} group, found only {remaining} remaining output(s)",
                                ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID
                        && candidate.output_id
                            == ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AdaptiveBandpassTriggerOscillator {
                                delta_bits: *delta_bits,
                                alpha_bits: *alpha_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: expected one \
                                 contiguous {:?} group at (delta_bits={delta_bits:#018x}, \
                                 alpha_bits={alpha_bits:#018x})",
                                ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(
                    ResolvedClassicCudaLaunch::AdaptiveBandpassTriggerOscillator {
                        routes: std::array::from_fn(|offset| candidates[offset].clone()),
                        delta_bits: *delta_bits,
                        alpha_bits: *alpha_bits,
                    },
                );
                index += group_len;
            }
            ClassicCudaResolvedRoute::AdaptiveBoundsRsi {
                rsi_length,
                alpha_bits,
            } => {
                let group_len = ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_BOUNDS_RSI_ID}: expected one contiguous {:?} group, \
                                 found only {remaining} remaining output(s)",
                                ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ADAPTIVE_BOUNDS_RSI_ID
                        && candidate.output_id == ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AdaptiveBoundsRsi {
                                rsi_length: *rsi_length,
                                alpha_bits: *alpha_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_BOUNDS_RSI_ID}: expected one contiguous {:?} group at \
                                 (rsi_length={rsi_length}, alpha_bits={alpha_bits:#018x})",
                                ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AdaptiveBoundsRsi {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    rsi_length: *rsi_length,
                    alpha_bits: *alpha_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AdaptiveMacd {
                length,
                fast_period,
                slow_period,
                signal_period,
            } => {
                let group_len = ADAPTIVE_MACD_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_MACD_ID}: expected one contiguous {:?} group, found \
                                 only {remaining} remaining output(s)",
                                ADAPTIVE_MACD_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ADAPTIVE_MACD_ID
                        && candidate.output_id == ADAPTIVE_MACD_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AdaptiveMacd {
                                length: *length,
                                fast_period: *fast_period,
                                slow_period: *slow_period,
                                signal_period: *signal_period,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_MACD_ID}: expected one contiguous {:?} group at tuple \
                                 ({length}, {fast_period}, {slow_period}, {signal_period})",
                                ADAPTIVE_MACD_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AdaptiveMacd {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                    fast_period: *fast_period,
                    slow_period: *slow_period,
                    signal_period: *signal_period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AdaptiveMomentumOscillator {
                length,
                smoothing_length,
            } => {
                let group_len = ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: expected one contiguous {:?} \
                                 group, found only {remaining} remaining output(s)",
                                ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ADAPTIVE_MOMENTUM_OSCILLATOR_ID
                        && candidate.output_id == ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AdaptiveMomentumOscillator {
                                length: *length,
                                smoothing_length: *smoothing_length,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: expected one contiguous {:?} \
                                 group at tuple ({length}, {smoothing_length})",
                                ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AdaptiveMomentumOscillator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                    smoothing_length: *smoothing_length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AdaptiveSchaffTrendCycle {
                adaptive_length,
                stc_length,
                smoothing_factor_bits,
                fast_length,
                slow_length,
            } => {
                let group_len = ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: expected one contiguous {:?} \
                                 group, found only {remaining} remaining output(s)",
                                ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ADAPTIVE_SCHAFF_TREND_CYCLE_ID
                        && candidate.output_id == ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AdaptiveSchaffTrendCycle {
                                adaptive_length: *adaptive_length,
                                stc_length: *stc_length,
                                smoothing_factor_bits: *smoothing_factor_bits,
                                fast_length: *fast_length,
                                slow_length: *slow_length,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: expected one contiguous {:?} \
                                 group at tuple ({adaptive_length}, {stc_length}, \
                                 {}, {fast_length}, {slow_length})",
                                ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS,
                                f64::from_bits(*smoothing_factor_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AdaptiveSchaffTrendCycle {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    adaptive_length: *adaptive_length,
                    stc_length: *stc_length,
                    smoothing_factor_bits: *smoothing_factor_bits,
                    fast_length: *fast_length,
                    slow_length: *slow_length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AdjustableMaAlternatingExtremities {
                length,
                mult_bits,
                alpha_bits,
                beta_bits,
            } => {
                let group_len = ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADJUSTABLE_MA_ID}: expected one contiguous admitted {:?} \
                                 group, found only {remaining} remaining output(s)",
                                ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ADJUSTABLE_MA_ID
                        && candidate.output_id == ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AdjustableMaAlternatingExtremities {
                                length: *length,
                                mult_bits: *mult_bits,
                                alpha_bits: *alpha_bits,
                                beta_bits: *beta_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ADJUSTABLE_MA_ID}: expected one contiguous admitted {:?} group \
                                 at tuple ({length}, {}, {}, {})",
                                ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS,
                                f64::from_bits(*mult_bits),
                                f64::from_bits(*alpha_bits),
                                f64::from_bits(*beta_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(
                    ResolvedClassicCudaLaunch::AdjustableMaAlternatingExtremities {
                        routes: std::array::from_fn(|offset| candidates[offset].clone()),
                        length: *length,
                        mult_bits: *mult_bits,
                        alpha_bits: *alpha_bits,
                        beta_bits: *beta_bits,
                    },
                );
                index += group_len;
            }
            ClassicCudaResolvedRoute::Alligator {
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
            } => {
                let group_len = ALLIGATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ALLIGATOR_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                ALLIGATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ALLIGATOR_ID
                        && candidate.output_id == ALLIGATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Alligator {
                                jaw_period: *jaw_period,
                                jaw_offset: *jaw_offset,
                                teeth_period: *teeth_period,
                                teeth_offset: *teeth_offset,
                                lips_period: *lips_period,
                                lips_offset: *lips_offset,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ALLIGATOR_ID}: expected one contiguous {:?} group at tuple \
                                 ({jaw_period}, {jaw_offset}, {teeth_period}, {teeth_offset}, \
                                 {lips_period}, {lips_offset})",
                                ALLIGATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Alligator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    jaw_period: *jaw_period,
                    jaw_offset: *jaw_offset,
                    teeth_period: *teeth_period,
                    teeth_offset: *teeth_offset,
                    lips_period: *lips_period,
                    lips_offset: *lips_offset,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AlphaTrend {
                coeff_bits,
                period,
                no_volume,
            } => {
                let group_len = ALPHATREND_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ALPHATREND_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                ALPHATREND_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ALPHATREND_ID
                        && candidate.output_id == ALPHATREND_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AlphaTrend {
                                coeff_bits: *coeff_bits,
                                period: *period,
                                no_volume: *no_volume,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ALPHATREND_ID}: expected one contiguous {:?} group at tuple \
                                 ({:?}, {period}, {no_volume})",
                                ALPHATREND_OUTPUT_IDS,
                                f64::from_bits(*coeff_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AlphaTrend {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    coeff_bits: *coeff_bits,
                    period: *period,
                    no_volume: *no_volume,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Acosc => {
                let group_len = ACOSC_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ACOSC_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                ACOSC_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ACOSC_ID
                        && candidate.output_id == ACOSC_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route == ClassicCudaResolvedRoute::Acosc
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ACOSC_ID}: expected one contiguous exact {:?} no-parameter group",
                                ACOSC_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Acosc {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AndeanOscillator {
                length,
                signal_length,
            } => {
                let group_len = ANDEAN_OSCILLATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ANDEAN_OSCILLATOR_ID}: expected one contiguous {:?} group, found \
                                 only {remaining} remaining output(s)",
                                ANDEAN_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ANDEAN_OSCILLATOR_ID
                        && candidate.output_id == ANDEAN_OSCILLATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AndeanOscillator {
                                length: *length,
                                signal_length: *signal_length,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ANDEAN_OSCILLATOR_ID}: expected one contiguous {:?} group at \
                                 tuple ({length}, {signal_length})",
                                ANDEAN_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AndeanOscillator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                    signal_length: *signal_length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Aroon { length } => {
                let group_len = AROON_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{AROON_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                AROON_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == AROON_ID
                        && candidate.output_id == AROON_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route == (ClassicCudaResolvedRoute::Aroon { length: *length })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{AROON_ID}: expected one contiguous {:?} group at length \
                                 {length}",
                                AROON_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Aroon {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Aso { period, mode } => {
                let group_len = ASO_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ASO_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                ASO_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ASO_ID
                        && candidate.output_id == ASO_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Aso {
                                period: *period,
                                mode: *mode,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ASO_ID}: expected one contiguous {:?} group at tuple \
                                 ({period}, {mode})",
                                ASO_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Aso {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    mode: *mode,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::AutocorrelationIndicator {
                length,
                lag,
                use_test_signal,
            } => {
                let group_len = AUTOCORRELATION_INDICATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{AUTOCORRELATION_INDICATOR_ID}: expected one contiguous {:?} \
                                 group, found only {remaining} remaining output(s)",
                                AUTOCORRELATION_INDICATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == AUTOCORRELATION_INDICATOR_ID
                        && candidate.output_id == AUTOCORRELATION_INDICATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::AutocorrelationIndicator {
                                length: *length,
                                lag: *lag,
                                use_test_signal: *use_test_signal,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{AUTOCORRELATION_INDICATOR_ID}: expected one contiguous {:?} \
                                 group at tuple ({length}, {lag}, {use_test_signal})",
                                AUTOCORRELATION_INDICATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::AutocorrelationIndicator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                    lag: *lag,
                    use_test_signal: *use_test_signal,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Avsl {
                fast_period,
                slow_period,
                multiplier_bits,
            } => {
                let candidate = &resolved[index];
                let valid = candidate.node.indicator_id == AVSL_ID
                    && candidate.output_id == AVSL_OUTPUT_IDS[0]
                    && candidate.route
                        == (ClassicCudaResolvedRoute::Avsl {
                            fast_period: *fast_period,
                            slow_period: *slow_period,
                            multiplier_bits: *multiplier_bits,
                        });
                if !valid {
                    push_gap(
                        &mut gaps,
                        &candidate.node,
                        ClassicCudaGapReason::NamedFamilyContractMismatch,
                        format!(
                            "{AVSL_ID}: expected one exact {:?} row at tuple ({fast_period}, \
                             {slow_period}, {:?})",
                            AVSL_OUTPUT_IDS,
                            f64::from_bits(*multiplier_bits)
                        ),
                    );
                    index += 1;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Avsl {
                    routes: [candidate.clone()],
                    fast_period: *fast_period,
                    slow_period: *slow_period,
                    multiplier_bits: *multiplier_bits,
                });
                index += 1;
            }
            ClassicCudaResolvedRoute::Bandpass {
                period,
                bandwidth_bits,
            } => {
                let group_len = BANDPASS_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BANDPASS_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                BANDPASS_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == BANDPASS_ID
                        && candidate.output_id == BANDPASS_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Bandpass {
                                period: *period,
                                bandwidth_bits: *bandwidth_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BANDPASS_ID}: expected one contiguous {:?} group at tuple \
                                 ({period}, {:?})",
                                BANDPASS_OUTPUT_IDS,
                                f64::from_bits(*bandwidth_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Bandpass {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    bandwidth_bits: *bandwidth_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::BollingerBands {
                period,
                devup_bits,
                devdn_bits,
            } => {
                let group_len = BOLLINGER_BANDS_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BOLLINGER_BANDS_ID}: expected one contiguous {:?} group, \
                                 found only {remaining} remaining output(s)",
                                BOLLINGER_BANDS_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == BOLLINGER_BANDS_ID
                        && candidate.output_id == BOLLINGER_BANDS_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::BollingerBands {
                                period: *period,
                                devup_bits: *devup_bits,
                                devdn_bits: *devdn_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BOLLINGER_BANDS_ID}: expected one contiguous {:?} group at \
                                 tuple ({period}, {:?}, {:?})",
                                BOLLINGER_BANDS_OUTPUT_IDS,
                                f64::from_bits(*devup_bits),
                                f64::from_bits(*devdn_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::BollingerBands {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    devup_bits: *devup_bits,
                    devdn_bits: *devdn_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::BuffAverages {
                fast_period,
                slow_period,
            } => {
                let group_len = BUFF_AVERAGES_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BUFF_AVERAGES_ID}: expected one contiguous {:?} group, found \
                                 only {remaining} remaining output(s)",
                                BUFF_AVERAGES_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == BUFF_AVERAGES_ID
                        && candidate.output_id == BUFF_AVERAGES_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::BuffAverages {
                                fast_period: *fast_period,
                                slow_period: *slow_period,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BUFF_AVERAGES_ID}: expected one contiguous {:?} group at tuple \
                                 ({fast_period}, {slow_period})",
                                BUFF_AVERAGES_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::BuffAverages {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    fast_period: *fast_period,
                    slow_period: *slow_period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::CandleStrengthOscillator {
                period,
                atr_enabled,
                atr_length,
                mode,
            } => {
                let group_len = CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CANDLE_STRENGTH_OSCILLATOR_ID}: expected one contiguous {:?} \
                                 group, found only {remaining} remaining output(s)",
                                CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == CANDLE_STRENGTH_OSCILLATOR_ID
                        && candidate.output_id == CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::CandleStrengthOscillator {
                                period: *period,
                                atr_enabled: *atr_enabled,
                                atr_length: *atr_length,
                                mode: *mode,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CANDLE_STRENGTH_OSCILLATOR_ID}: expected one contiguous {:?} \
                                 group at tuple ({period}, {atr_enabled}, {atr_length}, {mode:?})",
                                CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::CandleStrengthOscillator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    atr_enabled: *atr_enabled,
                    atr_length: *atr_length,
                    mode: *mode,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::ChandelierExit {
                period,
                mult_bits,
                use_close,
            } => {
                let group_len = CHANDELIER_EXIT_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CHANDELIER_EXIT_ID}: expected one contiguous {:?} group, found \
                                 only {remaining} remaining output(s)",
                                CHANDELIER_EXIT_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == CHANDELIER_EXIT_ID
                        && candidate.output_id == CHANDELIER_EXIT_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::ChandelierExit {
                                period: *period,
                                mult_bits: *mult_bits,
                                use_close: *use_close,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CHANDELIER_EXIT_ID}: expected one contiguous {:?} group at tuple \
                                 ({period}, mult_bits={mult_bits:#018x}, {use_close})",
                                CHANDELIER_EXIT_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::ChandelierExit {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    mult_bits: *mult_bits,
                    use_close: *use_close,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Cksp { p, x_bits, q } => {
                let group_len = CKSP_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CKSP_ID}: expected one contiguous {:?} default group, found only \
                                 {remaining} remaining output(s)",
                                CKSP_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == CKSP_ID
                        && candidate.output_id == CKSP_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Cksp {
                                p: *p,
                                x_bits: *x_bits,
                                q: *q,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CKSP_ID}: expected one contiguous {:?} group at exact default \
                                 tuple ({p}, x_bits={x_bits:#018x}, {q})",
                                CKSP_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Cksp {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    p: *p,
                    x_bits: *x_bits,
                    q: *q,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Coppock {
                short_roc_period,
                long_roc_period,
                ma_period,
            } => {
                let candidate = &resolved[index];
                let expected_route = ClassicCudaResolvedRoute::Coppock {
                    short_roc_period: *short_roc_period,
                    long_roc_period: *long_roc_period,
                    ma_period: *ma_period,
                };
                if candidate.node.indicator_id != COPPOCK_ID
                    || candidate.output_id != COPPOCK_OUTPUT_ID
                    || candidate.route != expected_route
                {
                    push_gap(
                        &mut gaps,
                        &candidate.node,
                        ClassicCudaGapReason::NamedFamilyContractMismatch,
                        format!(
                            "{COPPOCK_ID}: expected sole `{COPPOCK_OUTPUT_ID}` route at exact \
                             tuple ({short_roc_period}, {long_roc_period}, {ma_period})"
                        ),
                    );
                    index += 1;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Coppock {
                    route: candidate.clone(),
                    short_roc_period: *short_roc_period,
                    long_roc_period: *long_roc_period,
                    ma_period: *ma_period,
                });
                index += 1;
            }
            ClassicCudaResolvedRoute::CorrelationCycle {
                period,
                threshold_bits,
            } => {
                let group_len = CORRELATION_CYCLE_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CORRELATION_CYCLE_ID}: expected one contiguous {:?} group, \
                                 found only {remaining} remaining output(s)",
                                CORRELATION_CYCLE_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == CORRELATION_CYCLE_ID
                        && candidate.output_id == CORRELATION_CYCLE_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::CorrelationCycle {
                                period: *period,
                                threshold_bits: *threshold_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CORRELATION_CYCLE_ID}: expected one contiguous {:?} group at \
                                 tuple ({period}, {:?})",
                                CORRELATION_CYCLE_OUTPUT_IDS,
                                f64::from_bits(*threshold_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::CorrelationCycle {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    threshold_bits: *threshold_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Cvi { period } => {
                let candidate = &resolved[index];
                let expected_route = ClassicCudaResolvedRoute::Cvi { period: *period };
                if candidate.node.indicator_id != CVI_ID
                    || candidate.output_id != CVI_OUTPUT_ID
                    || candidate.route != expected_route
                {
                    push_gap(
                        &mut gaps,
                        &candidate.node,
                        ClassicCudaGapReason::NamedFamilyContractMismatch,
                        format!(
                            "{CVI_ID}: expected sole `{CVI_OUTPUT_ID}` route at exact period \
                             {period}"
                        ),
                    );
                    index += 1;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Cvi {
                    route: candidate.clone(),
                    period: *period,
                });
                index += 1;
            }
            ClassicCudaResolvedRoute::CyberpunkValueTrendAnalyzer {
                entry_level,
                exit_level,
            } => {
                let group_len = CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: expected one contiguous \
                                 {:?} group, found only {remaining} remaining output(s)",
                                CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == CYBERPUNK_VALUE_TREND_ANALYZER_ID
                        && candidate.output_id == CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::CyberpunkValueTrendAnalyzer {
                                entry_level: *entry_level,
                                exit_level: *exit_level,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: expected one contiguous \
                                 {:?} group at threshold tuple ({entry_level}, {exit_level})",
                                CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::CyberpunkValueTrendAnalyzer {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    entry_level: *entry_level,
                    exit_level: *exit_level,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::CycleChannelOscillator {
                short_cycle_length,
                medium_cycle_length,
                short_multiplier_bits,
                medium_multiplier_bits,
            } => {
                let group_len = CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CYCLE_CHANNEL_OSCILLATOR_ID}: expected one contiguous {:?} \
                                 group, found only {remaining} remaining output(s)",
                                CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == CYCLE_CHANNEL_OSCILLATOR_ID
                        && candidate.output_id == CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::CycleChannelOscillator {
                                short_cycle_length: *short_cycle_length,
                                medium_cycle_length: *medium_cycle_length,
                                short_multiplier_bits: *short_multiplier_bits,
                                medium_multiplier_bits: *medium_multiplier_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{CYCLE_CHANNEL_OSCILLATOR_ID}: expected one contiguous {:?} \
                                 group at tuple ({short_cycle_length}, {medium_cycle_length}, \
                                 {short_multiplier_bits:#018x}, {medium_multiplier_bits:#018x})",
                                CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::CycleChannelOscillator {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    short_cycle_length: *short_cycle_length,
                    medium_cycle_length: *medium_cycle_length,
                    short_multiplier_bits: *short_multiplier_bits,
                    medium_multiplier_bits: *medium_multiplier_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::DailyFactor {
                threshold_level_bits,
            } => {
                let group_len = DAILY_FACTOR_PRODUCTION_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DAILY_FACTOR_ID}: expected one contiguous {:?} group, found \
                                 only {remaining} remaining output(s)",
                                DAILY_FACTOR_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DAILY_FACTOR_ID
                        && candidate.output_id == DAILY_FACTOR_PRODUCTION_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::DailyFactor {
                                threshold_level_bits: *threshold_level_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DAILY_FACTOR_ID}: expected one contiguous {:?} group at \
                                 threshold bits {threshold_level_bits:#018x}",
                                DAILY_FACTOR_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::DailyFactor {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    threshold_level_bits: *threshold_level_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::DamianiVolatmeter {
                vis_atr,
                vis_std,
                sed_atr,
                sed_std,
                threshold_bits,
            } => {
                let group_len = DAMIANI_VOLATMETER_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DAMIANI_VOLATMETER_ID}: expected one contiguous {:?} group, \
                                 found only {remaining} remaining output(s)",
                                DAMIANI_VOLATMETER_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DAMIANI_VOLATMETER_ID
                        && candidate.output_id == DAMIANI_VOLATMETER_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::DamianiVolatmeter {
                                vis_atr: *vis_atr,
                                vis_std: *vis_std,
                                sed_atr: *sed_atr,
                                sed_std: *sed_std,
                                threshold_bits: *threshold_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DAMIANI_VOLATMETER_ID}: expected one contiguous {:?} group at \
                                 tuple ({vis_atr}, {vis_std}, {sed_atr}, {sed_std}, \
                                 {threshold_bits:#018x})",
                                DAMIANI_VOLATMETER_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::DamianiVolatmeter {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    vis_atr: *vis_atr,
                    vis_std: *vis_std,
                    sed_atr: *sed_atr,
                    sed_std: *sed_std,
                    threshold_bits: *threshold_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Di { period } => {
                let group_len = DI_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DI_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                DI_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DI_ID
                        && candidate.output_id == DI_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route == (ClassicCudaResolvedRoute::Di { period: *period })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DI_ID}: expected one contiguous {:?} group at period {period}",
                                DI_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Di {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::DidiIndex {
                short_length,
                medium_length,
                long_length,
            } => {
                let group_len = DIDI_INDEX_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DIDI_INDEX_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                DIDI_INDEX_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DIDI_INDEX_ID
                        && candidate.output_id == DIDI_INDEX_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::DidiIndex {
                                short_length: *short_length,
                                medium_length: *medium_length,
                                long_length: *long_length,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DIDI_INDEX_ID}: expected one contiguous {:?} group at tuple \
                                 ({short_length}, {medium_length}, {long_length})",
                                DIDI_INDEX_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::DidiIndex {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    short_length: *short_length,
                    medium_length: *medium_length,
                    long_length: *long_length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::DirectionalImbalanceIndex { length, period } => {
                let group_len = DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DIRECTIONAL_IMBALANCE_INDEX_ID}: expected one contiguous {:?} \
                                 group, found only {remaining} remaining output(s)",
                                DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DIRECTIONAL_IMBALANCE_INDEX_ID
                        && candidate.output_id == DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::DirectionalImbalanceIndex {
                                length: *length,
                                period: *period,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DIRECTIONAL_IMBALANCE_INDEX_ID}: expected one contiguous {:?} \
                                 group at tuple ({length}, {period})",
                                DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::DirectionalImbalanceIndex {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                    period: *period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::DisparityIndex {
                ema_period,
                lookback_period,
                smoothing_period,
                smoothing_is_sma,
            } => {
                let candidate = &resolved[index];
                let valid = candidate.node.indicator_id == DISPARITY_INDEX_ID
                    && candidate.output_id == DISPARITY_INDEX_OUTPUT_ID
                    && candidate.route
                        == (ClassicCudaResolvedRoute::DisparityIndex {
                            ema_period: *ema_period,
                            lookback_period: *lookback_period,
                            smoothing_period: *smoothing_period,
                            smoothing_is_sma: *smoothing_is_sma,
                        });
                if !valid {
                    push_gap(
                        &mut gaps,
                        &candidate.node,
                        ClassicCudaGapReason::NamedFamilyContractMismatch,
                        format!(
                            "{DISPARITY_INDEX_ID}: expected sole `{DISPARITY_INDEX_OUTPUT_ID}` \
                             route at tuple ({ema_period}, {lookback_period}, \
                             {smoothing_period}, smoothing_is_sma={smoothing_is_sma})"
                        ),
                    );
                    index += 1;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::DisparityIndex {
                    route: candidate.clone(),
                    ema_period: *ema_period,
                    lookback_period: *lookback_period,
                    smoothing_period: *smoothing_period,
                    smoothing_is_sma: *smoothing_is_sma,
                });
                index += 1;
            }
            ClassicCudaResolvedRoute::Dm { period } => {
                let group_len = DM_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DM_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                DM_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DM_ID
                        && candidate.output_id == DM_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route == (ClassicCudaResolvedRoute::Dm { period: *period })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DM_ID}: expected one contiguous {:?} group at period {period}",
                                DM_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Dm {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Donchian { period } => {
                let group_len = DONCHIAN_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DONCHIAN_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                DONCHIAN_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DONCHIAN_ID
                        && candidate.output_id == DONCHIAN_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Donchian { period: *period })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DONCHIAN_ID}: expected one contiguous {:?} group at period \
                                 {period}",
                                DONCHIAN_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Donchian {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::DualUlcerIndex {
                period,
                auto_threshold,
                threshold_bits,
            } => {
                let group_len = DUAL_ULCER_INDEX_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DUAL_ULCER_INDEX_ID}: expected one contiguous {:?} group, \
                                 found only {remaining} remaining output(s)",
                                DUAL_ULCER_INDEX_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DUAL_ULCER_INDEX_ID
                        && candidate.output_id == DUAL_ULCER_INDEX_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::DualUlcerIndex {
                                period: *period,
                                auto_threshold: *auto_threshold,
                                threshold_bits: *threshold_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DUAL_ULCER_INDEX_ID}: expected one contiguous {:?} group at \
                                 tuple ({period}, auto_threshold={auto_threshold}, \
                                 threshold_bits={threshold_bits:#018x})",
                                DUAL_ULCER_INDEX_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::DualUlcerIndex {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    auto_threshold: *auto_threshold,
                    threshold_bits: *threshold_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Dvdiqqe {
                period,
                smoothing_period,
                fast_multiplier_bits,
                slow_multiplier_bits,
                use_tick_only,
                dynamic_center,
                tick_size_bits,
            } => {
                let group_len = DVDIQQE_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DVDIQQE_ID}: expected one contiguous {:?} group, found only \
                                 {remaining} remaining output(s)",
                                DVDIQQE_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == DVDIQQE_ID
                        && candidate.output_id == DVDIQQE_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Dvdiqqe {
                                period: *period,
                                smoothing_period: *smoothing_period,
                                fast_multiplier_bits: *fast_multiplier_bits,
                                slow_multiplier_bits: *slow_multiplier_bits,
                                use_tick_only: *use_tick_only,
                                dynamic_center: *dynamic_center,
                                tick_size_bits: *tick_size_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{DVDIQQE_ID}: expected one contiguous {:?} group at tuple \
                                 ({period}, {smoothing_period}, fast_bits=\
                                 {fast_multiplier_bits:#018x}, slow_bits=\
                                 {slow_multiplier_bits:#018x}, use_tick_only={use_tick_only}, \
                                 dynamic_center={dynamic_center}, tick_bits={tick_size_bits:#018x})",
                                DVDIQQE_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::Dvdiqqe {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    smoothing_period: *smoothing_period,
                    fast_multiplier_bits: *fast_multiplier_bits,
                    slow_multiplier_bits: *slow_multiplier_bits,
                    use_tick_only: *use_tick_only,
                    dynamic_center: *dynamic_center,
                    tick_size_bits: *tick_size_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::EhlersAutocorrelationPeriodogram {
                min_period,
                max_period,
                avg_length,
                enhance,
            } => {
                let group_len = EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: expected one \
                                 contiguous {:?} group, found only {remaining} remaining output(s)",
                                EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EHLERS_AUTOCORRELATION_PERIODOGRAM_ID
                        && candidate.output_id
                            == EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EhlersAutocorrelationPeriodogram {
                                min_period: *min_period,
                                max_period: *max_period,
                                avg_length: *avg_length,
                                enhance: *enhance,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: expected one \
                                 contiguous {:?} group at tuple ({min_period}, {max_period}, \
                                 {avg_length}, enhance={enhance})",
                                EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(
                    ResolvedClassicCudaLaunch::EhlersAutocorrelationPeriodogram {
                        routes: std::array::from_fn(|offset| candidates[offset].clone()),
                        min_period: *min_period,
                        max_period: *max_period,
                        avg_length: *avg_length,
                        enhance: *enhance,
                    },
                );
                index += group_len;
            }
            ClassicCudaResolvedRoute::EhlersLinearExtrapolationPredictor {
                high_pass_length,
                low_pass_length,
                gain_bits,
                bars_forward,
                signal_mode,
            } => {
                let group_len = EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID
                        && candidate.output_id
                            == EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EhlersLinearExtrapolationPredictor {
                                high_pass_length: *high_pass_length,
                                low_pass_length: *low_pass_length,
                                gain_bits: *gain_bits,
                                bars_forward: *bars_forward,
                                signal_mode: *signal_mode,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: expected one contiguous {:?} group at tuple ({high_pass_length}, {low_pass_length}, gain_bits={gain_bits:#018x}, bars_forward={bars_forward}, signal_mode={signal_mode})",
                                EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(
                    ResolvedClassicCudaLaunch::EhlersLinearExtrapolationPredictor {
                        routes: std::array::from_fn(|offset| candidates[offset].clone()),
                        high_pass_length: *high_pass_length,
                        low_pass_length: *low_pass_length,
                        gain_bits: *gain_bits,
                        bars_forward: *bars_forward,
                        signal_mode: *signal_mode,
                    },
                );
                index += group_len;
            }
            ClassicCudaResolvedRoute::EhlersUndersampledDoubleMovingAverage {
                fast_length,
                slow_length,
                sample_length,
            } => {
                let group_len = EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID
                        && candidate.output_id
                            == EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EhlersUndersampledDoubleMovingAverage {
                                fast_length: *fast_length,
                                slow_length: *slow_length,
                                sample_length: *sample_length,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: expected one contiguous {:?} group at tuple ({fast_length}, {slow_length}, {sample_length})",
                                EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(
                    ResolvedClassicCudaLaunch::EhlersUndersampledDoubleMovingAverage {
                        routes: std::array::from_fn(|offset| candidates[offset].clone()),
                        fast_length: *fast_length,
                        slow_length: *slow_length,
                        sample_length: *sample_length,
                    },
                );
                index += group_len;
            }
            ClassicCudaResolvedRoute::EmaDeviationCorrectedT3 {
                period,
                hot_bits,
                t3_mode,
            } => {
                let group_len = EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EMA_DEVIATION_CORRECTED_T3_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EMA_DEVIATION_CORRECTED_T3_ID
                        && candidate.output_id == EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EmaDeviationCorrectedT3 {
                                period: *period,
                                hot_bits: *hot_bits,
                                t3_mode: *t3_mode,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EMA_DEVIATION_CORRECTED_T3_ID}: expected one contiguous {:?} group at tuple ({period}, hot_bits={hot_bits:#018x}, t3_mode={t3_mode})",
                                EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::EmaDeviationCorrectedT3 {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    hot_bits: *hot_bits,
                    t3_mode: *t3_mode,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Emd {
                period,
                delta_bits,
                fraction_bits,
            } => {
                let group_len = EMD_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EMD_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                EMD_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EMD_ID
                        && candidate.output_id == EMD_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Emd {
                                period: *period,
                                delta_bits: *delta_bits,
                                fraction_bits: *fraction_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EMD_ID}: expected one contiguous {:?} group at tuple ({period}, delta_bits={delta_bits:#018x}, fraction_bits={fraction_bits:#018x})",
                                EMD_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Emd {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    delta_bits: *delta_bits,
                    fraction_bits: *fraction_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::EmdTrend { length, mult_bits } => {
                let group_len = EMD_TREND_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EMD_TREND_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                EMD_TREND_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EMD_TREND_ID
                        && candidate.output_id == EMD_TREND_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EmdTrend {
                                length: *length,
                                mult_bits: *mult_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EMD_TREND_ID}: expected one contiguous {:?} group at tuple ({length}, mult_bits={mult_bits:#018x})",
                                EMD_TREND_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::EmdTrend {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                    mult_bits: *mult_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Eri { period } => {
                let group_len = ERI_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ERI_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                ERI_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == ERI_ID
                        && candidate.output_id == ERI_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route == (ClassicCudaResolvedRoute::Eri { period: *period })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{ERI_ID}: expected one contiguous {:?} group at period {period}",
                                ERI_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Eri {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::EvasiveSupertrend {
                atr_length,
                base_multiplier_bits,
                noise_threshold_bits,
                expansion_alpha_bits,
            } => {
                let group_len = EVASIVE_SUPERTREND_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EVASIVE_SUPERTREND_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                EVASIVE_SUPERTREND_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EVASIVE_SUPERTREND_ID
                        && candidate.output_id == EVASIVE_SUPERTREND_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EvasiveSupertrend {
                                atr_length: *atr_length,
                                base_multiplier_bits: *base_multiplier_bits,
                                noise_threshold_bits: *noise_threshold_bits,
                                expansion_alpha_bits: *expansion_alpha_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EVASIVE_SUPERTREND_ID}: expected one contiguous {:?} group at tuple ({atr_length}, base_multiplier_bits={base_multiplier_bits:#018x}, noise_threshold_bits={noise_threshold_bits:#018x}, expansion_alpha_bits={expansion_alpha_bits:#018x})",
                                EVASIVE_SUPERTREND_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::EvasiveSupertrend {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    atr_length: *atr_length,
                    base_multiplier_bits: *base_multiplier_bits,
                    noise_threshold_bits: *noise_threshold_bits,
                    expansion_alpha_bits: *expansion_alpha_bits,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::FibonacciTrailingStop {
                left_bars,
                right_bars,
                level_bits,
                trigger_mode,
            } => {
                let group_len = FIBONACCI_TRAILING_STOP_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FIBONACCI_TRAILING_STOP_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                FIBONACCI_TRAILING_STOP_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == FIBONACCI_TRAILING_STOP_ID
                        && candidate.output_id == FIBONACCI_TRAILING_STOP_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::FibonacciTrailingStop {
                                left_bars: *left_bars,
                                right_bars: *right_bars,
                                level_bits: *level_bits,
                                trigger_mode: *trigger_mode,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FIBONACCI_TRAILING_STOP_ID}: expected one contiguous {:?} group at tuple ({left_bars}, {right_bars}, level_bits={level_bits:#018x}, trigger_mode={trigger_mode})",
                                FIBONACCI_TRAILING_STOP_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::FibonacciTrailingStop {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    left_bars: *left_bars,
                    right_bars: *right_bars,
                    level_bits: *level_bits,
                    trigger_mode: *trigger_mode,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Fisher { period } => {
                let group_len = FISHER_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FISHER_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                FISHER_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == FISHER_ID
                        && candidate.output_id == FISHER_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route == (ClassicCudaResolvedRoute::Fisher { period: *period })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FISHER_ID}: expected one contiguous {:?} group at period {period}",
                                FISHER_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Fisher {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::ForwardBackwardExponentialOscillator { length, smooth } => {
                let group_len = FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID
                        && candidate.output_id
                            == FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::ForwardBackwardExponentialOscillator {
                                length: *length,
                                smooth: *smooth,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: expected one contiguous {:?} group at length {length}, smooth {smooth}",
                                FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(
                    ResolvedClassicCudaLaunch::ForwardBackwardExponentialOscillator {
                        routes: std::array::from_fn(|offset| candidates[offset].clone()),
                        length: *length,
                        smooth: *smooth,
                    },
                );
                index += group_len;
            }
            ClassicCudaResolvedRoute::FvgTrailingStop {
                unmitigated_fvg_lookback,
                smoothing_length,
                reset_on_cross,
            } => {
                let group_len = FVG_TRAILING_STOP_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FVG_TRAILING_STOP_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                FVG_TRAILING_STOP_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == FVG_TRAILING_STOP_ID
                        && candidate.output_id == FVG_TRAILING_STOP_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::FvgTrailingStop {
                                unmitigated_fvg_lookback: *unmitigated_fvg_lookback,
                                smoothing_length: *smoothing_length,
                                reset_on_cross: *reset_on_cross,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FVG_TRAILING_STOP_ID}: expected one contiguous {:?} group at lookback {unmitigated_fvg_lookback}, smoothing {smoothing_length}, reset {reset_on_cross}",
                                FVG_TRAILING_STOP_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::FvgTrailingStop {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    unmitigated_fvg_lookback: *unmitigated_fvg_lookback,
                    smoothing_length: *smoothing_length,
                    reset_on_cross: *reset_on_cross,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Gatorosc {
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
            } => {
                let group_len = GATOROSC_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{GATOROSC_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                GATOROSC_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == GATOROSC_ID
                        && candidate.output_id == GATOROSC_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Gatorosc {
                                jaws_length: *jaws_length,
                                jaws_shift: *jaws_shift,
                                teeth_length: *teeth_length,
                                teeth_shift: *teeth_shift,
                                lips_length: *lips_length,
                                lips_shift: *lips_shift,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{GATOROSC_ID}: expected one contiguous {:?} group at tuple [{jaws_length},{jaws_shift},{teeth_length},{teeth_shift},{lips_length},{lips_shift}]",
                                GATOROSC_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Gatorosc {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    jaws_length: *jaws_length,
                    jaws_shift: *jaws_shift,
                    teeth_length: *teeth_length,
                    teeth_shift: *teeth_shift,
                    lips_length: *lips_length,
                    lips_shift: *lips_shift,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::Halftrend {
                amplitude,
                channel_deviation_bits,
                atr_period,
            } => {
                let group_len = HALFTREND_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{HALFTREND_ID}: expected one contiguous {:?} group, found only {remaining} remaining output(s)",
                                HALFTREND_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == HALFTREND_ID
                        && candidate.output_id == HALFTREND_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::Halftrend {
                                amplitude: *amplitude,
                                channel_deviation_bits: *channel_deviation_bits,
                                atr_period: *atr_period,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{HALFTREND_ID}: expected one contiguous {:?} group at amplitude {amplitude}, channel_deviation_bits {channel_deviation_bits}, atr_period {atr_period}",
                                HALFTREND_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::Halftrend {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    amplitude: *amplitude,
                    channel_deviation_bits: *channel_deviation_bits,
                    atr_period: *atr_period,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::FibonacciEntryBands { length } => {
                let group_len = FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FIBONACCI_ENTRY_BANDS_ID}: expected one contiguous admitted {:?} group, found only {remaining} remaining output(s)",
                                FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == FIBONACCI_ENTRY_BANDS_ID
                        && candidate.output_id
                            == FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::FibonacciEntryBands { length: *length })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{FIBONACCI_ENTRY_BANDS_ID}: expected one contiguous admitted {:?} group at length {length}",
                                FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::FibonacciEntryBands {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::EhlersDataSamplingRsi { length } => {
                let group_len = EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_DATA_SAMPLING_RSI_ID}: expected one contiguous admitted {:?} group, found only {remaining} remaining output(s)",
                                EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }
                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == EHLERS_DATA_SAMPLING_RSI_ID
                        && candidate.output_id
                            == EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::EhlersDataSamplingRsi { length: *length })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{EHLERS_DATA_SAMPLING_RSI_ID}: expected one contiguous admitted {:?} group at length {length}",
                                EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }
                launches.push(ResolvedClassicCudaLaunch::EhlersDataSamplingRsi {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    length: *length,
                });
                index += group_len;
            }
            ClassicCudaResolvedRoute::BullsVBears {
                period,
                ma_type,
                calculation_method,
                normalized_bars_back,
                raw_rolling_period,
                raw_threshold_percentile_bits,
                threshold_level_bits,
            } => {
                let group_len = BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS.len();
                let remaining = resolved.len() - index;
                if remaining < group_len {
                    for candidate in &resolved[index..] {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BULLS_V_BEARS_ID}: expected one contiguous admitted {:?} \
                                 group, found only {remaining} remaining output(s)",
                                BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS
                            ),
                        );
                    }
                    break;
                }

                let candidates = &resolved[index..index + group_len];
                let first = &candidates[0];
                let valid = candidates.iter().enumerate().all(|(offset, candidate)| {
                    candidate.node.indicator_id == BULLS_V_BEARS_ID
                        && candidate.output_id == BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS[offset]
                        && candidate.node.stage == first.node.stage
                        && candidate.node.parameters == first.node.parameters
                        && candidate.route
                            == (ClassicCudaResolvedRoute::BullsVBears {
                                period: *period,
                                ma_type: *ma_type,
                                calculation_method: *calculation_method,
                                normalized_bars_back: *normalized_bars_back,
                                raw_rolling_period: *raw_rolling_period,
                                raw_threshold_percentile_bits: *raw_threshold_percentile_bits,
                                threshold_level_bits: *threshold_level_bits,
                            })
                });
                if !valid {
                    for candidate in candidates {
                        push_gap(
                            &mut gaps,
                            &candidate.node,
                            ClassicCudaGapReason::NamedFamilyContractMismatch,
                            format!(
                                "{BULLS_V_BEARS_ID}: expected one contiguous admitted {:?} group \
                                 at tuple ({period}, {ma_type:?}, {calculation_method:?}, \
                                 {normalized_bars_back}, {raw_rolling_period}, {}, {})",
                                BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS,
                                f64::from_bits(*raw_threshold_percentile_bits),
                                f64::from_bits(*threshold_level_bits)
                            ),
                        );
                    }
                    index += group_len;
                    continue;
                }

                launches.push(ResolvedClassicCudaLaunch::BullsVBears {
                    routes: std::array::from_fn(|offset| candidates[offset].clone()),
                    period: *period,
                    ma_type: *ma_type,
                    calculation_method: *calculation_method,
                    normalized_bars_back: *normalized_bars_back,
                    raw_rolling_period: *raw_rolling_period,
                    raw_threshold_percentile_bits: *raw_threshold_percentile_bits,
                    threshold_level_bits: *threshold_level_bits,
                });
                index += group_len;
            }
        }
    }

    if gaps.is_empty() {
        Ok(launches)
    } else {
        Err(gaps)
    }
}

fn gap_manifest(gaps: &[ClassicCudaGap]) -> String {
    gaps.iter()
        .map(|gap| {
            format!(
                "{} [{} output={:?} params={:?}]:{}:{}",
                gap.column_name,
                gap.indicator_id,
                gap.requested_output_id,
                gap.parameters,
                gap.reason,
                gap.detail
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Resolve the complete admitted graph without creating a CUDA context or
/// allocating a feature column.  The same resolved-plan contract is consumed
/// by the run-level preflight and immediately before execution, so the cheap
/// preflight cannot describe a different routing policy from the launch path.
pub(crate) fn resolve_gpu_only_classic_plan(
    plan: &ClassicCudaPlan,
) -> Result<Vec<ResolvedClassicCudaLaunch>> {
    let resolved = preflight_exact_classic_cuda_plan(plan).map_err(|gaps| {
        anyhow::anyhow!(
            "GpuOnly exact-plan preflight found {} unrouteable admitted output contract(s) \
             before the first CUDA context/launch. No output was excluded and no CPU/f32 \
             substitute is permitted. Complete ordered manifest: {}",
            gaps.len(),
            gap_manifest(&gaps)
        )
    })?;
    let resolved_output_count = resolved
        .iter()
        .map(ResolvedClassicCudaLaunch::output_count)
        .sum::<usize>();
    ensure!(
        resolved_output_count == plan.nodes.len(),
        "GpuOnly preflight resolved {resolved_output_count} of {} outputs without returning its \
         missing manifest",
        plan.nodes.len()
    );
    Ok(resolved)
}

fn swept_point_fits_frame(node: &ClassicCudaNode, rows: usize) -> bool {
    node.parameters
        .swept_period()
        .is_none_or(|period| (period as f64) * 1.25 < rows as f64)
}

enum PendingClassicColumn {
    PrimaryResident {
        route: ResolvedClassicCudaNode,
        output: IndicatorCudaOutputF64,
    },
    NamedResident {
        routes: Vec<ResolvedClassicCudaNode>,
        output: F64NamedOutputsResult,
    },
    WarmupPlaceholder(ResolvedClassicCudaNode),
}

/// Execute a fully preflighted typed output plan through one resident engine.
/// Unsupported named/discrete gaps never reach this function's launch loop.
pub(crate) fn execute_gpu_only_classic_plan(
    ohlcv: &Ohlcv,
    plan: ClassicCudaPlan,
    admission: ClassicTaAdmissionPlan,
) -> Result<ClassicTaComputation> {
    let resolved = resolve_gpu_only_classic_plan(&plan)?;

    let engine = GpuIndicatorEngine::new(ohlcv, 0)?;
    let mut pending = Vec::with_capacity(resolved.len());
    for launch in resolved {
        match launch {
            ResolvedClassicCudaLaunch::Primary(route) => {
                if swept_point_fits_frame(&route.node, plan.rows) {
                    let cuda_period = match &route.route {
                        ClassicCudaResolvedRoute::Primary { cuda_period } => *cuda_period,
                        ClassicCudaResolvedRoute::AbsoluteStrengthIndexOscillator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AdaptiveBandpassTriggerOscillator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AdaptiveBoundsRsi { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AdaptiveMacd { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AdaptiveMomentumOscillator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AdaptiveSchaffTrendCycle { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AdjustableMaAlternatingExtremities { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Alligator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AlphaTrend { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Acosc => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AndeanOscillator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Aroon { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Aso { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::AutocorrelationIndicator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Avsl { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Bandpass { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::BollingerBands { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::BuffAverages { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::CandleStrengthOscillator { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::ChandelierExit { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Cksp { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Coppock { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::CorrelationCycle { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Cvi { .. } => {
                            unreachable!("preflight placed a typed CVI route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::CyberpunkValueTrendAnalyzer { .. } => {
                            unreachable!(
                                "preflight placed a typed Cyberpunk Value Trend Analyzer route \
                                 in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::CycleChannelOscillator { .. } => {
                            unreachable!(
                                "preflight placed a typed Cycle Channel Oscillator route in a \
                                 primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::DailyFactor { .. } => {
                            unreachable!(
                                "preflight placed a typed Daily Factor route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::DamianiVolatmeter { .. } => {
                            unreachable!(
                                "preflight placed a typed Damiani Volatmeter route in a primary \
                                 launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Di { .. } => {
                            unreachable!("preflight placed a typed DI route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::DidiIndex { .. } => {
                            unreachable!(
                                "preflight placed a typed Didi Index route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::DirectionalImbalanceIndex { .. } => {
                            unreachable!(
                                "preflight placed a typed Directional Imbalance Index route in \
                                 a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::DisparityIndex { .. } => {
                            unreachable!(
                                "preflight placed a typed Disparity Index route in a primary \
                                 launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Dm { .. } => {
                            unreachable!("preflight placed a typed DM route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::Donchian { .. } => {
                            unreachable!(
                                "preflight placed a typed Donchian route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::DualUlcerIndex { .. } => {
                            unreachable!(
                                "preflight placed a typed Dual Ulcer Index route in a primary \
                                 launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Dvdiqqe { .. } => {
                            unreachable!(
                                "preflight placed a typed DVDIQQE route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::EhlersAutocorrelationPeriodogram { .. } => {
                            unreachable!(
                                "preflight placed a typed Ehlers Autocorrelation Periodogram \
                                 route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::EhlersLinearExtrapolationPredictor { .. } => {
                            unreachable!(
                                "preflight placed a typed Ehlers Linear Extrapolation Predictor route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::EhlersUndersampledDoubleMovingAverage {
                            ..
                        } => {
                            unreachable!(
                                "preflight placed a typed Ehlers Undersampled Double Moving Average route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::EmaDeviationCorrectedT3 { .. } => {
                            unreachable!(
                                "preflight placed a typed EMA-Deviation-Corrected T3 route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Emd { .. } => {
                            unreachable!("preflight placed a typed EMD route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::EmdTrend { .. } => {
                            unreachable!(
                                "preflight placed a typed EMD Trend route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Eri { .. } => {
                            unreachable!("preflight placed a typed ERI route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::EvasiveSupertrend { .. } => {
                            unreachable!(
                                "preflight placed a typed Evasive Supertrend route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::FibonacciTrailingStop { .. } => {
                            unreachable!(
                                "preflight placed a typed Fibonacci Trailing Stop route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Fisher { .. } => {
                            unreachable!(
                                "preflight placed a typed Fisher route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::ForwardBackwardExponentialOscillator {
                            ..
                        } => {
                            unreachable!("preflight placed a typed FBEO route in a primary launch")
                        }
                        ClassicCudaResolvedRoute::FvgTrailingStop { .. } => {
                            unreachable!(
                                "preflight placed a typed FVG Trailing Stop route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Gatorosc { .. } => {
                            unreachable!(
                                "preflight placed a typed Gator Oscillator route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::Halftrend { .. } => {
                            unreachable!(
                                "preflight placed a typed HalfTrend route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::FibonacciEntryBands { .. } => {
                            unreachable!(
                                "preflight placed a typed Fibonacci Entry Bands route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::EhlersDataSamplingRsi { .. } => {
                            unreachable!(
                                "preflight placed a typed Ehlers Data Sampling RSI route in a primary launch"
                            )
                        }
                        ClassicCudaResolvedRoute::BullsVBears { .. } => {
                            unreachable!("preflight placed a named route in a primary launch")
                        }
                    };
                    let output =
                        engine.compute_primary_device(route.node.indicator_id, &[cuda_period])?;
                    ensure!(
                        output.output_id == route.output_id,
                        "{}: planned output `{}` but CUDA returned `{}`",
                        route.node.column_name,
                        route.output_id,
                        output.output_id
                    );
                    pending.push(PendingClassicColumn::PrimaryResident { route, output });
                } else {
                    pending.push(PendingClassicColumn::WarmupPlaceholder(route));
                }
            }
            ResolvedClassicCudaLaunch::AbsoluteStrengthIndexOscillator {
                routes,
                ema_length,
                signal_length,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let output = engine.compute_absolute_strength_index_oscillator_outputs_device(
                        &[(ema_length, signal_length)],
                    )?;
                    ensure!(
                        output.indicator_id == ASI_ID,
                        "planned `{ASI_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    ensure!(
                        output.outputs.len() == routes.len(),
                        "{ASI_ID}: planned {} named outputs but CUDA returned {}",
                        routes.len(),
                        output.outputs.len()
                    );
                    for (route, named_output) in routes.iter().zip(&output.outputs) {
                        ensure!(
                            route.output_id == named_output.output_id,
                            "{}: planned output `{}` but CUDA returned `{}`",
                            route.node.column_name,
                            route.output_id,
                            named_output.output_id
                        );
                    }
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AdaptiveBandpassTriggerOscillator {
                routes,
                delta_bits,
                alpha_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let output = engine
                        .compute_adaptive_bandpass_trigger_oscillator_outputs_device(&[(
                            f64::from_bits(delta_bits),
                            f64::from_bits(alpha_bits),
                        )])?;
                    ensure!(
                        output.indicator_id == ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID,
                        "planned `{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS,
                        "{ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID}: full CUDA output contract \
                         {:?} != {:?}",
                        returned_output_ids,
                        ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AdaptiveBoundsRsi {
                routes,
                rsi_length,
                alpha_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let parameter_rows = [AdaptiveBoundsRsiParams {
                        rsi_length: Some(rsi_length),
                        alpha: Some(f64::from_bits(alpha_bits)),
                    }];
                    let output =
                        engine.compute_adaptive_bounds_rsi_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == ADAPTIVE_BOUNDS_RSI_ID,
                        "planned `{ADAPTIVE_BOUNDS_RSI_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ADAPTIVE_BOUNDS_RSI_KERNEL_OUTPUT_IDS,
                        "{ADAPTIVE_BOUNDS_RSI_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ADAPTIVE_BOUNDS_RSI_KERNEL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AdaptiveMacd {
                routes,
                length,
                fast_period,
                slow_period,
                signal_period,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let output = engine.compute_adaptive_macd_outputs_device(&[(
                        length,
                        fast_period,
                        slow_period,
                        signal_period,
                    )])?;
                    ensure!(
                        output.indicator_id == ADAPTIVE_MACD_ID,
                        "planned `{ADAPTIVE_MACD_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ADAPTIVE_MACD_OUTPUT_IDS,
                        "{ADAPTIVE_MACD_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ADAPTIVE_MACD_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AdaptiveMomentumOscillator {
                routes,
                length,
                smoothing_length,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let output =
                        engine.compute_adaptive_momentum_oscillator_outputs_device(&[(
                            length,
                            smoothing_length,
                        )])?;
                    ensure!(
                        output.indicator_id == ADAPTIVE_MOMENTUM_OSCILLATOR_ID,
                        "planned `{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS,
                        "{ADAPTIVE_MOMENTUM_OSCILLATOR_ID}: full CUDA output contract {:?} != \
                         {:?}",
                        returned_output_ids,
                        ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AdaptiveSchaffTrendCycle {
                routes,
                adaptive_length,
                stc_length,
                smoothing_factor_bits,
                fast_length,
                slow_length,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let parameter_rows = [AdaptiveSchaffTrendCycleParams {
                        adaptive_length: Some(adaptive_length),
                        stc_length: Some(stc_length),
                        smoothing_factor: Some(f64::from_bits(smoothing_factor_bits)),
                        fast_length: Some(fast_length),
                        slow_length: Some(slow_length),
                    }];
                    let output = engine
                        .compute_adaptive_schaff_trend_cycle_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == ADAPTIVE_SCHAFF_TREND_CYCLE_ID,
                        "planned `{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS,
                        "{ADAPTIVE_SCHAFF_TREND_CYCLE_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AdjustableMaAlternatingExtremities {
                routes,
                length,
                mult_bits,
                alpha_bits,
                beta_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS,
                        "{ADJUSTABLE_MA_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS
                    );
                    let parameter_rows = [AdjustableMaAlternatingExtremitiesParams {
                        length: Some(length),
                        mult: Some(f64::from_bits(mult_bits)),
                        alpha: Some(f64::from_bits(alpha_bits)),
                        beta: Some(f64::from_bits(beta_bits)),
                    }];
                    let output = engine
                        .compute_adjustable_ma_alternating_extremities_outputs_device(
                            &parameter_rows,
                        )?;
                    ensure!(
                        output.indicator_id == ADJUSTABLE_MA_ID,
                        "planned `{ADJUSTABLE_MA_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ADJUSTABLE_MA_FULL_OUTPUT_IDS,
                        "{ADJUSTABLE_MA_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ADJUSTABLE_MA_FULL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Alligator {
                routes,
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == ALLIGATOR_OUTPUT_IDS,
                        "{ALLIGATOR_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        ALLIGATOR_OUTPUT_IDS
                    );
                    let parameter_rows = [AlligatorParams {
                        jaw_period: Some(jaw_period),
                        jaw_offset: Some(jaw_offset),
                        teeth_period: Some(teeth_period),
                        teeth_offset: Some(teeth_offset),
                        lips_period: Some(lips_period),
                        lips_offset: Some(lips_offset),
                    }];
                    let output = engine.compute_alligator_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == ALLIGATOR_ID,
                        "planned `{ALLIGATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ALLIGATOR_OUTPUT_IDS,
                        "{ALLIGATOR_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ALLIGATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AlphaTrend {
                routes,
                coeff_bits,
                period,
                no_volume,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == ALPHATREND_OUTPUT_IDS,
                        "{ALPHATREND_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        ALPHATREND_OUTPUT_IDS
                    );
                    let parameter_rows = [AlphaTrendParams {
                        coeff: Some(f64::from_bits(coeff_bits)),
                        period: Some(period),
                        no_volume: Some(no_volume),
                    }];
                    let output = engine.compute_alphatrend_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == ALPHATREND_ID,
                        "planned `{ALPHATREND_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ALPHATREND_OUTPUT_IDS,
                        "{ALPHATREND_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ALPHATREND_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Acosc { routes } => {
                let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                ensure!(
                    planned_output_ids == ACOSC_OUTPUT_IDS,
                    "{ACOSC_ID}: typed admitted output contract {:?} != {:?}",
                    planned_output_ids,
                    ACOSC_OUTPUT_IDS
                );
                let output = engine.compute_acosc_outputs_device()?;
                ensure!(
                    output.indicator_id == ACOSC_ID,
                    "planned `{ACOSC_ID}` but CUDA returned `{}`",
                    output.indicator_id
                );
                let returned_output_ids = output
                    .outputs
                    .iter()
                    .map(|named_output| named_output.output_id)
                    .collect::<Vec<_>>();
                ensure!(
                    returned_output_ids == ACOSC_OUTPUT_IDS,
                    "{ACOSC_ID}: full CUDA output contract {:?} != {:?}",
                    returned_output_ids,
                    ACOSC_OUTPUT_IDS
                );
                pending.push(PendingClassicColumn::NamedResident {
                    routes: routes.into(),
                    output,
                });
            }
            ResolvedClassicCudaLaunch::AndeanOscillator {
                routes,
                length,
                signal_length,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == ANDEAN_OSCILLATOR_OUTPUT_IDS,
                        "{ANDEAN_OSCILLATOR_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        ANDEAN_OSCILLATOR_OUTPUT_IDS
                    );
                    let parameter_tuples = [(length, signal_length)];
                    let output =
                        engine.compute_andean_oscillator_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == ANDEAN_OSCILLATOR_ID,
                        "planned `{ANDEAN_OSCILLATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ANDEAN_OSCILLATOR_OUTPUT_IDS,
                        "{ANDEAN_OSCILLATOR_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ANDEAN_OSCILLATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Aroon { routes, length } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == AROON_OUTPUT_IDS,
                        "{AROON_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        AROON_OUTPUT_IDS
                    );
                    let output = engine.compute_aroon_outputs_device(&[length])?;
                    ensure!(
                        output.indicator_id == AROON_ID,
                        "planned `{AROON_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == AROON_OUTPUT_IDS,
                        "{AROON_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        AROON_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Aso {
                routes,
                period,
                mode,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == ASO_OUTPUT_IDS,
                        "{ASO_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        ASO_OUTPUT_IDS
                    );
                    let output = engine.compute_aso_outputs_device(&[(period, mode)])?;
                    ensure!(
                        output.indicator_id == ASO_ID,
                        "planned `{ASO_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ASO_OUTPUT_IDS,
                        "{ASO_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        ASO_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::AutocorrelationIndicator {
                routes,
                length,
                lag,
                use_test_signal,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == AUTOCORRELATION_INDICATOR_OUTPUT_IDS,
                        "{AUTOCORRELATION_INDICATOR_ID}: typed admitted output contract {:?} != \
                         {:?}",
                        planned_output_ids,
                        AUTOCORRELATION_INDICATOR_OUTPUT_IDS
                    );
                    let parameter_tuples = [(length, lag, use_test_signal)];
                    let output = engine
                        .compute_autocorrelation_indicator_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == AUTOCORRELATION_INDICATOR_ID,
                        "planned `{AUTOCORRELATION_INDICATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == AUTOCORRELATION_INDICATOR_OUTPUT_IDS,
                        "{AUTOCORRELATION_INDICATOR_ID}: selected-output CUDA contract {:?} != \
                         {:?}",
                        returned_output_ids,
                        AUTOCORRELATION_INDICATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Avsl {
                routes,
                fast_period,
                slow_period,
                multiplier_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == AVSL_OUTPUT_IDS,
                        "{AVSL_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        AVSL_OUTPUT_IDS
                    );
                    let parameter_tuples =
                        [(fast_period, slow_period, f64::from_bits(multiplier_bits))];
                    let output = engine.compute_avsl_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == AVSL_ID,
                        "planned `{AVSL_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == AVSL_OUTPUT_IDS,
                        "{AVSL_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        AVSL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Bandpass {
                routes,
                period,
                bandwidth_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == BANDPASS_OUTPUT_IDS,
                        "{BANDPASS_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        BANDPASS_OUTPUT_IDS
                    );
                    let parameter_tuples = [(period, f64::from_bits(bandwidth_bits))];
                    let output = engine.compute_bandpass_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == BANDPASS_ID,
                        "planned `{BANDPASS_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == BANDPASS_OUTPUT_IDS,
                        "{BANDPASS_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        BANDPASS_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::BollingerBands {
                routes,
                period,
                devup_bits,
                devdn_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == BOLLINGER_BANDS_OUTPUT_IDS,
                        "{BOLLINGER_BANDS_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        BOLLINGER_BANDS_OUTPUT_IDS
                    );
                    let parameter_tuples = [(
                        period,
                        f64::from_bits(devup_bits),
                        f64::from_bits(devdn_bits),
                    )];
                    let output =
                        engine.compute_bollinger_bands_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == BOLLINGER_BANDS_ID,
                        "planned `{BOLLINGER_BANDS_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == BOLLINGER_BANDS_OUTPUT_IDS,
                        "{BOLLINGER_BANDS_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        BOLLINGER_BANDS_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::BuffAverages {
                routes,
                fast_period,
                slow_period,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == BUFF_AVERAGES_OUTPUT_IDS,
                        "{BUFF_AVERAGES_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        BUFF_AVERAGES_OUTPUT_IDS
                    );
                    let parameter_tuples = [(fast_period, slow_period)];
                    let output = engine.compute_buff_averages_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == BUFF_AVERAGES_ID,
                        "planned `{BUFF_AVERAGES_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == BUFF_AVERAGES_OUTPUT_IDS,
                        "{BUFF_AVERAGES_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        BUFF_AVERAGES_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::CandleStrengthOscillator {
                routes,
                period,
                atr_enabled,
                atr_length,
                mode,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS,
                        "{CANDLE_STRENGTH_OSCILLATOR_ID}: typed admitted output contract {:?} \
                         != {:?}",
                        planned_output_ids,
                        CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
                    );
                    let mode = match mode {
                        ClassicCandleStrengthMode::Bollinger => "bollinger",
                    };
                    let parameter_rows = [CandleStrengthOscillatorParams {
                        period: Some(period),
                        atr_enabled: Some(atr_enabled),
                        atr_length: Some(atr_length),
                        mode: Some(mode.to_string()),
                    }];
                    let output = engine
                        .compute_candle_strength_oscillator_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == CANDLE_STRENGTH_OSCILLATOR_ID,
                        "planned `{CANDLE_STRENGTH_OSCILLATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS,
                        "{CANDLE_STRENGTH_OSCILLATOR_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::ChandelierExit {
                routes,
                period,
                mult_bits,
                use_close,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == CHANDELIER_EXIT_OUTPUT_IDS,
                        "{CHANDELIER_EXIT_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        CHANDELIER_EXIT_OUTPUT_IDS
                    );
                    let parameter_rows = [ChandelierExitParams {
                        period: Some(period),
                        mult: Some(f64::from_bits(mult_bits)),
                        use_close: Some(use_close),
                    }];
                    let output = engine.compute_chandelier_exit_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == CHANDELIER_EXIT_ID,
                        "planned `{CHANDELIER_EXIT_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == CHANDELIER_EXIT_OUTPUT_IDS,
                        "{CHANDELIER_EXIT_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        CHANDELIER_EXIT_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Cksp {
                routes,
                p,
                x_bits,
                q,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == CKSP_OUTPUT_IDS,
                        "{CKSP_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        CKSP_OUTPUT_IDS
                    );
                    let parameter_rows = [CkspParams {
                        p: Some(p),
                        x: Some(f64::from_bits(x_bits)),
                        q: Some(q),
                    }];
                    let output = engine.compute_cksp_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == CKSP_ID,
                        "planned `{CKSP_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == CKSP_OUTPUT_IDS,
                        "{CKSP_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        CKSP_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Coppock {
                route,
                short_roc_period,
                long_roc_period,
                ma_period,
            } => {
                if swept_point_fits_frame(&route.node, plan.rows) {
                    ensure!(
                        route.output_id == COPPOCK_OUTPUT_ID,
                        "{COPPOCK_ID}: typed admitted output `{}` != `{COPPOCK_OUTPUT_ID}`",
                        route.output_id
                    );
                    let parameter_rows = [CoppockParams {
                        short_roc_period: Some(short_roc_period),
                        long_roc_period: Some(long_roc_period),
                        ma_period: Some(ma_period),
                        ma_type: Some("wma".to_string()),
                    }];
                    let output = engine.compute_coppock_output_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == COPPOCK_ID,
                        "planned `{COPPOCK_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == [COPPOCK_OUTPUT_ID],
                        "{COPPOCK_ID}: production CUDA contract {:?} != \
                         [{COPPOCK_OUTPUT_ID:?}]",
                        returned_output_ids
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: vec![route],
                        output,
                    });
                } else {
                    pending.push(PendingClassicColumn::WarmupPlaceholder(route));
                }
            }
            ResolvedClassicCudaLaunch::CorrelationCycle {
                routes,
                period,
                threshold_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == CORRELATION_CYCLE_OUTPUT_IDS,
                        "{CORRELATION_CYCLE_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        CORRELATION_CYCLE_OUTPUT_IDS
                    );
                    let parameter_tuples = [(period, f64::from_bits(threshold_bits))];
                    let output =
                        engine.compute_correlation_cycle_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == CORRELATION_CYCLE_ID,
                        "planned `{CORRELATION_CYCLE_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == CORRELATION_CYCLE_OUTPUT_IDS,
                        "{CORRELATION_CYCLE_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        CORRELATION_CYCLE_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Cvi { route, period } => {
                if swept_point_fits_frame(&route.node, plan.rows) {
                    ensure!(
                        route.output_id == CVI_OUTPUT_ID,
                        "{CVI_ID}: typed admitted output `{}` != `{CVI_OUTPUT_ID}`",
                        route.output_id
                    );
                    let output = engine.compute_cvi_output_device(&[period])?;
                    ensure!(
                        output.indicator_id == CVI_ID,
                        "planned `{CVI_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == [CVI_OUTPUT_ID],
                        "{CVI_ID}: production CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        [CVI_OUTPUT_ID]
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: vec![route],
                        output,
                    });
                } else {
                    pending.push(PendingClassicColumn::WarmupPlaceholder(route));
                }
            }
            ResolvedClassicCudaLaunch::CyberpunkValueTrendAnalyzer {
                routes,
                entry_level,
                exit_level,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS,
                        "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: typed admitted output contract \
                         {:?} != {:?}",
                        planned_output_ids,
                        CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
                    );
                    let parameter_tuples = [(entry_level, exit_level)];
                    let output = engine
                        .compute_cyberpunk_value_trend_analyzer_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == CYBERPUNK_VALUE_TREND_ANALYZER_ID,
                        "planned `{CYBERPUNK_VALUE_TREND_ANALYZER_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS,
                        "{CYBERPUNK_VALUE_TREND_ANALYZER_ID}: production CUDA contract {:?} != \
                         {:?}",
                        returned_output_ids,
                        CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::CycleChannelOscillator {
                routes,
                short_cycle_length,
                medium_cycle_length,
                short_multiplier_bits,
                medium_multiplier_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS,
                        "{CYCLE_CHANNEL_OSCILLATOR_ID}: typed admitted output contract {:?} != \
                         {:?}",
                        planned_output_ids,
                        CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
                    );
                    let parameter_tuples = [(
                        short_cycle_length,
                        medium_cycle_length,
                        f64::from_bits(short_multiplier_bits),
                        f64::from_bits(medium_multiplier_bits),
                    )];
                    let output = engine
                        .compute_cycle_channel_oscillator_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == CYCLE_CHANNEL_OSCILLATOR_ID,
                        "planned `{CYCLE_CHANNEL_OSCILLATOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS,
                        "{CYCLE_CHANNEL_OSCILLATOR_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::DailyFactor {
                routes,
                threshold_level_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DAILY_FACTOR_PRODUCTION_OUTPUT_IDS,
                        "{DAILY_FACTOR_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DAILY_FACTOR_PRODUCTION_OUTPUT_IDS
                    );
                    let threshold_levels = [f64::from_bits(threshold_level_bits)];
                    let output = engine.compute_daily_factor_outputs_device(&threshold_levels)?;
                    ensure!(
                        output.indicator_id == DAILY_FACTOR_ID,
                        "planned `{DAILY_FACTOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DAILY_FACTOR_FULL_OUTPUT_IDS,
                        "{DAILY_FACTOR_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        DAILY_FACTOR_FULL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::DamianiVolatmeter {
                routes,
                vis_atr,
                vis_std,
                sed_atr,
                sed_std,
                threshold_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DAMIANI_VOLATMETER_OUTPUT_IDS,
                        "{DAMIANI_VOLATMETER_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DAMIANI_VOLATMETER_OUTPUT_IDS
                    );
                    let parameter_tuples = [(
                        vis_atr,
                        vis_std,
                        sed_atr,
                        sed_std,
                        f64::from_bits(threshold_bits),
                    )];
                    let output =
                        engine.compute_damiani_volatmeter_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == DAMIANI_VOLATMETER_ID,
                        "planned `{DAMIANI_VOLATMETER_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DAMIANI_VOLATMETER_OUTPUT_IDS,
                        "{DAMIANI_VOLATMETER_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DAMIANI_VOLATMETER_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Di { routes, period } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DI_OUTPUT_IDS,
                        "{DI_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DI_OUTPUT_IDS
                    );
                    let output = engine.compute_di_outputs_device(&[period])?;
                    ensure!(
                        output.indicator_id == DI_ID,
                        "planned `{DI_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DI_OUTPUT_IDS,
                        "{DI_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DI_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::DidiIndex {
                routes,
                short_length,
                medium_length,
                long_length,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DIDI_INDEX_OUTPUT_IDS,
                        "{DIDI_INDEX_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DIDI_INDEX_OUTPUT_IDS
                    );
                    let parameter_tuples = [(short_length, medium_length, long_length)];
                    let output = engine.compute_didi_index_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == DIDI_INDEX_ID,
                        "planned `{DIDI_INDEX_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DIDI_INDEX_OUTPUT_IDS,
                        "{DIDI_INDEX_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DIDI_INDEX_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::DirectionalImbalanceIndex {
                routes,
                length,
                period,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS,
                        "{DIRECTIONAL_IMBALANCE_INDEX_ID}: typed admitted output contract {:?} \
                         != {:?}",
                        planned_output_ids,
                        DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
                    );
                    let parameter_tuples = [(length, period)];
                    let output = engine
                        .compute_directional_imbalance_index_outputs_device(&parameter_tuples)?;
                    ensure!(
                        output.indicator_id == DIRECTIONAL_IMBALANCE_INDEX_ID,
                        "planned `{DIRECTIONAL_IMBALANCE_INDEX_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS,
                        "{DIRECTIONAL_IMBALANCE_INDEX_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::DisparityIndex {
                route,
                ema_period,
                lookback_period,
                smoothing_period,
                smoothing_is_sma,
            } => {
                if swept_point_fits_frame(&route.node, plan.rows) {
                    ensure!(
                        route.output_id == DISPARITY_INDEX_OUTPUT_ID,
                        "{DISPARITY_INDEX_ID}: typed admitted output `{}` != \
                         `{DISPARITY_INDEX_OUTPUT_ID}`",
                        route.output_id
                    );
                    let parameter_rows = [(
                        ema_period,
                        lookback_period,
                        smoothing_period,
                        smoothing_is_sma,
                    )];
                    let output = engine.compute_disparity_index_output_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == DISPARITY_INDEX_ID,
                        "planned `{DISPARITY_INDEX_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == [DISPARITY_INDEX_OUTPUT_ID],
                        "{DISPARITY_INDEX_ID}: production CUDA contract {:?} != \
                         [{DISPARITY_INDEX_OUTPUT_ID:?}]",
                        returned_output_ids
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: vec![route],
                        output,
                    });
                } else {
                    pending.push(PendingClassicColumn::WarmupPlaceholder(route));
                }
            }
            ResolvedClassicCudaLaunch::Dm { routes, period } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DM_OUTPUT_IDS,
                        "{DM_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DM_OUTPUT_IDS
                    );
                    let output = engine.compute_dm_outputs_device(&[period])?;
                    ensure!(
                        output.indicator_id == DM_ID,
                        "planned `{DM_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DM_OUTPUT_IDS,
                        "{DM_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DM_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Donchian { routes, period } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DONCHIAN_OUTPUT_IDS,
                        "{DONCHIAN_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DONCHIAN_OUTPUT_IDS
                    );
                    let output = engine.compute_donchian_outputs_device(&[period])?;
                    ensure!(
                        output.indicator_id == DONCHIAN_ID,
                        "planned `{DONCHIAN_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DONCHIAN_OUTPUT_IDS,
                        "{DONCHIAN_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DONCHIAN_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::DualUlcerIndex {
                routes,
                period,
                auto_threshold,
                threshold_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DUAL_ULCER_INDEX_OUTPUT_IDS,
                        "{DUAL_ULCER_INDEX_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DUAL_ULCER_INDEX_OUTPUT_IDS
                    );
                    let parameter_rows = [(period, auto_threshold, f64::from_bits(threshold_bits))];
                    let output = engine.compute_dual_ulcer_index_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == DUAL_ULCER_INDEX_ID,
                        "planned `{DUAL_ULCER_INDEX_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DUAL_ULCER_INDEX_OUTPUT_IDS,
                        "{DUAL_ULCER_INDEX_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DUAL_ULCER_INDEX_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Dvdiqqe {
                routes,
                period,
                smoothing_period,
                fast_multiplier_bits,
                slow_multiplier_bits,
                use_tick_only,
                dynamic_center,
                tick_size_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == DVDIQQE_OUTPUT_IDS,
                        "{DVDIQQE_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        DVDIQQE_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        period,
                        smoothing_period,
                        f64::from_bits(fast_multiplier_bits),
                        f64::from_bits(slow_multiplier_bits),
                        use_tick_only,
                        dynamic_center,
                        f64::from_bits(tick_size_bits),
                    )];
                    let output = engine.compute_dvdiqqe_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == DVDIQQE_ID,
                        "planned `{DVDIQQE_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == DVDIQQE_OUTPUT_IDS,
                        "{DVDIQQE_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        DVDIQQE_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EhlersAutocorrelationPeriodogram {
                routes,
                min_period,
                max_period,
                avg_length,
                enhance,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS,
                        "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: typed admitted output \
                         contract {:?} != {:?}",
                        planned_output_ids,
                        EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
                    );
                    let parameter_rows = [(min_period, max_period, avg_length, enhance)];
                    let output = engine.compute_ehlers_autocorrelation_periodogram_outputs_device(
                        &parameter_rows,
                    )?;
                    ensure!(
                        output.indicator_id == EHLERS_AUTOCORRELATION_PERIODOGRAM_ID,
                        "planned `{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS,
                        "{EHLERS_AUTOCORRELATION_PERIODOGRAM_ID}: production CUDA contract \
                         {:?} != {:?}",
                        returned_output_ids,
                        EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EhlersLinearExtrapolationPredictor {
                routes,
                high_pass_length,
                low_pass_length,
                gain_bits,
                bars_forward,
                signal_mode,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS,
                        "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        high_pass_length,
                        low_pass_length,
                        f64::from_bits(gain_bits),
                        bars_forward,
                        signal_mode,
                    )];
                    let output = engine
                        .compute_ehlers_linear_extrapolation_predictor_outputs_device(
                            &parameter_rows,
                        )?;
                    ensure!(
                        output.indicator_id == EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID,
                        "planned `{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS,
                        "{EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EhlersUndersampledDoubleMovingAverage {
                routes,
                fast_length,
                slow_length,
                sample_length,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS,
                        "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
                    );
                    let parameter_rows = [(fast_length, slow_length, sample_length)];
                    let output = engine
                        .compute_ehlers_undersampled_double_moving_average_outputs_device(
                            &parameter_rows,
                        )?;
                    ensure!(
                        output.indicator_id == EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID,
                        "planned `{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS,
                        "{EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EmaDeviationCorrectedT3 {
                routes,
                period,
                hot_bits,
                t3_mode,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS,
                        "{EMA_DEVIATION_CORRECTED_T3_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
                    );
                    let parameter_rows = [(period, f64::from_bits(hot_bits), t3_mode)];
                    let output = engine
                        .compute_ema_deviation_corrected_t3_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == EMA_DEVIATION_CORRECTED_T3_ID,
                        "planned `{EMA_DEVIATION_CORRECTED_T3_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS,
                        "{EMA_DEVIATION_CORRECTED_T3_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Emd {
                routes,
                period,
                delta_bits,
                fraction_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EMD_OUTPUT_IDS,
                        "{EMD_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EMD_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        period,
                        f64::from_bits(delta_bits),
                        f64::from_bits(fraction_bits),
                    )];
                    let output = engine.compute_emd_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == EMD_ID,
                        "planned `{EMD_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EMD_OUTPUT_IDS,
                        "{EMD_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EMD_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EmdTrend {
                routes,
                length,
                mult_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EMD_TREND_OUTPUT_IDS,
                        "{EMD_TREND_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EMD_TREND_OUTPUT_IDS
                    );
                    let output = engine
                        .compute_emd_trend_outputs_device(&[(length, f64::from_bits(mult_bits))])?;
                    ensure!(
                        output.indicator_id == EMD_TREND_ID,
                        "planned `{EMD_TREND_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EMD_TREND_OUTPUT_IDS,
                        "{EMD_TREND_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EMD_TREND_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Eri { routes, period } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == ERI_OUTPUT_IDS,
                        "{ERI_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        ERI_OUTPUT_IDS
                    );
                    let output = engine.compute_eri_outputs_device(&[period])?;
                    ensure!(
                        output.indicator_id == ERI_ID,
                        "planned `{ERI_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == ERI_OUTPUT_IDS,
                        "{ERI_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        ERI_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EvasiveSupertrend {
                routes,
                atr_length,
                base_multiplier_bits,
                noise_threshold_bits,
                expansion_alpha_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EVASIVE_SUPERTREND_OUTPUT_IDS,
                        "{EVASIVE_SUPERTREND_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EVASIVE_SUPERTREND_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        atr_length,
                        f64::from_bits(base_multiplier_bits),
                        f64::from_bits(noise_threshold_bits),
                        f64::from_bits(expansion_alpha_bits),
                    )];
                    let output =
                        engine.compute_evasive_supertrend_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == EVASIVE_SUPERTREND_ID,
                        "planned `{EVASIVE_SUPERTREND_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EVASIVE_SUPERTREND_OUTPUT_IDS,
                        "{EVASIVE_SUPERTREND_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EVASIVE_SUPERTREND_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::FibonacciTrailingStop {
                routes,
                left_bars,
                right_bars,
                level_bits,
                trigger_mode,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == FIBONACCI_TRAILING_STOP_OUTPUT_IDS,
                        "{FIBONACCI_TRAILING_STOP_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        FIBONACCI_TRAILING_STOP_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        left_bars,
                        right_bars,
                        f64::from_bits(level_bits),
                        trigger_mode,
                    )];
                    let output =
                        engine.compute_fibonacci_trailing_stop_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == FIBONACCI_TRAILING_STOP_ID,
                        "planned `{FIBONACCI_TRAILING_STOP_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == FIBONACCI_TRAILING_STOP_OUTPUT_IDS,
                        "{FIBONACCI_TRAILING_STOP_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        FIBONACCI_TRAILING_STOP_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Fisher { routes, period } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == FISHER_OUTPUT_IDS,
                        "{FISHER_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        FISHER_OUTPUT_IDS
                    );
                    let output = engine.compute_fisher_outputs_device(&[period])?;
                    ensure!(
                        output.indicator_id == FISHER_ID,
                        "planned `{FISHER_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == FISHER_OUTPUT_IDS,
                        "{FISHER_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        FISHER_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::ForwardBackwardExponentialOscillator {
                routes,
                length,
                smooth,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS,
                        "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
                    );
                    let parameter_rows = [(length, smooth)];
                    let output = engine
                        .compute_forward_backward_exponential_oscillator_outputs_device(
                            &parameter_rows,
                        )?;
                    ensure!(
                        output.indicator_id == FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID,
                        "planned {FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID} but CUDA returned {}",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS,
                        "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::FvgTrailingStop {
                routes,
                unmitigated_fvg_lookback,
                smoothing_length,
                reset_on_cross,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == FVG_TRAILING_STOP_OUTPUT_IDS,
                        "{FVG_TRAILING_STOP_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        FVG_TRAILING_STOP_OUTPUT_IDS
                    );
                    let parameter_rows =
                        [(unmitigated_fvg_lookback, smoothing_length, reset_on_cross)];
                    let output =
                        engine.compute_fvg_trailing_stop_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == FVG_TRAILING_STOP_ID,
                        "planned {FVG_TRAILING_STOP_ID} but CUDA returned {}",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == FVG_TRAILING_STOP_OUTPUT_IDS,
                        "{FVG_TRAILING_STOP_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        FVG_TRAILING_STOP_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Gatorosc {
                routes,
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == GATOROSC_OUTPUT_IDS,
                        "{GATOROSC_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        GATOROSC_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        jaws_length,
                        jaws_shift,
                        teeth_length,
                        teeth_shift,
                        lips_length,
                        lips_shift,
                    )];
                    let output = engine.compute_gatorosc_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == GATOROSC_ID,
                        "planned {GATOROSC_ID} but CUDA returned {}",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == GATOROSC_OUTPUT_IDS,
                        "{GATOROSC_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        GATOROSC_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::Halftrend {
                routes,
                amplitude,
                channel_deviation_bits,
                atr_period,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == HALFTREND_OUTPUT_IDS,
                        "{HALFTREND_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        HALFTREND_OUTPUT_IDS
                    );
                    let parameter_rows = [(
                        amplitude,
                        f64::from_bits(channel_deviation_bits),
                        atr_period,
                    )];
                    let output = engine.compute_halftrend_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == HALFTREND_ID,
                        "planned {HALFTREND_ID} but CUDA returned {}",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == HALFTREND_OUTPUT_IDS,
                        "{HALFTREND_ID}: production CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        HALFTREND_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::FibonacciEntryBands { routes, length } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS,
                        "{FIBONACCI_ENTRY_BANDS_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS
                    );
                    let sweep = FibonacciEntryBandsBatchRange {
                        length: (length, length, 0),
                        atr_length: (14, 14, 0),
                        source: "hlc3".into(),
                        use_atr: true,
                        tp_aggressiveness: "low".into(),
                    };
                    let output = engine.compute_fibonacci_entry_bands_outputs_device(&sweep)?;
                    ensure!(
                        output.indicator_id == FIBONACCI_ENTRY_BANDS_ID,
                        "planned `{FIBONACCI_ENTRY_BANDS_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS,
                        "{FIBONACCI_ENTRY_BANDS_ID}: full CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        FIBONACCI_ENTRY_BANDS_FULL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::EhlersDataSamplingRsi { routes, length } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS,
                        "{EHLERS_DATA_SAMPLING_RSI_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS
                    );
                    let output =
                        engine.compute_ehlers_data_sampling_rsi_outputs_device(&[length])?;
                    ensure!(
                        output.indicator_id == EHLERS_DATA_SAMPLING_RSI_ID,
                        "planned `{EHLERS_DATA_SAMPLING_RSI_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS,
                        "{EHLERS_DATA_SAMPLING_RSI_ID}: full CUDA contract {:?} != {:?}",
                        returned_output_ids,
                        EHLERS_DATA_SAMPLING_RSI_FULL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
            ResolvedClassicCudaLaunch::BullsVBears {
                routes,
                period,
                ma_type,
                calculation_method,
                normalized_bars_back,
                raw_rolling_period,
                raw_threshold_percentile_bits,
                threshold_level_bits,
            } => {
                if swept_point_fits_frame(&routes[0].node, plan.rows) {
                    let planned_output_ids = routes.each_ref().map(|route| route.output_id);
                    ensure!(
                        planned_output_ids == BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS,
                        "{BULLS_V_BEARS_ID}: typed admitted output contract {:?} != {:?}",
                        planned_output_ids,
                        BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS
                    );
                    let parameter_rows = [BullsVBearsParams {
                        period: Some(period),
                        ma_type: Some(ma_type),
                        calculation_method: Some(calculation_method),
                        normalized_bars_back: Some(normalized_bars_back),
                        raw_rolling_period: Some(raw_rolling_period),
                        raw_threshold_percentile: Some(f64::from_bits(
                            raw_threshold_percentile_bits,
                        )),
                        threshold_level: Some(f64::from_bits(threshold_level_bits)),
                    }];
                    let output = engine.compute_bulls_v_bears_outputs_device(&parameter_rows)?;
                    ensure!(
                        output.indicator_id == BULLS_V_BEARS_ID,
                        "planned `{BULLS_V_BEARS_ID}` but CUDA returned `{}`",
                        output.indicator_id
                    );
                    let returned_output_ids = output
                        .outputs
                        .iter()
                        .map(|named_output| named_output.output_id)
                        .collect::<Vec<_>>();
                    ensure!(
                        returned_output_ids == BULLS_V_BEARS_FULL_OUTPUT_IDS,
                        "{BULLS_V_BEARS_ID}: full CUDA output contract {:?} != {:?}",
                        returned_output_ids,
                        BULLS_V_BEARS_FULL_OUTPUT_IDS
                    );
                    pending.push(PendingClassicColumn::NamedResident {
                        routes: routes.into(),
                        output,
                    });
                } else {
                    pending.extend(
                        routes
                            .into_iter()
                            .map(PendingClassicColumn::WarmupPlaceholder),
                    );
                }
            }
        }
    }
    engine.synchronize()?;

    let mut ledger = admission.admission_ledger();
    let mut columns = Vec::with_capacity(plan.nodes.len());
    for pending_column in pending {
        match pending_column {
            PendingClassicColumn::PrimaryResident { route, output } => {
                let values = engine.download_primary_output_f64(output)?;
                ensure!(
                    values.len() == plan.rows,
                    "{}: downloaded {} values for {} rows",
                    route.node.column_name,
                    values.len(),
                    plan.rows
                );
                ledger.produced(route.node.indicator_id);
                columns.push((route.node.column_name, values));
            }
            PendingClassicColumn::NamedResident { routes, output } => {
                let planned_output_ids = routes
                    .iter()
                    .map(|route| route.output_id)
                    .collect::<Vec<_>>();
                let downloaded = engine.download_named_outputs_f64(&output, &planned_output_ids)?;
                ensure!(
                    downloaded.len() == routes.len(),
                    "{}: downloaded {} named outputs for {} planned columns",
                    output.indicator_id,
                    downloaded.len(),
                    routes.len()
                );
                for (route, (output_id, values)) in routes.into_iter().zip(downloaded) {
                    ensure!(
                        output_id == route.output_id,
                        "{}: planned output `{}` but downloaded `{output_id}`",
                        route.node.column_name,
                        route.output_id
                    );
                    ensure!(
                        values.len() == plan.rows,
                        "{}: downloaded {} values for {} rows",
                        route.node.column_name,
                        values.len(),
                        plan.rows
                    );
                    ledger.produced(route.node.indicator_id);
                    columns.push((route.node.column_name, values));
                }
            }
            PendingClassicColumn::WarmupPlaceholder(route) => {
                let period = route
                    .node
                    .parameters
                    .swept_period()
                    .expect("only swept nodes have a warmup placeholder");
                ledger.dropped(
                    route.node.indicator_id,
                    &route.node.column_name,
                    DropReason::PreflightWarmup,
                    format!(
                        "period {period} * 1.25 >= {} bars (#212 pre-flight guard); column emitted \
                         as all-NaN to keep the column set independent of the frame length",
                        plan.rows
                    ),
                );
                columns.push((route.node.column_name, vec![f64::NAN; plan.rows]));
            }
        }
    }

    let mut names = HashSet::with_capacity(columns.len());
    let mut fingerprints = HashSet::with_capacity(columns.len());
    for (name, values) in &columns {
        ensure!(
            names.insert(name.as_str()),
            "duplicate Classic CUDA column name `{name}`"
        );
        if let Some((row, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| value.is_infinite())
        {
            bail!("Classic CUDA column `{name}` emitted infinity {value} at row {row}");
        }
        if !fingerprints.insert(series_fingerprint(values)) {
            ledger.duplicate_column(name);
        }
        if !has_finite_variation(values) {
            ledger.degenerate_column(name);
        }
    }

    ledger.log_summary("classic-ta-gpu-only", plan.rows);
    if plan.rows >= VOCABULARY_FLOOR_MIN_ROWS {
        ledger.enforce_floor(
            "classic-ta-gpu-only",
            plan.rows,
            MIN_PRODUCING_INDICATOR_IDS,
            MIN_BASE_VOCABULARY_COLUMNS,
            admission.admitted_indicator_ids.len(),
            admission.admitted_base_columns,
        )?;
    }
    let historical_sweep_produced_columns = plan
        .nodes
        .iter()
        .filter(|node| node.stage == ClassicCudaStage::Historical)
        .count();
    let report = admission.execution_report(historical_sweep_produced_columns, columns.len());
    Ok(ClassicTaComputation {
        columns,
        report,
        ledger,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_plan_expands_dynamic_patterns_and_preserves_stage_order() {
        let patterns = vector_ta::indicators::pattern_recognition::list_patterns();
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &["rsi", "pattern_recognition"],
            &["ema"],
            &[("sma", vec![21, 50])],
        )
        .unwrap();
        assert_eq!(plan.nodes[0].column_name, "rsi");
        assert_eq!(
            plan.nodes[1].column_name,
            format!("pattern_recognition_{}", patterns[0].id)
        );
        assert_eq!(
            plan.nodes[patterns.len()].column_name,
            format!("pattern_recognition_{}", patterns[patterns.len() - 1].id)
        );
        assert_eq!(plan.nodes[patterns.len() + 1].column_name, "ema_7");
        assert_eq!(plan.nodes.last().unwrap().column_name, "sma_50");
        assert!(
            !patterns.is_empty(),
            "vector-ta pattern graph is unexpectedly empty"
        );
    }

    #[test]
    fn preflight_reports_every_discrete_row_and_dispatcher_blocker_in_order() {
        let plan =
            build_exact_classic_cuda_plan(1_000, &["ma", "pattern_recognition", "rsi"], &[], &[])
                .unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan).unwrap_err();
        assert!(gaps.iter().any(|gap| {
            gap.indicator_id == "ma"
                && gap.reason == ClassicCudaGapReason::MissingMovingAverageDispatcherRoute
        }));
        let pattern_gaps = gaps
            .iter()
            .filter(|gap| gap.reason == ClassicCudaGapReason::MissingDiscreteMatrixRoute)
            .collect::<Vec<_>>();
        assert_eq!(
            pattern_gaps.len(),
            vector_ta::indicators::pattern_recognition::list_patterns().len()
        );
        assert!(
            gaps.iter().all(|gap| gap.indicator_id != "rsi"),
            "supported primary rsi route was incorrectly rejected: {gaps:#?}"
        );
    }

    #[test]
    fn asi_default_outputs_preflight_as_one_exact_typed_launch() {
        let plan = build_exact_classic_cuda_plan(1_000, &[ASI_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AbsoluteStrengthIndexOscillator {
                routes,
                ema_length,
                signal_length,
            },
        ] = launches.as_slice()
        else {
            panic!("ASI must preflight as exactly one typed all-output launch");
        };
        assert_eq!((*ema_length, *signal_length), (21, 34));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ASI_OUTPUT_IDS,
            "canonical output identities/order drifted"
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "absolute_strength_index_oscillator_oscillator",
                "absolute_strength_index_oscillator_signal",
                "absolute_strength_index_oscillator_histogram",
            ]
        );
    }

    #[test]
    fn asi_sweep_preserves_the_registry_window_ratio_in_one_launch() {
        let plan = build_exact_classic_cuda_plan(1_000, &[], &[], &[(ASI_ID, vec![68])]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AbsoluteStrengthIndexOscillator {
                routes,
                ema_length,
                signal_length,
            },
        ] = launches.as_slice()
        else {
            panic!("swept ASI must preflight as exactly one typed all-output launch");
        };
        assert_eq!((*ema_length, *signal_length), (42, 68));
        assert!(
            routes
                .iter()
                .all(|route| route.node.stage == ClassicCudaStage::Extended)
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "absolute_strength_index_oscillator_68_oscillator",
                "absolute_strength_index_oscillator_68_signal",
                "absolute_strength_index_oscillator_68_histogram",
            ]
        );
    }

    #[test]
    fn adaptive_bounds_default_admitted_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ADAPTIVE_BOUNDS_RSI_ID),
            ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS.map(Some),
            "formula-level RSI exclusion or canonical output order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(1_000, &[ADAPTIVE_BOUNDS_RSI_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveBoundsRsi {
                routes,
                rsi_length,
                alpha_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Adaptive Bounds RSI must preflight as one typed all-output launch");
        };
        assert_eq!(*rsi_length, 14);
        assert_eq!(*alpha_bits, 0.1_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ADAPTIVE_BOUNDS_RSI_PRODUCTION_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "adaptive_bounds_rsi_lower",
                "adaptive_bounds_rsi_lower_mid",
                "adaptive_bounds_rsi_middle",
                "adaptive_bounds_rsi_upper_mid",
                "adaptive_bounds_rsi_upper",
                "adaptive_bounds_rsi_regime",
                "adaptive_bounds_rsi_regime_flip",
                "adaptive_bounds_rsi_lower_signal",
                "adaptive_bounds_rsi_upper_signal",
            ]
        );
    }

    #[test]
    fn adaptive_bounds_sweep_changes_only_rsi_length_and_keeps_alpha_bits() {
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(ADAPTIVE_BOUNDS_RSI_ID, vec![28])])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveBoundsRsi {
                routes,
                rsi_length,
                alpha_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Adaptive Bounds RSI must remain one typed launch");
        };
        assert_eq!(*rsi_length, 28);
        assert_eq!(*alpha_bits, 0.1_f64.to_bits());
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route
                    .node
                    .column_name
                    .starts_with("adaptive_bounds_rsi_28_")
        }));
    }

    #[test]
    fn adaptive_macd_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ADAPTIVE_MACD_ID),
            ADAPTIVE_MACD_OUTPUT_IDS.map(Some),
            "canonical Adaptive MACD output identities or order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ADAPTIVE_MACD_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveMacd {
                routes,
                length,
                fast_period,
                slow_period,
                signal_period,
            },
        ] = launches.as_slice()
        else {
            panic!("Adaptive MACD must preflight as one typed all-output launch");
        };
        assert_eq!(
            (*length, *fast_period, *slow_period, *signal_period),
            (20, 10, 20, 9)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ADAPTIVE_MACD_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "adaptive_macd_macd",
                "adaptive_macd_signal",
                "adaptive_macd_hist"
            ]
        );
    }

    #[test]
    fn adaptive_macd_sweep_changes_only_length_and_keeps_other_defaults() {
        let plan = build_exact_classic_cuda_plan(1_000, &[], &[], &[(ADAPTIVE_MACD_ID, vec![50])])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveMacd {
                routes,
                length,
                fast_period,
                slow_period,
                signal_period,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Adaptive MACD must remain one typed launch");
        };
        assert_eq!(
            (*length, *fast_period, *slow_period, *signal_period),
            (50, 10, 20, 9)
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route.node.column_name.starts_with("adaptive_macd_50_")
        }));
    }

    #[test]
    fn adaptive_momentum_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ADAPTIVE_MOMENTUM_OSCILLATOR_ID),
            ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS.map(Some),
            "canonical Adaptive Momentum Oscillator output identities or order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(1_000, &[ADAPTIVE_MOMENTUM_OSCILLATOR_ID], &[], &[])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveMomentumOscillator {
                routes,
                length,
                smoothing_length,
            },
        ] = launches.as_slice()
        else {
            panic!("Adaptive Momentum Oscillator must preflight as one typed all-output launch");
        };
        assert_eq!((*length, *smoothing_length), (14, 9));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ADAPTIVE_MOMENTUM_OSCILLATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "adaptive_momentum_oscillator_amo",
                "adaptive_momentum_oscillator_ama",
            ]
        );
    }

    #[test]
    fn adaptive_momentum_sweep_changes_only_length_and_keeps_smoothing_default() {
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(ADAPTIVE_MOMENTUM_OSCILLATOR_ID, vec![50])],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveMomentumOscillator {
                routes,
                length,
                smoothing_length,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Adaptive Momentum Oscillator must remain one typed launch");
        };
        assert_eq!((*length, *smoothing_length), (50, 9));
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route
                    .node
                    .column_name
                    .starts_with("adaptive_momentum_oscillator_50_")
        }));
    }

    #[test]
    fn adaptive_schaff_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ADAPTIVE_SCHAFF_TREND_CYCLE_ID),
            ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS.map(Some),
            "canonical ASTC output identities or order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(1_000, &[ADAPTIVE_SCHAFF_TREND_CYCLE_ID], &[], &[])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveSchaffTrendCycle {
                routes,
                adaptive_length,
                stc_length,
                smoothing_factor_bits,
                fast_length,
                slow_length,
            },
        ] = launches.as_slice()
        else {
            panic!("ASTC must preflight as one typed all-output launch");
        };
        assert_eq!(
            (
                *adaptive_length,
                *stc_length,
                f64::from_bits(*smoothing_factor_bits),
                *fast_length,
                *slow_length,
            ),
            (55, 12, 0.45, 26, 50)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ADAPTIVE_SCHAFF_TREND_CYCLE_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "adaptive_schaff_trend_cycle_stc",
                "adaptive_schaff_trend_cycle_histogram",
            ]
        );
    }

    #[test]
    fn adaptive_schaff_sweep_scales_four_windows_and_keeps_smoothing_bits() {
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(ADAPTIVE_SCHAFF_TREND_CYCLE_ID, vec![110])],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveSchaffTrendCycle {
                routes,
                adaptive_length,
                stc_length,
                smoothing_factor_bits,
                fast_length,
                slow_length,
            },
        ] = launches.as_slice()
        else {
            panic!("swept ASTC must remain one typed all-output launch");
        };
        assert_eq!(
            (*adaptive_length, *stc_length, *fast_length, *slow_length),
            (110, 24, 52, 100)
        );
        assert_eq!(*smoothing_factor_bits, 0.45_f64.to_bits());
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route
                    .node
                    .column_name
                    .starts_with("adaptive_schaff_trend_cycle_110_")
        }));
    }

    #[test]
    fn adjustable_ma_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ADJUSTABLE_MA_ID),
            ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS.map(Some),
            "the reviewed smoothed_close duplicate exclusion or admitted order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ADJUSTABLE_MA_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdjustableMaAlternatingExtremities {
                routes,
                length,
                mult_bits,
                alpha_bits,
                beta_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Adjustable MA must preflight as one typed admitted-output launch");
        };
        assert_eq!(
            (
                *length,
                f64::from_bits(*mult_bits),
                f64::from_bits(*alpha_bits),
                f64::from_bits(*beta_bits),
            ),
            (50, 2.0, 1.0, 0.5)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ADJUSTABLE_MA_PRODUCTION_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "adjustable_ma_alternating_extremities_ma",
                "adjustable_ma_alternating_extremities_upper",
                "adjustable_ma_alternating_extremities_lower",
                "adjustable_ma_alternating_extremities_extremity",
                "adjustable_ma_alternating_extremities_state",
                "adjustable_ma_alternating_extremities_changed",
                "adjustable_ma_alternating_extremities_smoothed_open",
                "adjustable_ma_alternating_extremities_smoothed_high",
                "adjustable_ma_alternating_extremities_smoothed_low",
            ]
        );
        assert!(
            plan.nodes
                .iter()
                .all(|node| node.requested_output_id != Some("smoothed_close")),
            "the structurally duplicate smoothed_close output entered production"
        );
    }

    #[test]
    fn adjustable_ma_sweep_changes_length_and_keeps_three_float_defaults_by_bits() {
        let plan = build_exact_classic_cuda_plan(1_000, &[], &[], &[(ADJUSTABLE_MA_ID, vec![100])])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdjustableMaAlternatingExtremities {
                routes,
                length,
                mult_bits,
                alpha_bits,
                beta_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Adjustable MA must remain one typed admitted-output launch");
        };
        assert_eq!(*length, 100);
        assert_eq!(*mult_bits, 2.0_f64.to_bits());
        assert_eq!(*alpha_bits, 1.0_f64.to_bits());
        assert_eq!(*beta_bits, 0.5_f64.to_bits());
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route
                    .node
                    .column_name
                    .starts_with("adjustable_ma_alternating_extremities_100_")
                && route.node.requested_output_id != Some("smoothed_close")
        }));
    }

    #[test]
    fn alligator_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ALLIGATOR_ID),
            ALLIGATOR_OUTPUT_IDS.map(Some),
            "canonical jaw/teeth/lips output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ALLIGATOR_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Alligator {
                routes,
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
            },
        ] = launches.as_slice()
        else {
            panic!("Alligator must preflight as one typed three-output launch");
        };
        assert_eq!(
            (
                *jaw_period,
                *jaw_offset,
                *teeth_period,
                *teeth_offset,
                *lips_period,
                *lips_offset,
            ),
            (13, 8, 8, 5, 5, 3)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ALLIGATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["alligator_jaw", "alligator_teeth", "alligator_lips"]
        );
    }

    #[test]
    fn alphatrend_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ALPHATREND_ID),
            ALPHATREND_OUTPUT_IDS.map(Some),
            "canonical k1/k2 output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ALPHATREND_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AlphaTrend {
                routes,
                coeff_bits,
                period,
                no_volume,
            },
        ] = launches.as_slice()
        else {
            panic!("AlphaTrend must preflight as one typed two-output launch");
        };
        assert_eq!(*coeff_bits, 1.0_f64.to_bits());
        assert_eq!(*period, 14);
        assert!(!*no_volume);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ALPHATREND_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["alphatrend_k1", "alphatrend_k2"]
        );
    }

    #[test]
    fn andean_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ANDEAN_OSCILLATOR_ID),
            ANDEAN_OSCILLATOR_OUTPUT_IDS.map(Some),
            "canonical bull/bear/signal output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ANDEAN_OSCILLATOR_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AndeanOscillator {
                routes,
                length,
                signal_length,
            },
        ] = launches.as_slice()
        else {
            panic!("Andean Oscillator must preflight as one typed three-output launch");
        };
        assert_eq!((*length, *signal_length), (50, 9));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ANDEAN_OSCILLATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "andean_oscillator_bull",
                "andean_oscillator_bear",
                "andean_oscillator_signal",
            ]
        );
    }

    #[test]
    fn aroon_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(AROON_ID),
            AROON_OUTPUT_IDS.map(Some),
            "canonical up/down output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[AROON_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Aroon { routes, length }] = launches.as_slice() else {
            panic!("Aroon must preflight as one typed two-output launch");
        };
        assert_eq!(*length, 14);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            AROON_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["aroon_up", "aroon_down"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(14),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn aso_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ASO_ID),
            ASO_OUTPUT_IDS.map(Some),
            "canonical bulls/bears output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ASO_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Aso {
                routes,
                period,
                mode,
            },
        ] = launches.as_slice()
        else {
            panic!("ASO must preflight as one typed two-output launch");
        };
        assert_eq!((*period, *mode), (10, 0));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ASO_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["aso_bulls", "aso_bears"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(10),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn autocorrelation_indicator_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(AUTOCORRELATION_INDICATOR_ID),
            AUTOCORRELATION_INDICATOR_OUTPUT_IDS.map(Some),
            "canonical filtered/correlation output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[AUTOCORRELATION_INDICATOR_ID], &[], &[])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AutocorrelationIndicator {
                routes,
                length,
                lag,
                use_test_signal,
            },
        ] = launches.as_slice()
        else {
            panic!("Autocorrelation Indicator must preflight as one typed two-output launch");
        };
        assert_eq!((*length, *lag, *use_test_signal), (20, 1, false));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            AUTOCORRELATION_INDICATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "autocorrelation_indicator_filtered",
                "autocorrelation_indicator_correlation",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn avsl_default_output_forms_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(AVSL_ID),
            AVSL_REQUESTED_OUTPUT_IDS,
            "canonical single-output AVSL request must retain the CPU default-output receipt"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[AVSL_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Avsl {
                routes,
                fast_period,
                slow_period,
                multiplier_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("AVSL must preflight as one typed single-output launch");
        };
        assert_eq!(
            (*fast_period, *slow_period, *multiplier_bits),
            (12, 26, 2.0_f64.to_bits())
        );
        assert_eq!(routes[0].output_id, "value");
        assert_eq!(routes[0].node.requested_output_id, None);
        assert_eq!(routes[0].node.column_name, "avsl");
        assert_eq!(
            routes[0].node.parameters,
            ClassicCudaParameters::Defaults {
                anchor: ClassicCudaAnchor::Resolved(26),
                require_period_invariant_kernel: false,
            }
        );
    }

    #[test]
    fn acosc_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ACOSC_ID),
            ACOSC_OUTPUT_IDS.map(Some),
            "canonical osc/change output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[ACOSC_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Acosc { routes }] = launches.as_slice() else {
            panic!("ACOSC must preflight as one typed two-output launch");
        };
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ACOSC_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["acosc_osc", "acosc_change"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
        }));
    }

    #[test]
    fn adaptive_bandpass_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID),
            ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS.map(Some),
            "canonical in_phase/lead output order drifted"
        );
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_ID],
            &[],
            &[],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AdaptiveBandpassTriggerOscillator {
                routes,
                delta_bits,
                alpha_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Adaptive Bandpass must preflight as one typed two-output launch");
        };
        assert_eq!(*delta_bits, 0.1_f64.to_bits());
        assert_eq!(*alpha_bits, 0.07_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ADAPTIVE_BANDPASS_TRIGGER_OSCILLATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "adaptive_bandpass_trigger_oscillator_in_phase",
                "adaptive_bandpass_trigger_oscillator_lead",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
        }));
        let swept = ClassicCudaParameters::Swept {
            period: 50,
            overrides: vec![("period", 50)],
            anchor: ClassicCudaAnchor::Resolved(50),
        };
        let error = resolve_adaptive_bandpass_trigger_oscillator_parameters(&swept)
            .expect_err("Adaptive Bandpass coefficients must not enter an integer period sweep");
        assert!(error.contains("not a canonical integer period sweep"));
    }

    #[test]
    fn alphatrend_sweep_changes_only_period_and_keeps_other_defaults_exact() {
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(ALPHATREND_ID, vec![100])]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AlphaTrend {
                routes,
                coeff_bits,
                period,
                no_volume,
            },
        ] = launches.as_slice()
        else {
            panic!("swept AlphaTrend must remain one typed two-output launch");
        };
        assert_eq!(*coeff_bits, 1.0_f64.to_bits());
        assert_eq!(*period, 100);
        assert!(!*no_volume);
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route.node.column_name.starts_with("alphatrend_100_")
        }));
    }

    #[test]
    fn andean_length_sweep_keeps_signal_default_in_one_exact_typed_launch() {
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(ANDEAN_OSCILLATOR_ID, vec![100])])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::AndeanOscillator {
                routes,
                length,
                signal_length,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Andean Oscillator must remain one typed three-output launch");
        };
        assert_eq!((*length, *signal_length), (100, 9));
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route.node.column_name.starts_with("andean_oscillator_100_")
                && route.node.parameters
                    == ClassicCudaParameters::Swept {
                        period: 100,
                        overrides: vec![("length", 100)],
                        anchor: ClassicCudaAnchor::Resolved(100),
                    }
        }));
    }

    #[test]
    fn aroon_length_sweep_forms_five_exact_typed_two_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(AROON_ID, periods.clone())]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_length) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Aroon { routes, length } = launch else {
                panic!("every Aroon sweep point must remain one typed two-output launch");
            };
            assert_eq!(*length, expected_length);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                AROON_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("aroon_{expected_length}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_length,
                            overrides: vec![("length", expected_length as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_length),
                        }
            }));
        }
    }

    #[test]
    fn aso_period_sweep_forms_five_exact_typed_two_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(ASO_ID, periods.clone())]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Aso {
                routes,
                period,
                mode,
            } = launch
            else {
                panic!("every ASO sweep point must remain one typed two-output launch");
            };
            assert_eq!((*period, *mode), (expected_period, 0));
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                ASO_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("aso_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
            }));
        }
    }

    #[test]
    fn autocorrelation_indicator_length_sweep_forms_five_exact_typed_two_output_launches() {
        let lengths = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(AUTOCORRELATION_INDICATOR_ID, lengths.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), lengths.len());
        for (launch, expected_length) in launches.iter().zip(lengths) {
            let ResolvedClassicCudaLaunch::AutocorrelationIndicator {
                routes,
                length,
                lag,
                use_test_signal,
            } = launch
            else {
                panic!("every ACI sweep point must remain one typed two-output launch");
            };
            assert_eq!(
                (*length, *lag, *use_test_signal),
                (expected_length, 1, false)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                AUTOCORRELATION_INDICATOR_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("autocorrelation_indicator_{expected_length}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_length,
                            overrides: vec![("length", expected_length as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_length),
                        }
            }));
        }
    }

    #[test]
    fn avsl_ratio_sweep_forms_five_exact_typed_single_output_launches() {
        let slow_periods = vec![7, 21, 50, 100, 200];
        let expected_fast_periods = [3, 10, 23, 46, 92];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(AVSL_ID, slow_periods.clone())])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), slow_periods.len());
        for ((launch, expected_slow), expected_fast) in
            launches.iter().zip(slow_periods).zip(expected_fast_periods)
        {
            let ResolvedClassicCudaLaunch::Avsl {
                routes,
                fast_period,
                slow_period,
                multiplier_bits,
            } = launch
            else {
                panic!("every AVSL sweep point must remain one typed single-output launch");
            };
            assert_eq!(
                (*fast_period, *slow_period, *multiplier_bits),
                (expected_fast, expected_slow, 2.0_f64.to_bits())
            );
            assert_eq!(routes[0].output_id, "value");
            assert_eq!(
                routes[0].node.parameters,
                ClassicCudaParameters::Swept {
                    period: expected_slow,
                    overrides: vec![
                        ("fast_period", expected_fast as i64),
                        ("slow_period", expected_slow as i64),
                    ],
                    anchor: ClassicCudaAnchor::Resolved(expected_slow),
                }
            );
            assert_eq!(routes[0].node.column_name, format!("avsl_{expected_slow}"));
            assert_eq!(routes[0].node.requested_output_id, None);
        }
    }

    #[test]
    fn classic_bandpass_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(BANDPASS_ID),
            BANDPASS_OUTPUT_IDS.map(Some),
            "canonical Bandpass output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(4_000, &[BANDPASS_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Bandpass {
                routes,
                period,
                bandwidth_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Bandpass must preflight as one typed four-output launch");
        };
        assert_eq!((*period, *bandwidth_bits), (20, 0.3_f64.to_bits()));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            BANDPASS_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "bandpass_bp",
                "bandpass_bp_normalized",
                "bandpass_signal",
                "bandpass_trigger",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn classic_bandpass_period_sweep_forms_five_exact_typed_four_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(4_000, &[], &[], &[(BANDPASS_ID, periods.clone())])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Bandpass {
                routes,
                period,
                bandwidth_bits,
            } = launch
            else {
                panic!("every Bandpass sweep point must remain one typed four-output launch");
            };
            assert_eq!(
                (*period, *bandwidth_bits),
                (expected_period, 0.3_f64.to_bits())
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                BANDPASS_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("bandpass_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
            }));
        }
    }

    #[test]
    fn classic_bollinger_bands_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(BOLLINGER_BANDS_ID),
            BOLLINGER_BANDS_OUTPUT_IDS.map(Some),
            "canonical Bollinger Bands output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[BOLLINGER_BANDS_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::BollingerBands {
                routes,
                period,
                devup_bits,
                devdn_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Bollinger Bands must preflight as one typed three-output launch");
        };
        assert_eq!(
            (*period, *devup_bits, *devdn_bits),
            (20, 2.0_f64.to_bits(), 2.0_f64.to_bits())
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            BOLLINGER_BANDS_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "bollinger_bands_upper",
                "bollinger_bands_middle",
                "bollinger_bands_lower",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn classic_bollinger_bands_period_sweep_forms_five_exact_typed_three_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(BOLLINGER_BANDS_ID, periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::BollingerBands {
                routes,
                period,
                devup_bits,
                devdn_bits,
            } = launch
            else {
                panic!(
                    "every Bollinger Bands sweep point must remain one typed three-output launch"
                );
            };
            assert_eq!(
                (*period, *devup_bits, *devdn_bits),
                (expected_period, 2.0_f64.to_bits(), 2.0_f64.to_bits())
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                BOLLINGER_BANDS_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("bollinger_bands_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
            }));
        }
    }

    #[test]
    fn buff_averages_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(BUFF_AVERAGES_ID),
            BUFF_AVERAGES_OUTPUT_IDS.map(Some),
            "canonical Buff Averages output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[BUFF_AVERAGES_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::BuffAverages {
                routes,
                fast_period,
                slow_period,
            },
        ] = launches.as_slice()
        else {
            panic!("Buff Averages must preflight as one typed two-output launch");
        };
        assert_eq!((*fast_period, *slow_period), (5, 20));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            BUFF_AVERAGES_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["buff_averages_fast", "buff_averages_slow"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn buff_averages_ratio_sweep_forms_five_exact_typed_two_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let expected_fast_periods = [2, 5, 13, 25, 50];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(BUFF_AVERAGES_ID, periods.clone())])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for ((launch, expected_slow_period), expected_fast_period) in
            launches.iter().zip(periods).zip(expected_fast_periods)
        {
            let ResolvedClassicCudaLaunch::BuffAverages {
                routes,
                fast_period,
                slow_period,
            } = launch
            else {
                panic!("every Buff Averages sweep point must remain one typed two-output launch");
            };
            assert_eq!(
                (*fast_period, *slow_period),
                (expected_fast_period, expected_slow_period)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                BUFF_AVERAGES_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_slow_period,
                            overrides: vec![
                                ("fast_period", expected_fast_period as i64),
                                ("slow_period", expected_slow_period as i64),
                            ],
                            anchor: ClassicCudaAnchor::Resolved(expected_slow_period),
                        }
            }));
        }
    }

    #[test]
    fn candle_strength_oscillator_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(CANDLE_STRENGTH_OSCILLATOR_ID),
            CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS.map(Some),
            "canonical Candle Strength Oscillator output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[CANDLE_STRENGTH_OSCILLATOR_ID], &[], &[])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::CandleStrengthOscillator {
                routes,
                period,
                atr_enabled,
                atr_length,
                mode,
            },
        ] = launches.as_slice()
        else {
            panic!("Candle Strength Oscillator must preflight as one typed six-output launch");
        };
        assert_eq!(
            (*period, *atr_enabled, *atr_length, *mode),
            (50, false, 50, ClassicCandleStrengthMode::Bollinger)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "candle_strength_oscillator_strength",
                "candle_strength_oscillator_highs",
                "candle_strength_oscillator_lows",
                "candle_strength_oscillator_mid",
                "candle_strength_oscillator_long_signal",
                "candle_strength_oscillator_short_signal",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(50),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn candle_strength_oscillator_period_sweep_forms_five_exact_typed_six_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(CANDLE_STRENGTH_OSCILLATOR_ID, periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::CandleStrengthOscillator {
                routes,
                period,
                atr_enabled,
                atr_length,
                mode,
            } = launch
            else {
                panic!(
                    "every Candle Strength Oscillator sweep point must remain one typed \
                     six-output launch"
                );
            };
            assert_eq!(
                (*period, *atr_enabled, *atr_length, *mode),
                (
                    expected_period,
                    false,
                    50,
                    ClassicCandleStrengthMode::Bollinger,
                )
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("candle_strength_oscillator_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
            }));
        }
    }

    #[test]
    fn chandelier_exit_default_outputs_form_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(CHANDELIER_EXIT_ID),
            CHANDELIER_EXIT_OUTPUT_IDS.map(Some),
            "canonical Chandelier Exit output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[CHANDELIER_EXIT_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::ChandelierExit {
                routes,
                period,
                mult_bits,
                use_close,
            },
        ] = launches.as_slice()
        else {
            panic!("Chandelier Exit must preflight as one typed two-output launch");
        };
        assert_eq!(
            (*period, *mult_bits, *use_close),
            (22, 3.0_f64.to_bits(), true)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            CHANDELIER_EXIT_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["chandelier_exit_long_stop", "chandelier_exit_short_stop"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(22),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn chandelier_exit_period_sweep_forms_five_exact_typed_pair_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(CHANDELIER_EXIT_ID, periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::ChandelierExit {
                routes,
                period,
                mult_bits,
                use_close,
            } = launch
            else {
                panic!("every Chandelier Exit sweep point must remain one typed pair launch");
            };
            assert_eq!(
                (*period, *mult_bits, *use_close),
                (expected_period, 3.0_f64.to_bits(), true)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                CHANDELIER_EXIT_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("chandelier_exit_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
            }));
        }
    }

    #[test]
    fn cksp_default_outputs_form_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(CKSP_ID),
            CKSP_OUTPUT_IDS.map(Some),
            "canonical CKSP output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[CKSP_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Cksp {
                routes,
                p,
                x_bits,
                q,
            },
        ] = launches.as_slice()
        else {
            panic!("CKSP must preflight as one typed two-output default launch");
        };
        assert_eq!((*p, *x_bits, *q), (10, 1.0_f64.to_bits(), 9));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            CKSP_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["cksp_long_values", "cksp_short_values"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
        }));
    }

    #[test]
    fn cksp_synthetic_period_sweep_fails_closed_before_launch() {
        let plan = build_exact_classic_cuda_plan(1_000, &[], &[], &[(CKSP_ID, vec![7])]).unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan).unwrap_err();
        assert!(
            gaps.iter().any(|gap| {
                gap.indicator_id == CKSP_ID
                    && gap.reason == ClassicCudaGapReason::MissingParameterContract
                    && (gap.detail.contains(
                        "canonical CPU sweep produced no explicit integer window overrides",
                    ) || gap.detail.contains("default-only"))
            }),
            "synthetic CKSP sweep must fail before any launch: {gaps:#?}"
        );
    }

    #[test]
    fn alligator_ratio_sweep_scales_three_periods_and_preserves_offsets() {
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(ALLIGATOR_ID, vec![100])]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Alligator {
                routes,
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Alligator must remain one typed three-output launch");
        };
        assert_eq!(
            (
                *jaw_period,
                *jaw_offset,
                *teeth_period,
                *teeth_offset,
                *lips_period,
                *lips_offset,
            ),
            (100, 8, 62, 5, 38, 3)
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route.node.column_name.starts_with("alligator_100_")
        }));
    }

    #[test]
    fn bulls_v_bears_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(BULLS_V_BEARS_ID),
            BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS.map(Some),
            "the three reviewed Bulls v Bears exclusions or admitted order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[BULLS_V_BEARS_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::BullsVBears {
                routes,
                period,
                ma_type,
                calculation_method,
                normalized_bars_back,
                raw_rolling_period,
                raw_threshold_percentile_bits,
                threshold_level_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Bulls v Bears must preflight as one typed admitted-output launch");
        };
        assert_eq!(*period, 14);
        assert_eq!(*ma_type, BullsVBearsMaType::Ema);
        assert_eq!(
            *calculation_method,
            BullsVBearsCalculationMethod::Normalized
        );
        assert_eq!(*normalized_bars_back, 120);
        assert_eq!(*raw_rolling_period, 50);
        assert_eq!(*raw_threshold_percentile_bits, 95.0_f64.to_bits());
        assert_eq!(*threshold_level_bits, 80.0_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            BULLS_V_BEARS_PRODUCTION_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "bulls_v_bears_value",
                "bulls_v_bears_bull",
                "bulls_v_bears_bear",
                "bulls_v_bears_bullish_signal",
                "bulls_v_bears_bearish_signal",
                "bulls_v_bears_zero_cross_up",
                "bulls_v_bears_zero_cross_down",
            ]
        );
        for excluded in ["ma", "upper", "lower"] {
            assert!(
                plan.nodes
                    .iter()
                    .all(|node| node.requested_output_id != Some(excluded)),
                "reviewed excluded output `{excluded}` entered production"
            );
        }
    }

    #[test]
    fn bulls_v_bears_sweep_changes_period_and_keeps_six_auxiliary_defaults() {
        let plan = build_exact_classic_cuda_plan(1_000, &[], &[], &[(BULLS_V_BEARS_ID, vec![100])])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::BullsVBears {
                routes,
                period,
                ma_type,
                calculation_method,
                normalized_bars_back,
                raw_rolling_period,
                raw_threshold_percentile_bits,
                threshold_level_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("swept Bulls v Bears must remain one typed admitted-output launch");
        };
        assert_eq!(*period, 100);
        assert_eq!(*ma_type, BullsVBearsMaType::Ema);
        assert_eq!(
            *calculation_method,
            BullsVBearsCalculationMethod::Normalized
        );
        assert_eq!(*normalized_bars_back, 120);
        assert_eq!(*raw_rolling_period, 50);
        assert_eq!(*raw_threshold_percentile_bits, 95.0_f64.to_bits());
        assert_eq!(*threshold_level_bits, 80.0_f64.to_bits());
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Extended
                && route.node.column_name.starts_with("bulls_v_bears_100_")
                && ![Some("ma"), Some("upper"), Some("lower")]
                    .contains(&route.node.requested_output_id)
        }));
    }

    #[test]
    fn exact_plan_uses_output_ids_for_and_budget_counting_without_readding_exclusions() {
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &["adjustable_ma_alternating_extremities", "mwdx"],
            &[],
            &[],
        )
        .unwrap();
        assert!(
            plan.nodes
                .iter()
                .all(|node| node.column_name
                    != "adjustable_ma_alternating_extremities_smoothed_close")
        );
        assert!(plan.nodes.iter().all(|node| node.indicator_id != "mwdx"));
        assert_eq!(
            plan.nodes.len(),
            planned_output_count("adjustable_ma_alternating_extremities")
                + planned_output_count("mwdx")
        );
    }

    #[test]
    fn coppock_default_output_forms_one_exact_typed_tuple_launch() {
        assert_eq!(
            output_ids_for("coppock"),
            [None],
            "canonical Coppock receipt must remain the sole unsuffixed value output"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &["coppock"], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Coppock {
                route,
                short_roc_period,
                long_roc_period,
                ma_period,
            },
        ] = launches.as_slice()
        else {
            panic!("Coppock must preflight as one typed single-output tuple launch");
        };
        assert_eq!(
            (*short_roc_period, *long_roc_period, *ma_period),
            (11, 14, 10)
        );
        assert_eq!(route.output_id, "value");
        assert_eq!(route.node.column_name, "coppock");
        assert_eq!(route.node.stage, ClassicCudaStage::Base);
        assert_eq!(
            route.node.parameters,
            ClassicCudaParameters::Defaults {
                anchor: ClassicCudaAnchor::Resolved(14),
                require_period_invariant_kernel: false,
            }
        );
    }

    #[test]
    fn coppock_ratio_sweep_forms_five_exact_typed_tuple_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let expected = [
            (6, 7, 5),
            (17, 21, 15),
            (39, 50, 36),
            (79, 100, 71),
            (157, 200, 143),
        ];
        let plan = build_exact_classic_cuda_plan(1_000, &[], &[], &[("coppock", periods.clone())])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for ((launch, period), (expected_short, expected_long, expected_ma)) in
            launches.iter().zip(periods).zip(expected)
        {
            let ResolvedClassicCudaLaunch::Coppock {
                route,
                short_roc_period,
                long_roc_period,
                ma_period,
            } = launch
            else {
                panic!("every Coppock ratio point must remain one typed value launch");
            };
            assert_eq!(
                (*short_roc_period, *long_roc_period, *ma_period),
                (expected_short, expected_long, expected_ma)
            );
            assert_eq!(route.output_id, "value");
            assert_eq!(route.node.column_name, format!("coppock_{period}"));
            assert_eq!(route.node.stage, ClassicCudaStage::Extended);
            assert_eq!(
                route.node.parameters,
                ClassicCudaParameters::Swept {
                    period,
                    overrides: vec![
                        ("short_roc_period", expected_short as i64),
                        ("long_roc_period", expected_long as i64),
                        ("ma_period", expected_ma as i64),
                    ],
                    anchor: ClassicCudaAnchor::Resolved(period),
                }
            );
        }
    }

    #[test]
    fn correlation_cycle_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for("correlation_cycle"),
            [Some("real"), Some("imag"), Some("angle"), Some("state")],
            "canonical Correlation Cycle output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(4_000, &["correlation_cycle"], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::CorrelationCycle {
                routes,
                period,
                threshold_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Correlation Cycle must preflight as one typed four-output launch");
        };
        assert_eq!((*period, *threshold_bits), (20, 9.0_f64.to_bits()));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ["real", "imag", "angle", "state"]
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "correlation_cycle_real",
                "correlation_cycle_imag",
                "correlation_cycle_angle",
                "correlation_cycle_state",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
        }));
    }

    #[test]
    fn correlation_cycle_period_sweep_forms_five_exact_typed_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            4_000,
            &[],
            &[],
            &[("correlation_cycle", periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::CorrelationCycle {
                routes,
                period,
                threshold_bits,
            } = launch
            else {
                panic!("every Correlation Cycle sweep must remain one typed four-output launch");
            };
            assert_eq!(
                (*period, *threshold_bits),
                (expected_period, 9.0_f64.to_bits())
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                ["real", "imag", "angle", "state"]
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("correlation_cycle_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
            }));
        }
    }

    #[test]
    fn cvi_default_output_forms_one_exact_typed_primary_launch() {
        assert_eq!(
            output_ids_for(CVI_ID),
            [None],
            "canonical CVI single-output receipt must stay unsuffixed"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[CVI_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Cvi { route, period }] = launches.as_slice() else {
            panic!("CVI must preflight as one typed resident primary launch");
        };
        assert_eq!(*period, 10);
        assert_eq!(route.node.indicator_id, CVI_ID);
        assert_eq!(route.output_id, "value");
        assert_eq!(route.node.column_name, CVI_ID);
        assert_eq!(route.node.stage, ClassicCudaStage::Base);
        assert_eq!(
            route.node.parameters,
            ClassicCudaParameters::Defaults {
                anchor: ClassicCudaAnchor::Resolved(10),
                require_period_invariant_kernel: false,
            }
        );
        assert_eq!(route.route, ClassicCudaResolvedRoute::Cvi { period: 10 });
    }

    #[test]
    fn cvi_period_sweep_forms_five_exact_typed_primary_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(CVI_ID, periods.clone())]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Cvi { route, period } = launch else {
                panic!("every CVI period must remain one typed resident primary launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(route.node.indicator_id, CVI_ID);
            assert_eq!(route.output_id, "value");
            assert_eq!(
                route.node.column_name,
                format!("{CVI_ID}_{expected_period}")
            );
            assert_eq!(route.node.stage, ClassicCudaStage::Extended);
            assert_eq!(
                route.node.parameters,
                ClassicCudaParameters::Swept {
                    period: expected_period,
                    overrides: vec![("period", expected_period as i64)],
                    anchor: ClassicCudaAnchor::Resolved(expected_period),
                }
            );
            assert_eq!(
                route.route,
                ClassicCudaResolvedRoute::Cvi {
                    period: expected_period,
                }
            );
        }
    }

    #[test]
    fn cvi_period_above_compiled_ring_bound_fails_before_launch() {
        let plan = build_exact_classic_cuda_plan(2_000, &[], &[], &[(CVI_ID, vec![513])]).unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan)
            .expect_err("CVI period 513 must fail before a CUDA session exists");
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].indicator_id, CVI_ID);
        assert_eq!(
            gaps[0].reason,
            ClassicCudaGapReason::MissingParameterContract
        );
        assert!(gaps[0].detail.contains("compiled CUDA ring bound 512"));
    }

    #[test]
    fn cyberpunk_value_trend_analyzer_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(CYBERPUNK_VALUE_TREND_ANALYZER_ID),
            CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS.map(Some),
            "canonical Cyberpunk Value Trend Analyzer output identities/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(1_000, &[CYBERPUNK_VALUE_TREND_ANALYZER_ID], &[], &[])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::CyberpunkValueTrendAnalyzer {
                routes,
                entry_level,
                exit_level,
            },
        ] = launches.as_slice()
        else {
            panic!("Cyberpunk Value Trend Analyzer must preflight as one typed six-output launch");
        };
        assert_eq!((*entry_level, *exit_level), (30, 75));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::CyberpunkValueTrendAnalyzer {
                        entry_level: 30,
                        exit_level: 75,
                    }
        }));
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "cyberpunk_value_trend_analyzer_value_trend",
                "cyberpunk_value_trend_analyzer_value_trend_lag",
                "cyberpunk_value_trend_analyzer_deviation_index",
                "cyberpunk_value_trend_analyzer_overbought_signal",
                "cyberpunk_value_trend_analyzer_buy_signal",
                "cyberpunk_value_trend_analyzer_sell_signal",
            ]
        );
    }

    #[test]
    fn cyberpunk_value_trend_analyzer_synthetic_period_sweep_fails_before_launch() {
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(CYBERPUNK_VALUE_TREND_ANALYZER_ID, vec![21])],
        )
        .unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan)
            .expect_err("Cyberpunk Value Trend Analyzer has no canonical period sweep");
        assert!(gaps.iter().all(|gap| {
            gap.indicator_id == CYBERPUNK_VALUE_TREND_ANALYZER_ID
                && gap.reason == ClassicCudaGapReason::MissingParameterContract
        }));
        for output_id in CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS {
            assert!(
                gaps.iter().any(|gap| {
                    gap.requested_output_id == Some(output_id)
                        && (gap.detail.contains("has no canonical period sweep")
                            || gap.detail.contains(
                                "canonical CPU sweep produced no explicit integer window overrides",
                            ))
                }),
                "synthetic sweep did not fail closed for `{output_id}`: {gaps:#?}"
            );
        }
    }

    #[test]
    fn cycle_channel_oscillator_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(CYCLE_CHANNEL_OSCILLATOR_ID),
            CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS.map(Some),
            "canonical Cycle Channel Oscillator output identities/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(1_000, &[CYCLE_CHANNEL_OSCILLATOR_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::CycleChannelOscillator {
                routes,
                short_cycle_length,
                medium_cycle_length,
                short_multiplier_bits,
                medium_multiplier_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Cycle Channel Oscillator must preflight as one typed two-output launch");
        };
        assert_eq!(
            (
                *short_cycle_length,
                *medium_cycle_length,
                *short_multiplier_bits,
                *medium_multiplier_bits,
            ),
            (10, 30, 1.0_f64.to_bits(), 3.0_f64.to_bits())
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "cycle_channel_oscillator_fast",
                "cycle_channel_oscillator_slow",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(30),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::CycleChannelOscillator {
                        short_cycle_length: 10,
                        medium_cycle_length: 30,
                        short_multiplier_bits: 1.0_f64.to_bits(),
                        medium_multiplier_bits: 3.0_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn cycle_channel_oscillator_ratio_sweep_forms_four_exact_typed_launches() {
        let periods = vec![21, 50, 100, 200];
        let expected_tuples = [(7_usize, 21_usize), (17, 50), (33, 100), (67, 200)];
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(CYCLE_CHANNEL_OSCILLATOR_ID, periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), expected_tuples.len());
        for ((launch, expected_period), (expected_short, expected_medium)) in
            launches.iter().zip(periods).zip(expected_tuples)
        {
            let ResolvedClassicCudaLaunch::CycleChannelOscillator {
                routes,
                short_cycle_length,
                medium_cycle_length,
                short_multiplier_bits,
                medium_multiplier_bits,
            } = launch
            else {
                panic!("every Cycle Channel Oscillator ratio tuple must remain one typed launch");
            };
            assert_eq!(
                (
                    *short_cycle_length,
                    *medium_cycle_length,
                    *short_multiplier_bits,
                    *medium_multiplier_bits,
                ),
                (
                    expected_short,
                    expected_medium,
                    1.0_f64.to_bits(),
                    3.0_f64.to_bits(),
                )
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![
                                ("short_cycle_length", expected_short as i64),
                                ("medium_cycle_length", expected_medium as i64),
                            ],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("cycle_channel_oscillator_{expected_period}_"))
            }));
        }
    }

    #[test]
    fn cycle_channel_oscillator_formula_excluded_period_fails_before_launch() {
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(CYCLE_CHANNEL_OSCILLATOR_ID, vec![7])],
        )
        .unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan)
            .expect_err("the formula-excluded Cycle Channel Oscillator point must fail closed");
        assert_eq!(gaps.len(), CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS.len());
        assert!(gaps.iter().all(|gap| {
            gap.indicator_id == CYCLE_CHANNEL_OSCILLATOR_ID
                && gap.reason == ClassicCudaGapReason::MissingParameterContract
                && gap.detail.contains("internal short delay zero")
        }));
    }

    #[test]
    fn daily_factor_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(DAILY_FACTOR_ID),
            DAILY_FACTOR_PRODUCTION_OUTPUT_IDS.map(Some),
            "reviewed admitted Daily Factor output identities/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DAILY_FACTOR_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::DailyFactor {
                routes,
                threshold_level_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Daily Factor must preflight as one typed two-output launch");
        };
        assert_eq!(*threshold_level_bits, 0.35_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DAILY_FACTOR_PRODUCTION_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["daily_factor_value", "daily_factor_signal"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::DailyFactor {
                        threshold_level_bits: 0.35_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn daily_factor_synthetic_period_sweep_fails_before_launch() {
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(DAILY_FACTOR_ID, vec![21])]).unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan).expect_err(
            "Daily Factor synthetic sweep must fail in pure preflight before engine \
                 allocation or launch",
        );
        let parameters = ClassicCudaParameters::Swept {
            period: 21,
            overrides: Vec::new(),
            anchor: ClassicCudaAnchor::Resolved(21),
        };
        assert_eq!(
            gaps,
            vec![
                ClassicCudaGap {
                    column_name: "daily_factor_21_value".to_string(),
                    indicator_id: DAILY_FACTOR_ID,
                    requested_output_id: Some("value"),
                    parameters: parameters.clone(),
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: "canonical CPU sweep produced no explicit integer window overrides"
                        .to_string(),
                },
                ClassicCudaGap {
                    column_name: "daily_factor_21_value".to_string(),
                    indicator_id: DAILY_FACTOR_ID,
                    requested_output_id: Some("value"),
                    parameters: parameters.clone(),
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: format!("{DAILY_FACTOR_ID}: formula has no canonical period sweep"),
                },
                ClassicCudaGap {
                    column_name: "daily_factor_21_signal".to_string(),
                    indicator_id: DAILY_FACTOR_ID,
                    requested_output_id: Some("signal"),
                    parameters: parameters.clone(),
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: "canonical CPU sweep produced no explicit integer window overrides"
                        .to_string(),
                },
                ClassicCudaGap {
                    column_name: "daily_factor_21_signal".to_string(),
                    indicator_id: DAILY_FACTOR_ID,
                    requested_output_id: Some("signal"),
                    parameters,
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: format!("{DAILY_FACTOR_ID}: formula has no canonical period sweep"),
                },
            ],
            "Daily Factor must preserve both independent fail-closed receipts for each admitted \
             output without constructing an executable launch"
        );
    }

    #[test]
    fn damiani_volatmeter_default_outputs_form_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(DAMIANI_VOLATMETER_ID),
            DAMIANI_VOLATMETER_OUTPUT_IDS.map(Some),
            "canonical Damiani Volatmeter output identities/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(3_200, &[DAMIANI_VOLATMETER_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::DamianiVolatmeter {
                routes,
                vis_atr,
                vis_std,
                sed_atr,
                sed_std,
                threshold_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Damiani Volatmeter must preflight as one typed two-output default launch");
        };
        assert_eq!(
            (*vis_atr, *vis_std, *sed_atr, *sed_std, *threshold_bits),
            (13, 20, 40, 100, 1.4_f64.to_bits())
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DAMIANI_VOLATMETER_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["damiani_volatmeter_vol", "damiani_volatmeter_anti"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::DamianiVolatmeter {
                        vis_atr: 13,
                        vis_std: 20,
                        sed_atr: 40,
                        sed_std: 100,
                        threshold_bits: 1.4_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn damiani_volatmeter_synthetic_period_sweep_fails_before_launch() {
        let plan =
            build_exact_classic_cuda_plan(3_200, &[], &[], &[(DAMIANI_VOLATMETER_ID, vec![21])])
                .unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan).expect_err(
            "Damiani Volatmeter synthetic sweep must fail in pure preflight before engine \
             allocation or launch",
        );
        let parameters = ClassicCudaParameters::Swept {
            period: 21,
            overrides: Vec::new(),
            anchor: ClassicCudaAnchor::Resolved(21),
        };
        assert_eq!(
            gaps,
            vec![
                ClassicCudaGap {
                    column_name: "damiani_volatmeter_21_vol".to_string(),
                    indicator_id: DAMIANI_VOLATMETER_ID,
                    requested_output_id: Some("vol"),
                    parameters: parameters.clone(),
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: "canonical CPU sweep produced no explicit integer window overrides"
                        .to_string(),
                },
                ClassicCudaGap {
                    column_name: "damiani_volatmeter_21_vol".to_string(),
                    indicator_id: DAMIANI_VOLATMETER_ID,
                    requested_output_id: Some("vol"),
                    parameters: parameters.clone(),
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: format!(
                        "{DAMIANI_VOLATMETER_ID}: formula has no canonical period sweep"
                    ),
                },
                ClassicCudaGap {
                    column_name: "damiani_volatmeter_21_anti".to_string(),
                    indicator_id: DAMIANI_VOLATMETER_ID,
                    requested_output_id: Some("anti"),
                    parameters: parameters.clone(),
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: "canonical CPU sweep produced no explicit integer window overrides"
                        .to_string(),
                },
                ClassicCudaGap {
                    column_name: "damiani_volatmeter_21_anti".to_string(),
                    indicator_id: DAMIANI_VOLATMETER_ID,
                    requested_output_id: Some("anti"),
                    parameters,
                    reason: ClassicCudaGapReason::MissingParameterContract,
                    detail: format!(
                        "{DAMIANI_VOLATMETER_ID}: formula has no canonical period sweep"
                    ),
                },
            ],
            "Damiani Volatmeter must preserve both independent fail-closed receipts for each \
             canonical output without constructing an executable launch"
        );
    }

    #[test]
    fn di_default_outputs_form_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(DI_ID),
            DI_OUTPUT_IDS.map(Some),
            "canonical DI plus/minus identity and order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DI_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Di { routes, period }] = launches.as_slice() else {
            panic!("DI must preflight as one typed two-output default launch");
        };
        assert_eq!(*period, 14);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DI_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["di_plus", "di_minus"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(14),
                        require_period_invariant_kernel: false,
                    }
                && route.route == ClassicCudaResolvedRoute::Di { period: 14 }
        }));
    }

    #[test]
    fn di_period_sweep_forms_five_exact_typed_pair_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(DI_ID, periods.clone())]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Di { routes, period } = launch else {
                panic!("every DI period must remain one typed two-output launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DI_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("di_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Di {
                            period: expected_period,
                        }
            }));
        }
    }

    #[test]
    fn didi_index_default_outputs_form_one_exact_typed_quad_launch() {
        assert_eq!(
            output_ids_for(DIDI_INDEX_ID),
            DIDI_INDEX_OUTPUT_IDS.map(Some),
            "canonical Didi Index output identity and order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DIDI_INDEX_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::DidiIndex {
                routes,
                short_length,
                medium_length,
                long_length,
            },
        ] = launches.as_slice()
        else {
            panic!("Didi Index must preflight as one typed four-output default launch");
        };
        assert_eq!((*short_length, *medium_length, *long_length), (3, 8, 20));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DIDI_INDEX_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "didi_index_short",
                "didi_index_long",
                "didi_index_crossover",
                "didi_index_crossunder",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::DidiIndex {
                        short_length: 3,
                        medium_length: 8,
                        long_length: 20,
                    }
        }));
    }

    #[test]
    fn didi_index_registry_ratio_sweep_forms_five_exact_typed_quad_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let expected_tuples = [
            (1, 3, 7),
            (3, 8, 21),
            (8, 20, 50),
            (15, 40, 100),
            (30, 80, 200),
        ];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(DIDI_INDEX_ID, periods.clone())])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for ((launch, expected_period), expected_tuple) in
            launches.iter().zip(periods).zip(expected_tuples)
        {
            let ResolvedClassicCudaLaunch::DidiIndex {
                routes,
                short_length,
                medium_length,
                long_length,
            } = launch
            else {
                panic!("every Didi Index ratio tuple must remain one typed four-output launch");
            };
            assert_eq!(
                (*short_length, *medium_length, *long_length),
                expected_tuple
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DIDI_INDEX_OUTPUT_IDS
            );
            let expected_overrides = vec![
                ("short_length", expected_tuple.0 as i64),
                ("medium_length", expected_tuple.1 as i64),
                ("long_length", expected_tuple.2 as i64),
            ];
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("didi_index_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: expected_overrides.clone(),
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::DidiIndex {
                            short_length: expected_tuple.0,
                            medium_length: expected_tuple.1,
                            long_length: expected_tuple.2,
                        }
            }));
        }
    }

    #[test]
    fn directional_imbalance_index_default_outputs_form_one_exact_typed_six_launch() {
        assert_eq!(
            output_ids_for(DIRECTIONAL_IMBALANCE_INDEX_ID),
            DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS.map(Some),
            "canonical Directional Imbalance Index output identity/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(1_000, &[DIRECTIONAL_IMBALANCE_INDEX_ID], &[], &[])
                .unwrap();
        assert_eq!(
            plan.nodes.len(),
            6,
            "canonical registry repair intentionally replaces one aliased base column with six"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::DirectionalImbalanceIndex {
                routes,
                length,
                period,
            },
        ] = launches.as_slice()
        else {
            panic!("Directional Imbalance Index must preflight as one typed six-output launch");
        };
        assert_eq!((*length, *period), (10, 70));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "directional_imbalance_index_up",
                "directional_imbalance_index_down",
                "directional_imbalance_index_bulls",
                "directional_imbalance_index_bears",
                "directional_imbalance_index_upper",
                "directional_imbalance_index_lower",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(70),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::DirectionalImbalanceIndex {
                        length: 10,
                        period: 70,
                    }
        }));
    }

    #[test]
    fn directional_imbalance_index_period_sweep_forms_thirty_canonical_columns_in_five_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(DIRECTIONAL_IMBALANCE_INDEX_ID, periods.clone())],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            30,
            "five exact periods times six canonical outputs is the intentional dataset identity"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::DirectionalImbalanceIndex {
                routes,
                length,
                period,
            } = launch
            else {
                panic!("every period must remain one typed six-output launch");
            };
            assert_eq!((*length, *period), (10, expected_period));
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
            );
            assert!(routes.iter().all(|route| {
                route.node.stage == ClassicCudaStage::Extended
                    && route
                        .node
                        .column_name
                        .starts_with(&format!("directional_imbalance_index_{expected_period}_"))
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::DirectionalImbalanceIndex {
                            length: 10,
                            period: expected_period,
                        }
            }));
        }
    }

    #[test]
    fn disparity_index_default_output_forms_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(DISPARITY_INDEX_ID),
            [None],
            "canonical Disparity Index single-output receipt must stay unsuffixed"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DISPARITY_INDEX_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::DisparityIndex {
                route,
                ema_period,
                lookback_period,
                smoothing_period,
                smoothing_is_sma,
            },
        ] = launches.as_slice()
        else {
            panic!("Disparity Index must preflight as one typed resident launch");
        };
        assert_eq!(
            (
                *ema_period,
                *lookback_period,
                *smoothing_period,
                *smoothing_is_sma,
            ),
            (14, 14, 9, false)
        );
        assert_eq!(route.node.indicator_id, DISPARITY_INDEX_ID);
        assert_eq!(route.output_id, DISPARITY_INDEX_OUTPUT_ID);
        assert_eq!(route.node.column_name, DISPARITY_INDEX_ID);
        assert_eq!(route.node.stage, ClassicCudaStage::Base);
        assert_eq!(
            route.node.parameters,
            ClassicCudaParameters::Defaults {
                anchor: ClassicCudaAnchor::Resolved(14),
                require_period_invariant_kernel: false,
            }
        );
        assert_eq!(
            route.route,
            ClassicCudaResolvedRoute::DisparityIndex {
                ema_period: 14,
                lookback_period: 14,
                smoothing_period: 9,
                smoothing_is_sma: false,
            }
        );
    }

    #[test]
    fn disparity_index_lookback_sweep_forms_five_exact_typed_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(DISPARITY_INDEX_ID, periods.clone())],
        )
        .unwrap();
        assert_eq!(plan.nodes.len(), periods.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_lookback) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::DisparityIndex {
                route,
                ema_period,
                lookback_period,
                smoothing_period,
                smoothing_is_sma,
            } = launch
            else {
                panic!("every Disparity Index lookback must remain one typed resident launch");
            };
            assert_eq!(
                (
                    *ema_period,
                    *lookback_period,
                    *smoothing_period,
                    *smoothing_is_sma,
                ),
                (14, expected_lookback, 9, false)
            );
            assert_eq!(route.node.indicator_id, DISPARITY_INDEX_ID);
            assert_eq!(route.output_id, DISPARITY_INDEX_OUTPUT_ID);
            assert_eq!(
                route.node.column_name,
                format!("{DISPARITY_INDEX_ID}_{expected_lookback}")
            );
            assert_eq!(route.node.stage, ClassicCudaStage::Extended);
            assert_eq!(
                route.node.parameters,
                ClassicCudaParameters::Swept {
                    period: expected_lookback,
                    overrides: vec![("lookback_period", expected_lookback as i64)],
                    anchor: ClassicCudaAnchor::Resolved(expected_lookback),
                }
            );
            assert_eq!(
                route.route,
                ClassicCudaResolvedRoute::DisparityIndex {
                    ema_period: 14,
                    lookback_period: expected_lookback,
                    smoothing_period: 9,
                    smoothing_is_sma: false,
                }
            );
        }
    }

    #[test]
    fn dm_default_outputs_form_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(DM_ID),
            DM_OUTPUT_IDS.map(Some),
            "canonical DM plus/minus identity and order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DM_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Dm { routes, period }] = launches.as_slice() else {
            panic!("DM must preflight as one typed two-output default launch");
        };
        assert_eq!(*period, 14);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DM_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["dm_plus", "dm_minus"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(14),
                        require_period_invariant_kernel: false,
                    }
                && route.route == ClassicCudaResolvedRoute::Dm { period: 14 }
        }));
    }

    #[test]
    fn dm_period_sweep_forms_five_exact_typed_pair_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(DM_ID, periods.clone())]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Dm { routes, period } = launch else {
                panic!("every DM period must remain one typed two-output launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DM_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("dm_{expected_period}_{}", DM_OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Dm {
                            period: expected_period,
                        }
            }));
        }
    }

    #[test]
    fn donchian_default_outputs_form_one_exact_typed_triple_launch() {
        assert_eq!(
            output_ids_for(DONCHIAN_ID),
            DONCHIAN_OUTPUT_IDS.map(Some),
            "canonical Donchian upper/middle/lower identity and order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DONCHIAN_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Donchian { routes, period }] = launches.as_slice() else {
            panic!("Donchian must preflight as one typed three-output default launch");
        };
        assert_eq!(*period, 20);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DONCHIAN_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["donchian_upper", "donchian_middle", "donchian_lower"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
                && route.route == ClassicCudaResolvedRoute::Donchian { period: 20 }
        }));
    }

    #[test]
    fn donchian_period_sweep_forms_five_exact_typed_triple_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(1_000, &[], &[], &[(DONCHIAN_ID, periods.clone())])
                .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Donchian { routes, period } = launch else {
                panic!("every Donchian period must remain one typed three-output launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DONCHIAN_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("donchian_{expected_period}_{}", DONCHIAN_OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Donchian {
                            period: expected_period,
                        }
            }));
        }
    }

    #[test]
    fn dual_ulcer_index_default_outputs_form_one_exact_typed_triple_launch() {
        assert_eq!(
            output_ids_for(DUAL_ULCER_INDEX_ID),
            DUAL_ULCER_INDEX_OUTPUT_IDS.map(Some),
            "canonical Dual Ulcer Index identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(1_000, &[DUAL_ULCER_INDEX_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::DualUlcerIndex {
                routes,
                period,
                auto_threshold,
                threshold_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("Dual Ulcer Index must preflight as one typed three-output default launch");
        };
        assert_eq!(*period, 5);
        assert!(*auto_threshold);
        assert_eq!(*threshold_bits, 0.1_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DUAL_ULCER_INDEX_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "dual_ulcer_index_long_ulcer",
                "dual_ulcer_index_short_ulcer",
                "dual_ulcer_index_threshold",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(5),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::DualUlcerIndex {
                        period: 5,
                        auto_threshold: true,
                        threshold_bits: 0.1_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn dual_ulcer_index_period_sweep_forms_five_exact_typed_triple_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            1_000,
            &[],
            &[],
            &[(DUAL_ULCER_INDEX_ID, periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::DualUlcerIndex {
                routes,
                period,
                auto_threshold,
                threshold_bits,
            } = launch
            else {
                panic!("every Dual Ulcer Index period must remain one typed triple launch");
            };
            assert_eq!(*period, expected_period);
            assert!(*auto_threshold);
            assert_eq!(*threshold_bits, 0.1_f64.to_bits());
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DUAL_ULCER_INDEX_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "dual_ulcer_index_{expected_period}_{}",
                            DUAL_ULCER_INDEX_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::DualUlcerIndex {
                            period: expected_period,
                            auto_threshold: true,
                            threshold_bits: 0.1_f64.to_bits(),
                        }
            }));
        }
    }

    #[test]
    fn dvdiqqe_default_outputs_form_one_exact_typed_four_output_launch() {
        assert_eq!(
            output_ids_for(DVDIQQE_ID),
            DVDIQQE_OUTPUT_IDS.map(Some),
            "canonical DVDIQQE identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[DVDIQQE_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Dvdiqqe {
                routes,
                period,
                smoothing_period,
                fast_multiplier_bits,
                slow_multiplier_bits,
                use_tick_only,
                dynamic_center,
                tick_size_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("DVDIQQE must preflight as one typed four-output default launch");
        };
        assert_eq!(*period, 13);
        assert_eq!(*smoothing_period, 6);
        assert_eq!(*fast_multiplier_bits, 2.618_f64.to_bits());
        assert_eq!(*slow_multiplier_bits, 4.236_f64.to_bits());
        assert!(!*use_tick_only);
        assert!(*dynamic_center);
        assert_eq!(*tick_size_bits, 0.01_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            DVDIQQE_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "dvdiqqe_dvdi",
                "dvdiqqe_fast_tl",
                "dvdiqqe_slow_tl",
                "dvdiqqe_center_line",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(13),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::Dvdiqqe {
                        period: 13,
                        smoothing_period: 6,
                        fast_multiplier_bits: 2.618_f64.to_bits(),
                        slow_multiplier_bits: 4.236_f64.to_bits(),
                        use_tick_only: false,
                        dynamic_center: true,
                        tick_size_bits: 0.01_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn dvdiqqe_period_sweep_forms_five_exact_typed_four_output_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(2_000, &[], &[], &[(DVDIQQE_ID, periods.clone())])
            .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Dvdiqqe {
                routes,
                period,
                smoothing_period,
                fast_multiplier_bits,
                slow_multiplier_bits,
                use_tick_only,
                dynamic_center,
                tick_size_bits,
            } = launch
            else {
                panic!("every DVDIQQE period must remain one typed four-output launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(*smoothing_period, 6);
            assert_eq!(*fast_multiplier_bits, 2.618_f64.to_bits());
            assert_eq!(*slow_multiplier_bits, 4.236_f64.to_bits());
            assert!(!*use_tick_only);
            assert!(*dynamic_center);
            assert_eq!(*tick_size_bits, 0.01_f64.to_bits());
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                DVDIQQE_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("dvdiqqe_{expected_period}_{}", DVDIQQE_OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Dvdiqqe {
                            period: expected_period,
                            smoothing_period: 6,
                            fast_multiplier_bits: 2.618_f64.to_bits(),
                            slow_multiplier_bits: 4.236_f64.to_bits(),
                            use_tick_only: false,
                            dynamic_center: true,
                            tick_size_bits: 0.01_f64.to_bits(),
                        }
            }));
        }
    }

    #[test]
    fn ehlers_autocorrelation_periodogram_default_forms_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(EHLERS_AUTOCORRELATION_PERIODOGRAM_ID),
            EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS.map(Some),
            "canonical Ehlers Autocorrelation Periodogram identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[EHLERS_AUTOCORRELATION_PERIODOGRAM_ID],
            &[],
            &[],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::EhlersAutocorrelationPeriodogram {
                routes,
                min_period,
                max_period,
                avg_length,
                enhance,
            },
        ] = launches.as_slice()
        else {
            panic!("the default periodogram pair must form one typed resident launch");
        };
        assert_eq!(
            (*min_period, *max_period, *avg_length, *enhance),
            (8, 48, 3, true)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "ehlers_autocorrelation_periodogram_dominant_cycle",
                "ehlers_autocorrelation_periodogram_normalized_power",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(48),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::EhlersAutocorrelationPeriodogram {
                        min_period: 8,
                        max_period: 48,
                        avg_length: 3,
                        enhance: true,
                    }
        }));
    }

    #[test]
    fn ehlers_autocorrelation_periodogram_valid_registry_ratios_form_four_exact_pair_launches() {
        let expected = [
            (21, 4, 21, 1),
            (50, 8, 50, 3),
            (100, 17, 100, 6),
            (200, 33, 200, 13),
        ];
        let periods = expected
            .iter()
            .map(|(period, ..)| *period)
            .collect::<Vec<_>>();
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(EHLERS_AUTOCORRELATION_PERIODOGRAM_ID, periods)],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), expected.len());
        for (launch, (anchor, expected_min, expected_max, expected_avg)) in
            launches.iter().zip(expected)
        {
            let ResolvedClassicCudaLaunch::EhlersAutocorrelationPeriodogram {
                routes,
                min_period,
                max_period,
                avg_length,
                enhance,
            } = launch
            else {
                panic!("every admitted periodogram ratio must remain one typed pair launch");
            };
            assert_eq!(
                (*min_period, *max_period, *avg_length, *enhance),
                (expected_min, expected_max, expected_avg, true)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
            );
            let expected_overrides = vec![
                ("min_period", expected_min as i64),
                ("max_period", expected_max as i64),
                ("avg_length", expected_avg as i64),
            ];
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "ehlers_autocorrelation_periodogram_{anchor}_{}",
                            EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: anchor,
                            overrides: expected_overrides.clone(),
                            anchor: ClassicCudaAnchor::Resolved(anchor),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EhlersAutocorrelationPeriodogram {
                            min_period: expected_min,
                            max_period: expected_max,
                            avg_length: expected_avg,
                            enhance: true,
                        }
            }));
        }
    }

    #[test]
    fn ehlers_data_sampling_rsi_default_forms_one_exact_typed_admitted_pair_launch() {
        assert_eq!(
            output_ids_for(EHLERS_DATA_SAMPLING_RSI_ID),
            EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS.map(Some),
            "canonical EDSRSI admitted identity/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(2_000, &[EHLERS_DATA_SAMPLING_RSI_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::EhlersDataSamplingRsi { routes, length }] =
            launches.as_slice()
        else {
            panic!("the default EDSRSI admitted pair must form one typed resident launch");
        };
        assert_eq!(*length, 14);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "ehlers_data_sampling_relative_strength_indicator_ds_rsi",
                "ehlers_data_sampling_relative_strength_indicator_signal",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(14),
                        require_period_invariant_kernel: false,
                    }
                && route.route == ClassicCudaResolvedRoute::EhlersDataSamplingRsi { length: 14 }
        }));
    }

    #[test]
    fn ehlers_data_sampling_rsi_length_sweep_forms_five_exact_typed_pair_launches() {
        let periods = crate::core::hpc_ta::ALT_PERIODS.to_vec();
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(EHLERS_DATA_SAMPLING_RSI_ID, periods.clone())],
        )
        .unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_length) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::EhlersDataSamplingRsi { routes, length } = launch else {
                panic!("every EDSRSI length sweep must remain one typed admitted-pair launch");
            };
            assert_eq!(*length, expected_length);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "ehlers_data_sampling_relative_strength_indicator_{expected_length}_{}",
                            EHLERS_DATA_SAMPLING_RSI_PRODUCTION_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_length,
                            overrides: vec![("length", expected_length as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_length),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EhlersDataSamplingRsi {
                            length: expected_length,
                        }
            }));
        }
    }

    #[test]
    fn ehlers_linear_extrapolation_predictor_default_forms_one_exact_typed_quint_launch() {
        assert_eq!(
            output_ids_for(EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID),
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS.map(Some),
            "canonical ELEP output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS.len()
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::EhlersLinearExtrapolationPredictor {
                routes,
                high_pass_length,
                low_pass_length,
                gain_bits,
                bars_forward,
                signal_mode,
            },
        ] = launches.as_slice()
        else {
            panic!("the default ELEP quint must form one typed resident launch");
        };
        assert_eq!(
            (
                *high_pass_length,
                *low_pass_length,
                *gain_bits,
                *bars_forward,
                *signal_mode,
            ),
            (125, 12, 0.7_f64.to_bits(), 5, 0)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "ehlers_linear_extrapolation_predictor_prediction",
                "ehlers_linear_extrapolation_predictor_filter",
                "ehlers_linear_extrapolation_predictor_state",
                "ehlers_linear_extrapolation_predictor_go_long",
                "ehlers_linear_extrapolation_predictor_go_short",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(125),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::EhlersLinearExtrapolationPredictor {
                        high_pass_length: 125,
                        low_pass_length: 12,
                        gain_bits: 0.7_f64.to_bits(),
                        bars_forward: 5,
                        signal_mode: 0,
                    }
        }));
    }

    #[test]
    fn ehlers_linear_extrapolation_predictor_ratios_form_five_exact_quint_launches() {
        let expected = [
            (7, 7, 1),
            (21, 21, 2),
            (50, 50, 5),
            (100, 100, 10),
            (200, 200, 19),
        ];
        let periods = expected
            .iter()
            .map(|(anchor, ..)| *anchor)
            .collect::<Vec<_>>();
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_ID, periods)],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            expected.len() * EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS.len(),
            "ELEP must retain all 25 admitted sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), expected.len());
        for (launch, (anchor, expected_high_pass, expected_low_pass)) in
            launches.iter().zip(expected)
        {
            let ResolvedClassicCudaLaunch::EhlersLinearExtrapolationPredictor {
                routes,
                high_pass_length,
                low_pass_length,
                gain_bits,
                bars_forward,
                signal_mode,
            } = launch
            else {
                panic!("every admitted ELEP ratio must remain one typed quint launch");
            };
            assert_eq!(
                (
                    *high_pass_length,
                    *low_pass_length,
                    *gain_bits,
                    *bars_forward,
                    *signal_mode,
                ),
                (
                    expected_high_pass,
                    expected_low_pass,
                    0.7_f64.to_bits(),
                    5,
                    0,
                )
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
            );
            let expected_overrides = vec![
                ("high_pass_length", expected_high_pass as i64),
                ("low_pass_length", expected_low_pass as i64),
            ];
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "ehlers_linear_extrapolation_predictor_{anchor}_{}",
                            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: anchor,
                            overrides: expected_overrides.clone(),
                            anchor: ClassicCudaAnchor::Resolved(anchor),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EhlersLinearExtrapolationPredictor {
                            high_pass_length: expected_high_pass,
                            low_pass_length: expected_low_pass,
                            gain_bits: 0.7_f64.to_bits(),
                            bars_forward: 5,
                            signal_mode: 0,
                        }
            }));
        }
    }

    #[test]
    fn ehlers_undersampled_double_moving_average_default_forms_one_exact_typed_pair_launch() {
        assert_eq!(
            output_ids_for(EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID),
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS.map(Some),
            "canonical EUDMA output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS.len()
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::EhlersUndersampledDoubleMovingAverage {
                routes,
                fast_length,
                slow_length,
                sample_length,
            },
        ] = launches.as_slice()
        else {
            panic!("the default EUDMA pair must form one typed resident launch");
        };
        assert_eq!((*fast_length, *slow_length, *sample_length), (6, 12, 5));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "ehlers_undersampled_double_moving_average_fast",
                "ehlers_undersampled_double_moving_average_slow",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(12),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::EhlersUndersampledDoubleMovingAverage {
                        fast_length: 6,
                        slow_length: 12,
                        sample_length: 5,
                    }
        }));
    }

    #[test]
    fn ehlers_undersampled_double_moving_average_ratios_form_five_exact_pair_launches() {
        let expected = [
            (7, 4, 7, 3),
            (21, 11, 21, 9),
            (50, 25, 50, 21),
            (100, 50, 100, 42),
            (200, 100, 200, 83),
        ];
        let periods = expected
            .iter()
            .map(|(anchor, ..)| *anchor)
            .collect::<Vec<_>>();
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_ID, periods)],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            expected.len() * EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS.len(),
            "EUDMA must retain all ten admitted sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), expected.len());
        for (launch, (anchor, expected_fast, expected_slow, expected_sample)) in
            launches.iter().zip(expected)
        {
            let ResolvedClassicCudaLaunch::EhlersUndersampledDoubleMovingAverage {
                routes,
                fast_length,
                slow_length,
                sample_length,
            } = launch
            else {
                panic!("every admitted EUDMA ratio must remain one typed pair launch");
            };
            assert_eq!(
                (*fast_length, *slow_length, *sample_length),
                (expected_fast, expected_slow, expected_sample)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
            );
            let expected_overrides = vec![
                ("fast_length", expected_fast as i64),
                ("slow_length", expected_slow as i64),
                ("sample_length", expected_sample as i64),
            ];
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "ehlers_undersampled_double_moving_average_{anchor}_{}",
                            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: anchor,
                            overrides: expected_overrides.clone(),
                            anchor: ClassicCudaAnchor::Resolved(anchor),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EhlersUndersampledDoubleMovingAverage {
                            fast_length: expected_fast,
                            slow_length: expected_slow,
                            sample_length: expected_sample,
                        }
            }));
        }
    }

    #[test]
    fn ema_deviation_corrected_t3_default_forms_one_exact_period_ten_pair_launch() {
        assert_eq!(
            output_ids_for(EMA_DEVIATION_CORRECTED_T3_ID),
            EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.map(Some),
            "canonical EDCT3 output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[EMA_DEVIATION_CORRECTED_T3_ID], &[], &[])
            .unwrap();
        assert_eq!(
            plan.nodes.len(),
            EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.len()
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::EmaDeviationCorrectedT3 {
                routes,
                period,
                hot_bits,
                t3_mode,
            },
        ] = launches.as_slice()
        else {
            panic!("the default EDCT3 pair must form one typed resident launch");
        };
        assert_eq!((*period, *hot_bits, *t3_mode), (10, 0.7_f64.to_bits(), 0));
        assert_ne!(
            *period, 14,
            "the fabricated generic-MA base must not survive"
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "ema_deviation_corrected_t3_corrected",
                "ema_deviation_corrected_t3_t3",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(10),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::EmaDeviationCorrectedT3 {
                        period: 10,
                        hot_bits: 0.7_f64.to_bits(),
                        t3_mode: 0,
                    }
        }));
    }

    #[test]
    fn ema_deviation_corrected_t3_period_sweeps_form_five_exact_pair_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(EMA_DEVIATION_CORRECTED_T3_ID, periods.clone())],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            periods.len() * EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS.len(),
            "EDCT3 must retain all ten admitted sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::EmaDeviationCorrectedT3 {
                routes,
                period,
                hot_bits,
                t3_mode,
            } = launch
            else {
                panic!("every admitted EDCT3 period must remain one typed pair launch");
            };
            assert_eq!(
                (*period, *hot_bits, *t3_mode),
                (expected_period, 0.7_f64.to_bits(), 0)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "ema_deviation_corrected_t3_{expected_period}_{}",
                            EMA_DEVIATION_CORRECTED_T3_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EmaDeviationCorrectedT3 {
                            period: expected_period,
                            hot_bits: 0.7_f64.to_bits(),
                            t3_mode: 0,
                        }
            }));
        }
    }

    #[test]
    fn emd_default_forms_one_exact_period_twenty_triple_launch() {
        assert_eq!(
            output_ids_for(EMD_ID),
            EMD_OUTPUT_IDS.map(Some),
            "canonical EMD output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[EMD_ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), EMD_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Emd {
                routes,
                period,
                delta_bits,
                fraction_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("the default EMD outputs must form one typed resident launch");
        };
        assert_eq!(
            (*period, *delta_bits, *fraction_bits),
            (20, 0.5_f64.to_bits(), 0.1_f64.to_bits())
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EMD_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["emd_upperband", "emd_middleband", "emd_lowerband"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::Emd {
                        period: 20,
                        delta_bits: 0.5_f64.to_bits(),
                        fraction_bits: 0.1_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn emd_period_sweeps_form_five_exact_triple_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(EMD_ID, periods.clone())]).unwrap();
        assert_eq!(
            plan.nodes.len(),
            periods.len() * EMD_OUTPUT_IDS.len(),
            "EMD must retain all fifteen admitted sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Emd {
                routes,
                period,
                delta_bits,
                fraction_bits,
            } = launch
            else {
                panic!("every admitted EMD period must remain one typed triple launch");
            };
            assert_eq!(
                (*period, *delta_bits, *fraction_bits),
                (expected_period, 0.5_f64.to_bits(), 0.1_f64.to_bits())
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EMD_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("emd_{expected_period}_{}", EMD_OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Emd {
                            period: expected_period,
                            delta_bits: 0.5_f64.to_bits(),
                            fraction_bits: 0.1_f64.to_bits(),
                        }
            }));
        }
    }

    #[test]
    fn emd_trend_default_forms_one_exact_length_twenty_eight_quad_launch() {
        const ID: &str = "emd_trend";
        const OUTPUT_IDS: [&str; 4] = ["direction", "average", "upper", "lower"];
        assert_eq!(
            output_ids_for(ID),
            OUTPUT_IDS.map(Some),
            "canonical EMD Trend output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::EmdTrend {
                routes,
                length,
                mult_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("the default EMD Trend outputs must form one typed resident launch");
        };
        assert_eq!((*length, *mult_bits), (28, 1.0_f64.to_bits()));
        assert_eq!(routes.each_ref().map(|route| route.output_id), OUTPUT_IDS);
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "emd_trend_direction",
                "emd_trend_average",
                "emd_trend_upper",
                "emd_trend_lower",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(28),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::EmdTrend {
                        length: 28,
                        mult_bits: 1.0_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn emd_trend_length_sweeps_form_five_exact_quad_launches() {
        const ID: &str = "emd_trend";
        const OUTPUT_IDS: [&str; 4] = ["direction", "average", "upper", "lower"];
        let lengths = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(ID, lengths.clone())]).unwrap();
        assert_eq!(
            plan.nodes.len(),
            lengths.len() * OUTPUT_IDS.len(),
            "EMD Trend must retain all twenty admitted length-sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), lengths.len());
        for (launch, expected_length) in launches.iter().zip(lengths) {
            let ResolvedClassicCudaLaunch::EmdTrend {
                routes,
                length,
                mult_bits,
            } = launch
            else {
                panic!("every admitted EMD Trend length must remain one typed quad launch");
            };
            assert_eq!((*length, *mult_bits), (expected_length, 1.0_f64.to_bits()));
            assert_eq!(routes.each_ref().map(|route| route.output_id), OUTPUT_IDS);
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("emd_trend_{expected_length}_{}", OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_length,
                            overrides: vec![("length", expected_length as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_length),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EmdTrend {
                            length: expected_length,
                            mult_bits: 1.0_f64.to_bits(),
                        }
            }));
        }
    }

    #[test]
    fn eri_default_forms_one_exact_period_thirteen_pair_launch() {
        assert_eq!(
            output_ids_for(ERI_ID),
            ERI_OUTPUT_IDS.map(Some),
            "canonical ERI output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[ERI_ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), ERI_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Eri { routes, period }] = launches.as_slice() else {
            panic!("the default ERI outputs must form one typed resident launch");
        };
        assert_eq!(*period, 13);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            ERI_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["eri_bull", "eri_bear"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(13),
                        require_period_invariant_kernel: false,
                    }
                && route.route == ClassicCudaResolvedRoute::Eri { period: 13 }
        }));
    }

    #[test]
    fn eri_period_sweeps_form_five_exact_pair_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(ERI_ID, periods.clone())]).unwrap();
        assert_eq!(
            plan.nodes.len(),
            periods.len() * ERI_OUTPUT_IDS.len(),
            "ERI must retain all ten admitted period-sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Eri { routes, period } = launch else {
                panic!("every admitted ERI period must remain one typed pair launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                ERI_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("eri_{expected_period}_{}", ERI_OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Eri {
                            period: expected_period,
                        }
            }));
        }
    }

    #[test]
    fn evasive_supertrend_default_forms_one_exact_period_ten_quad_launch() {
        assert_eq!(
            output_ids_for(EVASIVE_SUPERTREND_ID),
            EVASIVE_SUPERTREND_OUTPUT_IDS.map(Some),
            "canonical Evasive Supertrend output identity/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(2_000, &[EVASIVE_SUPERTREND_ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), EVASIVE_SUPERTREND_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::EvasiveSupertrend {
                routes,
                atr_length,
                base_multiplier_bits,
                noise_threshold_bits,
                expansion_alpha_bits,
            },
        ] = launches.as_slice()
        else {
            panic!("the default Evasive Supertrend outputs must form one typed resident launch");
        };
        assert_eq!(*atr_length, 10);
        assert_eq!(*base_multiplier_bits, 3.0_f64.to_bits());
        assert_eq!(*noise_threshold_bits, 1.0_f64.to_bits());
        assert_eq!(*expansion_alpha_bits, 0.5_f64.to_bits());
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            EVASIVE_SUPERTREND_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "evasive_supertrend_band",
                "evasive_supertrend_state",
                "evasive_supertrend_noisy",
                "evasive_supertrend_changed",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(10),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::EvasiveSupertrend {
                        atr_length: 10,
                        base_multiplier_bits: 3.0_f64.to_bits(),
                        noise_threshold_bits: 1.0_f64.to_bits(),
                        expansion_alpha_bits: 0.5_f64.to_bits(),
                    }
        }));
    }

    #[test]
    fn evasive_supertrend_atr_length_sweeps_form_five_exact_quad_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(EVASIVE_SUPERTREND_ID, periods.clone())],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            periods.len() * EVASIVE_SUPERTREND_OUTPUT_IDS.len(),
            "Evasive Supertrend must retain all twenty admitted sweep receipts"
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());
        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::EvasiveSupertrend {
                routes,
                atr_length,
                base_multiplier_bits,
                noise_threshold_bits,
                expansion_alpha_bits,
            } = launch
            else {
                panic!(
                    "every admitted Evasive Supertrend period must remain one typed quad launch"
                );
            };
            assert_eq!(*atr_length, expected_period);
            assert_eq!(*base_multiplier_bits, 3.0_f64.to_bits());
            assert_eq!(*noise_threshold_bits, 1.0_f64.to_bits());
            assert_eq!(*expansion_alpha_bits, 0.5_f64.to_bits());
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                EVASIVE_SUPERTREND_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "evasive_supertrend_{expected_period}_{}",
                            EVASIVE_SUPERTREND_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("atr_length", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::EvasiveSupertrend {
                            atr_length: expected_period,
                            base_multiplier_bits: 3.0_f64.to_bits(),
                            noise_threshold_bits: 1.0_f64.to_bits(),
                            expansion_alpha_bits: 0.5_f64.to_bits(),
                        }
            }));
        }
    }

    #[test]
    fn fibonacci_entry_bands_base_and_length_sweeps_form_five_exact_sixteen_receipt_launches() {
        assert_eq!(
            output_ids_for(FIBONACCI_ENTRY_BANDS_ID),
            FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS.map(Some),
            "reviewed TP duplicates must remain excluded from the admitted graph"
        );
        let swept_lengths = vec![7, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[FIBONACCI_ENTRY_BANDS_ID],
            &[],
            &[(FIBONACCI_ENTRY_BANDS_ID, swept_lengths.clone())],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            80,
            "five canonical tuples times sixteen admitted outputs must retain exactly eighty receipts"
        );

        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), 5);
        for (launch, expected_length) in launches.iter().zip([21, 7, 50, 100, 200]) {
            let ResolvedClassicCudaLaunch::FibonacciEntryBands { routes, length } = launch else {
                panic!("every Fibonacci Entry Bands tuple must form one typed resident launch");
            };
            assert_eq!(*length, expected_length);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                let expected_column = if expected_length == 21 {
                    format!(
                        "fibonacci_entry_bands_{}",
                        FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS[offset]
                    )
                } else {
                    format!(
                        "fibonacci_entry_bands_{expected_length}_{}",
                        FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS[offset]
                    )
                };
                let expected_parameters = if expected_length == 21 {
                    ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(21),
                        require_period_invariant_kernel: false,
                    }
                } else {
                    ClassicCudaParameters::Swept {
                        period: expected_length,
                        overrides: vec![("length", expected_length as i64)],
                        anchor: ClassicCudaAnchor::Resolved(expected_length),
                    }
                };
                route.node.column_name == expected_column
                    && route.node.stage
                        == if expected_length == 21 {
                            ClassicCudaStage::Base
                        } else {
                            ClassicCudaStage::Extended
                        }
                    && route.node.parameters == expected_parameters
                    && route.route
                        == ClassicCudaResolvedRoute::FibonacciEntryBands {
                            length: expected_length,
                        }
            }));
        }
    }

    #[test]
    fn fibonacci_entry_bands_rejects_noncanonical_sweep_before_any_launch() {
        let mut plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(FIBONACCI_ENTRY_BANDS_ID, vec![7])])
                .unwrap();
        assert_eq!(
            plan.nodes.len(),
            FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS.len()
        );
        for node in &mut plan.nodes {
            node.parameters = ClassicCudaParameters::Swept {
                period: 7,
                overrides: vec![("atr_length", 7)],
                anchor: ClassicCudaAnchor::Resolved(7),
            };
        }

        let gaps = preflight_exact_classic_cuda_plan(&plan)
            .expect_err("noncanonical ATR-length injection must fail before allocation or launch");
        assert_eq!(
            gaps.len(),
            FIBONACCI_ENTRY_BANDS_PRODUCTION_OUTPUT_IDS.len()
        );
        assert!(gaps.iter().all(|gap| {
            gap.reason == ClassicCudaGapReason::MissingParameterContract
                && gap.detail.contains("exact length overrides")
        }));
    }

    #[test]
    fn fibonacci_trailing_stop_default_outputs_form_one_exact_typed_launch() {
        assert_eq!(
            output_ids_for(FIBONACCI_TRAILING_STOP_ID),
            FIBONACCI_TRAILING_STOP_OUTPUT_IDS.map(Some),
            "canonical Fibonacci Trailing Stop identity/order drifted"
        );
        let plan =
            build_exact_classic_cuda_plan(2_000, &[FIBONACCI_TRAILING_STOP_ID], &[], &[]).unwrap();
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::FibonacciTrailingStop {
                routes,
                left_bars,
                right_bars,
                level_bits,
                trigger_mode,
            },
        ] = launches.as_slice()
        else {
            panic!("Fibonacci Trailing Stop must form one typed four-output default launch");
        };
        assert_eq!(
            (*left_bars, *right_bars, *level_bits, *trigger_mode),
            (20, 1, (-0.382_f64).to_bits(), 0)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            FIBONACCI_TRAILING_STOP_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "fibonacci_trailing_stop_trailing_stop",
                "fibonacci_trailing_stop_long_stop",
                "fibonacci_trailing_stop_short_stop",
                "fibonacci_trailing_stop_direction",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(1),
                        require_period_invariant_kernel: true,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::FibonacciTrailingStop {
                        left_bars: 20,
                        right_bars: 1,
                        level_bits: (-0.382_f64).to_bits(),
                        trigger_mode: 0,
                    }
        }));
    }

    #[test]
    fn fibonacci_trailing_stop_synthetic_period_sweep_fails_before_launch() {
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(FIBONACCI_TRAILING_STOP_ID, vec![21])],
        )
        .unwrap();
        let gaps = preflight_exact_classic_cuda_plan(&plan)
            .expect_err("synthetic period must fail before engine allocation or launch");
        let parameters = ClassicCudaParameters::Swept {
            period: 21,
            overrides: Vec::new(),
            anchor: ClassicCudaAnchor::Resolved(21),
        };
        let mut expected = Vec::new();
        for output_id in FIBONACCI_TRAILING_STOP_OUTPUT_IDS {
            let column_name = format!("{FIBONACCI_TRAILING_STOP_ID}_21_{output_id}");
            expected.push(ClassicCudaGap {
                column_name: column_name.clone(),
                indicator_id: FIBONACCI_TRAILING_STOP_ID,
                requested_output_id: Some(output_id),
                parameters: parameters.clone(),
                reason: ClassicCudaGapReason::MissingParameterContract,
                detail: "canonical CPU sweep produced no explicit integer window overrides"
                    .to_string(),
            });
            expected.push(ClassicCudaGap {
                column_name,
                indicator_id: FIBONACCI_TRAILING_STOP_ID,
                requested_output_id: Some(output_id),
                parameters: parameters.clone(),
                reason: ClassicCudaGapReason::MissingParameterContract,
                detail: format!(
                    "{FIBONACCI_TRAILING_STOP_ID}: formula has no canonical period sweep"
                ),
            });
        }
        assert_eq!(
            gaps, expected,
            "all four receipts must retain ordered generic and family-specific refusals"
        );
    }

    #[test]
    fn fisher_default_outputs_form_one_exact_period_nine_pair_launch() {
        assert_eq!(
            output_ids_for(FISHER_ID),
            FISHER_OUTPUT_IDS.map(Some),
            "canonical Fisher output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[FISHER_ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), FISHER_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [ResolvedClassicCudaLaunch::Fisher { routes, period }] = launches.as_slice() else {
            panic!("Fisher must preflight as one typed two-output default launch");
        };
        assert_eq!(*period, 9);
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            FISHER_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            ["fisher_fisher", "fisher_signal"]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(9),
                        require_period_invariant_kernel: false,
                    }
                && route.route == ClassicCudaResolvedRoute::Fisher { period: 9 }
        }));
    }

    #[test]
    fn fisher_period_sweeps_form_five_exact_pair_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(2_000, &[], &[], &[(FISHER_ID, periods.clone())])
            .unwrap();
        assert_eq!(plan.nodes.len(), periods.len() * FISHER_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());

        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Fisher { routes, period } = launch else {
                panic!("every Fisher period must remain one typed two-output launch");
            };
            assert_eq!(*period, expected_period);
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                FISHER_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("fisher_{expected_period}_{}", FISHER_OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_period,
                            overrides: vec![("period", expected_period as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_period),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::Fisher {
                            period: expected_period,
                        }
            }));
        }
    }

    #[test]
    fn forward_backward_exponential_oscillator_default_outputs_form_one_exact_triple_launch() {
        assert_eq!(
            output_ids_for(FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID),
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS.map(Some),
            "canonical FBEO output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS.len()
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::ForwardBackwardExponentialOscillator {
                routes,
                length,
                smooth,
            },
        ] = launches.as_slice()
        else {
            panic!("FBEO must preflight as one typed three-output default launch");
        };
        assert_eq!((*length, *smooth), (20, 10));
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "forward_backward_exponential_oscillator_forward_backward",
                "forward_backward_exponential_oscillator_backward",
                "forward_backward_exponential_oscillator_histogram",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(20),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::ForwardBackwardExponentialOscillator {
                        length: 20,
                        smooth: 10,
                    }
        }));
    }

    #[test]
    fn forward_backward_exponential_oscillator_length_sweeps_form_five_exact_triple_launches() {
        let periods = vec![7, 21, 50, 100, 200];
        let plan = build_exact_classic_cuda_plan(
            2_000,
            &[],
            &[],
            &[(FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID, periods.clone())],
        )
        .unwrap();
        assert_eq!(
            plan.nodes.len(),
            periods.len() * FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS.len()
        );
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());

        for (launch, expected_length) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::ForwardBackwardExponentialOscillator {
                routes,
                length,
                smooth,
            } = launch
            else {
                panic!("every FBEO length must remain one typed three-output launch");
            };
            assert_eq!((*length, *smooth), (expected_length, 10));
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
            );
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!(
                            "{FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_ID}_{expected_length}_{}",
                            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS[offset]
                        )
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_length,
                            overrides: vec![("length", expected_length as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_length),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::ForwardBackwardExponentialOscillator {
                            length: expected_length,
                            smooth: 10,
                        }
            }));
        }
    }

    #[test]
    fn fvg_trailing_stop_default_outputs_form_one_exact_four_output_launch() {
        const ID: &str = "fvg_trailing_stop";
        const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];
        assert_eq!(
            output_ids_for(ID),
            OUTPUT_IDS.map(Some),
            "canonical FVG Trailing Stop output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::FvgTrailingStop {
                routes,
                unmitigated_fvg_lookback,
                smoothing_length,
                reset_on_cross,
            },
        ] = launches.as_slice()
        else {
            panic!("FVG Trailing Stop must preflight as one typed four-output default launch");
        };
        assert_eq!(
            (
                *unmitigated_fvg_lookback,
                *smoothing_length,
                *reset_on_cross
            ),
            (5, 9, false)
        );
        assert_eq!(routes.each_ref().map(|route| route.output_id), OUTPUT_IDS);
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "fvg_trailing_stop_upper",
                "fvg_trailing_stop_lower",
                "fvg_trailing_stop_upper_ts",
                "fvg_trailing_stop_lower_ts",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(9),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::FvgTrailingStop {
                        unmitigated_fvg_lookback: 5,
                        smoothing_length: 9,
                        reset_on_cross: false,
                    }
        }));
    }

    #[test]
    fn fvg_trailing_stop_smoothing_sweeps_form_five_exact_four_output_launches() {
        const ID: &str = "fvg_trailing_stop";
        const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];
        let smoothing_lengths = vec![7, 21, 50, 100, 200];
        let plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(ID, smoothing_lengths.clone())])
                .unwrap();
        assert_eq!(plan.nodes.len(), smoothing_lengths.len() * OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), smoothing_lengths.len());

        for (launch, expected_smoothing) in launches.iter().zip(smoothing_lengths) {
            let ResolvedClassicCudaLaunch::FvgTrailingStop {
                routes,
                unmitigated_fvg_lookback,
                smoothing_length,
                reset_on_cross,
            } = launch
            else {
                panic!("every admitted smoothing length must stay one typed four-output launch");
            };
            assert_eq!(
                (
                    *unmitigated_fvg_lookback,
                    *smoothing_length,
                    *reset_on_cross
                ),
                (5, expected_smoothing, false)
            );
            assert_eq!(routes.each_ref().map(|route| route.output_id), OUTPUT_IDS);
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("{ID}_{expected_smoothing}_{}", OUTPUT_IDS[offset])
                    && route.node.parameters
                        == ClassicCudaParameters::Swept {
                            period: expected_smoothing,
                            overrides: vec![("smoothing_length", expected_smoothing as i64)],
                            anchor: ClassicCudaAnchor::Resolved(expected_smoothing),
                        }
                    && route.route
                        == ClassicCudaResolvedRoute::FvgTrailingStop {
                            unmitigated_fvg_lookback: 5,
                            smoothing_length: expected_smoothing,
                            reset_on_cross: false,
                        }
            }));
        }
    }

    #[test]
    fn gatorosc_default_outputs_form_one_exact_four_output_launch() {
        assert_eq!(
            output_ids_for(GATOROSC_ID),
            GATOROSC_OUTPUT_IDS.map(Some),
            "canonical Gator Oscillator output identity/order drifted"
        );
        let plan = build_exact_classic_cuda_plan(2_000, &[GATOROSC_ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), GATOROSC_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Gatorosc {
                routes,
                jaws_length,
                jaws_shift,
                teeth_length,
                teeth_shift,
                lips_length,
                lips_shift,
            },
        ] = launches.as_slice()
        else {
            panic!("Gator Oscillator must preflight as one typed four-output default launch");
        };
        assert_eq!(
            (
                *jaws_length,
                *jaws_shift,
                *teeth_length,
                *teeth_shift,
                *lips_length,
                *lips_shift,
            ),
            (13, 8, 8, 5, 5, 3)
        );
        assert_eq!(
            routes.each_ref().map(|route| route.output_id),
            GATOROSC_OUTPUT_IDS
        );
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "gatorosc_upper",
                "gatorosc_lower",
                "gatorosc_upper_change",
                "gatorosc_lower_change",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(13),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::Gatorosc {
                        jaws_length: 13,
                        jaws_shift: 8,
                        teeth_length: 8,
                        teeth_shift: 5,
                        lips_length: 5,
                        lips_shift: 3,
                    }
        }));
    }

    #[test]
    fn gatorosc_registry_ratio_sweeps_form_five_exact_four_output_launches() {
        let expected = [
            (7, 7, 4, 3),
            (21, 21, 13, 8),
            (50, 50, 31, 19),
            (100, 100, 62, 38),
            (200, 200, 123, 77),
        ];
        let periods = expected.iter().map(|tuple| tuple.0).collect::<Vec<_>>();
        let plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(GATOROSC_ID, periods)]).unwrap();
        assert_eq!(plan.nodes.len(), expected.len() * GATOROSC_OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), expected.len());

        for (launch, &(period, jaws_length, teeth_length, lips_length)) in
            launches.iter().zip(&expected)
        {
            let ResolvedClassicCudaLaunch::Gatorosc {
                routes,
                jaws_length: actual_jaws,
                jaws_shift,
                teeth_length: actual_teeth,
                teeth_shift,
                lips_length: actual_lips,
                lips_shift,
            } = launch
            else {
                panic!("every admitted Gator Oscillator ratio must stay one typed launch");
            };
            assert_eq!(
                (
                    *actual_jaws,
                    *jaws_shift,
                    *actual_teeth,
                    *teeth_shift,
                    *actual_lips,
                    *lips_shift,
                ),
                (jaws_length, 8, teeth_length, 5, lips_length, 3)
            );
            assert_eq!(
                routes.each_ref().map(|route| route.output_id),
                GATOROSC_OUTPUT_IDS
            );
            let expected_parameters = ClassicCudaParameters::Swept {
                period,
                overrides: vec![
                    ("jaws_length", jaws_length as i64),
                    ("teeth_length", teeth_length as i64),
                    ("lips_length", lips_length as i64),
                ],
                anchor: ClassicCudaAnchor::Resolved(period),
            };
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("{GATOROSC_ID}_{period}_{}", GATOROSC_OUTPUT_IDS[offset])
                    && route.node.parameters == expected_parameters
                    && route.route
                        == ClassicCudaResolvedRoute::Gatorosc {
                            jaws_length,
                            jaws_shift: 8,
                            teeth_length,
                            teeth_shift: 5,
                            lips_length,
                            lips_shift: 3,
                        }
            }));
        }
    }

    #[test]
    fn halftrend_default_outputs_form_one_exact_six_output_launch() {
        const ID: &str = "halftrend";
        const OUTPUT_IDS: [&str; 6] = [
            "halftrend",
            "trend",
            "atr_high",
            "atr_low",
            "buy_signal",
            "sell_signal",
        ];
        assert_eq!(
            output_ids_for(ID),
            OUTPUT_IDS.map(Some),
            "canonical HalfTrend output identity/order drifted"
        );

        let plan = build_exact_classic_cuda_plan(2_000, &[ID], &[], &[]).unwrap();
        assert_eq!(plan.nodes.len(), OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        let [
            ResolvedClassicCudaLaunch::Halftrend {
                routes,
                amplitude,
                channel_deviation_bits,
                atr_period,
            },
        ] = launches.as_slice()
        else {
            panic!("HalfTrend must preflight as one typed six-output default launch");
        };
        assert_eq!(
            (*amplitude, *channel_deviation_bits, *atr_period),
            (2, 2.0_f64.to_bits(), 100)
        );
        assert_eq!(routes.each_ref().map(|route| route.output_id), OUTPUT_IDS);
        assert_eq!(
            routes
                .each_ref()
                .map(|route| route.node.column_name.as_str()),
            [
                "halftrend_halftrend",
                "halftrend_trend",
                "halftrend_atr_high",
                "halftrend_atr_low",
                "halftrend_buy_signal",
                "halftrend_sell_signal",
            ]
        );
        assert!(routes.iter().all(|route| {
            route.node.stage == ClassicCudaStage::Base
                && route.node.parameters
                    == ClassicCudaParameters::Defaults {
                        anchor: ClassicCudaAnchor::Resolved(100),
                        require_period_invariant_kernel: false,
                    }
                && route.route
                    == ClassicCudaResolvedRoute::Halftrend {
                        amplitude: 2,
                        channel_deviation_bits: 2.0_f64.to_bits(),
                        atr_period: 100,
                    }
        }));
    }

    #[test]
    fn halftrend_registry_ratio_sweeps_form_four_exact_six_output_launches() {
        const ID: &str = "halftrend";
        const OUTPUT_IDS: [&str; 6] = [
            "halftrend",
            "trend",
            "atr_high",
            "atr_low",
            "buy_signal",
            "sell_signal",
        ];
        let periods = vec![7, 21, 50, 200];
        let plan =
            build_exact_classic_cuda_plan(2_000, &[], &[], &[(ID, periods.clone())]).unwrap();
        assert_eq!(plan.nodes.len(), periods.len() * OUTPUT_IDS.len());
        let launches = preflight_exact_classic_cuda_plan(&plan).unwrap();
        assert_eq!(launches.len(), periods.len());

        for (launch, expected_period) in launches.iter().zip(periods) {
            let ResolvedClassicCudaLaunch::Halftrend {
                routes,
                amplitude,
                channel_deviation_bits,
                atr_period,
            } = launch
            else {
                panic!("every admitted HalfTrend ATR period must stay one typed launch");
            };
            assert_eq!(
                (*amplitude, *channel_deviation_bits, *atr_period),
                (2, 2.0_f64.to_bits(), expected_period)
            );
            assert_eq!(routes.each_ref().map(|route| route.output_id), OUTPUT_IDS);
            let expected_parameters = ClassicCudaParameters::Swept {
                period: expected_period,
                overrides: vec![("atr_period", expected_period as i64)],
                anchor: ClassicCudaAnchor::Resolved(expected_period),
            };
            assert!(routes.iter().enumerate().all(|(offset, route)| {
                route.node.stage == ClassicCudaStage::Extended
                    && route.node.column_name
                        == format!("{ID}_{expected_period}_{}", OUTPUT_IDS[offset])
                    && route.node.parameters == expected_parameters
                    && route.route
                        == ClassicCudaResolvedRoute::Halftrend {
                            amplitude: 2,
                            channel_deviation_bits: 2.0_f64.to_bits(),
                            atr_period: expected_period,
                        }
            }));
        }
    }
}
