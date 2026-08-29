#[cfg(feature = "cuda-build-native")]
use std::sync::OnceLock;
#[cfg(all(feature = "cuda-build-native", test))]
use std::{
    cell::Cell,
    sync::{Mutex, MutexGuard},
};

#[cfg(feature = "cuda-build-native")]
pub mod accumulation_swing_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ad_wrapper;
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub mod adaptive_schaff_trend_cycle_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod advance_decline_line_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod adx_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod adxr_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod alligator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod alphatrend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod andean_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod aroon_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod atr_percentile_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod atr_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod autocorrelation_indicator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod avsl_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod bandpass_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod bench;
#[cfg(feature = "cuda-build-native")]
pub mod candle_strength_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod chande_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod cvi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod cyberpunk_value_trend_analyzer_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod cycle_channel_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod daily_factor_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod device_types;
/// The f64 half of the device vocabulary. ADDITIVE — `device_types` is
/// untouched because 180 f32 wrappers and the generated f32 dispatcher depend
/// on it.
#[cfg(feature = "cuda-build-native")]
pub mod device_types_f64;
#[cfg(feature = "cuda-build-native")]
pub mod di_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod disparity_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod dm_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod donchian_channel_width_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod donchian_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod dx_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod dynamic_momentum_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_autocorrelation_periodogram_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_data_sampling_relative_strength_indicator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_detrending_filter_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_linear_extrapolation_predictor_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_smoothed_adaptive_momentum_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod eri_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod evasive_supertrend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ewma_volatility_wrapper;
/// Launch planning shared by the from-scratch f64 kernels: the slot count is
/// chosen from FREE VRAM, never from how wide a sweep the operator asked for.
#[cfg(feature = "cuda-build-native")]
pub mod f64_launch;
#[cfg(feature = "cuda-build-native")]
pub mod goertzel_cycle_composite_wave_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod hypertrend_wrapper;
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub mod impulse_macd_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod insync_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod kairi_relative_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod keltner_channel_width_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod keltner_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod market_meanness_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod market_structure_trailing_stop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod marketefi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod medprice_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod mesa_stochastic_multi_length_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod module_loader;
#[cfg(feature = "cuda-build-native")]
pub mod moving_average_cross_probability_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod moving_averages;
#[cfg(feature = "cuda-build-native")]
pub mod multi_length_stochastic_average_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod normalized_resonator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod possible_rsi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod price_moving_average_ratio_percentile_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod qstick_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod range_breakout_signals_wrapper;
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub mod rocr_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod runtime;
#[cfg(feature = "cuda-build-native")]
pub mod smooth_theil_sen_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod standardized_psar_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod statistical_trailing_stop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod supertrend_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod supertrend_recovery_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod velocity_acceleration_convergence_divergence_indicator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volume_weighted_stochastic_rsi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vwap_deviation_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vwap_zscore_with_signals_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod wavetrend;

#[cfg(feature = "cuda-build-native")]
pub use accumulation_swing_index_wrapper::{
    CudaAccumulationSwingIndex, CudaAccumulationSwingIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use ad_wrapper::{CudaAd, CudaAdError};
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub use adaptive_schaff_trend_cycle_wrapper::{
    CudaAdaptiveSchaffTrendCycle, CudaAdaptiveSchaffTrendCycleError,
};
#[cfg(feature = "cuda-build-native")]
pub use advance_decline_line_wrapper::{CudaAdvanceDeclineLine, CudaAdvanceDeclineLineError};
#[cfg(feature = "cuda-build-native")]
pub use adx_wrapper::{CudaAdx, CudaAdxError};
#[cfg(feature = "cuda-build-native")]
pub use adxr_wrapper::{CudaAdxr, CudaAdxrError};
#[cfg(feature = "cuda-build-native")]
pub use alligator_wrapper::{
    CudaAlligator, CudaAlligatorBatchResult, CudaAlligatorError, DeviceArrayF32Trio,
};
#[cfg(feature = "cuda-build-native")]
pub use alphatrend_wrapper::{CudaAlphaTrend, CudaAlphaTrendError};
#[cfg(feature = "cuda-build-native")]
pub use andean_oscillator_wrapper::{CudaAndeanOscillator, CudaAndeanOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use aroon_wrapper::{CudaAroon, CudaAroonError};
#[cfg(feature = "cuda-build-native")]
pub use atr_percentile_wrapper::{CudaAtrPercentile, CudaAtrPercentileError};
#[cfg(feature = "cuda-build-native")]
pub use atr_wrapper::CudaAtr;
#[cfg(feature = "cuda-build-native")]
pub use autocorrelation_indicator_wrapper::{
    CudaAutocorrelationIndicator, CudaAutocorrelationIndicatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use avsl_wrapper::{CudaAvsl, CudaAvslError};
#[cfg(feature = "cuda-build-native")]
pub use bandpass_wrapper::{CudaBandpass, CudaBandpassBatchResult, DeviceArrayF32Quad};
#[cfg(feature = "cuda-build-native")]
pub use bench::{CudaBenchScenario, CudaBenchState};
#[cfg(feature = "cuda-build-native")]
pub use candle_strength_oscillator_wrapper::{
    CudaCandleStrengthOscillator, CudaCandleStrengthOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use chande_wrapper::CudaChande;
#[cfg(feature = "cuda-build-native")]
pub use cvi_wrapper::{CudaCvi, CudaCviError};
#[cfg(feature = "cuda-build-native")]
pub use cyberpunk_value_trend_analyzer_wrapper::{
    CudaCyberpunkValueTrendAnalyzer, CudaCyberpunkValueTrendAnalyzerError,
};
#[cfg(feature = "cuda-build-native")]
pub use cycle_channel_oscillator_wrapper::{
    CudaCycleChannelOscillator, CudaCycleChannelOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use daily_factor_wrapper::{CudaDailyFactor, CudaDailyFactorError};
#[cfg(feature = "cuda-build-native")]
pub use device_types::{
    CudaDeviceCloseVolumeRef, CudaDeviceHighLowRef, CudaDeviceMatrix, CudaDeviceMatrixF32,
    CudaDeviceMatrixF32Ref, CudaDeviceOhlc, CudaDeviceOhlcRef, CudaDeviceOhlcv, CudaDeviceOhlcvRef,
    CudaDeviceSliceF32Ref, CudaDeviceSliceI32Ref, CudaDeviceSliceI64Ref, CudaDeviceVector,
    CudaDeviceVectorF32, CudaDeviceVectorI32, CudaDeviceVectorI64, CudaDeviceViewError,
};
#[cfg(feature = "cuda-build-native")]
pub use device_types_f64::{
    CudaDeviceCloseVolumeF64Ref, CudaDeviceHighLowF64Ref, CudaDeviceMatrixF64,
    CudaDeviceMatrixF64Ref, CudaDeviceOhlcF64Ref, CudaDeviceOhlcvF64, CudaDeviceOhlcvF64Ref,
    CudaDeviceSliceF64Ref, CudaDeviceVectorF64,
};
#[cfg(feature = "cuda-build-native")]
pub use di_wrapper::{CudaDi, CudaDiError, DeviceArrayF32Pair};
#[cfg(feature = "cuda-build-native")]
pub use disparity_index_wrapper::{CudaDisparityIndex, CudaDisparityIndexError};
#[cfg(feature = "cuda-build-native")]
pub use dm_wrapper::{CudaDm, CudaDmError};
#[cfg(feature = "cuda-build-native")]
pub use donchian_channel_width_wrapper::{CudaDonchianChannelWidth, CudaDonchianChannelWidthError};
#[cfg(feature = "cuda-build-native")]
pub use donchian_wrapper::{CudaDonchian, CudaDonchianError};
#[cfg(feature = "cuda-build-native")]
pub use dx_wrapper::{CudaDx, CudaDxError};
#[cfg(feature = "cuda-build-native")]
pub use dynamic_momentum_index_wrapper::{CudaDynamicMomentumIndex, CudaDynamicMomentumIndexError};
#[cfg(feature = "cuda-build-native")]
pub use ehlers_autocorrelation_periodogram_wrapper::{
    CudaEhlersAutocorrelationPeriodogram, CudaEhlersAutocorrelationPeriodogramError,
};
#[cfg(feature = "cuda-build-native")]
pub use ehlers_data_sampling_relative_strength_indicator_wrapper::{
    CudaEhlersDataSamplingRelativeStrengthIndicator,
    CudaEhlersDataSamplingRelativeStrengthIndicatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use ehlers_detrending_filter_wrapper::{
    CudaEhlersDetrendingFilter, CudaEhlersDetrendingFilterError,
};
#[cfg(feature = "cuda-build-native")]
pub use ehlers_linear_extrapolation_predictor_wrapper::{
    CudaEhlersLinearExtrapolationPredictor, CudaEhlersLinearExtrapolationPredictorError,
};
#[cfg(feature = "cuda-build-native")]
pub use ehlers_smoothed_adaptive_momentum_wrapper::{
    CudaEhlersSmoothedAdaptiveMomentum, CudaEhlersSmoothedAdaptiveMomentumError,
};
#[cfg(feature = "cuda-build-native")]
pub use eri_wrapper::{CudaEri, CudaEriError};
#[cfg(feature = "cuda-build-native")]
pub use evasive_supertrend_wrapper::{CudaEvasiveSuperTrend, CudaEvasiveSuperTrendError};
#[cfg(feature = "cuda-build-native")]
pub use ewma_volatility_wrapper::{CudaEwmaVolatility, CudaEwmaVolatilityError};
#[cfg(feature = "cuda-build-native")]
pub use goertzel_cycle_composite_wave_wrapper::{
    CudaGoertzelCycleCompositeWave, CudaGoertzelCycleCompositeWaveError,
};
#[cfg(feature = "cuda-build-native")]
pub use hypertrend_wrapper::{CudaHyperTrend, CudaHyperTrendError};
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub use impulse_macd_wrapper::{CudaImpulseMacd, CudaImpulseMacdError};
#[cfg(feature = "cuda-build-native")]
pub use insync_index_wrapper::{CudaInsyncIndex, CudaInsyncIndexError};
#[cfg(feature = "cuda-build-native")]
pub use kairi_relative_index_wrapper::{CudaKairiRelativeIndex, CudaKairiRelativeIndexError};
#[cfg(feature = "cuda-build-native")]
pub use keltner_channel_width_oscillator_wrapper::{
    CudaKeltnerChannelWidthOscillator, CudaKeltnerChannelWidthOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use keltner_wrapper::{
    CudaKeltner, CudaKeltnerBatchResult, CudaKeltnerError, DeviceKeltnerTriplet,
};
#[cfg(feature = "cuda-build-native")]
pub use market_meanness_index_wrapper::{CudaMarketMeannessIndex, CudaMarketMeannessIndexError};
#[cfg(feature = "cuda-build-native")]
pub use market_structure_trailing_stop_wrapper::{
    CudaMarketStructureTrailingStop, CudaMarketStructureTrailingStopError,
};
#[cfg(feature = "cuda-build-native")]
pub use marketefi_wrapper::{CudaMarketefi, CudaMarketefiError};
#[cfg(feature = "cuda-build-native")]
pub use medprice_wrapper::CudaMedprice;
#[cfg(feature = "cuda-build-native")]
pub use mesa_stochastic_multi_length_wrapper::{
    CudaMesaStochasticMultiLength, CudaMesaStochasticMultiLengthError,
};
#[cfg(feature = "cuda-build-native")]
pub use moving_average_cross_probability_wrapper::{
    CudaMovingAverageCrossProbability, CudaMovingAverageCrossProbabilityError,
};
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::rsmk_wrapper::{CudaRsmk, CudaRsmkError};
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::wclprice_wrapper::CudaWclprice;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::{
    CudaAlma, CudaDma, CudaEhlersPma, CudaGaussian, CudaJma, CudaMaDeviceDataRef, CudaMama,
    CudaReflex, CudaSqwma, CudaTema, CudaVwma, DeviceArrayF32, DeviceEhlersPmaPair, DeviceMamaPair,
};
#[cfg(feature = "cuda-build-native")]
pub use multi_length_stochastic_average_wrapper::{
    CudaMultiLengthStochasticAverage, CudaMultiLengthStochasticAverageError,
};
#[cfg(feature = "cuda-build-native")]
pub use normalized_resonator_wrapper::{CudaNormalizedResonator, CudaNormalizedResonatorError};
#[cfg(feature = "cuda-build-native")]
pub use possible_rsi_wrapper::{CudaPossibleRsi, CudaPossibleRsiError};
#[cfg(feature = "cuda-build-native")]
pub use price_moving_average_ratio_percentile_wrapper::{
    CudaPriceMovingAverageRatioPercentile, CudaPriceMovingAverageRatioPercentileError,
};
#[cfg(feature = "cuda-build-native")]
pub use qstick_wrapper::{
    BatchKernelPolicy as QsBatchKernelPolicy, CudaQstick, CudaQstickError, CudaQstickPolicy,
    ManySeriesKernelPolicy as QsManySeriesKernelPolicy,
};
#[cfg(feature = "cuda-build-native")]
pub use range_breakout_signals_wrapper::{CudaRangeBreakoutSignals, CudaRangeBreakoutSignalsError};
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub use rocr_wrapper::{CudaRocr, CudaRocrError};
#[cfg(feature = "cuda-build-native")]
pub use runtime::{CudaRuntime, CudaRuntimeError, CudaSession, CudaSessionIdentity};
#[cfg(feature = "cuda-build-native")]
pub use smooth_theil_sen_wrapper::{CudaSmoothTheilSen, CudaSmoothTheilSenError};
#[cfg(feature = "cuda-build-native")]
pub use standardized_psar_oscillator_wrapper::{
    CudaStandardizedPsarOscillator, CudaStandardizedPsarOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use statistical_trailing_stop_wrapper::{
    CudaStatisticalTrailingStop, CudaStatisticalTrailingStopError,
};
#[cfg(feature = "cuda-build-native")]
pub use supertrend_oscillator_wrapper::{CudaSupertrendOscillator, CudaSupertrendOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use supertrend_recovery_wrapper::{CudaSuperTrendRecovery, CudaSuperTrendRecoveryError};
#[cfg(feature = "cuda-build-native")]
pub use velocity_acceleration_convergence_divergence_indicator_wrapper::{
    CudaVelocityAccelerationConvergenceDivergenceIndicator,
    CudaVelocityAccelerationConvergenceDivergenceIndicatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use volume_weighted_stochastic_rsi_wrapper::{
    CudaVolumeWeightedStochasticRsi, CudaVolumeWeightedStochasticRsiError,
};
#[cfg(feature = "cuda-build-native")]
pub use vwap_deviation_oscillator_wrapper::{
    CudaVwapDeviationOscillator, CudaVwapDeviationOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use vwap_zscore_with_signals_wrapper::{
    CudaVwapZscoreWithSignals, CudaVwapZscoreWithSignalsError,
};
#[cfg(feature = "cuda-build-native")]
pub mod oscillators;
#[cfg(feature = "cuda-build-native")]
pub use oscillators::msw_wrapper::{CudaMsw, CudaMswError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::qqe_wrapper::{CudaQqe, CudaQqeError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::rvi_wrapper::{CudaRvi, CudaRviError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::stc_wrapper::{CudaStc, CudaStcError};
#[cfg(feature = "cuda-build-native")]
pub mod bollinger_bands_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod dvdiqqe_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod er_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod nadaraya_watson_envelope_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod nvi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod pfe_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod pvi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod supertrend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ttm_trend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vertical_horizontal_filter_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vpt_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vwmacd_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod wto_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod zig_zag_channels_wrapper;

#[cfg(feature = "cuda-build-native")]
pub use dvdiqqe_wrapper::{CudaDvdiqqe, CudaDvdiqqeError};
#[cfg(feature = "cuda-build-native")]
pub use er_wrapper::{CudaEr, CudaErError};
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::cwma_wrapper::CudaCwma;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::ehlers_ecema_wrapper::CudaEhlersEcema;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::epma_wrapper::CudaEpma;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::highpass_wrapper::CudaHighpass;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::kama_wrapper::CudaKama;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::nama_wrapper::CudaNama;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::sinwma_wrapper::CudaSinwma;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::supersmoother_3_pole_wrapper::CudaSupersmoother3Pole;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::tradjema_wrapper::CudaTradjema;
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::wma_wrapper::CudaWma;
#[cfg(feature = "cuda-build-native")]
pub use nadaraya_watson_envelope_wrapper::{CudaNwe, CudaNweError, DeviceNwePair};
#[cfg(feature = "cuda-build-native")]
pub use nvi_wrapper::{CudaNvi, CudaNviError};
#[cfg(feature = "cuda-build-native")]
pub use pfe_wrapper::{CudaPfe, CudaPfeError};
#[cfg(feature = "cuda-build-native")]
pub use pvi_wrapper::{CudaPvi, CudaPviError};
#[cfg(feature = "cuda-build-native")]
pub use supertrend_wrapper::{CudaSupertrend, CudaSupertrendError};
#[cfg(feature = "cuda-build-native")]
pub use ttm_trend_wrapper::{CudaTtmTrend, CudaTtmTrendError};
#[cfg(feature = "cuda-build-native")]
pub use vertical_horizontal_filter_wrapper::{
    CudaVerticalHorizontalFilter, CudaVerticalHorizontalFilterBatchResult,
    CudaVerticalHorizontalFilterError,
};
#[cfg(feature = "cuda-build-native")]
pub use vpt_wrapper::{CudaVpt, CudaVptError};
#[cfg(feature = "cuda-build-native")]
pub use vwmacd_wrapper::{CudaVwmacd, CudaVwmacdError};
#[cfg(feature = "cuda-build-native")]
pub use wto_wrapper::{CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};
#[cfg(feature = "cuda-build-native")]
pub use zig_zag_channels_wrapper::{CudaZigZagChannels, CudaZigZagChannelsError};
#[cfg(feature = "cuda-build-native")]
pub mod bollinger_bands_width_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod bull_power_vs_bear_power_wrapper;
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub mod chandelier_exit_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod cksp_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod correl_hl_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod damiani_volatmeter_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod decisionpoint_breadth_swenlin_trading_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod demand_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod deviation_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod devstop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod didi_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod directional_imbalance_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod dual_ulcer_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod efi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_fm_demodulator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod emd_trend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod emd_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod exponential_trend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod fibonacci_trailing_stop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod forward_backward_exponential_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod fractal_dimension_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod fvg_positioning_average_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod fvg_trailing_stop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod garman_klass_volatility_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod geometric_bias_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod gmma_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod gopalakrishnan_range_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod grover_llorens_cycle_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod half_causal_estimator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod halftrend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod historical_volatility_percentile_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod historical_volatility_rank_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod historical_volatility_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod hull_butterfly_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod intraday_momentum_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod kaufmanstop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod kurtosis_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod l1_ehlers_phasor_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod l2_ehlers_signal_to_noise_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod leavitt_convolution_acceleration_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod linear_correlation_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod linear_regression_intensity_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod lpc_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod macd_wave_signal_pro_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod mass_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod mean_ad_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod medium_ad_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod minmax_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod mod_god_mode_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod momentum_ratio_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod monotonicity_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod natr_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod neighboring_trailing_stop_wrapper;
/// The NeoEthos f64 indicator lane: one module, ten `*_batch_f64` kernels, no
/// narrowing and no fast math. See `kernels/cuda/neoethos_f64_kernels.cu`.
#[cfg(feature = "cuda-build-native")]
pub mod neoethos_f64_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod net_myrsi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod nonlinear_regression_zero_lag_moving_average_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod normalized_volume_true_range_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod obv_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod on_balance_volume_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod parkinson_volatility_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod pattern_recognition_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod percentile_nearest_rank_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod polynomial_regression_extrapolation_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod premier_rsi_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod pretty_good_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod price_density_market_noise_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod projection_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod psychological_line_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod qqe_weighted_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod random_walk_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod range_filter_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod rank_correlation_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod regression_slope_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod relative_strength_index_wave_indicator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod reversal_signals_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod rogers_satchell_volatility_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod rolling_skewness_kurtosis_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod rolling_z_score_trend_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod safezonestop_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod sar_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod smoothed_gaussian_trend_filter_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod spearman_correlation_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod squeeze_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod stddev_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod stochastic_adaptive_d_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod stochastic_connors_rsi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod stochastic_distance_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod stochastic_money_flow_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod trend_continuation_factor_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod trend_direction_force_index_wrapper;
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub mod trend_follower_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod trend_trigger_factor_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod twiggs_money_flow_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ui_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod var_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod velocity_acceleration_indicator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod velocity_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volatility_quality_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volatility_ratio_adaptive_rsx_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volume_energy_reservoirs_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volume_weighted_relative_strength_index_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volume_weighted_rsi_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volume_zone_oscillator_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vosc_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod voss_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vpci_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod wad_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod yang_zhang_volatility_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod zscore_wrapper;

#[cfg(feature = "cuda-build-native")]
pub use oscillators::{CudaDecOsc, CudaDecOscError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::{CudaFisher, CudaFisherError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::{CudaIftRsi, CudaIftRsiError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::{CudaMfi, CudaMfiError};

#[cfg(feature = "cuda-build-native")]
pub use bollinger_bands_width_wrapper::{CudaBbw, CudaBbwError};
#[cfg(feature = "cuda-build-native")]
pub use bull_power_vs_bear_power_wrapper::{
    CudaBullPowerVsBearPower, CudaBullPowerVsBearPowerError,
};
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub use chande_wrapper::CudaChandeError;
#[cfg(feature = "cuda-build-native")]
pub use cksp_wrapper::{CudaCksp, CudaCkspError};
#[cfg(feature = "cuda-build-native")]
pub use decisionpoint_breadth_swenlin_trading_oscillator_wrapper::{
    CudaDecisionPointBreadthSwenlinTradingOscillator,
    CudaDecisionPointBreadthSwenlinTradingOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use demand_index_wrapper::{CudaDemandIndex, CudaDemandIndexError};
#[cfg(feature = "cuda-build-native")]
pub use deviation_wrapper::{CudaDeviation, CudaDeviationError};
#[cfg(feature = "cuda-build-native")]
pub use devstop_wrapper::{CudaDevStop, CudaDevStopError};
#[cfg(feature = "cuda-build-native")]
pub use didi_index_wrapper::{CudaDidiIndex, CudaDidiIndexError};
#[cfg(feature = "cuda-build-native")]
pub use directional_imbalance_index_wrapper::{
    CudaDirectionalImbalanceIndex, CudaDirectionalImbalanceIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use dual_ulcer_index_wrapper::{CudaDualUlcerIndex, CudaDualUlcerIndexError};
#[cfg(feature = "cuda-build-native")]
pub use ehlers_fm_demodulator_wrapper::{CudaEhlersFmDemodulator, CudaEhlersFmDemodulatorError};
#[cfg(feature = "cuda-build-native")]
pub use emd_trend_wrapper::{CudaEmdTrend, CudaEmdTrendError};
#[cfg(feature = "cuda-build-native")]
pub use emd_wrapper::{CudaEmd, CudaEmdBatchResult, CudaEmdError, DeviceArrayF32Triple};
#[cfg(feature = "cuda-build-native")]
pub use exponential_trend_wrapper::{CudaExponentialTrend, CudaExponentialTrendError};
#[cfg(feature = "cuda-build-native")]
pub use fibonacci_trailing_stop_wrapper::{
    CudaFibonacciTrailingStop, CudaFibonacciTrailingStopError,
};
#[cfg(feature = "cuda-build-native")]
pub use forward_backward_exponential_oscillator_wrapper::{
    CudaForwardBackwardExponentialOscillator, CudaForwardBackwardExponentialOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use fractal_dimension_index_wrapper::{
    CudaFractalDimensionIndex, CudaFractalDimensionIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use fvg_positioning_average_wrapper::{
    CudaFvgPositioningAverage, CudaFvgPositioningAverageError,
};
#[cfg(feature = "cuda-build-native")]
pub use fvg_trailing_stop_wrapper::{CudaFvgTs, CudaFvgTsError};
#[cfg(feature = "cuda-build-native")]
pub use garman_klass_volatility_wrapper::{
    CudaGarmanKlassBatchResult, CudaGarmanKlassVolatility, CudaGarmanKlassVolatilityError,
};
#[cfg(feature = "cuda-build-native")]
pub use geometric_bias_oscillator_wrapper::{
    CudaGeometricBiasOscillator, CudaGeometricBiasOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use gmma_oscillator_wrapper::{CudaGmmaOscillator, CudaGmmaOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use gopalakrishnan_range_index_wrapper::{
    CudaGopalakrishnanRangeIndex, CudaGopalakrishnanRangeIndexBatchResult,
    CudaGopalakrishnanRangeIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use grover_llorens_cycle_oscillator_wrapper::{
    CudaGroverLlorensCycleOscillator, CudaGroverLlorensCycleOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use historical_volatility_percentile_wrapper::{
    CudaHistoricalVolatilityPercentile, CudaHistoricalVolatilityPercentileError,
};
#[cfg(feature = "cuda-build-native")]
pub use historical_volatility_rank_wrapper::{
    CudaHistoricalVolatilityRank, CudaHistoricalVolatilityRankError,
};
#[cfg(feature = "cuda-build-native")]
pub use historical_volatility_wrapper::{
    CudaHistoricalVolatility, CudaHistoricalVolatilityBatchResult, CudaHistoricalVolatilityError,
};
#[cfg(feature = "cuda-build-native")]
pub use hull_butterfly_oscillator_wrapper::{
    CudaHullButterflyOscillator, CudaHullButterflyOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use intraday_momentum_index_wrapper::{
    CudaIntradayMomentumIndex, CudaIntradayMomentumIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use kaufmanstop_wrapper::{CudaKaufmanstop, CudaKaufmanstopError};
#[cfg(feature = "cuda-build-native")]
pub use l1_ehlers_phasor_wrapper::{CudaL1EhlersPhasor, CudaL1EhlersPhasorError};
#[cfg(feature = "cuda-build-native")]
pub use l2_ehlers_signal_to_noise_wrapper::{
    CudaL2EhlersSignalToNoise, CudaL2EhlersSignalToNoiseError,
};
#[cfg(feature = "cuda-build-native")]
pub use leavitt_convolution_acceleration_wrapper::{
    CudaLeavittConvolutionAcceleration, CudaLeavittConvolutionAccelerationError,
};
#[cfg(feature = "cuda-build-native")]
pub use linear_correlation_oscillator_wrapper::{
    CudaLinearCorrelationOscillator, CudaLinearCorrelationOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use linear_regression_intensity_wrapper::{
    CudaLinearRegressionIntensity, CudaLinearRegressionIntensityError,
};
#[cfg(feature = "cuda-build-native")]
pub use macd_wave_signal_pro_wrapper::{CudaMacdWaveSignalPro, CudaMacdWaveSignalProError};
#[cfg(feature = "cuda-build-native")]
pub use mass_wrapper::{CudaMass, CudaMassError};
#[cfg(feature = "cuda-build-native")]
pub use mean_ad_wrapper::{CudaMeanAd, CudaMeanAdError};
#[cfg(feature = "cuda-build-native")]
pub use medium_ad_wrapper::{CudaMediumAd, CudaMediumAdError};
#[cfg(feature = "cuda-build-native")]
pub use minmax_wrapper::{CudaMinmax, CudaMinmaxError};
#[cfg(feature = "cuda-build-native")]
pub use mod_god_mode_wrapper::{CudaModGodMode, CudaModGodModeBatchResult};
#[cfg(feature = "cuda-build-native")]
pub use momentum_ratio_oscillator_wrapper::{
    CudaMomentumRatioOscillator, CudaMomentumRatioOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use monotonicity_index_wrapper::{CudaMonotonicityIndex, CudaMonotonicityIndexError};
#[cfg(feature = "cuda-build-native")]
pub use moving_averages::{
    CudaApo, CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma,
    CudaHmaError, CudaLinearregSlope, CudaLinearregSlopeError, CudaLinreg, CudaLinregError,
    CudaLinregIntercept, CudaLinregInterceptError, CudaNma, CudaNmaError, CudaSma, CudaSmaError,
    CudaSuperSmoother, CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaTsf,
    CudaTsfError, CudaVidya, CudaVidyaError, CudaVlma, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError,
};
#[cfg(feature = "cuda-build-native")]
pub use natr_wrapper::{CudaNatr, CudaNatrError};
#[cfg(feature = "cuda-build-native")]
pub use neighboring_trailing_stop_wrapper::{
    CudaNeighboringTrailingStop, CudaNeighboringTrailingStopError,
};
#[cfg(feature = "cuda-build-native")]
pub use neoethos_f64_wrapper::{
    CudaF64IndicatorError, CudaF64Indicators, F64_EXACT_MATH_AUTHORITY_V3, F64Inputs, F64Kernel,
    F64NamedDeviceOutput, F64NamedOutputsResult, F64ResidentNamedPartsV3,
    F64ResidentObservedRouteManifestV3, F64ResidentSingleSweepAllocationPlanV4,
    F64ResidentSweepResultV3, F64SweepResult, MFI_MAX_PERIOD,
    preflight_resident_single_sweep_allocation_v4,
};
#[cfg(feature = "cuda-build-native")]
pub use net_myrsi_wrapper::{CudaNetMyrsi, CudaNetMyrsiError};
#[cfg(feature = "cuda-build-native")]
pub use nonlinear_regression_zero_lag_moving_average_wrapper::{
    CudaNonlinearRegressionZeroLagMovingAverage, CudaNonlinearRegressionZeroLagMovingAverageError,
};
#[cfg(feature = "cuda-build-native")]
pub use normalized_volume_true_range_wrapper::{
    CudaNormalizedVolumeTrueRange, CudaNormalizedVolumeTrueRangeError,
};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::adosc_wrapper::{CudaAdosc, CudaAdoscError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::ao_wrapper::{CudaAo, CudaAoError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::cfo_wrapper::{CudaCfo, CudaCfoError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::coppock_wrapper::{CudaCoppock, CudaCoppockError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::dpo_wrapper::{CudaDpo, CudaDpoError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::fosc_wrapper::{CudaFosc, CudaFoscError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::gatorosc_wrapper::{CudaGatorOsc, CudaGatorOscError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::kvo_wrapper::{CudaKvo, CudaKvoError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::macd_wrapper::{CudaMacd, CudaMacdError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::ppo_wrapper::{CudaPpo, CudaPpoError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::tsi_wrapper::{CudaTsi, CudaTsiError};
#[cfg(feature = "cuda-build-native")]
pub use parkinson_volatility_wrapper::{
    CudaParkinsonVolatility, CudaParkinsonVolatilityBatchResult, CudaParkinsonVolatilityError,
    ParkinsonDeviceArrayF32Pair,
};
#[cfg(feature = "cuda-build-native")]
pub use pattern_recognition_wrapper::{
    CudaPatternRecognition, CudaPatternRecognitionError, DevicePatternFeatures, NativeSubsetRows,
};
#[cfg(feature = "cuda-build-native")]
pub use percentile_nearest_rank_wrapper::{CudaPercentileNearestRank, CudaPnrError};
#[cfg(feature = "cuda-build-native")]
pub use polynomial_regression_extrapolation_wrapper::{
    CudaPolynomialRegressionExtrapolation, CudaPolynomialRegressionExtrapolationError,
};
#[cfg(feature = "cuda-build-native")]
pub use premier_rsi_oscillator_wrapper::{CudaPremierRsiOscillator, CudaPremierRsiOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use price_density_market_noise_wrapper::{
    CudaPriceDensityMarketNoise, CudaPriceDensityMarketNoiseError,
};
#[cfg(feature = "cuda-build-native")]
pub use projection_oscillator_wrapper::{CudaProjectionOscillator, CudaProjectionOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use psychological_line_wrapper::{CudaPsychologicalLine, CudaPsychologicalLineError};
#[cfg(feature = "cuda-build-native")]
pub use qqe_weighted_oscillator_wrapper::{
    CudaQqeWeightedOscillator, CudaQqeWeightedOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use random_walk_index_wrapper::{CudaRandomWalkIndex, CudaRandomWalkIndexError};
#[cfg(feature = "cuda-build-native")]
pub use range_filter_wrapper::{CudaRangeFilter, CudaRangeFilterError, DeviceRangeFilterTrio};
#[cfg(feature = "cuda-build-native")]
pub use rank_correlation_index_wrapper::{CudaRankCorrelationIndex, CudaRankCorrelationIndexError};
#[cfg(feature = "cuda-build-native")]
pub use regression_slope_oscillator_wrapper::{
    CudaRegressionSlopeOscillator, CudaRegressionSlopeOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use relative_strength_index_wave_indicator_wrapper::{
    CudaRelativeStrengthIndexWaveIndicator, CudaRelativeStrengthIndexWaveIndicatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use reversal_signals_wrapper::{CudaReversalSignals, CudaReversalSignalsError};
#[cfg(feature = "cuda-build-native")]
pub use rogers_satchell_volatility_wrapper::{
    CudaRogersSatchellBatchResult, CudaRogersSatchellManySeriesResult,
    CudaRogersSatchellVolatility, CudaRogersSatchellVolatilityError,
    DeviceArrayF32Pair as RogersSatchellDeviceArrayF32Pair,
};
#[cfg(feature = "cuda-build-native")]
pub use rolling_skewness_kurtosis_wrapper::{
    CudaRollingSkewnessKurtosis, CudaRollingSkewnessKurtosisError,
};
#[cfg(feature = "cuda-build-native")]
pub use rolling_z_score_trend_wrapper::{CudaRollingZScoreTrend, CudaRollingZScoreTrendError};
#[cfg(feature = "cuda-build-native")]
pub use sar_wrapper::{CudaSar, CudaSarError};
#[cfg(feature = "cuda-build-native")]
pub use smoothed_gaussian_trend_filter_wrapper::{
    CudaSmoothedGaussianTrendFilter, CudaSmoothedGaussianTrendFilterError,
};
#[cfg(feature = "cuda-build-native")]
pub use spearman_correlation_wrapper::{CudaSpearmanCorrelation, CudaSpearmanCorrelationError};
#[cfg(feature = "cuda-build-native")]
pub use squeeze_index_wrapper::{CudaSqueezeIndex, CudaSqueezeIndexError};
#[cfg(feature = "cuda-build-native")]
pub use stochastic_adaptive_d_wrapper::{CudaStochasticAdaptiveD, CudaStochasticAdaptiveDError};
#[cfg(feature = "cuda-build-native")]
pub use stochastic_connors_rsi_wrapper::{CudaStochasticConnorsRsi, CudaStochasticConnorsRsiError};
#[cfg(feature = "cuda-build-native")]
pub use stochastic_distance_wrapper::{CudaStochasticDistance, CudaStochasticDistanceError};
#[cfg(feature = "cuda-build-native")]
pub use stochastic_money_flow_index_wrapper::{
    CudaStochasticMoneyFlowIndex, CudaStochasticMoneyFlowIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use trend_continuation_factor_wrapper::{
    CudaTrendContinuationFactor, CudaTrendContinuationFactorError,
};
#[cfg(feature = "cuda-build-native")]
pub use trend_direction_force_index_wrapper::{
    CudaTrendDirectionForceIndex, CudaTrendDirectionForceIndexError,
};
#[cfg(feature = "cuda-build-native")]
#[cfg(feature = "cuda-build-native")]
pub use trend_follower_wrapper::{CudaTrendFollower, CudaTrendFollowerError};
#[cfg(feature = "cuda-build-native")]
pub use trend_trigger_factor_wrapper::{CudaTrendTriggerFactor, CudaTrendTriggerFactorError};
#[cfg(feature = "cuda-build-native")]
pub use twiggs_money_flow_wrapper::{CudaTwiggsMoneyFlow, CudaTwiggsMoneyFlowError};
#[cfg(feature = "cuda-build-native")]
pub use var_wrapper::{CudaVar, CudaVarError};
#[cfg(feature = "cuda-build-native")]
pub use velocity_acceleration_indicator_wrapper::{
    CudaVelocityAccelerationIndicator, CudaVelocityAccelerationIndicatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use velocity_wrapper::{CudaVelocity, CudaVelocityError};
#[cfg(feature = "cuda-build-native")]
pub use vi_wrapper::{CudaVi, CudaViError};
#[cfg(feature = "cuda-build-native")]
pub use volatility_quality_index_wrapper::{
    CudaVolatilityQualityIndex, CudaVolatilityQualityIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use volatility_ratio_adaptive_rsx_wrapper::{
    CudaVolatilityRatioAdaptiveRsx, CudaVolatilityRatioAdaptiveRsxError,
};
#[cfg(feature = "cuda-build-native")]
pub use volume_energy_reservoirs_wrapper::{
    CudaVolumeEnergyReservoirs, CudaVolumeEnergyReservoirsError,
};
#[cfg(feature = "cuda-build-native")]
pub use volume_weighted_relative_strength_index_wrapper::{
    CudaVolumeWeightedRelativeStrengthIndex, CudaVolumeWeightedRelativeStrengthIndexError,
};
#[cfg(feature = "cuda-build-native")]
pub use volume_weighted_rsi_wrapper::{CudaVolumeWeightedRsi, CudaVolumeWeightedRsiError};
#[cfg(feature = "cuda-build-native")]
pub use volume_zone_oscillator_wrapper::{CudaVolumeZoneOscillator, CudaVolumeZoneOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use voss_wrapper::{CudaVoss, CudaVossError};
#[cfg(feature = "cuda-build-native")]
pub use vpci_wrapper::{CudaVpci, CudaVpciError};
#[cfg(feature = "cuda-build-native")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda-build-native")]
pub use yang_zhang_volatility_wrapper::{
    CudaYangZhangBatchResult, CudaYangZhangVolatility, CudaYangZhangVolatilityError,
};
#[cfg(feature = "cuda-build-native")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda-build-native")]
pub mod linearreg_angle_wrapper;
#[cfg(feature = "cuda-build-native")]
pub use bollinger_bands_wrapper::{CudaBollingerBands, CudaBollingerError};
#[cfg(feature = "cuda-build-native")]
pub use chandelier_exit_wrapper::{CudaCeError, CudaChandelierExit};
#[cfg(feature = "cuda-build-native")]
pub use correl_hl_wrapper::{CudaCorrelHl, CudaCorrelHlError};
#[cfg(feature = "cuda-build-native")]
pub use damiani_volatmeter_wrapper::{CudaDamianiError, CudaDamianiVolatmeter};
#[cfg(feature = "cuda-build-native")]
pub use efi_wrapper::{CudaEfi, CudaEfiError};
#[cfg(feature = "cuda-build-native")]
pub use half_causal_estimator_wrapper::{CudaHalfCausalEstimator, CudaHalfCausalEstimatorError};
#[cfg(feature = "cuda-build-native")]
pub use halftrend_wrapper::{CudaHalftrend, CudaHalftrendError};
#[cfg(feature = "cuda-build-native")]
pub use kurtosis_wrapper::{CudaKurtosis, CudaKurtosisError};
#[cfg(feature = "cuda-build-native")]
pub use linearreg_angle_wrapper::{CudaLinearregAngle, CudaLinearregAngleError};
#[cfg(feature = "cuda-build-native")]
pub use lpc_wrapper::{
    BatchKernelPolicy as LpcBatchKernelPolicy, CudaLpc, CudaLpcError, CudaLpcPolicy,
    ManySeriesKernelPolicy as LpcManySeriesKernelPolicy,
};
#[cfg(feature = "cuda-build-native")]
pub use obv_wrapper::{CudaObv, CudaObvError};
#[cfg(feature = "cuda-build-native")]
pub use on_balance_volume_oscillator_wrapper::{
    CudaOnBalanceVolumeOscillator, CudaOnBalanceVolumeOscillatorError,
};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::cg_wrapper::{CudaCg, CudaCgError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::cmo_wrapper::{CudaCmo, CudaCmoError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::dti_wrapper::{CudaDti, CudaDtiError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::emv_wrapper::{CudaEmv, CudaEmvError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::kdj_wrapper::{CudaKdj, CudaKdjError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::reverse_rsi_wrapper::{CudaReverseRsi, CudaReverseRsiError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::squeeze_momentum_wrapper::{CudaSmiError, CudaSqueezeMomentum};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::stochf_wrapper::{CudaStochf, CudaStochfError};
#[cfg(feature = "cuda-build-native")]
pub use oscillators::ttm_squeeze_wrapper::{CudaTtmSqueeze, CudaTtmSqueezeError};
#[cfg(feature = "cuda-build-native")]
pub use pretty_good_oscillator_wrapper::{CudaPrettyGoodOscillator, CudaPrettyGoodOscillatorError};
#[cfg(feature = "cuda-build-native")]
pub use safezonestop_wrapper::{CudaSafeZoneStop, CudaSafeZoneStopError};
#[cfg(feature = "cuda-build-native")]
pub use stddev_wrapper::{CudaStddev, CudaStddevError};
#[cfg(feature = "cuda-build-native")]
pub use ui_wrapper::{CudaUi, CudaUiError};
#[cfg(feature = "cuda-build-native")]
pub use vosc_wrapper::{
    BatchKernelPolicy as VoscBatchKernelPolicy, CudaVosc, CudaVoscError, CudaVoscPolicy,
    ManySeriesKernelPolicy as VoscManySeriesKernelPolicy,
};

#[cfg(all(feature = "cuda-build-native", test))]
pub(crate) struct CudaTestLock {
    _guard: Option<MutexGuard<'static, ()>>,
}

#[cfg(all(feature = "cuda-build-native", test))]
impl Drop for CudaTestLock {
    fn drop(&mut self) {
        CUDA_TEST_LOCK_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0);
            depth.set(current.saturating_sub(1));
        });
    }
}

#[cfg(all(feature = "cuda-build-native", test))]
thread_local! {
    static CUDA_TEST_LOCK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(feature = "cuda-build-native", test))]
pub(crate) fn cuda_test_lock() -> CudaTestLock {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = CUDA_TEST_LOCK_DEPTH.with(|depth| {
        let current = depth.get();
        depth.set(current + 1);
        if current == 0 {
            Some(
                LOCK.get_or_init(|| Mutex::new(()))
                    .lock()
                    .unwrap_or_else(|err| err.into_inner()),
            )
        } else {
            None
        }
    });
    CudaTestLock { _guard: guard }
}

#[inline]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda-build-native")]
    {
        static CUDA_AVAILABLE_CACHED: OnceLock<bool> = OnceLock::new();
        static CUDA_PROBE_CONTEXT_0: OnceLock<Option<cust::context::Context>> = OnceLock::new();

        if std::env::var("CUDA_PLACEHOLDER_ON_FAIL").ok().as_deref() == Some("1")
            || std::env::var("CUDA_FORCE_SKIP").ok().as_deref() == Some("1")
        {
            return false;
        }
        *CUDA_AVAILABLE_CACHED.get_or_init(|| {
            use cust::{
                device::Device,
                function::BlockSize,
                function::GridSize,
                prelude::CudaFlags,
                stream::{Stream, StreamFlags},
            };

            let debug = std::env::var("CUDA_PROBE_DEBUG").ok().as_deref() == Some("1");

            if let Err(err) = cust::init(CudaFlags::empty()) {
                if debug {
                    eprintln!("cuda_available: cust::init failed: {err:?}");
                }
                return false;
            }

            let ndev = match Device::num_devices() {
                Ok(n) => n,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: Device::num_devices failed: {err:?}");
                    }
                    0
                }
            };
            if ndev == 0 {
                if debug {
                    eprintln!("cuda_available: no CUDA devices reported");
                }
                return false;
            }

            let device = match Device::get_device(0) {
                Ok(d) => d,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: Device::get_device(0) failed: {err:?}");
                    }
                    return false;
                }
            };

            let ctx0 =
                CUDA_PROBE_CONTEXT_0.get_or_init(|| match cust::context::Context::new(device) {
                    Ok(c) => Some(c),
                    Err(err) => {
                        if debug {
                            eprintln!("cuda_available: Context::new failed: {err:?}");
                        }
                        None
                    }
                });
            if ctx0.is_none() {
                return false;
            }

            if debug {
                eprintln!("cuda_available: loading exact native-SASS probe cubin");
            }
            let module = match crate::load_cuda_embedded_module!("vector_ta_native_probe") {
                Ok(module) => module,
                Err(error) => {
                    if debug {
                        eprintln!("cuda_available: native-SASS probe load failed: {error:?}");
                    }
                    return false;
                }
            };
            let func = match module.get_function("vector_ta_native_probe") {
                Ok(f) => f,
                Err(err) => {
                    if debug {
                        eprintln!(
                            "cuda_available: module.get_function(\"vector_ta_native_probe\") \
                             failed: {err:?}"
                        );
                    }
                    return false;
                }
            };
            let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
                Ok(s) => s,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: Stream::new failed: {err:?}");
                    }
                    return false;
                }
            };
            unsafe {
                let args: &mut [*mut std::ffi::c_void] = &mut [];
                if let Err(err) =
                    stream.launch(&func, GridSize::xy(1, 1), BlockSize::xyz(1, 1, 1), 0, args)
                {
                    if debug {
                        eprintln!("cuda_available: stream.launch failed: {err:?}");
                    }
                    return false;
                }
            }
            if let Err(err) = stream.synchronize() {
                if debug {
                    eprintln!("cuda_available: stream.synchronize failed: {err:?}");
                }
                return false;
            }
            true
        })
    }

    #[cfg(not(feature = "cuda-build-native"))]
    {
        false
    }
}

#[inline]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda-build-native")]
    {
        use cust::{device::Device, prelude::CudaFlags};
        if cust::init(CudaFlags::empty()).is_err() {
            return 0;
        }
        match Device::num_devices() {
            Ok(n) => n as usize,
            Err(_) => 0,
        }
    }

    #[cfg(not(feature = "cuda-build-native"))]
    {
        0
    }
}
