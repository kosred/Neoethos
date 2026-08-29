use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::str::FromStr;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 14;
const DEFAULT_MA_TYPE: BullsVBearsMaType = BullsVBearsMaType::Ema;
const DEFAULT_CALCULATION_METHOD: BullsVBearsCalculationMethod =
    BullsVBearsCalculationMethod::Normalized;
const DEFAULT_NORMALIZED_BARS_BACK: usize = 120;
const DEFAULT_RAW_ROLLING_PERIOD: usize = 50;
const DEFAULT_RAW_THRESHOLD_PERCENTILE: f64 = 95.0;
const DEFAULT_THRESHOLD_LEVEL: f64 = 80.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BullsVBearsMaType {
    Ema,
    Sma,
    Wma,
}

impl Default for BullsVBearsMaType {
    fn default() -> Self {
        DEFAULT_MA_TYPE
    }
}

impl BullsVBearsMaType {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Ema => "ema",
            Self::Sma => "sma",
            Self::Wma => "wma",
        }
    }

    #[inline(always)]
    fn warmup(self, period: usize) -> usize {
        match self {
            Self::Ema => 0,
            Self::Sma | Self::Wma => period.saturating_sub(1),
        }
    }
}

impl FromStr for BullsVBearsMaType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ema" => Ok(Self::Ema),
            "sma" => Ok(Self::Sma),
            "wma" => Ok(Self::Wma),
            _ => Err(format!("invalid ma_type: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BullsVBearsCalculationMethod {
    Normalized,
    Raw,
}

impl Default for BullsVBearsCalculationMethod {
    fn default() -> Self {
        DEFAULT_CALCULATION_METHOD
    }
}

impl BullsVBearsCalculationMethod {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Normalized => "normalized",
            Self::Raw => "raw",
        }
    }
}

impl FromStr for BullsVBearsCalculationMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "normalized" => Ok(Self::Normalized),
            "raw" => Ok(Self::Raw),
            _ => Err(format!("invalid calculation_method: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BullsVBearsData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct BullsVBearsOutput {
    pub value: Vec<f64>,
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub ma: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub bullish_signal: Vec<f64>,
    pub bearish_signal: Vec<f64>,
    pub zero_cross_up: Vec<f64>,
    pub zero_cross_down: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BullsVBearsOutputField {
    Value,
    Bull,
    Bear,
    Ma,
    Upper,
    Lower,
    BullishSignal,
    BearishSignal,
    ZeroCrossUp,
    ZeroCrossDown,
}

#[derive(Debug, Clone)]
pub struct BullsVBearsParams {
    pub period: Option<usize>,
    pub ma_type: Option<BullsVBearsMaType>,
    pub calculation_method: Option<BullsVBearsCalculationMethod>,
    pub normalized_bars_back: Option<usize>,
    pub raw_rolling_period: Option<usize>,
    pub raw_threshold_percentile: Option<f64>,
    pub threshold_level: Option<f64>,
}

impl Default for BullsVBearsParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            ma_type: Some(DEFAULT_MA_TYPE),
            calculation_method: Some(DEFAULT_CALCULATION_METHOD),
            normalized_bars_back: Some(DEFAULT_NORMALIZED_BARS_BACK),
            raw_rolling_period: Some(DEFAULT_RAW_ROLLING_PERIOD),
            raw_threshold_percentile: Some(DEFAULT_RAW_THRESHOLD_PERCENTILE),
            threshold_level: Some(DEFAULT_THRESHOLD_LEVEL),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BullsVBearsInput<'a> {
    pub data: BullsVBearsData<'a>,
    pub params: BullsVBearsParams,
}

impl<'a> BullsVBearsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: BullsVBearsParams) -> Self {
        Self {
            data: BullsVBearsData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: BullsVBearsParams,
    ) -> Self {
        Self {
            data: BullsVBearsData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, BullsVBearsParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BullsVBearsBuilder {
    period: Option<usize>,
    ma_type: Option<BullsVBearsMaType>,
    calculation_method: Option<BullsVBearsCalculationMethod>,
    normalized_bars_back: Option<usize>,
    raw_rolling_period: Option<usize>,
    raw_threshold_percentile: Option<f64>,
    threshold_level: Option<f64>,
    kernel: Kernel,
}

impl Default for BullsVBearsBuilder {
    fn default() -> Self {
        Self {
            period: None,
            ma_type: None,
            calculation_method: None,
            normalized_bars_back: None,
            raw_rolling_period: None,
            raw_threshold_percentile: None,
            threshold_level: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BullsVBearsBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, value: BullsVBearsMaType) -> Self {
        self.ma_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn calculation_method(mut self, value: BullsVBearsCalculationMethod) -> Self {
        self.calculation_method = Some(value);
        self
    }

    #[inline(always)]
    pub fn normalized_bars_back(mut self, value: usize) -> Self {
        self.normalized_bars_back = Some(value);
        self
    }

    #[inline(always)]
    pub fn raw_rolling_period(mut self, value: usize) -> Self {
        self.raw_rolling_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn raw_threshold_percentile(mut self, value: f64) -> Self {
        self.raw_threshold_percentile = Some(value);
        self
    }

    #[inline(always)]
    pub fn threshold_level(mut self, value: f64) -> Self {
        self.threshold_level = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<BullsVBearsOutput, BullsVBearsError> {
        let input = BullsVBearsInput::from_candles(
            candles,
            BullsVBearsParams {
                period: self.period,
                ma_type: self.ma_type,
                calculation_method: self.calculation_method,
                normalized_bars_back: self.normalized_bars_back,
                raw_rolling_period: self.raw_rolling_period,
                raw_threshold_percentile: self.raw_threshold_percentile,
                threshold_level: self.threshold_level,
            },
        );
        bulls_v_bears_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<BullsVBearsOutput, BullsVBearsError> {
        let input = BullsVBearsInput::from_slices(
            high,
            low,
            close,
            BullsVBearsParams {
                period: self.period,
                ma_type: self.ma_type,
                calculation_method: self.calculation_method,
                normalized_bars_back: self.normalized_bars_back,
                raw_rolling_period: self.raw_rolling_period,
                raw_threshold_percentile: self.raw_threshold_percentile,
                threshold_level: self.threshold_level,
            },
        );
        bulls_v_bears_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<BullsVBearsStream, BullsVBearsError> {
        BullsVBearsStream::try_new(BullsVBearsParams {
            period: self.period,
            ma_type: self.ma_type,
            calculation_method: self.calculation_method,
            normalized_bars_back: self.normalized_bars_back,
            raw_rolling_period: self.raw_rolling_period,
            raw_threshold_percentile: self.raw_threshold_percentile,
            threshold_level: self.threshold_level,
        })
    }
}

#[derive(Debug, Error)]
pub enum BullsVBearsError {
    #[error("bulls_v_bears: Input data slice is empty.")]
    EmptyInputData,
    #[error("bulls_v_bears: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "bulls_v_bears: Inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("bulls_v_bears: Invalid period: {period}")]
    InvalidPeriod { period: usize },
    #[error("bulls_v_bears: Invalid normalized_bars_back: {normalized_bars_back}")]
    InvalidNormalizedBarsBack { normalized_bars_back: usize },
    #[error("bulls_v_bears: Invalid raw_rolling_period: {raw_rolling_period}")]
    InvalidRawRollingPeriod { raw_rolling_period: usize },
    #[error("bulls_v_bears: Invalid raw_threshold_percentile: {raw_threshold_percentile}")]
    InvalidRawThresholdPercentile { raw_threshold_percentile: f64 },
    #[error("bulls_v_bears: Invalid threshold_level: {threshold_level}")]
    InvalidThresholdLevel { threshold_level: f64 },
    #[error("bulls_v_bears: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("bulls_v_bears: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("bulls_v_bears: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    period: usize,
    ma_type: BullsVBearsMaType,
    calculation_method: BullsVBearsCalculationMethod,
    normalized_bars_back: usize,
    raw_rolling_period: usize,
    raw_threshold_percentile: f64,
    threshold_level: f64,
}

#[inline(always)]
fn extract_hlc<'a>(
    input: &'a BullsVBearsInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), BullsVBearsError> {
    let (high, low, close) = match &input.data {
        BullsVBearsData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        BullsVBearsData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(BullsVBearsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(BullsVBearsError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    Ok((high, low, close))
}

#[inline(always)]
fn first_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
}

#[inline(always)]
fn resolve_params(params: &BullsVBearsParams) -> Result<ResolvedParams, BullsVBearsError> {
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if period == 0 {
        return Err(BullsVBearsError::InvalidPeriod { period });
    }
    let normalized_bars_back = params
        .normalized_bars_back
        .unwrap_or(DEFAULT_NORMALIZED_BARS_BACK);
    if normalized_bars_back == 0 {
        return Err(BullsVBearsError::InvalidNormalizedBarsBack {
            normalized_bars_back,
        });
    }
    let raw_rolling_period = params
        .raw_rolling_period
        .unwrap_or(DEFAULT_RAW_ROLLING_PERIOD);
    if raw_rolling_period == 0 {
        return Err(BullsVBearsError::InvalidRawRollingPeriod { raw_rolling_period });
    }
    let raw_threshold_percentile = params
        .raw_threshold_percentile
        .unwrap_or(DEFAULT_RAW_THRESHOLD_PERCENTILE);
    if !raw_threshold_percentile.is_finite() || !(80.0..=99.0).contains(&raw_threshold_percentile) {
        return Err(BullsVBearsError::InvalidRawThresholdPercentile {
            raw_threshold_percentile,
        });
    }
    let threshold_level = params.threshold_level.unwrap_or(DEFAULT_THRESHOLD_LEVEL);
    if !threshold_level.is_finite() || !(0.0..=100.0).contains(&threshold_level) {
        return Err(BullsVBearsError::InvalidThresholdLevel { threshold_level });
    }
    Ok(ResolvedParams {
        period,
        ma_type: params.ma_type.unwrap_or(DEFAULT_MA_TYPE),
        calculation_method: params
            .calculation_method
            .unwrap_or(DEFAULT_CALCULATION_METHOD),
        normalized_bars_back,
        raw_rolling_period,
        raw_threshold_percentile,
        threshold_level,
    })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a BullsVBearsInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        ResolvedParams,
        usize,
        Kernel,
    ),
    BullsVBearsError,
> {
    let (high, low, close) = extract_hlc(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid_hlc(high, low, close).ok_or(BullsVBearsError::AllValuesNaN)?;
    Ok((high, low, close, params, first, kernel.to_non_batch()))
}

#[inline(always)]
fn check_output_len(out: &[f64], expected: usize) -> Result<(), BullsVBearsError> {
    if out.len() != expected {
        return Err(BullsVBearsError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn fill_moving_average(
    close: &[f64],
    params: ResolvedParams,
    out_ma: &mut [f64],
) -> Result<(), BullsVBearsError> {
    let len = close.len();
    check_output_len(out_ma, len)?;

    match params.ma_type {
        BullsVBearsMaType::Ema => {
            let alpha = 2.0 / (params.period as f64 + 1.0);
            let mut prev = f64::NAN;
            for i in 0..len {
                let x = close[i];
                if !x.is_finite() {
                    prev = f64::NAN;
                    out_ma[i] = f64::NAN;
                    continue;
                }
                if prev.is_finite() {
                    prev += alpha * (x - prev);
                } else {
                    prev = x;
                }
                out_ma[i] = prev;
            }
        }
        BullsVBearsMaType::Sma => {
            let mut sum = 0.0;
            let mut finite_count = 0usize;
            for i in 0..len {
                let x = close[i];
                if x.is_finite() {
                    sum += x;
                    finite_count += 1;
                }
                if i >= params.period {
                    let old = close[i - params.period];
                    if old.is_finite() {
                        sum -= old;
                        finite_count -= 1;
                    }
                }
                out_ma[i] = if i + 1 >= params.period && finite_count == params.period {
                    sum / params.period as f64
                } else {
                    f64::NAN
                };
            }
        }
        BullsVBearsMaType::Wma => {
            let denom = (params.period * (params.period + 1) / 2) as f64;
            let mut window: VecDeque<f64> = VecDeque::with_capacity(params.period);
            let mut sum = 0.0;
            let mut finite_count = 0usize;
            let mut weighted = 0.0;
            let mut prev_full_valid = false;

            for i in 0..len {
                let x = close[i];
                let old_window_sum = sum;
                let popped = if window.len() == params.period {
                    let old = window.pop_front().unwrap();
                    if old.is_finite() {
                        sum -= old;
                        finite_count -= 1;
                    }
                    Some(old)
                } else {
                    None
                };
                window.push_back(x);
                if x.is_finite() {
                    sum += x;
                    finite_count += 1;
                }

                let full_valid = window.len() == params.period && finite_count == params.period;
                if full_valid {
                    if prev_full_valid && popped.is_some() && x.is_finite() {
                        weighted = weighted + params.period as f64 * x - old_window_sum;
                    } else {
                        weighted = 0.0;
                        for (idx, value) in window.iter().enumerate() {
                            weighted += *value * (idx + 1) as f64;
                        }
                    }
                    out_ma[i] = weighted / denom;
                    prev_full_valid = true;
                } else {
                    out_ma[i] = f64::NAN;
                    prev_full_valid = false;
                    weighted = 0.0;
                }
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn push_min_queue(queue: &mut VecDeque<(usize, f64)>, idx: usize, value: f64) {
    while let Some((_, back)) = queue.back() {
        if *back <= value {
            break;
        }
        queue.pop_back();
    }
    queue.push_back((idx, value));
}

#[inline(always)]
fn push_max_queue(queue: &mut VecDeque<(usize, f64)>, idx: usize, value: f64) {
    while let Some((_, back)) = queue.back() {
        if *back >= value {
            break;
        }
        queue.pop_back();
    }
    queue.push_back((idx, value));
}

#[inline(always)]
fn expire_queue(queue: &mut VecDeque<(usize, f64)>, min_index: usize) {
    while let Some((idx, _)) = queue.front() {
        if *idx >= min_index {
            break;
        }
        queue.pop_front();
    }
}

#[inline(always)]
fn compute_signals(
    out_value: &[f64],
    out_upper: &[f64],
    out_lower: &[f64],
    out_bullish_signal: &mut [f64],
    out_bearish_signal: &mut [f64],
    out_zero_cross_up: &mut [f64],
    out_zero_cross_down: &mut [f64],
) -> Result<(), BullsVBearsError> {
    let len = out_value.len();
    check_output_len(out_upper, len)?;
    check_output_len(out_lower, len)?;
    check_output_len(out_bullish_signal, len)?;
    check_output_len(out_bearish_signal, len)?;
    check_output_len(out_zero_cross_up, len)?;
    check_output_len(out_zero_cross_down, len)?;

    let mut prev_total = f64::NAN;
    for i in 0..len {
        let total = out_value[i];
        let upper = out_upper[i];
        let lower = out_lower[i];
        if total.is_finite() && upper.is_finite() && lower.is_finite() {
            out_bullish_signal[i] = if total > upper { 1.0 } else { 0.0 };
            out_bearish_signal[i] = if total < lower { 1.0 } else { 0.0 };
            out_zero_cross_up[i] = if prev_total.is_finite() && total > 0.0 && prev_total <= 0.0 {
                1.0
            } else {
                0.0
            };
            out_zero_cross_down[i] = if prev_total.is_finite() && total < 0.0 && prev_total >= 0.0 {
                1.0
            } else {
                0.0
            };
            prev_total = total;
        } else {
            out_bullish_signal[i] = f64::NAN;
            out_bearish_signal[i] = f64::NAN;
            out_zero_cross_up[i] = f64::NAN;
            out_zero_cross_down[i] = f64::NAN;
            prev_total = f64::NAN;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn bulls_v_bears_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    out_value: &mut [f64],
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_ma: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_bullish_signal: &mut [f64],
    out_bearish_signal: &mut [f64],
    out_zero_cross_up: &mut [f64],
    out_zero_cross_down: &mut [f64],
) -> Result<(), BullsVBearsError> {
    let len = close.len();
    check_output_len(out_value, len)?;
    check_output_len(out_bull, len)?;
    check_output_len(out_bear, len)?;
    check_output_len(out_ma, len)?;
    check_output_len(out_upper, len)?;
    check_output_len(out_lower, len)?;
    check_output_len(out_bullish_signal, len)?;
    check_output_len(out_bearish_signal, len)?;
    check_output_len(out_zero_cross_up, len)?;
    check_output_len(out_zero_cross_down, len)?;

    fill_moving_average(close, params, out_ma)?;

    for i in 0..len {
        let h = high[i];
        let l = low[i];
        let ma = out_ma[i];
        if h.is_finite() && l.is_finite() && ma.is_finite() {
            out_bull[i] = h - ma;
            out_bear[i] = ma - l;
        } else {
            out_bull[i] = f64::NAN;
            out_bear[i] = f64::NAN;
        }
    }

    match params.calculation_method {
        BullsVBearsCalculationMethod::Normalized => {
            let mut bull_min = VecDeque::new();
            let mut bull_max = VecDeque::new();
            let mut bear_min = VecDeque::new();
            let mut bear_max = VecDeque::new();

            for i in 0..len {
                out_upper[i] = params.threshold_level;
                out_lower[i] = -params.threshold_level;

                let bull = out_bull[i];
                let bear = out_bear[i];
                if !(bull.is_finite() && bear.is_finite()) {
                    bull_min.clear();
                    bull_max.clear();
                    bear_min.clear();
                    bear_max.clear();
                    out_value[i] = f64::NAN;
                    continue;
                }

                let min_index = i
                    .saturating_add(1)
                    .saturating_sub(params.normalized_bars_back);
                expire_queue(&mut bull_min, min_index);
                expire_queue(&mut bull_max, min_index);
                expire_queue(&mut bear_min, min_index);
                expire_queue(&mut bear_max, min_index);

                push_min_queue(&mut bull_min, i, bull);
                push_max_queue(&mut bull_max, i, bull);
                push_min_queue(&mut bear_min, i, bear);
                push_max_queue(&mut bear_max, i, bear);

                let bull_min_value = bull_min.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bull_max_value = bull_max.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bear_min_value = bear_min.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bear_max_value = bear_max.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bull_range = bull_max_value - bull_min_value;
                let bear_range = bear_max_value - bear_min_value;

                if bull_range > 0.0 && bear_range > 0.0 {
                    let norm_bull = ((bull - bull_min_value) / bull_range - 0.5) * 100.0;
                    let norm_bear = ((bear - bear_min_value) / bear_range - 0.5) * 100.0;
                    out_value[i] = norm_bull - norm_bear;
                } else {
                    out_value[i] = f64::NAN;
                }
            }
        }
        BullsVBearsCalculationMethod::Raw => {
            let mut raw_min = VecDeque::new();
            let mut raw_max = VecDeque::new();
            let upper_factor = params.raw_threshold_percentile / 100.0;
            let lower_factor = (100.0 - params.raw_threshold_percentile) / 100.0;

            for i in 0..len {
                let bull = out_bull[i];
                let bear = out_bear[i];
                out_value[i] = if bull.is_finite() && bear.is_finite() {
                    bull - bear
                } else {
                    f64::NAN
                };
            }

            for i in 0..len {
                let total = out_value[i];
                if !total.is_finite() {
                    raw_min.clear();
                    raw_max.clear();
                    out_upper[i] = f64::NAN;
                    out_lower[i] = f64::NAN;
                    continue;
                }

                let min_index = i
                    .saturating_add(1)
                    .saturating_sub(params.raw_rolling_period);
                expire_queue(&mut raw_min, min_index);
                expire_queue(&mut raw_max, min_index);

                push_min_queue(&mut raw_min, i, total);
                push_max_queue(&mut raw_max, i, total);

                let lowest = raw_min.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let highest = raw_max.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                if lowest.is_finite() && highest.is_finite() {
                    let range = highest - lowest;
                    out_upper[i] = lowest + range * upper_factor;
                    out_lower[i] = lowest + range * lower_factor;
                } else {
                    out_upper[i] = f64::NAN;
                    out_lower[i] = f64::NAN;
                }
            }
        }
    }

    compute_signals(
        out_value,
        out_upper,
        out_lower,
        out_bullish_signal,
        out_bearish_signal,
        out_zero_cross_up,
        out_zero_cross_down,
    )?;
    Ok(())
}

#[inline]
pub fn bulls_v_bears(input: &BullsVBearsInput) -> Result<BullsVBearsOutput, BullsVBearsError> {
    bulls_v_bears_with_kernel(input, Kernel::Auto)
}

pub fn bulls_v_bears_with_kernel(
    input: &BullsVBearsInput,
    kernel: Kernel,
) -> Result<BullsVBearsOutput, BullsVBearsError> {
    let (high, low, close, params, _first, _kernel) = validate_input(input, kernel)?;
    let len = close.len();
    let warm = params.ma_type.warmup(params.period);

    let mut value = alloc_with_nan_prefix(len, warm);
    let mut bull = alloc_with_nan_prefix(len, warm);
    let mut bear = alloc_with_nan_prefix(len, warm);
    let mut ma = alloc_with_nan_prefix(len, warm);
    let mut upper = alloc_with_nan_prefix(len, warm);
    let mut lower = alloc_with_nan_prefix(len, warm);
    let mut bullish_signal = alloc_with_nan_prefix(len, warm);
    let mut bearish_signal = alloc_with_nan_prefix(len, warm);
    let mut zero_cross_up = alloc_with_nan_prefix(len, warm);
    let mut zero_cross_down = alloc_with_nan_prefix(len, warm);

    bulls_v_bears_compute_into(
        high,
        low,
        close,
        params,
        &mut value,
        &mut bull,
        &mut bear,
        &mut ma,
        &mut upper,
        &mut lower,
        &mut bullish_signal,
        &mut bearish_signal,
        &mut zero_cross_up,
        &mut zero_cross_down,
    )?;

    Ok(BullsVBearsOutput {
        value,
        bull,
        bear,
        ma,
        upper,
        lower,
        bullish_signal,
        bearish_signal,
        zero_cross_up,
        zero_cross_down,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bulls_v_bears_into(
    out_value: &mut [f64],
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_ma: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_bullish_signal: &mut [f64],
    out_bearish_signal: &mut [f64],
    out_zero_cross_up: &mut [f64],
    out_zero_cross_down: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: BullsVBearsParams,
) -> Result<(), BullsVBearsError> {
    bulls_v_bears_into_slice(
        out_value,
        out_bull,
        out_bear,
        out_ma,
        out_upper,
        out_lower,
        out_bullish_signal,
        out_bearish_signal,
        out_zero_cross_up,
        out_zero_cross_down,
        high,
        low,
        close,
        params,
        Kernel::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn bulls_v_bears_into_slice(
    out_value: &mut [f64],
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_ma: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_bullish_signal: &mut [f64],
    out_bearish_signal: &mut [f64],
    out_zero_cross_up: &mut [f64],
    out_zero_cross_down: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: BullsVBearsParams,
    kernel: Kernel,
) -> Result<(), BullsVBearsError> {
    let input = BullsVBearsInput::from_slices(high, low, close, params);
    let (_, _, _, resolved, _, _) = validate_input(&input, kernel)?;
    bulls_v_bears_compute_into(
        high,
        low,
        close,
        resolved,
        out_value,
        out_bull,
        out_bear,
        out_ma,
        out_upper,
        out_lower,
        out_bullish_signal,
        out_bearish_signal,
        out_zero_cross_up,
        out_zero_cross_down,
    )
}

pub fn bulls_v_bears_output_into_slice(
    dst: &mut [f64],
    input: &BullsVBearsInput,
    kernel: Kernel,
    field: BullsVBearsOutputField,
) -> Result<(), BullsVBearsError> {
    let (high, low, close, _params, _first, _kernel) = validate_input(input, kernel)?;
    if dst.len() != close.len() {
        return Err(BullsVBearsError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    let mut stream = BullsVBearsStream::try_new(input.params.clone())?;
    for i in 0..close.len() {
        let point = stream.update(high[i], low[i], close[i]);
        dst[i] = match field {
            BullsVBearsOutputField::Value => point.0,
            BullsVBearsOutputField::Bull => point.1,
            BullsVBearsOutputField::Bear => point.2,
            BullsVBearsOutputField::Ma => point.3,
            BullsVBearsOutputField::Upper => point.4,
            BullsVBearsOutputField::Lower => point.5,
            BullsVBearsOutputField::BullishSignal => point.6,
            BullsVBearsOutputField::BearishSignal => point.7,
            BullsVBearsOutputField::ZeroCrossUp => point.8,
            BullsVBearsOutputField::ZeroCrossDown => point.9,
        };
    }
    Ok(())
}

#[derive(Debug, Clone)]
enum StreamMaState {
    Ema {
        alpha: f64,
        prev: f64,
    },
    Sma {
        period: usize,
        window: VecDeque<f64>,
        sum: f64,
        finite_count: usize,
    },
    Wma {
        period: usize,
        denom: f64,
        window: VecDeque<f64>,
        sum: f64,
        finite_count: usize,
        weighted: f64,
        prev_full_valid: bool,
    },
}

impl StreamMaState {
    fn new(params: ResolvedParams) -> Self {
        match params.ma_type {
            BullsVBearsMaType::Ema => Self::Ema {
                alpha: 2.0 / (params.period as f64 + 1.0),
                prev: f64::NAN,
            },
            BullsVBearsMaType::Sma => Self::Sma {
                period: params.period,
                window: VecDeque::<f64>::with_capacity(params.period),
                sum: 0.0,
                finite_count: 0,
            },
            BullsVBearsMaType::Wma => Self::Wma {
                period: params.period,
                denom: (params.period * (params.period + 1) / 2) as f64,
                window: VecDeque::<f64>::with_capacity(params.period),
                sum: 0.0,
                finite_count: 0,
                weighted: 0.0,
                prev_full_valid: false,
            },
        }
    }

    fn update(&mut self, close: f64) -> f64 {
        match self {
            Self::Ema { alpha, prev } => {
                if !close.is_finite() {
                    *prev = f64::NAN;
                    return f64::NAN;
                }
                if prev.is_finite() {
                    *prev += *alpha * (close - *prev);
                } else {
                    *prev = close;
                }
                *prev
            }
            Self::Sma {
                period,
                window,
                sum,
                finite_count,
            } => {
                if window.len() == *period {
                    let old = window.pop_front().unwrap();
                    if old.is_finite() {
                        *sum -= old;
                        *finite_count -= 1;
                    }
                }
                window.push_back(close);
                if close.is_finite() {
                    *sum += close;
                    *finite_count += 1;
                }
                if window.len() == *period && *finite_count == *period {
                    *sum / *period as f64
                } else {
                    f64::NAN
                }
            }
            Self::Wma {
                period,
                denom,
                window,
                sum,
                finite_count,
                weighted,
                prev_full_valid,
            } => {
                let old_window_sum = *sum;
                let popped = if window.len() == *period {
                    let old = window.pop_front().unwrap();
                    if old.is_finite() {
                        *sum -= old;
                        *finite_count -= 1;
                    }
                    Some(old)
                } else {
                    None
                };
                window.push_back(close);
                if close.is_finite() {
                    *sum += close;
                    *finite_count += 1;
                }
                let full_valid = window.len() == *period && *finite_count == *period;
                if full_valid {
                    if *prev_full_valid && popped.is_some() && close.is_finite() {
                        *weighted = *weighted + *period as f64 * close - old_window_sum;
                    } else {
                        *weighted = 0.0;
                        for (idx, value) in window.iter().enumerate() {
                            *weighted += *value * (idx + 1) as f64;
                        }
                    }
                    *prev_full_valid = true;
                    *weighted / *denom
                } else {
                    *prev_full_valid = false;
                    *weighted = 0.0;
                    f64::NAN
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BullsVBearsStream {
    params: ResolvedParams,
    index: usize,
    ma_state: StreamMaState,
    bull_min: VecDeque<(usize, f64)>,
    bull_max: VecDeque<(usize, f64)>,
    bear_min: VecDeque<(usize, f64)>,
    bear_max: VecDeque<(usize, f64)>,
    raw_min: VecDeque<(usize, f64)>,
    raw_max: VecDeque<(usize, f64)>,
    prev_total: f64,
}

impl BullsVBearsStream {
    pub fn try_new(params: BullsVBearsParams) -> Result<Self, BullsVBearsError> {
        let params = resolve_params(&params)?;
        Ok(Self {
            params,
            index: 0,
            ma_state: StreamMaState::new(params),
            bull_min: VecDeque::new(),
            bull_max: VecDeque::new(),
            bear_min: VecDeque::new(),
            bear_max: VecDeque::new(),
            raw_min: VecDeque::new(),
            raw_max: VecDeque::new(),
            prev_total: f64::NAN,
        })
    }

    #[inline]
    fn reset_derived_segment(&mut self) {
        self.bull_min.clear();
        self.bull_max.clear();
        self.bear_min.clear();
        self.bear_max.clear();
        self.raw_min.clear();
        self.raw_max.clear();
        self.prev_total = f64::NAN;
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let idx = self.index;
        self.index = self.index.saturating_add(1);

        let ma = self.ma_state.update(close);
        if !(high.is_finite() && low.is_finite() && ma.is_finite()) {
            self.reset_derived_segment();
            let (upper, lower) = match self.params.calculation_method {
                BullsVBearsCalculationMethod::Normalized => {
                    (self.params.threshold_level, -self.params.threshold_level)
                }
                BullsVBearsCalculationMethod::Raw => (f64::NAN, f64::NAN),
            };
            return (
                f64::NAN,
                f64::NAN,
                f64::NAN,
                ma,
                upper,
                lower,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            );
        }

        let bull = high - ma;
        let bear = ma - low;
        let (value, upper, lower) = match self.params.calculation_method {
            BullsVBearsCalculationMethod::Normalized => {
                let min_index = idx
                    .saturating_add(1)
                    .saturating_sub(self.params.normalized_bars_back);
                expire_queue(&mut self.bull_min, min_index);
                expire_queue(&mut self.bull_max, min_index);
                expire_queue(&mut self.bear_min, min_index);
                expire_queue(&mut self.bear_max, min_index);
                push_min_queue(&mut self.bull_min, idx, bull);
                push_max_queue(&mut self.bull_max, idx, bull);
                push_min_queue(&mut self.bear_min, idx, bear);
                push_max_queue(&mut self.bear_max, idx, bear);
                let bull_min_value = self.bull_min.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bull_max_value = self.bull_max.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bear_min_value = self.bear_min.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bear_max_value = self.bear_max.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let bull_range = bull_max_value - bull_min_value;
                let bear_range = bear_max_value - bear_min_value;
                let total = if bull_range > 0.0 && bear_range > 0.0 {
                    ((bull - bull_min_value) / bull_range - 0.5) * 100.0
                        - ((bear - bear_min_value) / bear_range - 0.5) * 100.0
                } else {
                    f64::NAN
                };
                (
                    total,
                    self.params.threshold_level,
                    -self.params.threshold_level,
                )
            }
            BullsVBearsCalculationMethod::Raw => {
                let total = bull - bear;
                let min_index = idx
                    .saturating_add(1)
                    .saturating_sub(self.params.raw_rolling_period);
                expire_queue(&mut self.raw_min, min_index);
                expire_queue(&mut self.raw_max, min_index);
                push_min_queue(&mut self.raw_min, idx, total);
                push_max_queue(&mut self.raw_max, idx, total);
                let raw_lowest = self.raw_min.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let raw_highest = self.raw_max.front().map(|(_, v)| *v).unwrap_or(f64::NAN);
                let raw_range = raw_highest - raw_lowest;
                let upper = raw_lowest + raw_range * (self.params.raw_threshold_percentile / 100.0);
                let lower = raw_lowest
                    + raw_range * ((100.0 - self.params.raw_threshold_percentile) / 100.0);
                (total, upper, lower)
            }
        };

        if !(value.is_finite() && upper.is_finite() && lower.is_finite()) {
            self.prev_total = f64::NAN;
            return (
                f64::NAN,
                bull,
                bear,
                ma,
                upper,
                lower,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            );
        }

        let bullish_signal = if value > upper { 1.0 } else { 0.0 };
        let bearish_signal = if value < lower { 1.0 } else { 0.0 };
        let zero_cross_up = if self.prev_total.is_finite() && value > 0.0 && self.prev_total <= 0.0
        {
            1.0
        } else {
            0.0
        };
        let zero_cross_down =
            if self.prev_total.is_finite() && value < 0.0 && self.prev_total >= 0.0 {
                1.0
            } else {
                0.0
            };
        self.prev_total = value;
        (
            value,
            bull,
            bear,
            ma,
            upper,
            lower,
            bullish_signal,
            bearish_signal,
            zero_cross_up,
            zero_cross_down,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BullsVBearsBatchRange {
    pub period: (usize, usize, usize),
    pub normalized_bars_back: (usize, usize, usize),
    pub raw_rolling_period: (usize, usize, usize),
    pub raw_threshold_percentile: (f64, f64, f64),
    pub threshold_level: (f64, f64, f64),
    pub ma_type: BullsVBearsMaType,
    pub calculation_method: BullsVBearsCalculationMethod,
}

impl Default for BullsVBearsBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
            normalized_bars_back: (
                DEFAULT_NORMALIZED_BARS_BACK,
                DEFAULT_NORMALIZED_BARS_BACK,
                0,
            ),
            raw_rolling_period: (DEFAULT_RAW_ROLLING_PERIOD, DEFAULT_RAW_ROLLING_PERIOD, 0),
            raw_threshold_percentile: (
                DEFAULT_RAW_THRESHOLD_PERCENTILE,
                DEFAULT_RAW_THRESHOLD_PERCENTILE,
                0.0,
            ),
            threshold_level: (DEFAULT_THRESHOLD_LEVEL, DEFAULT_THRESHOLD_LEVEL, 0.0),
            ma_type: DEFAULT_MA_TYPE,
            calculation_method: DEFAULT_CALCULATION_METHOD,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BullsVBearsBatchOutput {
    pub value: Vec<f64>,
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub ma: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub bullish_signal: Vec<f64>,
    pub bearish_signal: Vec<f64>,
    pub zero_cross_up: Vec<f64>,
    pub zero_cross_down: Vec<f64>,
    pub combos: Vec<BullsVBearsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct BullsVBearsBatchBuilder {
    range: BullsVBearsBatchRange,
    kernel: Kernel,
}

impl Default for BullsVBearsBatchBuilder {
    fn default() -> Self {
        Self {
            range: BullsVBearsBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl BullsVBearsBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: BullsVBearsBatchRange) -> Self {
        self.range = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<BullsVBearsBatchOutput, BullsVBearsError> {
        bulls_v_bears_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<BullsVBearsBatchOutput, BullsVBearsError> {
        bulls_v_bears_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, BullsVBearsError> {
    if start > end || (start != end && step == 0) {
        return Err(BullsVBearsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        current = current
            .checked_add(step)
            .ok_or_else(|| BullsVBearsError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            })?;
        if current > end && out.last().copied() != Some(end) {
            break;
        }
        if out.len() > 1_000_000 {
            return Err(BullsVBearsError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
fn expand_float_range(start: f64, end: f64, step: f64) -> Result<Vec<f64>, BullsVBearsError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(BullsVBearsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if start > end || ((start - end).abs() > f64::EPSILON && step <= 0.0) {
        return Err(BullsVBearsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end + 1e-12 {
        out.push(current);
        if out.len() > 1_000_000 {
            return Err(BullsVBearsError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if (current - end).abs() <= 1e-12 {
            break;
        }
        current += step;
    }
    Ok(out)
}

pub fn bulls_v_bears_expand_grid(
    sweep: &BullsVBearsBatchRange,
) -> Result<Vec<BullsVBearsParams>, BullsVBearsError> {
    let periods = expand_usize_range(sweep.period.0, sweep.period.1, sweep.period.2)?;
    let normalized_bars_backs = expand_usize_range(
        sweep.normalized_bars_back.0,
        sweep.normalized_bars_back.1,
        sweep.normalized_bars_back.2,
    )?;
    let raw_rolling_periods = expand_usize_range(
        sweep.raw_rolling_period.0,
        sweep.raw_rolling_period.1,
        sweep.raw_rolling_period.2,
    )?;
    let raw_threshold_percentiles = expand_float_range(
        sweep.raw_threshold_percentile.0,
        sweep.raw_threshold_percentile.1,
        sweep.raw_threshold_percentile.2,
    )?;
    let threshold_levels = expand_float_range(
        sweep.threshold_level.0,
        sweep.threshold_level.1,
        sweep.threshold_level.2,
    )?;

    let mut out = Vec::with_capacity(
        periods.len()
            * normalized_bars_backs.len()
            * raw_rolling_periods.len()
            * raw_threshold_percentiles.len()
            * threshold_levels.len(),
    );
    for period in periods {
        for normalized_bars_back in &normalized_bars_backs {
            for raw_rolling_period in &raw_rolling_periods {
                for raw_threshold_percentile in &raw_threshold_percentiles {
                    for threshold_level in &threshold_levels {
                        out.push(BullsVBearsParams {
                            period: Some(period),
                            ma_type: Some(sweep.ma_type),
                            calculation_method: Some(sweep.calculation_method),
                            normalized_bars_back: Some(*normalized_bars_back),
                            raw_rolling_period: Some(*raw_rolling_period),
                            raw_threshold_percentile: Some(*raw_threshold_percentile),
                            threshold_level: Some(*threshold_level),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
fn validate_raw_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<usize, BullsVBearsError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(BullsVBearsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(BullsVBearsError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    first_valid_hlc(high, low, close).ok_or(BullsVBearsError::AllValuesNaN)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, BullsVBearsError> {
    rows.checked_mul(cols)
        .ok_or_else(|| BullsVBearsError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn bulls_v_bears_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullsVBearsBatchRange,
    kernel: Kernel,
) -> Result<BullsVBearsBatchOutput, BullsVBearsError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(BullsVBearsError::InvalidKernelForBatch(kernel)),
    };
    bulls_v_bears_batch_par_slice(high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn bulls_v_bears_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullsVBearsBatchRange,
    kernel: Kernel,
) -> Result<BullsVBearsBatchOutput, BullsVBearsError> {
    bulls_v_bears_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn bulls_v_bears_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullsVBearsBatchRange,
    kernel: Kernel,
) -> Result<BullsVBearsBatchOutput, BullsVBearsError> {
    bulls_v_bears_batch_inner(high, low, close, sweep, kernel, true)
}

fn bulls_v_bears_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullsVBearsBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<BullsVBearsBatchOutput, BullsVBearsError> {
    let combos = bulls_v_bears_expand_grid(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total = batch_shape(rows, cols)?;
    validate_raw_slices(high, low, close)?;
    let warmups = combos
        .iter()
        .map(|params| resolve_params(params).map(|p| p.ma_type.warmup(p.period)))
        .collect::<Result<Vec<_>, _>>()?;

    let mut value_buf = make_uninit_matrix(rows, cols);
    let mut bull_buf = make_uninit_matrix(rows, cols);
    let mut bear_buf = make_uninit_matrix(rows, cols);
    let mut ma_buf = make_uninit_matrix(rows, cols);
    let mut upper_buf = make_uninit_matrix(rows, cols);
    let mut lower_buf = make_uninit_matrix(rows, cols);
    let mut bullish_signal_buf = make_uninit_matrix(rows, cols);
    let mut bearish_signal_buf = make_uninit_matrix(rows, cols);
    let mut zero_cross_up_buf = make_uninit_matrix(rows, cols);
    let mut zero_cross_down_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut value_buf, cols, &warmups);
    init_matrix_prefixes(&mut bull_buf, cols, &warmups);
    init_matrix_prefixes(&mut bear_buf, cols, &warmups);
    init_matrix_prefixes(&mut ma_buf, cols, &warmups);
    init_matrix_prefixes(&mut upper_buf, cols, &warmups);
    init_matrix_prefixes(&mut lower_buf, cols, &warmups);
    init_matrix_prefixes(&mut bullish_signal_buf, cols, &warmups);
    init_matrix_prefixes(&mut bearish_signal_buf, cols, &warmups);
    init_matrix_prefixes(&mut zero_cross_up_buf, cols, &warmups);
    init_matrix_prefixes(&mut zero_cross_down_buf, cols, &warmups);

    let mut value_guard = ManuallyDrop::new(value_buf);
    let mut bull_guard = ManuallyDrop::new(bull_buf);
    let mut bear_guard = ManuallyDrop::new(bear_buf);
    let mut ma_guard = ManuallyDrop::new(ma_buf);
    let mut upper_guard = ManuallyDrop::new(upper_buf);
    let mut lower_guard = ManuallyDrop::new(lower_buf);
    let mut bullish_signal_guard = ManuallyDrop::new(bullish_signal_buf);
    let mut bearish_signal_guard = ManuallyDrop::new(bearish_signal_buf);
    let mut zero_cross_up_guard = ManuallyDrop::new(zero_cross_up_buf);
    let mut zero_cross_down_guard = ManuallyDrop::new(zero_cross_down_buf);

    let out_value = unsafe {
        core::slice::from_raw_parts_mut(value_guard.as_mut_ptr() as *mut f64, value_guard.len())
    };
    let out_bull = unsafe {
        core::slice::from_raw_parts_mut(bull_guard.as_mut_ptr() as *mut f64, bull_guard.len())
    };
    let out_bear = unsafe {
        core::slice::from_raw_parts_mut(bear_guard.as_mut_ptr() as *mut f64, bear_guard.len())
    };
    let out_ma = unsafe {
        core::slice::from_raw_parts_mut(ma_guard.as_mut_ptr() as *mut f64, ma_guard.len())
    };
    let out_upper = unsafe {
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let out_lower = unsafe {
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };
    let out_bullish_signal = unsafe {
        core::slice::from_raw_parts_mut(
            bullish_signal_guard.as_mut_ptr() as *mut f64,
            bullish_signal_guard.len(),
        )
    };
    let out_bearish_signal = unsafe {
        core::slice::from_raw_parts_mut(
            bearish_signal_guard.as_mut_ptr() as *mut f64,
            bearish_signal_guard.len(),
        )
    };
    let out_zero_cross_up = unsafe {
        core::slice::from_raw_parts_mut(
            zero_cross_up_guard.as_mut_ptr() as *mut f64,
            zero_cross_up_guard.len(),
        )
    };
    let out_zero_cross_down = unsafe {
        core::slice::from_raw_parts_mut(
            zero_cross_down_guard.as_mut_ptr() as *mut f64,
            zero_cross_down_guard.len(),
        )
    };

    bulls_v_bears_batch_inner_into(
        high,
        low,
        close,
        sweep,
        kernel,
        parallel,
        out_value,
        out_bull,
        out_bear,
        out_ma,
        out_upper,
        out_lower,
        out_bullish_signal,
        out_bearish_signal,
        out_zero_cross_up,
        out_zero_cross_down,
    )?;

    let value = unsafe {
        Vec::from_raw_parts(
            value_guard.as_mut_ptr() as *mut f64,
            total,
            value_guard.capacity(),
        )
    };
    let bull = unsafe {
        Vec::from_raw_parts(
            bull_guard.as_mut_ptr() as *mut f64,
            total,
            bull_guard.capacity(),
        )
    };
    let bear = unsafe {
        Vec::from_raw_parts(
            bear_guard.as_mut_ptr() as *mut f64,
            total,
            bear_guard.capacity(),
        )
    };
    let ma = unsafe {
        Vec::from_raw_parts(
            ma_guard.as_mut_ptr() as *mut f64,
            total,
            ma_guard.capacity(),
        )
    };
    let upper = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            total,
            upper_guard.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            total,
            lower_guard.capacity(),
        )
    };
    let bullish_signal = unsafe {
        Vec::from_raw_parts(
            bullish_signal_guard.as_mut_ptr() as *mut f64,
            total,
            bullish_signal_guard.capacity(),
        )
    };
    let bearish_signal = unsafe {
        Vec::from_raw_parts(
            bearish_signal_guard.as_mut_ptr() as *mut f64,
            total,
            bearish_signal_guard.capacity(),
        )
    };
    let zero_cross_up = unsafe {
        Vec::from_raw_parts(
            zero_cross_up_guard.as_mut_ptr() as *mut f64,
            total,
            zero_cross_up_guard.capacity(),
        )
    };
    let zero_cross_down = unsafe {
        Vec::from_raw_parts(
            zero_cross_down_guard.as_mut_ptr() as *mut f64,
            total,
            zero_cross_down_guard.capacity(),
        )
    };

    Ok(BullsVBearsBatchOutput {
        value,
        bull,
        bear,
        ma,
        upper,
        lower,
        bullish_signal,
        bearish_signal,
        zero_cross_up,
        zero_cross_down,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bulls_v_bears_batch_into_slice(
    out_value: &mut [f64],
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_ma: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_bullish_signal: &mut [f64],
    out_bearish_signal: &mut [f64],
    out_zero_cross_up: &mut [f64],
    out_zero_cross_down: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullsVBearsBatchRange,
    kernel: Kernel,
) -> Result<(), BullsVBearsError> {
    bulls_v_bears_batch_inner_into(
        high,
        low,
        close,
        sweep,
        kernel,
        false,
        out_value,
        out_bull,
        out_bear,
        out_ma,
        out_upper,
        out_lower,
        out_bullish_signal,
        out_bearish_signal,
        out_zero_cross_up,
        out_zero_cross_down,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bulls_v_bears_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullsVBearsBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_value: &mut [f64],
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_ma: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_bullish_signal: &mut [f64],
    out_bearish_signal: &mut [f64],
    out_zero_cross_up: &mut [f64],
    out_zero_cross_down: &mut [f64],
) -> Result<Vec<BullsVBearsParams>, BullsVBearsError> {
    let combos = bulls_v_bears_expand_grid(sweep)?;
    validate_raw_slices(high, low, close)?;
    let rows = combos.len();
    let cols = close.len();
    let total = batch_shape(rows, cols)?;
    check_output_len(out_value, total)?;
    check_output_len(out_bull, total)?;
    check_output_len(out_bear, total)?;
    check_output_len(out_ma, total)?;
    check_output_len(out_upper, total)?;
    check_output_len(out_lower, total)?;
    check_output_len(out_bullish_signal, total)?;
    check_output_len(out_bearish_signal, total)?;
    check_output_len(out_zero_cross_up, total)?;
    check_output_len(out_zero_cross_down, total)?;

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        let results: Vec<Result<(), BullsVBearsError>> = out_value
            .par_chunks_mut(cols)
            .zip(out_bull.par_chunks_mut(cols))
            .zip(out_bear.par_chunks_mut(cols))
            .zip(out_ma.par_chunks_mut(cols))
            .zip(out_upper.par_chunks_mut(cols))
            .zip(out_lower.par_chunks_mut(cols))
            .zip(out_bullish_signal.par_chunks_mut(cols))
            .zip(out_bearish_signal.par_chunks_mut(cols))
            .zip(out_zero_cross_up.par_chunks_mut(cols))
            .zip(out_zero_cross_down.par_chunks_mut(cols))
            .zip(combos.par_iter())
            .map(
                |(
                    (
                        (
                            (
                                (
                                    (
                                        ((((value_row, bull_row), bear_row), ma_row), upper_row),
                                        lower_row,
                                    ),
                                    bullish_row,
                                ),
                                bearish_row,
                            ),
                            up_row,
                        ),
                        down_row,
                    ),
                    params,
                )| {
                    let resolved = resolve_params(params)?;
                    bulls_v_bears_compute_into(
                        high,
                        low,
                        close,
                        resolved,
                        value_row,
                        bull_row,
                        bear_row,
                        ma_row,
                        upper_row,
                        lower_row,
                        bullish_row,
                        bearish_row,
                        up_row,
                        down_row,
                    )
                },
            )
            .collect();
        for result in results {
            result?;
        }
    }
    if !parallel || cfg!(target_arch = "wasm32") {
        for (row, params) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let resolved = resolve_params(params)?;
            bulls_v_bears_compute_into(
                high,
                low,
                close,
                resolved,
                &mut out_value[start..end],
                &mut out_bull[start..end],
                &mut out_bear[start..end],
                &mut out_ma[start..end],
                &mut out_upper[start..end],
                &mut out_lower[start..end],
                &mut out_bullish_signal[start..end],
                &mut out_bearish_signal[start..end],
                &mut out_zero_cross_up[start..end],
                &mut out_zero_cross_down[start..end],
            )?;
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV, ParamValue,
        compute_cpu_batch,
    };

    fn sample_hlc() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = (0..160)
            .map(|i| 100.0 + (i as f64 * 0.25) + ((i % 7) as f64 - 3.0) * 0.4)
            .collect::<Vec<_>>();
        let high = close.iter().map(|v| *v + 1.5).collect::<Vec<_>>();
        let low = close.iter().map(|v| *v - 1.25).collect::<Vec<_>>();
        (high, low, close)
    }

    fn assert_vec_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let diff = (a - b).abs();
            assert!(diff <= 1e-10, "mismatch at {idx}: {a} vs {b}");
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN, got {actual}");
            return;
        }
        assert!(actual.is_finite(), "expected {expected}, got {actual}");
        let tolerance = 1e-12_f64.max(expected.abs() * 1e-12);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn params(
        ma_type: BullsVBearsMaType,
        calculation_method: BullsVBearsCalculationMethod,
    ) -> BullsVBearsParams {
        BullsVBearsParams {
            period: Some(2),
            ma_type: Some(ma_type),
            calculation_method: Some(calculation_method),
            normalized_bars_back: Some(3),
            raw_rolling_period: Some(3),
            raw_threshold_percentile: Some(80.0),
            threshold_level: Some(30.0),
        }
    }

    #[test]
    fn normalized_formula_matches_hand_derived_reference() {
        // Official definition: bull = high - MA, bear = MA - low, then subtract
        // their independently min/max-normalized values over `bars back`.
        let high = [11.0, 15.0, 12.0, 18.0];
        let low = [8.0, 9.0, 7.0, 10.0];
        let close = [10.0, 12.0, 11.0, 14.0];
        let out = bulls_v_bears(&BullsVBearsInput::from_slices(
            &high,
            &low,
            &close,
            params(
                BullsVBearsMaType::Ema,
                BullsVBearsCalculationMethod::Normalized,
            ),
        ))
        .unwrap();

        let expected_ma = [10.0, 34.0 / 3.0, 100.0 / 9.0, 352.0 / 27.0];
        let expected_bull = [1.0, 11.0 / 3.0, 8.0 / 9.0, 134.0 / 27.0];
        let expected_bear = [2.0, 7.0 / 3.0, 37.0 / 9.0, 82.0 / 27.0];
        let expected_value = [f64::NAN, 0.0, -100.0, 725.0 / 12.0];
        let expected_bullish = [f64::NAN, 0.0, 0.0, 1.0];
        let expected_bearish = [f64::NAN, 0.0, 1.0, 0.0];
        let expected_cross_up = [f64::NAN, 0.0, 0.0, 1.0];
        let expected_cross_down = [f64::NAN, 0.0, 1.0, 0.0];

        for i in 0..close.len() {
            assert_close(out.ma[i], expected_ma[i]);
            assert_close(out.bull[i], expected_bull[i]);
            assert_close(out.bear[i], expected_bear[i]);
            assert_close(out.value[i], expected_value[i]);
            assert_close(out.upper[i], 30.0);
            assert_close(out.lower[i], -30.0);
            assert_close(out.bullish_signal[i], expected_bullish[i]);
            assert_close(out.bearish_signal[i], expected_bearish[i]);
            assert_close(out.zero_cross_up[i], expected_cross_up[i]);
            assert_close(out.zero_cross_down[i], expected_cross_down[i]);
        }
    }

    #[test]
    fn raw_formula_and_signals_match_hand_derived_reference() {
        let high = [11.0, 15.0, 12.0, 18.0];
        let low = [8.0, 9.0, 7.0, 10.0];
        let close = [10.0, 12.0, 11.0, 14.0];
        let out = bulls_v_bears(&BullsVBearsInput::from_slices(
            &high,
            &low,
            &close,
            params(BullsVBearsMaType::Ema, BullsVBearsCalculationMethod::Raw),
        ))
        .unwrap();

        let expected_value = [-1.0, 4.0 / 3.0, -29.0 / 9.0, 52.0 / 27.0];
        let expected_upper = [-1.0, 13.0 / 15.0, 19.0 / 45.0, 121.0 / 135.0];
        let expected_lower = [-1.0, -8.0 / 15.0, -104.0 / 45.0, -296.0 / 135.0];
        let expected_bullish = [0.0, 1.0, 0.0, 1.0];
        let expected_bearish = [0.0, 0.0, 1.0, 0.0];
        let expected_cross_up = [0.0, 1.0, 0.0, 1.0];
        let expected_cross_down = [0.0, 0.0, 1.0, 0.0];

        for i in 0..close.len() {
            assert_close(out.value[i], expected_value[i]);
            assert_close(out.upper[i], expected_upper[i]);
            assert_close(out.lower[i], expected_lower[i]);
            assert_close(out.bullish_signal[i], expected_bullish[i]);
            assert_close(out.bearish_signal[i], expected_bearish[i]);
            assert_close(out.zero_cross_up[i], expected_cross_up[i]);
            assert_close(out.zero_cross_down[i], expected_cross_down[i]);
        }
    }

    #[test]
    fn sma_and_wma_match_hand_derived_reference() {
        let close = [1.0, 2.0, 4.0, 8.0];
        let high = [2.0, 3.0, 5.0, 9.0];
        let low = [0.0, 1.0, 3.0, 7.0];

        for (ma_type, expected) in [
            (
                BullsVBearsMaType::Sma,
                [f64::NAN, f64::NAN, 7.0 / 3.0, 14.0 / 3.0],
            ),
            (
                BullsVBearsMaType::Wma,
                [f64::NAN, f64::NAN, 17.0 / 6.0, 34.0 / 6.0],
            ),
        ] {
            let mut p = params(ma_type, BullsVBearsCalculationMethod::Raw);
            p.period = Some(3);
            let out =
                bulls_v_bears(&BullsVBearsInput::from_slices(&high, &low, &close, p)).unwrap();
            for (actual, expected) in out.ma.iter().zip(expected) {
                assert_close(*actual, expected);
            }
        }
    }

    #[test]
    fn normalized_gap_starts_a_new_segment_without_crossing_history() {
        let high = [11.0, 13.0, f64::NAN, 12.0, 14.0];
        let low = [8.0, 9.0, f64::NAN, 7.0, 8.0];
        let close = [10.0, 10.0, f64::NAN, 10.0, 10.0];
        let params = params(
            BullsVBearsMaType::Ema,
            BullsVBearsCalculationMethod::Normalized,
        );
        let input = BullsVBearsInput::from_slices(&high, &low, &close, params.clone());
        let batch = bulls_v_bears(&input).unwrap();

        assert_close(batch.value[1], 100.0);
        assert!(batch.value[2].is_nan());
        assert!(batch.value[3].is_nan());
        assert_close(batch.value[4], 100.0);
        assert_close(batch.zero_cross_up[4], 0.0);
        assert_close(batch.zero_cross_down[4], 0.0);

        let mut stream = BullsVBearsStream::try_new(params).unwrap();
        let streamed = (0..close.len())
            .map(|i| stream.update(high[i], low[i], close[i]))
            .collect::<Vec<_>>();
        for i in 0..close.len() {
            assert_close(streamed[i].0, batch.value[i]);
            assert_close(streamed[i].4, batch.upper[i]);
            assert_close(streamed[i].5, batch.lower[i]);
            assert_close(streamed[i].8, batch.zero_cross_up[i]);
            assert_close(streamed[i].9, batch.zero_cross_down[i]);
        }
    }

    #[test]
    fn raw_gap_resets_dynamic_thresholds_and_crossing_history() {
        let high = [10.0, 10.0, f64::NAN, 12.0];
        let low = [8.0, 9.0, f64::NAN, 10.0];
        let close = [10.0, 10.0, f64::NAN, 10.0];
        let params = params(BullsVBearsMaType::Ema, BullsVBearsCalculationMethod::Raw);
        let input = BullsVBearsInput::from_slices(&high, &low, &close, params.clone());
        let batch = bulls_v_bears(&input).unwrap();

        assert!(batch.value[2].is_nan());
        assert!(batch.upper[2].is_nan());
        assert!(batch.lower[2].is_nan());
        assert_close(batch.value[3], 2.0);
        assert_close(batch.upper[3], 2.0);
        assert_close(batch.lower[3], 2.0);
        assert_close(batch.bullish_signal[3], 0.0);
        assert_close(batch.bearish_signal[3], 0.0);
        assert_close(batch.zero_cross_up[3], 0.0);
        assert_close(batch.zero_cross_down[3], 0.0);

        let mut stream = BullsVBearsStream::try_new(params).unwrap();
        let streamed = (0..close.len())
            .map(|i| stream.update(high[i], low[i], close[i]))
            .collect::<Vec<_>>();
        for i in 0..close.len() {
            assert_close(streamed[i].0, batch.value[i]);
            assert_close(streamed[i].4, batch.upper[i]);
            assert_close(streamed[i].5, batch.lower[i]);
            assert_close(streamed[i].6, batch.bullish_signal[i]);
            assert_close(streamed[i].7, batch.bearish_signal[i]);
            assert_close(streamed[i].8, batch.zero_cross_up[i]);
            assert_close(streamed[i].9, batch.zero_cross_down[i]);
        }
    }

    #[test]
    fn every_selected_output_matches_the_full_family() {
        let (high, low, close) = sample_hlc();
        let params = params(BullsVBearsMaType::Wma, BullsVBearsCalculationMethod::Raw);
        let input = BullsVBearsInput::from_slices(&high, &low, &close, params);
        let full = bulls_v_bears(&input).unwrap();
        let fields = [
            (BullsVBearsOutputField::Value, full.value.as_slice()),
            (BullsVBearsOutputField::Bull, full.bull.as_slice()),
            (BullsVBearsOutputField::Bear, full.bear.as_slice()),
            (BullsVBearsOutputField::Ma, full.ma.as_slice()),
            (BullsVBearsOutputField::Upper, full.upper.as_slice()),
            (BullsVBearsOutputField::Lower, full.lower.as_slice()),
            (
                BullsVBearsOutputField::BullishSignal,
                full.bullish_signal.as_slice(),
            ),
            (
                BullsVBearsOutputField::BearishSignal,
                full.bearish_signal.as_slice(),
            ),
            (
                BullsVBearsOutputField::ZeroCrossUp,
                full.zero_cross_up.as_slice(),
            ),
            (
                BullsVBearsOutputField::ZeroCrossDown,
                full.zero_cross_down.as_slice(),
            ),
        ];

        for (field, expected) in fields {
            let mut selected = vec![f64::NAN; close.len()];
            bulls_v_bears_output_into_slice(&mut selected, &input, Kernel::Auto, field).unwrap();
            assert_vec_close(&selected, expected);
        }
    }

    #[test]
    fn normalized_stream_matches_batch() {
        let (high, low, close) = sample_hlc();
        let params = BullsVBearsParams::default();
        let batch = bulls_v_bears(&BullsVBearsInput::from_slices(
            &high,
            &low,
            &close,
            params.clone(),
        ))
        .unwrap();
        let mut stream = BullsVBearsStream::try_new(params).unwrap();
        let mut value = Vec::with_capacity(close.len());
        let mut bull = Vec::with_capacity(close.len());
        let mut bear = Vec::with_capacity(close.len());
        let mut ma = Vec::with_capacity(close.len());
        let mut upper = Vec::with_capacity(close.len());
        let mut lower = Vec::with_capacity(close.len());
        let mut bullish_signal = Vec::with_capacity(close.len());
        let mut bearish_signal = Vec::with_capacity(close.len());
        let mut zero_cross_up = Vec::with_capacity(close.len());
        let mut zero_cross_down = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            let out = stream.update(high[i], low[i], close[i]);
            value.push(out.0);
            bull.push(out.1);
            bear.push(out.2);
            ma.push(out.3);
            upper.push(out.4);
            lower.push(out.5);
            bullish_signal.push(out.6);
            bearish_signal.push(out.7);
            zero_cross_up.push(out.8);
            zero_cross_down.push(out.9);
        }

        assert_vec_close(&value, &batch.value);
        assert_vec_close(&bull, &batch.bull);
        assert_vec_close(&bear, &batch.bear);
        assert_vec_close(&ma, &batch.ma);
        assert_vec_close(&upper, &batch.upper);
        assert_vec_close(&lower, &batch.lower);
        assert_vec_close(&bullish_signal, &batch.bullish_signal);
        assert_vec_close(&bearish_signal, &batch.bearish_signal);
        assert_vec_close(&zero_cross_up, &batch.zero_cross_up);
        assert_vec_close(&zero_cross_down, &batch.zero_cross_down);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (high, low, close) = sample_hlc();
        let single = bulls_v_bears(&BullsVBearsInput::from_slices(
            &high,
            &low,
            &close,
            BullsVBearsParams {
                period: Some(14),
                ma_type: Some(BullsVBearsMaType::Ema),
                calculation_method: Some(BullsVBearsCalculationMethod::Raw),
                normalized_bars_back: Some(120),
                raw_rolling_period: Some(50),
                raw_threshold_percentile: Some(95.0),
                threshold_level: Some(80.0),
            },
        ))
        .unwrap();
        let batch = bulls_v_bears_batch_slice(
            &high,
            &low,
            &close,
            &BullsVBearsBatchRange {
                period: (14, 16, 2),
                normalized_bars_back: (120, 120, 0),
                raw_rolling_period: (50, 50, 0),
                raw_threshold_percentile: (95.0, 95.0, 0.0),
                threshold_level: (80.0, 80.0, 0.0),
                ma_type: BullsVBearsMaType::Ema,
                calculation_method: BullsVBearsCalculationMethod::Raw,
            },
            Kernel::Auto,
        )
        .unwrap();
        let cols = close.len();
        assert_eq!(batch.rows, 2);
        assert_vec_close(&batch.value[..cols], &single.value);
        assert_vec_close(&batch.upper[..cols], &single.upper);
        assert_vec_close(&batch.lower[..cols], &single.lower);
    }

    #[test]
    fn invalid_period_fails() {
        let (high, low, close) = sample_hlc();
        let err = bulls_v_bears(&BullsVBearsInput::from_slices(
            &high,
            &low,
            &close,
            BullsVBearsParams {
                period: Some(0),
                ..BullsVBearsParams::default()
            },
        ))
        .unwrap_err();
        assert!(err.to_string().contains("Invalid period"));
    }

    #[test]
    fn cpu_dispatch_matches_direct() {
        let (high, low, close) = sample_hlc();
        let request = IndicatorBatchRequest {
            indicator_id: "bulls_v_bears",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &[IndicatorParamSet {
                params: &[
                    ParamKV {
                        key: "period",
                        value: ParamValue::Int(14),
                    },
                    ParamKV {
                        key: "ma_type",
                        value: ParamValue::EnumString("ema"),
                    },
                    ParamKV {
                        key: "calculation_method",
                        value: ParamValue::EnumString("raw"),
                    },
                ],
            }],
            kernel: Kernel::Auto,
        };

        let output = compute_cpu_batch(request).unwrap();
        let values = output.values_f64.unwrap();
        let direct = bulls_v_bears(&BullsVBearsInput::from_slices(
            &high,
            &low,
            &close,
            BullsVBearsParams {
                period: Some(14),
                ma_type: Some(BullsVBearsMaType::Ema),
                calculation_method: Some(BullsVBearsCalculationMethod::Raw),
                ..BullsVBearsParams::default()
            },
        ))
        .unwrap();
        assert_vec_close(&values[..close.len()], &direct.value);
    }
}
