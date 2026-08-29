use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_FAST_LENGTH: usize = 20;
const DEFAULT_SLOW_LENGTH: usize = 40;
const ATR_PERIOD: usize = 14;
const ATR_SCALE: f64 = 0.5;

#[derive(Debug, Clone)]
pub enum HemaTrendLevelsData<'a> {
    Candles(&'a Candles),
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct HemaTrendLevelsOutput {
    pub fast_hema: Vec<f64>,
    pub slow_hema: Vec<f64>,
    pub trend_direction: Vec<f64>,
    pub bar_state: Vec<f64>,
    pub bullish_crossover: Vec<f64>,
    pub bearish_crossunder: Vec<f64>,
    pub box_offset: Vec<f64>,
    pub bull_box_top: Vec<f64>,
    pub bull_box_bottom: Vec<f64>,
    pub bear_box_top: Vec<f64>,
    pub bear_box_bottom: Vec<f64>,
    pub bullish_test: Vec<f64>,
    pub bearish_test: Vec<f64>,
    pub bullish_test_level: Vec<f64>,
    pub bearish_test_level: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HemaTrendLevelsOutputField {
    FastHema,
    SlowHema,
    TrendDirection,
    BarState,
    BullishCrossover,
    BearishCrossunder,
    BoxOffset,
    BullBoxTop,
    BullBoxBottom,
    BearBoxTop,
    BearBoxBottom,
    BullishTest,
    BearishTest,
    BullishTestLevel,
    BearishTestLevel,
}

#[derive(Debug, Clone, Copy)]
pub struct HemaTrendLevelsPoint {
    pub fast_hema: f64,
    pub slow_hema: f64,
    pub trend_direction: f64,
    pub bar_state: f64,
    pub bullish_crossover: f64,
    pub bearish_crossunder: f64,
    pub box_offset: f64,
    pub bull_box_top: f64,
    pub bull_box_bottom: f64,
    pub bear_box_top: f64,
    pub bear_box_bottom: f64,
    pub bullish_test: f64,
    pub bearish_test: f64,
    pub bullish_test_level: f64,
    pub bearish_test_level: f64,
}

impl HemaTrendLevelsPoint {
    #[inline(always)]
    fn nan() -> Self {
        Self {
            fast_hema: f64::NAN,
            slow_hema: f64::NAN,
            trend_direction: f64::NAN,
            bar_state: f64::NAN,
            bullish_crossover: f64::NAN,
            bearish_crossunder: f64::NAN,
            box_offset: f64::NAN,
            bull_box_top: f64::NAN,
            bull_box_bottom: f64::NAN,
            bear_box_top: f64::NAN,
            bear_box_bottom: f64::NAN,
            bullish_test: f64::NAN,
            bearish_test: f64::NAN,
            bullish_test_level: f64::NAN,
            bearish_test_level: f64::NAN,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HemaTrendLevelsParams {
    pub fast_length: Option<usize>,
    pub slow_length: Option<usize>,
}

impl Default for HemaTrendLevelsParams {
    fn default() -> Self {
        Self {
            fast_length: Some(DEFAULT_FAST_LENGTH),
            slow_length: Some(DEFAULT_SLOW_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HemaTrendLevelsInput<'a> {
    pub data: HemaTrendLevelsData<'a>,
    pub params: HemaTrendLevelsParams,
}

impl<'a> HemaTrendLevelsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: HemaTrendLevelsParams) -> Self {
        Self {
            data: HemaTrendLevelsData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: HemaTrendLevelsParams,
    ) -> Self {
        Self {
            data: HemaTrendLevelsData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, HemaTrendLevelsParams::default())
    }

    #[inline]
    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            HemaTrendLevelsData::Candles(candles) => {
                (&candles.open, &candles.high, &candles.low, &candles.close)
            }
            HemaTrendLevelsData::Slices {
                open,
                high,
                low,
                close,
            } => (open, high, low, close),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HemaTrendLevelsBuilder {
    fast_length: Option<usize>,
    slow_length: Option<usize>,
    kernel: Kernel,
}

impl HemaTrendLevelsBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn fast_length(mut self, value: usize) -> Self {
        self.fast_length = Some(value);
        self
    }

    #[inline]
    pub fn slow_length(mut self, value: usize) -> Self {
        self.slow_length = Some(value);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<HemaTrendLevelsOutput, HemaTrendLevelsError> {
        let input = HemaTrendLevelsInput::from_candles(
            candles,
            HemaTrendLevelsParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
            },
        );
        hema_trend_levels_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<HemaTrendLevelsOutput, HemaTrendLevelsError> {
        let input = HemaTrendLevelsInput::from_slices(
            open,
            high,
            low,
            close,
            HemaTrendLevelsParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
            },
        );
        hema_trend_levels_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<HemaTrendLevelsStream, HemaTrendLevelsError> {
        HemaTrendLevelsStream::try_new(HemaTrendLevelsParams {
            fast_length: self.fast_length,
            slow_length: self.slow_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum HemaTrendLevelsError {
    #[error("hema_trend_levels: Input data slice is empty.")]
    EmptyInputData,
    #[error("hema_trend_levels: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "hema_trend_levels: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("hema_trend_levels: Invalid fast_length: {fast_length}")]
    InvalidFastLength { fast_length: usize },
    #[error("hema_trend_levels: Invalid slow_length: {slow_length}")]
    InvalidSlowLength { slow_length: usize },
    #[error("hema_trend_levels: Output length mismatch: expected = {expected}")]
    OutputLengthMismatch { expected: usize },
    #[error("hema_trend_levels: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("hema_trend_levels: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    fast_length: usize,
    slow_length: usize,
    fast_half: usize,
    slow_half: usize,
    fast_sqrt: usize,
    slow_sqrt: usize,
}

#[inline(always)]
fn resolve_params(params: &HemaTrendLevelsParams) -> Result<ResolvedParams, HemaTrendLevelsError> {
    let fast_length = params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
    let slow_length = params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
    if fast_length == 0 {
        return Err(HemaTrendLevelsError::InvalidFastLength { fast_length });
    }
    if slow_length == 0 {
        return Err(HemaTrendLevelsError::InvalidSlowLength { slow_length });
    }
    Ok(ResolvedParams {
        fast_length,
        slow_length,
        fast_half: ((fast_length as f64) / 2.0).round().max(1.0) as usize,
        slow_half: ((slow_length as f64) / 2.0).round().max(1.0) as usize,
        fast_sqrt: (fast_length as f64).sqrt().round().max(1.0) as usize,
        slow_sqrt: (slow_length as f64).sqrt().round().max(1.0) as usize,
    })
}

#[inline(always)]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut i = 0usize;
    while i < close.len() {
        if open[i].is_finite() && high[i].is_finite() && low[i].is_finite() && close[i].is_finite()
        {
            return i;
        }
        i += 1;
    }
    close.len()
}

#[derive(Debug, Clone, Copy)]
struct EmaState {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl EmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            value: 0.0,
            initialized: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value += self.alpha * (value - self.value);
        }
        self.value
    }
}

#[derive(Debug, Clone, Copy)]
struct HemaState {
    ema_half: EmaState,
    ema_full: EmaState,
    ema_diff: EmaState,
}

impl HemaState {
    #[inline(always)]
    fn new(length: usize, half_length: usize, sqrt_length: usize) -> Self {
        Self {
            ema_half: EmaState::new(half_length),
            ema_full: EmaState::new(length),
            ema_diff: EmaState::new(sqrt_length),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.ema_half.reset();
        self.ema_full.reset();
        self.ema_diff.reset();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        let ema_half = self.ema_half.update(value);
        let ema_full = self.ema_full.update(value);
        self.ema_diff.update(2.0 * ema_half - ema_full)
    }
}

#[derive(Debug, Clone, Copy)]
struct AtrState {
    prev_close: f64,
    sum: f64,
    value: f64,
    count: usize,
    initialized: bool,
}

impl AtrState {
    #[inline(always)]
    fn new() -> Self {
        Self {
            prev_close: f64::NAN,
            sum: 0.0,
            value: 0.0,
            count: 0,
            initialized: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.sum = 0.0;
        self.value = 0.0;
        self.count = 0;
        self.initialized = false;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if self.prev_close.is_finite() {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };
        self.prev_close = close;

        if !self.initialized {
            self.sum += tr;
            self.count += 1;
            if self.count >= ATR_PERIOD {
                self.value = self.sum / ATR_PERIOD as f64;
                self.initialized = true;
                Some(self.value)
            } else {
                None
            }
        } else {
            self.value = ((ATR_PERIOD - 1) as f64 * self.value + tr) / ATR_PERIOD as f64;
            Some(self.value)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BoxState {
    top: f64,
    bottom: f64,
}

#[derive(Debug, Clone)]
struct HemaTrendLevelsCoreState {
    fast: HemaState,
    slow: HemaState,
    atr: AtrState,
    prev_fast: f64,
    prev_slow: f64,
    bull_box: Option<BoxState>,
    bear_box: Option<BoxState>,
}

impl HemaTrendLevelsCoreState {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            fast: HemaState::new(params.fast_length, params.fast_half, params.fast_sqrt),
            slow: HemaState::new(params.slow_length, params.slow_half, params.slow_sqrt),
            atr: AtrState::new(),
            prev_fast: f64::NAN,
            prev_slow: f64::NAN,
            bull_box: None,
            bear_box: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.atr.reset();
        self.prev_fast = f64::NAN;
        self.prev_slow = f64::NAN;
        self.bull_box = None;
        self.bear_box = None;
    }

    #[inline(always)]
    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> HemaTrendLevelsPoint {
        let fast_hema = self.fast.update(close);
        let slow_hema = self.slow.update(close);
        let box_offset = self
            .atr
            .update(high, low, close)
            .map(|atr| atr * ATR_SCALE)
            .unwrap_or(f64::NAN);

        let bullish_crossover = self.prev_fast.is_finite()
            && self.prev_slow.is_finite()
            && self.prev_fast <= self.prev_slow
            && fast_hema > slow_hema;
        let bearish_crossunder = self.prev_fast.is_finite()
            && self.prev_slow.is_finite()
            && self.prev_fast >= self.prev_slow
            && fast_hema < slow_hema;

        if bullish_crossover && box_offset.is_finite() {
            self.bull_box = Some(BoxState {
                top: low + box_offset,
                bottom: low,
            });
        } else if bearish_crossunder && box_offset.is_finite() {
            self.bear_box = Some(BoxState {
                top: high - box_offset,
                bottom: high,
            });
        }

        let trend_direction = if fast_hema > slow_hema {
            1.0
        } else if fast_hema < slow_hema {
            -1.0
        } else {
            0.0
        };
        let bullish_condition = close > fast_hema && fast_hema > slow_hema;
        let bearish_condition = close < fast_hema && fast_hema < slow_hema;
        let bar_state = if bullish_condition {
            1.0
        } else if bearish_condition {
            -1.0
        } else {
            0.0
        };

        let bull_box_top = self.bull_box.map(|b| b.top).unwrap_or(f64::NAN);
        let bull_box_bottom = self.bull_box.map(|b| b.bottom).unwrap_or(f64::NAN);
        let bear_box_top = self.bear_box.map(|b| b.top).unwrap_or(f64::NAN);
        let bear_box_bottom = self.bear_box.map(|b| b.bottom).unwrap_or(f64::NAN);
        let mut bullish_test = 0.0;
        let mut bearish_test = 0.0;
        let mut bullish_test_level = f64::NAN;
        let mut bearish_test_level = f64::NAN;

        if let Some(bull_box) = self.bull_box {
            if low < bull_box.top
                && high > bull_box.top
                && open > bull_box.top
                && close > bull_box.top
            {
                bullish_test = 1.0;
                bullish_test_level = bull_box.bottom;
            }
        }
        if let Some(bear_box) = self.bear_box {
            if high > bear_box.top
                && low < bear_box.top
                && open < bear_box.top
                && close < bear_box.top
            {
                bearish_test = 1.0;
                bearish_test_level = bear_box.bottom;
            }
        }

        self.prev_fast = fast_hema;
        self.prev_slow = slow_hema;

        HemaTrendLevelsPoint {
            fast_hema,
            slow_hema,
            trend_direction,
            bar_state,
            bullish_crossover: if bullish_crossover { 1.0 } else { 0.0 },
            bearish_crossunder: if bearish_crossunder { 1.0 } else { 0.0 },
            box_offset,
            bull_box_top,
            bull_box_bottom,
            bear_box_top,
            bear_box_bottom,
            bullish_test,
            bearish_test,
            bullish_test_level,
            bearish_test_level,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HemaTrendLevelsStream {
    state: HemaTrendLevelsCoreState,
    warmup_period: usize,
}

impl HemaTrendLevelsStream {
    #[inline]
    pub fn try_new(params: HemaTrendLevelsParams) -> Result<Self, HemaTrendLevelsError> {
        let params = resolve_params(&params)?;
        Ok(Self {
            state: HemaTrendLevelsCoreState::new(params),
            warmup_period: 0,
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<HemaTrendLevelsPoint> {
        if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.state.reset();
            return None;
        }
        Some(self.state.update(open, high, low, close))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state.reset();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.warmup_period
    }
}

#[allow(clippy::too_many_arguments)]
fn hema_trend_levels_row_from_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    fast_hema_out: &mut [f64],
    slow_hema_out: &mut [f64],
    trend_direction_out: &mut [f64],
    bar_state_out: &mut [f64],
    bullish_crossover_out: &mut [f64],
    bearish_crossunder_out: &mut [f64],
    box_offset_out: &mut [f64],
    bull_box_top_out: &mut [f64],
    bull_box_bottom_out: &mut [f64],
    bear_box_top_out: &mut [f64],
    bear_box_bottom_out: &mut [f64],
    bullish_test_out: &mut [f64],
    bearish_test_out: &mut [f64],
    bullish_test_level_out: &mut [f64],
    bearish_test_level_out: &mut [f64],
) {
    let mut state = HemaTrendLevelsCoreState::new(params);
    for i in 0..close.len() {
        let point = if open[i].is_finite()
            && high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
        {
            state.update(open[i], high[i], low[i], close[i])
        } else {
            state.reset();
            HemaTrendLevelsPoint::nan()
        };
        fast_hema_out[i] = point.fast_hema;
        slow_hema_out[i] = point.slow_hema;
        trend_direction_out[i] = point.trend_direction;
        bar_state_out[i] = point.bar_state;
        bullish_crossover_out[i] = point.bullish_crossover;
        bearish_crossunder_out[i] = point.bearish_crossunder;
        box_offset_out[i] = point.box_offset;
        bull_box_top_out[i] = point.bull_box_top;
        bull_box_bottom_out[i] = point.bull_box_bottom;
        bear_box_top_out[i] = point.bear_box_top;
        bear_box_bottom_out[i] = point.bear_box_bottom;
        bullish_test_out[i] = point.bullish_test;
        bearish_test_out[i] = point.bearish_test;
        bullish_test_level_out[i] = point.bullish_test_level;
        bearish_test_level_out[i] = point.bearish_test_level;
    }
}

fn hema_trend_levels_selected_row_from_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    field: HemaTrendLevelsOutputField,
    out: &mut [f64],
) {
    let mut state = HemaTrendLevelsCoreState::new(params);
    for i in 0..close.len() {
        let point = if open[i].is_finite()
            && high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
        {
            state.update(open[i], high[i], low[i], close[i])
        } else {
            state.reset();
            HemaTrendLevelsPoint::nan()
        };
        out[i] = match field {
            HemaTrendLevelsOutputField::FastHema => point.fast_hema,
            HemaTrendLevelsOutputField::SlowHema => point.slow_hema,
            HemaTrendLevelsOutputField::TrendDirection => point.trend_direction,
            HemaTrendLevelsOutputField::BarState => point.bar_state,
            HemaTrendLevelsOutputField::BullishCrossover => point.bullish_crossover,
            HemaTrendLevelsOutputField::BearishCrossunder => point.bearish_crossunder,
            HemaTrendLevelsOutputField::BoxOffset => point.box_offset,
            HemaTrendLevelsOutputField::BullBoxTop => point.bull_box_top,
            HemaTrendLevelsOutputField::BullBoxBottom => point.bull_box_bottom,
            HemaTrendLevelsOutputField::BearBoxTop => point.bear_box_top,
            HemaTrendLevelsOutputField::BearBoxBottom => point.bear_box_bottom,
            HemaTrendLevelsOutputField::BullishTest => point.bullish_test,
            HemaTrendLevelsOutputField::BearishTest => point.bearish_test,
            HemaTrendLevelsOutputField::BullishTestLevel => point.bullish_test_level,
            HemaTrendLevelsOutputField::BearishTestLevel => point.bearish_test_level,
        };
    }
}

pub fn hema_trend_levels(
    input: &HemaTrendLevelsInput,
) -> Result<HemaTrendLevelsOutput, HemaTrendLevelsError> {
    hema_trend_levels_with_kernel(input, Kernel::Auto)
}

pub fn hema_trend_levels_with_kernel(
    input: &HemaTrendLevelsInput,
    _kernel: Kernel,
) -> Result<HemaTrendLevelsOutput, HemaTrendLevelsError> {
    let (open, high, low, close) = input.as_slices();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(HemaTrendLevelsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(HemaTrendLevelsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if first_valid_ohlc(open, high, low, close) >= close.len() {
        return Err(HemaTrendLevelsError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    let len = close.len();
    let mut fast_hema = alloc_with_nan_prefix(len, 0);
    let mut slow_hema = alloc_with_nan_prefix(len, 0);
    let mut trend_direction = alloc_with_nan_prefix(len, 0);
    let mut bar_state = alloc_with_nan_prefix(len, 0);
    let mut bullish_crossover = alloc_with_nan_prefix(len, 0);
    let mut bearish_crossunder = alloc_with_nan_prefix(len, 0);
    let mut box_offset = alloc_with_nan_prefix(len, 0);
    let mut bull_box_top = alloc_with_nan_prefix(len, 0);
    let mut bull_box_bottom = alloc_with_nan_prefix(len, 0);
    let mut bear_box_top = alloc_with_nan_prefix(len, 0);
    let mut bear_box_bottom = alloc_with_nan_prefix(len, 0);
    let mut bullish_test = alloc_with_nan_prefix(len, 0);
    let mut bearish_test = alloc_with_nan_prefix(len, 0);
    let mut bullish_test_level = alloc_with_nan_prefix(len, 0);
    let mut bearish_test_level = alloc_with_nan_prefix(len, 0);
    hema_trend_levels_row_from_slices(
        open,
        high,
        low,
        close,
        params,
        &mut fast_hema,
        &mut slow_hema,
        &mut trend_direction,
        &mut bar_state,
        &mut bullish_crossover,
        &mut bearish_crossunder,
        &mut box_offset,
        &mut bull_box_top,
        &mut bull_box_bottom,
        &mut bear_box_top,
        &mut bear_box_bottom,
        &mut bullish_test,
        &mut bearish_test,
        &mut bullish_test_level,
        &mut bearish_test_level,
    );
    Ok(HemaTrendLevelsOutput {
        fast_hema,
        slow_hema,
        trend_direction,
        bar_state,
        bullish_crossover,
        bearish_crossunder,
        box_offset,
        bull_box_top,
        bull_box_bottom,
        bear_box_top,
        bear_box_bottom,
        bullish_test,
        bearish_test,
        bullish_test_level,
        bearish_test_level,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn hema_trend_levels_into_slices(
    fast_hema_out: &mut [f64],
    slow_hema_out: &mut [f64],
    trend_direction_out: &mut [f64],
    bar_state_out: &mut [f64],
    bullish_crossover_out: &mut [f64],
    bearish_crossunder_out: &mut [f64],
    box_offset_out: &mut [f64],
    bull_box_top_out: &mut [f64],
    bull_box_bottom_out: &mut [f64],
    bear_box_top_out: &mut [f64],
    bear_box_bottom_out: &mut [f64],
    bullish_test_out: &mut [f64],
    bearish_test_out: &mut [f64],
    bullish_test_level_out: &mut [f64],
    bearish_test_level_out: &mut [f64],
    input: &HemaTrendLevelsInput,
    _kernel: Kernel,
) -> Result<(), HemaTrendLevelsError> {
    let (open, high, low, close) = input.as_slices();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(HemaTrendLevelsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(HemaTrendLevelsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let expected = close.len();
    if fast_hema_out.len() != expected
        || slow_hema_out.len() != expected
        || trend_direction_out.len() != expected
        || bar_state_out.len() != expected
        || bullish_crossover_out.len() != expected
        || bearish_crossunder_out.len() != expected
        || box_offset_out.len() != expected
        || bull_box_top_out.len() != expected
        || bull_box_bottom_out.len() != expected
        || bear_box_top_out.len() != expected
        || bear_box_bottom_out.len() != expected
        || bullish_test_out.len() != expected
        || bearish_test_out.len() != expected
        || bullish_test_level_out.len() != expected
        || bearish_test_level_out.len() != expected
    {
        return Err(HemaTrendLevelsError::OutputLengthMismatch { expected });
    }
    if first_valid_ohlc(open, high, low, close) >= close.len() {
        return Err(HemaTrendLevelsError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    hema_trend_levels_row_from_slices(
        open,
        high,
        low,
        close,
        params,
        fast_hema_out,
        slow_hema_out,
        trend_direction_out,
        bar_state_out,
        bullish_crossover_out,
        bearish_crossunder_out,
        box_offset_out,
        bull_box_top_out,
        bull_box_bottom_out,
        bear_box_top_out,
        bear_box_bottom_out,
        bullish_test_out,
        bearish_test_out,
        bullish_test_level_out,
        bearish_test_level_out,
    );
    Ok(())
}

pub fn hema_trend_levels_output_into_slice(
    out: &mut [f64],
    input: &HemaTrendLevelsInput,
    _kernel: Kernel,
    field: HemaTrendLevelsOutputField,
) -> Result<(), HemaTrendLevelsError> {
    let (open, high, low, close) = input.as_slices();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(HemaTrendLevelsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(HemaTrendLevelsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let expected = close.len();
    if out.len() != expected {
        return Err(HemaTrendLevelsError::OutputLengthMismatch { expected });
    }
    if first_valid_ohlc(open, high, low, close) >= close.len() {
        return Err(HemaTrendLevelsError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    hema_trend_levels_selected_row_from_slices(open, high, low, close, params, field, out);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn hema_trend_levels_into(
    fast_hema_out: &mut [f64],
    slow_hema_out: &mut [f64],
    trend_direction_out: &mut [f64],
    bar_state_out: &mut [f64],
    bullish_crossover_out: &mut [f64],
    bearish_crossunder_out: &mut [f64],
    box_offset_out: &mut [f64],
    bull_box_top_out: &mut [f64],
    bull_box_bottom_out: &mut [f64],
    bear_box_top_out: &mut [f64],
    bear_box_bottom_out: &mut [f64],
    bullish_test_out: &mut [f64],
    bearish_test_out: &mut [f64],
    bullish_test_level_out: &mut [f64],
    bearish_test_level_out: &mut [f64],
    input: &HemaTrendLevelsInput,
) -> Result<(), HemaTrendLevelsError> {
    hema_trend_levels_into_slices(
        fast_hema_out,
        slow_hema_out,
        trend_direction_out,
        bar_state_out,
        bullish_crossover_out,
        bearish_crossunder_out,
        box_offset_out,
        bull_box_top_out,
        bull_box_bottom_out,
        bear_box_top_out,
        bear_box_bottom_out,
        bullish_test_out,
        bearish_test_out,
        bullish_test_level_out,
        bearish_test_level_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HemaTrendLevelsBatchRange {
    pub fast_length: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
}

impl Default for HemaTrendLevelsBatchRange {
    fn default() -> Self {
        Self {
            fast_length: (DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
            slow_length: (DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HemaTrendLevelsBatchOutput {
    pub fast_hema: Vec<f64>,
    pub slow_hema: Vec<f64>,
    pub trend_direction: Vec<f64>,
    pub bar_state: Vec<f64>,
    pub bullish_crossover: Vec<f64>,
    pub bearish_crossunder: Vec<f64>,
    pub box_offset: Vec<f64>,
    pub bull_box_top: Vec<f64>,
    pub bull_box_bottom: Vec<f64>,
    pub bear_box_top: Vec<f64>,
    pub bear_box_bottom: Vec<f64>,
    pub bullish_test: Vec<f64>,
    pub bearish_test: Vec<f64>,
    pub bullish_test_level: Vec<f64>,
    pub bearish_test_level: Vec<f64>,
    pub combos: Vec<HemaTrendLevelsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct HemaTrendLevelsBatchBuilder {
    range: HemaTrendLevelsBatchRange,
    kernel: Kernel,
}

impl HemaTrendLevelsBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn fast_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_length = (start, end, step);
        self
    }

    pub fn slow_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_length = (start, end, step);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<HemaTrendLevelsBatchOutput, HemaTrendLevelsError> {
        hema_trend_levels_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }
}

fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, HemaTrendLevelsError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }
    if out.is_empty() {
        return Err(HemaTrendLevelsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_hema_trend_levels(
    sweep: &HemaTrendLevelsBatchRange,
) -> Result<Vec<HemaTrendLevelsParams>, HemaTrendLevelsError> {
    let fast_lengths = expand_axis_usize(sweep.fast_length)?;
    let slow_lengths = expand_axis_usize(sweep.slow_length)?;
    let mut combos = Vec::with_capacity(fast_lengths.len().saturating_mul(slow_lengths.len()));
    for fast_length in fast_lengths {
        for &slow_length in &slow_lengths {
            let params = HemaTrendLevelsParams {
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
            };
            let _ = resolve_params(&params)?;
            combos.push(params);
        }
    }
    Ok(combos)
}

pub fn hema_trend_levels_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HemaTrendLevelsBatchRange,
    kernel: Kernel,
) -> Result<HemaTrendLevelsBatchOutput, HemaTrendLevelsError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HemaTrendLevelsError::InvalidKernelForBatch(other)),
    };
    hema_trend_levels_batch_par_slice(open, high, low, close, sweep, batch_kernel.to_non_batch())
}

pub fn hema_trend_levels_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HemaTrendLevelsBatchRange,
    kernel: Kernel,
) -> Result<HemaTrendLevelsBatchOutput, HemaTrendLevelsError> {
    hema_trend_levels_batch_inner(open, high, low, close, sweep, kernel, false)
}

pub fn hema_trend_levels_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HemaTrendLevelsBatchRange,
    kernel: Kernel,
) -> Result<HemaTrendLevelsBatchOutput, HemaTrendLevelsError> {
    hema_trend_levels_batch_inner(open, high, low, close, sweep, kernel, true)
}

#[allow(clippy::too_many_lines)]
fn hema_trend_levels_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HemaTrendLevelsBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<HemaTrendLevelsBatchOutput, HemaTrendLevelsError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(HemaTrendLevelsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(HemaTrendLevelsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if first_valid_ohlc(open, high, low, close) >= close.len() {
        return Err(HemaTrendLevelsError::AllValuesNaN);
    }

    let combos = expand_grid_hema_trend_levels(sweep)?;
    let resolved = combos
        .iter()
        .map(resolve_params)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HemaTrendLevelsError::OutputLengthMismatch {
            expected: usize::MAX,
        })?;
    let zero_prefixes = vec![0usize; rows];

    macro_rules! alloc_matrix {
        ($mu:ident, $guard:ident, $slice:ident) => {
            let mut $mu = make_uninit_matrix(rows, cols);
            init_matrix_prefixes(&mut $mu, cols, &zero_prefixes);
            let mut $guard = ManuallyDrop::new($mu);
            let $slice =
                unsafe { std::slice::from_raw_parts_mut($guard.as_mut_ptr() as *mut f64, total) };
        };
    }

    alloc_matrix!(fast_mu, fast_guard, fast_out);
    alloc_matrix!(slow_mu, slow_guard, slow_out);
    alloc_matrix!(trend_mu, trend_guard, trend_out);
    alloc_matrix!(bar_mu, bar_guard, bar_out);
    alloc_matrix!(bull_cross_mu, bull_cross_guard, bull_cross_out);
    alloc_matrix!(bear_cross_mu, bear_cross_guard, bear_cross_out);
    alloc_matrix!(offset_mu, offset_guard, offset_out);
    alloc_matrix!(bull_top_mu, bull_top_guard, bull_top_out);
    alloc_matrix!(bull_bottom_mu, bull_bottom_guard, bull_bottom_out);
    alloc_matrix!(bear_top_mu, bear_top_guard, bear_top_out);
    alloc_matrix!(bear_bottom_mu, bear_bottom_guard, bear_bottom_out);
    alloc_matrix!(bull_test_mu, bull_test_guard, bull_test_out);
    alloc_matrix!(bear_test_mu, bear_test_guard, bear_test_out);
    alloc_matrix!(
        bull_test_level_mu,
        bull_test_level_guard,
        bull_test_level_out
    );
    alloc_matrix!(
        bear_test_level_mu,
        bear_test_level_guard,
        bear_test_level_out
    );

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let fp = fast_out.as_mut_ptr() as usize;
            let sp = slow_out.as_mut_ptr() as usize;
            let tp = trend_out.as_mut_ptr() as usize;
            let bp = bar_out.as_mut_ptr() as usize;
            let bcp = bull_cross_out.as_mut_ptr() as usize;
            let bxp = bear_cross_out.as_mut_ptr() as usize;
            let op = offset_out.as_mut_ptr() as usize;
            let btp = bull_top_out.as_mut_ptr() as usize;
            let bbp = bull_bottom_out.as_mut_ptr() as usize;
            let rtp = bear_top_out.as_mut_ptr() as usize;
            let rbp = bear_bottom_out.as_mut_ptr() as usize;
            let bstp = bull_test_out.as_mut_ptr() as usize;
            let rstp = bear_test_out.as_mut_ptr() as usize;
            let bslp = bull_test_level_out.as_mut_ptr() as usize;
            let rslp = bear_test_level_out.as_mut_ptr() as usize;
            resolved
                .par_iter()
                .enumerate()
                .for_each(|(row, params)| unsafe {
                    let start = row * cols;
                    hema_trend_levels_row_from_slices(
                        open,
                        high,
                        low,
                        close,
                        *params,
                        std::slice::from_raw_parts_mut((fp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((sp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((tp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((bp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((bcp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((bxp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((op as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((btp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((bbp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((rtp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((rbp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((bstp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((rstp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((bslp as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut((rslp as *mut f64).add(start), cols),
                    );
                });
        }
        #[cfg(target_arch = "wasm32")]
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            hema_trend_levels_row_from_slices(
                open,
                high,
                low,
                close,
                *params,
                &mut fast_out[start..end],
                &mut slow_out[start..end],
                &mut trend_out[start..end],
                &mut bar_out[start..end],
                &mut bull_cross_out[start..end],
                &mut bear_cross_out[start..end],
                &mut offset_out[start..end],
                &mut bull_top_out[start..end],
                &mut bull_bottom_out[start..end],
                &mut bear_top_out[start..end],
                &mut bear_bottom_out[start..end],
                &mut bull_test_out[start..end],
                &mut bear_test_out[start..end],
                &mut bull_test_level_out[start..end],
                &mut bear_test_level_out[start..end],
            );
        }
    } else {
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            hema_trend_levels_row_from_slices(
                open,
                high,
                low,
                close,
                *params,
                &mut fast_out[start..end],
                &mut slow_out[start..end],
                &mut trend_out[start..end],
                &mut bar_out[start..end],
                &mut bull_cross_out[start..end],
                &mut bear_cross_out[start..end],
                &mut offset_out[start..end],
                &mut bull_top_out[start..end],
                &mut bull_bottom_out[start..end],
                &mut bear_top_out[start..end],
                &mut bear_bottom_out[start..end],
                &mut bull_test_out[start..end],
                &mut bear_test_out[start..end],
                &mut bull_test_level_out[start..end],
                &mut bear_test_level_out[start..end],
            );
        }
    }

    macro_rules! into_vec {
        ($guard:ident) => {
            unsafe {
                Vec::from_raw_parts(
                    $guard.as_mut_ptr() as *mut f64,
                    $guard.len(),
                    $guard.capacity(),
                )
            }
        };
    }

    Ok(HemaTrendLevelsBatchOutput {
        fast_hema: into_vec!(fast_guard),
        slow_hema: into_vec!(slow_guard),
        trend_direction: into_vec!(trend_guard),
        bar_state: into_vec!(bar_guard),
        bullish_crossover: into_vec!(bull_cross_guard),
        bearish_crossunder: into_vec!(bear_cross_guard),
        box_offset: into_vec!(offset_guard),
        bull_box_top: into_vec!(bull_top_guard),
        bull_box_bottom: into_vec!(bull_bottom_guard),
        bear_box_top: into_vec!(bear_top_guard),
        bear_box_bottom: into_vec!(bear_bottom_guard),
        bullish_test: into_vec!(bull_test_guard),
        bearish_test: into_vec!(bear_test_guard),
        bullish_test_level: into_vec!(bull_test_level_guard),
        bearish_test_level: into_vec!(bear_test_level_guard),
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn hema_trend_levels_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &HemaTrendLevelsBatchRange,
    kernel: Kernel,
    parallel: bool,
    fast_hema: &mut [f64],
    slow_hema: &mut [f64],
    trend_direction: &mut [f64],
    bar_state: &mut [f64],
    bullish_crossover: &mut [f64],
    bearish_crossunder: &mut [f64],
    box_offset: &mut [f64],
    bull_box_top: &mut [f64],
    bull_box_bottom: &mut [f64],
    bear_box_top: &mut [f64],
    bear_box_bottom: &mut [f64],
    bullish_test: &mut [f64],
    bearish_test: &mut [f64],
    bullish_test_level: &mut [f64],
    bearish_test_level: &mut [f64],
) -> Result<Vec<HemaTrendLevelsParams>, HemaTrendLevelsError> {
    let out = hema_trend_levels_batch_inner(open, high, low, close, sweep, kernel, parallel)?;
    let total = out.rows * out.cols;
    if fast_hema.len() != total
        || slow_hema.len() != total
        || trend_direction.len() != total
        || bar_state.len() != total
        || bullish_crossover.len() != total
        || bearish_crossunder.len() != total
        || box_offset.len() != total
        || bull_box_top.len() != total
        || bull_box_bottom.len() != total
        || bear_box_top.len() != total
        || bear_box_bottom.len() != total
        || bullish_test.len() != total
        || bearish_test.len() != total
        || bullish_test_level.len() != total
        || bearish_test_level.len() != total
    {
        return Err(HemaTrendLevelsError::OutputLengthMismatch { expected: total });
    }
    fast_hema.copy_from_slice(&out.fast_hema);
    slow_hema.copy_from_slice(&out.slow_hema);
    trend_direction.copy_from_slice(&out.trend_direction);
    bar_state.copy_from_slice(&out.bar_state);
    bullish_crossover.copy_from_slice(&out.bullish_crossover);
    bearish_crossunder.copy_from_slice(&out.bearish_crossunder);
    box_offset.copy_from_slice(&out.box_offset);
    bull_box_top.copy_from_slice(&out.bull_box_top);
    bull_box_bottom.copy_from_slice(&out.bull_box_bottom);
    bear_box_top.copy_from_slice(&out.bear_box_top);
    bear_box_bottom.copy_from_slice(&out.bear_box_bottom);
    bullish_test.copy_from_slice(&out.bullish_test);
    bearish_test.copy_from_slice(&out.bearish_test);
    bullish_test_level.copy_from_slice(&out.bullish_test_level);
    bearish_test_level.copy_from_slice(&out.bearish_test_level);
    Ok(out.combos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(length);
        let mut high = Vec::with_capacity(length);
        let mut low = Vec::with_capacity(length);
        let mut close = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            let o = if i < length / 3 {
                100.0 - x * 0.08 + (x * 0.03).sin() * 0.2
            } else if i < 2 * length / 3 {
                94.0 + x * 0.18 + (x * 0.05).sin() * 0.3
            } else {
                140.0 - x * 0.14 + (x * 0.04).cos() * 0.35
            };
            let c = o + (x * 0.07).cos() * 0.6;
            let h = o.max(c) + 0.75;
            let l = o.min(c) - 0.75;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (open, high, low, close)
    }

    #[test]
    fn hema_trend_levels_matches_hand_derived_cross_box_and_reset_vector() {
        // Published HEMA definition: EMA(2*EMA(close, length/2) -
        // EMA(close, length), sqrt(length)).  Choosing fast=1 and slow=2
        // makes the independent arithmetic especially transparent:
        // fast=close and slow=2*close-EMA_2(close), alpha(EMA_2)=2/3.
        // Fourteen flat true ranges seed ATR=2 before the two deliberate
        // direction changes create first a bear box, then a bull box.
        let mut open = vec![100.0; 19];
        let mut high = vec![101.0; 19];
        let mut low = vec![99.0; 19];
        let mut close = vec![100.0; 19];

        open[14] = 102.0;
        high[14] = 103.0;
        low[14] = 101.0;
        close[14] = 102.0;

        open[15] = 101.0;
        high[15] = 103.0;
        low[15] = 100.0;
        close[15] = 101.0;

        open[16] = 102.0;
        high[16] = 103.0;
        low[16] = 100.0;
        close[16] = 102.0;

        open[17] = f64::NAN;
        high[17] = f64::NAN;
        low[17] = f64::NAN;
        close[17] = f64::NAN;

        let out = hema_trend_levels(&HemaTrendLevelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            HemaTrendLevelsParams {
                fast_length: Some(1),
                slow_length: Some(2),
            },
        ))
        .unwrap();

        let close_enough = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "actual={actual:.17} expected={expected:.17}"
            );
        };

        // ATR seed: fourteen ranges of exactly 2, therefore offset=ATR/2=1.
        close_enough(out.fast_hema[13], 100.0);
        close_enough(out.slow_hema[13], 100.0);
        close_enough(out.box_offset[13], 1.0);

        // Bar 14: EMA_2=304/3, slow=308/3. A bearish cross seeds the
        // resistance box at high - (29/14)/2 = 2855/28.
        close_enough(out.fast_hema[14], 102.0);
        close_enough(out.slow_hema[14], 308.0 / 3.0);
        assert_eq!(out.trend_direction[14], -1.0);
        assert_eq!(out.bar_state[14], 0.0);
        assert_eq!(out.bullish_crossover[14], 0.0);
        assert_eq!(out.bearish_crossunder[14], 1.0);
        close_enough(out.box_offset[14], 29.0 / 28.0);
        assert!(out.bull_box_top[14].is_nan());
        assert!(out.bull_box_bottom[14].is_nan());
        close_enough(out.bear_box_top[14], 2855.0 / 28.0);
        close_enough(out.bear_box_bottom[14], 103.0);
        assert_eq!(out.bullish_test[14], 0.0);
        assert_eq!(out.bearish_test[14], 0.0);
        assert!(out.bullish_test_level[14].is_nan());
        assert!(out.bearish_test_level[14].is_nan());

        // Bar 15: EMA_2=910/9, slow=908/9. The bullish cross seeds a
        // support box. The same candle cleanly retests the still-live bear
        // box, whose reported test level is its 103 bottom.
        close_enough(out.fast_hema[15], 101.0);
        close_enough(out.slow_hema[15], 908.0 / 9.0);
        assert_eq!(out.trend_direction[15], 1.0);
        assert_eq!(out.bullish_crossover[15], 1.0);
        assert_eq!(out.bearish_crossunder[15], 0.0);
        close_enough(out.box_offset[15], 419.0 / 392.0);
        close_enough(out.bull_box_top[15], 100.0 + 419.0 / 392.0);
        close_enough(out.bull_box_bottom[15], 100.0);
        close_enough(out.bear_box_top[15], 2855.0 / 28.0);
        close_enough(out.bear_box_bottom[15], 103.0);
        assert_eq!(out.bullish_test[15], 0.0);
        assert_eq!(out.bearish_test[15], 1.0);
        assert!(out.bullish_test_level[15].is_nan());
        close_enough(out.bearish_test_level[15], 103.0);

        // Bar 16 reverses again, replaces the bear box and cleanly retests
        // the persistent bull box from bar 15.
        close_enough(out.fast_hema[16], 102.0);
        close_enough(out.slow_hema[16], 2762.0 / 27.0);
        assert_eq!(out.trend_direction[16], -1.0);
        assert_eq!(out.bullish_crossover[16], 0.0);
        assert_eq!(out.bearish_crossunder[16], 1.0);
        close_enough(out.box_offset[16], 6035.0 / 5488.0);
        close_enough(out.bear_box_top[16], 103.0 - 6035.0 / 5488.0);
        close_enough(out.bear_box_bottom[16], 103.0);
        assert_eq!(out.bullish_test[16], 1.0);
        assert_eq!(out.bearish_test[16], 0.0);
        close_enough(out.bullish_test_level[16], 100.0);
        assert!(out.bearish_test_level[16].is_nan());

        // A non-finite OHLC bar invalidates every output and clears every
        // recurrence/box. The next finite bar restarts both HEMAs from close,
        // with no stale crossover, ATR or box state leaking across the gap.
        for series in [
            &out.fast_hema,
            &out.slow_hema,
            &out.trend_direction,
            &out.bar_state,
            &out.bullish_crossover,
            &out.bearish_crossunder,
            &out.box_offset,
            &out.bull_box_top,
            &out.bull_box_bottom,
            &out.bear_box_top,
            &out.bear_box_bottom,
            &out.bullish_test,
            &out.bearish_test,
            &out.bullish_test_level,
            &out.bearish_test_level,
        ] {
            assert!(series[17].is_nan());
        }
        close_enough(out.fast_hema[18], 100.0);
        close_enough(out.slow_hema[18], 100.0);
        assert_eq!(out.trend_direction[18], 0.0);
        assert_eq!(out.bar_state[18], 0.0);
        assert_eq!(out.bullish_crossover[18], 0.0);
        assert_eq!(out.bearish_crossunder[18], 0.0);
        assert!(out.box_offset[18].is_nan());
        assert!(out.bull_box_top[18].is_nan());
        assert!(out.bear_box_top[18].is_nan());
        assert_eq!(out.bullish_test[18], 0.0);
        assert_eq!(out.bearish_test[18], 0.0);
    }

    #[test]
    fn hema_trend_levels_output_contract() {
        let (open, high, low, close) = sample_ohlc(240);
        let out = hema_trend_levels(&HemaTrendLevelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            HemaTrendLevelsParams::default(),
        ))
        .unwrap();
        assert_eq!(out.fast_hema.len(), close.len());
        assert!(out.fast_hema.iter().any(|v| v.is_finite()));
        assert!(
            out.bullish_crossover
                .iter()
                .all(|v| v.is_nan() || *v == 0.0 || *v == 1.0)
        );
    }

    #[test]
    fn hema_trend_levels_crossovers_seed_boxes() {
        let (open, high, low, close) = sample_ohlc(260);
        let out = hema_trend_levels(&HemaTrendLevelsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            HemaTrendLevelsParams::default(),
        ))
        .unwrap();
        let mut saw_bull = false;
        let mut saw_bear = false;
        for i in 0..close.len() {
            if out.bullish_crossover[i] == 1.0 && out.box_offset[i].is_finite() {
                saw_bull = true;
                assert!((out.bull_box_top[i] - (low[i] + out.box_offset[i])).abs() <= 1e-12);
            }
            if out.bearish_crossunder[i] == 1.0 && out.box_offset[i].is_finite() {
                saw_bear = true;
                assert!((out.bear_box_top[i] - (high[i] - out.box_offset[i])).abs() <= 1e-12);
            }
        }
        assert!(saw_bull && saw_bear);
    }
}
