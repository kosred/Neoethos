use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_FACTOR: f64 = 5.0;
const DEFAULT_SLOPE: f64 = 14.0;
const DEFAULT_WIDTH_PERCENT: f64 = 80.0;
const ATR_PERIOD: usize = 200;

#[derive(Debug, Clone)]
pub enum HyperTrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct HyperTrendOutput {
    pub upper: Vec<f64>,
    pub average: Vec<f64>,
    pub lower: Vec<f64>,
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperTrendOutputField {
    Upper,
    Average,
    Lower,
    Trend,
    Changed,
}

#[derive(Debug, Clone)]
pub struct HyperTrendParams {
    pub factor: Option<f64>,
    pub slope: Option<f64>,
    pub width_percent: Option<f64>,
}

impl Default for HyperTrendParams {
    fn default() -> Self {
        Self {
            factor: Some(DEFAULT_FACTOR),
            slope: Some(DEFAULT_SLOPE),
            width_percent: Some(DEFAULT_WIDTH_PERCENT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HyperTrendInput<'a> {
    pub data: HyperTrendData<'a>,
    pub params: HyperTrendParams,
}

impl<'a> HyperTrendInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HyperTrendParams) -> Self {
        Self {
            data: HyperTrendData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
        params: HyperTrendParams,
    ) -> Self {
        Self {
            data: HyperTrendData::Slices { high, low, source },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", HyperTrendParams::default())
    }

    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(DEFAULT_FACTOR)
    }

    #[inline]
    pub fn get_slope(&self) -> f64 {
        self.params.slope.unwrap_or(DEFAULT_SLOPE)
    }

    #[inline]
    pub fn get_width_percent(&self) -> f64 {
        self.params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            HyperTrendData::Candles { candles, source } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                source_type(candles, source),
            ),
            HyperTrendData::Slices { high, low, source } => (*high, *low, *source),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HyperTrendBuilder {
    factor: Option<f64>,
    slope: Option<f64>,
    width_percent: Option<f64>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for HyperTrendBuilder {
    fn default() -> Self {
        Self {
            factor: None,
            slope: None,
            width_percent: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HyperTrendBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn factor(mut self, value: f64) -> Self {
        self.factor = Some(value);
        self
    }

    #[inline]
    pub fn slope(mut self, value: f64) -> Self {
        self.slope = Some(value);
        self
    }

    #[inline]
    pub fn width_percent(mut self, value: f64) -> Self {
        self.width_percent = Some(value);
        self
    }

    #[inline]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<HyperTrendOutput, HyperTrendError> {
        let input = HyperTrendInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("close"),
            HyperTrendParams {
                factor: self.factor,
                slope: self.slope,
                width_percent: self.width_percent,
            },
        );
        hypertrend_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<HyperTrendOutput, HyperTrendError> {
        let input = HyperTrendInput::from_slices(
            high,
            low,
            source,
            HyperTrendParams {
                factor: self.factor,
                slope: self.slope,
                width_percent: self.width_percent,
            },
        );
        hypertrend_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<HyperTrendStream, HyperTrendError> {
        HyperTrendStream::try_new(HyperTrendParams {
            factor: self.factor,
            slope: self.slope,
            width_percent: self.width_percent,
        })
    }
}

#[derive(Debug, Error)]
pub enum HyperTrendError {
    #[error("hypertrend: Empty input data.")]
    EmptyInputData,
    #[error("hypertrend: Input length mismatch: high={high}, low={low}, source={source_len}")]
    DataLengthMismatch {
        high: usize,
        low: usize,
        source_len: usize,
    },
    #[error("hypertrend: All input values are invalid.")]
    AllValuesNaN,
    #[error("hypertrend: Invalid factor: {factor}")]
    InvalidFactor { factor: f64 },
    #[error("hypertrend: Invalid slope: {slope}")]
    InvalidSlope { slope: f64 },
    #[error("hypertrend: Invalid width_percent: {width_percent}")]
    InvalidWidthPercent { width_percent: f64 },
    #[error("hypertrend: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("hypertrend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("hypertrend: Invalid float range: start={start}, end={end}, step={step}")]
    InvalidFloatRange { start: f64, end: f64, step: f64 },
    #[error("hypertrend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && source.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], source: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| valid_bar(high[i], low[i], source[i]))
}

#[inline(always)]
fn normalize_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[inline(always)]
fn validate_lengths(high: &[f64], low: &[f64], source: &[f64]) -> Result<(), HyperTrendError> {
    if high.is_empty() || low.is_empty() || source.is_empty() {
        return Err(HyperTrendError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != source.len() {
        return Err(HyperTrendError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            source_len: source.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_params(factor: f64, slope: f64, width_percent: f64) -> Result<(), HyperTrendError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(HyperTrendError::InvalidFactor { factor });
    }
    if !slope.is_finite() || slope <= 0.0 {
        return Err(HyperTrendError::InvalidSlope { slope });
    }
    if !width_percent.is_finite() || !(0.0..=100.0).contains(&width_percent) {
        return Err(HyperTrendError::InvalidWidthPercent { width_percent });
    }
    Ok(())
}

#[inline(always)]
fn pine_sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    if prev_close.is_finite() {
        let a = high - low;
        let b = (high - prev_close).abs();
        let c = (low - prev_close).abs();
        a.max(b).max(c)
    } else {
        high - low
    }
}

fn compute_atr_zeroed(high: &[f64], low: &[f64], source: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; source.len()];
    let mut prev_close = f64::NAN;
    let mut seed_sum = 0.0;
    let mut seed_count = 0usize;
    let mut atr = f64::NAN;

    for i in 0..source.len() {
        if !valid_bar(high[i], low[i], source[i]) {
            out[i] = 0.0;
            prev_close = f64::NAN;
            seed_sum = 0.0;
            seed_count = 0;
            atr = f64::NAN;
            continue;
        }

        let tr = true_range(high[i], low[i], prev_close);
        prev_close = source[i];

        if seed_count < ATR_PERIOD {
            seed_sum += tr;
            seed_count += 1;
            if seed_count == ATR_PERIOD {
                atr = seed_sum / ATR_PERIOD as f64;
                out[i] = atr;
            }
            continue;
        }

        atr = ((atr * (ATR_PERIOD as f64 - 1.0)) + tr) / ATR_PERIOD as f64;
        out[i] = atr;
    }

    out
}

#[inline(always)]
fn hypertrend_row_scalar(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    factor: f64,
    slope: f64,
    width_ratio: f64,
    atr_values: &[f64],
    out_upper: &mut [f64],
    out_average: &mut [f64],
    out_lower: &mut [f64],
    out_trend: &mut [f64],
    out_changed: &mut [f64],
) {
    let mut initialized = false;
    let mut avg = 0.0;
    let mut hold = 0.0;
    let mut os = 1.0;

    for i in 0..source.len() {
        let src = source[i];
        if !valid_bar(high[i], low[i], src) {
            out_upper[i] = f64::NAN;
            out_average[i] = f64::NAN;
            out_lower[i] = f64::NAN;
            out_trend[i] = f64::NAN;
            out_changed[i] = f64::NAN;
            initialized = false;
            avg = 0.0;
            hold = 0.0;
            os = 1.0;
            continue;
        }

        if !initialized {
            avg = src;
            hold = 0.0;
            os = 1.0;
            out_average[i] = avg;
            out_upper[i] = avg;
            out_lower[i] = avg;
            out_trend[i] = os;
            out_changed[i] = 0.0;
            initialized = true;
            continue;
        }

        let atr = atr_values[i] * factor;
        let next_avg = if (src - avg).abs() > atr {
            0.5 * (src + avg)
        } else {
            avg + os * (hold / factor / slope)
        };
        let next_os = pine_sign(next_avg - avg);
        let changed = if next_os != os { 1.0 } else { 0.0 };
        let next_hold = if changed != 0.0 { atr } else { hold };
        let upper = next_avg + width_ratio * next_hold;
        let lower = next_avg - width_ratio * next_hold;

        out_upper[i] = upper;
        out_average[i] = next_avg;
        out_lower[i] = lower;
        out_trend[i] = next_os;
        out_changed[i] = changed;

        avg = next_avg;
        hold = next_hold;
        os = next_os;
    }
}

#[inline(always)]
fn hypertrend_selected_row_scalar(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    factor: f64,
    slope: f64,
    width_ratio: f64,
    atr_values: &[f64],
    field: HyperTrendOutputField,
    out: &mut [f64],
) {
    let mut initialized = false;
    let mut avg = 0.0;
    let mut hold = 0.0;
    let mut os = 1.0;

    for i in 0..source.len() {
        let src = source[i];
        if !valid_bar(high[i], low[i], src) {
            out[i] = f64::NAN;
            initialized = false;
            avg = 0.0;
            hold = 0.0;
            os = 1.0;
            continue;
        }

        let (upper, average, lower, trend, changed) = if !initialized {
            avg = src;
            hold = 0.0;
            os = 1.0;
            initialized = true;
            (avg, avg, avg, os, 0.0)
        } else {
            let atr = atr_values[i] * factor;
            let next_avg = if (src - avg).abs() > atr {
                0.5 * (src + avg)
            } else {
                avg + os * (hold / factor / slope)
            };
            let next_os = pine_sign(next_avg - avg);
            let changed = if next_os != os { 1.0 } else { 0.0 };
            let next_hold = if changed != 0.0 { atr } else { hold };
            let upper = next_avg + width_ratio * next_hold;
            let lower = next_avg - width_ratio * next_hold;
            avg = next_avg;
            hold = next_hold;
            os = next_os;
            (upper, next_avg, lower, next_os, changed)
        };

        out[i] = match field {
            HyperTrendOutputField::Upper => upper,
            HyperTrendOutputField::Average => average,
            HyperTrendOutputField::Lower => lower,
            HyperTrendOutputField::Trend => trend,
            HyperTrendOutputField::Changed => changed,
        };
    }
}

#[inline]
pub fn hypertrend(input: &HyperTrendInput) -> Result<HyperTrendOutput, HyperTrendError> {
    hypertrend_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn hypertrend_with_kernel(
    input: &HyperTrendInput,
    kernel: Kernel,
) -> Result<HyperTrendOutput, HyperTrendError> {
    let (high, low, source) = input.as_refs();
    validate_lengths(high, low, source)?;

    let factor = input.get_factor();
    let slope = input.get_slope();
    let width_percent = input.get_width_percent();
    validate_params(factor, slope, width_percent)?;

    let first_valid = first_valid_bar(high, low, source).ok_or(HyperTrendError::AllValuesNaN)?;
    let _kernel = normalize_kernel(kernel);
    let atr_values = compute_atr_zeroed(high, low, source);
    let width_ratio = width_percent * 0.01;
    let len = source.len();

    let mut upper = alloc_with_nan_prefix(len, first_valid);
    let mut average = alloc_with_nan_prefix(len, first_valid);
    let mut lower = alloc_with_nan_prefix(len, first_valid);
    let mut trend = alloc_with_nan_prefix(len, first_valid);
    let mut changed = alloc_with_nan_prefix(len, first_valid);

    hypertrend_row_scalar(
        high,
        low,
        source,
        factor,
        slope,
        width_ratio,
        &atr_values,
        &mut upper,
        &mut average,
        &mut lower,
        &mut trend,
        &mut changed,
    );

    Ok(HyperTrendOutput {
        upper,
        average,
        lower,
        trend,
        changed,
    })
}

#[inline]
pub fn hypertrend_into_slice(
    out_upper: &mut [f64],
    out_average: &mut [f64],
    out_lower: &mut [f64],
    out_trend: &mut [f64],
    out_changed: &mut [f64],
    input: &HyperTrendInput,
    kernel: Kernel,
) -> Result<(), HyperTrendError> {
    let (high, low, source) = input.as_refs();
    validate_lengths(high, low, source)?;
    let len = source.len();
    if out_upper.len() != len
        || out_average.len() != len
        || out_lower.len() != len
        || out_trend.len() != len
        || out_changed.len() != len
    {
        return Err(HyperTrendError::OutputLengthMismatch {
            expected: len,
            got: out_upper
                .len()
                .max(out_average.len())
                .max(out_lower.len())
                .max(out_trend.len())
                .max(out_changed.len()),
        });
    }

    let factor = input.get_factor();
    let slope = input.get_slope();
    let width_percent = input.get_width_percent();
    validate_params(factor, slope, width_percent)?;
    let _kernel = normalize_kernel(kernel);
    let atr_values = compute_atr_zeroed(high, low, source);

    hypertrend_row_scalar(
        high,
        low,
        source,
        factor,
        slope,
        width_percent * 0.01,
        &atr_values,
        out_upper,
        out_average,
        out_lower,
        out_trend,
        out_changed,
    );
    Ok(())
}

#[inline]
pub fn hypertrend_output_into_slice(
    out: &mut [f64],
    input: &HyperTrendInput,
    kernel: Kernel,
    field: HyperTrendOutputField,
) -> Result<(), HyperTrendError> {
    let (high, low, source) = input.as_refs();
    validate_lengths(high, low, source)?;
    let len = source.len();
    if out.len() != len {
        return Err(HyperTrendError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let factor = input.get_factor();
    let slope = input.get_slope();
    let width_percent = input.get_width_percent();
    validate_params(factor, slope, width_percent)?;
    let _kernel = normalize_kernel(kernel);
    let atr_values = compute_atr_zeroed(high, low, source);

    hypertrend_selected_row_scalar(
        high,
        low,
        source,
        factor,
        slope,
        width_percent * 0.01,
        &atr_values,
        field,
        out,
    );
    Ok(())
}

#[inline]
pub fn hypertrend_into(
    input: &HyperTrendInput,
    out_upper: &mut [f64],
    out_average: &mut [f64],
    out_lower: &mut [f64],
    out_trend: &mut [f64],
    out_changed: &mut [f64],
) -> Result<(), HyperTrendError> {
    hypertrend_into_slice(
        out_upper,
        out_average,
        out_lower,
        out_trend,
        out_changed,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
struct HyperTrendAtrState {
    prev_close: f64,
    seed_sum: f64,
    seed_count: usize,
    atr: f64,
}

impl HyperTrendAtrState {
    #[inline]
    fn new() -> Self {
        Self {
            prev_close: f64::NAN,
            seed_sum: 0.0,
            seed_count: 0,
            atr: f64::NAN,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.seed_sum = 0.0;
        self.seed_count = 0;
        self.atr = f64::NAN;
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64, source: f64) -> Option<f64> {
        if !valid_bar(high, low, source) {
            self.reset();
            return None;
        }

        let tr = true_range(high, low, self.prev_close);
        self.prev_close = source;

        if self.seed_count < ATR_PERIOD {
            self.seed_sum += tr;
            self.seed_count += 1;
            if self.seed_count == ATR_PERIOD {
                self.atr = self.seed_sum / ATR_PERIOD as f64;
                return Some(self.atr);
            }
            return Some(0.0);
        }

        self.atr = ((self.atr * (ATR_PERIOD as f64 - 1.0)) + tr) / ATR_PERIOD as f64;
        Some(self.atr)
    }
}

#[derive(Clone, Debug)]
pub struct HyperTrendStream {
    factor: f64,
    slope: f64,
    width_ratio: f64,
    atr: HyperTrendAtrState,
    initialized: bool,
    avg: f64,
    hold: f64,
    os: f64,
}

impl HyperTrendStream {
    #[inline]
    pub fn try_new(params: HyperTrendParams) -> Result<Self, HyperTrendError> {
        let factor = params.factor.unwrap_or(DEFAULT_FACTOR);
        let slope = params.slope.unwrap_or(DEFAULT_SLOPE);
        let width_percent = params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT);
        validate_params(factor, slope, width_percent)?;
        Ok(Self {
            factor,
            slope,
            width_ratio: width_percent * 0.01,
            atr: HyperTrendAtrState::new(),
            initialized: false,
            avg: 0.0,
            hold: 0.0,
            os: 1.0,
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        source: f64,
    ) -> Option<(f64, f64, f64, f64, f64)> {
        let atr_raw = self.atr.update(high, low, source)?;
        if !self.initialized {
            self.avg = source;
            self.hold = 0.0;
            self.os = 1.0;
            self.initialized = true;
            return Some((source, source, source, 1.0, 0.0));
        }

        let atr = atr_raw * self.factor;
        let next_avg = if (source - self.avg).abs() > atr {
            0.5 * (source + self.avg)
        } else {
            self.avg + self.os * (self.hold / self.factor / self.slope)
        };
        let next_os = pine_sign(next_avg - self.avg);
        let changed = if next_os != self.os { 1.0 } else { 0.0 };
        let next_hold = if changed != 0.0 { atr } else { self.hold };
        let upper = next_avg + self.width_ratio * next_hold;
        let lower = next_avg - self.width_ratio * next_hold;

        self.avg = next_avg;
        self.hold = next_hold;
        self.os = next_os;

        Some((upper, next_avg, lower, next_os, changed))
    }
}

#[derive(Clone, Debug)]
pub struct HyperTrendBatchRange {
    pub factor: (f64, f64, f64),
    pub slope: (f64, f64, f64),
    pub width_percent: (f64, f64, f64),
}

impl Default for HyperTrendBatchRange {
    fn default() -> Self {
        Self {
            factor: (DEFAULT_FACTOR, DEFAULT_FACTOR, 0.0),
            slope: (DEFAULT_SLOPE, DEFAULT_SLOPE, 0.0),
            width_percent: (DEFAULT_WIDTH_PERCENT, DEFAULT_WIDTH_PERCENT, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HyperTrendBatchOutput {
    pub upper: Vec<f64>,
    pub average: Vec<f64>,
    pub lower: Vec<f64>,
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<HyperTrendParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct HyperTrendBatchBuilder {
    range: HyperTrendBatchRange,
    kernel: Kernel,
}

impl Default for HyperTrendBatchBuilder {
    fn default() -> Self {
        Self {
            range: HyperTrendBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl HyperTrendBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn factor_range(mut self, range: (f64, f64, f64)) -> Self {
        self.range.factor = range;
        self
    }

    #[inline]
    pub fn slope_range(mut self, range: (f64, f64, f64)) -> Self {
        self.range.slope = range;
        self
    }

    #[inline]
    pub fn width_percent_range(mut self, range: (f64, f64, f64)) -> Self {
        self.range.width_percent = range;
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<HyperTrendBatchOutput, HyperTrendError> {
        hypertrend_batch_with_kernel(high, low, source, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<HyperTrendBatchOutput, HyperTrendError> {
        let source = source_type(candles, "close");
        hypertrend_batch_with_kernel(
            &candles.high,
            &candles.low,
            source,
            &self.range,
            self.kernel,
        )
    }
}

pub fn expand_grid_hypertrend(
    range: &HyperTrendBatchRange,
) -> Result<Vec<HyperTrendParams>, HyperTrendError> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, HyperTrendError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(HyperTrendError::InvalidFloatRange { start, end, step });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let step = step.abs();
        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += step;
            }
        } else {
            let mut x = start;
            while x + 1e-12 >= end {
                out.push(x);
                x -= step;
            }
        }

        if out.is_empty() {
            return Err(HyperTrendError::InvalidFloatRange { start, end, step });
        }
        Ok(out)
    }

    let factors = axis_f64(range.factor)?;
    let slopes = axis_f64(range.slope)?;
    let widths = axis_f64(range.width_percent)?;

    let cap = factors
        .len()
        .checked_mul(slopes.len())
        .and_then(|value| value.checked_mul(widths.len()))
        .ok_or(HyperTrendError::InvalidRange {
            start: range.factor.0.to_string(),
            end: range.factor.1.to_string(),
            step: range.factor.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &factor in &factors {
        for &slope in &slopes {
            for &width_percent in &widths {
                out.push(HyperTrendParams {
                    factor: Some(factor),
                    slope: Some(slope),
                    width_percent: Some(width_percent),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn hypertrend_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &HyperTrendBatchRange,
    kernel: Kernel,
) -> Result<HyperTrendBatchOutput, HyperTrendError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HyperTrendError::InvalidKernelForBatch(other)),
    };
    hypertrend_batch_par_slice(high, low, source, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn hypertrend_batch_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &HyperTrendBatchRange,
    kernel: Kernel,
) -> Result<HyperTrendBatchOutput, HyperTrendError> {
    hypertrend_batch_inner(high, low, source, sweep, kernel, false)
}

#[inline]
pub fn hypertrend_batch_par_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &HyperTrendBatchRange,
    kernel: Kernel,
) -> Result<HyperTrendBatchOutput, HyperTrendError> {
    hypertrend_batch_inner(high, low, source, sweep, kernel, true)
}

fn hypertrend_batch_inner(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &HyperTrendBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<HyperTrendBatchOutput, HyperTrendError> {
    validate_lengths(high, low, source)?;
    let combos = expand_grid_hypertrend(sweep)?;
    for params in &combos {
        validate_params(
            params.factor.unwrap_or(DEFAULT_FACTOR),
            params.slope.unwrap_or(DEFAULT_SLOPE),
            params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT),
        )?;
    }

    let first_valid = first_valid_bar(high, low, source).ok_or(HyperTrendError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HyperTrendError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;
    let atr_values = compute_atr_zeroed(high, low, source);

    let mut upper_matrix = make_uninit_matrix(rows, cols);
    let mut average_matrix = make_uninit_matrix(rows, cols);
    let mut lower_matrix = make_uninit_matrix(rows, cols);
    let mut trend_matrix = make_uninit_matrix(rows, cols);
    let mut changed_matrix = make_uninit_matrix(rows, cols);
    let warmups = vec![first_valid; rows];
    init_matrix_prefixes(&mut upper_matrix, cols, &warmups);
    init_matrix_prefixes(&mut average_matrix, cols, &warmups);
    init_matrix_prefixes(&mut lower_matrix, cols, &warmups);
    init_matrix_prefixes(&mut trend_matrix, cols, &warmups);
    init_matrix_prefixes(&mut changed_matrix, cols, &warmups);

    let mut upper_guard = ManuallyDrop::new(upper_matrix);
    let mut average_guard = ManuallyDrop::new(average_matrix);
    let mut lower_guard = ManuallyDrop::new(lower_matrix);
    let mut trend_guard = ManuallyDrop::new(trend_matrix);
    let mut changed_guard = ManuallyDrop::new(changed_matrix);

    let upper_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(upper_guard.as_mut_ptr(), upper_guard.len()) };
    let average_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(average_guard.as_mut_ptr(), average_guard.len()) };
    let lower_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(lower_guard.as_mut_ptr(), lower_guard.len()) };
    let trend_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(trend_guard.as_mut_ptr(), trend_guard.len()) };
    let changed_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(changed_guard.as_mut_ptr(), changed_guard.len()) };

    let do_row = |row: usize,
                  row_upper: &mut [MaybeUninit<f64>],
                  row_average: &mut [MaybeUninit<f64>],
                  row_lower: &mut [MaybeUninit<f64>],
                  row_trend: &mut [MaybeUninit<f64>],
                  row_changed: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let dst_upper =
            unsafe { std::slice::from_raw_parts_mut(row_upper.as_mut_ptr() as *mut f64, cols) };
        let dst_average =
            unsafe { std::slice::from_raw_parts_mut(row_average.as_mut_ptr() as *mut f64, cols) };
        let dst_lower =
            unsafe { std::slice::from_raw_parts_mut(row_lower.as_mut_ptr() as *mut f64, cols) };
        let dst_trend =
            unsafe { std::slice::from_raw_parts_mut(row_trend.as_mut_ptr() as *mut f64, cols) };
        let dst_changed =
            unsafe { std::slice::from_raw_parts_mut(row_changed.as_mut_ptr() as *mut f64, cols) };

        hypertrend_row_scalar(
            high,
            low,
            source,
            params.factor.unwrap_or(DEFAULT_FACTOR),
            params.slope.unwrap_or(DEFAULT_SLOPE),
            params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT) * 0.01,
            &atr_values,
            dst_upper,
            dst_average,
            dst_lower,
            dst_trend,
            dst_changed,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        upper_mu
            .par_chunks_mut(cols)
            .zip(average_mu.par_chunks_mut(cols))
            .zip(lower_mu.par_chunks_mut(cols))
            .zip(trend_mu.par_chunks_mut(cols))
            .zip(changed_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, ((((row_upper, row_average), row_lower), row_trend), row_changed))| {
                    do_row(
                        row,
                        row_upper,
                        row_average,
                        row_lower,
                        row_trend,
                        row_changed,
                    )
                },
            );

        #[cfg(target_arch = "wasm32")]
        for (row, ((((row_upper, row_average), row_lower), row_trend), row_changed)) in upper_mu
            .chunks_mut(cols)
            .zip(average_mu.chunks_mut(cols))
            .zip(lower_mu.chunks_mut(cols))
            .zip(trend_mu.chunks_mut(cols))
            .zip(changed_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(
                row,
                row_upper,
                row_average,
                row_lower,
                row_trend,
                row_changed,
            );
        }
    } else {
        for (row, ((((row_upper, row_average), row_lower), row_trend), row_changed)) in upper_mu
            .chunks_mut(cols)
            .zip(average_mu.chunks_mut(cols))
            .zip(lower_mu.chunks_mut(cols))
            .zip(trend_mu.chunks_mut(cols))
            .zip(changed_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(
                row,
                row_upper,
                row_average,
                row_lower,
                row_trend,
                row_changed,
            );
        }
    }

    let upper = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            total,
            upper_guard.capacity(),
        )
    };
    let average = unsafe {
        Vec::from_raw_parts(
            average_guard.as_mut_ptr() as *mut f64,
            total,
            average_guard.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            total,
            lower_guard.capacity(),
        )
    };
    let trend = unsafe {
        Vec::from_raw_parts(
            trend_guard.as_mut_ptr() as *mut f64,
            total,
            trend_guard.capacity(),
        )
    };
    let changed = unsafe {
        Vec::from_raw_parts(
            changed_guard.as_mut_ptr() as *mut f64,
            total,
            changed_guard.capacity(),
        )
    };

    Ok(HyperTrendBatchOutput {
        upper,
        average,
        lower,
        trend,
        changed,
        combos,
        rows,
        cols,
    })
}

fn hypertrend_batch_inner_into(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &HyperTrendBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_upper: &mut [f64],
    out_average: &mut [f64],
    out_lower: &mut [f64],
    out_trend: &mut [f64],
    out_changed: &mut [f64],
) -> Result<Vec<HyperTrendParams>, HyperTrendError> {
    validate_lengths(high, low, source)?;
    let combos = expand_grid_hypertrend(sweep)?;
    for params in &combos {
        validate_params(
            params.factor.unwrap_or(DEFAULT_FACTOR),
            params.slope.unwrap_or(DEFAULT_SLOPE),
            params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT),
        )?;
    }

    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HyperTrendError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;
    if out_upper.len() != total
        || out_average.len() != total
        || out_lower.len() != total
        || out_trend.len() != total
        || out_changed.len() != total
    {
        return Err(HyperTrendError::OutputLengthMismatch {
            expected: total,
            got: out_upper
                .len()
                .max(out_average.len())
                .max(out_lower.len())
                .max(out_trend.len())
                .max(out_changed.len()),
        });
    }

    let _kernel = kernel;
    let atr_values = compute_atr_zeroed(high, low, source);
    let do_row = |row: usize,
                  dst_upper: &mut [f64],
                  dst_average: &mut [f64],
                  dst_lower: &mut [f64],
                  dst_trend: &mut [f64],
                  dst_changed: &mut [f64]| {
        let params = &combos[row];
        hypertrend_row_scalar(
            high,
            low,
            source,
            params.factor.unwrap_or(DEFAULT_FACTOR),
            params.slope.unwrap_or(DEFAULT_SLOPE),
            params.width_percent.unwrap_or(DEFAULT_WIDTH_PERCENT) * 0.01,
            &atr_values,
            dst_upper,
            dst_average,
            dst_lower,
            dst_trend,
            dst_changed,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_upper
            .par_chunks_mut(cols)
            .zip(out_average.par_chunks_mut(cols))
            .zip(out_lower.par_chunks_mut(cols))
            .zip(out_trend.par_chunks_mut(cols))
            .zip(out_changed.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, ((((dst_upper, dst_average), dst_lower), dst_trend), dst_changed))| {
                    do_row(
                        row,
                        dst_upper,
                        dst_average,
                        dst_lower,
                        dst_trend,
                        dst_changed,
                    )
                },
            );

        #[cfg(target_arch = "wasm32")]
        for (row, ((((dst_upper, dst_average), dst_lower), dst_trend), dst_changed)) in out_upper
            .chunks_mut(cols)
            .zip(out_average.chunks_mut(cols))
            .zip(out_lower.chunks_mut(cols))
            .zip(out_trend.chunks_mut(cols))
            .zip(out_changed.chunks_mut(cols))
            .enumerate()
        {
            do_row(
                row,
                dst_upper,
                dst_average,
                dst_lower,
                dst_trend,
                dst_changed,
            );
        }
    } else {
        for (row, ((((dst_upper, dst_average), dst_lower), dst_trend), dst_changed)) in out_upper
            .chunks_mut(cols)
            .zip(out_average.chunks_mut(cols))
            .zip(out_lower.chunks_mut(cols))
            .zip(out_trend.chunks_mut(cols))
            .zip(out_changed.chunks_mut(cols))
            .enumerate()
        {
            do_row(
                row,
                dst_upper,
                dst_average,
                dst_lower,
                dst_trend,
                dst_changed,
            );
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

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let lhs = a[i];
            let rhs = b[i];
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {i}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= tol,
                    "mismatch at {i}: {lhs} vs {rhs} with tol {tol}"
                );
            }
        }
    }

    fn sample_hls(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut source = Vec::with_capacity(len);

        for i in 0..len {
            let base = 100.0 + i as f64 * 0.17 + (i as f64 * 0.031).sin() * 2.75;
            let spread = 1.0 + (i as f64 * 0.07).cos().abs() * 1.8;
            let src = base + (i as f64 * 0.11).sin() * 0.8;
            high.push(base + spread);
            low.push(base - spread);
            source.push(src);
        }

        (high, low, source)
    }

    fn check_output_contract(kernel: Kernel) {
        let (high, low, source) = sample_hls(320);
        let input = HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(5.0),
                slope: Some(14.0),
                width_percent: Some(80.0),
            },
        );
        let out = hypertrend_with_kernel(&input, kernel).expect("indicator");
        assert_eq!(out.upper.len(), source.len());
        assert_eq!(out.average.len(), source.len());
        assert_eq!(out.lower.len(), source.len());
        assert_eq!(out.trend.len(), source.len());
        assert_eq!(out.changed.len(), source.len());
        assert!(out.average.iter().any(|v| v.is_finite()));
        assert!(
            out.upper
                .iter()
                .zip(&out.lower)
                .all(|(u, l)| (u.is_nan() && l.is_nan()) || (*u >= *l))
        );
    }

    fn check_into_matches_api(kernel: Kernel) {
        let (high, low, source) = sample_hls(240);
        let input = HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(4.5),
                slope: Some(10.0),
                width_percent: Some(60.0),
            },
        );
        let baseline = hypertrend_with_kernel(&input, kernel).expect("baseline");
        let mut upper = vec![0.0; source.len()];
        let mut average = vec![0.0; source.len()];
        let mut lower = vec![0.0; source.len()];
        let mut trend = vec![0.0; source.len()];
        let mut changed = vec![0.0; source.len()];
        hypertrend_into_slice(
            &mut upper,
            &mut average,
            &mut lower,
            &mut trend,
            &mut changed,
            &input,
            kernel,
        )
        .expect("into");

        assert_close(&baseline.upper, &upper, 1e-12);
        assert_close(&baseline.average, &average, 1e-12);
        assert_close(&baseline.lower, &lower, 1e-12);
        assert_close(&baseline.trend, &trend, 1e-12);
        assert_close(&baseline.changed, &changed, 1e-12);
    }

    fn check_stream_matches_batch() {
        let (high, low, source) = sample_hls(260);
        let input = HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(4.0),
                slope: Some(12.0),
                width_percent: Some(55.0),
            },
        );
        let batch = hypertrend(&input).expect("batch");
        let mut stream = HyperTrendStream::try_new(HyperTrendParams {
            factor: Some(4.0),
            slope: Some(12.0),
            width_percent: Some(55.0),
        })
        .expect("stream");

        let mut upper = vec![f64::NAN; source.len()];
        let mut average = vec![f64::NAN; source.len()];
        let mut lower = vec![f64::NAN; source.len()];
        let mut trend = vec![f64::NAN; source.len()];
        let mut changed = vec![f64::NAN; source.len()];

        for i in 0..source.len() {
            if let Some((u, a, l, t, c)) = stream.update(high[i], low[i], source[i]) {
                upper[i] = u;
                average[i] = a;
                lower[i] = l;
                trend[i] = t;
                changed[i] = c;
            }
        }

        assert_close(&batch.upper, &upper, 1e-12);
        assert_close(&batch.average, &average, 1e-12);
        assert_close(&batch.lower, &lower, 1e-12);
        assert_close(&batch.trend, &trend, 1e-12);
        assert_close(&batch.changed, &changed, 1e-12);
    }

    fn check_batch_single_matches_single(kernel: Kernel) {
        let (high, low, source) = sample_hls(180);
        let batch = hypertrend_batch_with_kernel(
            &high,
            &low,
            &source,
            &HyperTrendBatchRange {
                factor: (5.0, 5.0, 0.0),
                slope: (14.0, 14.0, 0.0),
                width_percent: (80.0, 80.0, 0.0),
            },
            kernel,
        )
        .expect("batch");
        let single = hypertrend(&HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(5.0),
                slope: Some(14.0),
                width_percent: Some(80.0),
            },
        ))
        .expect("single");

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        assert_close(&batch.upper[..source.len()], &single.upper, 1e-12);
        assert_close(&batch.average[..source.len()], &single.average, 1e-12);
        assert_close(&batch.lower[..source.len()], &single.lower, 1e-12);
        assert_close(&batch.trend[..source.len()], &single.trend, 1e-12);
        assert_close(&batch.changed[..source.len()], &single.changed, 1e-12);
    }

    #[test]
    fn hypertrend_invalid_params() {
        let (high, low, source) = sample_hls(64);

        let err = hypertrend(&HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(0.0),
                slope: Some(14.0),
                width_percent: Some(80.0),
            },
        ))
        .expect_err("invalid factor");
        assert!(matches!(err, HyperTrendError::InvalidFactor { .. }));

        let err = hypertrend(&HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(5.0),
                slope: Some(0.0),
                width_percent: Some(80.0),
            },
        ))
        .expect_err("invalid slope");
        assert!(matches!(err, HyperTrendError::InvalidSlope { .. }));

        let err = hypertrend(&HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(5.0),
                slope: Some(14.0),
                width_percent: Some(120.0),
            },
        ))
        .expect_err("invalid width");
        assert!(matches!(err, HyperTrendError::InvalidWidthPercent { .. }));
    }

    #[test]
    fn hypertrend_output_contract() {
        check_output_contract(Kernel::Auto);
        check_output_contract(Kernel::Scalar);
    }

    #[test]
    fn hypertrend_into_matches_api() {
        check_into_matches_api(Kernel::Auto);
        check_into_matches_api(Kernel::Scalar);
    }

    #[test]
    fn hypertrend_stream_matches_batch() {
        check_stream_matches_batch();
    }

    #[test]
    fn hypertrend_batch_single_matches_single() {
        check_batch_single_matches_single(Kernel::Auto);
    }

    #[test]
    fn hypertrend_dispatch_matches_direct() {
        let (high, low, source) = sample_hls(160);
        let combo = [
            ParamKV {
                key: "factor",
                value: ParamValue::Float(5.0),
            },
            ParamKV {
                key: "slope",
                value: ParamValue::Float(14.0),
            },
            ParamKV {
                key: "width_percent",
                value: ParamValue::Float(80.0),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "hypertrend",
            output_id: Some("average"),
            data: IndicatorDataRef::Ohlc {
                open: &source,
                high: &high,
                low: &low,
                close: &source,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };

        let batch = compute_cpu_batch(req).expect("dispatch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());

        let direct = hypertrend(&HyperTrendInput::from_slices(
            &high,
            &low,
            &source,
            HyperTrendParams {
                factor: Some(5.0),
                slope: Some(14.0),
                width_percent: Some(80.0),
            },
        ))
        .expect("direct");
        let row = &batch.values_f64.as_ref().expect("f64 output")[0..source.len()];
        assert_close(row, &direct.average, 1e-12);
    }
}
