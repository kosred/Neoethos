use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "hlcc4";
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_LENGTH1: usize = 2;
const DEFAULT_LENGTH2: usize = 5;
const DEFAULT_LENGTH3: usize = 9;
const DEFAULT_LENGTH4: usize = 13;

#[derive(Debug, Clone)]
pub enum RelativeStrengthIndexWaveIndicatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        source: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct RelativeStrengthIndexWaveIndicatorOutput {
    pub rsi_ma1: Vec<f64>,
    pub rsi_ma2: Vec<f64>,
    pub rsi_ma3: Vec<f64>,
    pub rsi_ma4: Vec<f64>,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RelativeStrengthIndexWaveIndicatorParams {
    pub rsi_length: Option<usize>,
    pub length1: Option<usize>,
    pub length2: Option<usize>,
    pub length3: Option<usize>,
    pub length4: Option<usize>,
}

impl Default for RelativeStrengthIndexWaveIndicatorParams {
    fn default() -> Self {
        Self {
            rsi_length: Some(DEFAULT_RSI_LENGTH),
            length1: Some(DEFAULT_LENGTH1),
            length2: Some(DEFAULT_LENGTH2),
            length3: Some(DEFAULT_LENGTH3),
            length4: Some(DEFAULT_LENGTH4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelativeStrengthIndexWaveIndicatorInput<'a> {
    pub data: RelativeStrengthIndexWaveIndicatorData<'a>,
    pub params: RelativeStrengthIndexWaveIndicatorParams,
}

impl<'a> RelativeStrengthIndexWaveIndicatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: RelativeStrengthIndexWaveIndicatorParams,
    ) -> Self {
        Self {
            data: RelativeStrengthIndexWaveIndicatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        params: RelativeStrengthIndexWaveIndicatorParams,
    ) -> Self {
        Self {
            data: RelativeStrengthIndexWaveIndicatorData::Slices { source, high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            RelativeStrengthIndexWaveIndicatorParams::default(),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RelativeStrengthIndexWaveIndicatorBuilder {
    source: Option<&'static str>,
    rsi_length: Option<usize>,
    length1: Option<usize>,
    length2: Option<usize>,
    length3: Option<usize>,
    length4: Option<usize>,
    kernel: Kernel,
}

impl Default for RelativeStrengthIndexWaveIndicatorBuilder {
    fn default() -> Self {
        Self {
            source: None,
            rsi_length: None,
            length1: None,
            length2: None,
            length3: None,
            length4: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RelativeStrengthIndexWaveIndicatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn rsi_length(mut self, value: usize) -> Self {
        self.rsi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn length1(mut self, value: usize) -> Self {
        self.length1 = Some(value);
        self
    }

    #[inline(always)]
    pub fn length2(mut self, value: usize) -> Self {
        self.length2 = Some(value);
        self
    }

    #[inline(always)]
    pub fn length3(mut self, value: usize) -> Self {
        self.length3 = Some(value);
        self
    }

    #[inline(always)]
    pub fn length4(mut self, value: usize) -> Self {
        self.length4 = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<RelativeStrengthIndexWaveIndicatorOutput, RelativeStrengthIndexWaveIndicatorError>
    {
        let input = RelativeStrengthIndexWaveIndicatorInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            RelativeStrengthIndexWaveIndicatorParams {
                rsi_length: self.rsi_length,
                length1: self.length1,
                length2: self.length2,
                length3: self.length3,
                length4: self.length4,
            },
        );
        relative_strength_index_wave_indicator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        high: &[f64],
        low: &[f64],
    ) -> Result<RelativeStrengthIndexWaveIndicatorOutput, RelativeStrengthIndexWaveIndicatorError>
    {
        let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
            source,
            high,
            low,
            RelativeStrengthIndexWaveIndicatorParams {
                rsi_length: self.rsi_length,
                length1: self.length1,
                length2: self.length2,
                length3: self.length3,
                length4: self.length4,
            },
        );
        relative_strength_index_wave_indicator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<RelativeStrengthIndexWaveIndicatorStream, RelativeStrengthIndexWaveIndicatorError>
    {
        RelativeStrengthIndexWaveIndicatorStream::try_new(
            RelativeStrengthIndexWaveIndicatorParams {
                rsi_length: self.rsi_length,
                length1: self.length1,
                length2: self.length2,
                length3: self.length3,
                length4: self.length4,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum RelativeStrengthIndexWaveIndicatorError {
    #[error("relative_strength_index_wave_indicator: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "relative_strength_index_wave_indicator: Input slices must have the same length: source={source_len}, high={high_len}, low={low_len}"
    )]
    LengthMismatch {
        source_len: usize,
        high_len: usize,
        low_len: usize,
    },
    #[error("relative_strength_index_wave_indicator: All values are NaN.")]
    AllValuesNaN,
    #[error("relative_strength_index_wave_indicator: Invalid rsi_length: {rsi_length}")]
    InvalidRsiLength { rsi_length: usize },
    #[error("relative_strength_index_wave_indicator: Invalid length1: {length1}")]
    InvalidLength1 { length1: usize },
    #[error("relative_strength_index_wave_indicator: Invalid length2: {length2}")]
    InvalidLength2 { length2: usize },
    #[error("relative_strength_index_wave_indicator: Invalid length3: {length3}")]
    InvalidLength3 { length3: usize },
    #[error("relative_strength_index_wave_indicator: Invalid length4: {length4}")]
    InvalidLength4 { length4: usize },
    #[error(
        "relative_strength_index_wave_indicator: Output length mismatch: expected={expected}, got={got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "relative_strength_index_wave_indicator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("relative_strength_index_wave_indicator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    rsi_length: usize,
    length1: usize,
    length2: usize,
    length3: usize,
    length4: usize,
}

#[inline(always)]
fn first_valid_triplet(source: &[f64], high: &[f64], low: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| source[i].is_finite() && high[i].is_finite() && low[i].is_finite())
}

#[inline(always)]
fn validate_raw_slices(
    source: &[f64],
    high: &[f64],
    low: &[f64],
) -> Result<usize, RelativeStrengthIndexWaveIndicatorError> {
    if source.is_empty() || high.is_empty() || low.is_empty() {
        return Err(RelativeStrengthIndexWaveIndicatorError::EmptyInputData);
    }
    if source.len() != high.len() || source.len() != low.len() {
        return Err(RelativeStrengthIndexWaveIndicatorError::LengthMismatch {
            source_len: source.len(),
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    first_valid_triplet(source, high, low)
        .ok_or(RelativeStrengthIndexWaveIndicatorError::AllValuesNaN)
}

#[inline(always)]
fn resolve_params(
    params: &RelativeStrengthIndexWaveIndicatorParams,
) -> Result<ResolvedParams, RelativeStrengthIndexWaveIndicatorError> {
    let rsi_length = params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
    if rsi_length == 0 {
        return Err(RelativeStrengthIndexWaveIndicatorError::InvalidRsiLength { rsi_length });
    }
    let length1 = params.length1.unwrap_or(DEFAULT_LENGTH1);
    if length1 == 0 {
        return Err(RelativeStrengthIndexWaveIndicatorError::InvalidLength1 { length1 });
    }
    let length2 = params.length2.unwrap_or(DEFAULT_LENGTH2);
    if length2 == 0 {
        return Err(RelativeStrengthIndexWaveIndicatorError::InvalidLength2 { length2 });
    }
    let length3 = params.length3.unwrap_or(DEFAULT_LENGTH3);
    if length3 == 0 {
        return Err(RelativeStrengthIndexWaveIndicatorError::InvalidLength3 { length3 });
    }
    let length4 = params.length4.unwrap_or(DEFAULT_LENGTH4);
    if length4 == 0 {
        return Err(RelativeStrengthIndexWaveIndicatorError::InvalidLength4 { length4 });
    }
    Ok(ResolvedParams {
        rsi_length,
        length1,
        length2,
        length3,
        length4,
    })
}

#[inline(always)]
fn extract_slices<'a>(
    input: &'a RelativeStrengthIndexWaveIndicatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), RelativeStrengthIndexWaveIndicatorError> {
    let (source, high, low) = match &input.data {
        RelativeStrengthIndexWaveIndicatorData::Candles { candles, source } => (
            source_type(candles, source),
            candles.high.as_slice(),
            candles.low.as_slice(),
        ),
        RelativeStrengthIndexWaveIndicatorData::Slices { source, high, low } => {
            (*source, *high, *low)
        }
    };
    validate_raw_slices(source, high, low)?;
    Ok((source, high, low))
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a RelativeStrengthIndexWaveIndicatorInput<'a>,
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
    RelativeStrengthIndexWaveIndicatorError,
> {
    let (source, high, low) = extract_slices(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid_triplet(source, high, low)
        .ok_or(RelativeStrengthIndexWaveIndicatorError::AllValuesNaN)?;
    Ok((source, high, low, params, first, kernel.to_non_batch()))
}

#[derive(Clone, Debug)]
struct RsiCore {
    period: usize,
    inv_p: f64,
    beta: f64,
    initialized: bool,
    prev: f64,
    deltas_seen: usize,
    sum_gain: f64,
    sum_loss: f64,
    avg_gain: f64,
    avg_loss: f64,
    ready: bool,
}

impl RsiCore {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let inv_p = 1.0 / period as f64;
        Self {
            period,
            inv_p,
            beta: 1.0 - inv_p,
            initialized: false,
            prev: f64::NAN,
            deltas_seen: 0,
            sum_gain: 0.0,
            sum_loss: 0.0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            ready: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.initialized = false;
        self.prev = f64::NAN;
        self.deltas_seen = 0;
        self.sum_gain = 0.0;
        self.sum_loss = 0.0;
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.ready = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        if !value.is_finite() {
            self.reset();
            return f64::NAN;
        }
        if !self.initialized {
            self.prev = value;
            self.initialized = true;
            return f64::NAN;
        }
        let delta = value - self.prev;
        self.prev = value;
        let gain = delta.max(0.0);
        let loss = (-delta).max(0.0);

        if !self.ready {
            self.sum_gain += gain;
            self.sum_loss += loss;
            self.deltas_seen += 1;
            if self.deltas_seen < self.period {
                return f64::NAN;
            }
            self.avg_gain = self.sum_gain * self.inv_p;
            self.avg_loss = self.sum_loss * self.inv_p;
            self.ready = true;
        } else {
            self.avg_gain = self.avg_gain.mul_add(self.beta, self.inv_p * gain);
            self.avg_loss = self.avg_loss.mul_add(self.beta, self.inv_p * loss);
        }

        let denom = self.avg_gain + self.avg_loss;
        if denom == 0.0 {
            50.0
        } else {
            100.0 * self.avg_gain / denom
        }
    }
}

#[derive(Clone, Debug)]
struct WmaCore {
    len: usize,
    denom: f64,
    buf: Vec<f64>,
    pos: usize,
    count: usize,
    sum: f64,
    weighted_sum: f64,
}

impl WmaCore {
    #[inline(always)]
    fn new(len: usize) -> Self {
        Self {
            len,
            denom: (len * (len + 1) / 2) as f64,
            buf: vec![0.0; len],
            pos: 0,
            count: 0,
            sum: 0.0,
            weighted_sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.pos = 0;
        self.count = 0;
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        if !value.is_finite() {
            self.reset();
            return f64::NAN;
        }

        if self.count < self.len {
            self.buf[self.count] = value;
            self.count += 1;
            self.sum += value;
            self.weighted_sum += self.count as f64 * value;
            if self.count == self.len {
                return self.weighted_sum / self.denom;
            }
            return f64::NAN;
        }

        let old_sum = self.sum;
        let old = self.buf[self.pos];
        self.buf[self.pos] = value;
        self.pos += 1;
        if self.pos == self.len {
            self.pos = 0;
        }
        self.weighted_sum = self.weighted_sum + self.len as f64 * value - old_sum;
        self.sum = old_sum + value - old;
        self.weighted_sum / self.denom
    }
}

#[derive(Clone, Debug)]
struct RelativeStrengthIndexWaveCore {
    rsi_source: RsiCore,
    rsi_high: RsiCore,
    rsi_low: RsiCore,
    wma1: WmaCore,
    wma2: WmaCore,
    wma3: WmaCore,
    wma4: WmaCore,
    prev_slo: Option<f64>,
}

impl RelativeStrengthIndexWaveCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            rsi_source: RsiCore::new(params.rsi_length),
            rsi_high: RsiCore::new(params.rsi_length),
            rsi_low: RsiCore::new(params.rsi_length),
            wma1: WmaCore::new(params.length1),
            wma2: WmaCore::new(params.length2),
            wma3: WmaCore::new(params.length3),
            wma4: WmaCore::new(params.length4),
            prev_slo: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.rsi_source.reset();
        self.rsi_high.reset();
        self.rsi_low.reset();
        self.wma1.reset();
        self.wma2.reset();
        self.wma3.reset();
        self.wma4.reset();
        self.prev_slo = None;
    }

    #[inline(always)]
    fn update(&mut self, source: f64, high: f64, low: f64) -> (f64, f64, f64, f64, f64) {
        if !source.is_finite() || !high.is_finite() || !low.is_finite() {
            self.reset();
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }

        let custom_rsi = self.rsi_source.update(source);
        let high_rsi = self.rsi_high.update(high);
        let low_rsi = self.rsi_low.update(low);
        if !custom_rsi.is_finite() || !high_rsi.is_finite() || !low_rsi.is_finite() {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }

        let hlc_rsi = (high_rsi + low_rsi + 2.0 * custom_rsi) * 0.25;
        let rsi_ma1 = self.wma1.update(hlc_rsi);
        let rsi_ma2 = self.wma2.update(hlc_rsi);
        let rsi_ma3 = self.wma3.update(hlc_rsi);
        let rsi_ma4 = self.wma4.update(hlc_rsi);

        let state = if rsi_ma1.is_finite() && rsi_ma2.is_finite() {
            let slo = rsi_ma1 - rsi_ma2;
            let prev = self.prev_slo.unwrap_or(0.0);
            self.prev_slo = Some(slo);
            if slo > 0.0 {
                if slo > prev { 2.0 } else { 1.0 }
            } else if slo < 0.0 {
                if slo < prev { -2.0 } else { -1.0 }
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        (rsi_ma1, rsi_ma2, rsi_ma3, rsi_ma4, state)
    }
}

#[inline(always)]
fn compute_relative_strength_index_wave_indicator_into(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    params: ResolvedParams,
    out_rsi_ma1: &mut [f64],
    out_rsi_ma2: &mut [f64],
    out_rsi_ma3: &mut [f64],
    out_rsi_ma4: &mut [f64],
    out_state: &mut [f64],
) -> Result<(), RelativeStrengthIndexWaveIndicatorError> {
    let len = source.len();
    if out_rsi_ma1.len() != len {
        return Err(
            RelativeStrengthIndexWaveIndicatorError::OutputLengthMismatch {
                expected: len,
                got: out_rsi_ma1.len(),
            },
        );
    }
    if out_rsi_ma2.len() != len {
        return Err(
            RelativeStrengthIndexWaveIndicatorError::OutputLengthMismatch {
                expected: len,
                got: out_rsi_ma2.len(),
            },
        );
    }
    if out_rsi_ma3.len() != len {
        return Err(
            RelativeStrengthIndexWaveIndicatorError::OutputLengthMismatch {
                expected: len,
                got: out_rsi_ma3.len(),
            },
        );
    }
    if out_rsi_ma4.len() != len {
        return Err(
            RelativeStrengthIndexWaveIndicatorError::OutputLengthMismatch {
                expected: len,
                got: out_rsi_ma4.len(),
            },
        );
    }
    if out_state.len() != len {
        return Err(
            RelativeStrengthIndexWaveIndicatorError::OutputLengthMismatch {
                expected: len,
                got: out_state.len(),
            },
        );
    }

    let mut core = RelativeStrengthIndexWaveCore::new(params);
    for i in 0..len {
        let (a, b, c, d, state) = core.update(source[i], high[i], low[i]);
        out_rsi_ma1[i] = a;
        out_rsi_ma2[i] = b;
        out_rsi_ma3[i] = c;
        out_rsi_ma4[i] = d;
        out_state[i] = state;
    }
    Ok(())
}

#[inline]
pub fn relative_strength_index_wave_indicator(
    input: &RelativeStrengthIndexWaveIndicatorInput,
) -> Result<RelativeStrengthIndexWaveIndicatorOutput, RelativeStrengthIndexWaveIndicatorError> {
    relative_strength_index_wave_indicator_with_kernel(input, Kernel::Auto)
}

pub fn relative_strength_index_wave_indicator_with_kernel(
    input: &RelativeStrengthIndexWaveIndicatorInput,
    kernel: Kernel,
) -> Result<RelativeStrengthIndexWaveIndicatorOutput, RelativeStrengthIndexWaveIndicatorError> {
    let (source, high, low, params, first, _kernel) = validate_input(input, kernel)?;
    let len = source.len();
    let mut rsi_ma1 = alloc_with_nan_prefix(len, first);
    let mut rsi_ma2 = alloc_with_nan_prefix(len, first);
    let mut rsi_ma3 = alloc_with_nan_prefix(len, first);
    let mut rsi_ma4 = alloc_with_nan_prefix(len, first);
    let mut state = alloc_with_nan_prefix(len, first);
    compute_relative_strength_index_wave_indicator_into(
        source,
        high,
        low,
        params,
        &mut rsi_ma1,
        &mut rsi_ma2,
        &mut rsi_ma3,
        &mut rsi_ma4,
        &mut state,
    )?;
    Ok(RelativeStrengthIndexWaveIndicatorOutput {
        rsi_ma1,
        rsi_ma2,
        rsi_ma3,
        rsi_ma4,
        state,
    })
}

pub fn relative_strength_index_wave_indicator_into(
    input: &RelativeStrengthIndexWaveIndicatorInput,
    out_rsi_ma1: &mut [f64],
    out_rsi_ma2: &mut [f64],
    out_rsi_ma3: &mut [f64],
    out_rsi_ma4: &mut [f64],
    out_state: &mut [f64],
) -> Result<(), RelativeStrengthIndexWaveIndicatorError> {
    relative_strength_index_wave_indicator_into_slice(
        out_rsi_ma1,
        out_rsi_ma2,
        out_rsi_ma3,
        out_rsi_ma4,
        out_state,
        input,
        Kernel::Auto,
    )
}

#[inline]
pub fn relative_strength_index_wave_indicator_into_slice(
    out_rsi_ma1: &mut [f64],
    out_rsi_ma2: &mut [f64],
    out_rsi_ma3: &mut [f64],
    out_rsi_ma4: &mut [f64],
    out_state: &mut [f64],
    input: &RelativeStrengthIndexWaveIndicatorInput,
    kernel: Kernel,
) -> Result<(), RelativeStrengthIndexWaveIndicatorError> {
    let (source, high, low, params, _first, _kernel) = validate_input(input, kernel)?;
    compute_relative_strength_index_wave_indicator_into(
        source,
        high,
        low,
        params,
        out_rsi_ma1,
        out_rsi_ma2,
        out_rsi_ma3,
        out_rsi_ma4,
        out_state,
    )
}

#[derive(Debug, Clone)]
pub struct RelativeStrengthIndexWaveIndicatorStream {
    core: RelativeStrengthIndexWaveCore,
}

impl RelativeStrengthIndexWaveIndicatorStream {
    #[inline(always)]
    pub fn try_new(
        params: RelativeStrengthIndexWaveIndicatorParams,
    ) -> Result<Self, RelativeStrengthIndexWaveIndicatorError> {
        let resolved = resolve_params(&params)?;
        Ok(Self {
            core: RelativeStrengthIndexWaveCore::new(resolved),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, source: f64, high: f64, low: f64) -> (f64, f64, f64, f64, f64) {
        self.core.update(source, high, low)
    }
}

#[derive(Debug, Clone)]
pub struct RelativeStrengthIndexWaveIndicatorBatchRange {
    pub rsi_length: (usize, usize, usize),
    pub length1: (usize, usize, usize),
    pub length2: (usize, usize, usize),
    pub length3: (usize, usize, usize),
    pub length4: (usize, usize, usize),
}

impl Default for RelativeStrengthIndexWaveIndicatorBatchRange {
    fn default() -> Self {
        Self {
            rsi_length: (DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
            length1: (DEFAULT_LENGTH1, DEFAULT_LENGTH1, 0),
            length2: (DEFAULT_LENGTH2, DEFAULT_LENGTH2, 0),
            length3: (DEFAULT_LENGTH3, DEFAULT_LENGTH3, 0),
            length4: (DEFAULT_LENGTH4, DEFAULT_LENGTH4, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelativeStrengthIndexWaveIndicatorBatchOutput {
    pub rsi_ma1: Vec<f64>,
    pub rsi_ma2: Vec<f64>,
    pub rsi_ma3: Vec<f64>,
    pub rsi_ma4: Vec<f64>,
    pub state: Vec<f64>,
    pub combos: Vec<RelativeStrengthIndexWaveIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct RelativeStrengthIndexWaveIndicatorBatchBuilder {
    source: Option<&'static str>,
    range: RelativeStrengthIndexWaveIndicatorBatchRange,
    kernel: Kernel,
}

impl Default for RelativeStrengthIndexWaveIndicatorBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: RelativeStrengthIndexWaveIndicatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl RelativeStrengthIndexWaveIndicatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn rsi_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.rsi_length = value;
        self
    }

    #[inline(always)]
    pub fn length1_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length1 = value;
        self
    }

    #[inline(always)]
    pub fn length2_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length2 = value;
        self
    }

    #[inline(always)]
    pub fn length3_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length3 = value;
        self
    }

    #[inline(always)]
    pub fn length4_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length4 = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        RelativeStrengthIndexWaveIndicatorBatchOutput,
        RelativeStrengthIndexWaveIndicatorError,
    > {
        relative_strength_index_wave_indicator_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            candles.high.as_slice(),
            candles.low.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        high: &[f64],
        low: &[f64],
    ) -> Result<
        RelativeStrengthIndexWaveIndicatorBatchOutput,
        RelativeStrengthIndexWaveIndicatorError,
    > {
        relative_strength_index_wave_indicator_batch_with_kernel(
            source,
            high,
            low,
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, RelativeStrengthIndexWaveIndicatorError> {
    if step == 0 {
        if start != end {
            return Err(RelativeStrengthIndexWaveIndicatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end {
        return Err(RelativeStrengthIndexWaveIndicatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end {
        out.push(current);
        current = current.checked_add(step).ok_or_else(|| {
            RelativeStrengthIndexWaveIndicatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            }
        })?;
    }
    Ok(out)
}

pub fn expand_grid(
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
) -> Result<Vec<RelativeStrengthIndexWaveIndicatorParams>, RelativeStrengthIndexWaveIndicatorError>
{
    let rsi_lengths =
        expand_usize_range(sweep.rsi_length.0, sweep.rsi_length.1, sweep.rsi_length.2)?;
    let length1s = expand_usize_range(sweep.length1.0, sweep.length1.1, sweep.length1.2)?;
    let length2s = expand_usize_range(sweep.length2.0, sweep.length2.1, sweep.length2.2)?;
    let length3s = expand_usize_range(sweep.length3.0, sweep.length3.1, sweep.length3.2)?;
    let length4s = expand_usize_range(sweep.length4.0, sweep.length4.1, sweep.length4.2)?;

    let mut combos = Vec::with_capacity(
        rsi_lengths.len() * length1s.len() * length2s.len() * length3s.len() * length4s.len(),
    );
    for rsi_length in rsi_lengths {
        for length1 in length1s.iter().copied() {
            for length2 in length2s.iter().copied() {
                for length3 in length3s.iter().copied() {
                    for length4 in length4s.iter().copied() {
                        combos.push(RelativeStrengthIndexWaveIndicatorParams {
                            rsi_length: Some(rsi_length),
                            length1: Some(length1),
                            length2: Some(length2),
                            length3: Some(length3),
                            length4: Some(length4),
                        });
                    }
                }
            }
        }
    }
    Ok(combos)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, RelativeStrengthIndexWaveIndicatorError> {
    rows.checked_mul(cols)
        .ok_or_else(|| RelativeStrengthIndexWaveIndicatorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn relative_strength_index_wave_indicator_batch_with_kernel(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    kernel: Kernel,
) -> Result<RelativeStrengthIndexWaveIndicatorBatchOutput, RelativeStrengthIndexWaveIndicatorError>
{
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(RelativeStrengthIndexWaveIndicatorError::InvalidKernelForBatch(kernel)),
    };
    relative_strength_index_wave_indicator_batch_par_slice(
        source,
        high,
        low,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn relative_strength_index_wave_indicator_batch_slice(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    kernel: Kernel,
) -> Result<RelativeStrengthIndexWaveIndicatorBatchOutput, RelativeStrengthIndexWaveIndicatorError>
{
    relative_strength_index_wave_indicator_batch_inner(source, high, low, sweep, kernel, false)
}

#[inline(always)]
pub fn relative_strength_index_wave_indicator_batch_par_slice(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    kernel: Kernel,
) -> Result<RelativeStrengthIndexWaveIndicatorBatchOutput, RelativeStrengthIndexWaveIndicatorError>
{
    relative_strength_index_wave_indicator_batch_inner(source, high, low, sweep, kernel, true)
}

fn relative_strength_index_wave_indicator_batch_inner(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<RelativeStrengthIndexWaveIndicatorBatchOutput, RelativeStrengthIndexWaveIndicatorError>
{
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(source, high, low)?;
    let rows = combos.len();
    let cols = source.len();
    let total = batch_shape(rows, cols)?;
    let warmups = vec![first.min(cols); rows];

    let mut rsi_ma1_buf = make_uninit_matrix(rows, cols);
    let mut rsi_ma2_buf = make_uninit_matrix(rows, cols);
    let mut rsi_ma3_buf = make_uninit_matrix(rows, cols);
    let mut rsi_ma4_buf = make_uninit_matrix(rows, cols);
    let mut state_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut rsi_ma1_buf, cols, &warmups);
    init_matrix_prefixes(&mut rsi_ma2_buf, cols, &warmups);
    init_matrix_prefixes(&mut rsi_ma3_buf, cols, &warmups);
    init_matrix_prefixes(&mut rsi_ma4_buf, cols, &warmups);
    init_matrix_prefixes(&mut state_buf, cols, &warmups);

    let mut rsi_ma1_guard = ManuallyDrop::new(rsi_ma1_buf);
    let mut rsi_ma2_guard = ManuallyDrop::new(rsi_ma2_buf);
    let mut rsi_ma3_guard = ManuallyDrop::new(rsi_ma3_buf);
    let mut rsi_ma4_guard = ManuallyDrop::new(rsi_ma4_buf);
    let mut state_guard = ManuallyDrop::new(state_buf);

    let out_rsi_ma1: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(rsi_ma1_guard.as_mut_ptr() as *mut f64, rsi_ma1_guard.len())
    };
    let out_rsi_ma2: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(rsi_ma2_guard.as_mut_ptr() as *mut f64, rsi_ma2_guard.len())
    };
    let out_rsi_ma3: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(rsi_ma3_guard.as_mut_ptr() as *mut f64, rsi_ma3_guard.len())
    };
    let out_rsi_ma4: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(rsi_ma4_guard.as_mut_ptr() as *mut f64, rsi_ma4_guard.len())
    };
    let out_state: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(state_guard.as_mut_ptr() as *mut f64, state_guard.len())
    };

    relative_strength_index_wave_indicator_batch_inner_into(
        source,
        high,
        low,
        sweep,
        kernel,
        parallel,
        out_rsi_ma1,
        out_rsi_ma2,
        out_rsi_ma3,
        out_rsi_ma4,
        out_state,
    )?;

    let rsi_ma1 = unsafe {
        Vec::from_raw_parts(
            rsi_ma1_guard.as_mut_ptr() as *mut f64,
            total,
            rsi_ma1_guard.capacity(),
        )
    };
    let rsi_ma2 = unsafe {
        Vec::from_raw_parts(
            rsi_ma2_guard.as_mut_ptr() as *mut f64,
            total,
            rsi_ma2_guard.capacity(),
        )
    };
    let rsi_ma3 = unsafe {
        Vec::from_raw_parts(
            rsi_ma3_guard.as_mut_ptr() as *mut f64,
            total,
            rsi_ma3_guard.capacity(),
        )
    };
    let rsi_ma4 = unsafe {
        Vec::from_raw_parts(
            rsi_ma4_guard.as_mut_ptr() as *mut f64,
            total,
            rsi_ma4_guard.capacity(),
        )
    };
    let state = unsafe {
        Vec::from_raw_parts(
            state_guard.as_mut_ptr() as *mut f64,
            total,
            state_guard.capacity(),
        )
    };

    Ok(RelativeStrengthIndexWaveIndicatorBatchOutput {
        rsi_ma1,
        rsi_ma2,
        rsi_ma3,
        rsi_ma4,
        state,
        combos,
        rows,
        cols,
    })
}

pub fn relative_strength_index_wave_indicator_batch_into_slice(
    out_rsi_ma1: &mut [f64],
    out_rsi_ma2: &mut [f64],
    out_rsi_ma3: &mut [f64],
    out_rsi_ma4: &mut [f64],
    out_state: &mut [f64],
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    kernel: Kernel,
) -> Result<(), RelativeStrengthIndexWaveIndicatorError> {
    relative_strength_index_wave_indicator_batch_inner_into(
        source,
        high,
        low,
        sweep,
        kernel,
        false,
        out_rsi_ma1,
        out_rsi_ma2,
        out_rsi_ma3,
        out_rsi_ma4,
        out_state,
    )?;
    Ok(())
}

fn relative_strength_index_wave_indicator_batch_inner_into(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &RelativeStrengthIndexWaveIndicatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_rsi_ma1: &mut [f64],
    out_rsi_ma2: &mut [f64],
    out_rsi_ma3: &mut [f64],
    out_rsi_ma4: &mut [f64],
    out_state: &mut [f64],
) -> Result<Vec<RelativeStrengthIndexWaveIndicatorParams>, RelativeStrengthIndexWaveIndicatorError>
{
    let combos = expand_grid(sweep)?;
    let _first = validate_raw_slices(source, high, low)?;
    let rows = combos.len();
    let cols = source.len();
    let total = batch_shape(rows, cols)?;
    for out in [
        out_rsi_ma1.len(),
        out_rsi_ma2.len(),
        out_rsi_ma3.len(),
        out_rsi_ma4.len(),
        out_state.len(),
    ] {
        if out != total {
            return Err(
                RelativeStrengthIndexWaveIndicatorError::OutputLengthMismatch {
                    expected: total,
                    got: out,
                },
            );
        }
    }

    let compute_row = |row: usize,
                       dst1: &mut [f64],
                       dst2: &mut [f64],
                       dst3: &mut [f64],
                       dst4: &mut [f64],
                       dst_state: &mut [f64]|
     -> Result<(), RelativeStrengthIndexWaveIndicatorError> {
        let params = resolve_params(&combos[row])?;
        compute_relative_strength_index_wave_indicator_into(
            source, high, low, params, dst1, dst2, dst3, dst4, dst_state,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_rsi_ma1
                .par_chunks_mut(cols)
                .zip(out_rsi_ma2.par_chunks_mut(cols))
                .zip(out_rsi_ma3.par_chunks_mut(cols))
                .zip(out_rsi_ma4.par_chunks_mut(cols))
                .zip(out_state.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, ((((dst1, dst2), dst3), dst4), dst_state))| {
                    compute_row(row, dst1, dst2, dst3, dst4, dst_state)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                compute_row(
                    row,
                    &mut out_rsi_ma1[start..end],
                    &mut out_rsi_ma2[start..end],
                    &mut out_rsi_ma3[start..end],
                    &mut out_rsi_ma4[start..end],
                    &mut out_state[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            compute_row(
                row,
                &mut out_rsi_ma1[start..end],
                &mut out_rsi_ma2[start..end],
                &mut out_rsi_ma3[start..end],
                &mut out_rsi_ma4[start..end],
                &mut out_state[start..end],
            )?;
        }
    }
    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_source_high_low(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut source = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            let close = 100.0 + (x * 0.11).sin() * 2.2 + (x * 0.03).cos() * 1.1 + x * 0.04;
            let h = close + 0.7 + (x * 0.07).sin().abs() * 0.4;
            let l = close - 0.8 - (x * 0.05).cos().abs() * 0.35;
            source.push((h + l + 2.0 * close) * 0.25);
            high.push(h);
            low.push(l);
        }
        (source, high, low)
    }

    fn assert_series_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(a[i].is_nan() && b[i].is_nan(), "NaN mismatch at {i}");
            } else {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-12,
                    "mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let (source, high, low) = sample_source_high_low(160);
        let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
            &source,
            &high,
            &low,
            RelativeStrengthIndexWaveIndicatorParams::default(),
        );
        let out = relative_strength_index_wave_indicator(&input).unwrap();
        let params = resolve_params(&RelativeStrengthIndexWaveIndicatorParams::default()).unwrap();
        let mut core = RelativeStrengthIndexWaveCore::new(params);
        let mut rsi_ma1 = vec![f64::NAN; source.len()];
        let mut rsi_ma2 = vec![f64::NAN; source.len()];
        let mut rsi_ma3 = vec![f64::NAN; source.len()];
        let mut rsi_ma4 = vec![f64::NAN; source.len()];
        let mut state = vec![f64::NAN; source.len()];
        for i in 0..source.len() {
            let (a, b, c, d, s) = core.update(source[i], high[i], low[i]);
            rsi_ma1[i] = a;
            rsi_ma2[i] = b;
            rsi_ma3[i] = c;
            rsi_ma4[i] = d;
            state[i] = s;
        }
        assert_series_eq(&out.rsi_ma1, &rsi_ma1);
        assert_series_eq(&out.rsi_ma2, &rsi_ma2);
        assert_series_eq(&out.rsi_ma3, &rsi_ma3);
        assert_series_eq(&out.rsi_ma4, &rsi_ma4);
        assert_series_eq(&out.state, &state);
    }

    #[test]
    fn stream_matches_batch() {
        let (source, high, low) = sample_source_high_low(144);
        let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
            &source,
            &high,
            &low,
            RelativeStrengthIndexWaveIndicatorParams::default(),
        );
        let batch = relative_strength_index_wave_indicator(&input).unwrap();
        let mut stream =
            RelativeStrengthIndexWaveIndicatorStream::try_new(Default::default()).unwrap();
        let mut rsi_ma1 = vec![f64::NAN; source.len()];
        let mut rsi_ma2 = vec![f64::NAN; source.len()];
        let mut rsi_ma3 = vec![f64::NAN; source.len()];
        let mut rsi_ma4 = vec![f64::NAN; source.len()];
        let mut state = vec![f64::NAN; source.len()];
        for i in 0..source.len() {
            let (a, b, c, d, s) = stream.update(source[i], high[i], low[i]);
            rsi_ma1[i] = a;
            rsi_ma2[i] = b;
            rsi_ma3[i] = c;
            rsi_ma4[i] = d;
            state[i] = s;
        }
        assert_series_eq(&batch.rsi_ma1, &rsi_ma1);
        assert_series_eq(&batch.rsi_ma2, &rsi_ma2);
        assert_series_eq(&batch.rsi_ma3, &rsi_ma3);
        assert_series_eq(&batch.rsi_ma4, &rsi_ma4);
        assert_series_eq(&batch.state, &state);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (source, high, low) = sample_source_high_low(128);
        let sweep = RelativeStrengthIndexWaveIndicatorBatchRange {
            rsi_length: (14, 15, 1),
            ..Default::default()
        };
        let batch = relative_strength_index_wave_indicator_batch_slice(
            &source,
            &high,
            &low,
            &sweep,
            Kernel::ScalarBatch,
        )
        .unwrap();
        let single = relative_strength_index_wave_indicator(
            &RelativeStrengthIndexWaveIndicatorInput::from_slices(
                &source,
                &high,
                &low,
                RelativeStrengthIndexWaveIndicatorParams::default(),
            ),
        )
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, source.len());
        assert_series_eq(&batch.rsi_ma1[..source.len()], &single.rsi_ma1);
        assert_series_eq(&batch.rsi_ma2[..source.len()], &single.rsi_ma2);
        assert_series_eq(&batch.rsi_ma3[..source.len()], &single.rsi_ma3);
        assert_series_eq(&batch.rsi_ma4[..source.len()], &single.rsi_ma4);
        assert_series_eq(&batch.state[..source.len()], &single.state);
    }

    #[test]
    fn into_slice_matches_single() {
        let (source, high, low) = sample_source_high_low(96);
        let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
            &source,
            &high,
            &low,
            RelativeStrengthIndexWaveIndicatorParams::default(),
        );
        let expected = relative_strength_index_wave_indicator(&input).unwrap();
        let mut rsi_ma1 = vec![f64::NAN; source.len()];
        let mut rsi_ma2 = vec![f64::NAN; source.len()];
        let mut rsi_ma3 = vec![f64::NAN; source.len()];
        let mut rsi_ma4 = vec![f64::NAN; source.len()];
        let mut state = vec![f64::NAN; source.len()];
        relative_strength_index_wave_indicator_into_slice(
            &mut rsi_ma1,
            &mut rsi_ma2,
            &mut rsi_ma3,
            &mut rsi_ma4,
            &mut state,
            &input,
            Kernel::Auto,
        )
        .unwrap();
        assert_series_eq(&expected.rsi_ma1, &rsi_ma1);
        assert_series_eq(&expected.rsi_ma2, &rsi_ma2);
        assert_series_eq(&expected.rsi_ma3, &rsi_ma3);
        assert_series_eq(&expected.rsi_ma4, &rsi_ma4);
        assert_series_eq(&expected.state, &state);
    }

    #[test]
    fn invalid_lengths_are_rejected() {
        let (source, high, low) = sample_source_high_low(64);
        let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
            &source,
            &high,
            &low,
            RelativeStrengthIndexWaveIndicatorParams {
                rsi_length: Some(0),
                ..Default::default()
            },
        );
        let err = relative_strength_index_wave_indicator(&input).unwrap_err();
        assert!(matches!(
            err,
            RelativeStrengthIndexWaveIndicatorError::InvalidRsiLength { .. }
        ));
    }
}
