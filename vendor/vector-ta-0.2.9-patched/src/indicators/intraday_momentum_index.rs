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

const DEFAULT_LENGTH: usize = 14;
const DEFAULT_LENGTH_MA: usize = 6;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_LENGTH_BB: usize = 20;
const DEFAULT_APPLY_SMOOTHING: bool = false;
const DEFAULT_LOW_BAND: usize = 10;
const PI: f64 = std::f64::consts::PI;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[derive(Debug, Clone)]
pub enum IntradayMomentumIndexData<'a> {
    Candles { candles: &'a Candles },
    Slices { open: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct IntradayMomentumIndexOutput {
    pub imi: Vec<f64>,
    pub upper_hit: Vec<f64>,
    pub lower_hit: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct IntradayMomentumIndexParams {
    pub length: Option<usize>,
    pub length_ma: Option<usize>,
    pub mult: Option<f64>,
    pub length_bb: Option<usize>,
    pub apply_smoothing: Option<bool>,
    pub low_band: Option<usize>,
}

impl Default for IntradayMomentumIndexParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            length_ma: Some(DEFAULT_LENGTH_MA),
            mult: Some(DEFAULT_MULT),
            length_bb: Some(DEFAULT_LENGTH_BB),
            apply_smoothing: Some(DEFAULT_APPLY_SMOOTHING),
            low_band: Some(DEFAULT_LOW_BAND),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IntradayMomentumIndexInput<'a> {
    pub data: IntradayMomentumIndexData<'a>,
    pub params: IntradayMomentumIndexParams,
}

impl<'a> IntradayMomentumIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: IntradayMomentumIndexParams) -> Self {
        Self {
            data: IntradayMomentumIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        close: &'a [f64],
        params: IntradayMomentumIndexParams,
    ) -> Self {
        Self {
            data: IntradayMomentumIndexData::Slices { open, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, IntradayMomentumIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_length_ma(&self) -> usize {
        self.params.length_ma.unwrap_or(DEFAULT_LENGTH_MA)
    }

    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(DEFAULT_MULT)
    }

    #[inline]
    pub fn get_length_bb(&self) -> usize {
        self.params.length_bb.unwrap_or(DEFAULT_LENGTH_BB)
    }

    #[inline]
    pub fn get_apply_smoothing(&self) -> bool {
        self.params
            .apply_smoothing
            .unwrap_or(DEFAULT_APPLY_SMOOTHING)
    }

    #[inline]
    pub fn get_low_band(&self) -> usize {
        self.params.low_band.unwrap_or(DEFAULT_LOW_BAND)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct IntradayMomentumIndexBuilder {
    length: Option<usize>,
    length_ma: Option<usize>,
    mult: Option<f64>,
    length_bb: Option<usize>,
    apply_smoothing: Option<bool>,
    low_band: Option<usize>,
    kernel: Kernel,
}

impl Default for IntradayMomentumIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            length_ma: None,
            mult: None,
            length_bb: None,
            apply_smoothing: None,
            low_band: None,
            kernel: Kernel::Auto,
        }
    }
}

impl IntradayMomentumIndexBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline]
    pub fn length_ma(mut self, length_ma: usize) -> Self {
        self.length_ma = Some(length_ma);
        self
    }

    #[inline]
    pub fn mult(mut self, mult: f64) -> Self {
        self.mult = Some(mult);
        self
    }

    #[inline]
    pub fn length_bb(mut self, length_bb: usize) -> Self {
        self.length_bb = Some(length_bb);
        self
    }

    #[inline]
    pub fn apply_smoothing(mut self, apply_smoothing: bool) -> Self {
        self.apply_smoothing = Some(apply_smoothing);
        self
    }

    #[inline]
    pub fn low_band(mut self, low_band: usize) -> Self {
        self.low_band = Some(low_band);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<IntradayMomentumIndexOutput, IntradayMomentumIndexError> {
        let input = IntradayMomentumIndexInput::from_candles(
            candles,
            IntradayMomentumIndexParams {
                length: self.length,
                length_ma: self.length_ma,
                mult: self.mult,
                length_bb: self.length_bb,
                apply_smoothing: self.apply_smoothing,
                low_band: self.low_band,
            },
        );
        intraday_momentum_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<IntradayMomentumIndexOutput, IntradayMomentumIndexError> {
        let input = IntradayMomentumIndexInput::from_slices(
            open,
            close,
            IntradayMomentumIndexParams {
                length: self.length,
                length_ma: self.length_ma,
                mult: self.mult,
                length_bb: self.length_bb,
                apply_smoothing: self.apply_smoothing,
                low_band: self.low_band,
            },
        );
        intraday_momentum_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<IntradayMomentumIndexStream, IntradayMomentumIndexError> {
        IntradayMomentumIndexStream::try_new(IntradayMomentumIndexParams {
            length: self.length,
            length_ma: self.length_ma,
            mult: self.mult,
            length_bb: self.length_bb,
            apply_smoothing: self.apply_smoothing,
            low_band: self.low_band,
        })
    }
}

#[derive(Debug, Error)]
pub enum IntradayMomentumIndexError {
    #[error("intraday_momentum_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("intraday_momentum_index: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "intraday_momentum_index: Inconsistent slice lengths: open={open_len}, close={close_len}"
    )]
    InconsistentSliceLengths { open_len: usize, close_len: usize },
    #[error("intraday_momentum_index: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "intraday_momentum_index: Invalid length_ma: length_ma = {length_ma}, data length = {data_len}"
    )]
    InvalidLengthMa { length_ma: usize, data_len: usize },
    #[error(
        "intraday_momentum_index: Invalid length_bb: length_bb = {length_bb}, data length = {data_len}"
    )]
    InvalidLengthBb { length_bb: usize, data_len: usize },
    #[error("intraday_momentum_index: Invalid mult: {mult}. Must be finite and >= 0.")]
    InvalidMult { mult: f64 },
    #[error(
        "intraday_momentum_index: Invalid low_band: {low_band}. Must be >= 1 when smoothing is enabled."
    )]
    InvalidLowBand { low_band: usize },
    #[error("intraday_momentum_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "intraday_momentum_index: Output length mismatch: expected = {expected}, imi = {imi_got}, upper_hit = {upper_hit_got}, lower_hit = {lower_hit_got}, signal = {signal_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        imi_got: usize,
        upper_hit_got: usize,
        lower_hit_got: usize,
        signal_got: usize,
    },
    #[error("intraday_momentum_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("intraday_momentum_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct EmaState {
    alpha: f64,
    seeded: bool,
    value: f64,
}

impl EmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            seeded: false,
            value: f64::NAN,
        }
    }

    #[inline]
    fn update(&mut self, x: f64) -> f64 {
        if !self.seeded {
            self.seeded = true;
            self.value = x;
        } else {
            self.value = self.alpha.mul_add(x, (1.0 - self.alpha) * self.value);
        }
        self.value
    }
}

#[derive(Debug, Clone)]
pub struct IntradayMomentumIndexStream {
    length: usize,
    mult: f64,
    length_bb: usize,
    apply_smoothing: bool,
    gains: Vec<f64>,
    losses: Vec<f64>,
    valid: Vec<u8>,
    idx: usize,
    count: usize,
    valid_count: usize,
    sum_gain: f64,
    sum_loss: f64,
    signal_ema: EmaState,
    basis_ema: EmaState,
    coeff1: f64,
    coeff2: f64,
    coeff3: f64,
    prev_price: f64,
    prev_filt1: f64,
    prev_filt2: f64,
    bb_values: Vec<f64>,
    bb_valid: Vec<u8>,
    bb_idx: usize,
    bb_count: usize,
    bb_valid_count: usize,
    bb_sum: f64,
    bb_sumsq: f64,
}

impl IntradayMomentumIndexStream {
    pub fn try_new(
        params: IntradayMomentumIndexParams,
    ) -> Result<IntradayMomentumIndexStream, IntradayMomentumIndexError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length == 0 {
            return Err(IntradayMomentumIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        let length_ma = params.length_ma.unwrap_or(DEFAULT_LENGTH_MA);
        if length_ma == 0 {
            return Err(IntradayMomentumIndexError::InvalidLengthMa {
                length_ma,
                data_len: 0,
            });
        }
        let length_bb = params.length_bb.unwrap_or(DEFAULT_LENGTH_BB);
        if length_bb == 0 {
            return Err(IntradayMomentumIndexError::InvalidLengthBb {
                length_bb,
                data_len: 0,
            });
        }
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        if !mult.is_finite() || mult < 0.0 {
            return Err(IntradayMomentumIndexError::InvalidMult { mult });
        }
        let apply_smoothing = params.apply_smoothing.unwrap_or(DEFAULT_APPLY_SMOOTHING);
        let low_band = params.low_band.unwrap_or(DEFAULT_LOW_BAND);
        if apply_smoothing && low_band == 0 {
            return Err(IntradayMomentumIndexError::InvalidLowBand { low_band });
        }
        let (coeff1, coeff2, coeff3) = if apply_smoothing {
            supersmoother_coefficients(low_band as f64)
        } else {
            (0.0, 0.0, 0.0)
        };
        Ok(Self {
            length,
            mult,
            length_bb,
            apply_smoothing,
            gains: vec![0.0; length],
            losses: vec![0.0; length],
            valid: vec![0; length],
            idx: 0,
            count: 0,
            valid_count: 0,
            sum_gain: 0.0,
            sum_loss: 0.0,
            signal_ema: EmaState::new(length_ma),
            basis_ema: EmaState::new(length_bb),
            coeff1,
            coeff2,
            coeff3,
            prev_price: 0.0,
            prev_filt1: 0.0,
            prev_filt2: 0.0,
            bb_values: vec![0.0; length_bb],
            bb_valid: vec![0; length_bb],
            bb_idx: 0,
            bb_count: 0,
            bb_valid_count: 0,
            bb_sum: 0.0,
            bb_sumsq: 0.0,
        })
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.length.saturating_sub(1)
    }

    pub fn update(&mut self, open: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        let valid_bar = open.is_finite() && close.is_finite();
        if self.count >= self.length {
            let old = self.idx;
            if self.valid[old] != 0 {
                self.valid_count = self.valid_count.saturating_sub(1);
                self.sum_gain -= self.gains[old];
                self.sum_loss -= self.losses[old];
            }
        } else {
            self.count += 1;
        }

        if valid_bar {
            let diff = close - open;
            let gain = diff.max(0.0);
            let loss = (-diff).max(0.0);
            self.gains[self.idx] = gain;
            self.losses[self.idx] = loss;
            self.valid[self.idx] = 1;
            self.valid_count += 1;
            self.sum_gain += gain;
            self.sum_loss += loss;
        } else {
            self.gains[self.idx] = 0.0;
            self.losses[self.idx] = 0.0;
            self.valid[self.idx] = 0;
        }
        self.idx += 1;
        if self.idx == self.length {
            self.idx = 0;
        }

        let mut imi = f64::NAN;
        if self.count >= self.length && self.valid_count == self.length {
            let denom = self.sum_gain + self.sum_loss;
            if denom > 0.0 && denom.is_finite() {
                let raw_imi = 100.0 * (self.sum_gain / denom);
                imi = if self.apply_smoothing {
                    let filt = self.coeff1 * (raw_imi + self.prev_price) * 0.5
                        + self.coeff2 * self.prev_filt1
                        + self.coeff3 * self.prev_filt2;
                    self.prev_price = raw_imi;
                    self.prev_filt2 = self.prev_filt1;
                    self.prev_filt1 = filt;
                    filt
                } else {
                    raw_imi
                };
            }
        }

        if !imi.is_finite() {
            if self.apply_smoothing {
                self.prev_price = 0.0;
                self.prev_filt1 = 0.0;
                self.prev_filt2 = 0.0;
            }
            self.push_bb(f64::NAN);
            return None;
        }

        let signal = self.signal_ema.update(imi);
        let basis = self.basis_ema.update(imi);
        let dev = self.push_bb(imi);
        let upper_hit = if dev.is_finite() {
            let upper = basis + self.mult * dev;
            if imi >= upper { imi } else { f64::NAN }
        } else {
            f64::NAN
        };
        let lower_hit = if dev.is_finite() {
            let lower = basis - self.mult * dev;
            if imi <= lower { imi } else { f64::NAN }
        } else {
            f64::NAN
        };

        Some((imi, upper_hit, lower_hit, signal))
    }

    #[inline]
    fn push_bb(&mut self, value: f64) -> f64 {
        if self.bb_count >= self.length_bb {
            let old = self.bb_idx;
            if self.bb_valid[old] != 0 {
                self.bb_valid_count = self.bb_valid_count.saturating_sub(1);
                let old_value = self.bb_values[old];
                self.bb_sum -= old_value;
                self.bb_sumsq -= old_value * old_value;
            }
        } else {
            self.bb_count += 1;
        }

        if value.is_finite() {
            self.bb_values[self.bb_idx] = value;
            self.bb_valid[self.bb_idx] = 1;
            self.bb_valid_count += 1;
            self.bb_sum += value;
            self.bb_sumsq += value * value;
        } else {
            self.bb_values[self.bb_idx] = 0.0;
            self.bb_valid[self.bb_idx] = 0;
        }

        self.bb_idx += 1;
        if self.bb_idx == self.length_bb {
            self.bb_idx = 0;
        }

        if self.bb_count < self.length_bb || self.bb_valid_count != self.length_bb {
            return f64::NAN;
        }
        let n = self.length_bb as f64;
        let mean = self.bb_sum / n;
        let variance = (self.bb_sumsq / n - mean * mean).max(0.0);
        variance.sqrt()
    }
}

#[inline]
fn supersmoother_coefficients(low_band: f64) -> (f64, f64, f64) {
    let a1 = (-PI * SQRT_2 / low_band).exp();
    let coeff2 = 2.0 * a1 * (SQRT_2 * PI / low_band).cos();
    let coeff3 = -(a1 * a1);
    let coeff1 = 1.0 - coeff2 - coeff3;
    (coeff1, coeff2, coeff3)
}

#[inline]
fn first_valid_open_close(open: &[f64], close: &[f64]) -> usize {
    open.iter()
        .zip(close.iter())
        .position(|(&o, &c)| o.is_finite() && c.is_finite())
        .unwrap_or(open.len())
}

#[inline]
fn count_valid_open_close(open: &[f64], close: &[f64]) -> usize {
    open.iter()
        .zip(close.iter())
        .filter(|&(o, c)| o.is_finite() && c.is_finite())
        .count()
}

fn intraday_momentum_index_row_from_slices(
    open: &[f64],
    close: &[f64],
    params: &IntradayMomentumIndexParams,
    imi_out: &mut [f64],
    upper_hit_out: &mut [f64],
    lower_hit_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), IntradayMomentumIndexError> {
    imi_out.fill(f64::NAN);
    upper_hit_out.fill(f64::NAN);
    lower_hit_out.fill(f64::NAN);
    signal_out.fill(f64::NAN);

    let mut stream = IntradayMomentumIndexStream::try_new(params.clone())?;
    for i in 0..open.len() {
        if let Some((imi, upper_hit, lower_hit, signal)) = stream.update(open[i], close[i]) {
            imi_out[i] = imi;
            upper_hit_out[i] = upper_hit;
            lower_hit_out[i] = lower_hit;
            signal_out[i] = signal;
        }
    }
    Ok(())
}

#[inline]
fn intraday_momentum_index_prepare<'a>(
    input: &'a IntradayMomentumIndexInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize, Kernel), IntradayMomentumIndexError> {
    let (open, close) = match &input.data {
        IntradayMomentumIndexData::Candles { candles } => {
            (&candles.open[..], candles.close.as_slice())
        }
        IntradayMomentumIndexData::Slices { open, close } => {
            if open.len() != close.len() {
                return Err(IntradayMomentumIndexError::InconsistentSliceLengths {
                    open_len: open.len(),
                    close_len: close.len(),
                });
            }
            (*open, *close)
        }
    };

    let len = open.len();
    if len == 0 {
        return Err(IntradayMomentumIndexError::EmptyInputData);
    }

    let first = first_valid_open_close(open, close);
    if first >= len {
        return Err(IntradayMomentumIndexError::AllValuesNaN);
    }

    let length = input.get_length();
    if length == 0 || length > len {
        return Err(IntradayMomentumIndexError::InvalidLength {
            length,
            data_len: len,
        });
    }

    let length_ma = input.get_length_ma();
    if length_ma == 0 || length_ma > len {
        return Err(IntradayMomentumIndexError::InvalidLengthMa {
            length_ma,
            data_len: len,
        });
    }

    let length_bb = input.get_length_bb();
    if length_bb == 0 || length_bb > len {
        return Err(IntradayMomentumIndexError::InvalidLengthBb {
            length_bb,
            data_len: len,
        });
    }

    let mult = input.get_mult();
    if !mult.is_finite() || mult < 0.0 {
        return Err(IntradayMomentumIndexError::InvalidMult { mult });
    }

    let apply_smoothing = input.get_apply_smoothing();
    let low_band = input.get_low_band();
    if apply_smoothing && low_band == 0 {
        return Err(IntradayMomentumIndexError::InvalidLowBand { low_band });
    }

    let valid = count_valid_open_close(open, close);
    if valid < length {
        return Err(IntradayMomentumIndexError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((open, close, first, length, length_bb, chosen))
}

#[inline]
pub fn intraday_momentum_index(
    input: &IntradayMomentumIndexInput,
) -> Result<IntradayMomentumIndexOutput, IntradayMomentumIndexError> {
    intraday_momentum_index_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn intraday_momentum_index_with_kernel(
    input: &IntradayMomentumIndexInput,
    kernel: Kernel,
) -> Result<IntradayMomentumIndexOutput, IntradayMomentumIndexError> {
    let (open, close, first, length, length_bb, _chosen) =
        intraday_momentum_index_prepare(input, kernel)?;
    let imi_warmup = first + length - 1;
    let band_warmup = first + length + length_bb - 2;
    let mut imi = alloc_with_nan_prefix(open.len(), imi_warmup.min(open.len()));
    let mut signal = alloc_with_nan_prefix(open.len(), imi_warmup.min(open.len()));
    let mut upper_hit = alloc_with_nan_prefix(open.len(), band_warmup.min(open.len()));
    let mut lower_hit = alloc_with_nan_prefix(open.len(), band_warmup.min(open.len()));
    intraday_momentum_index_row_from_slices(
        open,
        close,
        &input.params,
        &mut imi,
        &mut upper_hit,
        &mut lower_hit,
        &mut signal,
    )?;
    Ok(IntradayMomentumIndexOutput {
        imi,
        upper_hit,
        lower_hit,
        signal,
    })
}

#[inline]
pub fn intraday_momentum_index_into_slices(
    imi_out: &mut [f64],
    upper_hit_out: &mut [f64],
    lower_hit_out: &mut [f64],
    signal_out: &mut [f64],
    input: &IntradayMomentumIndexInput,
    kernel: Kernel,
) -> Result<(), IntradayMomentumIndexError> {
    let (open, close, _first, _length, _length_bb, _chosen) =
        intraday_momentum_index_prepare(input, kernel)?;
    if imi_out.len() != open.len()
        || upper_hit_out.len() != open.len()
        || lower_hit_out.len() != open.len()
        || signal_out.len() != open.len()
    {
        return Err(IntradayMomentumIndexError::OutputLengthMismatch {
            expected: open.len(),
            imi_got: imi_out.len(),
            upper_hit_got: upper_hit_out.len(),
            lower_hit_got: lower_hit_out.len(),
            signal_got: signal_out.len(),
        });
    }
    intraday_momentum_index_row_from_slices(
        open,
        close,
        &input.params,
        imi_out,
        upper_hit_out,
        lower_hit_out,
        signal_out,
    )
}

#[inline]
pub fn intraday_momentum_index_into(
    input: &IntradayMomentumIndexInput,
    imi_out: &mut [f64],
    upper_hit_out: &mut [f64],
    lower_hit_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), IntradayMomentumIndexError> {
    intraday_momentum_index_into_slices(
        imi_out,
        upper_hit_out,
        lower_hit_out,
        signal_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct IntradayMomentumIndexBatchRange {
    pub length: (usize, usize, usize),
    pub length_ma: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    pub length_bb: (usize, usize, usize),
    pub apply_smoothing: Option<bool>,
    pub low_band: (usize, usize, usize),
}

impl Default for IntradayMomentumIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            length_ma: (DEFAULT_LENGTH_MA, DEFAULT_LENGTH_MA, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
            length_bb: (DEFAULT_LENGTH_BB, DEFAULT_LENGTH_BB, 0),
            apply_smoothing: Some(DEFAULT_APPLY_SMOOTHING),
            low_band: (DEFAULT_LOW_BAND, DEFAULT_LOW_BAND, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IntradayMomentumIndexBatchBuilder {
    range: IntradayMomentumIndexBatchRange,
    kernel: Kernel,
}

impl IntradayMomentumIndexBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length_ma = (start, end, step);
        self
    }

    #[inline]
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }

    #[inline]
    pub fn length_bb_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length_bb = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_smoothing(mut self, apply_smoothing: bool) -> Self {
        self.range.apply_smoothing = Some(apply_smoothing);
        self
    }

    #[inline]
    pub fn low_band_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.low_band = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<IntradayMomentumIndexBatchOutput, IntradayMomentumIndexError> {
        intraday_momentum_index_batch_with_kernel(open, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<IntradayMomentumIndexBatchOutput, IntradayMomentumIndexError> {
        intraday_momentum_index_batch_with_kernel(
            &candles.open,
            &candles.close,
            &self.range,
            self.kernel,
        )
    }
}

#[derive(Debug, Clone)]
pub struct IntradayMomentumIndexBatchOutput {
    pub imi: Vec<f64>,
    pub upper_hit: Vec<f64>,
    pub lower_hit: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<IntradayMomentumIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, IntradayMomentumIndexError> {
    if start > end {
        return Err(IntradayMomentumIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    let mut cur = start;
    while cur <= end {
        out.push(cur);
        match cur.checked_add(step) {
            Some(next) if next > cur => cur = next,
            _ => break,
        }
    }
    Ok(out)
}

fn expand_f64_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, IntradayMomentumIndexError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end || step < 0.0 {
        return Err(IntradayMomentumIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    let mut cur = start;
    while cur <= end + 1e-12 {
        out.push(cur);
        cur += step;
    }
    Ok(out)
}

fn expand_grid_intraday_momentum_index(
    sweep: &IntradayMomentumIndexBatchRange,
) -> Result<Vec<IntradayMomentumIndexParams>, IntradayMomentumIndexError> {
    let lengths = expand_usize_range(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let length_mas = expand_usize_range(sweep.length_ma.0, sweep.length_ma.1, sweep.length_ma.2)?;
    let mults = expand_f64_range(sweep.mult.0, sweep.mult.1, sweep.mult.2)?;
    let length_bbs = expand_usize_range(sweep.length_bb.0, sweep.length_bb.1, sweep.length_bb.2)?;
    let low_bands = expand_usize_range(sweep.low_band.0, sweep.low_band.1, sweep.low_band.2)?;
    let apply_smoothing = sweep.apply_smoothing.unwrap_or(DEFAULT_APPLY_SMOOTHING);

    let mut out = Vec::with_capacity(
        lengths.len() * length_mas.len() * mults.len() * length_bbs.len() * low_bands.len(),
    );
    for &length in &lengths {
        for &length_ma in &length_mas {
            for &mult in &mults {
                for &length_bb in &length_bbs {
                    for &low_band in &low_bands {
                        out.push(IntradayMomentumIndexParams {
                            length: Some(length),
                            length_ma: Some(length_ma),
                            mult: Some(mult),
                            length_bb: Some(length_bb),
                            apply_smoothing: Some(apply_smoothing),
                            low_band: Some(low_band),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn intraday_momentum_index_batch_with_kernel(
    open: &[f64],
    close: &[f64],
    sweep: &IntradayMomentumIndexBatchRange,
    kernel: Kernel,
) -> Result<IntradayMomentumIndexBatchOutput, IntradayMomentumIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(IntradayMomentumIndexError::InvalidKernelForBatch(other)),
    };
    intraday_momentum_index_batch_par_slice(open, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn intraday_momentum_index_batch_slice(
    open: &[f64],
    close: &[f64],
    sweep: &IntradayMomentumIndexBatchRange,
    kernel: Kernel,
) -> Result<IntradayMomentumIndexBatchOutput, IntradayMomentumIndexError> {
    intraday_momentum_index_batch_inner(open, close, sweep, kernel, false)
}

#[inline]
pub fn intraday_momentum_index_batch_par_slice(
    open: &[f64],
    close: &[f64],
    sweep: &IntradayMomentumIndexBatchRange,
    kernel: Kernel,
) -> Result<IntradayMomentumIndexBatchOutput, IntradayMomentumIndexError> {
    intraday_momentum_index_batch_inner(open, close, sweep, kernel, true)
}

fn intraday_momentum_index_batch_inner(
    open: &[f64],
    close: &[f64],
    sweep: &IntradayMomentumIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<IntradayMomentumIndexBatchOutput, IntradayMomentumIndexError> {
    if open.is_empty() || close.is_empty() {
        return Err(IntradayMomentumIndexError::EmptyInputData);
    }
    if open.len() != close.len() {
        return Err(IntradayMomentumIndexError::InconsistentSliceLengths {
            open_len: open.len(),
            close_len: close.len(),
        });
    }
    let combos = expand_grid_intraday_momentum_index(sweep)?;
    let rows = combos.len();
    let cols = open.len();
    let first = first_valid_open_close(open, close);
    if first >= cols {
        return Err(IntradayMomentumIndexError::AllValuesNaN);
    }
    let valid = count_valid_open_close(open, close);

    let mut imi_mu = make_uninit_matrix(rows, cols);
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let mut imi_warms = Vec::with_capacity(rows);
    let mut band_warms = Vec::with_capacity(rows);

    for combo in &combos {
        let length = combo.length.unwrap_or(DEFAULT_LENGTH);
        let length_bb = combo.length_bb.unwrap_or(DEFAULT_LENGTH_BB);
        if valid < length {
            return Err(IntradayMomentumIndexError::NotEnoughValidData {
                needed: length,
                valid,
            });
        }
        imi_warms.push((first + length - 1).min(cols));
        band_warms.push((first + length + length_bb - 2).min(cols));
    }

    init_matrix_prefixes(&mut imi_mu, cols, &imi_warms);
    init_matrix_prefixes(&mut signal_mu, cols, &imi_warms);
    init_matrix_prefixes(&mut upper_mu, cols, &band_warms);
    init_matrix_prefixes(&mut lower_mu, cols, &band_warms);

    let mut imi_guard = ManuallyDrop::new(imi_mu);
    let mut upper_guard = ManuallyDrop::new(upper_mu);
    let mut lower_guard = ManuallyDrop::new(lower_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);

    let imi_out = unsafe {
        std::slice::from_raw_parts_mut(imi_guard.as_mut_ptr() as *mut f64, imi_guard.len())
    };
    let upper_out = unsafe {
        std::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let lower_out = unsafe {
        std::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };
    let signal_out = unsafe {
        std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            imi_out
                .par_chunks_mut(cols)
                .zip(upper_out.par_chunks_mut(cols))
                .zip(lower_out.par_chunks_mut(cols))
                .zip(signal_out.par_chunks_mut(cols))
                .zip(combos.par_iter())
                .for_each(|((((dst_imi, dst_upper), dst_lower), dst_signal), combo)| {
                    let _ = intraday_momentum_index_row_from_slices(
                        open, close, combo, dst_imi, dst_upper, dst_lower, dst_signal,
                    );
                });
        }
    } else {
        let _ = kernel;
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            intraday_momentum_index_row_from_slices(
                open,
                close,
                combo,
                &mut imi_out[start..end],
                &mut upper_out[start..end],
                &mut lower_out[start..end],
                &mut signal_out[start..end],
            )?;
        }
    }

    let imi = unsafe {
        Vec::from_raw_parts(
            imi_guard.as_mut_ptr() as *mut f64,
            imi_guard.len(),
            imi_guard.capacity(),
        )
    };
    let upper_hit = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            upper_guard.len(),
            upper_guard.capacity(),
        )
    };
    let lower_hit = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            lower_guard.len(),
            lower_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };
    core::mem::forget(imi_guard);
    core::mem::forget(upper_guard);
    core::mem::forget(lower_guard);
    core::mem::forget(signal_guard);

    Ok(IntradayMomentumIndexBatchOutput {
        imi,
        upper_hit,
        lower_hit,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn intraday_momentum_index_batch_inner_into(
    open: &[f64],
    close: &[f64],
    sweep: &IntradayMomentumIndexBatchRange,
    kernel: Kernel,
    imi_out: &mut [f64],
    upper_hit_out: &mut [f64],
    lower_hit_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<IntradayMomentumIndexParams>, IntradayMomentumIndexError> {
    let out = intraday_momentum_index_batch_inner(open, close, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if imi_out.len() != total
        || upper_hit_out.len() != total
        || lower_hit_out.len() != total
        || signal_out.len() != total
    {
        return Err(IntradayMomentumIndexError::OutputLengthMismatch {
            expected: total,
            imi_got: imi_out.len(),
            upper_hit_got: upper_hit_out.len(),
            lower_hit_got: lower_hit_out.len(),
            signal_got: signal_out.len(),
        });
    }
    imi_out.copy_from_slice(&out.imi);
    upper_hit_out.copy_from_slice(&out.upper_hit);
    lower_hit_out.copy_from_slice(&out.lower_hit);
    signal_out.copy_from_slice(&out.signal);
    Ok(out.combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    fn load_open_close() -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let candles = read_candles_from_vortex("src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex")?;
        Ok((candles.open.clone(), candles.close.clone()))
    }

    fn assert_series_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (idx, (&a, &b)) in left.iter().zip(right.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= 1e-10,
                "mismatch at {}: left={}, right={}",
                idx,
                a,
                b
            );
        }
    }

    #[test]
    fn intraday_momentum_index_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, close) = load_open_close()?;
        let input = IntradayMomentumIndexInput::from_slices(
            &open,
            &close,
            IntradayMomentumIndexParams::default(),
        );
        let out = intraday_momentum_index_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.imi.len(), close.len());
        assert_eq!(out.upper_hit.len(), close.len());
        assert_eq!(out.lower_hit.len(), close.len());
        assert_eq!(out.signal.len(), close.len());
        assert!(out.imi.iter().any(|v| v.is_finite()));
        assert!(out.signal.iter().any(|v| v.is_finite()));
        for &value in out.imi.iter().filter(|v| v.is_finite()).take(64) {
            assert!((0.0..=100.0).contains(&value));
        }
        Ok(())
    }

    #[test]
    fn intraday_momentum_index_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let (open, close) = load_open_close()?;
        let input = IntradayMomentumIndexInput::from_slices(
            &open,
            &close,
            IntradayMomentumIndexParams {
                length: Some(11),
                length_ma: Some(5),
                mult: Some(1.75),
                length_bb: Some(16),
                apply_smoothing: Some(true),
                low_band: Some(9),
            },
        );
        let auto = intraday_momentum_index_with_kernel(&input, Kernel::Auto)?;
        let scalar = intraday_momentum_index_with_kernel(&input, Kernel::Scalar)?;
        assert_series_eq(&auto.imi, &scalar.imi);
        assert_series_eq(&auto.upper_hit, &scalar.upper_hit);
        assert_series_eq(&auto.lower_hit, &scalar.lower_hit);
        assert_series_eq(&auto.signal, &scalar.signal);
        Ok(())
    }

    #[test]
    fn intraday_momentum_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, close) = load_open_close()?;
        let input = IntradayMomentumIndexInput::from_slices(
            &open,
            &close,
            IntradayMomentumIndexParams {
                length: Some(14),
                length_ma: Some(6),
                mult: Some(2.0),
                length_bb: Some(20),
                apply_smoothing: Some(true),
                low_band: Some(10),
            },
        );
        let expected = intraday_momentum_index(&input)?;
        let mut imi = vec![f64::NAN; close.len()];
        let mut upper = vec![f64::NAN; close.len()];
        let mut lower = vec![f64::NAN; close.len()];
        let mut signal = vec![f64::NAN; close.len()];
        intraday_momentum_index_into(&input, &mut imi, &mut upper, &mut lower, &mut signal)?;
        assert_series_eq(&imi, &expected.imi);
        assert_series_eq(&upper, &expected.upper_hit);
        assert_series_eq(&lower, &expected.lower_hit);
        assert_series_eq(&signal, &expected.signal);
        Ok(())
    }

    #[test]
    fn intraday_momentum_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (open, close) = load_open_close()?;
        let params = IntradayMomentumIndexParams {
            length: Some(14),
            length_ma: Some(6),
            mult: Some(2.0),
            length_bb: Some(20),
            apply_smoothing: Some(true),
            low_band: Some(10),
        };
        let batch = intraday_momentum_index(&IntradayMomentumIndexInput::from_slices(
            &open,
            &close,
            params.clone(),
        ))?;
        let mut stream = IntradayMomentumIndexStream::try_new(params)?;
        let mut imi = Vec::with_capacity(close.len());
        let mut upper = Vec::with_capacity(close.len());
        let mut lower = Vec::with_capacity(close.len());
        let mut signal = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            if let Some((imi_v, upper_v, lower_v, signal_v)) = stream.update(open[i], close[i]) {
                imi.push(imi_v);
                upper.push(upper_v);
                lower.push(lower_v);
                signal.push(signal_v);
            } else {
                imi.push(f64::NAN);
                upper.push(f64::NAN);
                lower.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }
        assert_series_eq(&imi, &batch.imi);
        assert_series_eq(&upper, &batch.upper_hit);
        assert_series_eq(&lower, &batch.lower_hit);
        assert_series_eq(&signal, &batch.signal);
        Ok(())
    }

    #[test]
    fn intraday_momentum_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, close) = load_open_close()?;
        let open = &open[..256];
        let close = &close[..256];
        let single = intraday_momentum_index_with_kernel(
            &IntradayMomentumIndexInput::from_slices(
                open,
                close,
                IntradayMomentumIndexParams::default(),
            ),
            Kernel::Scalar,
        )?;
        let batch = intraday_momentum_index_batch_with_kernel(
            open,
            close,
            &IntradayMomentumIndexBatchRange::default(),
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.imi[..close.len()], &single.imi);
        assert_series_eq(&batch.upper_hit[..close.len()], &single.upper_hit);
        assert_series_eq(&batch.lower_hit[..close.len()], &single.lower_hit);
        assert_series_eq(&batch.signal[..close.len()], &single.signal);
        Ok(())
    }

    #[test]
    fn intraday_momentum_index_rejects_invalid_params() {
        let open = [1.0, 2.0, 3.0];
        let close = [1.5, 2.5, 3.5];
        let err = intraday_momentum_index(&IntradayMomentumIndexInput::from_slices(
            &open,
            &close,
            IntradayMomentumIndexParams {
                length: Some(0),
                ..IntradayMomentumIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            IntradayMomentumIndexError::InvalidLength { .. }
        ));
    }
}
