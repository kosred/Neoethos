#[cfg(feature = "cuda-build-native")]
use crate::cuda::{CudaZscore, CudaZscoreError};
use crate::indicators::deviation::{
    DevError, DevInput, DevParams, DeviationData, DeviationOutput, deviation,
};
use crate::indicators::moving_averages::ma::{MaData, ma};
use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for ZscoreInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ZscoreData::Slice(slice) => slice,
            ZscoreData::Candles { candles, source } => {
                if source.eq_ignore_ascii_case("close") {
                    candles.close.as_slice()
                } else {
                    source_type(candles, source)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ZscoreData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZscoreOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ZscoreParams {
    pub period: Option<usize>,
    pub ma_type: Option<String>,
    pub nbdev: Option<f64>,
    pub devtype: Option<usize>,
}

impl Default for ZscoreParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            ma_type: Some("sma".to_string()),
            nbdev: Some(1.0),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZscoreInput<'a> {
    pub data: ZscoreData<'a>,
    pub params: ZscoreParams,
}

impl<'a> ZscoreInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: ZscoreParams) -> Self {
        Self {
            data: ZscoreData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slice(slice: &'a [f64], params: ZscoreParams) -> Self {
        Self {
            data: ZscoreData::Slice(slice),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", ZscoreParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
    #[inline]
    pub fn get_nbdev(&self) -> f64 {
        self.params.nbdev.unwrap_or(1.0)
    }
    #[inline]
    pub fn get_devtype(&self) -> usize {
        self.params.devtype.unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct ZscoreBuilder {
    period: Option<usize>,
    ma_type: Option<String>,
    nbdev: Option<f64>,
    devtype: Option<usize>,
    kernel: Kernel,
}

impl Default for ZscoreBuilder {
    fn default() -> Self {
        Self {
            period: None,
            ma_type: None,
            nbdev: None,
            devtype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ZscoreBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_type<T: Into<String>>(mut self, t: T) -> Self {
        self.ma_type = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn nbdev(mut self, x: f64) -> Self {
        self.nbdev = Some(x);
        self
    }
    #[inline(always)]
    pub fn devtype(mut self, dt: usize) -> Self {
        self.devtype = Some(dt);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<ZscoreOutput, ZscoreError> {
        let p = ZscoreParams {
            period: self.period,
            ma_type: self.ma_type,
            nbdev: self.nbdev,
            devtype: self.devtype,
        };
        let i = ZscoreInput::from_candles(candles, "close", p);
        zscore_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<ZscoreOutput, ZscoreError> {
        let p = ZscoreParams {
            period: self.period,
            ma_type: self.ma_type,
            nbdev: self.nbdev,
            devtype: self.devtype,
        };
        let i = ZscoreInput::from_slice(data, p);
        zscore_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ZscoreStream, ZscoreError> {
        let p = ZscoreParams {
            period: self.period,
            ma_type: self.ma_type,
            nbdev: self.nbdev,
            devtype: self.devtype,
        };
        ZscoreStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ZscoreError {
    #[error("zscore: Input data slice is empty.")]
    EmptyInputData,
    #[error("zscore: All values are NaN.")]
    AllValuesNaN,
    #[error("zscore: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("zscore: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("zscore: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("zscore: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("zscore: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("zscore: DevError {0}")]
    DevError(#[from] DevError),
    #[error("zscore: MaError {0}")]
    MaError(String),
}

#[inline]
pub fn zscore(input: &ZscoreInput) -> Result<ZscoreOutput, ZscoreError> {
    zscore_with_kernel(input, Kernel::Auto)
}

pub fn zscore_with_kernel(
    input: &ZscoreInput,
    kernel: Kernel,
) -> Result<ZscoreOutput, ZscoreError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(ZscoreError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZscoreError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ZscoreError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(ZscoreError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let ma_type = input.get_ma_type();
    let nbdev = input.get_nbdev();
    let devtype = input.get_devtype();

    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                zscore_scalar(data, period, first, &ma_type, nbdev, devtype)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                zscore_avx2(data, period, first, &ma_type, nbdev, devtype)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                zscore_avx512(data, period, first, &ma_type, nbdev, devtype)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                zscore_scalar(data, period, first, &ma_type, nbdev, devtype)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn zscore_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
    if devtype == 0 {
        if ma_type == "sma" {
            return zscore_scalar_classic_sma(data, period, first, nbdev);
        } else if ma_type == "ema" {
            return zscore_scalar_classic_ema(data, period, first, nbdev);
        }
    }

    let means = ma(ma_type, MaData::Slice(data), period)
        .map_err(|e| ZscoreError::MaError(e.to_string()))?;
    let dev_input = DevInput {
        data: DeviationData::Slice(data),
        params: DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    };
    let mut sigmas = deviation(&dev_input)?.values;
    for v in &mut sigmas {
        *v *= nbdev;
    }

    let warmup_end = first + period - 1;
    let n = data.len();
    let mut out = alloc_with_nan_prefix(n, warmup_end);
    let mut i = warmup_end;
    while i < n {
        let mean = *means.get_unchecked(i);
        let sigma = *sigmas.get_unchecked(i);
        let value = *data.get_unchecked(i);
        *out.get_unchecked_mut(i) = if sigma == 0.0 || sigma.is_nan() {
            f64::NAN
        } else {
            (value - mean) / sigma
        };
        i += 1;
    }
    Ok(ZscoreOutput { values: out })
}

#[inline]
pub unsafe fn zscore_scalar_classic_sma(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
) -> Result<ZscoreOutput, ZscoreError> {
    let warmup_end = first + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);

    let inv = 1.0 / (period as f64);
    let mut sum = 0.0f64;
    let mut sum_sqr = 0.0f64;
    {
        let mut j = first;
        while j <= warmup_end {
            let v = *data.get_unchecked(j);
            sum += v;

            sum_sqr = v.mul_add(v, sum_sqr);
            j += 1;
        }
    }
    let mut mean = sum * inv;

    let mut variance = (-mean).mul_add(mean, sum_sqr * inv);
    if variance < 0.0 {
        variance = 0.0;
    }
    let mut stddev = if variance == 0.0 {
        0.0
    } else {
        variance.sqrt() * nbdev
    };

    let xw = *data.get_unchecked(warmup_end);
    *out.get_unchecked_mut(warmup_end) = if stddev == 0.0 || stddev.is_nan() {
        f64::NAN
    } else {
        (xw - mean) / stddev
    };

    let n = data.len();
    let mut i = warmup_end + 1;
    while i < n {
        let old_val = *data.get_unchecked(i - period);
        let new_val = *data.get_unchecked(i);
        let dd = new_val - old_val;
        sum += dd;

        sum_sqr = (new_val + old_val).mul_add(dd, sum_sqr);
        mean = sum * inv;

        variance = (-mean).mul_add(mean, sum_sqr * inv);
        if variance < 0.0 {
            variance = 0.0;
        }
        stddev = if variance == 0.0 {
            0.0
        } else {
            variance.sqrt() * nbdev
        };

        *out.get_unchecked_mut(i) = if stddev == 0.0 || stddev.is_nan() {
            f64::NAN
        } else {
            (new_val - mean) / stddev
        };
        i += 1;
    }

    Ok(ZscoreOutput { values: out })
}

#[inline]
pub unsafe fn zscore_scalar_classic_ema(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
) -> Result<ZscoreOutput, ZscoreError> {
    let n = data.len();
    let warmup_end = first + period - 1;
    let mut out = alloc_with_nan_prefix(n, warmup_end);

    if n <= warmup_end {
        return Ok(ZscoreOutput { values: out });
    }

    let den = period as f64;
    let inv = 1.0 / den;
    let alpha = 2.0 / (den + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    {
        let mut j = first;
        while j <= warmup_end {
            let v = *data.get_unchecked(j);
            sum += v;
            sum2 = v.mul_add(v, sum2);
            j += 1;
        }
    }
    let mut ema = sum * inv;

    let mut ex = sum * inv;
    let mut ex2 = sum2 * inv;
    let mut mse = (-2.0 * ema).mul_add(ex, ema.mul_add(ema, ex2));
    if mse < 0.0 {
        mse = 0.0;
    }
    let mut sd = mse.sqrt() * nbdev;

    let xw = *data.get_unchecked(warmup_end);
    *out.get_unchecked_mut(warmup_end) = if sd == 0.0 || sd.is_nan() {
        f64::NAN
    } else {
        (xw - ema) / sd
    };

    let mut i = warmup_end + 1;
    while i < n {
        let new = *data.get_unchecked(i);
        let old = *data.get_unchecked(i - period);

        let dd = new - old;
        sum += dd;
        sum2 = (new + old).mul_add(dd, sum2);
        ex = sum * inv;
        ex2 = sum2 * inv;

        ema = ema.mul_add(one_minus_alpha, alpha * new);

        mse = (-2.0 * ema).mul_add(ex, ema.mul_add(ema, ex2));
        if mse < 0.0 {
            mse = 0.0;
        }
        sd = mse.sqrt() * nbdev;

        *out.get_unchecked_mut(i) = if sd == 0.0 || sd.is_nan() {
            f64::NAN
        } else {
            (new - ema) / sd
        };
        i += 1;
    }

    Ok(ZscoreOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
    zscore_scalar(data, period, first, ma_type, nbdev, devtype)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
    if period <= 32 {
        zscore_avx512_short(data, period, first, ma_type, nbdev, devtype)
    } else {
        zscore_avx512_long(data, period, first, ma_type, nbdev, devtype)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
    zscore_scalar(data, period, first, ma_type, nbdev, devtype)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
    zscore_scalar(data, period, first, ma_type, nbdev, devtype)
}

#[derive(Debug, Clone)]
pub struct ZscoreStream {
    period: usize,
    ma_type: String,
    nbdev: f64,
    devtype: usize,

    buffer: Vec<f64>,
    head: usize,
    filled: bool,

    sum: f64,
    sum2: f64,

    wsum: f64,

    ema: f64,
    ema_inited: bool,

    nan_count: usize,

    inv_period: f64,
    wma_denom: f64,
    inv_wma_denom: f64,
    inv_nbdev: f64,

    kind: MaKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum MaKind {
    Sma,
    Ema,
    Wma,
    Other,
}

impl ZscoreStream {
    pub fn try_new(params: ZscoreParams) -> Result<Self, ZscoreError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ZscoreError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let ma_type = params.ma_type.unwrap_or_else(|| "sma".to_string());
        let nbdev = params.nbdev.unwrap_or(1.0);
        let devtype = params.devtype.unwrap_or(0);

        let kind = match ma_type.to_ascii_lowercase().as_str() {
            "sma" => MaKind::Sma,
            "ema" => MaKind::Ema,
            "wma" => MaKind::Wma,
            _ => MaKind::Other,
        };

        let n = period as f64;
        let wden = n * (n + 1.0) * 0.5;
        let inv_nbdev = if nbdev != 0.0 {
            1.0 / nbdev
        } else {
            f64::INFINITY
        };

        Ok(Self {
            period,
            ma_type,
            nbdev,
            devtype,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,

            sum: 0.0,
            sum2: 0.0,
            wsum: 0.0,

            ema: 0.0,
            ema_inited: false,

            nan_count: period,

            inv_period: 1.0 / n,
            wma_denom: wden,
            inv_wma_denom: 1.0 / wden,
            inv_nbdev,
            kind,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        let just_filled = !self.filled && self.head == 0;
        if just_filled {
            self.filled = true;
        }

        if old.is_nan() {
            self.nan_count = self.nan_count.saturating_sub(1);
        }
        if value.is_nan() {
            self.nan_count += 1;
        }

        let sum_prev = self.sum;

        let old_c = if old.is_nan() { 0.0 } else { old };
        let new_c = if value.is_nan() { 0.0 } else { value };

        self.sum = self.sum + new_c - old_c;
        self.sum2 = self.sum2 + new_c * new_c - old_c * old_c;

        self.wsum = self.wsum - sum_prev + (self.period as f64) * new_c;

        if !self.filled {
            return None;
        }

        if self.devtype == 0 && self.nbdev != 0.0 && self.nan_count == 0 {
            let mean = match self.kind {
                MaKind::Sma => self.sum * self.inv_period,
                MaKind::Ema => {
                    if !self.ema_inited {
                        self.ema = self.sum * self.inv_period;
                        self.ema_inited = true;
                        self.ema
                    } else {
                        let alpha = 2.0 / ((self.period as f64) + 1.0);
                        self.ema = self.ema.mul_add(1.0 - alpha, alpha * new_c);
                        self.ema
                    }
                }
                MaKind::Wma => self.wsum * self.inv_wma_denom,
                MaKind::Other => {
                    return Some(self.compute_zscore_slow());
                }
            };

            let ex = self.sum * self.inv_period;
            let ex2 = self.sum2 * self.inv_period;

            let var = if self.kind == MaKind::Sma {
                let v = ex2 - mean * mean;
                if v < 0.0 { 0.0 } else { v }
            } else {
                let v = ex2 - 2.0 * mean * ex + mean * mean;
                if v < 0.0 { 0.0 } else { v }
            };

            let sd = var.sqrt();
            if sd == 0.0 || !sd.is_finite() {
                return Some(f64::NAN);
            }

            let last_idx = if self.head == 0 {
                self.period - 1
            } else {
                self.head - 1
            };
            let last = self.buffer[last_idx];
            let z = (last - mean) / (sd * self.nbdev);
            return Some(z);
        }

        Some(self.compute_zscore_slow())
    }

    #[inline(always)]
    fn compute_zscore_slow(&self) -> f64 {
        let mut ordered = vec![0.0; self.period];
        let mut idx = self.head;
        for i in 0..self.period {
            ordered[i] = self.buffer[idx];
            idx = (idx + 1) % self.period;
        }

        let means = match ma(&self.ma_type, MaData::Slice(&ordered), self.period) {
            Ok(m) => m,
            Err(_) => return f64::NAN,
        };

        let dev_input = DevInput {
            data: DeviationData::Slice(&ordered),
            params: DevParams {
                period: Some(self.period),
                devtype: Some(self.devtype),
            },
        };
        let mut sigmas = match deviation(&dev_input) {
            Ok(d) => d.values,
            Err(_) => return f64::NAN,
        };
        for s in &mut sigmas {
            *s *= self.nbdev;
        }

        let mean = means[self.period - 1];
        let sigma = sigmas[self.period - 1];
        let value = ordered[self.period - 1];
        if sigma == 0.0 || sigma.is_nan() {
            f64::NAN
        } else {
            (value - mean) / sigma
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZscoreBatchRange {
    pub period: (usize, usize, usize),
    pub ma_type: (String, String, String),
    pub nbdev: (f64, f64, f64),
    pub devtype: (usize, usize, usize),
}

impl Default for ZscoreBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
            ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
            nbdev: (1.0, 1.0, 0.0),
            devtype: (0, 0, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ZscoreBatchBuilder {
    range: ZscoreBatchRange,
    kernel: Kernel,
}

impl ZscoreBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn ma_type_static<T: Into<String>>(mut self, s: T) -> Self {
        let val = s.into();
        self.range.ma_type = (val.clone(), val.clone(), "".to_string());
        self
    }
    #[inline]
    pub fn nbdev_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nbdev = (start, end, step);
        self
    }
    #[inline]
    pub fn nbdev_static(mut self, x: f64) -> Self {
        self.range.nbdev = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.devtype = (start, end, step);
        self
    }
    #[inline]
    pub fn devtype_static(mut self, x: usize) -> Self {
        self.range.devtype = (x, x, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<ZscoreBatchOutput, ZscoreError> {
        zscore_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ZscoreBatchOutput, ZscoreError> {
        ZscoreBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ZscoreBatchOutput, ZscoreError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ZscoreBatchOutput, ZscoreError> {
        ZscoreBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn zscore_batch_with_kernel(
    data: &[f64],
    sweep: &ZscoreBatchRange,
    k: Kernel,
) -> Result<ZscoreBatchOutput, ZscoreError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(ZscoreError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    zscore_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ZscoreBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ZscoreParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ZscoreBatchOutput {
    pub fn row_for_params(&self, p: &ZscoreParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && c.ma_type.as_ref().unwrap_or(&"sma".to_string())
                    == p.ma_type.as_ref().unwrap_or(&"sma".to_string())
                && (c.nbdev.unwrap_or(1.0) - p.nbdev.unwrap_or(1.0)).abs() < 1e-12
                && c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
        })
    }
    pub fn values_for(&self, p: &ZscoreParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ZscoreBatchRange) -> Result<Vec<ZscoreParams>, ZscoreError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, ZscoreError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(next) => {
                        if next == v {
                            break;
                        }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                vals.push(v);
                if v == end {
                    break;
                }
                let next = v.saturating_sub(step);
                if next == v {
                    break;
                }
                v = next;
                if v < end {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(ZscoreError::InvalidRange {
                start: start as f64,
                end: end as f64,
                step: step as f64,
            });
        }
        Ok(vals)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, ZscoreError> {
        if !step.is_finite() {
            return Err(ZscoreError::InvalidRange { start, end, step });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        let tol = 1e-12;
        if start <= end {
            let step_pos = step.abs();
            let mut x = start;
            while x <= end + tol {
                vals.push(x);
                x += step_pos;
                if !x.is_finite() {
                    break;
                }
            }
        } else {
            let step_neg = -step.abs();
            let mut x = start;
            while x >= end - tol {
                vals.push(x);
                x += step_neg;
                if !x.is_finite() {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(ZscoreError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    fn axis_string((start, end, _step): (String, String, String)) -> Vec<String> {
        if start == end {
            vec![start]
        } else {
            vec![start]
        }
    }

    let periods = axis_usize(r.period)?;
    let ma_types = axis_string(r.ma_type.clone());
    let nbdevs = axis_f64(r.nbdev)?;
    let devtypes = axis_usize(r.devtype)?;

    let total = periods
        .len()
        .checked_mul(ma_types.len())
        .and_then(|v| v.checked_mul(nbdevs.len()))
        .and_then(|v| v.checked_mul(devtypes.len()))
        .ok_or(ZscoreError::InvalidRange {
            start: periods.len() as f64,
            end: devtypes.len() as f64,
            step: nbdevs.len() as f64,
        })?;

    let mut out = Vec::with_capacity(total);
    for &p in &periods {
        for mt in &ma_types {
            for &n in &nbdevs {
                for &dt in &devtypes {
                    out.push(ZscoreParams {
                        period: Some(p),
                        ma_type: Some(mt.clone()),
                        nbdev: Some(n),
                        devtype: Some(dt),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn zscore_batch_slice(
    data: &[f64],
    sweep: &ZscoreBatchRange,
    kern: Kernel,
) -> Result<ZscoreBatchOutput, ZscoreError> {
    zscore_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn zscore_batch_par_slice(
    data: &[f64],
    sweep: &ZscoreBatchRange,
    kern: Kernel,
) -> Result<ZscoreBatchOutput, ZscoreError> {
    zscore_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn zscore_batch_inner(
    data: &[f64],
    sweep: &ZscoreBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ZscoreBatchOutput, ZscoreError> {
    if data.is_empty() {
        return Err(ZscoreError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZscoreError::AllValuesNaN)?;
    let cols = data.len();
    let mut max_p = 0usize;
    for prm in &combos {
        let period = prm.period.unwrap();
        if period == 0 || period > cols {
            return Err(ZscoreError::InvalidPeriod {
                period,
                data_len: cols,
            });
        }
        if period > max_p {
            max_p = period;
        }
    }
    if cols - first < max_p {
        return Err(ZscoreError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let rows = combos.len();

    rows.checked_mul(cols).ok_or(ZscoreError::InvalidRange {
        start: rows as f64,
        end: cols as f64,
        step: 0.0,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            first
                .checked_add(p)
                .and_then(|v| v.checked_sub(1))
                .ok_or(ZscoreError::InvalidRange {
                    start: first as f64,
                    end: p as f64,
                    step: 1.0,
                })
        })
        .collect::<Result<_, _>>()?;

    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let mut groups: HashMap<(usize, MaKind), Vec<(usize, f64)>> = HashMap::new();
    let mut fallback_rows: Vec<usize> = Vec::new();
    for (row_idx, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let ma_type = prm.ma_type.as_ref().unwrap();
        let devtype = prm.devtype.unwrap();
        match zscore_fast_batch_kind(ma_type, devtype) {
            Some(kind) => {
                groups
                    .entry((period, kind))
                    .or_default()
                    .push((row_idx, prm.nbdev.unwrap()));
            }
            None => fallback_rows.push(row_idx),
        }
    }

    let prefixes = if groups.keys().any(|(_, kind)| matches!(kind, MaKind::Sma)) {
        Some(build_sma_std_prefixes(data))
    } else {
        None
    };

    let writer = RowWriter {
        ptr: out.as_mut_ptr(),
        cols,
    };

    for ((period, kind), rows_for_period) in groups.into_iter() {
        let warmup_end = first + period - 1;
        let mut base = vec![f64::NAN; cols];

        match kind {
            MaKind::Sma => match kern {
                Kernel::Scalar => {
                    let pre = prefixes.as_ref().expect("prefixes missing for scalar path");
                    zscore_sma_std_from_prefix_scalar(data, period, warmup_end, pre, &mut base);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => unsafe {
                    let pre = prefixes.as_ref().expect("prefixes missing for AVX2 path");
                    zscore_sma_std_from_prefix_avx2(data, period, warmup_end, pre, &mut base);
                },
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => unsafe {
                    let pre = prefixes.as_ref().expect("prefixes missing for AVX512 path");
                    zscore_sma_std_from_prefix_avx512(data, period, warmup_end, pre, &mut base);
                },
                _ => unreachable!(),
            },
            MaKind::Ema => unsafe {
                zscore_row_scalar_classic_ema(data, first, period, 1.0, &mut base);
            },
            _ => unreachable!(),
        }

        let base_ref = &base;

        let write_scalar = |row_idx: usize, nbdev: f64| unsafe {
            writer.with_row(row_idx, |dst| {
                scale_copy_row_scalar(base_ref, warmup_end, nbdev, dst);
            });
        };

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        let write_avx2 = |row_idx: usize, nbdev: f64| unsafe {
            writer.with_row(row_idx, |dst| {
                scale_copy_row_avx2(base_ref, warmup_end, nbdev, dst);
            });
        };

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        let write_avx512 = |row_idx: usize, nbdev: f64| unsafe {
            writer.with_row(row_idx, |dst| {
                scale_copy_row_avx512(base_ref, warmup_end, nbdev, dst);
            });
        };

        let dispatch_write = |row_idx: usize, nbdev: f64| match kern {
            Kernel::Scalar => write_scalar(row_idx, nbdev),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => write_avx2(row_idx, nbdev),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => write_avx512(row_idx, nbdev),
            _ => unreachable!(),
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                rows_for_period
                    .par_iter()
                    .for_each(|(row_idx, nb)| dispatch_write(*row_idx, *nb));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (row_idx, nb) in rows_for_period.iter() {
                    dispatch_write(*row_idx, *nb);
                }
            }
        } else {
            for (row_idx, nb) in rows_for_period.iter() {
                dispatch_write(*row_idx, *nb);
            }
        }
    }

    if !fallback_rows.is_empty() {
        let do_row = |row: usize| unsafe {
            let prm = &combos[row];
            let period = prm.period.unwrap();
            let ma_type = prm.ma_type.as_ref().unwrap();
            let nbdev = prm.nbdev.unwrap();
            let devtype = prm.devtype.unwrap();
            writer.with_row(row, |dst| match kern {
                Kernel::Scalar => {
                    zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, dst)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => zscore_row_avx2(data, first, period, ma_type, nbdev, devtype, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => {
                    zscore_row_avx512(data, first, period, ma_type, nbdev, devtype, dst)
                }
                _ => unreachable!(),
            });
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                fallback_rows.par_iter().for_each(|&row| do_row(row));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for &row in &fallback_rows {
                    do_row(row);
                }
            }
        } else {
            for &row in &fallback_rows {
                do_row(row);
            }
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(ZscoreBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn zscore_fast_batch_kind(ma_type: &str, devtype: usize) -> Option<MaKind> {
    if devtype != 0 {
        None
    } else if ma_type.eq_ignore_ascii_case("sma") {
        Some(MaKind::Sma)
    } else if ma_type.eq_ignore_ascii_case("ema") {
        Some(MaKind::Ema)
    } else {
        None
    }
}

#[inline(always)]
unsafe fn zscore_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) {
    if devtype == 0 {
        if ma_type == "sma" {
            zscore_row_scalar_classic_sma(data, first, period, nbdev, out);
            return;
        } else if ma_type == "ema" {
            zscore_row_scalar_classic_ema(data, first, period, nbdev, out);
            return;
        }
    }

    let means = match ma(ma_type, MaData::Slice(data), period) {
        Ok(m) => m,
        Err(_) => {
            out.fill(f64::NAN);
            return;
        }
    };
    let dev_input = DevInput {
        data: DeviationData::Slice(data),
        params: DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    };
    let mut sigmas = match deviation(&dev_input) {
        Ok(d) => d.values,
        Err(_) => {
            out.fill(f64::NAN);
            return;
        }
    };
    for v in &mut sigmas {
        *v *= nbdev;
    }
    let warmup_end = first + period - 1;
    for i in warmup_end..data.len() {
        let mean = means[i];
        let sigma = sigmas[i];
        let value = data[i];
        out[i] = if sigma == 0.0 || sigma.is_nan() {
            f64::NAN
        } else {
            (value - mean) / sigma
        };
    }
}

#[inline(always)]
unsafe fn zscore_row_scalar_classic_sma(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    let warmup_end = first + period - 1;

    let mut sum = 0.0;
    let mut sum_sqr = 0.0;
    for j in first..=warmup_end {
        let val = data[j];
        sum += val;
        sum_sqr += val * val;
    }

    let mut mean = sum / period as f64;
    let mut variance = (sum_sqr / period as f64) - (mean * mean);
    let mut stddev = if variance <= 0.0 {
        0.0
    } else {
        variance.sqrt() * nbdev
    };

    out[warmup_end] = if stddev == 0.0 || stddev.is_nan() {
        f64::NAN
    } else {
        (data[warmup_end] - mean) / stddev
    };

    for i in warmup_end + 1..data.len() {
        let old_val = data[i - period];
        let new_val = data[i];

        sum += new_val - old_val;
        sum_sqr += new_val * new_val - old_val * old_val;

        mean = sum / period as f64;
        variance = (sum_sqr / period as f64) - (mean * mean);
        stddev = if variance <= 0.0 {
            0.0
        } else {
            variance.sqrt() * nbdev
        };

        out[i] = if stddev == 0.0 || stddev.is_nan() {
            f64::NAN
        } else {
            (new_val - mean) / stddev
        };
    }
}

#[derive(Clone, Debug)]
struct SmaStdPrefixes {
    ps: Vec<f64>,
    ps2: Vec<f64>,
    pnan: Vec<i32>,
}

#[inline]
fn build_sma_std_prefixes(data: &[f64]) -> SmaStdPrefixes {
    let n = data.len();
    let mut ps = vec![0.0f64; n + 1];
    let mut ps2 = vec![0.0f64; n + 1];
    let mut pnan = vec![0i32; n + 1];

    for i in 0..n {
        let v = data[i];
        if v.is_nan() {
            ps[i + 1] = ps[i];
            ps2[i + 1] = ps2[i];
            pnan[i + 1] = pnan[i] + 1;
        } else {
            ps[i + 1] = ps[i] + v;
            ps2[i + 1] = ps2[i] + v * v;
            pnan[i + 1] = pnan[i];
        }
    }

    SmaStdPrefixes { ps, ps2, pnan }
}

#[inline]
fn zscore_sma_std_from_prefix_scalar(
    data: &[f64],
    period: usize,
    warmup_end: usize,
    pre: &SmaStdPrefixes,
    base_out: &mut [f64],
) {
    let n = data.len();
    debug_assert_eq!(base_out.len(), n);

    for v in &mut base_out[..warmup_end] {
        *v = f64::NAN;
    }
    if n <= warmup_end {
        return;
    }

    let denom = period as f64;
    for i in warmup_end..n {
        let nan_count = pre.pnan[i + 1] - pre.pnan[i + 1 - period];
        if nan_count > 0 {
            base_out[i] = f64::NAN;
            continue;
        }

        let sum = pre.ps[i + 1] - pre.ps[i + 1 - period];
        let sum2 = pre.ps2[i + 1] - pre.ps2[i + 1 - period];
        let mean = sum / denom;
        let variance = sum2 / denom - mean * mean;
        let stdv = if variance <= 0.0 {
            0.0
        } else {
            variance.sqrt()
        };
        base_out[i] = if stdv == 0.0 || stdv.is_nan() {
            f64::NAN
        } else {
            (data[i] - mean) / stdv
        };
    }
}

#[inline]
fn scale_copy_row_scalar(src_base: &[f64], warmup_end: usize, nbdev: f64, dst: &mut [f64]) {
    debug_assert_eq!(src_base.len(), dst.len());

    if warmup_end > 0 {
        dst[..warmup_end].copy_from_slice(&src_base[..warmup_end]);
    }

    if dst.len() <= warmup_end {
        return;
    }

    if nbdev == 0.0 {
        for v in &mut dst[warmup_end..] {
            *v = f64::NAN;
        }
        return;
    }

    for (d, s) in dst[warmup_end..].iter_mut().zip(&src_base[warmup_end..]) {
        *d = *s / nbdev;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn zscore_sma_std_from_prefix_avx2(
    data: &[f64],
    period: usize,
    warmup_end: usize,
    pre: &SmaStdPrefixes,
    base_out: &mut [f64],
) {
    let n = data.len();
    debug_assert_eq!(base_out.len(), n);

    for v in &mut base_out[..warmup_end] {
        *v = f64::NAN;
    }
    if n <= warmup_end {
        return;
    }

    let den = _mm256_set1_pd(period as f64);
    let zero = _mm256_set1_pd(0.0);
    let nanv = _mm256_set1_pd(f64::NAN);

    let mut i = warmup_end;
    while i + 4 <= n {
        let s_hi = _mm256_loadu_pd(pre.ps.as_ptr().add(i + 1));
        let s_lo = _mm256_loadu_pd(pre.ps.as_ptr().add(i + 1 - period));
        let sum = _mm256_sub_pd(s_hi, s_lo);

        let q_hi = _mm256_loadu_pd(pre.ps2.as_ptr().add(i + 1));
        let q_lo = _mm256_loadu_pd(pre.ps2.as_ptr().add(i + 1 - period));
        let sum2 = _mm256_sub_pd(q_hi, q_lo);

        let mean = _mm256_div_pd(sum, den);
        let var = _mm256_sub_pd(_mm256_div_pd(sum2, den), _mm256_mul_pd(mean, mean));
        let var_nz = _mm256_max_pd(var, zero);
        let stdv = _mm256_sqrt_pd(var_nz);

        let x = _mm256_loadu_pd(data.as_ptr().add(i));
        let z = _mm256_div_pd(_mm256_sub_pd(x, mean), stdv);

        let m_std0 = _mm256_cmp_pd(stdv, zero, _CMP_EQ_OQ);
        let m_stdnan = _mm256_cmp_pd(stdv, stdv, _CMP_UNORD_Q);

        let cur = _mm_loadu_si128(pre.pnan.as_ptr().add(i + 1) as *const _);
        let prev = _mm_loadu_si128(pre.pnan.as_ptr().add(i + 1 - period) as *const _);
        let diff = _mm_sub_epi32(cur, prev);
        let diff_pd = _mm256_cvtepi32_pd(diff);
        let m_hasnan = _mm256_cmp_pd(diff_pd, zero, _CMP_GT_OQ);

        let mask = _mm256_or_pd(_mm256_or_pd(m_std0, m_stdnan), m_hasnan);
        let res = _mm256_blendv_pd(z, nanv, mask);
        _mm256_storeu_pd(base_out.as_mut_ptr().add(i), res);

        i += 4;
    }

    let den_s = period as f64;
    while i < n {
        let count = pre.pnan[i + 1] - pre.pnan[i + 1 - period];
        if count > 0 {
            base_out[i] = f64::NAN;
        } else {
            let sum = pre.ps[i + 1] - pre.ps[i + 1 - period];
            let sum2 = pre.ps2[i + 1] - pre.ps2[i + 1 - period];
            let mean = sum / den_s;
            let variance = sum2 / den_s - mean * mean;
            let sd = if variance <= 0.0 {
                0.0
            } else {
                variance.sqrt()
            };
            base_out[i] = if sd == 0.0 || sd.is_nan() {
                f64::NAN
            } else {
                (data[i] - mean) / sd
            };
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zscore_sma_std_from_prefix_avx512(
    data: &[f64],
    period: usize,
    warmup_end: usize,
    pre: &SmaStdPrefixes,
    base_out: &mut [f64],
) {
    let n = data.len();
    debug_assert_eq!(base_out.len(), n);

    for v in &mut base_out[..warmup_end] {
        *v = f64::NAN;
    }
    if n <= warmup_end {
        return;
    }

    let den = _mm512_set1_pd(period as f64);
    let zero = _mm512_set1_pd(0.0);
    let nanv = _mm512_set1_pd(f64::NAN);

    let mut i = warmup_end;
    while i + 8 <= n {
        let s_hi = _mm512_loadu_pd(pre.ps.as_ptr().add(i + 1));
        let s_lo = _mm512_loadu_pd(pre.ps.as_ptr().add(i + 1 - period));
        let sum = _mm512_sub_pd(s_hi, s_lo);

        let q_hi = _mm512_loadu_pd(pre.ps2.as_ptr().add(i + 1));
        let q_lo = _mm512_loadu_pd(pre.ps2.as_ptr().add(i + 1 - period));
        let sum2 = _mm512_sub_pd(q_hi, q_lo);

        let mean = _mm512_div_pd(sum, den);
        let var = _mm512_sub_pd(_mm512_div_pd(sum2, den), _mm512_mul_pd(mean, mean));
        let var_nz = _mm512_max_pd(var, zero);
        let stdv = _mm512_sqrt_pd(var_nz);

        let x = _mm512_loadu_pd(data.as_ptr().add(i));
        let z = _mm512_div_pd(_mm512_sub_pd(x, mean), stdv);

        let k_std0 = _mm512_cmp_pd_mask(stdv, zero, _CMP_EQ_OQ);
        let k_stdnan = _mm512_cmp_pd_mask(stdv, stdv, _CMP_UNORD_Q);

        let cur_i = _mm256_loadu_si256(pre.pnan.as_ptr().add(i + 1) as *const _);
        let prev_i = _mm256_loadu_si256(pre.pnan.as_ptr().add(i + 1 - period) as *const _);
        let diff_i = _mm256_sub_epi32(cur_i, prev_i);
        let diff_pd = _mm512_cvtepi32_pd(diff_i);
        let k_hasnan = _mm512_cmp_pd_mask(diff_pd, zero, _CMP_GT_OQ);

        let k_bad = k_std0 | k_stdnan | k_hasnan;
        let res = _mm512_mask_mov_pd(z, k_bad, nanv);
        _mm512_storeu_pd(base_out.as_mut_ptr().add(i), res);

        i += 8;
    }

    let den_s = period as f64;
    while i < n {
        let count = pre.pnan[i + 1] - pre.pnan[i + 1 - period];
        if count > 0 {
            base_out[i] = f64::NAN;
        } else {
            let sum = pre.ps[i + 1] - pre.ps[i + 1 - period];
            let sum2 = pre.ps2[i + 1] - pre.ps2[i + 1 - period];
            let mean = sum / den_s;
            let variance = sum2 / den_s - mean * mean;
            let sd = if variance <= 0.0 {
                0.0
            } else {
                variance.sqrt()
            };
            base_out[i] = if sd == 0.0 || sd.is_nan() {
                f64::NAN
            } else {
                (data[i] - mean) / sd
            };
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn scale_copy_row_avx2(src_base: &[f64], warmup_end: usize, nbdev: f64, dst: &mut [f64]) {
    debug_assert_eq!(src_base.len(), dst.len());

    if warmup_end > 0 {
        dst[..warmup_end].copy_from_slice(&src_base[..warmup_end]);
    }

    if dst.len() <= warmup_end {
        return;
    }

    if nbdev == 0.0 {
        for v in &mut dst[warmup_end..] {
            *v = f64::NAN;
        }
        return;
    }

    let inv = _mm256_set1_pd(1.0 / nbdev);
    let mut i = warmup_end;
    while i + 4 <= dst.len() {
        let v = _mm256_loadu_pd(src_base.as_ptr().add(i));
        let y = _mm256_mul_pd(v, inv);
        _mm256_storeu_pd(dst.as_mut_ptr().add(i), y);
        i += 4;
    }
    while i < dst.len() {
        dst[i] = src_base[i] / nbdev;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn scale_copy_row_avx512(src_base: &[f64], warmup_end: usize, nbdev: f64, dst: &mut [f64]) {
    debug_assert_eq!(src_base.len(), dst.len());

    if warmup_end > 0 {
        dst[..warmup_end].copy_from_slice(&src_base[..warmup_end]);
    }

    if dst.len() <= warmup_end {
        return;
    }

    if nbdev == 0.0 {
        for v in &mut dst[warmup_end..] {
            *v = f64::NAN;
        }
        return;
    }

    let inv = _mm512_set1_pd(1.0 / nbdev);
    let mut i = warmup_end;
    while i + 8 <= dst.len() {
        let v = _mm512_loadu_pd(src_base.as_ptr().add(i));
        let y = _mm512_mul_pd(v, inv);
        _mm512_storeu_pd(dst.as_mut_ptr().add(i), y);
        i += 8;
    }
    while i < dst.len() {
        dst[i] = src_base[i] / nbdev;
        i += 1;
    }
}

#[derive(Clone, Copy)]
struct RowWriter {
    ptr: *mut f64,
    cols: usize,
}

unsafe impl Send for RowWriter {}
unsafe impl Sync for RowWriter {}

impl RowWriter {
    #[inline(always)]
    unsafe fn with_row<F>(&self, row: usize, mut f: F)
    where
        F: FnOnce(&mut [f64]),
    {
        let slice = std::slice::from_raw_parts_mut(self.ptr.add(row * self.cols), self.cols);
        f(slice);
    }
}

#[inline(always)]
unsafe fn zscore_row_scalar_classic_ema(
    data: &[f64],
    first: usize,
    period: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    let n = data.len();
    let warmup_end = first + period - 1;

    if n <= warmup_end {
        return;
    }

    let den = period as f64;
    let alpha = 2.0 / (den + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    let mut sum = 0.0;
    let mut sum2 = 0.0;
    {
        let mut j = first;
        while j <= warmup_end {
            let v = data[j];
            sum += v;
            sum2 += v * v;
            j += 1;
        }
    }
    let mut ema = sum / den;

    let mut mse = (sum2 / den) - 2.0 * ema * (sum / den) + ema * ema;
    if mse < 0.0 {
        mse = 0.0;
    }
    let mut sd = mse.sqrt() * nbdev;

    out[warmup_end] = if sd == 0.0 || sd.is_nan() {
        f64::NAN
    } else {
        (data[warmup_end] - ema) / sd
    };

    let mut i = warmup_end + 1;
    while i < n {
        let new = data[i];
        let old = data[i - period];

        sum += new - old;
        sum2 += new * new - old * old;

        ema = alpha * new + one_minus_alpha * ema;

        mse = (sum2 / den) - 2.0 * ema * (sum / den) + ema * ema;
        if mse < 0.0 {
            mse = 0.0;
        }
        sd = mse.sqrt() * nbdev;

        out[i] = if sd == 0.0 || sd.is_nan() {
            f64::NAN
        } else {
            (new - ema) / sd
        };
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) {
    zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        zscore_row_avx512_short(data, first, period, ma_type, nbdev, devtype, out)
    } else {
        zscore_row_avx512_long(data, first, period, ma_type, nbdev, devtype, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) {
    zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) {
    zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out)
}

#[inline(always)]
pub fn zscore_batch_inner_into(
    data: &[f64],
    sweep: &ZscoreBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ZscoreParams>, ZscoreError> {
    if data.is_empty() {
        return Err(ZscoreError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZscoreError::AllValuesNaN)?;
    let cols = data.len();
    let mut max_p = 0usize;
    for prm in &combos {
        let period = prm.period.unwrap();
        if period == 0 || period > cols {
            return Err(ZscoreError::InvalidPeriod {
                period,
                data_len: cols,
            });
        }
        if period > max_p {
            max_p = period;
        }
    }
    if cols - first < max_p {
        return Err(ZscoreError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let rows = combos.len();

    let expected = rows.checked_mul(cols).ok_or(ZscoreError::InvalidRange {
        start: rows as f64,
        end: cols as f64,
        step: 0.0,
    })?;
    if out.len() != expected {
        return Err(ZscoreError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    {
        let out_uninit = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                out.len(),
            )
        };
        init_matrix_prefixes(out_uninit, cols, &warm);
    }

    let mut groups: HashMap<(usize, MaKind), Vec<(usize, f64)>> = HashMap::new();
    let mut fallback_rows: Vec<usize> = Vec::new();
    for (row_idx, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let ma_type = prm.ma_type.as_ref().unwrap();
        let devtype = prm.devtype.unwrap();
        match zscore_fast_batch_kind(ma_type, devtype) {
            Some(kind) => {
                groups
                    .entry((period, kind))
                    .or_default()
                    .push((row_idx, prm.nbdev.unwrap()));
            }
            None => fallback_rows.push(row_idx),
        }
    }

    let prefixes = if groups.keys().any(|(_, kind)| matches!(kind, MaKind::Sma)) {
        Some(build_sma_std_prefixes(data))
    } else {
        None
    };

    let writer = RowWriter {
        ptr: out.as_mut_ptr(),
        cols,
    };

    for ((period, kind), rows_for_period) in groups.into_iter() {
        let warmup_end = first + period - 1;
        let mut base = vec![f64::NAN; cols];

        match kind {
            MaKind::Sma => match kern {
                Kernel::Scalar => {
                    let pre = prefixes.as_ref().expect("prefixes missing for scalar path");
                    zscore_sma_std_from_prefix_scalar(data, period, warmup_end, pre, &mut base);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => unsafe {
                    let pre = prefixes.as_ref().expect("prefixes missing for AVX2 path");
                    zscore_sma_std_from_prefix_avx2(data, period, warmup_end, pre, &mut base);
                },
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => unsafe {
                    let pre = prefixes.as_ref().expect("prefixes missing for AVX512 path");
                    zscore_sma_std_from_prefix_avx512(data, period, warmup_end, pre, &mut base);
                },
                _ => unreachable!(),
            },
            MaKind::Ema => unsafe {
                zscore_row_scalar_classic_ema(data, first, period, 1.0, &mut base);
            },
            _ => unreachable!(),
        }

        let base_ref = &base;

        let write_scalar = |row_idx: usize, nbdev: f64| unsafe {
            writer.with_row(row_idx, |dst| {
                scale_copy_row_scalar(base_ref, warmup_end, nbdev, dst);
            });
        };

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        let write_avx2 = |row_idx: usize, nbdev: f64| unsafe {
            writer.with_row(row_idx, |dst| {
                scale_copy_row_avx2(base_ref, warmup_end, nbdev, dst);
            });
        };

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        let write_avx512 = |row_idx: usize, nbdev: f64| unsafe {
            writer.with_row(row_idx, |dst| {
                scale_copy_row_avx512(base_ref, warmup_end, nbdev, dst);
            });
        };

        let dispatch_write = |row_idx: usize, nbdev: f64| match kern {
            Kernel::Scalar => write_scalar(row_idx, nbdev),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => write_avx2(row_idx, nbdev),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => write_avx512(row_idx, nbdev),
            _ => unreachable!(),
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                rows_for_period
                    .par_iter()
                    .for_each(|(row_idx, nb)| dispatch_write(*row_idx, *nb));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (row_idx, nb) in rows_for_period.iter() {
                    dispatch_write(*row_idx, *nb);
                }
            }
        } else {
            for (row_idx, nb) in rows_for_period.iter() {
                dispatch_write(*row_idx, *nb);
            }
        }
    }

    if !fallback_rows.is_empty() {
        let do_row = |row: usize| unsafe {
            let prm = &combos[row];
            let period = prm.period.unwrap();
            let ma_type = prm.ma_type.as_ref().unwrap();
            let nbdev = prm.nbdev.unwrap();
            let devtype = prm.devtype.unwrap();
            writer.with_row(row, |dst| match kern {
                Kernel::Scalar => {
                    zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, dst)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => zscore_row_avx2(data, first, period, ma_type, nbdev, devtype, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => {
                    zscore_row_avx512(data, first, period, ma_type, nbdev, devtype, dst)
                }
                _ => unreachable!(),
            });
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                fallback_rows.par_iter().for_each(|&row| do_row(row));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for &row in &fallback_rows {
                    do_row(row);
                }
            }
        } else {
            for &row in &fallback_rows {
                do_row(row);
            }
        }
    }

    Ok(combos)
}

pub fn zscore_into_slice(
    dst: &mut [f64],
    input: &ZscoreInput,
    kern: Kernel,
) -> Result<(), ZscoreError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(ZscoreError::EmptyInputData);
    }
    if dst.len() != data.len() {
        return Err(ZscoreError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ZscoreError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ZscoreError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(ZscoreError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let ma_type = input.get_ma_type();
    let nbdev = input.get_nbdev();
    let devtype = input.get_devtype();

    let chosen = match kern {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                zscore_compute_into_scalar(data, period, first, &ma_type, nbdev, devtype, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                zscore_compute_into_avx2(data, period, first, &ma_type, nbdev, devtype, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                zscore_compute_into_avx512(data, period, first, &ma_type, nbdev, devtype, dst)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                zscore_compute_into_scalar(data, period, first, &ma_type, nbdev, devtype, dst)
            }
            _ => {
                return Err(ZscoreError::InvalidPeriod {
                    period: 0,
                    data_len: 0,
                });
            }
        }
    }?;

    Ok(())
}

pub fn zscore_into(input: &ZscoreInput, out: &mut [f64]) -> Result<(), ZscoreError> {
    zscore_into_slice(out, input, Kernel::Auto)
}

#[inline]
unsafe fn zscore_compute_into_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) -> Result<(), ZscoreError> {
    let warmup_end = first + period - 1;
    for v in &mut out[..warmup_end] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    if data.len() <= warmup_end {
        return Ok(());
    }

    if devtype == 0 {
        if ma_type == "sma" {
            let inv = 1.0 / (period as f64);
            let mut sum = 0.0f64;
            let mut sum_sqr = 0.0f64;
            {
                let mut j = first;
                while j <= warmup_end {
                    let v = *data.get_unchecked(j);
                    sum += v;
                    sum_sqr = v.mul_add(v, sum_sqr);
                    j += 1;
                }
            }
            let mut mean = sum * inv;
            let mut variance = (-mean).mul_add(mean, sum_sqr * inv);
            if variance < 0.0 {
                variance = 0.0;
            }
            let mut sd = if variance == 0.0 {
                0.0
            } else {
                variance.sqrt() * nbdev
            };

            let xw = *data.get_unchecked(warmup_end);
            *out.get_unchecked_mut(warmup_end) = if sd == 0.0 || sd.is_nan() {
                f64::NAN
            } else {
                (xw - mean) / sd
            };

            let n = data.len();
            let mut i = warmup_end + 1;
            while i < n {
                let old_val = *data.get_unchecked(i - period);
                let new_val = *data.get_unchecked(i);
                let dd = new_val - old_val;
                sum += dd;
                sum_sqr = (new_val + old_val).mul_add(dd, sum_sqr);
                mean = sum * inv;

                variance = (-mean).mul_add(mean, sum_sqr * inv);
                if variance < 0.0 {
                    variance = 0.0;
                }
                sd = if variance == 0.0 {
                    0.0
                } else {
                    variance.sqrt() * nbdev
                };

                *out.get_unchecked_mut(i) = if sd == 0.0 || sd.is_nan() {
                    f64::NAN
                } else {
                    (new_val - mean) / sd
                };
                i += 1;
            }

            return Ok(());
        }

        if ma_type == "ema" {
            let den = period as f64;
            let inv = 1.0 / den;
            let alpha = 2.0 / (den + 1.0);
            let one_minus_alpha = 1.0 - alpha;

            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            {
                let mut j = first;
                while j <= warmup_end {
                    let v = *data.get_unchecked(j);
                    sum += v;
                    sum2 = v.mul_add(v, sum2);
                    j += 1;
                }
            }
            let mut ema = sum * inv;

            let mut ex = sum * inv;
            let mut ex2 = sum2 * inv;
            let mut mse = (-2.0 * ema).mul_add(ex, ema.mul_add(ema, ex2));
            if mse < 0.0 {
                mse = 0.0;
            }
            let mut sd = mse.sqrt() * nbdev;

            let xw = *data.get_unchecked(warmup_end);
            *out.get_unchecked_mut(warmup_end) = if sd == 0.0 || sd.is_nan() {
                f64::NAN
            } else {
                (xw - ema) / sd
            };

            let n = data.len();
            let mut i = warmup_end + 1;
            while i < n {
                let new = *data.get_unchecked(i);
                let old = *data.get_unchecked(i - period);

                let dd = new - old;
                sum += dd;
                sum2 = (new + old).mul_add(dd, sum2);
                ex = sum * inv;
                ex2 = sum2 * inv;

                ema = ema.mul_add(one_minus_alpha, alpha * new);

                mse = (-2.0 * ema).mul_add(ex, ema.mul_add(ema, ex2));
                if mse < 0.0 {
                    mse = 0.0;
                }
                sd = mse.sqrt() * nbdev;

                *out.get_unchecked_mut(i) = if sd == 0.0 || sd.is_nan() {
                    f64::NAN
                } else {
                    (new - ema) / sd
                };
                i += 1;
            }

            return Ok(());
        }
    }

    let means = ma(ma_type, MaData::Slice(data), period)
        .map_err(|e| ZscoreError::MaError(e.to_string()))?;
    let dev_input = DevInput {
        data: DeviationData::Slice(data),
        params: DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    };
    let mut sigmas = deviation(&dev_input)?.values;
    for v in &mut sigmas {
        *v *= nbdev;
    }

    for i in warmup_end..data.len() {
        let mean = means[i];
        let sigma = sigmas[i];
        let value = data[i];
        out[i] = if sigma == 0.0 || sigma.is_nan() {
            f64::NAN
        } else {
            (value - mean) / sigma
        };
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn zscore_compute_into_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) -> Result<(), ZscoreError> {
    zscore_compute_into_scalar(data, period, first, ma_type, nbdev, devtype, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn zscore_compute_into_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    ma_type: &str,
    nbdev: f64,
    devtype: usize,
    out: &mut [f64],
) -> Result<(), ZscoreError> {
    zscore_compute_into_scalar(data, period, first, ma_type, nbdev, devtype, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    fn check_zscore_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = ZscoreParams {
            period: None,
            ma_type: None,
            nbdev: None,
            devtype: None,
        };
        let input = ZscoreInput::from_candles(&candles, "close", default_params);
        let output = zscore_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_zscore_with_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ZscoreParams {
            period: Some(0),
            ma_type: None,
            nbdev: None,
            devtype: None,
        };
        let input = ZscoreInput::from_slice(&input_data, params);
        let res = zscore_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Zscore should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_zscore_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ZscoreParams {
            period: Some(10),
            ma_type: None,
            nbdev: None,
            devtype: None,
        };
        let input = ZscoreInput::from_slice(&data_small, params);
        let res = zscore_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Zscore should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_zscore_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ZscoreParams {
            period: Some(14),
            ma_type: None,
            nbdev: None,
            devtype: None,
        };
        let input = ZscoreInput::from_slice(&single_point, params);
        let res = zscore_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Zscore should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_zscore_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = ZscoreParams::default();
        let input = ZscoreInput::from_slice(&input_data, params);
        let res = zscore_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Zscore should fail when all values are NaN",
            test_name
        );
        Ok(())
    }
    fn check_zscore_input_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ZscoreInput::with_default_candles(&candles);
        match input.data {
            ZscoreData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ZscoreData::Candles"),
        }
        Ok(())
    }
    fn check_zscore_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = ZscoreInput::from_candles(&candles, "close", ZscoreParams::default());
        let result = zscore_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -0.3040683926967643,
            -0.41042159719064014,
            -0.5411993612192193,
            -0.1673226261513698,
            -1.431635486349618,
        ];
        let start = result.values.len().saturating_sub(5);

        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Zscore {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    macro_rules! generate_all_zscore_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }
    #[cfg(debug_assertions)]
    fn check_zscore_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            ZscoreParams::default(),
            ZscoreParams {
                period: Some(2),
                ma_type: Some("sma".to_string()),
                nbdev: Some(1.0),
                devtype: Some(0),
            },
            ZscoreParams {
                period: Some(5),
                ma_type: Some("ema".to_string()),
                nbdev: Some(1.0),
                devtype: Some(0),
            },
            ZscoreParams {
                period: Some(10),
                ma_type: Some("wma".to_string()),
                nbdev: Some(2.0),
                devtype: Some(0),
            },
            ZscoreParams {
                period: Some(20),
                ma_type: Some("sma".to_string()),
                nbdev: Some(1.5),
                devtype: Some(1),
            },
            ZscoreParams {
                period: Some(30),
                ma_type: Some("ema".to_string()),
                nbdev: Some(2.5),
                devtype: Some(2),
            },
            ZscoreParams {
                period: Some(50),
                ma_type: Some("wma".to_string()),
                nbdev: Some(3.0),
                devtype: Some(0),
            },
            ZscoreParams {
                period: Some(100),
                ma_type: Some("sma".to_string()),
                nbdev: Some(1.0),
                devtype: Some(1),
            },
            ZscoreParams {
                period: Some(14),
                ma_type: Some("ema".to_string()),
                nbdev: Some(0.5),
                devtype: Some(0),
            },
            ZscoreParams {
                period: Some(14),
                ma_type: Some("sma".to_string()),
                nbdev: Some(0.1),
                devtype: Some(2),
            },
            ZscoreParams {
                period: Some(25),
                ma_type: Some("wma".to_string()),
                nbdev: Some(4.0),
                devtype: Some(1),
            },
            ZscoreParams {
                period: Some(7),
                ma_type: Some("ema".to_string()),
                nbdev: Some(1.618),
                devtype: Some(0),
            },
            ZscoreParams {
                period: Some(21),
                ma_type: Some("sma".to_string()),
                nbdev: Some(2.718),
                devtype: Some(1),
            },
            ZscoreParams {
                period: Some(42),
                ma_type: Some("wma".to_string()),
                nbdev: Some(3.14159),
                devtype: Some(2),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = ZscoreInput::from_candles(&candles, "close", params.clone());
            let output = zscore_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, ma_type={}, nbdev={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        params.nbdev.unwrap_or(1.0),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, ma_type={}, nbdev={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        params.nbdev.unwrap_or(1.0),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, ma_type={}, nbdev={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.ma_type.as_deref().unwrap_or("sma"),
                        params.nbdev.unwrap_or(1.0),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_zscore_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_zscore_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e5f64..1e5f64).prop_filter("finite", |x| x.is_finite()),
                    period + 10..400,
                ),
                Just(period),
                prop::sample::select(vec!["sma", "ema", "wma"]),
                0.5f64..3.0f64,
                0usize..=2,
            )
        });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, period, ma_type, nbdev, devtype)| {
                let params = ZscoreParams {
                    period: Some(period),
                    ma_type: Some(ma_type.to_string()),
                    nbdev: Some(nbdev),
                    devtype: Some(devtype),
                };
                let input = ZscoreInput::from_slice(&data, params.clone());

                let ZscoreOutput { values: out } = zscore_with_kernel(&input, kernel)?;

                let ZscoreOutput { values: ref_out } = zscore_with_kernel(&input, Kernel::Scalar)?;

                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                for i in 0..(period - 1) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                for i in (period - 1)..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/infinite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff = y_bits.abs_diff(r_bits);

                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at index {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON) {
                    for i in (period - 1)..data.len() {
                        prop_assert!(
                            out[i].is_nan() || devtype != 0,
                            "Expected NaN for constant data with stddev at index {}, got {}",
                            i,
                            out[i]
                        );
                    }
                }

                if period == 2 && devtype == 0 && ma_type == "sma" {
                    for i in 1..data.len() {
                        if out[i].is_finite() {
                            let mean = (data[i - 1] + data[i]) / 2.0;
                            let diff1 = (data[i - 1] - mean).powi(2);
                            let diff2 = (data[i] - mean).powi(2);
                            let variance = (diff1 + diff2) / 2.0;
                            let stddev = variance.sqrt();

                            if stddev > f64::EPSILON {
                                let expected = (data[i] - mean) / (stddev * nbdev);
                                prop_assert!(
                                    (out[i] - expected).abs() <= 1e-6,
                                    "Zscore calculation mismatch at index {}: {} vs expected {}",
                                    i,
                                    out[i],
                                    expected
                                );
                            }
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    generate_all_zscore_tests!(
        check_zscore_partial_params,
        check_zscore_with_zero_period,
        check_zscore_period_exceeds_length,
        check_zscore_very_small_dataset,
        check_zscore_all_nan,
        check_zscore_input_with_default_candles,
        check_zscore_accuracy,
        check_zscore_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_zscore_tests!(check_zscore_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = ZscoreBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = ZscoreParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 10, 2, 0.5, 2.0, 0.5, 0),
            (5, 25, 5, 1.0, 3.0, 1.0, 1),
            (10, 50, 10, 1.5, 3.5, 1.0, 2),
            (2, 5, 1, 0.1, 1.0, 0.3, 0),
            (14, 14, 0, 1.0, 4.0, 0.5, 0),
            (20, 40, 10, 2.0, 2.0, 0.0, 1),
        ];

        let ma_types = vec!["sma", "ema", "wma"];

        for (
            cfg_idx,
            &(period_start, period_end, period_step, nbdev_start, nbdev_end, nbdev_step, devtype),
        ) in test_configs.iter().enumerate()
        {
            for ma_type in &ma_types {
                let mut builder = ZscoreBatchBuilder::new().kernel(kernel);

                if period_step > 0 {
                    builder = builder.period_range(period_start, period_end, period_step);
                } else {
                    builder = builder.period_static(period_start);
                }

                if nbdev_step > 0.0 {
                    builder = builder.nbdev_range(nbdev_start, nbdev_end, nbdev_step);
                } else {
                    builder = builder.nbdev_static(nbdev_start);
                }

                builder = builder.ma_type_static(ma_type.to_string());

                builder = builder.devtype_static(devtype);

                let output = builder.apply_candles(&c, "close")?;

                for (idx, &val) in output.values.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();
                    let row = idx / output.cols;
                    let col = idx % output.cols;
                    let combo = &output.combos[row];

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Config {} (MA: {}): Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
                            test,
                            cfg_idx,
                            ma_type,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            combo.period.unwrap_or(14),
                            combo.ma_type.as_deref().unwrap_or("sma"),
                            combo.nbdev.unwrap_or(1.0),
                            combo.devtype.unwrap_or(0)
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Config {} (MA: {}): Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
                            test,
                            cfg_idx,
                            ma_type,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            combo.period.unwrap_or(14),
                            combo.ma_type.as_deref().unwrap_or("sma"),
                            combo.nbdev.unwrap_or(1.0),
                            combo.devtype.unwrap_or(0)
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Config {} (MA: {}): Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
                            test,
                            cfg_idx,
                            ma_type,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            combo.period.unwrap_or(14),
                            combo.ma_type.as_deref().unwrap_or("sma"),
                            combo.nbdev.unwrap_or(1.0),
                            combo.devtype.unwrap_or(0)
                        );
                    }
                }
            }
        }

        let devtype_test = ZscoreBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
            .nbdev_static(2.0)
            .ma_type_static("ema")
            .devtype_range(0, 2, 1)
            .apply_candles(&c, "close")?;

        for (idx, &val) in devtype_test.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / devtype_test.cols;
            let col = idx % devtype_test.cols;
            let combo = &devtype_test.combos[row];

            if bits == 0x11111111_11111111
                || bits == 0x22222222_22222222
                || bits == 0x33333333_33333333
            {
                let poison_type = if bits == 0x11111111_11111111 {
                    "alloc_with_nan_prefix"
                } else if bits == 0x22222222_22222222 {
                    "init_matrix_prefixes"
                } else {
                    "make_uninit_matrix"
                };

                panic!(
                    "[{}] Devtype test: Found {} poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
                    test,
                    poison_type,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    combo.period.unwrap_or(14),
                    combo.ma_type.as_deref().unwrap_or("sma"),
                    combo.nbdev.unwrap_or(1.0),
                    combo.devtype.unwrap_or(0)
                );
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_zscore_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = ZscoreInput::from_candles(&candles, "close", ZscoreParams::default());

        let baseline = zscore(&input)?.values;

        let mut out = vec![0.0; baseline.len()];
        {
            zscore_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());

        let eq_or_both_nan = |a: f64, b: f64| -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-9)
        };
        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }
}
