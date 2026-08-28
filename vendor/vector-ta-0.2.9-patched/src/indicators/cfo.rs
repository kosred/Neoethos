use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CfoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for CfoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CfoData::Slice(slice) => slice,
            CfoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CfoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CfoParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
}

impl Default for CfoParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CfoInput<'a> {
    pub data: CfoData<'a>,
    pub params: CfoParams,
}

impl<'a> CfoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CfoParams) -> Self {
        Self {
            data: CfoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CfoParams) -> Self {
        Self {
            data: CfoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CfoParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_scalar(&self) -> f64 {
        self.params.scalar.unwrap_or(100.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CfoBuilder {
    period: Option<usize>,
    scalar: Option<f64>,
    kernel: Kernel,
}

impl Default for CfoBuilder {
    fn default() -> Self {
        Self {
            period: None,
            scalar: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CfoBuilder {
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
    pub fn scalar(mut self, x: f64) -> Self {
        self.scalar = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CfoOutput, CfoError> {
        let p = CfoParams {
            period: self.period,
            scalar: self.scalar,
        };
        let i = CfoInput::from_candles(c, "close", p);
        cfo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CfoOutput, CfoError> {
        let p = CfoParams {
            period: self.period,
            scalar: self.scalar,
        };
        let i = CfoInput::from_slice(d, p);
        cfo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CfoStream, CfoError> {
        let p = CfoParams {
            period: self.period,
            scalar: self.scalar,
        };
        CfoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CfoError {
    #[error("cfo: All values are NaN.")]
    AllValuesNaN,
    #[error("cfo: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cfo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cfo: No data provided.")]
    EmptyInputData,
    #[error("cfo: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("cfo: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("cfo: Invalid range: start={start} end={end} step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
}

#[inline]
pub fn cfo(input: &CfoInput) -> Result<CfoOutput, CfoError> {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        cfo_with_kernel(input, cfo_auto_kernel())
    }
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        cfo_with_kernel(input, Kernel::Auto)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn cfo_auto_kernel() -> Kernel {
    if is_x86_feature_detected!("avx512f") {
        return Kernel::Avx512;
    }
    Kernel::Scalar
}

#[inline]
pub fn cfo_into_slice(dst: &mut [f64], input: &CfoInput, kernel: Kernel) -> Result<(), CfoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let period = input.get_period();
    let scalar = input.get_scalar();

    if len == 0 {
        return Err(CfoError::EmptyInputData);
    }

    if dst.len() != data.len() {
        return Err(CfoError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CfoError::AllValuesNaN)?;

    if period == 0 || period > len {
        return Err(CfoError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(CfoError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let warmup_end = first + period - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut dst[..warmup_end] {
        *v = qnan;
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => cfo_scalar(data, period, scalar, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cfo_avx2(data, period, scalar, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cfo_avx512(data, period, scalar, first, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                cfo_scalar(data, period, scalar, first, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn cfo_into(input: &CfoInput, out: &mut [f64]) -> Result<(), CfoError> {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        cfo_into_slice(out, input, cfo_auto_kernel())
    }
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        cfo_into_slice(out, input, Kernel::Auto)
    }
}

pub fn cfo_with_kernel(input: &CfoInput, kernel: Kernel) -> Result<CfoOutput, CfoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let period = input.get_period();
    let scalar = input.get_scalar();

    if len == 0 {
        return Err(CfoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CfoError::AllValuesNaN)?;

    if period == 0 || period > len {
        return Err(CfoError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(CfoError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first + period - 1);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cfo_scalar(data, period, scalar, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cfo_avx2(data, period, scalar, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cfo_avx512(data, period, scalar, first, &mut out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                cfo_scalar(data, period, scalar, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(CfoOutput { values: out })
}

#[inline]
pub fn cfo_scalar(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
    let size = data.len();

    let n = period as f64;
    let inv_n = 1.0 / n;
    let sx = ((period * (period + 1)) / 2) as f64;
    let sx2 = ((period * (period + 1) * (2 * period + 1)) / 6) as f64;
    let inv_denom = 1.0 / (n * sx2 - sx * sx);
    let half_nm1 = 0.5 * (n - 1.0);

    let start = first_valid;
    let pre = period - 1;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    for k in 0..pre {
        let v = data[start + k];
        let w = (k as f64) + 1.0;
        sum_y += v;
        sum_xy = v.mul_add(w, sum_xy);
    }

    let mut i = start + pre;
    while i < size {
        let v = data[i];
        sum_xy = v.mul_add(n, sum_xy);
        sum_y += v;
        let b = (-sx).mul_add(sum_y, n * sum_xy) * inv_denom;
        let f = b.mul_add(half_nm1, sum_y * inv_n);
        unsafe {
            *out.get_unchecked_mut(i) = if v.is_finite() && v != 0.0 {
                (v - f) * (scalar / v)
            } else {
                f64::NAN
            };
        }
        sum_xy -= sum_y;
        sum_y -= data[i - pre];
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cfo_avx512(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { cfo_avx512_short(data, period, scalar, first_valid, out) }
    } else {
        unsafe { cfo_avx512_long(data, period, scalar, first_valid, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn cfo_avx2(
    data: &[f64],
    period: usize,
    scalar: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn cfo_avx512_short(
    data: &[f64],
    period: usize,
    scalar: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn cfo_avx512_long(
    data: &[f64],
    period: usize,
    scalar: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn cfo_avx512_impl(
    data: &[f64],
    period: usize,
    scalar: f64,
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    if period < 2 {
        return cfo_scalar(data, period, scalar, first_valid, out);
    }

    let size = data.len();
    let n = period as f64;
    let inv_n = 1.0 / n;
    let sx = ((period * (period + 1)) / 2) as f64;
    let sx2 = ((period * (period + 1) * (2 * period + 1)) / 6) as f64;
    let inv_denom = 1.0 / (n * sx2 - sx * sx);
    let half_nm1 = 0.5 * (n - 1.0);

    let start = first_valid;
    let pre = period - 1;
    let end_init = start + pre;

    let dp = data.as_ptr();

    let mut acc_y = _mm512_set1_pd(0.0);
    let mut acc_xy = _mm512_set1_pd(0.0);
    let mut w = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let stepw = _mm512_set1_pd(8.0);

    let mut j = start;
    while j + 8 <= end_init {
        let v = _mm512_loadu_pd(dp.add(j));
        acc_y = _mm512_add_pd(acc_y, v);
        let prod = _mm512_mul_pd(v, w);
        acc_xy = _mm512_add_pd(acc_xy, prod);
        w = _mm512_add_pd(w, stepw);
        j += 8;
    }

    let mut tmp8 = [0.0f64; 8];
    _mm512_storeu_pd(tmp8.as_mut_ptr(), acc_y);
    let mut sum_y = tmp8.iter().sum::<f64>();
    _mm512_storeu_pd(tmp8.as_mut_ptr(), acc_xy);
    let mut sum_xy = tmp8.iter().sum::<f64>();

    let mut w_scalar = 1.0 + ((j - start) as f64);
    while j < end_init {
        let v = *dp.add(j);
        sum_y += v;
        sum_xy = v.mul_add(w_scalar, sum_xy);
        w_scalar += 1.0;
        j += 1;
    }

    let mut i = start + pre;
    while i < size {
        let v0 = *dp.add(i);
        sum_xy = v0.mul_add(n, sum_xy);
        sum_y += v0;
        let b0 = (-sx).mul_add(sum_y, n * sum_xy) * inv_denom;
        let f0 = b0.mul_add(half_nm1, sum_y * inv_n);
        *out.get_unchecked_mut(i) = if v0.is_finite() && v0 != 0.0 {
            (v0 - f0) * (scalar / v0)
        } else {
            f64::NAN
        };
        sum_xy -= sum_y;
        sum_y -= *dp.add(i - pre);
        i += 1;
    }
}

#[inline(always)]
pub fn cfo_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    scalar: f64,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    scalar: f64,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    scalar: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        cfo_row_avx512_short(data, first, period, stride, scalar, out)
    } else {
        cfo_row_avx512_long(data, first, period, stride, scalar, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    scalar: f64,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    scalar: f64,
    out: &mut [f64],
) {
    cfo_scalar(data, period, scalar, first, out)
}

#[derive(Debug, Clone)]
pub struct CfoStream {
    period: usize,
    scalar: f64,

    buf: Vec<f64>,
    idx: usize,
    filled: bool,

    sum_y: f64,
    sum_xy: f64,

    n: f64,
    inv_n: f64,
    sx: f64,
    inv_denom: f64,
    half_nm1: f64,
}

impl CfoStream {
    pub fn try_new(params: CfoParams) -> Result<Self, CfoError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CfoError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let scalar = params.scalar.unwrap_or(100.0);
        let n = period as f64;
        let inv_n = 1.0 / n;
        let sx = ((period * (period + 1)) / 2) as f64;
        let sx2 = ((period * (period + 1) * (2 * period + 1)) / 6) as f64;
        let inv_denom = 1.0 / (n * sx2 - sx * sx);
        let half_nm1 = 0.5 * (n - 1.0);

        Ok(Self {
            period,
            scalar,
            buf: vec![f64::NAN; period],
            idx: 0,
            filled: false,
            sum_y: 0.0,
            sum_xy: 0.0,
            n,
            inv_n,
            sx,
            inv_denom,
            half_nm1,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.filled {
            let k = (self.idx as f64) + 1.0;
            self.sum_y += value;
            self.sum_xy = value.mul_add(k, self.sum_xy);

            self.buf[self.idx] = value;
            self.idx += 1;

            if self.idx == self.period {
                self.idx = 0;
                self.filled = true;

                return Some(self.calc_current());
            } else {
                return None;
            }
        }

        let y_old = self.buf[self.idx];

        let new_sum_xy = (self.n * value) + (self.sum_xy - self.sum_y);
        let new_sum_y = self.sum_y - y_old + value;

        self.buf[self.idx] = value;
        self.sum_xy = new_sum_xy;
        self.sum_y = new_sum_y;

        self.idx = (self.idx + 1) % self.period;

        Some(self.calc_current())
    }

    #[inline(always)]
    fn calc_current(&self) -> f64 {
        debug_assert!(self.filled, "calc_current() called before buffer filled");

        let cur = self.buf[(self.idx + self.period - 1) % self.period];

        if cur.is_finite() && cur != 0.0 {
            let b = (-self.sx).mul_add(self.sum_y, self.n * self.sum_xy) * self.inv_denom;
            let f = b.mul_add(self.half_nm1, self.sum_y * self.inv_n);

            self.scalar.mul_add(-f / cur, self.scalar)
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
pub struct CfoBatchRange {
    pub period: (usize, usize, usize),
    pub scalar: (f64, f64, f64),
}

impl Default for CfoBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
            scalar: (100.0, 124.9, 0.1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CfoBatchBuilder {
    range: CfoBatchRange,
    kernel: Kernel,
}

impl CfoBatchBuilder {
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
    pub fn scalar_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.scalar = (start, end, step);
        self
    }
    #[inline]
    pub fn scalar_static(mut self, s: f64) -> Self {
        self.range.scalar = (s, s, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<CfoBatchOutput, CfoError> {
        cfo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CfoBatchOutput, CfoError> {
        CfoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CfoBatchOutput, CfoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CfoBatchOutput, CfoError> {
        CfoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn cfo_batch_with_kernel(
    data: &[f64],
    sweep: &CfoBatchRange,
    k: Kernel,
) -> Result<CfoBatchOutput, CfoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(CfoError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cfo_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CfoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CfoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CfoBatchOutput {
    pub fn row_for_params(&self, p: &CfoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.scalar.unwrap_or(100.0) - p.scalar.unwrap_or(100.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &CfoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CfoBatchRange) -> Result<Vec<CfoParams>, CfoError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CfoError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                vals.push(cur);
                match cur.checked_add(step) {
                    Some(next) => cur = next,
                    None => break,
                }
            }
        } else {
            return Err(CfoError::InvalidRange { start, end, step });
        }
        if vals.is_empty() {
            return Err(CfoError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CfoError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        let delta = if start <= end {
            step.abs()
        } else {
            -step.abs()
        };
        if delta.is_sign_positive() {
            let mut x = start;
            while x <= end + 1e-12 {
                vals.push(x);
                x += delta;
                if !x.is_finite() {
                    break;
                }
            }
        } else {
            let mut x = start;
            while x >= end - 1e-12 {
                vals.push(x);
                x += delta;
                if !x.is_finite() {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(CfoError::InvalidRange {
                start: 0,
                end: 0,
                step: 0,
            });
        }
        Ok(vals)
    }
    let periods = axis_usize(r.period)?;
    let scalars = axis_f64(r.scalar)?;
    let combos_len = periods
        .len()
        .checked_mul(scalars.len())
        .ok_or(CfoError::InvalidRange {
            start: periods.len(),
            end: scalars.len(),
            step: 0,
        })?;
    let mut out = Vec::with_capacity(combos_len);
    for &p in &periods {
        for &s in &scalars {
            out.push(CfoParams {
                period: Some(p),
                scalar: Some(s),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn cfo_batch_slice(
    data: &[f64],
    sweep: &CfoBatchRange,
    kern: Kernel,
) -> Result<CfoBatchOutput, CfoError> {
    cfo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cfo_batch_par_slice(
    data: &[f64],
    sweep: &CfoBatchRange,
    kern: Kernel,
) -> Result<CfoBatchOutput, CfoError> {
    cfo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cfo_batch_inner(
    data: &[f64],
    sweep: &CfoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CfoBatchOutput, CfoError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(CfoError::EmptyInputData);
    }

    for combo in &combos {
        let period = combo.period.unwrap();
        if period == 0 || period > data.len() {
            return Err(CfoError::InvalidPeriod {
                period,
                data_len: data.len(),
            });
        }
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CfoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CfoError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    rows.checked_mul(cols).ok_or(CfoError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    cfo_batch_prefix_scalar_rows(data, first, &combos, values, rows, cols, parallel);

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(CfoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline]
fn cfo_batch_prefix_scalar_rows(
    data: &[f64],
    first: usize,
    combos: &[CfoParams],
    values: &mut [f64],
    rows: usize,
    cols: usize,
    parallel: bool,
) {
    let valid_len = cols - first;
    if valid_len == 0 || rows == 0 {
        return;
    }

    let mut p = Vec::with_capacity(valid_len + 1);
    let mut q = Vec::with_capacity(valid_len + 1);
    p.push(0.0);
    q.push(0.0);
    let mut ps = 0.0f64;
    let mut qs = 0.0f64;
    for (idx, &v) in data[first..].iter().enumerate() {
        let j = (idx as f64) + 1.0;
        ps += v;
        qs = v.mul_add(j, qs);
        p.push(ps);
        q.push(qs);
    }

    let compute_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let scalar = combos[row].scalar.unwrap();
        let n = period as f64;
        let inv_n = 1.0 / n;
        let sx = ((period * (period + 1)) / 2) as f64;
        let sx2 = ((period * (period + 1) * (2 * period + 1)) / 6) as f64;
        let inv_denom = 1.0 / (n * sx2 - sx * sx);
        let half_nm1 = 0.5 * (n - 1.0);

        if cols < first + period {
            return;
        }

        let start_idx = first + period - 1;
        for di in start_idx..cols {
            let idx = di - first;

            let r1 = idx + 1;
            let l1_minus1 = idx + 1 - period;

            let sum_y = p[r1] - p[l1_minus1];
            let sum_xy = (q[r1] - q[l1_minus1]) - (l1_minus1 as f64) * sum_y;

            let b = (-sx).mul_add(sum_y, n * sum_xy) * inv_denom;
            let f = b.mul_add(half_nm1, sum_y * inv_n);
            let v = data[di];
            let y = if v.is_finite() && v != 0.0 {
                (v - f) * (scalar / v)
            } else {
                f64::NAN
            };
            out_row[di] = y;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| compute_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                compute_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            compute_row(row, slice);
        }
    }
}

#[inline(always)]
pub fn cfo_batch_inner_into(
    data: &[f64],
    sweep: &CfoBatchRange,
    kern: Kernel,
    parallel: bool,
    output: &mut [f64],
) -> Result<Vec<CfoParams>, CfoError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(CfoError::EmptyInputData);
    }

    for combo in &combos {
        let period = combo.period.unwrap();
        if period == 0 || period > data.len() {
            return Err(CfoError::InvalidPeriod {
                period,
                data_len: data.len(),
            });
        }
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CfoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CfoError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let total = rows.checked_mul(cols).ok_or(CfoError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;
    if output.len() != total {
        return Err(CfoError::OutputLengthMismatch {
            expected: total,
            got: output.len(),
        });
    }

    let out_mu: &mut [core::mem::MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            output.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            output.len(),
        )
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    let values: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len()) };
    cfo_batch_prefix_scalar_rows(data, first, &combos, values, rows, cols, parallel);

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_cfo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = CfoParams {
            period: None,
            scalar: None,
        };
        let input = CfoInput::from_candles(&candles, "close", default_params);
        let output = cfo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cfo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = CfoInput::from_candles(&candles, "close", CfoParams::default());
        let result = cfo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            0.5998626489475746,
            0.47578011282578453,
            0.20349744599816233,
            0.0919617952835795,
            -0.5676291145560617,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] CFO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_cfo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = CfoInput::with_default_candles(&candles);
        match input.data {
            CfoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CfoData::Candles"),
        }
        let output = cfo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cfo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CfoParams {
            period: Some(0),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_slice(&input_data, params);
        let res = cfo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CFO should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cfo_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CfoParams {
            period: Some(10),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_slice(&data_small, params);
        let res = cfo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CFO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cfo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_slice(&single_point, params);
        let res = cfo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CFO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cfo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let first_input = CfoInput::from_candles(&candles, "close", first_params);
        let first_result = cfo_with_kernel(&first_input, kernel)?;

        let second_params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let second_input = CfoInput::from_slice(&first_result.values, second_params);
        let second_result = cfo_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after idx 240, found NaN at {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_cfo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CfoInput::from_candles(
            &candles,
            "close",
            CfoParams {
                period: Some(14),
                scalar: Some(100.0),
            },
        );
        let res = cfo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_cfo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let period = 14;
        let scalar = 100.0;

        let input = CfoInput::from_candles(
            &candles,
            "close",
            CfoParams {
                period: Some(period),
                scalar: Some(scalar),
            },
        );
        let batch_output = cfo_with_kernel(&input, kernel)?.values;

        let mut stream = CfoStream::try_new(CfoParams {
            period: Some(period),
            scalar: Some(scalar),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] CFO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[test]
    fn test_cfo_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = Vec::with_capacity(256);

        data.push(f64::NAN);
        data.push(f64::NAN);
        for i in 1..=254u32 {
            data.push(50.0 + (i as f64) * 0.25);
        }

        let input = CfoInput::from_slice(&data, CfoParams::default());

        let baseline = cfo(&input)?.values;

        let mut out = vec![0.0; data.len()];
        {
            cfo_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_cfo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let param_sets = vec![
            CfoParams {
                period: Some(14),
                scalar: Some(100.0),
            },
            CfoParams {
                period: Some(7),
                scalar: Some(50.0),
            },
            CfoParams {
                period: Some(21),
                scalar: Some(200.0),
            },
            CfoParams {
                period: Some(30),
                scalar: Some(100.0),
            },
        ];

        for params in param_sets {
            let input = CfoInput::from_candles(&candles, "close", params.clone());
            let output = cfo_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params {:?}",
                        test_name, val, bits, i, params
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params {:?}",
                        test_name, val, bits, i, params
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params {:?}",
                        test_name, val, bits, i, params
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_cfo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cfo_tests {
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_cfo_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64)
                        .prop_filter("finite and non-zero", |x| x.is_finite() && x.abs() > 1e-10),
                    period..400,
                ),
                Just(period),
                50.0f64..200.0f64,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, scalar)| {
                let params = CfoParams {
                    period: Some(period),
                    scalar: Some(scalar),
                };
                let input = CfoInput::from_slice(&data, params);

                let CfoOutput { values: out } = cfo_with_kernel(&input, kernel).unwrap();
                let CfoOutput { values: ref_out } =
                    cfo_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                for i in 0..(period - 1) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                if data.len() >= period {
                    let first_valid_idx = period - 1;
                    prop_assert!(
                        out[first_valid_idx].is_finite(),
                        "Expected finite value at first valid index {}, got {}",
                        first_valid_idx,
                        out[first_valid_idx]
                    );
                }

                for i in (period - 1)..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    prop_assert!(
                        y.is_finite(),
                        "Expected finite output at index {}, got {}",
                        i,
                        y
                    );

                    let window_start = i.saturating_sub(period - 1);
                    let window = &data[window_start..=i];
                    if window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) {
                        let expected = 0.0;
                        prop_assert!(
                            (y - expected).abs() < 1e-6,
                            "For constant data at idx {}, expected CFO ≈ 0, got {}",
                            i,
                            y
                        );
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                if data.len() > period {
                    let params_double = CfoParams {
                        period: Some(period),
                        scalar: Some(scalar * 2.0),
                    };
                    let input_double = CfoInput::from_slice(&data, params_double);
                    let CfoOutput { values: out_double } =
                        cfo_with_kernel(&input_double, kernel).unwrap();

                    for i in (period - 1)..data.len().min(period + 10) {
                        if out[i].is_finite() && out_double[i].is_finite() && out[i].abs() > 1e-6 {
                            let ratio = out_double[i] / out[i];
                            prop_assert!(
                                (ratio - 2.0).abs() < 1e-9,
                                "Scale linearity failed at idx {}: ratio = {} (expected 2.0)",
                                i,
                                ratio
                            );
                        }
                    }
                }

                if period == 2 && data.len() >= 2 {
                    prop_assert!(out[0].is_nan(), "Period=2: first value should be NaN");

                    prop_assert!(
                        out[1].is_finite(),
                        "Period=2: second value should be finite, got {}",
                        out[1]
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_cfo_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    generate_all_cfo_tests!(
        check_cfo_partial_params,
        check_cfo_accuracy,
        check_cfo_default_candles,
        check_cfo_zero_period,
        check_cfo_period_exceeds_length,
        check_cfo_very_small_dataset,
        check_cfo_reinput,
        check_cfo_nan_handling,
        check_cfo_streaming,
        check_cfo_no_poison,
        check_cfo_property
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = CfoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = CfoParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            0.5998626489475746,
            0.47578011282578453,
            0.20349744599816233,
            0.0919617952835795,
            -0.5676291145560617,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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

        let output = CfoBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 30, 5)
            .scalar_range(50.0, 200.0, 50.0)
            .apply_candles(&c, "close")?;

        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
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
}
