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
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MaaqInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        maaq_data_slice(&self.data)
    }
}

#[inline(always)]
fn maaq_data_slice<'a>(data: &'a MaaqData<'a>) -> &'a [f64] {
    match data {
        MaaqData::Slice(slice) => slice,
        MaaqData::Candles { candles, source } => match *source {
            "open" => &candles.open,
            "high" => &candles.high,
            "low" => &candles.low,
            "close" => &candles.close,
            "volume" => &candles.volume,
            "hl2" => &candles.hl2,
            "hlc3" => &candles.hlc3,
            "ohlc4" => &candles.ohlc4,
            "hlcc4" | "hlcc" => &candles.hlcc4,
            _ => source_type(candles, source),
        },
    }
}

#[derive(Debug, Clone)]
pub enum MaaqData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MaaqOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MaaqParams {
    pub period: Option<usize>,
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
}

impl Default for MaaqParams {
    fn default() -> Self {
        Self {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaaqInput<'a> {
    pub data: MaaqData<'a>,
    pub params: MaaqParams,
}

impl<'a> MaaqInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MaaqParams) -> Self {
        Self {
            data: MaaqData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MaaqParams) -> Self {
        Self {
            data: MaaqData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MaaqParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(11)
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(2)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MaaqBuilder {
    period: Option<usize>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    kernel: Kernel,
}

impl Default for MaaqBuilder {
    fn default() -> Self {
        Self {
            period: None,
            fast_period: None,
            slow_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MaaqBuilder {
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
    pub fn fast_period(mut self, n: usize) -> Self {
        self.fast_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self {
        self.slow_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MaaqOutput, MaaqError> {
        let p = MaaqParams {
            period: self.period,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = MaaqInput::from_candles(c, "close", p);
        maaq_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MaaqOutput, MaaqError> {
        let p = MaaqParams {
            period: self.period,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = MaaqInput::from_slice(d, p);
        maaq_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MaaqStream, MaaqError> {
        let p = MaaqParams {
            period: self.period,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        MaaqStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MaaqError {
    #[error("maaq: Input data slice is empty.")]
    EmptyInputData,
    #[error("maaq: All values are NaN.")]
    AllValuesNaN,
    #[error("maaq: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("maaq: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("maaq: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("maaq: Invalid range (start={start}, end={end}, step={step})")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("maaq: Non-batch kernel passed to batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("maaq: periods cannot be zero: period = {period}, fast = {fast_p}, slow = {slow_p}")]
    ZeroPeriods {
        period: usize,
        fast_p: usize,
        slow_p: usize,
    },
}

#[inline]
pub fn maaq(input: &MaaqInput) -> Result<MaaqOutput, MaaqError> {
    maaq_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn maaq_auto_kernel() -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn maaq_compute_into(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    if out.len() != data.len() {
        return Err(MaaqError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    unsafe {
        if first > 0 {
            maaq_scalar(data, period, fast_p, slow_p, first, out)?;
        } else {
            match kernel {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    maaq_scalar(data, period, fast_p, slow_p, first, out)?;
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    maaq_avx2(data, period, fast_p, slow_p, first, out)?;
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    maaq_avx512(data, period, fast_p, slow_p, first, out)?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn maaq_prepare<'a>(
    input: &'a MaaqInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, Kernel), MaaqError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(MaaqError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;

    let period = input.get_period();
    let fast_p = input.get_fast_period();
    let slow_p = input.get_slow_period();

    if period == 0 || fast_p == 0 || slow_p == 0 {
        return Err(MaaqError::ZeroPeriods {
            period,
            fast_p,
            slow_p,
        });
    }
    if period >= len {
        return Err(MaaqError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(MaaqError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => maaq_auto_kernel(),
        k => k,
    };

    Ok((data, period, fast_p, slow_p, first, chosen))
}

pub fn maaq_with_kernel(input: &MaaqInput, kernel: Kernel) -> Result<MaaqOutput, MaaqError> {
    let (data, period, fast_p, slow_p, first, chosen) = maaq_prepare(input, kernel)?;

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    if out.len() != data.len() {
        return Err(MaaqError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    maaq_compute_into(data, period, fast_p, slow_p, first, chosen, &mut out)?;

    let warmup_end = first + period - 1;
    for v in &mut out[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(MaaqOutput { values: out })
}

#[inline]
pub fn maaq_scalar(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    let len = data.len();
    let fast_sc = 2.0 / (fast_p as f64 + 1.0);
    let slow_sc = 2.0 / (slow_p as f64 + 1.0);

    let mut diffs = vec![0.0f64; period];
    let mut vol_sum = 0.0;

    unsafe {
        let dp = data.as_ptr();
        let op = out.as_mut_ptr();
        let mut j = 1usize;
        while j < period {
            let d = (*dp.add(first + j) - *dp.add(first + j - 1)).abs();
            *diffs.get_unchecked_mut(j) = d;
            vol_sum += d;
            j += 1;
        }

        let warm_end = (first + period).min(len);
        if warm_end > first {
            core::ptr::copy_nonoverlapping(dp.add(first), op.add(first), warm_end - first);
        }

        let i0 = first + period;
        if i0 >= len {
            return Ok(());
        }

        let new_diff = (*dp.add(i0) - *dp.add(i0 - 1)).abs();
        *diffs.get_unchecked_mut(0) = new_diff;
        vol_sum += new_diff;

        let mut prev_val = *dp.add(i0 - 1);
        let er0 = if vol_sum > f64::EPSILON {
            (*dp.add(i0) - *dp.add(first)).abs() / vol_sum
        } else {
            0.0
        };
        let mut sc = fast_sc.mul_add(er0, slow_sc);
        sc *= sc;

        prev_val = sc.mul_add(*dp.add(i0) - prev_val, prev_val);
        *op.add(i0) = prev_val;

        let mut head = if period > 1 { 1usize } else { 0usize };

        let mut i = i0 + 1;
        while i < len {
            vol_sum -= *diffs.get_unchecked(head);
            let nd = (*dp.add(i) - *dp.add(i - 1)).abs();
            *diffs.get_unchecked_mut(head) = nd;
            vol_sum += nd;
            head += 1;
            if head == period {
                head = 0;
            }

            let er = if vol_sum > f64::EPSILON {
                (*dp.add(i) - *dp.add(i - period)).abs() / vol_sum
            } else {
                0.0
            };
            let mut sc = fast_sc.mul_add(er, slow_sc);
            sc *= sc;

            prev_val = sc.mul_add(*dp.add(i) - prev_val, prev_val);
            *op.add(i) = prev_val;
            i += 1;
        }
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn maaq_avx2(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    use core::arch::x86_64::*;

    let len = data.len();
    debug_assert_eq!(len, out.len());
    if len == 0 {
        return Ok(());
    }

    let fast_sc = 2.0 / (fast_p as f64 + 1.0);
    let slow_sc = 2.0 / (slow_p as f64 + 1.0);

    #[inline(always)]
    unsafe fn vabs_pd(x: __m256d) -> __m256d {
        let sign = _mm256_set1_pd(-0.0f64);
        _mm256_andnot_pd(sign, x)
    }

    #[inline(always)]
    fn fast_abs(x: f64) -> f64 {
        f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }

    let mut diffs: Vec<f64> = Vec::with_capacity(period);
    unsafe {
        diffs.set_len(period);
    }
    let mut vol_sum = 0.0f64;

    unsafe {
        let dp = data.as_ptr();

        let base = first + 1;
        let n = period.saturating_sub(1);

        let mut accv = _mm256_setzero_pd();
        let mut j = 0usize;
        while j + 4 <= n {
            let a = _mm256_loadu_pd(dp.add(base + j));
            let b = _mm256_loadu_pd(dp.add(base + j - 1));
            let d = vabs_pd(_mm256_sub_pd(a, b));
            _mm256_storeu_pd(diffs.as_mut_ptr().add(1 + j), d);
            accv = _mm256_add_pd(accv, d);
            j += 4;
        }

        let mut tmp = [0.0f64; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), accv);
        vol_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        while j < n {
            let k = base + j;
            let d = fast_abs(*dp.add(k) - *dp.add(k - 1));
            *diffs.get_unchecked_mut(1 + j) = d;
            vol_sum += d;
            j += 1;
        }

        let warm_end = (first + period).min(len);
        if warm_end > first {
            core::ptr::copy_nonoverlapping(
                dp.add(first),
                out.as_mut_ptr().add(first),
                warm_end - first,
            );
        }

        let i0 = first + period;
        if i0 >= len {
            return Ok(());
        }

        let new_diff = fast_abs(*dp.add(i0) - *dp.add(i0 - 1));
        *diffs.get_unchecked_mut(0) = new_diff;
        vol_sum += new_diff;

        let mut prev_val = *dp.add(i0 - 1);
        let er0 = if vol_sum > f64::EPSILON {
            fast_abs(*dp.add(i0) - *dp.add(first)) / vol_sum
        } else {
            0.0
        };
        let mut sc = fast_sc.mul_add(er0, slow_sc);
        sc *= sc;
        prev_val = sc.mul_add(*dp.add(i0) - prev_val, prev_val);
        *out.get_unchecked_mut(i0) = prev_val;

        let mut head = if period > 1 { 1usize } else { 0usize };

        let mut i = i0 + 1;
        let op = out.as_mut_ptr();
        while i < len {
            vol_sum -= *diffs.get_unchecked(head);

            let nd = fast_abs(*dp.add(i) - *dp.add(i - 1));
            *diffs.get_unchecked_mut(head) = nd;
            vol_sum += nd;

            head += 1;
            if head == period {
                head = 0;
            }

            let er = if vol_sum > f64::EPSILON {
                fast_abs(*dp.add(i) - *dp.add(i - period)) / vol_sum
            } else {
                0.0
            };

            let mut sc = fast_sc.mul_add(er, slow_sc);
            sc *= sc;
            prev_val = sc.mul_add(*dp.add(i) - prev_val, prev_val);

            *op.add(i) = prev_val;
            i += 1;
        }
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn maaq_avx512(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    maaq_avx2(data, period, fast_p, slow_p, first, out)
}

#[derive(Debug, Clone)]
pub struct MaaqStream {
    period: usize,
    fast_period: usize,
    slow_period: usize,
    buffer: Vec<f64>,
    diff: Vec<f64>,
    head: usize,
    filled: bool,
    last: f64,
    count: usize,

    vol_sum: f64,
    fast_sc: f64,
    slow_sc: f64,
}

impl MaaqStream {
    pub fn try_new(params: MaaqParams) -> Result<Self, MaaqError> {
        let period = params.period.unwrap_or(11);
        let fast_p = params.fast_period.unwrap_or(2);
        let slow_p = params.slow_period.unwrap_or(30);

        if period == 0 || fast_p == 0 || slow_p == 0 {
            return Err(MaaqError::ZeroPeriods {
                period,
                fast_p,
                slow_p,
            });
        }

        let fast_sc = 2.0 / (fast_p as f64 + 1.0);
        let slow_sc = 2.0 / (slow_p as f64 + 1.0);

        Ok(Self {
            period,
            fast_period: fast_p,
            slow_period: slow_p,
            buffer: vec![0.0; period],
            diff: vec![0.0; period],
            head: 0,
            filled: false,
            last: f64::NAN,
            count: 0,
            vol_sum: 0.0,
            fast_sc,
            slow_sc,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.filled {
            let prev = if self.count > 0 {
                let idx_prev = (self.head + self.period - 1) % self.period;
                self.buffer[idx_prev]
            } else {
                value
            };
            let d = (value - prev).abs();

            self.buffer[self.head] = value;
            self.diff[self.head] = d;
            self.vol_sum += d;

            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }

            self.count += 1;
            self.last = value;

            if self.count == self.period {
                self.filled = true;
            }
            return Some(value);
        }

        let idx_prev = (self.head + self.period - 1) % self.period;
        let prev_input = self.buffer[idx_prev];

        let old_diff = self.diff[self.head];
        self.vol_sum -= old_diff;

        let new_diff = (value - prev_input).abs();
        self.diff[self.head] = new_diff;
        self.vol_sum += new_diff;

        let old_value = self.buffer[self.head];

        self.buffer[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }

        let er = if self.vol_sum > f64::EPSILON {
            (value - old_value).abs() / self.vol_sum
        } else {
            0.0
        };

        let mut sc = self.fast_sc.mul_add(er, self.slow_sc);
        sc *= sc;

        let out = sc.mul_add(value - self.last, self.last);
        self.last = out;
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct MaaqBatchRange {
    pub period: (usize, usize, usize),
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
}

impl Default for MaaqBatchRange {
    fn default() -> Self {
        Self {
            period: (11, 260, 1),
            fast_period: (2, 2, 0),
            slow_period: (30, 30, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MaaqBatchBuilder {
    range: MaaqBatchRange,
    kernel: Kernel,
}

impl MaaqBatchBuilder {
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
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }
    #[inline]
    pub fn fast_period_static(mut self, x: usize) -> Self {
        self.range.fast_period = (x, x, 0);
        self
    }
    #[inline]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_period_static(mut self, s: usize) -> Self {
        self.range.slow_period = (s, s, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<MaaqBatchOutput, MaaqError> {
        maaq_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MaaqBatchOutput, MaaqError> {
        MaaqBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MaaqBatchOutput, MaaqError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MaaqBatchOutput, MaaqError> {
        MaaqBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn maaq_batch_with_kernel(
    data: &[f64],
    sweep: &MaaqBatchRange,
    k: Kernel,
) -> Result<MaaqBatchOutput, MaaqError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => return Err(MaaqError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    maaq_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MaaqBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MaaqParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MaaqBatchOutput {
    pub fn row_for_params(&self, p: &MaaqParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(11) == p.period.unwrap_or(11)
                && c.fast_period.unwrap_or(2) == p.fast_period.unwrap_or(2)
                && c.slow_period.unwrap_or(30) == p.slow_period.unwrap_or(30)
        })
    }
    pub fn values_for(&self, p: &MaaqParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid(r: &MaaqBatchRange) -> Vec<MaaqParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if start == end || step == 0 {
            return vec![start];
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(step) {
                    Some(nx) if nx > x => x = nx,
                    _ => break,
                }
            }
        } else {
            let mut x = start;
            while x >= end {
                v.push(x);
                match x.checked_sub(step) {
                    Some(nx) if nx < x => x = nx,
                    _ => break,
                }
                if x == 0 {
                    break;
                }
            }
        }
        v
    }
    let periods = axis_usize(r.period);
    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let mut out = Vec::with_capacity(periods.len() * fasts.len() * slows.len());
    for &p in &periods {
        for &f in &fasts {
            for &s in &slows {
                out.push(MaaqParams {
                    period: Some(p),
                    fast_period: Some(f),
                    slow_period: Some(s),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn maaq_batch_slice(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
) -> Result<MaaqBatchOutput, MaaqError> {
    maaq_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn maaq_batch_par_slice(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
) -> Result<MaaqBatchOutput, MaaqError> {
    maaq_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn maaq_batch_inner(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MaaqBatchOutput, MaaqError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MaaqError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MaaqError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    if rows.checked_mul(cols).is_none() {
        return Err(MaaqError::InvalidRange {
            start: rows,
            end: cols,
            step: 0,
        });
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let fast_p = combos[row].fast_period.unwrap();
        let slow_p = combos[row].slow_period.unwrap();

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => maaq_row_scalar(data, first, period, fast_p, slow_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => maaq_row_avx2(data, first, period, fast_p, slow_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => maaq_row_avx512(data, first, period, fast_p, slow_p, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(MaaqBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn maaq_batch_inner_into(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MaaqParams>, MaaqError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MaaqError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MaaqError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let expected = rows.checked_mul(cols).ok_or(MaaqError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;
    if out.len() != expected {
        return Err(MaaqError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let fast_p = combos[row].fast_period.unwrap();
        let slow_p = combos[row].slow_period.unwrap();

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                maaq_row_scalar(data, first, period, fast_p, slow_p, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                maaq_row_avx2(data, first, period, fast_p, slow_p, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                maaq_row_avx512(data, first, period, fast_p, slow_p, out_row)
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn maaq_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_scalar(data, period, fast_p, slow_p, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn maaq_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_avx2(data, period, fast_p, slow_p, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn maaq_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_avx2(data, period, fast_p, slow_p, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn maaq_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_row_scalar(data, first, period, fast_p, slow_p, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn maaq_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_row_scalar(data, first, period, fast_p, slow_p, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_maaq_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = MaaqParams {
            period: None,
            fast_period: None,
            slow_period: None,
        };
        let input = MaaqInput::from_candles(&candles, "close", default_params);
        let output = maaq_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_maaq_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MaaqInput::from_candles(&candles, "close", MaaqParams::default());
        let result = maaq_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59747.657115949725,
            59740.803138018055,
            59724.24153333905,
            59720.60576365108,
            59673.9954445178,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] MAAQ {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_maaq_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MaaqInput::with_default_candles(&candles);
        match input.data {
            MaaqData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MaaqData::Candles"),
        }
        let output = maaq_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_maaq_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MaaqParams {
            period: Some(0),
            fast_period: Some(0),
            slow_period: Some(0),
        };
        let input = MaaqInput::from_slice(&input_data, params);
        let res = maaq_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MAAQ should fail with zero periods",
            test_name
        );
        Ok(())
    }

    fn check_maaq_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MaaqParams {
            period: Some(10),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let input = MaaqInput::from_slice(&data_small, params);
        let res = maaq_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MAAQ should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_maaq_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MaaqParams {
            period: Some(9),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let input = MaaqInput::from_slice(&single_point, params);
        let res = maaq_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MAAQ should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_maaq_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = MaaqParams {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        };
        let first_input = MaaqInput::from_candles(&candles, "close", first_params);
        let first_result = maaq_with_kernel(&first_input, kernel)?;
        let second_params = MaaqParams {
            period: Some(5),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let second_input = MaaqInput::from_slice(&first_result.values, second_params);
        let second_result = maaq_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_maaq_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MaaqInput::from_candles(
            &candles,
            "close",
            MaaqParams {
                period: Some(11),
                fast_period: Some(2),
                slow_period: Some(30),
            },
        );
        let res = maaq_with_kernel(&input, kernel)?;
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

    fn check_maaq_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 11;
        let fast_p = 2;
        let slow_p = 30;
        let input = MaaqInput::from_candles(
            &candles,
            "close",
            MaaqParams {
                period: Some(period),
                fast_period: Some(fast_p),
                slow_period: Some(slow_p),
            },
        );
        let batch_output = maaq_with_kernel(&input, kernel)?.values;
        let mut stream = MaaqStream::try_new(MaaqParams {
            period: Some(period),
            fast_period: Some(fast_p),
            slow_period: Some(slow_p),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(maaq_val) => stream_values.push(maaq_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());

        for i in period..batch_output.len() {
            let b = batch_output[i];
            let s = stream_values[i];
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] MAAQ streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_maaq_tests {
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
    fn check_maaq_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_cases = vec![
            MaaqParams::default(),
            MaaqParams {
                period: Some(5),
                fast_period: Some(2),
                slow_period: Some(10),
            },
            MaaqParams {
                period: Some(8),
                fast_period: Some(3),
                slow_period: Some(20),
            },
            MaaqParams {
                period: Some(11),
                fast_period: Some(2),
                slow_period: Some(30),
            },
            MaaqParams {
                period: Some(15),
                fast_period: Some(4),
                slow_period: Some(40),
            },
            MaaqParams {
                period: Some(20),
                fast_period: Some(5),
                slow_period: Some(50),
            },
            MaaqParams {
                period: Some(30),
                fast_period: Some(6),
                slow_period: Some(60),
            },
            MaaqParams {
                period: Some(10),
                fast_period: Some(8),
                slow_period: Some(30),
            },
            MaaqParams {
                period: Some(25),
                fast_period: Some(1),
                slow_period: Some(100),
            },
        ];

        for params in test_cases {
            let input = MaaqInput::from_candles(&candles, "close", params.clone());
            let output = maaq_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params period={:?}, fast_period={:?}, slow_period={:?}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period,
                        params.fast_period,
                        params.slow_period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params period={:?}, fast_period={:?}, slow_period={:?}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period,
                        params.fast_period,
                        params.slow_period
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params period={:?}, fast_period={:?}, slow_period={:?}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period,
                        params.fast_period,
                        params.slow_period
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_maaq_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_maaq_tests!(
        check_maaq_partial_params,
        check_maaq_accuracy,
        check_maaq_default_candles,
        check_maaq_zero_period,
        check_maaq_period_exceeds_length,
        check_maaq_very_small_dataset,
        check_maaq_reinput,
        check_maaq_nan_handling,
        check_maaq_streaming,
        check_maaq_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = MaaqBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MaaqParams::default();
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
            ((5, 10, 2), (2, 4, 1), (10, 30, 5)),
            ((10, 20, 5), (2, 6, 2), (20, 50, 10)),
            ((20, 30, 5), (4, 8, 2), (40, 80, 20)),
            ((10, 15, 5), (5, 10, 5), (30, 60, 30)),
            ((8, 12, 1), (2, 5, 1), (15, 25, 5)),
        ];

        for (period_range, fast_range, slow_range) in test_configs {
            let output = MaaqBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_range.0, period_range.1, period_range.2)
                .fast_period_range(fast_range.0, fast_range.1, fast_range.2)
                .slow_period_range(slow_range.0, slow_range.1, slow_range.2)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let params = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, fast_period={:?}, slow_period={:?})",
                        test,
                        val,
                        bits,
                        row,
                        col,
                        params.period,
                        params.fast_period,
                        params.slow_period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, fast_period={:?}, slow_period={:?})",
                        test,
                        val,
                        bits,
                        row,
                        col,
                        params.period,
                        params.fast_period,
                        params.slow_period
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, fast_period={:?}, slow_period={:?})",
                        test,
                        val,
                        bits,
                        row,
                        col,
                        params.period,
                        params.fast_period,
                        params.slow_period
                    );
                }
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_maaq_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let main_strat = (
            proptest::collection::vec(
                (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                20..200,
            ),
            2usize..30,
            1usize..10,
            10usize..50,
        )
            .prop_filter("valid params", |(data, period, fast_p, slow_p)| {
                *period <= data.len() && *fast_p < *slow_p
            });

        proptest::test_runner::TestRunner::default().run(
            &main_strat,
            |(data, period, fast_p, slow_p)| {
                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params.clone());

                let result = maaq_with_kernel(&input, kernel)?;
                let reference = maaq_with_kernel(&input, Kernel::Scalar)?;

                prop_assert_eq!(
                    result.values.len(),
                    data.len(),
                    "Output length {} doesn't match input length {}",
                    result.values.len(),
                    data.len()
                );

                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_end = first_valid + period - 1;
                for i in 0..warmup_end.min(data.len()) {
                    prop_assert!(
                        result.values[i].is_nan(),
                        "Warmup value at {} should be NaN, got {}",
                        i,
                        result.values[i]
                    );
                }

                for i in 0..result.values.len() {
                    let y = result.values[i];
                    let r = reference.values[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/Inf mismatch at {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                        "SIMD mismatch at {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                let data_min = data.iter().copied().fold(f64::INFINITY, f64::min);
                let data_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = (data_max - data_min).abs();
                let tolerance = range * 0.02;

                for (i, &val) in result.values.iter().enumerate() {
                    if val.is_finite() && i >= period {
                        prop_assert!(
                            val >= data_min - tolerance && val <= data_max + tolerance,
                            "Value {} at index {} outside bounds [{}, {}]",
                            val,
                            i,
                            data_min - tolerance,
                            data_max + tolerance
                        );
                    }
                }

                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > period {
                    let constant_val = data[0];
                    for (i, &val) in result.values[period..].iter().enumerate() {
                        prop_assert!(
                            (val - constant_val).abs() < 1e-8,
                            "Constant data should produce constant output, got {} at index {}",
                            val,
                            i + period
                        );
                    }
                }

                Ok(())
            },
        )?;

        let maaq_strat = (
            proptest::collection::vec(
                (-100f64..100f64).prop_filter("finite", |x| x.is_finite()),
                50..100,
            ),
            5usize..15,
            1usize..5,
            20usize..40,
        )
            .prop_filter("valid maaq params", |(_data, period, fast_p, slow_p)| {
                *fast_p < *slow_p
            });

        proptest::test_runner::TestRunner::default().run(
            &maaq_strat,
            |(data, period, fast_p, slow_p)| {
                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params);
                let result = maaq_with_kernel(&input, kernel)?;

                let fast_sc = 2.0 / (fast_p as f64 + 1.0);
                let slow_sc = 2.0 / (slow_p as f64 + 1.0);

                for i in (period + 1)..data.len() {
                    let signal = (data[i] - data[i - period]).abs();
                    let noise: f64 = (1..=period)
                        .map(|j| (data[i - j + 1] - data[i - j]).abs())
                        .sum();

                    if noise > f64::EPSILON {
                        let er = signal / noise;
                        prop_assert!(
                            er >= 0.0 && er <= 1.0 + 1e-10,
                            "Efficiency ratio {} out of bounds at index {}",
                            er,
                            i
                        );

                        let sc = (er * fast_sc + slow_sc).powi(2);

                        let min_sc = slow_sc.powi(2);
                        let max_sc = (fast_sc + slow_sc).powi(2);
                        prop_assert!(
                            sc >= min_sc - 1e-10 && sc <= max_sc + 1e-10,
                            "Smoothing constant {} out of bounds [{}..{}] at index {}",
                            sc,
                            min_sc,
                            max_sc,
                            i
                        );
                    }
                }

                if data.len() >= period * 3 {
                    let trending_indices: Vec<usize> = (period..data.len())
                        .filter(|&i| {
                            let signal = (data[i] - data[i.saturating_sub(period)]).abs();
                            signal > 10.0
                        })
                        .collect();

                    for &i in trending_indices.iter().take(5) {
                        let tracking_error = (result.values[i] - data[i]).abs();
                        let price_range = data[i.saturating_sub(period)..=i]
                            .iter()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
                                (min.min(v), max.max(v))
                            });
                        let local_range = (price_range.1 - price_range.0).abs();

                        prop_assert!(
                            tracking_error <= local_range * 0.2 + 1.0,
                            "Poor tracking in trend at {}: error {} > 20% of range {}",
                            i,
                            tracking_error,
                            local_range
                        );
                    }
                }

                Ok(())
            },
        )?;

        let step_strat = (
            10usize..30,
            2usize..5,
            20usize..40,
            -100f64..100f64,
            -100f64..100f64,
        )
            .prop_filter("different levels", |(_p, _f, _s, init, final_level)| {
                (init - final_level).abs() > 1.0
            });

        proptest::test_runner::TestRunner::default().run(
            &step_strat,
            |(period, fast_p, slow_p, initial, final_level)| {
                let mut data = vec![initial; 50];
                data.extend(vec![final_level; 50]);

                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params);
                let result = maaq_with_kernel(&input, kernel)?;

                let last_values = &result.values[90..];
                let convergence_target = final_level;

                for &val in last_values {
                    let distance_to_target = (val - convergence_target).abs();
                    let initial_distance = (initial - final_level).abs();

                    prop_assert!(
                        distance_to_target < initial_distance * 0.3,
                        "Failed to converge: value {} too far from target {}",
                        val,
                        convergence_target
                    );
                }

                Ok(())
            },
        )?;

        let small_strat = (
            proptest::collection::vec(
                (-100f64..100f64).prop_filter("finite", |x| x.is_finite()),
                1..5,
            ),
            1usize..3,
            1usize..3,
            3usize..6,
        )
            .prop_filter("valid small params", |(data, period, _fast_p, _slow_p)| {
                *period <= data.len()
            });

        proptest::test_runner::TestRunner::default().run(
            &small_strat,
            |(data, period, fast_p, slow_p)| {
                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params);

                let result = maaq_with_kernel(&input, kernel)?;

                prop_assert_eq!(result.values.len(), data.len());

                for i in 0..period.min(data.len()) {
                    if data[i].is_finite() {
                        prop_assert!(
                            (result.values[i] - data[i]).abs() < 1e-10,
                            "Small data warmup mismatch at {}",
                            i
                        );
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_maaq_tests!(check_maaq_property);

    #[test]
    fn test_maaq_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data: Vec<f64> = vec![f64::NAN, f64::NAN, f64::NAN];
        for i in 0..256u32 {
            let x = (i as f64).sin() * 0.5 + (i as f64) * 0.1 + ((i % 7) as f64) * 0.01;
            data.push(x);
        }

        let input = MaaqInput::from_slice(&data, MaaqParams::default());

        let baseline = maaq(&input)?.values;

        let mut out = vec![0.0; data.len()];
        super::maaq_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        for (idx, (a, b)) in baseline.iter().zip(out.iter()).enumerate() {
            let equal = (a.is_nan() && b.is_nan()) || ((a - b).abs() <= 1e-12);
            assert!(equal, "Mismatch at {}: {} vs {}", idx, a, b);
        }

        Ok(())
    }
}

#[inline]
pub fn maaq_into_slice(dst: &mut [f64], input: &MaaqInput, kern: Kernel) -> Result<(), MaaqError> {
    let (data, period, fast_p, slow_p, first, chosen) = maaq_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(MaaqError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    maaq_compute_into(data, period, fast_p, slow_p, first, chosen, dst)?;

    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn maaq_into(input: &MaaqInput, out: &mut [f64]) -> Result<(), MaaqError> {
    maaq_into_slice(out, input, Kernel::Auto)
}
