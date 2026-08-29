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
use thiserror::Error;

impl<'a> AsRef<[f64]> for RocpInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RocpData::Slice(slice) => slice,
            RocpData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RocpData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocpOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocpParams {
    pub period: Option<usize>,
}

impl Default for RocpParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct RocpInput<'a> {
    pub data: RocpData<'a>,
    pub params: RocpParams,
}

impl<'a> RocpInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: RocpParams) -> Self {
        Self {
            data: RocpData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: RocpParams) -> Self {
        Self {
            data: RocpData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", RocpParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RocpBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for RocpBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RocpBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<RocpOutput, RocpError> {
        let p = RocpParams {
            period: self.period,
        };
        let i = RocpInput::from_candles(c, "close", p);
        rocp_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<RocpOutput, RocpError> {
        let p = RocpParams {
            period: self.period,
        };
        let i = RocpInput::from_slice(d, p);
        rocp_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RocpStream, RocpError> {
        let p = RocpParams {
            period: self.period,
        };
        RocpStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum RocpError {
    #[error("rocp: Input data slice is empty.")]
    EmptyInputData,
    #[error("rocp: All values are NaN.")]
    AllValuesNaN,
    #[error("rocp: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rocp: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rocp: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("rocp: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("rocp: Invalid kernel type for batch operation: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn rocp(input: &RocpInput) -> Result<RocpOutput, RocpError> {
    rocp_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn rocp_into(input: &RocpInput, out: &mut [f64]) -> Result<(), RocpError> {
    rocp_into_slice(out, input, Kernel::Auto)
}

pub fn rocp_with_kernel(input: &RocpInput, kernel: Kernel) -> Result<RocpOutput, RocpError> {
    let data: &[f64] = match &input.data {
        RocpData::Candles { candles, source } => source_type(candles, source),
        RocpData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(RocpError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocpError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(RocpError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(RocpError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first + period);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => rocp_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => rocp_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => rocp_avx512(data, period, first, &mut out),
            _ => rocp_scalar(data, period, first, &mut out),
        }
    }

    Ok(RocpOutput { values: out })
}

#[inline]
pub fn rocp_into_slice(dst: &mut [f64], input: &RocpInput, kern: Kernel) -> Result<(), RocpError> {
    let data: &[f64] = match &input.data {
        RocpData::Candles { candles, source } => source_type(candles, source),
        RocpData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(RocpError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocpError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(RocpError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(RocpError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    if dst.len() != len {
        return Err(RocpError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                match detect_best_kernel() {
                    Kernel::Avx512 if len < 1_000_000 => Kernel::Avx512,
                    Kernel::Avx512 | Kernel::Avx2 => Kernel::Avx2,
                    _ => Kernel::Scalar,
                }
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                Kernel::Scalar
            }
        }
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => rocp_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => rocp_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => rocp_avx512(data, period, first, dst),
            _ => rocp_scalar(data, period, first, dst),
        }
    }

    for v in &mut dst[..(first + period)] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn rocp_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let start = first_val + period;
    let n = data.len();
    if start >= n {
        return;
    }

    let curr = &data[start..];
    let prev = &data[(start - period)..(n - period)];
    let dst = &mut out[start..];

    for ((&c, &p), o) in curr.iter().zip(prev.iter()).zip(dst.iter_mut()) {
        *o = (c - p) / p;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocp_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    unsafe {
        if period <= 32 {
            rocp_avx512_short(data, period, first_val, out)
        } else {
            rocp_avx512_long(data, period, first_val, out)
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocp_avx2(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    unsafe { rocp_avx2_impl(data, period, first_val, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn rocp_avx2_impl(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let start = first_val + period;
    let n = data.len();
    if start >= n {
        return;
    }
    let mut i = start;
    let end = n - ((n - start) & 3);
    while i + 4 <= end {
        let c = _mm256_loadu_pd(data.as_ptr().add(i));
        let p = _mm256_loadu_pd(data.as_ptr().add(i - period));
        let num = _mm256_sub_pd(c, p);
        let div = _mm256_div_pd(num, p);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), div);
        i += 4;
    }
    while i < n {
        let prev = *data.get_unchecked(i - period);
        *out.get_unchecked_mut(i) = (*data.get_unchecked(i) - prev) / prev;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocp_avx512_short(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    rocp_avx512_impl(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocp_avx512_long(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    rocp_avx512_impl(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn rocp_avx512_impl(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let start = first_val + period;
    let n = data.len();
    if start >= n {
        return;
    }
    let mut i = start;
    let end = n - ((n - start) & 7);
    while i + 8 <= end {
        let c = _mm512_loadu_pd(data.as_ptr().add(i));
        let p = _mm512_loadu_pd(data.as_ptr().add(i - period));
        let num = _mm512_sub_pd(c, p);
        let div = _mm512_div_pd(num, p);
        _mm512_storeu_pd(out.as_mut_ptr().add(i), div);
        i += 8;
    }
    while i < n {
        let prev = *data.get_unchecked(i - period);
        *out.get_unchecked_mut(i) = (*data.get_unchecked(i) - prev) / prev;
        i += 1;
    }
}

#[derive(Debug, Clone)]
pub struct RocpStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,

    warmup: usize,

    inv: Vec<f64>,
}

impl RocpStream {
    pub fn try_new(params: RocpParams) -> Result<Self, RocpError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(RocpError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            warmup: period,
            inv: vec![f64::NAN; period],
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let idx = self.head;

        let prev = self.buffer[idx];
        let inv_prev = self.inv[idx];

        self.buffer[idx] = value;

        self.inv[idx] = 1.0f64 / value;

        let next = idx + 1;
        self.head = if next == self.period { 0 } else { next };

        if self.warmup > 1 {
            self.warmup -= 1;
            return None;
        } else if self.warmup == 1 {
            self.warmup = 0;

            return Some(value.mul_add(inv_prev, -1.0));
        }

        Some(value.mul_add(inv_prev, -1.0))
    }
}

#[derive(Clone, Debug)]
pub struct RocpBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for RocpBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 258, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RocpBatchBuilder {
    range: RocpBatchRange,
    kernel: Kernel,
}

impl RocpBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<RocpBatchOutput, RocpError> {
        rocp_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocpBatchOutput, RocpError> {
        RocpBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocpBatchOutput, RocpError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<RocpBatchOutput, RocpError> {
        RocpBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn rocp_batch_with_kernel(
    data: &[f64],
    sweep: &RocpBatchRange,
    k: Kernel,
) -> Result<RocpBatchOutput, RocpError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => {
            return Err(RocpError::InvalidKernelForBatch(other));
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    rocp_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocpBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<RocpParams>,
    pub rows: usize,
    pub cols: usize,
}
impl RocpBatchOutput {
    pub fn row_for_params(&self, p: &RocpParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
    }

    pub fn values_for(&self, p: &RocpParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &RocpBatchRange) -> Result<Vec<RocpParams>, RocpError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, RocpError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let vals: Vec<usize> = (start..=end).step_by(st).collect();
            if vals.is_empty() {
                return Err(RocpError::InvalidRange { start, end, step });
            }
            return Ok(vals);
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(RocpError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(RocpParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn rocp_batch_slice(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
) -> Result<RocpBatchOutput, RocpError> {
    rocp_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rocp_batch_par_slice(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
) -> Result<RocpBatchOutput, RocpError> {
    rocp_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rocp_batch_inner(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RocpBatchOutput, RocpError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(RocpError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocpError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(RocpError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let _ = rows.checked_mul(cols).ok_or(RocpError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warmup_periods: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let use_inv = rows >= 4;
    let inv: Option<Vec<f64>> = if use_inv {
        let mut v = Vec::with_capacity(cols);

        for &x in data.iter() {
            v.push(1.0f64 / x);
        }
        Some(v)
    } else {
        None
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        if let (Some(inv), Kernel::Scalar) = (&inv, kern) {
            rocp_row_scalar_with_inv(data, first, period, out_row, inv);
        } else {
            match kern {
                Kernel::Scalar => rocp_row_scalar(data, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => rocp_row_avx2(data, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => rocp_row_avx512(data, first, period, out_row),
                _ => rocp_row_scalar(data, first, period, out_row),
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(RocpBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn rocp_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let start = first + period;
    let n = data.len();
    if start >= n {
        return;
    }

    let curr = &data[start..];
    let prev = &data[(start - period)..(n - period)];
    let dst = &mut out[start..];
    for ((&c, &p), o) in curr.iter().zip(prev.iter()).zip(dst.iter_mut()) {
        *o = (c - p) / p;
    }
}

#[inline(always)]
fn rocp_row_scalar_with_inv(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
    inv: &[f64],
) {
    let start = first + period;
    let n = data.len();
    if start >= n {
        return;
    }

    let curr = &data[start..];
    let inv_prev = &inv[(start - period)..(n - period)];
    let dst = &mut out[start..];
    for ((&c, &ip), o) in curr.iter().zip(inv_prev.iter()).zip(dst.iter_mut()) {
        *o = c.mul_add(ip, -1.0);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rocp_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    rocp_row_avx2_impl(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        rocp_row_avx512_short(data, first, period, out);
    } else {
        rocp_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    rocp_row_avx512_impl(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocp_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    rocp_row_avx512_impl(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn rocp_row_avx2_impl(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let start = first + period;
    let n = data.len();
    if start >= n {
        return;
    }
    let mut i = start;
    let end = n - ((n - start) & 3);
    while i + 4 <= end {
        let c = _mm256_loadu_pd(data.as_ptr().add(i));
        let p = _mm256_loadu_pd(data.as_ptr().add(i - period));
        let num = _mm256_sub_pd(c, p);
        let div = _mm256_div_pd(num, p);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), div);
        i += 4;
    }
    while i < n {
        let prev = *data.get_unchecked(i - period);
        *out.get_unchecked_mut(i) = (*data.get_unchecked(i) - prev) / prev;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn rocp_row_avx512_impl(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let start = first + period;
    let n = data.len();
    if start >= n {
        return;
    }
    let mut i = start;
    let end = n - ((n - start) & 7);
    while i + 8 <= end {
        let c = _mm512_loadu_pd(data.as_ptr().add(i));
        let p = _mm512_loadu_pd(data.as_ptr().add(i - period));
        let num = _mm512_sub_pd(c, p);
        let div = _mm512_div_pd(num, p);
        _mm512_storeu_pd(out.as_mut_ptr().add(i), div);
        i += 8;
    }
    while i < n {
        let prev = *data.get_unchecked(i - period);
        *out.get_unchecked_mut(i) = (*data.get_unchecked(i) - prev) / prev;
        i += 1;
    }
}

#[inline(always)]
pub fn rocp_batch_inner_into(
    data: &[f64],
    sweep: &RocpBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<RocpParams>, RocpError> {
    use core::mem::MaybeUninit;

    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(RocpError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(RocpError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(RocpError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let expected = rows.checked_mul(cols).ok_or(RocpError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    if out.len() != expected {
        return Err(RocpError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    let use_inv = combos.len() >= 4;
    let inv: Option<Vec<f64>> = if use_inv {
        let mut v = Vec::with_capacity(cols);
        for &x in data.iter() {
            v.push(1.0f64 / x);
        }
        Some(v)
    } else {
        None
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let dst: &mut [f64] =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        if let (Some(inv), Kernel::Scalar | Kernel::ScalarBatch) = (&inv, kern) {
            rocp_row_scalar_with_inv(data, first, period, dst, inv);
        } else {
            match kern {
                Kernel::Scalar | Kernel::ScalarBatch => rocp_row_scalar(data, first, period, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => rocp_row_avx2(data, first, period, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => rocp_row_avx512(data, first, period, dst),
                _ => rocp_row_scalar(data, first, period, dst),
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
        }
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_rocp_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = RocpParams { period: None };
        let input = RocpInput::from_candles(&candles, "close", default_params);
        let output = rocp_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_rocp_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = RocpInput::from_candles(&candles, "close", RocpParams { period: Some(10) });
        let result = rocp_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -0.0022551709049293996,
            -0.005561903481650759,
            -0.003275201323586514,
            -0.004945415398072297,
            -0.015045927020537019,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-9,
                "[{}] ROCP {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_rocp_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = RocpInput::with_default_candles(&candles);
        match input.data {
            RocpData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected RocpData::Candles"),
        }
        let output = rocp_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_rocp_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = RocpParams { period: Some(0) };
        let input = RocpInput::from_slice(&input_data, params);
        let res = rocp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ROCP should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_rocp_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = RocpParams { period: Some(10) };
        let input = RocpInput::from_slice(&data_small, params);
        let res = rocp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ROCP should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_rocp_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = RocpParams { period: Some(9) };
        let input = RocpInput::from_slice(&single_point, params);
        let res = rocp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ROCP should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_rocp_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = RocpParams { period: Some(14) };
        let first_input = RocpInput::from_candles(&candles, "close", first_params);
        let first_result = rocp_with_kernel(&first_input, kernel)?;

        let second_params = RocpParams { period: Some(14) };
        let second_input = RocpInput::from_slice(&first_result.values, second_params);
        let second_result = rocp_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] ROCP Slice Reinput {:?} mismatch at idx {}: got NaN",
                test_name,
                kernel,
                i
            );
        }
        Ok(())
    }

    fn check_rocp_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = RocpInput::from_candles(&candles, "close", RocpParams { period: Some(9) });
        let res = rocp_with_kernel(&input, kernel)?;
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

    #[test]
    fn test_rocp_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = RocpParams { period: Some(10) };
        let input = RocpInput::from_candles(&candles, "close", params);

        let baseline = rocp(&input)?.values;

        let mut out = vec![0.0; candles.close.len()];
        {
            rocp_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());
        for i in 0..baseline.len() {
            let a = baseline[i];
            let b = out[i];
            let equal = (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12;
            assert!(
                equal,
                "ROCP parity mismatch at index {}: api={} into={}",
                i, a, b
            );
        }

        Ok(())
    }

    fn check_rocp_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let period = 9;

        let input = RocpInput::from_candles(
            &candles,
            "close",
            RocpParams {
                period: Some(period),
            },
        );
        let batch_output = rocp_with_kernel(&input, kernel)?.values;

        let mut stream = RocpStream::try_new(RocpParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(rocp_val) => stream_values.push(rocp_val),
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
                "[{}] ROCP streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_rocp_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            RocpParams::default(),
            RocpParams { period: Some(2) },
            RocpParams { period: Some(5) },
            RocpParams { period: Some(7) },
            RocpParams { period: Some(9) },
            RocpParams { period: Some(14) },
            RocpParams { period: Some(20) },
            RocpParams { period: Some(50) },
            RocpParams { period: Some(100) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = RocpInput::from_candles(&candles, "close", params.clone());
            let output = rocp_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_rocp_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_rocp_tests {
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

    generate_all_rocp_tests!(
        check_rocp_partial_params,
        check_rocp_accuracy,
        check_rocp_default_candles,
        check_rocp_zero_period,
        check_rocp_period_exceeds_length,
        check_rocp_very_small_dataset,
        check_rocp_reinput,
        check_rocp_nan_handling,
        check_rocp_streaming,
        check_rocp_no_poison
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = RocpBatchBuilder::new()
            .period_static(10)
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = RocpParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -0.0022551709049293996,
            -0.005561903481650759,
            -0.003275201323586514,
            -0.004945415398072297,
            -0.015045927020537019,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-9,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (5, 25, 5),
            (30, 60, 15),
            (2, 5, 1),
            (10, 50, 10),
            (7, 21, 7),
            (14, 28, 14),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = RocpBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

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
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_rocp_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (0.01f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = RocpParams {
                    period: Some(period),
                };
                let input = RocpInput::from_slice(&data, params);

                let RocpOutput { values: out } = rocp_with_kernel(&input, kernel).unwrap();

                let RocpOutput { values: ref_out } =
                    rocp_with_kernel(&input, Kernel::Scalar).unwrap();

                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

                for i in (first_valid + period)..data.len() {
                    let prev_value = data[i - period];
                    let curr_value = data[i];
                    let y = out[i];
                    let r = ref_out[i];

                    if prev_value != 0.0 && prev_value.is_finite() && curr_value.is_finite() {
                        let expected = (curr_value - prev_value) / prev_value;

                        prop_assert!(
                            y.is_nan() || (y - expected).abs() <= 1e-10,
                            "idx {}: ROCP mismatch. Got {}, expected {}, curr={}, prev={}",
                            i,
                            y,
                            expected,
                            curr_value,
                            prev_value
                        );
                    } else if prev_value == 0.0 {
                        prop_assert!(
                            !y.is_finite(),
                            "idx {}: Expected non-finite value when dividing by zero, got {}",
                            i,
                            y
                        );
                    }

                    let is_constant = data[first_valid..=i].windows(2).all(|w| {
                        let max_val = w[0].max(w[1]);
                        if max_val > 0.0 {
                            (w[0] - w[1]).abs() / max_val < 1e-10
                        } else {
                            w[0] == w[1]
                        }
                    });

                    if is_constant && data[first_valid].is_finite() && data[first_valid] != 0.0 {
                        prop_assert!(
                            y.abs() <= 1e-10,
                            "Constant data should produce ROCP=0, got {} at idx {}",
                            y,
                            i
                        );
                    }

                    let is_increasing = i >= first_valid + period
                        && (first_valid..=i).all(|j| j == first_valid || data[j] > data[j - 1]);

                    if is_increasing && y.is_finite() {
                        prop_assert!(
							y > -1e-10,
							"Strictly increasing data should produce positive ROCP, got {} at idx {}",
							y, i
						);
                    }

                    let is_decreasing = i >= first_valid + period
                        && (first_valid..=i).all(|j| j == first_valid || data[j] < data[j - 1]);

                    if is_decreasing && y.is_finite() {
                        prop_assert!(
							y < 1e-10,
							"Strictly decreasing data should produce negative ROCP, got {} at idx {}",
							y, i
						);
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-10 || ulp_diff <= 4,
                        "Kernel mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                for i in 0..(first_valid + period).min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at idx {}, got {}",
                        i,
                        out[i]
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_rocp_tests!(check_rocp_property);
}
