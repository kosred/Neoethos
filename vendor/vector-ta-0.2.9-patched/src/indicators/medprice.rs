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
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MedpriceData<'a> {
    Candles {
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MedpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct MedpriceParams;

#[derive(Debug, Clone)]
pub struct MedpriceInput<'a> {
    pub data: MedpriceData<'a>,
    pub params: MedpriceParams,
}

impl<'a> MedpriceInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
        params: MedpriceParams,
    ) -> Self {
        Self {
            data: MedpriceData::Candles {
                candles,
                high_source,
                low_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MedpriceParams) -> Self {
        Self {
            data: MedpriceData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "high", "low", MedpriceParams::default())
    }

    #[inline]
    pub fn get_high_low(&self) -> (&[f64], &[f64]) {
        match &self.data {
            MedpriceData::Candles {
                candles,
                high_source,
                low_source,
            } => (
                medprice_source(candles, high_source),
                medprice_source(candles, low_source),
            ),
            MedpriceData::Slices { high, low } => (high, low),
        }
    }
}

#[inline(always)]
fn medprice_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
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
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MedpriceBuilder {
    kernel: Kernel,
}

impl Default for MedpriceBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl MedpriceBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<MedpriceOutput, MedpriceError> {
        let input = MedpriceInput::with_default_candles(candles);
        medprice_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MedpriceOutput, MedpriceError> {
        let input = MedpriceInput::from_slices(high, low, MedpriceParams::default());
        medprice_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<MedpriceStream, MedpriceError> {
        MedpriceStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum MedpriceError {
    #[error("medprice: Empty data provided.")]
    EmptyInputData,
    #[error("medprice: Different lengths for high ({high_len}) and low ({low_len}).")]
    DifferentLength { high_len: usize, low_len: usize },
    #[error("medprice: All values are NaN.")]
    AllValuesNaN,
    #[error("medprice: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("medprice: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("medprice: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("medprice: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("medprice: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn medprice(input: &MedpriceInput) -> Result<MedpriceOutput, MedpriceError> {
    medprice_with_kernel(input, Kernel::Auto)
}

pub fn medprice_with_kernel(
    input: &MedpriceInput,
    kernel: Kernel,
) -> Result<MedpriceOutput, MedpriceError> {
    let (high, low) = input.get_high_low();

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first_valid_idx = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MedpriceError::AllValuesNaN)?;

    let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx);

    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                medprice_scalar(high, low, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first_valid_idx, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                medprice_avx512(high, low, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(MedpriceOutput { values: out })
}

#[inline]
pub fn medprice_into(input: &MedpriceInput, out: &mut [f64]) -> Result<(), MedpriceError> {
    medprice_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
pub fn medprice_compute_into(
    high: &[f64],
    low: &[f64],
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), MedpriceError> {
    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if out.len() != high.len() {
        return Err(MedpriceError::OutputLengthMismatch {
            expected: high.len(),
            got: out.len(),
        });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MedpriceError::AllValuesNaN)?;

    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => medprice_scalar(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => medprice_avx512(high, low, first, out),
            _ => unreachable!(),
        }
    }

    out[..first].fill(f64::NAN);
    Ok(())
}

#[inline]
pub fn medprice_scalar(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    let n = high.len();
    if first >= n {
        return;
    }
    unsafe {
        let mut hp = high.as_ptr().add(first);
        let mut lp = low.as_ptr().add(first);
        let mut op = out.as_mut_ptr().add(first);
        let end = high.as_ptr().add(n);
        while hp < end {
            *op = (*hp + *lp) * 0.5;
            hp = hp.add(1);
            lp = lp.add(1);
            op = op.add(1);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medprice_avx2(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_body(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
        let n = high.len();
        let mut i = first;

        let mut hp = high.as_ptr().add(first);
        let mut lp = low.as_ptr().add(first);
        let mut op = out.as_mut_ptr().add(first);

        let vhalf = _mm256_set1_pd(0.5);

        while i + 16 <= n {
            let h0 = _mm256_loadu_pd(hp);
            let l0 = _mm256_loadu_pd(lp);
            let h1 = _mm256_loadu_pd(hp.add(4));
            let l1 = _mm256_loadu_pd(lp.add(4));
            let h2 = _mm256_loadu_pd(hp.add(8));
            let l2 = _mm256_loadu_pd(lp.add(8));
            let h3 = _mm256_loadu_pd(hp.add(12));
            let l3 = _mm256_loadu_pd(lp.add(12));

            let r0 = _mm256_mul_pd(_mm256_add_pd(h0, l0), vhalf);
            let r1 = _mm256_mul_pd(_mm256_add_pd(h1, l1), vhalf);
            let r2 = _mm256_mul_pd(_mm256_add_pd(h2, l2), vhalf);
            let r3 = _mm256_mul_pd(_mm256_add_pd(h3, l3), vhalf);

            _mm256_storeu_pd(op, r0);
            _mm256_storeu_pd(op.add(4), r1);
            _mm256_storeu_pd(op.add(8), r2);
            _mm256_storeu_pd(op.add(12), r3);

            hp = hp.add(16);
            lp = lp.add(16);
            op = op.add(16);
            i += 16;
        }

        while i + 8 <= n {
            let h0 = _mm256_loadu_pd(hp);
            let l0 = _mm256_loadu_pd(lp);
            let h1 = _mm256_loadu_pd(hp.add(4));
            let l1 = _mm256_loadu_pd(lp.add(4));

            let r0 = _mm256_mul_pd(_mm256_add_pd(h0, l0), vhalf);
            let r1 = _mm256_mul_pd(_mm256_add_pd(h1, l1), vhalf);

            _mm256_storeu_pd(op, r0);
            _mm256_storeu_pd(op.add(4), r1);

            hp = hp.add(8);
            lp = lp.add(8);
            op = op.add(8);
            i += 8;
        }

        while i + 4 <= n {
            let h = _mm256_loadu_pd(hp);
            let l = _mm256_loadu_pd(lp);
            let r = _mm256_mul_pd(_mm256_add_pd(h, l), vhalf);
            _mm256_storeu_pd(op, r);

            hp = hp.add(4);
            lp = lp.add(4);
            op = op.add(4);
            i += 4;
        }

        while i < n {
            *op = (*hp + *lp) * 0.5;
            hp = hp.add(1);
            lp = lp.add(1);
            op = op.add(1);
            i += 1;
        }
    }

    unsafe { avx2_body(high, low, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medprice_avx512(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_body(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
        let n = high.len();
        let mut i = first;

        let mut hp = high.as_ptr().add(first);
        let mut lp = low.as_ptr().add(first);
        let mut op = out.as_mut_ptr().add(first);

        let vhalf = _mm512_set1_pd(0.5);

        while i + 32 <= n {
            let h0 = _mm512_loadu_pd(hp);
            let l0 = _mm512_loadu_pd(lp);
            let h1 = _mm512_loadu_pd(hp.add(8));
            let l1 = _mm512_loadu_pd(lp.add(8));
            let h2 = _mm512_loadu_pd(hp.add(16));
            let l2 = _mm512_loadu_pd(lp.add(16));
            let h3 = _mm512_loadu_pd(hp.add(24));
            let l3 = _mm512_loadu_pd(lp.add(24));

            let r0 = _mm512_mul_pd(_mm512_add_pd(h0, l0), vhalf);
            let r1 = _mm512_mul_pd(_mm512_add_pd(h1, l1), vhalf);
            let r2 = _mm512_mul_pd(_mm512_add_pd(h2, l2), vhalf);
            let r3 = _mm512_mul_pd(_mm512_add_pd(h3, l3), vhalf);

            _mm512_storeu_pd(op, r0);
            _mm512_storeu_pd(op.add(8), r1);
            _mm512_storeu_pd(op.add(16), r2);
            _mm512_storeu_pd(op.add(24), r3);

            hp = hp.add(32);
            lp = lp.add(32);
            op = op.add(32);
            i += 32;
        }

        while i + 16 <= n {
            let h0 = _mm512_loadu_pd(hp);
            let l0 = _mm512_loadu_pd(lp);
            let h1 = _mm512_loadu_pd(hp.add(8));
            let l1 = _mm512_loadu_pd(lp.add(8));

            let r0 = _mm512_mul_pd(_mm512_add_pd(h0, l0), vhalf);
            let r1 = _mm512_mul_pd(_mm512_add_pd(h1, l1), vhalf);

            _mm512_storeu_pd(op, r0);
            _mm512_storeu_pd(op.add(8), r1);

            hp = hp.add(16);
            lp = lp.add(16);
            op = op.add(16);
            i += 16;
        }

        while i + 8 <= n {
            let h = _mm512_loadu_pd(hp);
            let l = _mm512_loadu_pd(lp);
            let r = _mm512_mul_pd(_mm512_add_pd(h, l), vhalf);
            _mm512_storeu_pd(op, r);

            hp = hp.add(8);
            lp = lp.add(8);
            op = op.add(8);
            i += 8;
        }

        while i < n {
            *op = (*hp + *lp) * 0.5;
            hp = hp.add(1);
            lp = lp.add(1);
            op = op.add(1);
            i += 1;
        }
    }

    unsafe { avx512_body(high, low, first, out) }
}

#[inline(always)]
pub unsafe fn medprice_row_scalar(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_scalar(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx2(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx2(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx512(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512_short(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx512(high, low, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn medprice_row_avx512_long(high: &[f64], low: &[f64], first: usize, out: &mut [f64]) {
    medprice_avx512(high, low, first, out)
}

#[derive(Debug, Clone)]
pub struct MedpriceStream {
    started: bool,
}

impl MedpriceStream {
    pub fn try_new() -> Result<Self, MedpriceError> {
        Ok(Self { started: false })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() {
            return None;
        }
        Some((high + low) * 0.5)
    }
}

#[derive(Clone, Debug)]
pub struct MedpriceBatchRange {
    pub dummy: (usize, usize, usize),
}
impl Default for MedpriceBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MedpriceBatchBuilder {
    kernel: Kernel,
    range: MedpriceBatchRange,
}

impl MedpriceBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn dummy_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.dummy = range;
        self
    }
    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<MedpriceBatchOutput, MedpriceError> {
        let _ = expand_grid(&self.range)?;
        medprice_batch_with_kernel(high, low, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        high_src: &str,
        low_src: &str,
    ) -> Result<MedpriceBatchOutput, MedpriceError> {
        let high = source_type(c, high_src);
        let low = source_type(c, low_src);
        self.apply_slice(high, low)
    }
}

pub fn medprice_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    k: Kernel,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(MedpriceError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    medprice_batch_par_slice(high, low, simd)
}

#[derive(Clone, Debug)]
pub struct MedpriceBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, MedpriceError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut v = start;
        while v <= end {
            out.push(v);
            match v.checked_add(step) {
                Some(next) if next != v => v = next,
                _ => break,
            }
        }
    } else {
        let mut v = start;
        loop {
            out.push(v);
            if v <= end {
                break;
            }
            let dec = v.saturating_sub(step);
            if dec == v {
                break;
            }
            v = dec;
        }
        out.sort_unstable();
    }
    if out.is_empty() {
        return Err(MedpriceError::InvalidRange { start, end, step });
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid(range: &MedpriceBatchRange) -> Result<Vec<MedpriceParams>, MedpriceError> {
    let rows_axis = axis_usize(range.dummy)?;

    Ok(rows_axis
        .into_iter()
        .map(|_| MedpriceParams::default())
        .collect())
}

#[inline(always)]
pub fn medprice_batch_slice(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    medprice_batch_inner(high, low, kern, false)
}

#[inline(always)]
pub fn medprice_batch_par_slice(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    medprice_batch_inner(high, low, kern, true)
}

#[inline(always)]
fn medprice_batch_inner(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<MedpriceBatchOutput, MedpriceError> {
    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MedpriceError::AllValuesNaN)?;

    let rows: usize = 1;
    let cols: usize = high.len();
    let _total = rows.checked_mul(cols).ok_or(MedpriceError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmup_periods = vec![first];
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let chosen = match kern {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    medprice_batch_inner_into(high, low, chosen, _parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(MedpriceBatchOutput { values, rows, cols })
}

#[inline(always)]
fn medprice_batch_inner_into(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MedpriceParams>, MedpriceError> {
    let combos = vec![MedpriceParams::default()];

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MedpriceError::AllValuesNaN)?;

    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => medprice_scalar(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => medprice_avx512(high, low, first, out),
            _ => unreachable!(),
        }
    }

    Ok(combos)
}

#[inline]
pub fn medprice_into_slice_raw(
    dst: &mut [f64],
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<(), MedpriceError> {
    medprice_compute_into(high, low, kern, dst)
}

#[inline]
pub fn medprice_into_slice(
    dst: &mut [f64],
    input: &MedpriceInput,
    kern: Kernel,
) -> Result<(), MedpriceError> {
    let (high, low) = input.get_high_low();

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if dst.len() != high.len() {
        return Err(MedpriceError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }

    let first_valid_idx = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MedpriceError::AllValuesNaN)?;

    let chosen = match kern {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                medprice_scalar(high, low, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medprice_avx2(high, low, first_valid_idx, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                medprice_avx512(high, low, first_valid_idx, dst)
            }
            _ => unreachable!(),
        }
    }

    dst[..first_valid_idx].fill(f64::NAN);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use crate::utilities::enums::Kernel;

    fn check_medprice_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MedpriceInput::with_default_candles(&candles);
        let output = medprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_medprice_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MedpriceInput::from_candles(&candles, "high", "low", MedpriceParams);
        let result = medprice_with_kernel(&input, kernel)?;
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Output length mismatch"
        );
        let expected_last_five = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        assert!(result.values.len() >= 5, "Not enough data for comparison");
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (val - expected).abs() < 1e-1,
                "Mismatch at last five index {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
        Ok(())
    }

    fn check_medprice_empty_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [];
        let low = [];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for empty data");
        Ok(())
    }

    fn check_medprice_different_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for different slice lengths"
        );
        Ok(())
    }

    fn check_medprice_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all NaN data");
        Ok(())
    }

    fn check_medprice_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 100.0, 110.0];
        let low = [f64::NAN, 80.0, 90.0];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 3);
        assert!(result.values[0].is_nan());
        assert_eq!(result.values[1], 90.0);
        assert_eq!(result.values[2], 100.0);
        Ok(())
    }

    fn check_medprice_late_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 110.0, f64::NAN];
        let low = [80.0, 90.0, f64::NAN];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.values[0], 90.0);
        assert_eq!(result.values[1], 100.0);
        assert!(result.values[2].is_nan());
        Ok(())
    }

    fn check_medprice_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 110.0, 120.0];
        let low = [80.0, 90.0, 100.0];
        let mut stream = MedpriceStream::try_new()?;
        let mut values = Vec::with_capacity(high.len());
        for (&h, &l) in high.iter().zip(low.iter()) {
            values.push(stream.update(h, l));
        }
        assert_eq!(values[0], Some(90.0));
        assert_eq!(values[1], Some(100.0));
        assert_eq!(values[2], Some(110.0));
        Ok(())
    }

    fn check_medprice_batch(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 110.0, 120.0];
        let low = [80.0, 90.0, 100.0];
        let builder = MedpriceBatchBuilder::new().kernel(kernel);
        let batch = builder.apply_slice(&high, &low)?;
        assert_eq!(batch.values.len(), high.len());
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, 3);
        assert_eq!(batch.values, vec![90.0, 100.0, 110.0]);
        Ok(())
    }

    macro_rules! generate_all_medprice_tests {
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

    generate_all_medprice_tests!(
        check_medprice_with_default_candles,
        check_medprice_accuracy,
        check_medprice_empty_data,
        check_medprice_different_length,
        check_medprice_all_values_nan,
        check_medprice_nan_handling,
        check_medprice_late_nan_handling,
        check_medprice_streaming,
        check_medprice_batch
    );
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let high = source_type(&c, "high");
        let low = source_type(&c, "low");

        let output = MedpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(high, low)?;

        assert_eq!(output.rows, 1, "[{test}] batch output should have one row");
        assert_eq!(output.cols, high.len(), "[{test}] batch cols mismatch");
        assert_eq!(
            output.values.len(),
            output.cols,
            "[{test}] values shape mismatch"
        );

        let last_expected = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            assert!(
                (val - last_expected[i]).abs() < 1e-1,
                "[{test}] batch last-five mismatch idx {i}: got {val}, expected {}",
                last_expected[i]
            );
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);

    #[test]
    fn test_medprice_into_matches_api() {
        let n = 256usize;
        let mut ts: Vec<i64> = (0..n as i64).collect();
        let mut open = vec![0.0; n];
        let mut high = vec![0.0; n];
        let mut low = vec![0.0; n];
        let mut close = vec![0.0; n];
        let mut vol = vec![1.0; n];

        for i in 0..n {
            if i < 3 {
                high[i] = f64::NAN;
                low[i] = f64::NAN;
                open[i] = f64::NAN;
                close[i] = f64::NAN;
            } else {
                let x = i as f64;
                low[i] = 95.0 + (x.sin() * 2.0);
                high[i] = low[i] + 10.0 + (x.cos());
                open[i] = low[i] + 3.0;
                close[i] = high[i] - 4.0;
            }
        }

        let candles = crate::utilities::data_loader::Candles::new(
            ts.clone(),
            open,
            high.clone(),
            low.clone(),
            close,
            vol,
        );
        let input = MedpriceInput::with_default_candles(&candles);

        let baseline = medprice(&input).expect("baseline medprice failed").values;

        let mut out = vec![0.0; baseline.len()];
        {
            medprice_into(&input, &mut out).expect("medprice_into failed");
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "value mismatch at {}: baseline={:?}, into={:?}",
                i,
                baseline[i],
                out[i]
            );
        }
    }
}
