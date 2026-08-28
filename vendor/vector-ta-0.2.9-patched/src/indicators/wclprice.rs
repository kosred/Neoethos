use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WclpriceData<'a> {
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
pub struct WclpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WclpriceParams;

impl Default for WclpriceParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct WclpriceInput<'a> {
    pub data: WclpriceData<'a>,
    pub params: WclpriceParams,
}

impl<'a> WclpriceInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WclpriceData::Candles { candles },
            params: WclpriceParams::default(),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WclpriceData::Slices { high, low, close },
            params: WclpriceParams::default(),
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WclpriceBuilder {
    kernel: Kernel,
}
impl Default for WclpriceBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}
impl WclpriceBuilder {
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
    pub fn apply(self, candles: &Candles) -> Result<WclpriceOutput, WclpriceError> {
        let i = WclpriceInput::from_candles(candles);
        wclprice_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WclpriceOutput, WclpriceError> {
        let i = WclpriceInput::from_slices(high, low, close);
        wclprice_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn into_stream(self) -> WclpriceStream {
        WclpriceStream::default()
    }
}

#[derive(Debug, Error)]
pub enum WclpriceError {
    #[error("wclprice: empty input")]
    EmptyInputData,
    #[error("wclprice: all values are NaN")]
    AllValuesNaN,
    #[error("wclprice: invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("wclprice: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wclprice: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("wclprice: invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("wclprice: invalid kernel for batch mode: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("wclprice: missing candle field '{field}'")]
    MissingField { field: &'static str },
}

#[inline(always)]
fn wclprice_prepare<'a>(
    input: &'a WclpriceInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), WclpriceError> {
    let (high, low, close) = match &input.data {
        WclpriceData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        WclpriceData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyInputData);
    }
    let lh = high.len();
    let ll = low.len();
    let lc = close.len();
    let len = lh.min(ll).min(lc);

    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::ScalarBatch => Kernel::Scalar,
        k => k,
    };
    Ok((high, low, close, len, first, chosen))
}

#[inline]
pub fn wclprice(input: &WclpriceInput) -> Result<WclpriceOutput, WclpriceError> {
    wclprice_with_kernel(input, Kernel::Auto)
}

pub fn wclprice_with_kernel(
    input: &WclpriceInput,
    kernel: Kernel,
) -> Result<WclpriceOutput, WclpriceError> {
    let (high, low, close, len, first, chosen) = wclprice_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(len, first);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                wclprice_scalar(high, low, close, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wclprice_avx2(high, low, close, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wclprice_avx512(high, low, close, first, &mut out)
            }
            _ => wclprice_scalar(high, low, close, first, &mut out),
        }
    }
    Ok(WclpriceOutput { values: out })
}

#[inline]

pub fn wclprice_into(input: &WclpriceInput, out: &mut [f64]) -> Result<(), WclpriceError> {
    wclprice_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn wclprice_into_slice(
    dst: &mut [f64],
    input: &WclpriceInput,
    kern: Kernel,
) -> Result<(), WclpriceError> {
    let (high, low, close, len, first, chosen) = wclprice_prepare(input, kern)?;
    if dst.len() != len {
        return Err(WclpriceError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    if first > 0 {
        dst[..first].fill(f64::NAN);
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wclprice_scalar(high, low, close, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wclprice_avx2(high, low, close, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wclprice_avx512(high, low, close, first, dst),
            _ => wclprice_scalar(high, low, close, first, dst),
        }
    }
    Ok(())
}

#[inline]
pub fn wclprice_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    let len = high.len().min(low.len()).min(close.len());
    debug_assert_eq!(out.len(), len);

    const HALF: f64 = 0.5;
    const QUARTER: f64 = 0.25;

    let mut i = first_valid;
    let end = len;
    while i + 8 <= end {
        let h0 = high[i + 0];
        let l0 = low[i + 0];
        let c0 = close[i + 0];
        out[i + 0] = c0.mul_add(HALF, (h0 + l0) * QUARTER);

        let h1 = high[i + 1];
        let l1 = low[i + 1];
        let c1 = close[i + 1];
        out[i + 1] = c1.mul_add(HALF, (h1 + l1) * QUARTER);

        let h2 = high[i + 2];
        let l2 = low[i + 2];
        let c2 = close[i + 2];
        out[i + 2] = c2.mul_add(HALF, (h2 + l2) * QUARTER);

        let h3 = high[i + 3];
        let l3 = low[i + 3];
        let c3 = close[i + 3];
        out[i + 3] = c3.mul_add(HALF, (h3 + l3) * QUARTER);

        let h4 = high[i + 4];
        let l4 = low[i + 4];
        let c4 = close[i + 4];
        out[i + 4] = c4.mul_add(HALF, (h4 + l4) * QUARTER);

        let h5 = high[i + 5];
        let l5 = low[i + 5];
        let c5 = close[i + 5];
        out[i + 5] = c5.mul_add(HALF, (h5 + l5) * QUARTER);

        let h6 = high[i + 6];
        let l6 = low[i + 6];
        let c6 = close[i + 6];
        out[i + 6] = c6.mul_add(HALF, (h6 + l6) * QUARTER);

        let h7 = high[i + 7];
        let l7 = low[i + 7];
        let c7 = close[i + 7];
        out[i + 7] = c7.mul_add(HALF, (h7 + l7) * QUARTER);

        i += 8;
    }
    while i + 4 <= end {
        let h0 = high[i];
        let l0 = low[i];
        let c0 = close[i];
        out[i] = c0.mul_add(HALF, (h0 + l0) * QUARTER);

        let h1 = high[i + 1];
        let l1 = low[i + 1];
        let c1 = close[i + 1];
        out[i + 1] = c1.mul_add(HALF, (h1 + l1) * QUARTER);

        let h2 = high[i + 2];
        let l2 = low[i + 2];
        let c2 = close[i + 2];
        out[i + 2] = c2.mul_add(HALF, (h2 + l2) * QUARTER);

        let h3 = high[i + 3];
        let l3 = low[i + 3];
        let c3 = close[i + 3];
        out[i + 3] = c3.mul_add(HALF, (h3 + l3) * QUARTER);

        i += 4;
    }
    while i < end {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        out[i] = c.mul_add(HALF, (h + l) * QUARTER);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn wclprice_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len().min(low.len()).min(close.len());
    debug_assert_eq!(out.len(), len);

    let mut i = first_valid;
    let end = len;

    let vhalf = _mm256_set1_pd(0.5);
    let vquart = _mm256_set1_pd(0.25);

    const STEP: usize = 4;

    while i + 2 * STEP <= end {
        let h0 = _mm256_loadu_pd(high.as_ptr().add(i));
        let l0 = _mm256_loadu_pd(low.as_ptr().add(i));
        let c0 = _mm256_loadu_pd(close.as_ptr().add(i));
        let hl0 = _mm256_add_pd(h0, l0);
        let t0 = _mm256_mul_pd(hl0, vquart);

        let h1 = _mm256_loadu_pd(high.as_ptr().add(i + STEP));
        let l1 = _mm256_loadu_pd(low.as_ptr().add(i + STEP));
        let c1 = _mm256_loadu_pd(close.as_ptr().add(i + STEP));
        let hl1 = _mm256_add_pd(h1, l1);
        let t1 = _mm256_mul_pd(hl1, vquart);

        let y0 = _mm256_fmadd_pd(c0, vhalf, t0);
        let y1 = _mm256_fmadd_pd(c1, vhalf, t1);

        _mm256_storeu_pd(out.as_mut_ptr().add(i), y0);
        _mm256_storeu_pd(out.as_mut_ptr().add(i + STEP), y1);

        i += 2 * STEP;
    }

    while i + STEP <= end {
        let h = _mm256_loadu_pd(high.as_ptr().add(i));
        let l = _mm256_loadu_pd(low.as_ptr().add(i));
        let c = _mm256_loadu_pd(close.as_ptr().add(i));
        let hl = _mm256_add_pd(h, l);
        let t = _mm256_mul_pd(hl, vquart);
        let y = _mm256_fmadd_pd(c, vhalf, t);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), y);
        i += STEP;
    }
    while i < end {
        let h = *high.get_unchecked(i);
        let l = *low.get_unchecked(i);
        let c = *close.get_unchecked(i);
        *out.get_unchecked_mut(i) = c.mul_add(0.5, (h + l) * 0.25);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn wclprice_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len().min(low.len()).min(close.len());
    debug_assert_eq!(out.len(), len);

    let mut i = first_valid;
    let end = len;

    let vhalf = _mm512_set1_pd(0.5);
    let vquart = _mm512_set1_pd(0.25);

    const STEP: usize = 8;

    while i + 2 * STEP <= end {
        let h0 = _mm512_loadu_pd(high.as_ptr().add(i));
        let l0 = _mm512_loadu_pd(low.as_ptr().add(i));
        let c0 = _mm512_loadu_pd(close.as_ptr().add(i));
        let hl0 = _mm512_add_pd(h0, l0);
        let t0 = _mm512_mul_pd(hl0, vquart);

        let h1 = _mm512_loadu_pd(high.as_ptr().add(i + STEP));
        let l1 = _mm512_loadu_pd(low.as_ptr().add(i + STEP));
        let c1 = _mm512_loadu_pd(close.as_ptr().add(i + STEP));
        let hl1 = _mm512_add_pd(h1, l1);
        let t1 = _mm512_mul_pd(hl1, vquart);

        let y0 = _mm512_fmadd_pd(c0, vhalf, t0);
        let y1 = _mm512_fmadd_pd(c1, vhalf, t1);

        _mm512_storeu_pd(out.as_mut_ptr().add(i), y0);
        _mm512_storeu_pd(out.as_mut_ptr().add(i + STEP), y1);

        i += 2 * STEP;
    }

    while i + STEP <= end {
        let h = _mm512_loadu_pd(high.as_ptr().add(i));
        let l = _mm512_loadu_pd(low.as_ptr().add(i));
        let c = _mm512_loadu_pd(close.as_ptr().add(i));
        let hl = _mm512_add_pd(h, l);
        let t = _mm512_mul_pd(hl, vquart);
        let y = _mm512_fmadd_pd(c, vhalf, t);
        _mm512_storeu_pd(out.as_mut_ptr().add(i), y);
        i += STEP;
    }
    while i < end {
        let h = *high.get_unchecked(i);
        let l = *low.get_unchecked(i);
        let c = *close.get_unchecked(i);
        *out.get_unchecked_mut(i) = c.mul_add(0.5, (h + l) * 0.25);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[inline]
pub fn wclprice_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_scalar(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { wclprice_avx2(high, low, close, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { wclprice_avx512(high, low, close, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512_short(high, low, close, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wclprice_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first_valid: usize,
    out: &mut [f64],
) {
    wclprice_avx512_long(high, low, close, first_valid, out)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchRange;

impl Default for WclpriceBatchRange {
    fn default() -> Self {
        Self
    }
}

#[derive(Clone, Debug, Default)]
pub struct WclpriceBatchBuilder {
    kernel: Kernel,
}
impl WclpriceBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WclpriceBatchOutput, WclpriceError> {
        wclprice_batch_with_kernel(high, low, close, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
        let h = c
            .select_candle_field("high")
            .map_err(|_| WclpriceError::MissingField { field: "high" })?;
        let l = c
            .select_candle_field("low")
            .map_err(|_| WclpriceError::MissingField { field: "low" })?;
        let cl = c
            .select_candle_field("close")
            .map_err(|_| WclpriceError::MissingField { field: "close" })?;
        self.apply_slices(h, l, cl)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WclpriceBatchOutput, WclpriceError> {
        WclpriceBatchBuilder::new().apply_candles(c)
    }
}

pub fn wclprice_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(WclpriceError::InvalidKernelForBatch(other)),
    };
    wclprice_batch_par_slice(high, low, close, kernel)
}

#[derive(Clone, Debug)]
pub struct WclpriceBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WclpriceParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WclpriceBatchOutput {
    pub fn values_for(&self, _params: &WclpriceParams) -> Option<&[f64]> {
        if self.rows == 1 {
            Some(&self.values[..self.cols])
        } else {
            None
        }
    }
}

#[inline(always)]
pub fn wclprice_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    wclprice_batch_inner(high, low, close, kern, false)
}
#[inline(always)]
pub fn wclprice_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    wclprice_batch_inner(high, low, close, kern, true)
}
#[inline(always)]
pub fn wclprice_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<WclpriceBatchOutput, WclpriceError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyInputData);
    }
    let len = high.len().min(low.len()).min(close.len());
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    let mut buf_mu = make_uninit_matrix(1, len);
    init_matrix_prefixes(&mut buf_mu, len, &[first]);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let simd = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let map = match simd {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        other => other,
    };

    wclprice_batch_inner_into(high, low, close, map, _parallel, out_slice)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(WclpriceBatchOutput {
        values,
        combos: vec![WclpriceParams],
        rows: 1,
        cols: len,
    })
}

#[inline(always)]
fn expand_grid(_r: &WclpriceBatchRange) -> Vec<WclpriceParams> {
    vec![WclpriceParams]
}

#[inline(always)]
fn wclprice_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<WclpriceParams>, WclpriceError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyInputData);
    }
    let len = high.len().min(low.len()).min(close.len());
    if out.len() < len {
        return Err(WclpriceError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(WclpriceError::AllValuesNaN)?;

    if first > 0 {
        out[..first].fill(f64::NAN);
    }

    unsafe {
        match kern {
            Kernel::Scalar => wclprice_row_scalar(high, low, close, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => wclprice_row_avx2(high, low, close, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => wclprice_row_avx512(high, low, close, first, out),
            _ => wclprice_row_scalar(high, low, close, first, out),
        }
    }

    Ok(vec![WclpriceParams])
}

#[derive(Debug, Clone)]
pub struct WclpriceStream;
impl Default for WclpriceStream {
    fn default() -> Self {
        Self
    }
}
impl WclpriceStream {
    #[inline(always)]
    pub fn update(&mut self, h: f64, l: f64, c: f64) -> Option<f64> {
        if h.is_nan() | l.is_nan() | c.is_nan() {
            return None;
        }

        Some(c.mul_add(0.5, (h + l) * 0.25))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_wclprice_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file)?;

        let input = WclpriceInput::from_candles(&candles);

        let WclpriceOutput { values: expected } = wclprice(&input)?;

        let mut out = vec![0.0f64; expected.len()];
        wclprice_into(&input, &mut out)?;

        assert_eq!(out.len(), expected.len());
        for i in 0..expected.len() {
            let a = expected[i];
            let b = out[i];
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(equal, "mismatch at {}: expected={}, got={}", i, a, b);
        }
        Ok(())
    }

    fn check_wclprice_slices(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![59230.0, 59220.0, 59077.0, 59160.0, 58717.0];
        let low = vec![59222.0, 59211.0, 59077.0, 59143.0, 58708.0];
        let close = vec![59225.0, 59210.0, 59080.0, 59150.0, 58710.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice_with_kernel(&input, kernel)?;
        let expected = vec![59225.5, 59212.75, 59078.5, 59150.75, 58711.25];
        for (i, &v) in output.values.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
                "[{test}] mismatch at {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }
    fn check_wclprice_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file)?;
        let input = WclpriceInput::from_candles(&candles);
        let output = wclprice_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_wclprice_empty_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let res = wclprice_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with empty data", test);
        Ok(())
    }
    fn check_wclprice_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![f64::NAN, f64::NAN];
        let low = vec![f64::NAN, f64::NAN];
        let close = vec![f64::NAN, f64::NAN];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let res = wclprice_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with all NaN", test);
        Ok(())
    }
    fn check_wclprice_partial_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = vec![f64::NAN, 59000.0];
        let low = vec![f64::NAN, 58950.0];
        let close = vec![f64::NAN, 58975.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice_with_kernel(&input, kernel)?;
        assert!(output.values[0].is_nan());
        assert!((output.values[1] - (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0).abs() < 1e-8);
        Ok(())
    }

    fn check_wclprice_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = WclpriceInput::from_candles(&candles);
        let result = wclprice_with_kernel(&input, kernel)?;

        let expected_last_five = [59225.5, 59212.75, 59078.5, 59150.75, 58711.25];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] WCLPRICE {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_wclprice_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = WclpriceInput::from_candles(&candles);
        let output = wclprice_with_kernel(&input, kernel)?;

        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
					 in WCLPRICE with candle data",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
					 in WCLPRICE with candle data",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
					 in WCLPRICE with candle data",
                    test_name, val, bits, i
                );
            }
        }

        let test_patterns = vec![
            (
                "small dataset",
                vec![100.0, 101.0, 102.0],
                vec![99.0, 100.0, 101.0],
                vec![100.5, 101.5, 102.5],
            ),
            (
                "large values",
                vec![59000.0, 60000.0, 61000.0],
                vec![58900.0, 59900.0, 60900.0],
                vec![58950.0, 59950.0, 60950.0],
            ),
            (
                "with leading NaN",
                vec![f64::NAN, 100.0, 101.0],
                vec![f64::NAN, 99.0, 100.0],
                vec![f64::NAN, 100.5, 101.5],
            ),
            (
                "with trailing NaN",
                vec![100.0, 101.0, f64::NAN],
                vec![99.0, 100.0, f64::NAN],
                vec![100.5, 101.5, f64::NAN],
            ),
            (
                "mixed NaN pattern",
                vec![100.0, f64::NAN, 101.0, f64::NAN, 102.0],
                vec![99.0, f64::NAN, 100.0, f64::NAN, 101.0],
                vec![100.5, f64::NAN, 101.5, f64::NAN, 102.5],
            ),
            (
                "single valid value",
                vec![f64::NAN, f64::NAN, 100.0],
                vec![f64::NAN, f64::NAN, 99.0],
                vec![f64::NAN, f64::NAN, 100.5],
            ),
            (
                "extreme values",
                vec![1e-10, 1e10, 1e-10],
                vec![1e-10, 1e10, 1e-10],
                vec![1e-10, 1e10, 1e-10],
            ),
            (
                "zero values",
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.5, 0.0],
            ),
            (
                "negative values",
                vec![-100.0, -50.0, -25.0],
                vec![-101.0, -51.0, -26.0],
                vec![-100.5, -50.5, -25.5],
            ),
            (
                "large dataset",
                (0..1000).map(|i| 100.0 + i as f64).collect(),
                (0..1000).map(|i| 99.0 + i as f64).collect(),
                (0..1000).map(|i| 100.5 + i as f64).collect(),
            ),
        ];

        for (pattern_idx, (desc, high, low, close)) in test_patterns.iter().enumerate() {
            let input = WclpriceInput::from_slices(high, low, close);
            let output = wclprice_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in WCLPRICE with pattern '{}' (pattern {})",
                        test_name, val, bits, i, desc, pattern_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in WCLPRICE with pattern '{}' (pattern {})",
                        test_name, val, bits, i, desc, pattern_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in WCLPRICE with pattern '{}' (pattern {})",
                        test_name, val, bits, i, desc, pattern_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_wclprice_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_wclprice_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=400).prop_flat_map(|len| {
            prop::collection::vec(
                (0.0f64..1e6f64)
                    .prop_filter("finite non-negative price", |x| x.is_finite() && *x >= 0.0)
                    .prop_flat_map(|low| {
                        (0.0f64..10000.0f64)
                            .prop_filter("finite diff", |x| x.is_finite())
                            .prop_flat_map(move |high_diff| {
                                let high = low + high_diff;
                                (
                                    Just(low),
                                    Just(high),
                                    (low..=high).prop_filter("finite close", |x| x.is_finite()),
                                )
                            })
                    }),
                len,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |price_data| {
                let mut high = Vec::with_capacity(price_data.len());
                let mut low = Vec::with_capacity(price_data.len());
                let mut close = Vec::with_capacity(price_data.len());

                for (l, h, c) in price_data.iter() {
                    low.push(*l);
                    high.push(*h);
                    close.push(*c);
                }

                let input = WclpriceInput::from_slices(&high, &low, &close);
                let WclpriceOutput { values: out } = wclprice_with_kernel(&input, kernel)?;

                let WclpriceOutput { values: ref_out } =
                    wclprice_with_kernel(&input, Kernel::Scalar)?;

                for i in 0..price_data.len() {
                    let h = high[i];
                    let l = low[i];
                    let c = close[i];
                    let y = out[i];
                    let r = ref_out[i];

                    if h.is_finite() && l.is_finite() && c.is_finite() {
                        let expected = (h + l + 2.0 * c) / 4.0;
                        prop_assert!(
                            (y - expected).abs() <= 1e-9,
                            "Formula mismatch at idx {}: got {} expected {} (h={}, l={}, c={})",
                            i,
                            y,
                            expected,
                            h,
                            l,
                            c
                        );
                    } else {
                        prop_assert!(
                            y.is_nan(),
                            "Expected NaN at idx {} when input has non-finite values, got {}",
                            i,
                            y
                        );
                    }

                    if h.is_finite() && l.is_finite() && c.is_finite() {
                        let min_val = h.min(l).min(c);
                        let max_val = h.max(l).max(c);
                        prop_assert!(
                            y >= min_val - 1e-9 && y <= max_val + 1e-9,
                            "Output {} at idx {} outside bounds [{}, {}]",
                            y,
                            i,
                            min_val,
                            max_val
                        );
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y_bits == r_bits,
                            "NaN/infinite mismatch at idx {}: {} vs {} (bits: {:016x} vs {:016x})",
                            i,
                            y,
                            r,
                            y_bits,
                            r_bits
                        );
                    } else {
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

                    if (h - l).abs() < f64::EPSILON && (h - c).abs() < f64::EPSILON {
                        prop_assert!(
                            (y - h).abs() <= 1e-9,
                            "When all prices equal {}, WCLPRICE should be {}, got {}",
                            h,
                            h,
                            y
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_wclprice_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                   #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
    generate_all_wclprice_tests!(
        check_wclprice_slices,
        check_wclprice_candles,
        check_wclprice_empty_data,
        check_wclprice_all_nan,
        check_wclprice_partial_nan,
        check_wclprice_accuracy,
        check_wclprice_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_wclprice_tests!(check_wclprice_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = WclpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let row = output
            .values_for(&WclpriceParams)
            .expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = WclpriceBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, c.close.len());

        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
					 in WCLPRICE batch at index {} (candle data)",
                    test, val, bits, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) \
					 in WCLPRICE batch at index {} (candle data)",
                    test, val, bits, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) \
					 in WCLPRICE batch at index {} (candle data)",
                    test, val, bits, idx
                );
            }
        }

        let test_configs = vec![
            (
                "small data",
                vec![100.0, 101.0, 102.0],
                vec![99.0, 100.0, 101.0],
                vec![100.5, 101.5, 102.5],
            ),
            (
                "medium data",
                (0..100).map(|i| 100.0 + i as f64).collect(),
                (0..100).map(|i| 99.0 + i as f64).collect(),
                (0..100).map(|i| 100.5 + i as f64).collect(),
            ),
            (
                "large data",
                (0..5000).map(|i| 100.0 + (i as f64 * 0.1)).collect(),
                (0..5000).map(|i| 99.0 + (i as f64 * 0.1)).collect(),
                (0..5000).map(|i| 100.5 + (i as f64 * 0.1)).collect(),
            ),
            (
                "with NaN prefix",
                [
                    vec![f64::NAN; 10],
                    (0..90).map(|i| 100.0 + i as f64).collect(),
                ]
                .concat(),
                [
                    vec![f64::NAN; 10],
                    (0..90).map(|i| 99.0 + i as f64).collect(),
                ]
                .concat(),
                [
                    vec![f64::NAN; 10],
                    (0..90).map(|i| 100.5 + i as f64).collect(),
                ]
                .concat(),
            ),
            (
                "sparse NaN pattern",
                (0..50)
                    .map(|i| {
                        if i % 5 == 0 {
                            f64::NAN
                        } else {
                            100.0 + i as f64
                        }
                    })
                    .collect(),
                (0..50)
                    .map(|i| {
                        if i % 5 == 0 {
                            f64::NAN
                        } else {
                            99.0 + i as f64
                        }
                    })
                    .collect(),
                (0..50)
                    .map(|i| {
                        if i % 5 == 0 {
                            f64::NAN
                        } else {
                            100.5 + i as f64
                        }
                    })
                    .collect(),
            ),
            (
                "extreme values",
                vec![1e-100, 1e100, 1e-50, 1e50],
                vec![1e-100, 1e100, 1e-50, 1e50],
                vec![1e-100, 1e100, 1e-50, 1e50],
            ),
        ];

        for (cfg_idx, (desc, high, low, close)) in test_configs.iter().enumerate() {
            let output = WclpriceBatchBuilder::new()
                .kernel(kernel)
                .apply_slices(high, low, close)?;

            assert_eq!(
                output.rows, 1,
                "[{}] Config {}: Expected 1 row for WCLPRICE",
                test, cfg_idx
            );
            assert_eq!(
                output.cols,
                high.len(),
                "[{}] Config {}: Cols mismatch",
                test,
                cfg_idx
            );

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {} ({}): Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 in WCLPRICE batch at index {}",
                        test, cfg_idx, desc, val, bits, idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {} ({}): Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 in WCLPRICE batch at index {}",
                        test, cfg_idx, desc, val, bits, idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {} ({}): Found make_uninit_matrix poison value {} (0x{:016X}) \
						 in WCLPRICE batch at index {}",
                        test, cfg_idx, desc, val, bits, idx
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

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        }
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
