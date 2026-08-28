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

#[derive(Debug, Clone)]
pub enum NatrData<'a> {
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
pub struct NatrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NatrParams {
    pub period: Option<usize>,
}

impl Default for NatrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

// TA-Lib's private TA_IS_ZERO contract.  NATR uses it only for the close
// denominator; period one deliberately emits raw True Range.
const NATR_TA_EPSILON: f64 = 1.0e-14;

#[inline(always)]
fn natr_true_range(high: f64, low: f64, previous_close: f64) -> f64 {
    // TA-Lib compares and replaces in this exact order.  In particular, a
    // NaN initial high-low range is not rescued by either later comparison.
    let mut greatest = high - low;
    let high_distance = (previous_close - high).abs();
    if high_distance > greatest {
        greatest = high_distance;
    }
    let low_distance = (previous_close - low).abs();
    if low_distance > greatest {
        greatest = low_distance;
    }
    greatest
}

#[inline(always)]
fn natr_wilder_step(previous: f64, true_range: f64, period: usize) -> f64 {
    // Preserve TA-Lib's stated three-operation rounding contract.  Do not
    // rewrite as an EMA delta or fused multiply-add.
    let mut next = previous;
    next *= (period - 1) as f64;
    next += true_range;
    next /= period as f64;
    next
}

#[inline(always)]
fn natr_output_value(atr: f64, close: f64, period: usize) -> f64 {
    if period <= 1 {
        atr
    } else if close > -NATR_TA_EPSILON && close < NATR_TA_EPSILON {
        0.0
    } else {
        (atr / close) * 100.0
    }
}

#[derive(Debug, Clone)]
pub struct NatrInput<'a> {
    pub data: NatrData<'a>,
    pub params: NatrParams,
}

impl<'a> NatrInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: NatrParams) -> Self {
        Self {
            data: NatrData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: NatrParams,
    ) -> Self {
        Self {
            data: NatrData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: NatrData::Candles { candles },
            params: NatrParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NatrBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for NatrBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NatrBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<NatrOutput, NatrError> {
        let p = NatrParams {
            period: self.period,
        };
        let i = NatrInput::from_candles(c, p);
        natr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<NatrOutput, NatrError> {
        let p = NatrParams {
            period: self.period,
        };
        let i = NatrInput::from_slices(high, low, close, p);
        natr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NatrStream, NatrError> {
        let p = NatrParams {
            period: self.period,
        };
        NatrStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum NatrError {
    #[error("natr: Empty data provided for NATR.")]
    EmptyInputData,
    #[error("natr: All values are NaN.")]
    AllValuesNaN,
    #[error("natr: Empty data provided for NATR.")]
    EmptyData,
    #[error("natr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("natr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("natr: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("natr: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("natr: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("natr: Mismatched lengths: expected = {expected}, actual = {actual}")]
    MismatchedLength { expected: usize, actual: usize },
}

#[inline]
pub fn natr(input: &NatrInput) -> Result<NatrOutput, NatrError> {
    natr_with_kernel(input, Kernel::Auto)
}

pub fn natr_with_kernel(input: &NatrInput, kernel: Kernel) -> Result<NatrOutput, NatrError> {
    let (high, low, close) = match &input.data {
        NatrData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            (high, low, close)
        }
        NatrData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(NatrError::EmptyInputData);
    }

    let len_h = high.len();
    let len_l = low.len();
    let len_c = close.len();
    if len_h != len_l || len_h != len_c {
        return Err(NatrError::MismatchedLength {
            expected: len_h,
            actual: if len_l != len_h { len_l } else { len_c },
        });
    }
    let len = len_h;

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = {
        let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
        let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
        let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

        match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
            (Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
            _ => None,
        }
    };

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(NatrError::AllValuesNaN),
    };

    if (len - first_valid_idx) <= period {
        return Err(NatrError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first_valid_idx,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first_valid_idx + period);

    let chosen = match kernel {
        Kernel::Auto => natr_auto_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                natr_scalar(high, low, close, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                natr_avx2(high, low, close, period, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                natr_avx512(high, low, close, period, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(NatrOutput { values: out })
}

#[inline(always)]
fn natr_auto_kernel() -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Kernel::Avx2;
        }
    }
    Kernel::Scalar
}

#[inline]
pub fn natr_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = out.len();
    if first >= len {
        return;
    }

    let warm_end = first + period;
    if warm_end >= len {
        return;
    }
    let mut sum_tr = 0.0;

    for i in (first + 1)..=warm_end {
        let hi = high[i];
        let lo = low[i];
        let pc = close[i - 1];
        let tr = natr_true_range(hi, lo, pc);
        sum_tr += tr;
    }

    let mut atr = sum_tr / period as f64;
    out[warm_end] = natr_output_value(atr, close[warm_end], period);

    let mut idx = warm_end + 1;
    while idx + 3 < len {
        let pc0 = close[idx - 1];
        let pc1 = close[idx + 0];
        let pc2 = close[idx + 1];
        let pc3 = close[idx + 2];

        let h0 = high[idx + 0];
        let h1 = high[idx + 1];
        let h2 = high[idx + 2];
        let h3 = high[idx + 3];
        let l0 = low[idx + 0];
        let l1 = low[idx + 1];
        let l2 = low[idx + 2];
        let l3 = low[idx + 3];

        let tr0 = natr_true_range(h0, l0, pc0);
        let tr1 = natr_true_range(h1, l1, pc1);
        let tr2 = natr_true_range(h2, l2, pc2);
        let tr3 = natr_true_range(h3, l3, pc3);

        atr = natr_wilder_step(atr, tr0, period);
        out[idx] = natr_output_value(atr, close[idx], period);

        atr = natr_wilder_step(atr, tr1, period);
        out[idx + 1] = natr_output_value(atr, close[idx + 1], period);

        atr = natr_wilder_step(atr, tr2, period);
        out[idx + 2] = natr_output_value(atr, close[idx + 2], period);

        atr = natr_wilder_step(atr, tr3, period);
        out[idx + 3] = natr_output_value(atr, close[idx + 3], period);

        idx += 4;
    }

    while idx < len {
        let hi = high[idx];
        let lo = low[idx];
        let pc = close[idx - 1];
        let tr = natr_true_range(hi, lo, pc);
        atr = natr_wilder_step(atr, tr, period);
        out[idx] = natr_output_value(atr, close[idx], period);
        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn natr_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe {
        natr_avx512_body(high, low, close, period, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn natr_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert!(high.len() == low.len() && high.len() == close.len() && high.len() == out.len());
    let len = out.len();
    if first >= len {
        return;
    }

    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let o = out.as_mut_ptr();

    let warm_end = first + period;
    if warm_end >= len {
        return;
    }
    let mut i = first + 1;
    let mut sum_tr = 0.0;

    while i + 3 <= warm_end {
        let vh = _mm256_loadu_pd(h.add(i));
        let vl = _mm256_loadu_pd(l.add(i));
        let vpc = _mm256_loadu_pd(c.add(i - 1));

        let mut highs = [0.0f64; 4];
        let mut lows = [0.0f64; 4];
        let mut previous_closes = [0.0f64; 4];
        _mm256_storeu_pd(highs.as_mut_ptr(), vh);
        _mm256_storeu_pd(lows.as_mut_ptr(), vl);
        _mm256_storeu_pd(previous_closes.as_mut_ptr(), vpc);
        for offset in 0..4 {
            sum_tr += natr_true_range(highs[offset], lows[offset], previous_closes[offset]);
        }

        i += 4;
    }
    while i <= warm_end {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let pc = *c.add(i - 1);
        let tr = natr_true_range(hi, lo, pc);
        sum_tr += tr;
        i += 1;
    }

    let mut atr = sum_tr / period as f64;
    *o.add(warm_end) = natr_output_value(atr, *c.add(warm_end), period);

    let mut idx = warm_end + 1;
    while idx + 3 < len {
        let vh = _mm256_loadu_pd(h.add(idx));
        let vl = _mm256_loadu_pd(l.add(idx));
        let vpc = _mm256_loadu_pd(c.add(idx - 1));

        let mut highs = [0.0f64; 4];
        let mut lows = [0.0f64; 4];
        let mut previous_closes = [0.0f64; 4];
        _mm256_storeu_pd(highs.as_mut_ptr(), vh);
        _mm256_storeu_pd(lows.as_mut_ptr(), vl);
        _mm256_storeu_pd(previous_closes.as_mut_ptr(), vpc);

        for offset in 0..4 {
            let tr = natr_true_range(highs[offset], lows[offset], previous_closes[offset]);
            atr = natr_wilder_step(atr, tr, period);
            *o.add(idx + offset) = natr_output_value(atr, *c.add(idx + offset), period);
        }

        idx += 4;
    }
    while idx < len {
        let hi = *h.add(idx);
        let lo = *l.add(idx);
        let pc = *c.add(idx - 1);
        let tr = natr_true_range(hi, lo, pc);
        atr = natr_wilder_step(atr, tr, period);
        *o.add(idx) = natr_output_value(atr, *c.add(idx), period);
        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn natr_avx512_body(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    debug_assert!(high.len() == low.len() && high.len() == close.len() && high.len() == out.len());
    let len = out.len();
    if first >= len {
        return;
    }

    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let o = out.as_mut_ptr();

    let warm_end = first + period;
    if warm_end >= len {
        return;
    }
    let mut i = first + 1;
    let mut sum_tr = 0.0;

    while i + 7 <= warm_end {
        let vh = _mm512_loadu_pd(h.add(i));
        let vl = _mm512_loadu_pd(l.add(i));
        let vpc = _mm512_loadu_pd(c.add(i - 1));

        let mut highs = [0.0f64; 8];
        let mut lows = [0.0f64; 8];
        let mut previous_closes = [0.0f64; 8];
        _mm512_storeu_pd(highs.as_mut_ptr(), vh);
        _mm512_storeu_pd(lows.as_mut_ptr(), vl);
        _mm512_storeu_pd(previous_closes.as_mut_ptr(), vpc);
        for offset in 0..8 {
            sum_tr += natr_true_range(highs[offset], lows[offset], previous_closes[offset]);
        }

        i += 8;
    }
    while i <= warm_end {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let pc = *c.add(i - 1);
        let tr = natr_true_range(hi, lo, pc);
        sum_tr += tr;
        i += 1;
    }

    let mut atr = sum_tr / period as f64;
    *o.add(warm_end) = natr_output_value(atr, *c.add(warm_end), period);

    let mut idx = warm_end + 1;
    while idx + 7 < len {
        let vh = _mm512_loadu_pd(h.add(idx));
        let vl = _mm512_loadu_pd(l.add(idx));
        let vpc = _mm512_loadu_pd(c.add(idx - 1));

        let mut highs = [0.0f64; 8];
        let mut lows = [0.0f64; 8];
        let mut previous_closes = [0.0f64; 8];
        _mm512_storeu_pd(highs.as_mut_ptr(), vh);
        _mm512_storeu_pd(lows.as_mut_ptr(), vl);
        _mm512_storeu_pd(previous_closes.as_mut_ptr(), vpc);
        for offset in 0..8 {
            let tr = natr_true_range(highs[offset], lows[offset], previous_closes[offset]);
            atr = natr_wilder_step(atr, tr, period);
            *o.add(idx + offset) = natr_output_value(atr, *c.add(idx + offset), period);
        }

        idx += 8;
    }
    while idx < len {
        let hi = *h.add(idx);
        let lo = *l.add(idx);
        let pc = *c.add(idx - 1);
        let tr = natr_true_range(hi, lo, pc);
        atr = natr_wilder_step(atr, tr, period);
        *o.add(idx) = natr_output_value(atr, *c.add(idx), period);
        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn natr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out);
}

#[derive(Debug, Clone)]
pub struct NatrStream {
    period: usize,
    count: usize,
    sum_tr: f64,
    atr: f64,
    prev_close: f64,
    have_prev: bool,
    ready: bool,
}

impl NatrStream {
    #[inline(always)]
    pub fn try_new(params: NatrParams) -> Result<Self, NatrError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(NatrError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            count: 0,
            sum_tr: 0.0,
            atr: 0.0,
            prev_close: 0.0,
            have_prev: false,
            ready: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if !self.have_prev {
            self.prev_close = close;
            self.have_prev = true;
            return None;
        }

        let pc = self.prev_close;
        let tr = natr_true_range(high, low, pc);
        self.prev_close = close;

        if !self.ready {
            self.sum_tr += tr;
            self.count += 1;
            if self.count == self.period {
                self.atr = self.sum_tr / (self.period as f64);
                self.ready = true;
            } else {
                return None;
            }
        } else {
            self.atr = natr_wilder_step(self.atr, tr, self.period);
        }

        Some(natr_output_value(self.atr, close, self.period))
    }
}

#[derive(Clone, Debug)]
pub struct NatrBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for NatrBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NatrBatchBuilder {
    range: NatrBatchRange,
    kernel: Kernel,
}

impl NatrBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<NatrBatchOutput, NatrError> {
        natr_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<NatrBatchOutput, NatrError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<NatrBatchOutput, NatrError> {
        NatrBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub fn natr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    k: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(NatrError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    natr_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct NatrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NatrParams>,
    pub rows: usize,
    pub cols: usize,
}
impl NatrBatchOutput {
    pub fn row_for_params(&self, p: &NatrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &NatrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &NatrBatchRange) -> Result<Vec<NatrParams>, NatrError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, NatrError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        let step_u = step;

        if start <= end {
            let mut v = start;
            loop {
                if v > end {
                    break;
                }
                values.push(v);
                match v.checked_add(step_u) {
                    Some(next) => v = next,
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                if v < end {
                    break;
                }
                values.push(v);
                match v.checked_sub(step_u) {
                    Some(next) => v = next,
                    None => break,
                }
            }
        }

        if values.is_empty() {
            return Err(NatrError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }

        Ok(values)
    }

    let periods = axis_usize(r.period)?;

    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(NatrParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn natr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
    natr_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn natr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
) -> Result<NatrBatchOutput, NatrError> {
    natr_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn natr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<NatrBatchOutput, NatrError> {
    let combos = expand_grid(sweep)?;

    let len_h = high.len();
    let len_l = low.len();
    let len_c = close.len();
    if len_h != len_l || len_h != len_c {
        return Err(NatrError::MismatchedLength {
            expected: len_h,
            actual: if len_l != len_h { len_l } else { len_c },
        });
    }
    let len = len_h;

    let first = high
        .iter()
        .position(|x| !x.is_nan())
        .unwrap_or(0)
        .max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
        .max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first <= max_p {
        return Err(NatrError::NotEnoughValidData {
            needed: max_p + 1,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let use_tr_shared = combos.len() >= 24;

    if use_tr_shared {
        let mut tr: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, len);
        tr.resize(len, 0.0);
        if first + 1 < len {
            let mut i = first + 1;
            while i < len {
                let hi = high[i];
                let lo = low[i];
                let pc = close[i - 1];
                let trv = natr_true_range(hi, lo, pc);
                tr[i] = trv;
                i += 1;
            }
        }

        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            natr_row_scalar_from_tr(&tr, close, first, period, out_row);
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
    } else {
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match kern {
                Kernel::Scalar => natr_row_scalar(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => natr_row_avx2(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => natr_row_avx512(high, low, close, first, period, out_row),
                _ => unreachable!(),
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
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(NatrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn natr_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &NatrBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<NatrParams>, NatrError> {
    let combos = expand_grid(sweep)?;

    let len_h = high.len();
    let len_l = low.len();
    let len_c = close.len();
    if len_h != len_l || len_h != len_c {
        return Err(NatrError::MismatchedLength {
            expected: len_h,
            actual: if len_l != len_h { len_l } else { len_c },
        });
    }
    let len = len_h;

    let first = high
        .iter()
        .position(|x| !x.is_nan())
        .unwrap_or(0)
        .max(low.iter().position(|x| !x.is_nan()).unwrap_or(0))
        .max(close.iter().position(|x| !x.is_nan()).unwrap_or(0));
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first <= max_p {
        return Err(NatrError::NotEnoughValidData {
            needed: max_p + 1,
            valid: len - first,
        });
    }
    let rows = combos.len();
    let cols = len;

    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warmup_end = first + period;
        let row_start = row * cols;
        for i in 0..warmup_end.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let use_tr_shared = combos.len() >= 24;
    if use_tr_shared {
        let mut tr: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, len);
        tr.resize(len, 0.0);
        if first + 1 < len {
            let mut i = first + 1;
            while i < len {
                let hi = high[i];
                let lo = low[i];
                let pc = close[i - 1];
                let trv = natr_true_range(hi, lo, pc);
                tr[i] = trv;
                i += 1;
            }
        }

        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            natr_row_scalar_from_tr(&tr, close, first, period, out_row);
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
    } else {
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match kern {
                Kernel::Scalar => natr_row_scalar(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => natr_row_avx2(high, low, close, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => natr_row_avx512(high, low, close, first, period, out_row),
                _ => unreachable!(),
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
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn natr_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_scalar(high, low, close, period, first, out)
}

#[inline(always)]
unsafe fn natr_row_scalar_from_tr(
    tr: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = out.len();
    if first >= len {
        return;
    }
    let warm_end = first + period;
    if warm_end >= len {
        return;
    }

    let mut sum_tr = 0.0;
    for value in &tr[(first + 1)..=warm_end] {
        sum_tr += *value;
    }
    let mut atr = sum_tr / period as f64;
    out[warm_end] = natr_output_value(atr, close[warm_end], period);

    let mut i = warm_end + 1;
    while i < len {
        atr = natr_wilder_step(atr, tr[i], period);
        out[i] = natr_output_value(atr, close[i], period);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx2(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx512(high, low, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn natr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_avx512_body(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx2_from_tr(
    tr: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_row_scalar_from_tr(tr, close, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn natr_row_avx512_from_tr(
    tr: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    natr_row_scalar_from_tr(tr, close, first, period, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_natr_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = NatrInput::with_default_candles(&candles);

        let baseline = natr(&input)?.values;

        let mut out = vec![0.0; candles.close.len()];
        #[allow(unused_variables)]
        {
            {
                natr_into(&input, &mut out)?;
            }
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..baseline.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "NATR into parity mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }

    fn check_natr_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = NatrParams { period: None };
        let input_default = NatrInput::from_candles(&candles, default_params);
        let output_default = natr_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period_7 = NatrParams { period: Some(7) };
        let input_period_7 = NatrInput::from_candles(&candles, params_period_7);
        let output_period_7 = natr_with_kernel(&input_period_7, kernel)?;
        assert_eq!(output_period_7.values.len(), candles.close.len());
        Ok(())
    }

    fn check_natr_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close_prices = candles.select_candle_field("close").unwrap();
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_candles(&candles, params.clone());
        let natr_result = natr_with_kernel(&input, kernel)?;
        assert_eq!(natr_result.values.len(), close_prices.len());
        let expected_last_five = [
            1.5465877404905772,
            1.4773840355794576,
            1.4201627494720954,
            1.3556212509014807,
            1.3836271128536142,
        ];
        let start_index = natr_result.values.len() - 5;
        let result_last_five = &natr_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "NATR mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        let period = params.period.unwrap();
        for i in 0..(period - 1) {
            assert!(natr_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_natr_with_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 10.0, 15.0];
        let close = [7.0, 14.0, 25.0];
        let params = NatrParams { period: Some(0) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 10.0, 15.0];
        let close = [7.0, 14.0, 25.0];
        let params = NatrParams { period: Some(10) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [40.0];
        let close = [41.0];
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let params = NatrParams { period: Some(2) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 10.0];
        let low = [f64::NAN, 5.0];
        let close = [f64::NAN, 7.0];
        let params = NatrParams { period: Some(5) };
        let input = NatrInput::from_slices(&high, &low, &close, params);
        let result = natr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_natr_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = NatrParams { period: Some(14) };
        let first_input = NatrInput::from_candles(&candles, first_params);
        let first_result = natr_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = NatrParams { period: Some(14) };
        let second_input = NatrInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = natr_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());

        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, but found NaN at index {}",
                i
            );
        }
        Ok(())
    }

    fn check_natr_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_candles(&candles, params);
        let natr_result = natr_with_kernel(&input, kernel)?;
        assert_eq!(natr_result.values.len(), candles.close.len());
        if natr_result.values.len() > 30 {
            for i in 30..natr_result.values.len() {
                assert!(
                    !natr_result.values[i].is_nan(),
                    "Expected no NaN after index 30, but found NaN at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_natr_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = NatrInput::with_default_candles(&candles);
        match input.data {
            NatrData::Candles { .. } => {}
            _ => panic!("Expected NatrData::Candles variant"),
        }
        let output = natr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_natr_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 14;
        let high = &candles.high;
        let low = &candles.low;
        let close = &candles.close;
        let input = NatrInput::from_slices(
            high,
            low,
            close,
            NatrParams {
                period: Some(period),
            },
        );
        let batch_output = natr_with_kernel(&input, kernel)?.values;

        let mut stream = NatrStream::try_new(NatrParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(close.len());
        for ((&h, &l), &c) in high.iter().zip(low.iter()).zip(close.iter()) {
            match stream.update(h, l, c) {
                Some(natr_val) => stream_values.push(natr_val),
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
                "[{}] NATR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_natr_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            NatrParams::default(),
            NatrParams { period: Some(2) },
            NatrParams { period: Some(5) },
            NatrParams { period: Some(7) },
            NatrParams { period: Some(10) },
            NatrParams { period: Some(20) },
            NatrParams { period: Some(30) },
            NatrParams { period: Some(50) },
            NatrParams { period: Some(100) },
            NatrParams { period: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = NatrInput::from_candles(&candles, params.clone());
            let output = natr_with_kernel(&input, kernel)?;

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
                        params.period.unwrap_or(14),
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
                        params.period.unwrap_or(14),
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
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_natr_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = NatrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = NatrParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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
            (10, 20, 2),
            (50, 100, 10),
            (14, 14, 0),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = NatrBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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

    #[cfg(feature = "proptest")]
    fn check_natr_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50, 50usize..=400, 0usize..=2)
            .prop_flat_map(|(period, len, scenario)| {
                let close_strategy = match scenario {
                    0 => prop::collection::vec(
                        (1.0f64..1000.0f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    )
                    .boxed(),
                    1 => prop::collection::vec(
                        (0.01f64..1.0f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    )
                    .boxed(),
                    _ => (1.0f64..100.0f64)
                        .prop_map(move |val| vec![val; len])
                        .boxed(),
                };

                (close_strategy, Just(period), Just(len), Just(scenario))
            })
            .prop_flat_map(|(close_prices, period, len, scenario)| {
                let mut high_vec = Vec::with_capacity(len);
                let mut low_vec = Vec::with_capacity(len);

                for (i, &close) in close_prices.iter().enumerate() {
                    if scenario == 2 {
                        high_vec.push(close);
                        low_vec.push(close);
                    } else {
                        let volatility_factor = 0.001 + 0.20 * ((i * 7919) % 100) as f64 / 100.0;
                        let spread = close * volatility_factor;

                        let high = close + spread * 0.5;
                        let low = close - spread * 0.5;

                        high_vec.push(high);
                        low_vec.push(low.max(0.001));
                    }
                }

                (
                    Just(high_vec),
                    Just(low_vec),
                    Just(close_prices),
                    Just(period),
                    Just(scenario),
                )
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(high, low, close, period, scenario)| {
                let params = NatrParams {
                    period: Some(period),
                };
                let input = NatrInput::from_slices(&high, &low, &close, params);

                let result = natr_with_kernel(&input, kernel)?;

                let ref_result = natr_with_kernel(&input, Kernel::Scalar)?;

                prop_assert_eq!(result.values.len(), high.len());
                prop_assert_eq!(result.values.len(), low.len());
                prop_assert_eq!(result.values.len(), close.len());

                for i in 0..(period - 1) {
                    prop_assert!(
                        result.values[i].is_nan(),
                        "Expected NaN at index {} during warmup, got {}",
                        i,
                        result.values[i]
                    );
                }

                for i in period..result.values.len() {
                    if result.values[i].is_finite() {
                        prop_assert!(
                            result.values[i] >= 0.0,
                            "NATR should be non-negative at index {}: got {}",
                            i,
                            result.values[i]
                        );

                        prop_assert!(
                            result.values[i] < 10000.0,
                            "NATR seems unreasonably high at index {}: got {}",
                            i,
                            result.values[i]
                        );
                    }
                }

                for i in 0..result.values.len() {
                    let val = result.values[i];
                    let ref_val = ref_result.values[i];

                    if val.is_nan() && ref_val.is_nan() {
                        continue;
                    }

                    if val.is_finite() && ref_val.is_finite() {
                        let diff = (val - ref_val).abs();
                        let tolerance = (ref_val.abs() * 1e-10).max(1e-10);
                        prop_assert!(
                            diff <= tolerance,
                            "Kernel mismatch at index {}: {} vs {} (diff: {})",
                            i,
                            val,
                            ref_val,
                            diff
                        );
                    } else {
                        prop_assert_eq!(
                            val.is_finite(),
                            ref_val.is_finite(),
                            "Finite status mismatch at index {}: {} vs {}",
                            i,
                            val,
                            ref_val
                        );
                    }
                }

                if scenario == 2 {
                    let is_constant = high
                        .iter()
                        .zip(&low)
                        .zip(&close)
                        .all(|((h, l), c)| (*h - *l).abs() < 1e-10 && (*h - *c).abs() < 1e-10);

                    if is_constant && result.values.len() > period + 5 {
                        for i in (period + 5)..result.values.len() {
                            if result.values[i].is_finite() {
                                prop_assert!(
                                    result.values[i].abs() < 1e-10,
                                    "NATR should be 0 for constant prices at index {}, got {}",
                                    i,
                                    result.values[i]
                                );
                            }
                        }
                    }
                }

                if scenario == 1 {
                    for i in period..result.values.len() {
                        if result.values[i].is_finite() && close[i] > 0.0 {
                            prop_assert!(
                                result.values[i] >= 0.0 && result.values[i] < 100000.0,
                                "NATR out of bounds with small prices at index {}: got {}",
                                i,
                                result.values[i]
                            );
                        }
                    }
                }

                if close.iter().any(|&c| c.abs() < 1e-10) {
                    for (i, &c) in close.iter().enumerate() {
                        if c.abs() < 1e-10 && i >= period - 1 {
                            prop_assert!(
                                result.values[i] == 0.0 || result.values[i].is_nan(),
                                "NATR should be 0 or NaN when close is 0, got {} at index {}",
                                result.values[i],
                                i
                            );
                        }
                    }
                }

                #[cfg(debug_assertions)]
                {
                    for (i, &val) in result.values.iter().enumerate() {
                        if val.is_finite() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "Found poison value at index {}: {} (0x{:016X})",
                                i,
                                val,
                                bits
                            );
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[test]
    fn talib_natr_uses_previous_close_true_ranges_and_period_lookback() {
        // TA-Lib TA_NATR: the first output is at `period`, because every true
        // range consumes the preceding close.  The seed is TR[1..=period],
        // never the first bar's bare high-low range.
        let high = [10.0, 12.0, 13.0, 15.0, 16.0];
        let low = [8.0, 9.0, 10.0, 11.0, 12.0];
        let close = [9.0, 10.0, 12.0, 14.0, 0.0];
        let input = NatrInput::from_slices(&high, &low, &close, NatrParams { period: Some(2) });

        let values = natr_with_kernel(&input, Kernel::Scalar)
            .expect("TA-Lib-authoritative NATR fixture must evaluate")
            .values;

        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2].to_bits(), 25.0f64.to_bits());
        assert_eq!(values[3].to_bits(), 25.0f64.to_bits());
        assert_eq!(values[4].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn talib_natr_period_one_emits_raw_true_range() {
        // TA-Lib deliberately preserves its historical period-one contract:
        // raw TR, not TR/close*100.
        let high = [10.0, 12.0, 13.0];
        let low = [8.0, 9.0, 10.0];
        let close = [9.0, 10.0, 12.0];
        let input = NatrInput::from_slices(&high, &low, &close, NatrParams { period: Some(1) });

        let values = natr_with_kernel(&input, Kernel::Scalar)
            .expect("period-one NATR fixture must evaluate")
            .values;

        assert!(values[0].is_nan());
        assert_eq!(values[1].to_bits(), 3.0f64.to_bits());
        assert_eq!(values[2].to_bits(), 3.0f64.to_bits());
    }

    #[test]
    fn talib_natr_stream_and_batch_preserve_authoritative_edges() {
        let high = [10.0, 12.0, 13.0, 15.0, 16.0];
        let low = [8.0, 9.0, 10.0, 11.0, 12.0];
        let close = [9.0, 10.0, 12.0, 14.0, 5.0e-15];

        let mut stream =
            NatrStream::try_new(NatrParams { period: Some(2) }).expect("valid stream parameters");
        let streamed: Vec<f64> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .map(|((&h, &l), &c)| stream.update(h, l, c).unwrap_or(f64::NAN))
            .collect();
        assert!(streamed[1].is_nan());
        assert_eq!(streamed[2].to_bits(), 25.0f64.to_bits());
        assert_eq!(streamed[4].to_bits(), 0.0f64.to_bits());

        let batch = natr_batch_slice(
            &high,
            &low,
            &close,
            &NatrBatchRange { period: (2, 2, 0) },
            Kernel::Scalar,
        )
        .expect("TA-Lib-authoritative batch fixture must evaluate");
        assert!(batch.values[1].is_nan());
        assert_eq!(batch.values[2].to_bits(), 25.0f64.to_bits());
        assert_eq!(batch.values[4].to_bits(), 0.0f64.to_bits());
    }

    macro_rules! generate_all_natr_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    #[cfg(feature = "proptest")]
    generate_all_natr_tests!(check_natr_property);

    generate_all_natr_tests!(
        check_natr_partial_params,
        check_natr_accuracy,
        check_natr_with_zero_period,
        check_natr_period_exceeds_length,
        check_natr_very_small_dataset,
        check_natr_all_values_nan,
        check_natr_not_enough_valid_data,
        check_natr_slice_data_reinput,
        check_natr_nan_check,
        check_natr_default_candles,
        check_natr_streaming,
        check_natr_no_poison
    );

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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

pub fn natr_into_slice(dst: &mut [f64], input: &NatrInput, kern: Kernel) -> Result<(), NatrError> {
    let (high, low, close, period) = match &input.data {
        NatrData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            input.params.period.unwrap_or(14),
        ),
        NatrData::Slices { high, low, close } => {
            (*high, *low, *close, input.params.period.unwrap_or(14))
        }
    };

    let len = high.len().min(low.len()).min(close.len());

    if dst.len() != len {
        return Err(NatrError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    if len == 0 {
        return Err(NatrError::EmptyInputData);
    }
    if period == 0 {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period > len {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = {
        let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
        let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
        let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

        match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
            (Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
            _ => None,
        }
    };

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(NatrError::AllValuesNaN),
    };

    if (len - first_valid_idx) <= period {
        return Err(NatrError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first_valid_idx,
        });
    }

    let chosen = match kern {
        Kernel::Auto => natr_auto_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                natr_scalar(high, low, close, period, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                natr_avx2(high, low, close, period, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                natr_avx512(high, low, close, period, first_valid_idx, dst)
            }
            _ => unreachable!(),
        }
    }

    for v in &mut dst[..(first_valid_idx + period)] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn natr_into(input: &NatrInput, out: &mut [f64]) -> Result<(), NatrError> {
    natr_into_slice(out, input, Kernel::Auto)
}
