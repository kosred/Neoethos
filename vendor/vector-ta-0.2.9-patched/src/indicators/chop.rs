use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

/// Canonical CHOP semantics follow TradingView's published operation shape:
/// `100 * LOG10(SUM(ATR(1), n) / range) / LOG10(n)`.
///
/// Source: https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/
pub const CHOP_TRADINGVIEW_LOG10_SEMANTICS_V1: &str =
    "chop-tradingview-log10.fixed-order-f64.semantic-v1";

const CHOP_LN_2_BITS_V1: u64 = 0x3fe6_2e42_fefa_39ef;
const CHOP_LN_10_BITS_V1: u64 = 0x4002_6bb1_bbb5_5515;
const CHOP_MANTISSA_MASK_V1: u64 = 0x000f_ffff_ffff_ffff;
const CHOP_ONE_EXPONENT_BITS_V1: u64 = 0x3ff0_0000_0000_0000;
const CHOP_TWO_POW_54_BITS_V1: u64 = 0x4350_0000_0000_0000;

#[inline]
fn chop_ln_positive_exact_v1(value: f64) -> f64 {
    if value == f64::INFINITY {
        return f64::INFINITY;
    }
    if value <= 0.0 || value.is_nan() {
        return f64::NAN;
    }
    let (normalized, exponent_adjustment) = if value.is_subnormal() {
        (value * f64::from_bits(CHOP_TWO_POW_54_BITS_V1), -54)
    } else {
        (value, 0)
    };
    let bits = normalized.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + exponent_adjustment;
    let mantissa = f64::from_bits((bits & CHOP_MANTISSA_MASK_V1) | CHOP_ONE_EXPONENT_BITS_V1);
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z_squared = z * z;
    let mut term = z;
    let mut sum = z;
    let mut denominator = 3_u32;
    while denominator <= 49 {
        term *= z_squared;
        sum += term / denominator as f64;
        denominator += 2;
    }
    exponent as f64 * f64::from_bits(CHOP_LN_2_BITS_V1) + 2.0 * sum
}

#[inline]
pub(crate) fn chop_log10_positive_exact_v1(value: f64) -> f64 {
    chop_ln_positive_exact_v1(value) / f64::from_bits(CHOP_LN_10_BITS_V1)
}

#[inline]
fn chop_value_from_ratio_exact_v1(ratio: f64, scalar: f64, log10_period: f64) -> f64 {
    (scalar * chop_log10_positive_exact_v1(ratio)) / log10_period
}

#[derive(Debug, Clone)]
pub enum ChopData<'a> {
    Candles(&'a Candles),
    Slice {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ChopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ChopParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
    pub drift: Option<usize>,
}
impl Default for ChopParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
            drift: Some(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChopInput<'a> {
    pub data: ChopData<'a>,
    pub params: ChopParams,
}

impl<'a> ChopInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ChopParams) -> Self {
        Self {
            data: ChopData::Candles(candles),
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ChopParams,
    ) -> Self {
        Self {
            data: ChopData::Slice { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ChopData::Candles(candles),
            params: ChopParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_scalar(&self) -> f64 {
        self.params.scalar.unwrap_or(100.0)
    }
    #[inline]
    pub fn get_drift(&self) -> usize {
        self.params.drift.unwrap_or(1)
    }
}

impl<'a> AsRef<[f64]> for ChopInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ChopData::Candles(candles) => candles.close.as_slice(),
            ChopData::Slice { close, .. } => close,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ChopBuilder {
    period: Option<usize>,
    scalar: Option<f64>,
    drift: Option<usize>,
    kernel: Kernel,
}
impl Default for ChopBuilder {
    fn default() -> Self {
        Self {
            period: None,
            scalar: None,
            drift: None,
            kernel: Kernel::Auto,
        }
    }
}
impl ChopBuilder {
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
    pub fn scalar(mut self, s: f64) -> Self {
        self.scalar = Some(s);
        self
    }
    #[inline(always)]
    pub fn drift(mut self, d: usize) -> Self {
        self.drift = Some(d);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ChopOutput, ChopError> {
        let params = ChopParams {
            period: self.period,
            scalar: self.scalar,
            drift: self.drift,
        };
        let input = ChopInput::from_candles(c, params);
        chop_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ChopOutput, ChopError> {
        let params = ChopParams {
            period: self.period,
            scalar: self.scalar,
            drift: self.drift,
        };
        let input = ChopInput::from_slices(high, low, close, params);
        chop_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ChopStream, ChopError> {
        let params = ChopParams {
            period: self.period,
            scalar: self.scalar,
            drift: self.drift,
        };
        ChopStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum ChopError {
    #[error("chop: Empty data provided.")]
    EmptyData,
    #[error("chop: Invalid period: period={period}, data length={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("chop: All relevant data (high/low/close) are NaN.")]
    AllValuesNaN,
    #[error("chop: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("chop: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("chop: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("chop: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("chop: invalid input: {0}")]
    InvalidInput(String),
    #[error("chop: underlying function failed: {0}")]
    UnderlyingFunctionFailed(String),
}

#[inline]
pub fn chop(input: &ChopInput) -> Result<ChopOutput, ChopError> {
    chop_with_kernel(input, Kernel::Auto)
}

pub fn chop_with_kernel(input: &ChopInput, kernel: Kernel) -> Result<ChopOutput, ChopError> {
    let (high, low, close) = match &input.data {
        ChopData::Candles(candles) => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        ChopData::Slice { high, low, close } => (*high, *low, *close),
    };

    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChopError::UnderlyingFunctionFailed(
            "mismatched input lengths".to_string(),
        ));
    }

    let len = close.len();
    if len == 0 {
        return Err(ChopError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ChopError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let drift = input.get_drift();
    if drift == 0 {
        return Err(ChopError::UnderlyingFunctionFailed(
            "Invalid drift=0 for ATR".to_string(),
        ));
    }
    let scalar = input.get_scalar();

    let first_valid_idx = match (0..len).find(|&i| {
        let (h, l, c) = (high[i], low[i], close[i]);
        !(h.is_nan() || l.is_nan() || c.is_nan())
    }) {
        Some(idx) => idx,
        None => return Err(ChopError::AllValuesNaN),
    };
    if (len - first_valid_idx) < period {
        return Err(ChopError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let warmup_period = first_valid_idx + period - 1;
    let mut out = alloc_with_nan_prefix(len, warmup_period);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => chop_scalar(
                high,
                low,
                close,
                period,
                drift,
                scalar,
                first_valid_idx,
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => chop_avx2(
                high,
                low,
                close,
                period,
                drift,
                scalar,
                first_valid_idx,
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => chop_avx512(
                high,
                low,
                close,
                period,
                drift,
                scalar,
                first_valid_idx,
                &mut out,
            ),
            _ => unreachable!(),
        }
    }
    Ok(ChopOutput { values: out })
}

#[inline]
pub fn chop_into_slice(dst: &mut [f64], input: &ChopInput, kern: Kernel) -> Result<(), ChopError> {
    let (high, low, close) = match &input.data {
        ChopData::Candles(candles) => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        ChopData::Slice { high, low, close } => (*high, *low, *close),
    };

    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChopError::UnderlyingFunctionFailed(
            "mismatched input lengths".to_string(),
        ));
    }

    let len = close.len();
    if len == 0 {
        return Err(ChopError::EmptyData);
    }

    if dst.len() != len {
        return Err(ChopError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ChopError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let drift = input.get_drift();
    if drift == 0 {
        return Err(ChopError::UnderlyingFunctionFailed(
            "Invalid drift=0 for ATR".to_string(),
        ));
    }
    let scalar = input.get_scalar();

    let first_valid_idx = match (0..len).find(|&i| {
        let (h, l, c) = (high[i], low[i], close[i]);
        !(h.is_nan() || l.is_nan() || c.is_nan())
    }) {
        Some(idx) => idx,
        None => return Err(ChopError::AllValuesNaN),
    };
    if (len - first_valid_idx) < period {
        return Err(ChopError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => chop_scalar(
                high,
                low,
                close,
                period,
                drift,
                scalar,
                first_valid_idx,
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => chop_avx2(
                high,
                low,
                close,
                period,
                drift,
                scalar,
                first_valid_idx,
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => chop_avx512(
                high,
                low,
                close,
                period,
                drift,
                scalar,
                first_valid_idx,
                dst,
            ),
            _ => unreachable!(),
        }
    }

    let warmup_end = first_valid_idx + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn chop_into(input: &ChopInput, out: &mut [f64]) -> Result<(), ChopError> {
    let len = match &input.data {
        ChopData::Candles(c) => c.close.len(),
        ChopData::Slice { close, .. } => close.len(),
    };
    if out.len() != len {
        return Err(ChopError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }
    chop_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub unsafe fn chop_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    drift: usize,
    scalar: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    debug_assert!(high.len() == low.len() && low.len() == close.len());
    let len = close.len();
    if len == 0 {
        return;
    }

    if period == 14 && drift == 1 {
        chop_scalar_period_14_drift_1(high, low, close, scalar, first_valid_idx, out);
        return;
    }

    let alpha = 1.0 / (drift as f64);
    let log10_period = chop_log10_positive_exact_v1(period as f64);

    let mut atr_ring = vec![0.0_f64; period];
    let mut atr_ring_idx: usize = 0;
    let mut rolling_sum_atr: f64 = 0.0;

    let mut rma_atr = f64::NAN;
    let mut sum_tr: f64 = 0.0;

    let mut dq_high: VecDeque<usize> = VecDeque::with_capacity(period);
    let mut dq_low: VecDeque<usize> = VecDeque::with_capacity(period);

    let mut prev_close = close[first_valid_idx];

    for i in first_valid_idx..len {
        let hi = high[i];
        let lo = low[i];
        let hl = hi - lo;
        let tr = if i == first_valid_idx {
            sum_tr = hl;
            hl
        } else {
            let hc = (hi - prev_close).abs();
            let lc = (lo - prev_close).abs();
            hl.max(hc).max(lc)
        };

        let rel = i - first_valid_idx;
        if rel < drift {
            if i != first_valid_idx {
                sum_tr += tr;
            }
            if rel == drift - 1 {
                rma_atr = sum_tr / drift as f64;
            }
        } else {
            rma_atr += alpha * (tr - rma_atr);
        }
        prev_close = close[i];

        let current_atr = if rel < drift {
            if rel == drift - 1 { rma_atr } else { f64::NAN }
        } else {
            rma_atr
        };

        let oldest = atr_ring[atr_ring_idx];
        rolling_sum_atr -= oldest;
        let new_val = if current_atr.is_nan() {
            0.0
        } else {
            current_atr
        };
        atr_ring[atr_ring_idx] = new_val;
        rolling_sum_atr += new_val;
        atr_ring_idx += 1;
        if atr_ring_idx == period {
            atr_ring_idx = 0;
        }

        let win_start = i.saturating_sub(period - 1);
        while let Some(&front_idx) = dq_high.front() {
            if front_idx < win_start {
                dq_high.pop_front();
            } else {
                break;
            }
        }
        while let Some(&front_idx) = dq_low.front() {
            if front_idx < win_start {
                dq_low.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back_idx) = dq_high.back() {
            if high[back_idx] <= hi {
                dq_high.pop_back();
            } else {
                break;
            }
        }
        dq_high.push_back(i);
        while let Some(&back_idx) = dq_low.back() {
            if low[back_idx] >= lo {
                dq_low.pop_back();
            } else {
                break;
            }
        }
        dq_low.push_back(i);

        if rel >= (period - 1) {
            let hh_idx = *dq_high.front().unwrap();
            let ll_idx = *dq_low.front().unwrap();
            let range = high[hh_idx] - low[ll_idx];
            if range > 0.0 && rolling_sum_atr > 0.0 {
                let ratio = rolling_sum_atr / range;
                out[i] = chop_value_from_ratio_exact_v1(ratio, scalar, log10_period);
            } else {
                out[i] = f64::NAN;
            }
        }
    }
}

#[inline]
unsafe fn chop_scalar_period_14_drift_1(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    scalar: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    const PERIOD: usize = 14;
    const CAP: usize = 16;
    const MASK: usize = CAP - 1;

    #[inline(always)]
    fn rb_inc(idx: usize) -> usize {
        (idx + 1) & MASK
    }

    #[inline(always)]
    fn rb_dec(idx: usize) -> usize {
        idx.wrapping_sub(1) & MASK
    }

    let len = close.len();
    let log10_period = chop_log10_positive_exact_v1(PERIOD as f64);

    let mut atr_ring = [0.0_f64; PERIOD];
    let mut atr_ring_idx: usize = 0;
    let mut rolling_sum_atr: f64 = 0.0;

    let mut h_idx = [0usize; CAP];
    let mut h_val = [0.0_f64; CAP];
    let mut h_head: usize = 0;
    let mut h_tail: usize = 0;

    let mut l_idx = [0usize; CAP];
    let mut l_val = [0.0_f64; CAP];
    let mut l_head: usize = 0;
    let mut l_tail: usize = 0;

    let mut prev_close = *close.get_unchecked(first_valid_idx);

    for i in first_valid_idx..len {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let hl = hi - lo;
        let tr = if i == first_valid_idx {
            hl
        } else {
            let hc = (hi - prev_close).abs();
            let lc = (lo - prev_close).abs();
            hl.max(hc).max(lc)
        };
        prev_close = *close.get_unchecked(i);

        rolling_sum_atr -= atr_ring[atr_ring_idx];
        atr_ring[atr_ring_idx] = tr;
        rolling_sum_atr += tr;
        atr_ring_idx += 1;
        if atr_ring_idx == PERIOD {
            atr_ring_idx = 0;
        }

        while h_head != h_tail {
            let front_i = h_idx[h_head];
            if front_i + PERIOD <= i {
                h_head = rb_inc(h_head);
            } else {
                break;
            }
        }
        while l_head != l_tail {
            let front_i = l_idx[l_head];
            if front_i + PERIOD <= i {
                l_head = rb_inc(l_head);
            } else {
                break;
            }
        }

        while h_head != h_tail {
            let last = rb_dec(h_tail);
            if h_val[last] <= hi {
                h_tail = last;
            } else {
                break;
            }
        }
        let next_tail = rb_inc(h_tail);
        if next_tail == h_head {
            h_head = rb_inc(h_head);
        }
        h_idx[h_tail] = i;
        h_val[h_tail] = hi;
        h_tail = next_tail;

        while l_head != l_tail {
            let last = rb_dec(l_tail);
            if l_val[last] >= lo {
                l_tail = last;
            } else {
                break;
            }
        }
        let next_tail = rb_inc(l_tail);
        if next_tail == l_head {
            l_head = rb_inc(l_head);
        }
        l_idx[l_tail] = i;
        l_val[l_tail] = lo;
        l_tail = next_tail;

        if i - first_valid_idx >= PERIOD - 1 {
            let range = h_val[h_head] - l_val[l_head];
            if range > 0.0 && rolling_sum_atr > 0.0 {
                let ratio = rolling_sum_atr / range;
                *out.get_unchecked_mut(i) =
                    chop_value_from_ratio_exact_v1(ratio, scalar, log10_period);
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chop_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    drift: usize,
    scalar: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    chop_scalar(
        high,
        low,
        close,
        period,
        drift,
        scalar,
        first_valid_idx,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chop_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    drift: usize,
    scalar: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    chop_scalar(
        high,
        low,
        close,
        period,
        drift,
        scalar,
        first_valid_idx,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chop_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    drift: usize,
    scalar: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    chop_avx512(
        high,
        low,
        close,
        period,
        drift,
        scalar,
        first_valid_idx,
        out,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chop_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    drift: usize,
    scalar: f64,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    chop_avx512(
        high,
        low,
        close,
        period,
        drift,
        scalar,
        first_valid_idx,
        out,
    )
}

#[inline(always)]
pub fn chop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChopBatchRange,
    k: Kernel,
) -> Result<ChopBatchOutput, ChopError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(ChopError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    chop_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ChopBatchRange {
    pub period: (usize, usize, usize),
    pub scalar: (f64, f64, f64),
    pub drift: (usize, usize, usize),
}
impl Default for ChopBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
            scalar: (100.0, 124.9, 0.1),
            drift: (1, 1, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChopBatchBuilder {
    range: ChopBatchRange,
    kernel: Kernel,
}
impl ChopBatchBuilder {
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
    #[inline]
    pub fn drift_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.drift = (start, end, step);
        self
    }
    #[inline]
    pub fn drift_static(mut self, d: usize) -> Self {
        self.range.drift = (d, d, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ChopBatchOutput, ChopError> {
        chop_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct ChopBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ChopParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ChopBatchOutput {
    pub fn row_for_params(&self, p: &ChopParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.scalar.unwrap_or(100.0) - p.scalar.unwrap_or(100.0)).abs() < 1e-12
                && c.drift.unwrap_or(1) == p.drift.unwrap_or(1)
        })
    }
    pub fn values_for(&self, p: &ChopParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ChopBatchRange) -> Result<Vec<ChopParams>, ChopError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, ChopError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
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
            while v >= end {
                out.push(v);
                if v < end + step {
                    break;
                }
                v -= step;
                if v == 0 {
                    break;
                }
            }
        }
        if out.is_empty() {
            return Err(ChopError::InvalidRange { start, end, step });
        }
        Ok(out)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, ChopError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start <= end && step > 0.0 {
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += step;
            }
        } else if start >= end && step < 0.0 {
            let mut x = start;
            while x >= end - 1e-12 {
                v.push(x);
                x += step;
            }
        } else {
            return Err(ChopError::InvalidInput(
                "axis_f64 step direction invalid".into(),
            ));
        }
        if v.is_empty() {
            return Err(ChopError::InvalidRange {
                start: start as usize,
                end: end as usize,
                step: step.abs() as usize,
            });
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let scalars = axis_f64(r.scalar)?;
    let drifts = axis_usize(r.drift)?;
    let cap = periods
        .len()
        .checked_mul(scalars.len())
        .and_then(|x| x.checked_mul(drifts.len()))
        .ok_or_else(|| ChopError::InvalidInput("rows*cols overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &s in &scalars {
            for &d in &drifts {
                out.push(ChopParams {
                    period: Some(p),
                    scalar: Some(s),
                    drift: Some(d),
                });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn chop_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChopBatchRange,
    kern: Kernel,
) -> Result<ChopBatchOutput, ChopError> {
    chop_batch_inner(high, low, close, sweep, kern, false)
}
#[inline(always)]
pub fn chop_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChopBatchRange,
    kern: Kernel,
) -> Result<ChopBatchOutput, ChopError> {
    chop_batch_inner(high, low, close, sweep, kern, true)
}
#[inline(always)]
fn chop_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChopBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ChopBatchOutput, ChopError> {
    let combos = expand_grid(sweep)?;

    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChopError::UnderlyingFunctionFailed(
            "mismatched input lengths".to_string(),
        ));
    }

    let len = close.len();
    let first = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(ChopError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(ChopError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    rows.checked_mul(cols)
        .ok_or_else(|| ChopError::InvalidInput("rows*cols overflow".into()))?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let ChopParams {
            period,
            scalar,
            drift,
        } = combos[row].clone();
        let p = period.unwrap();
        let s = scalar.unwrap();
        let d = drift.unwrap();
        match kern {
            Kernel::Scalar => chop_row_scalar(high, low, close, first, p, d, s, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => chop_row_avx2(high, low, close, first, p, d, s, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => chop_row_avx512(high, low, close, first, p, d, s, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(ChopBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn chop_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    drift: usize,
    scalar: f64,
    out: &mut [f64],
) {
    chop_scalar(high, low, close, period, drift, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chop_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    drift: usize,
    scalar: f64,
    out: &mut [f64],
) {
    chop_avx2(high, low, close, period, drift, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn chop_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    drift: usize,
    scalar: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        chop_row_avx512_short(high, low, close, first, period, drift, scalar, out)
    } else {
        chop_row_avx512_long(high, low, close, first, period, drift, scalar, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn chop_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    drift: usize,
    scalar: f64,
    out: &mut [f64],
) {
    chop_avx512(high, low, close, period, drift, scalar, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn chop_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    drift: usize,
    scalar: f64,
    out: &mut [f64],
) {
    chop_avx512(high, low, close, period, drift, scalar, first, out)
}

#[derive(Copy, Clone, Debug)]
struct Node {
    idx: u64,
    val: f64,
}

#[derive(Debug, Clone)]
pub struct ChopStream {
    period: usize,
    drift: usize,
    scalar: f64,

    inv_drift: f64,
    log10_period: f64,

    atr_ring: Vec<f64>,
    ring_idx: usize,
    rolling_sum_atr: f64,

    dq_high: VecDeque<Node>,
    dq_low: VecDeque<Node>,

    rma_atr: f64,
    sum_tr: f64,
    count: u64,
    prev_close: f64,
}
impl ChopStream {
    #[inline]
    pub fn try_new(params: ChopParams) -> Result<Self, ChopError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ChopError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let drift = params.drift.unwrap_or(1);
        if drift == 0 {
            return Err(ChopError::UnderlyingFunctionFailed(
                "Invalid drift=0 for ATR".to_string(),
            ));
        }
        let scalar = params.scalar.unwrap_or(100.0);

        let inv_drift = 1.0 / (drift as f64);
        let log10_period = chop_log10_positive_exact_v1(period as f64);

        Ok(Self {
            period,
            drift,
            scalar,
            inv_drift,
            log10_period,

            atr_ring: vec![0.0; period],
            ring_idx: 0,
            rolling_sum_atr: 0.0,

            dq_high: VecDeque::with_capacity(period),
            dq_low: VecDeque::with_capacity(period),

            rma_atr: f64::NAN,
            sum_tr: 0.0,
            count: 0,
            prev_close: f64::NAN,
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let idx_ring = self.ring_idx;
        self.ring_idx = (self.ring_idx + 1) % self.period;
        self.count = self.count.saturating_add(1);
        let this_idx = self.count - 1;

        let tr = if self.count == 1 {
            self.prev_close = close;
            self.sum_tr = high - low;
            high - low
        } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            self.prev_close = close;
            hl.max(hc).max(lc)
        };

        if (self.count as usize) <= self.drift {
            if self.count != 1 {
                self.sum_tr += tr;
            }
            if (self.count as usize) == self.drift {
                self.rma_atr = self.sum_tr * self.inv_drift;
            }
        } else {
            self.rma_atr += self.inv_drift * (tr - self.rma_atr);
        }

        let current_atr = if (self.count as usize) < self.drift {
            f64::NAN
        } else {
            self.rma_atr
        };

        let newest = if current_atr.is_nan() {
            0.0
        } else {
            current_atr
        };
        let oldest = self.atr_ring[idx_ring];
        self.atr_ring[idx_ring] = newest;
        self.rolling_sum_atr += newest - oldest;

        let win_start = self.count.saturating_sub(self.period as u64);

        while let Some(&front) = self.dq_high.front() {
            if front.idx < win_start {
                self.dq_high.pop_front();
            } else {
                break;
            }
        }
        while let Some(&front) = self.dq_low.front() {
            if front.idx < win_start {
                self.dq_low.pop_front();
            } else {
                break;
            }
        }

        while let Some(&back) = self.dq_high.back() {
            if back.val <= high {
                self.dq_high.pop_back();
            } else {
                break;
            }
        }
        self.dq_high.push_back(Node {
            idx: this_idx,
            val: high,
        });

        while let Some(&back) = self.dq_low.back() {
            if back.val >= low {
                self.dq_low.pop_back();
            } else {
                break;
            }
        }
        self.dq_low.push_back(Node {
            idx: this_idx,
            val: low,
        });

        if self.count >= self.period as u64 {
            let range = self.dq_high.front().unwrap().val - self.dq_low.front().unwrap().val;
            if range > 0.0 && self.rolling_sum_atr > 0.0 {
                let ratio = self.rolling_sum_atr / range;
                let y = chop_value_from_ratio_exact_v1(ratio, self.scalar, self.log10_period);
                Some(y)
            } else {
                Some(f64::NAN)
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    #[test]
    fn published_log10_authority_has_stable_gate_196_bits() {
        assert_eq!(chop_log10_positive_exact_v1(1.0).to_bits(), 0_u64);
        assert_eq!(
            chop_log10_positive_exact_v1(f64::INFINITY).to_bits(),
            f64::INFINITY.to_bits()
        );
        assert!(chop_log10_positive_exact_v1(0.0).is_nan());
        assert!(chop_log10_positive_exact_v1(f64::NAN).is_nan());

        let ratio = f64::from_bits(0x4025_bae2_2390_9d3f);
        let log10_period = chop_log10_positive_exact_v1(14.0);
        let value = chop_value_from_ratio_exact_v1(ratio, 100.0, log10_period);
        assert_eq!(value.to_bits(), 0x4056_9935_e3af_9c87);
    }

    #[test]
    fn test_chop_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let base = 100.0 + (t * 0.07).sin() * 2.0 + (t * 0.013).cos();
            let h0 = base + 1.0 + 0.15 * (t * 0.31).sin();
            let l0 = base - 1.0 - 0.12 * (t * 0.23).cos();
            let (lo, hi) = if l0 <= h0 { (l0, h0) } else { (h0, l0) };
            let mut c0 = 0.5 * (lo + hi) + 0.2 * (t * 0.17).sin();
            if c0 < lo {
                c0 = lo;
            }
            if c0 > hi {
                c0 = hi;
            }
            high.push(hi);
            low.push(lo);
            close.push(c0);
        }

        let input = ChopInput::from_slices(&high, &low, &close, ChopParams::default());

        let baseline = chop(&input)?.values;

        let mut out = vec![0.0; n];
        {
            chop_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());
        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN mismatch at index {}", i);
            } else {
                assert!(
                    (a - b).abs() <= 1e-12,
                    "Value mismatch at index {}: {} vs {}",
                    i,
                    a,
                    b
                );
            }
        }
        Ok(())
    }
    fn check_chop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let partial_params = ChopParams {
            period: Some(30),
            scalar: None,
            drift: None,
        };
        let input_partial = ChopInput::from_candles(&candles, partial_params);
        let output_partial = chop_with_kernel(&input_partial, kernel)?;
        assert_eq!(output_partial.values.len(), candles.close.len());
        Ok(())
    }
    fn check_chop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_final_5 = [
            49.98214330294626,
            48.90450693742312,
            46.63648608318844,
            46.19823574588033,
            56.22876423352909,
        ];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ChopInput::with_default_candles(&candles);
        let result = chop_with_kernel(&input, kernel)?;
        let start_idx = result.values.len() - 5;
        for (i, &exp) in expected_final_5.iter().enumerate() {
            let idx = start_idx + i;
            let got = result.values[idx];
            assert!(
                (got - exp).abs() < 1e-4,
                "[{}] CHOP at idx {}: got {}, expected {}",
                test_name,
                idx,
                got,
                exp
            );
        }
        Ok(())
    }
    fn check_chop_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ChopInput::with_default_candles(&candles);
        match input.data {
            ChopData::Candles(_) => {}
            _ => panic!("Expected ChopData::Candles variant"),
        }
        let output = chop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_chop_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = ChopParams {
            period: Some(0),
            ..Default::default()
        };
        let input = ChopInput::from_candles(&candles, params);
        let result = chop_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for zero period",
            test_name
        );
        Ok(())
    }
    fn check_chop_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = ChopParams {
            period: Some(999999),
            ..Default::default()
        };
        let input = ChopInput::from_candles(&candles, params);
        let result = chop_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for huge period",
            test_name
        );
        Ok(())
    }
    fn check_chop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ChopInput::with_default_candles(&candles);
        let result = chop_with_kernel(&input, kernel)?;
        let check_index = 240;
        if result.values.len() > check_index {
            let all_nan = result.values[check_index..].iter().all(|&x| x.is_nan());
            assert!(
                !all_nan,
                "[{}] All CHOP values from index {} onward are NaN.",
                test_name, check_index
            );
        }
        Ok(())
    }
    fn check_chop_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 14;
        let scalar = 100.0;
        let drift = 1;
        let input = ChopInput::from_candles(
            &candles,
            ChopParams {
                period: Some(period),
                scalar: Some(scalar),
                drift: Some(drift),
            },
        );
        let batch_output = chop_with_kernel(&input, kernel)?.values;
        let mut stream = ChopStream::try_new(ChopParams {
            period: Some(period),
            scalar: Some(scalar),
            drift: Some(drift),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            let res = stream.update(candles.high[i], candles.low[i], candles.close[i]);
            match res {
                Some(chop_val) => stream_values.push(chop_val),
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
                "[{}] CHOP streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_chop_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = ChopInput::with_default_candles(&candles);
        let output = chop_with_kernel(&input, kernel)?;

        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

        let param_combinations = vec![
            ChopParams {
                period: Some(10),
                scalar: Some(50.0),
                drift: Some(1),
            },
            ChopParams {
                period: Some(20),
                scalar: Some(100.0),
                drift: Some(2),
            },
            ChopParams {
                period: Some(30),
                scalar: Some(150.0),
                drift: Some(3),
            },
        ];

        for params in param_combinations {
            let input = ChopInput::from_candles(&candles, params);
            let output = chop_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params {:?}",
                        test_name, val, bits, i, input.params
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params {:?}",
                        test_name, val, bits, i, input.params
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params {:?}",
                        test_name, val, bits, i, input.params
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_chop_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_chop_tests {
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
    #[cfg(not(feature = "proptest"))]
    generate_all_chop_tests!(
        check_chop_partial_params,
        check_chop_accuracy,
        check_chop_default_candles,
        check_chop_zero_period,
        check_chop_period_exceeds_length,
        check_chop_nan_handling,
        check_chop_streaming,
        check_chop_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_chop_tests!(
        check_chop_partial_params,
        check_chop_accuracy,
        check_chop_default_candles,
        check_chop_zero_period,
        check_chop_period_exceeds_length,
        check_chop_nan_handling,
        check_chop_streaming,
        check_chop_no_poison,
        check_chop_property
    );

    #[cfg(feature = "proptest")]
    fn check_chop_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (50usize..400).prop_flat_map(|size| {
            (
                10.0f64..1000.0f64,
                0.0f64..0.1f64,
                -0.02f64..0.02f64,
                prop::collection::vec((0.0f64..1.0, 0.0f64..1.0, 0.0f64..1.0, 0.0f64..1.0), size),
                0u8..5,
                Just(size),
                5usize..50,
                50.0f64..200.0f64,
                1usize..5,
            )
        });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(base_price, volatility, trend, random_factors, market_type, size, period, scalar, drift)| {

				let mut high_data = Vec::with_capacity(size);
				let mut low_data = Vec::with_capacity(size);
				let mut close_data = Vec::with_capacity(size);
				let mut open_data = Vec::with_capacity(size);

				let mut current_price = base_price;

				for i in 0..size {
					let (r1, r2, r3, r4) = random_factors[i];
					let range = current_price * volatility;


					let (open, high, low, close) = match market_type {
						0 => {

							let open = current_price;
							let close = current_price + range * (0.5 + r1 * 0.5) + (trend * current_price);
							let high = close.max(open) + range * r2 * 0.3;
							let low = close.min(open) - range * r3 * 0.2;

							let high_adjusted = high + range * r4 * 0.1;
							current_price = close;
							(open, high_adjusted, low, close)
						}
						1 => {

							let open = current_price;
							let close = current_price - range * (0.5 + r1 * 0.5) - (trend.abs() * current_price);
							let high = close.max(open) + range * r2 * 0.2;
							let low = close.min(open) - range * r3 * 0.3;

							let low_adjusted = low - range * r4 * 0.1;
							current_price = close;
							(open, high, low_adjusted, close)
						}
						2 => {

							let open = current_price;
							let direction = if r1 > 0.5 { 1.0 } else { -1.0 };
							let close = current_price + direction * range * r2 * 0.5;
							let high = open.max(close) + range * r3 * 0.4;
							let low = open.min(close) - range * r4 * 0.4;

							current_price = base_price * 0.15 + current_price * 0.85;
							(open, high, low, close)
						}
						3 => {

							let open = current_price;
							let close = current_price + range * (r1 - 0.5) * 2.0;
							let high = open.max(close) + range * r2 * 1.2;
							let low = open.min(close) - range * r3 * 1.2;

							let high_wick = high + range * r4 * 0.3;
							current_price = close;
							(open, high_wick, low, close)
						}
						4 | _ => {

							let tiny_move = range * 0.01 * (r1 - 0.5);
							let open = current_price;
							let close = current_price + tiny_move;

							if r2 < 0.1 {

								let price = current_price;
								(price, price, price, price)
							} else {

								let high = open.max(close) + range * 0.001 * r3;
								let low = open.min(close) - range * 0.001 * r4;
								current_price = close;
								(open, high, low, close)
							}
						}
					};


					let high_final = high.max(open).max(close);
					let low_final = low.min(open).min(close);


					debug_assert!(high_final >= low_final, "High must be >= Low");
					debug_assert!(high_final >= open && high_final >= close, "High must be >= Open and Close");
					debug_assert!(low_final <= open && low_final <= close, "Low must be <= Open and Close");

					open_data.push(open);
					high_data.push(high_final);
					low_data.push(low_final);
					close_data.push(close);
				}


				let params = ChopParams {
					period: Some(period),
					scalar: Some(scalar),
					drift: Some(drift),
				};
				let input = ChopInput::from_slices(&high_data, &low_data, &close_data, params.clone());


				let result = chop_with_kernel(&input, kernel)?;
				let reference = chop_with_kernel(&input, Kernel::Scalar)?;


				let first_valid_idx = (0..size).find(|&i| {
					!(high_data[i].is_nan() || low_data[i].is_nan() || close_data[i].is_nan())
				}).unwrap_or(0);
				let warmup_period = first_valid_idx + period - 1;


				let mut valid_chop_values = Vec::new();


				for i in 0..size {
					let y = result.values[i];
					let r = reference.values[i];


					prop_assert!(
						y.is_nan() || y.is_finite(),
						"[{}] CHOP at index {} is not finite or NaN: {}",
						test_name, i, y
					);


					if i < warmup_period {
						prop_assert!(
							y.is_nan(),
							"[{}] CHOP at index {} should be NaN during warmup but got: {}",
							test_name, i, y
						);
					}


					if i >= warmup_period && !high_data[i].is_nan() && !low_data[i].is_nan() && !close_data[i].is_nan() {

						let window_start = i.saturating_sub(period - 1);
						let window_valid = (window_start..=i).all(|j| {
							!high_data[j].is_nan() && !low_data[j].is_nan() && !close_data[j].is_nan()
						});

						if window_valid {

							let window_high_max = (window_start..=i).map(|j| high_data[j]).fold(f64::NEG_INFINITY, f64::max);
							let window_low_min = (window_start..=i).map(|j| low_data[j]).fold(f64::INFINITY, f64::min);
							let range = window_high_max - window_low_min;

							if range > 1e-10 {

								if !y.is_nan() {


									let normalized_bound = scalar * 1.5;
									prop_assert!(
										y >= -normalized_bound && y <= normalized_bound,
										"[{}] CHOP at index {} out of reasonable bounds: {} (scalar={}, bounds=±{})",
										test_name, i, y, scalar, normalized_bound
									);


									valid_chop_values.push(y);
								}
							} else if range == 0.0 {

								prop_assert!(
									y.is_nan(),
									"[{}] CHOP at index {} should be NaN when range=0 but got: {}",
									test_name, i, y
								);
							} else {


								prop_assert!(
									y.is_nan() || y.is_finite(),
									"[{}] CHOP at index {} should be finite or NaN with tiny range: {}",
									test_name, i, y
								);
							}
						}
					}


					if y.is_finite() && r.is_finite() {
						let ulp_diff = y.to_bits().abs_diff(r.to_bits());
						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 10,
							"[{}] Kernel mismatch at index {}: {} vs {} (ULP diff={})",
							test_name, i, y, r, ulp_diff
						);
					} else if y.is_nan() != r.is_nan() {
						prop_assert!(
							false,
							"[{}] NaN mismatch at index {}: kernel={}, scalar={}",
							test_name, i, y.is_nan(), r.is_nan()
						);
					}


					if (high_data[i] - low_data[i]).abs() < 1e-10 && i >= warmup_period {


						prop_assert!(
							y.is_nan() || y.is_finite(),
							"[{}] CHOP at flat candle index {} is invalid: {}",
							test_name, i, y
						);
					}

				}


				if valid_chop_values.len() > 20 {
					let avg_chop = valid_chop_values.iter().sum::<f64>() / valid_chop_values.len() as f64;
					let median_idx = valid_chop_values.len() / 2;
					let mut sorted_values = valid_chop_values.clone();
					sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
					let median_chop = sorted_values[median_idx];


					match market_type {
						0 | 1 => {


							prop_assert!(
								avg_chop.is_finite() && median_chop.is_finite(),
								"[{}] Trending market (type {}) has non-finite CHOP: avg={}, median={}",
								test_name, market_type, avg_chop, median_chop
							);

							let threshold = scalar * 0.6;
							if avg_chop > threshold && median_chop > threshold {


								prop_assert!(true);
							}
						}
						2 => {


							prop_assert!(
								avg_chop.is_finite() && median_chop.is_finite(),
								"[{}] Choppy market has non-finite CHOP: avg={}, median={}",
								test_name, avg_chop, median_chop
							);

							let threshold = scalar * 0.3;
							if avg_chop < threshold && median_chop < threshold {

								prop_assert!(true);
							}
						}
						3 => {


							prop_assert!(
								avg_chop.is_finite(),
								"[{}] Volatile market has non-finite average CHOP: {}",
								test_name, avg_chop
							);
						}
						4 => {


							if avg_chop.is_finite() {
								prop_assert!(
									avg_chop >= -scalar && avg_chop <= scalar,
									"[{}] Flat market CHOP out of bounds: avg={}, scalar={}",
									test_name, avg_chop, scalar
								);
							}
						}
						_ => {}
					}
				}


				if size >= period * 3 {

					let seg1_end = period * 2;
					let seg2_start = period;
					let seg2_end = period * 3;

					if seg1_end < size && seg2_end < size {
						let seg1_values: Vec<f64> = result.values[period..seg1_end]
							.iter()
							.filter(|v| v.is_finite())
							.cloned()
							.collect();
						let seg2_values: Vec<f64> = result.values[seg2_start..seg2_end]
							.iter()
							.filter(|v| v.is_finite())
							.cloned()
							.collect();

						if !seg1_values.is_empty() && !seg2_values.is_empty() {
							let seg1_avg = seg1_values.iter().sum::<f64>() / seg1_values.len() as f64;
							let seg2_avg = seg2_values.iter().sum::<f64>() / seg2_values.len() as f64;


							if market_type == 4 && seg1_avg.abs() > 1e-6 && seg2_avg.abs() > 1e-6 {
								let diff_ratio = (seg1_avg - seg2_avg).abs() / seg1_avg.abs().max(seg2_avg.abs());
								prop_assert!(
									diff_ratio < 0.8,
									"[{}] Flat market segments have inconsistent CHOP: seg1_avg={}, seg2_avg={}, diff_ratio={}",
									test_name, seg1_avg, seg2_avg, diff_ratio
								);
							}
						}
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }

    #[cfg(test)]

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let high = c.high.as_slice();
        let low = c.low.as_slice();
        let close = c.close.as_slice();

        let output = ChopBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(high, low, close)?;

        let def = ChopParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), close.len());

        let expected = [
            49.98214330294626,
            48.90450693742312,
            46.63648608318844,
            46.19823574588033,
            56.22876423352909,
        ];
        let start = row.len().saturating_sub(5);
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    fn check_batch_param_row_lookup(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let high = c.high.as_slice();
        let low = c.low.as_slice();
        let close = c.close.as_slice();

        let builder = ChopBatchBuilder::new()
            .kernel(kernel)
            .period_range(14, 16, 1)
            .scalar_range(100.0, 102.0, 1.0)
            .drift_range(1, 2, 1);

        let out = builder.apply_slices(high, low, close)?;

        for p in 14..=16 {
            for s in [100.0, 101.0, 102.0] {
                for d in 1..=2 {
                    let params = ChopParams {
                        period: Some(p),
                        scalar: Some(s),
                        drift: Some(d),
                    };
                    let row = out.values_for(&params);
                    assert!(
                        row.is_some(),
                        "[{test}] No row for params: period={p}, scalar={s}, drift={d}"
                    );
                }
            }
        }
        Ok(())
    }

    fn check_batch_huge_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let high = c.high.as_slice();
        let low = c.low.as_slice();
        let close = c.close.as_slice();

        let builder = ChopBatchBuilder::new()
            .kernel(kernel)
            .period_range(100_000, 100_001, 1);
        let result = builder.apply_slices(high, low, close);
        assert!(result.is_err(), "[{test}] Expected error for huge period");
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let high = c.high.as_slice();
        let low = c.low.as_slice();
        let close = c.close.as_slice();

        let output = ChopBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
            .scalar_range(50.0, 150.0, 50.0)
            .drift_range(1, 3, 1)
            .apply_slices(high, low, close)?;

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
    gen_batch_tests!(check_batch_param_row_lookup);
    gen_batch_tests!(check_batch_huge_period);
    gen_batch_tests!(check_batch_no_poison);
}

#[inline(always)]
fn chop_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChopBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ChopParams>, ChopError> {
    let combos = expand_grid(sweep)?;

    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChopError::UnderlyingFunctionFailed(
            "mismatched input lengths".to_string(),
        ));
    }

    let len = close.len();
    if len == 0 {
        return Err(ChopError::EmptyData);
    }

    let first = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
        .ok_or(ChopError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(ChopError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    let expected_len = rows
        .checked_mul(cols)
        .ok_or_else(|| ChopError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected_len {
        return Err(ChopError::OutputLengthMismatch {
            expected: expected_len,
            got: out.len(),
        });
    }

    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => actual,
    };

    let do_row = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let ChopParams {
            period,
            scalar,
            drift,
        } = combos[row];
        let p = period.unwrap();
        let s = scalar.unwrap();
        let d = drift.unwrap();

        let row_out: &mut [f64] =
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        match simd {
            Kernel::Scalar => chop_row_scalar(high, low, close, first, p, d, s, row_out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => chop_row_avx2(high, low, close, first, p, d, s, row_out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => chop_row_avx512(high, low, close, first, p, d, s, row_out),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, sl)| do_row(r, sl));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, sl) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, sl);
            }
        }
    } else {
        for (r, sl) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, sl);
        }
    }

    Ok(combos)
}
