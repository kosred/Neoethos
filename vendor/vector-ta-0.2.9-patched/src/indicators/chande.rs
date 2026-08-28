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
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ChandeData<'a> {
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
pub struct ChandeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ChandeParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub direction: Option<String>,
}

impl Default for ChandeParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChandeInput<'a> {
    pub data: ChandeData<'a>,
    pub params: ChandeParams,
}

impl<'a> ChandeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: ChandeParams) -> Self {
        Self {
            data: ChandeData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], p: ChandeParams) -> Self {
        Self {
            data: ChandeData::Slices { high, low, close },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, ChandeParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(22)
    }
    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(3.0)
    }
    #[inline]
    pub fn get_direction(&self) -> &str {
        self.params.direction.as_deref().unwrap_or("long")
    }
    #[inline]
    pub fn borrow_slices(&self) -> (&[f64], &[f64], &[f64]) {
        match &self.data {
            ChandeData::Candles { candles } => (
                source_type(candles, "high"),
                source_type(candles, "low"),
                source_type(candles, "close"),
            ),
            ChandeData::Slices { high, low, close } => (high, low, close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChandeBuilder {
    period: Option<usize>,
    mult: Option<f64>,
    direction: Option<String>,
    kernel: Kernel,
}

impl Default for ChandeBuilder {
    fn default() -> Self {
        Self {
            period: None,
            mult: None,
            direction: None,
            kernel: Kernel::Auto,
        }
    }
}
impl ChandeBuilder {
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
    pub fn mult(mut self, m: f64) -> Self {
        self.mult = Some(m);
        self
    }
    #[inline(always)]
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
        self.direction = Some(d.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ChandeOutput, ChandeError> {
        let p = ChandeParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
        };
        let i = ChandeInput::from_candles(c, p);
        chande_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ChandeOutput, ChandeError> {
        let p = ChandeParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
        };
        let i = ChandeInput::from_slices(high, low, close, p);
        chande_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ChandeStream, ChandeError> {
        let p = ChandeParams {
            period: self.period,
            mult: self.mult,
            direction: self.direction,
        };
        ChandeStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ChandeError {
    #[error("chande: Input series are empty.")]
    EmptyInputData,
    #[error("chande: All values are NaN.")]
    AllValuesNaN,
    #[error("chande: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("chande: not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("chande: input length mismatch: high={h}, low={l}, close={c}")]
    DataLengthMismatch { h: usize, l: usize, c: usize },
    #[error("chande: Invalid direction: {direction}")]
    InvalidDirection { direction: String },
    #[error("chande: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("chande: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: isize,
        end: isize,
        step: isize,
    },
    #[error("chande: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("chande: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
fn first_valid3(h: &[f64], l: &[f64], c: &[f64]) -> Option<usize> {
    let n = h.len().min(l.len()).min(c.len());
    (0..n).find(|&i| !h[i].is_nan() && !l[i].is_nan() && !c[i].is_nan())
}

#[inline]
pub fn chande(input: &ChandeInput) -> Result<ChandeOutput, ChandeError> {
    chande_with_kernel(input, Kernel::Auto)
}

pub fn chande_with_kernel(
    input: &ChandeInput,
    kernel: Kernel,
) -> Result<ChandeOutput, ChandeError> {
    let (high, low, close) = input.borrow_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }

    let len = high.len();
    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;
    let period = input.get_period();
    let mult = input.get_mult();
    let dir = {
        let d = input.get_direction();
        if d.eq_ignore_ascii_case("long") {
            "long"
        } else if d.eq_ignore_ascii_case("short") {
            "short"
        } else {
            return Err(ChandeError::InvalidDirection {
                direction: d.to_string(),
            });
        }
    };
    if period == 0 || period > len {
        return Err(ChandeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(ChandeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let chosen = match (
        chosen,
        cfg!(all(feature = "nightly-avx", target_arch = "x86_64")),
    ) {
        (Kernel::Avx512 | Kernel::Avx512Batch, false)
        | (Kernel::Avx2 | Kernel::Avx2Batch, false) => Kernel::Scalar,
        (k, _) => k,
    };

    let warmup = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warmup);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                chande_scalar(high, low, close, period, mult, dir, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                chande_avx2(high, low, close, period, mult, dir, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_avx512(high, low, close, period, mult, dir, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(ChandeOutput { values: out })
}

#[inline]
pub fn chande_into(input: &ChandeInput, out: &mut [f64]) -> Result<(), ChandeError> {
    chande_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn chande_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    direction: &str,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), ChandeError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }
    if out.len() != high.len() {
        return Err(ChandeError::OutputLengthMismatch {
            expected: high.len(),
            got: out.len(),
        });
    }
    let len = high.len();
    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(ChandeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(ChandeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let dir = if direction.eq_ignore_ascii_case("long") {
        "long"
    } else if direction.eq_ignore_ascii_case("short") {
        "short"
    } else {
        return Err(ChandeError::InvalidDirection {
            direction: direction.to_string(),
        });
    };

    let warmup = first + period - 1;
    let warmup_end = warmup.min(out.len());
    for v in &mut out[..warmup_end] {
        *v = f64::NAN;
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    let chosen = match (
        chosen,
        cfg!(all(feature = "nightly-avx", target_arch = "x86_64")),
    ) {
        (Kernel::Avx512 | Kernel::Avx512Batch, false)
        | (Kernel::Avx2 | Kernel::Avx2Batch, false) => Kernel::Scalar,
        (k, _) => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                chande_scalar(high, low, close, period, mult, dir, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                chande_avx2(high, low, close, period, mult, dir, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_avx512(high, low, close, period, mult, dir, first, out)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline]
pub fn chande_into_slice(
    dst: &mut [f64],
    input: &ChandeInput,
    kern: Kernel,
) -> Result<(), ChandeError> {
    let (high, low, close) = input.borrow_slices();
    let p = input.get_period();
    let m = input.get_mult();
    let d = input.get_direction();
    chande_compute_into(high, low, close, p, m, d, kern, dst)
}

#[inline]
pub fn chande_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    if period == 22 && mult == 3.0 && dir == "long" {
        chande_scalar_default_long(high, low, close, first, out);
        return;
    }

    let len = high.len();
    if first >= len {
        return;
    }

    let alpha = 1.0 / period as f64;
    let warmup = first + period - 1;

    let mut sum_tr = 0.0f64;
    let mut rma = 0.0f64;
    let mut prev_close = close[first];

    use std::collections::VecDeque;

    if dir == "long" {
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            let hi = high[i];
            let lo = low[i];
            let tr = if i == first {
                hi - lo
            } else {
                let hl = hi - lo;
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                hl.max(hc).max(lc)
            };

            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }

            while let Some(&j) = dq.back() {
                if high[j] <= hi {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;

                let max_h = high[*dq.front().expect("deque nonempty at warmup")];
                out[i] = (-rma).mul_add(mult, max_h);
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                let max_h = high[*dq.front().expect("deque nonempty in steady state")];
                out[i] = (-rma).mul_add(mult, max_h);
            }

            prev_close = close[i];
        }
    } else {
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            let hi = high[i];
            let lo = low[i];
            let tr = if i == first {
                hi - lo
            } else {
                let hl = hi - lo;
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                hl.max(hc).max(lc)
            };

            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }

            while let Some(&j) = dq.back() {
                if low[j] >= lo {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;

                let min_l = low[*dq.front().expect("deque nonempty at warmup")];
                out[i] = rma.mul_add(mult, min_l);
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                let min_l = low[*dq.front().expect("deque nonempty in steady state")];
                out[i] = rma.mul_add(mult, min_l);
            }

            prev_close = close[i];
        }
    }
}

#[inline(always)]
fn chande_scalar_default_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    out: &mut [f64],
) {
    const PERIOD: usize = 22;
    const MASK: usize = 31;
    const ALPHA: f64 = 1.0 / 22.0;
    const MULT: f64 = 3.0;

    let len = high.len();
    if first >= len {
        return;
    }

    let warmup = first + PERIOD - 1;
    let mut sum_tr = 0.0f64;
    let mut rma = 0.0f64;
    let mut prev_close = close[first];
    let mut dq = [0usize; 32];
    let mut head = 0usize;
    let mut count = 0usize;

    for i in first..len {
        let hi = high[i];
        let lo = low[i];
        let tr = if i == first {
            hi - lo
        } else {
            let hl = hi - lo;
            let hc = (hi - prev_close).abs();
            let lc = (lo - prev_close).abs();
            hl.max(hc).max(lc)
        };

        if i >= warmup {
            let window_start = i + 1 - PERIOD;
            while count != 0 && dq[head] < window_start {
                head = (head + 1) & MASK;
                count -= 1;
            }
        }

        while count != 0 {
            let back = (head + count - 1) & MASK;
            if high[dq[back]] <= hi {
                count -= 1;
            } else {
                break;
            }
        }
        let tail = (head + count) & MASK;
        dq[tail] = i;
        count += 1;

        if i < warmup {
            sum_tr += tr;
        } else if i == warmup {
            sum_tr += tr;
            rma = sum_tr / PERIOD as f64;
            let max_h = high[dq[head]];
            out[i] = (-rma).mul_add(MULT, max_h);
        } else {
            rma = ALPHA.mul_add(tr - rma, rma);
            let max_h = high[dq[head]];
            out[i] = (-rma).mul_add(MULT, max_h);
        }

        prev_close = close[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    unsafe { chande_fast_unchecked(high, low, close, period, mult, dir, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    unsafe { chande_fast_unchecked(high, low, close, period, mult, dir, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_fast_unchecked(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    mult: f64,
    dir: &str,
    first: usize,
    out: &mut [f64],
) {
    use std::collections::VecDeque;
    let len = high.len();
    if first >= len {
        return;
    }
    let alpha = 1.0 / period as f64;
    let warmup = first + period - 1;

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let op = out.as_mut_ptr();

    let mut prev_close = *cp.add(first);
    let mut sum_tr = 0.0f64;
    let mut rma = 0.0f64;

    if dir == "long" {
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            let hi = *hp.add(i);
            let lo = *lp.add(i);
            let hl = hi - lo;
            let tr = if i == first {
                hl
            } else {
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                let t = if hl >= hc { hl } else { hc };
                if t >= lc { t } else { lc }
            };

            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            while let Some(&j) = dq.back() {
                if *hp.add(j) <= hi {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;
                let max_h = *hp.add(*dq.front().unwrap());
                *op.add(i) = (-rma).mul_add(mult, max_h);
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                let max_h = *hp.add(*dq.front().unwrap());
                *op.add(i) = (-rma).mul_add(mult, max_h);
            }
            prev_close = *cp.add(i);
        }
    } else {
        let mut dq: VecDeque<usize> = VecDeque::with_capacity(period);
        for i in first..len {
            let hi = *hp.add(i);
            let lo = *lp.add(i);
            let hl = hi - lo;
            let tr = if i == first {
                hl
            } else {
                let hc = (hi - prev_close).abs();
                let lc = (lo - prev_close).abs();
                let t = if hl >= hc { hl } else { hc };
                if t >= lc { t } else { lc }
            };

            if i >= warmup {
                let window_start = i + 1 - period;
                while let Some(&j) = dq.front() {
                    if j < window_start {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            while let Some(&j) = dq.back() {
                if *lp.add(j) >= lo {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);

            if i < warmup {
                sum_tr += tr;
            } else if i == warmup {
                sum_tr += tr;
                rma = sum_tr / period as f64;
                let min_l = *lp.add(*dq.front().unwrap());
                *op.add(i) = rma.mul_add(mult, min_l);
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                let min_l = *lp.add(*dq.front().unwrap());
                *op.add(i) = rma.mul_add(mult, min_l);
            }
            prev_close = *cp.add(i);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChandeStream {
    period: usize,
    mult: f64,
    direction: String,
    is_long: bool,

    alpha: f64,

    atr: f64,
    close_prev: f64,
    t: usize,
    warm: usize,
    filled: bool,

    max_deque: std::collections::VecDeque<(f64, usize)>,
    min_deque: std::collections::VecDeque<(f64, usize)>,
}

impl ChandeStream {
    pub fn try_new(params: ChandeParams) -> Result<Self, ChandeError> {
        let period = params.period.unwrap_or(22);
        let mult = params.mult.unwrap_or(3.0);
        let direction = params
            .direction
            .unwrap_or_else(|| "long".into())
            .to_lowercase();

        if period == 0 {
            return Err(ChandeError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if direction != "long" && direction != "short" {
            return Err(ChandeError::InvalidDirection { direction });
        }

        let is_long = direction == "long";
        Ok(Self {
            period,
            mult,
            direction,
            is_long,
            alpha: 1.0 / period as f64,
            atr: 0.0,
            close_prev: f64::NAN,
            t: 0,
            warm: 0,
            filled: false,
            max_deque: std::collections::VecDeque::with_capacity(period),
            min_deque: std::collections::VecDeque::with_capacity(period),
        })
    }

    #[inline(always)]
    fn evict_old(&mut self) {
        let window_start = self.t.saturating_sub(self.period - 1);
        if self.is_long {
            while let Some(&(_, idx)) = self.max_deque.front() {
                if idx < window_start {
                    self.max_deque.pop_front();
                } else {
                    break;
                }
            }
        } else {
            while let Some(&(_, idx)) = self.min_deque.front() {
                if idx < window_start {
                    self.min_deque.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    #[inline(always)]
    fn push_max(&mut self, v: f64) {
        while let Some(&(back, _)) = self.max_deque.back() {
            if back <= v {
                self.max_deque.pop_back();
            } else {
                break;
            }
        }
        self.max_deque.push_back((v, self.t));
    }

    #[inline(always)]
    fn push_min(&mut self, v: f64) {
        while let Some(&(back, _)) = self.min_deque.back() {
            if back >= v {
                self.min_deque.pop_back();
            } else {
                break;
            }
        }
        self.min_deque.push_back((v, self.t));
    }

    #[inline(always)]
    fn tr(&self, high: f64, low: f64) -> f64 {
        if self.warm == 0 {
            high - low
        } else {
            let max_h = if high > self.close_prev {
                high
            } else {
                self.close_prev
            };
            let min_l = if low < self.close_prev {
                low
            } else {
                self.close_prev
            };
            max_h - min_l
        }
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = self.tr(high, low);

        if !self.filled {
            if self.is_long {
                self.push_max(high);
            } else {
                self.push_min(low);
            }
            self.atr += tr;
            self.warm += 1;

            let now_ready = self.warm == self.period;
            if now_ready {
                self.atr *= self.alpha;
                self.filled = true;
            }

            self.close_prev = close;
            self.t = self.t.wrapping_add(1);

            if !now_ready {
                return None;
            }

            if self.is_long {
                let m = self.max_deque.front().unwrap().0;
                Some((-self.atr).mul_add(self.mult, m))
            } else {
                let m = self.min_deque.front().unwrap().0;
                Some(self.atr.mul_add(self.mult, m))
            }
        } else {
            self.evict_old();
            if self.is_long {
                self.push_max(high);
            } else {
                self.push_min(low);
            }

            self.atr = self.alpha.mul_add(tr - self.atr, self.atr);

            self.close_prev = close;
            self.t = self.t.wrapping_add(1);

            if self.is_long {
                let m = self.max_deque.front().unwrap().0;
                Some((-self.atr).mul_add(self.mult, m))
            } else {
                let m = self.min_deque.front().unwrap().0;
                Some(self.atr.mul_add(self.mult, m))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChandeBatchRange {
    pub period: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for ChandeBatchRange {
    fn default() -> Self {
        Self {
            period: (22, 22, 0),
            mult: (3.0, 3.249, 0.001),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChandeBatchBuilder {
    range: ChandeBatchRange,
    direction: String,
    kernel: Kernel,
}

impl ChandeBatchBuilder {
    pub fn new() -> Self {
        Self {
            range: ChandeBatchRange::default(),
            direction: "long".into(),
            kernel: Kernel::Auto,
        }
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
        self.direction = d.into();
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }
    pub fn mult_static(mut self, m: f64) -> Self {
        self.range.mult = (m, m, 0.0);
        self
    }

    pub fn apply_candles(self, c: &Candles) -> Result<ChandeBatchOutput, ChandeError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        chande_batch_with_kernel(high, low, close, &self.range, &self.direction, self.kernel)
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ChandeBatchOutput, ChandeError> {
        chande_batch_with_kernel(high, low, close, &self.range, &self.direction, self.kernel)
    }
}

pub fn chande_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    direction: &str,
    k: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(ChandeError::InvalidKernelForBatch(other));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    chande_batch_par_slice(high, low, close, sweep, direction, simd)
}

#[derive(Clone, Debug)]
pub struct ChandeBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ChandeParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ChandeBatchOutput {
    pub fn row_for_params(&self, p: &ChandeParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(22) == p.period.unwrap_or(22)
                && (c.mult.unwrap_or(3.0) - p.mult.unwrap_or(3.0)).abs() < 1e-12
                && c.direction.as_deref().unwrap_or("long")
                    == p.direction.as_deref().unwrap_or("long")
        })
    }
    pub fn values_for(&self, p: &ChandeParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ChandeBatchRange, dir: &str) -> Result<Vec<ChandeParams>, ChandeError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, ChandeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        if start < end {
            if step == 0 {
                return Ok(vec![start]);
            }
            Ok((start..=end).step_by(step).collect())
        } else {
            let step_i = step as isize;
            if step_i == 0 {
                return Ok(vec![start]);
            }
            let mut vals = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            while x >= end_i {
                vals.push(x as usize);
                x = x.saturating_sub(step_i);
                if step_i <= 0 {
                    break;
                }
            }
            if vals.is_empty() {
                return Err(ChandeError::InvalidRange {
                    start: start as isize,
                    end: end as isize,
                    step: step as isize,
                });
            }
            Ok(vals)
        }
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, ChandeError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += step;
            }
        } else {
            let mut x = start;
            let st = -step.abs();
            while x >= end - 1e-12 {
                v.push(x);
                x += st;
            }
        }
        if v.is_empty() {
            return Err(ChandeError::InvalidRange {
                start: start as isize,
                end: end as isize,
                step: step as isize,
            });
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let mults = axis_f64(r.mult)?;

    let cap = periods
        .len()
        .checked_mul(mults.len())
        .ok_or(ChandeError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        })?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &m in &mults {
            out.push(ChandeParams {
                period: Some(p),
                mult: Some(m),
                direction: Some(dir.to_string()),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn chande_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    chande_batch_inner(high, low, close, sweep, dir, kern, false)
}

#[inline(always)]
pub fn chande_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    chande_batch_inner(high, low, close, sweep, dir, kern, true)
}

#[inline(always)]
fn chande_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
    parallel: bool,
) -> Result<ChandeBatchOutput, ChandeError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }

    let combos = expand_grid(sweep, dir)?;
    if combos.is_empty() {
        return Err(ChandeError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ChandeError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();

    let _total = rows
        .checked_mul(cols)
        .ok_or(ChandeError::InvalidInput("rows*cols overflow".into()))?;

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].mult.unwrap();
        let direction = combos[row].direction.as_deref().unwrap();
        match kern {
            Kernel::Scalar => {
                chande_row_scalar(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                chande_row_avx2(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                chande_row_avx512(high, low, close, first, period, mult, direction, out_row)
            }
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
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

    Ok(ChandeBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn chande_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    dir: &str,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ChandeParams>, ChandeError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ChandeError::EmptyInputData);
    }
    if !(high.len() == low.len() && low.len() == close.len()) {
        return Err(ChandeError::DataLengthMismatch {
            h: high.len(),
            l: low.len(),
            c: close.len(),
        });
    }

    let combos = expand_grid(sweep, dir)?;
    if combos.is_empty() {
        return Err(ChandeError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }

    let first = first_valid3(high, low, close).ok_or(ChandeError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ChandeError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }

    let cols = high.len();

    let expected = combos
        .len()
        .checked_mul(cols)
        .ok_or_else(|| ChandeError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(ChandeError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.period.unwrap() - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].mult.unwrap();
        let direction = combos[row].direction.as_deref().unwrap();
        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                chande_row_scalar(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                chande_row_avx2(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_row_avx512(high, low, close, first, period, mult, direction, out_row)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                chande_row_scalar(high, low, close, first, period, mult, direction, out_row)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
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

    Ok(combos)
}

#[inline(always)]
unsafe fn chande_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_scalar(high, low, close, period, mult, dir, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    if period <= 32 {
        chande_row_avx512_short(high, low, close, first, period, mult, dir, out)
    } else {
        chande_row_avx512_long(high, low, close, first, period, mult, dir, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    mult: f64,
    dir: &str,
    out: &mut [f64],
) {
    chande_fast_unchecked(high, low, close, period, mult, dir, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    #[test]
    fn test_chande_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = ChandeInput::with_default_candles(&candles);

        let baseline = chande(&input)?;

        let mut out = vec![0.0f64; candles.close.len()];
        {
            chande_into(&input, &mut out)?;
        }

        assert_eq!(baseline.values.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline.values[i], out[i]),
                "Mismatch at index {}: got {}, expected {}",
                i,
                out[i],
                baseline.values[i]
            );
        }
        Ok(())
    }

    fn check_chande_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = ChandeParams {
            period: None,
            mult: None,
            direction: None,
        };
        let input = ChandeInput::from_candles(&candles, default_params);
        let output = chande_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_chande_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close_prices = &candles.close;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let chande_result = chande_with_kernel(&input, kernel)?;

        assert_eq!(chande_result.values.len(), close_prices.len());

        let expected_last_five = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
        ];

        assert!(chande_result.values.len() >= 5);
        let start_idx = chande_result.values.len() - 5;
        let actual_last_five = &chande_result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] Chande Exits mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                exp,
                val
            );
        }

        let period = 22;
        for i in 0..(period - 1) {
            assert!(
                chande_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = ChandeInput::with_default_candles(&candles);
        let default_output = chande_with_kernel(&default_input, kernel)?;
        assert_eq!(default_output.values.len(), close_prices.len());
        Ok(())
    }

    fn check_chande_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = ChandeParams {
            period: Some(0),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Chande should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_chande_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = ChandeParams {
            period: Some(99999),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Chande should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_chande_bad_direction(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("bad".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Chande should fail with bad direction",
            test_name
        );
        Ok(())
    }

    fn check_chande_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let result = chande_with_kernel(&input, kernel)?;

        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] Unexpected NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_chande_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params.clone());
        let batch_output = chande_with_kernel(&input, kernel)?.values;

        let mut stream = ChandeStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), &c) in candles.high.iter().zip(&candles.low).zip(&candles.close) {
            match stream.update(h, l, c) {
                Some(chande_val) => stream_values.push(chande_val),
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
                diff < 1e-8,
                "[{}] Chande streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_chande_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let param_combinations = vec![
            ChandeParams {
                period: Some(10),
                mult: Some(2.0),
                direction: Some("long".into()),
            },
            ChandeParams {
                period: Some(22),
                mult: Some(3.0),
                direction: Some("short".into()),
            },
            ChandeParams {
                period: Some(50),
                mult: Some(5.0),
                direction: Some("long".into()),
            },
        ];

        for params in param_combinations {
            let input = ChandeInput::from_candles(&candles, params.clone());
            let output = chande_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap(),
                        params.mult.unwrap(),
                        params.direction.as_ref().unwrap()
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap(),
                        params.mult.unwrap(),
                        params.direction.as_ref().unwrap()
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap(),
                        params.mult.unwrap(),
                        params.direction.as_ref().unwrap()
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_chande_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_chande_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_chande_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                )
                .prop_flat_map(move |close| {
                    let len = close.len();
                    (
                        Just(close.clone()),
                        prop::collection::vec(0.0f64..1000.0f64, len),
                        prop::collection::vec(0.0f64..1000.0f64, len),
                    )
                        .prop_map(move |(c, high_spread, low_spread)| {
                            let high: Vec<f64> = c
                                .iter()
                                .zip(&high_spread)
                                .map(|(&close_val, &spread)| close_val + spread)
                                .collect();
                            let low: Vec<f64> = c
                                .iter()
                                .zip(&low_spread)
                                .map(|(&close_val, &spread)| close_val - spread)
                                .collect();
                            (high, low, c.clone())
                        })
                }),
                Just(period),
                0.1f64..10.0f64,
                prop::bool::ANY,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |((high, low, close), period, mult, is_long)| {
                let direction = if is_long { "long" } else { "short" };

                let candles = Candles {
                    high: high.clone(),
                    low: low.clone(),
                    close: close.clone(),
                    timestamp: vec![],
                    open: vec![],
                    volume: vec![],
                    fields: crate::utilities::data_loader::CandleFieldFlags {
                        open: false,
                        high: true,
                        low: true,
                        close: true,
                        volume: false,
                    },
                    hl2: vec![],
                    hlc3: vec![],
                    ohlc4: vec![],
                    hlcc4: vec![],
                };

                let params = ChandeParams {
                    period: Some(period),
                    mult: Some(mult),
                    direction: Some(direction.to_string()),
                };

                let input = ChandeInput::from_candles(&candles, params);

                let result = chande_with_kernel(&input, kernel);

                prop_assert!(result.is_ok(), "Chande should succeed for valid inputs");
                let output = result.unwrap();

                prop_assert_eq!(
                    output.values.len(),
                    high.len(),
                    "Output length should match input"
                );

                let first_valid = close.iter().position(|&x| !x.is_nan()).unwrap_or(0);
                let warmup_period = first_valid + period - 1;

                for i in 0..warmup_period.min(output.values.len()) {
                    prop_assert!(
                        output.values[i].is_nan(),
                        "Expected NaN during warmup at index {}",
                        i
                    );
                }

                if warmup_period < output.values.len() {
                    for i in warmup_period..output.values.len() {
                        let val = output.values[i];
                        prop_assert!(
                            val.is_finite(),
                            "Expected finite value after warmup at index {}, got {}",
                            i,
                            val
                        );
                    }
                }

                for i in warmup_period..output.values.len() {
                    let start_idx = i + 1 - period;
                    let period_high = high[start_idx..=i].iter().cloned().fold(f64::MIN, f64::max);
                    let period_low = low[start_idx..=i].iter().cloned().fold(f64::MAX, f64::min);
                    let val = output.values[i];

                    if is_long {
                        prop_assert!(
                            val <= period_high + 1e-6,
                            "Long exit {} should be <= period high {} at index {}",
                            val,
                            period_high,
                            i
                        );
                    } else {
                        prop_assert!(
                            val >= period_low - 1e-6,
                            "Short exit {} should be >= period low {} at index {}",
                            val,
                            period_low,
                            i
                        );
                    }
                }

                let ref_output = chande_with_kernel(&input, Kernel::Scalar).unwrap();
                for i in 0..output.values.len() {
                    let val = output.values[i];
                    let ref_val = ref_output.values[i];

                    if !val.is_finite() || !ref_val.is_finite() {
                        prop_assert_eq!(
                            val.to_bits(),
                            ref_val.to_bits(),
                            "NaN/Inf mismatch at index {}: {} vs {}",
                            i,
                            val,
                            ref_val
                        );
                        continue;
                    }

                    let val_bits = val.to_bits();
                    let ref_bits = ref_val.to_bits();
                    let ulp_diff = val_bits.abs_diff(ref_bits);

                    prop_assert!(
                        (val - ref_val).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        val,
                        ref_val,
                        ulp_diff
                    );
                }

                if period == 1 && warmup_period < output.values.len() {
                    for i in warmup_period..output.values.len() {
                        let val = output.values[i];
                        prop_assert!(
                            val.is_finite(),
                            "Period=1 should produce finite values at index {}",
                            i
                        );

                        if is_long {
                            prop_assert!(
                                val <= high[i] + 1e-6,
                                "Period=1 long exit {} should be <= high {} at index {}",
                                val,
                                high[i],
                                i
                            );
                        } else {
                            prop_assert!(
                                val >= low[i] - 1e-6,
                                "Period=1 short exit {} should be >= low {} at index {}",
                                val,
                                low[i],
                                i
                            );
                        }
                    }
                }

                let all_same_close = close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                let all_same_high = high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                let all_same_low = low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);

                if all_same_close
                    && all_same_high
                    && all_same_low
                    && warmup_period + 10 < output.values.len()
                {
                    let stable_start = warmup_period + period;
                    if stable_start + 2 < output.values.len() {
                        for i in stable_start..output.values.len() - 1 {
                            prop_assert!(
                                (output.values[i] - output.values[i + 1]).abs() <= 1e-6,
                                "Constant data should produce stable output at index {}: {} vs {}",
                                i,
                                output.values[i],
                                output.values[i + 1]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_chande_tests!(
        check_chande_partial_params,
        check_chande_accuracy,
        check_chande_zero_period,
        check_chande_period_exceeds_length,
        check_chande_bad_direction,
        check_chande_nan_handling,
        check_chande_streaming,
        check_chande_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_chande_tests!(check_chande_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = ChandeBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = ChandeParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
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

        let output = ChandeBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
            .mult_range(2.0, 5.0, 1.5)
            .direction("long")
            .apply_candles(&c)?;

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
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    params.period.unwrap(),
                    params.mult.unwrap(),
                    params.direction.as_ref().unwrap()
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    params.period.unwrap(),
                    params.mult.unwrap(),
                    params.direction.as_ref().unwrap()
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    params.period.unwrap(),
                    params.mult.unwrap(),
                    params.direction.as_ref().unwrap()
                );
            }
        }

        let output_short = ChandeBatchBuilder::new()
            .kernel(kernel)
            .period_range(15, 45, 15)
            .mult_range(1.0, 4.0, 1.5)
            .direction("short")
            .apply_candles(&c)?;

        for (idx, &val) in output_short.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output_short.cols;
            let col = idx % output_short.cols;
            let params = &output_short.combos[row];

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    params.period.unwrap(),
                    params.mult.unwrap(),
                    params.direction.as_ref().unwrap()
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    params.period.unwrap(),
                    params.mult.unwrap(),
                    params.direction.as_ref().unwrap()
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test,
                    val,
                    bits,
                    row,
                    col,
                    idx,
                    params.period.unwrap(),
                    params.mult.unwrap(),
                    params.direction.as_ref().unwrap()
                );
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
