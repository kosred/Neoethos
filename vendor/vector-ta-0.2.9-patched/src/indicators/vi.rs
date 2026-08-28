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
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ViData<'a> {
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
pub struct ViOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ViParams {
    pub period: Option<usize>,
}

impl Default for ViParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ViInput<'a> {
    pub data: ViData<'a>,
    pub params: ViParams,
}

impl<'a> ViInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ViParams) -> Self {
        Self {
            data: ViData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ViParams,
    ) -> Self {
        Self {
            data: ViData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ViData::Candles { candles },
            params: ViParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ViBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ViBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ViBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ViOutput, ViError> {
        let p = ViParams {
            period: self.period,
        };
        let i = ViInput::from_candles(c, p);
        vi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ViOutput, ViError> {
        let p = ViParams {
            period: self.period,
        };
        let i = ViInput::from_slices(high, low, close, p);
        vi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ViStream, ViError> {
        let p = ViParams {
            period: self.period,
        };
        ViStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ViError {
    #[error("vi: Empty data provided.")]
    EmptyInputData,
    #[error("vi: All values are NaN.")]
    AllValuesNaN,
    #[error("vi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vi: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vi: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("vi: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("vi: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn vi(input: &ViInput) -> Result<ViOutput, ViError> {
    vi_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn vi_prepare<'a>(
    input: &'a ViInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), ViError> {
    let (high, low, close) = match &input.data {
        ViData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        ViData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ViError::EmptyInputData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(ViError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ViError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(ViError::AllValuesNaN)?;

    if len - first < period {
        return Err(ViError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((high, low, close, period, first, chosen))
}

#[inline(always)]
fn vi_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vi_scalar(high, low, close, period, first, plus, minus)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vi_avx2(high, low, close, period, first, plus, minus)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vi_avx512(high, low, close, period, first, plus, minus)
            }
            _ => unreachable!(),
        }
    }
}

pub fn vi_with_kernel(input: &ViInput, kernel: Kernel) -> Result<ViOutput, ViError> {
    let (h, l, c, period, first, chosen) = vi_prepare(input, kernel)?;
    let mut plus = alloc_with_nan_prefix(h.len(), first + period - 1);
    let mut minus = alloc_with_nan_prefix(h.len(), first + period - 1);
    vi_compute_into(h, l, c, period, first, chosen, &mut plus, &mut minus);
    Ok(ViOutput { plus, minus })
}

pub fn vi_into_slice(
    dst_plus: &mut [f64],
    dst_minus: &mut [f64],
    input: &ViInput,
    kernel: Kernel,
) -> Result<(), ViError> {
    let (h, l, c, period, first, chosen) = vi_prepare(input, kernel)?;
    if dst_plus.len() != h.len() || dst_minus.len() != h.len() {
        let expected = h.len();
        let got = dst_plus.len().max(dst_minus.len());
        return Err(ViError::OutputLengthMismatch { expected, got });
    }
    vi_compute_into(h, l, c, period, first, chosen, dst_plus, dst_minus);
    let warm = first + period - 1;
    for i in 0..warm {
        dst_plus[i] = f64::NAN;
        dst_minus[i] = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
pub unsafe fn vi_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    let n = high.len();
    if n == 0 {
        return;
    }

    let warm = first + period - 1;

    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let p_out = plus.as_mut_ptr();
    let m_out = minus.as_mut_ptr();

    let mut tr_buf: Vec<f64> = Vec::with_capacity(period);
    let mut vp_buf: Vec<f64> = Vec::with_capacity(period);
    let mut vm_buf: Vec<f64> = Vec::with_capacity(period);
    tr_buf.set_len(period);
    vp_buf.set_len(period);
    vm_buf.set_len(period);
    let trp = tr_buf.as_mut_ptr();
    let vpp = vp_buf.as_mut_ptr();
    let vmp = vm_buf.as_mut_ptr();

    let mut prev_h = *h.add(first);
    let mut prev_l = *l.add(first);
    let mut prev_c = *c.add(first);

    let mut sum_tr = prev_h - prev_l;
    let mut sum_vp = 0.0f64;
    let mut sum_vm = 0.0f64;

    *trp.add(0) = sum_tr;
    *vpp.add(0) = 0.0;
    *vmp.add(0) = 0.0;

    if period == 1 {
        *p_out.add(warm) = 0.0;
        *m_out.add(warm) = 0.0;
    }

    let mut r = if period == 1 { 0 } else { 1 };

    let mut i = first + 1;
    while i < n {
        let hi = *h.add(i);
        let lo = *l.add(i);

        let hl = hi - lo;
        let hc = (hi - prev_c).abs();
        let lc = (lo - prev_c).abs();
        let mut tr_new = if hl > hc { hl } else { hc };
        if lc > tr_new {
            tr_new = lc;
        }

        let vp_new = (hi - prev_l).abs();
        let vm_new = (lo - prev_h).abs();

        if i <= warm {
            sum_tr += tr_new;
            sum_vp += vp_new;
            sum_vm += vm_new;

            *trp.add(r) = tr_new;
            *vpp.add(r) = vp_new;
            *vmp.add(r) = vm_new;

            if i == warm {
                *p_out.add(i) = sum_vp / sum_tr;
                *m_out.add(i) = sum_vm / sum_tr;
            }
        } else {
            let tr_old = *trp.add(r);
            let vp_old = *vpp.add(r);
            let vm_old = *vmp.add(r);

            sum_tr += tr_new - tr_old;
            sum_vp += vp_new - vp_old;
            sum_vm += vm_new - vm_old;

            *trp.add(r) = tr_new;
            *vpp.add(r) = vp_new;
            *vmp.add(r) = vm_new;

            *p_out.add(i) = sum_vp / sum_tr;
            *m_out.add(i) = sum_vm / sum_tr;
        }

        prev_h = hi;
        prev_l = lo;
        prev_c = *c.add(i);

        r += 1;
        if r == period {
            r = 0;
        }

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    if period <= 32 {
        vi_avx512_short(high, low, close, period, first, plus, minus);
    } else {
        vi_avx512_long(high, low, close, period, first, plus, minus);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

#[derive(Debug, Clone)]
pub struct ViStream {
    period: usize,
    tr: Vec<f64>,
    vp: Vec<f64>,
    vm: Vec<f64>,
    idx: usize,
    filled: bool,
    sum_tr: f64,
    sum_vp: f64,
    sum_vm: f64,
}

impl ViStream {
    pub fn try_new(params: ViParams) -> Result<Self, ViError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ViError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            tr: vec![0.0; period],
            vp: vec![0.0; period],
            vm: vec![0.0; period],
            idx: 0,
            filled: false,
            sum_tr: 0.0,
            sum_vp: 0.0,
            sum_vm: 0.0,
        })
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        prev_low: f64,
        prev_high: f64,
        prev_close: f64,
    ) -> Option<(f64, f64)> {
        let _ = close;

        let i = self.idx;

        let hl = high - low;
        let hc = (high - prev_close).abs();
        let lc = (low - prev_close).abs();
        let tr_new = hl.max(hc.max(lc));

        let vp_new = (high - prev_low).abs();
        let vm_new = (low - prev_high).abs();

        let tr_old = self.tr[i];
        let vp_old = self.vp[i];
        let vm_old = self.vm[i];

        self.sum_tr += tr_new - tr_old;
        self.sum_vp += vp_new - vp_old;
        self.sum_vm += vm_new - vm_old;

        self.tr[i] = tr_new;
        self.vp[i] = vp_new;
        self.vm[i] = vm_new;

        self.idx += 1;
        if self.idx == self.period {
            self.idx = 0;
            self.filled = true;
        }

        if self.filled {
            let inv_tr = 1.0 / self.sum_tr;
            let vi_p = self.sum_vp * inv_tr;
            let vi_m = self.sum_vm * inv_tr;
            Some((vi_p, vi_m))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct ViBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ViBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ViBatchBuilder {
    range: ViBatchRange,
    kernel: Kernel,
}

impl ViBatchBuilder {
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
    ) -> Result<ViBatchOutput, ViError> {
        vi_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<ViBatchOutput, ViError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

#[derive(Clone, Debug)]
pub struct ViBatchOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
    pub combos: Vec<ViParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ViBatchOutput {
    pub fn row_for_params(&self, p: &ViParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn plus_for(&self, p: &ViParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.plus[start..start + self.cols]
        })
    }
    pub fn minus_for(&self, p: &ViParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.minus[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ViBatchRange) -> Vec<ViParams> {
    fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, ViError> {
        let (start, end, step) = range;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(ViError::InvalidRange { start, end, step });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut cur = start;
        loop {
            v.push(cur);
            if cur == end {
                break;
            }
            cur = cur
                .checked_sub(step)
                .ok_or(ViError::InvalidRange { start, end, step })?;
            if cur < end {
                break;
            }
        }
        if v.is_empty() {
            return Err(ViError::InvalidRange { start, end, step });
        }
        Ok(v)
    }

    let periods = match axis_usize(r.period) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ViParams { period: Some(p) });
    }
    out
}

pub fn vi_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    k: Kernel,
) -> Result<ViBatchOutput, ViError> {
    let kernel = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx2Batch,
            other => other,
        },
        other if other.is_batch() => other,
        other => {
            return Err(ViError::InvalidKernelForBatch(other));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vi_batch_par_slice(high, low, close, sweep, simd)
}

pub fn vi_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kern: Kernel,
) -> Result<ViBatchOutput, ViError> {
    vi_batch_inner(high, low, close, sweep, kern, false)
}
pub fn vi_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kern: Kernel,
) -> Result<ViBatchOutput, ViError> {
    vi_batch_inner(high, low, close, sweep, kern, true)
}
#[inline(always)]
fn vi_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ViBatchOutput, ViError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ViError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ViError::EmptyInputData);
    }
    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(ViError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ViError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();
    rows.checked_mul(cols)
        .ok_or_else(|| ViError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;

    let mut plus_mu = make_uninit_matrix(rows, cols);
    let mut minus_mu = make_uninit_matrix(rows, cols);

    let mut warm: Vec<usize> = Vec::with_capacity(combos.len());
    for c in &combos {
        let p = c.period.unwrap();
        let warm_i = first
            .checked_add(p)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| ViError::InvalidPeriod {
                period: p,
                data_len: high.len(),
            })?;
        warm.push(warm_i);
    }

    init_matrix_prefixes(&mut plus_mu, cols, &warm);
    init_matrix_prefixes(&mut minus_mu, cols, &warm);

    let mut plus_guard = core::mem::ManuallyDrop::new(plus_mu);
    let mut minus_guard = core::mem::ManuallyDrop::new(minus_mu);
    let plus: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(plus_guard.as_mut_ptr() as *mut f64, plus_guard.len())
    };
    let minus: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(minus_guard.as_mut_ptr() as *mut f64, minus_guard.len())
    };

    let mut pfx_tr = vec![0.0f64; cols];
    let mut pfx_vp = vec![0.0f64; cols];
    let mut pfx_vm = vec![0.0f64; cols];
    if cols > 0 && first < cols {
        unsafe {
            match kern {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => vi_prefix_avx512(
                    high,
                    low,
                    close,
                    first,
                    &mut pfx_tr,
                    &mut pfx_vp,
                    &mut pfx_vm,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => vi_prefix_avx2(
                    high,
                    low,
                    close,
                    first,
                    &mut pfx_tr,
                    &mut pfx_vp,
                    &mut pfx_vm,
                ),
                _ => vi_prefix_scalar(
                    high,
                    low,
                    close,
                    first,
                    &mut pfx_tr,
                    &mut pfx_vp,
                    &mut pfx_vm,
                ),
            }
        }
    }

    let do_row = |row: usize, plus_row: &mut [f64], minus_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let warm = first + period - 1;
        if warm >= cols {
            return;
        }
        let mut i = warm;
        while i < cols {
            let tr_sum = if i >= period {
                pfx_tr[i] - pfx_tr[i - period]
            } else {
                pfx_tr[i]
            };
            let vp_sum = if i >= period {
                pfx_vp[i] - pfx_vp[i - period]
            } else {
                pfx_vp[i]
            };
            let vm_sum = if i >= period {
                pfx_vm[i] - pfx_vm[i - period]
            } else {
                pfx_vm[i]
            };
            plus_row[i] = vp_sum / tr_sum;
            minus_row[i] = vm_sum / tr_sum;
            i += 1;
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            plus.par_chunks_mut(cols)
                .zip(minus.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (p, m))| do_row(row, p, m));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for ((row, p), m) in plus
                .chunks_mut(cols)
                .enumerate()
                .zip(minus.chunks_mut(cols))
            {
                do_row(row, p, m);
            }
        }
    } else {
        for ((row, p), m) in plus
            .chunks_mut(cols)
            .enumerate()
            .zip(minus.chunks_mut(cols))
        {
            do_row(row, p, m);
        }
    }

    let plus_vec = unsafe {
        Vec::from_raw_parts(
            plus_guard.as_mut_ptr() as *mut f64,
            plus_guard.len(),
            plus_guard.capacity(),
        )
    };
    let minus_vec = unsafe {
        Vec::from_raw_parts(
            minus_guard.as_mut_ptr() as *mut f64,
            minus_guard.len(),
            minus_guard.capacity(),
        )
    };

    Ok(ViBatchOutput {
        plus: plus_vec,
        minus: minus_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn vi_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    if period <= 32 {
        vi_row_avx512_short(high, low, close, first, period, plus, minus);
    } else {
        vi_row_avx512_long(high, low, close, first, period, plus, minus);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

fn vi_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_plus: &mut [f64],
    out_minus: &mut [f64],
) -> Result<Vec<ViParams>, ViError> {
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = close.len();
    if rows == 0 {
        return Err(ViError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| ViError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if out_plus.len() != expected || out_minus.len() != expected {
        let got = out_plus.len().max(out_minus.len());
        return Err(ViError::OutputLengthMismatch { expected, got });
    }

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ViError::EmptyInputData);
    }
    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(ViError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ViError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }

    let cols = close.len();
    let mut pfx_tr = vec![0.0f64; cols];
    let mut pfx_vp = vec![0.0f64; cols];
    let mut pfx_vm = vec![0.0f64; cols];
    if cols > 0 && first < cols {
        unsafe {
            match kernel {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => vi_prefix_avx512(
                    high,
                    low,
                    close,
                    first,
                    &mut pfx_tr,
                    &mut pfx_vp,
                    &mut pfx_vm,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => vi_prefix_avx2(
                    high,
                    low,
                    close,
                    first,
                    &mut pfx_tr,
                    &mut pfx_vp,
                    &mut pfx_vm,
                ),
                _ => vi_prefix_scalar(
                    high,
                    low,
                    close,
                    first,
                    &mut pfx_tr,
                    &mut pfx_vp,
                    &mut pfx_vm,
                ),
            }
        }
    }

    let do_row = |row: usize, p_row: &mut [f64], m_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let warm = first + period - 1;

        for i in 0..warm.min(cols) {
            p_row[i] = f64::NAN;
            m_row[i] = f64::NAN;
        }
        if warm >= cols {
            return;
        }
        let mut i = warm;
        while i < cols {
            let tr_sum = if i >= period {
                pfx_tr[i] - pfx_tr[i - period]
            } else {
                pfx_tr[i]
            };
            let vp_sum = if i >= period {
                pfx_vp[i] - pfx_vp[i - period]
            } else {
                pfx_vp[i]
            };
            let vm_sum = if i >= period {
                pfx_vm[i] - pfx_vm[i - period]
            } else {
                pfx_vm[i]
            };
            p_row[i] = vp_sum / tr_sum;
            m_row[i] = vm_sum / tr_sum;
            i += 1;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_plus
            .par_chunks_mut(cols)
            .zip(out_minus.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (p, m))| do_row(row, p, m));
        #[cfg(target_arch = "wasm32")]
        for ((row, p), m) in out_plus
            .chunks_mut(cols)
            .enumerate()
            .zip(out_minus.chunks_mut(cols))
        {
            do_row(row, p, m);
        }
    } else {
        for ((row, p), m) in out_plus
            .chunks_mut(cols)
            .enumerate()
            .zip(out_minus.chunks_mut(cols))
        {
            do_row(row, p, m);
        }
    }
    Ok(combos)
}

#[inline(always)]
unsafe fn vi_prefix_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    pfx_tr: &mut [f64],
    pfx_vp: &mut [f64],
    pfx_vm: &mut [f64],
) {
    let n = high.len();
    pfx_tr[first] = high[first] - low[first];
    pfx_vp[first] = 0.0;
    pfx_vm[first] = 0.0;
    let mut prev_h = high[first];
    let mut prev_l = low[first];
    let mut prev_c = close[first];
    let mut i = first + 1;
    while i < n {
        let hi = high[i];
        let lo = low[i];
        let hl = hi - lo;
        let hc = (hi - prev_c).abs();
        let lc = (lo - prev_c).abs();
        let tr_i = hl.max(hc.max(lc));
        let vp_i = (hi - prev_l).abs();
        let vm_i = (lo - prev_h).abs();
        pfx_tr[i] = pfx_tr[i - 1] + tr_i;
        pfx_vp[i] = pfx_vp[i - 1] + vp_i;
        pfx_vm[i] = pfx_vm[i - 1] + vm_i;
        prev_h = hi;
        prev_l = lo;
        prev_c = close[i];
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn abs256(x: __m256d) -> __m256d {
    let zero = _mm256_set1_pd(0.0);
    _mm256_max_pd(x, _mm256_sub_pd(zero, x))
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_prefix_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    pfx_tr: &mut [f64],
    pfx_vp: &mut [f64],
    pfx_vm: &mut [f64],
) {
    use core::arch::x86_64::*;
    let n = high.len();
    pfx_tr[first] = high[first] - low[first];
    pfx_vp[first] = 0.0;
    pfx_vm[first] = 0.0;
    let mut i = first + 1;

    let mut carry_tr = pfx_tr[i - 1];
    let mut carry_vp = pfx_vp[i - 1];
    let mut carry_vm = pfx_vm[i - 1];
    let step = 4;
    while i + step <= n {
        let v_hi = _mm256_loadu_pd(high.as_ptr().add(i));
        let v_lo = _mm256_loadu_pd(low.as_ptr().add(i));
        let v_cl_prev = _mm256_loadu_pd(close.as_ptr().add(i - 1));
        let v_lo_prev = _mm256_loadu_pd(low.as_ptr().add(i - 1));
        let v_hi_prev = _mm256_loadu_pd(high.as_ptr().add(i - 1));

        let hl = _mm256_sub_pd(v_hi, v_lo);
        let hc = abs256(_mm256_sub_pd(v_hi, v_cl_prev));
        let lc = abs256(_mm256_sub_pd(v_lo, v_cl_prev));
        let tr_v = _mm256_max_pd(hl, _mm256_max_pd(hc, lc));
        let vp_v = abs256(_mm256_sub_pd(v_hi, v_lo_prev));
        let vm_v = abs256(_mm256_sub_pd(v_lo, v_hi_prev));

        let mut tr_tmp = [0.0f64; 4];
        let mut vp_tmp = [0.0f64; 4];
        let mut vm_tmp = [0.0f64; 4];
        _mm256_storeu_pd(tr_tmp.as_mut_ptr(), tr_v);
        _mm256_storeu_pd(vp_tmp.as_mut_ptr(), vp_v);
        _mm256_storeu_pd(vm_tmp.as_mut_ptr(), vm_v);
        let mut k = 0;
        while k < step {
            carry_tr += tr_tmp[k];
            carry_vp += vp_tmp[k];
            carry_vm += vm_tmp[k];
            pfx_tr[i + k] = carry_tr;
            pfx_vp[i + k] = carry_vp;
            pfx_vm[i + k] = carry_vm;
            k += 1;
        }

        i += step;
    }
    while i < n {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let prev_c = *close.get_unchecked(i - 1);
        let prev_l = *low.get_unchecked(i - 1);
        let prev_h = *high.get_unchecked(i - 1);
        let hl = hi - lo;
        let hc = (hi - prev_c).abs();
        let lc = (lo - prev_c).abs();
        let tr_i = hl.max(hc.max(lc));
        let vp_i = (hi - prev_l).abs();
        let vm_i = (lo - prev_h).abs();
        carry_tr += tr_i;
        carry_vp += vp_i;
        carry_vm += vm_i;
        pfx_tr[i] = carry_tr;
        pfx_vp[i] = carry_vp;
        pfx_vm[i] = carry_vm;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn abs512(x: __m512d) -> __m512d {
    let zero = _mm512_set1_pd(0.0);
    _mm512_max_pd(x, _mm512_sub_pd(zero, x))
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_prefix_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    pfx_tr: &mut [f64],
    pfx_vp: &mut [f64],
    pfx_vm: &mut [f64],
) {
    use core::arch::x86_64::*;
    let n = high.len();
    pfx_tr[first] = high[first] - low[first];
    pfx_vp[first] = 0.0;
    pfx_vm[first] = 0.0;
    let mut i = first + 1;
    let mut carry_tr = pfx_tr[i - 1];
    let mut carry_vp = pfx_vp[i - 1];
    let mut carry_vm = pfx_vm[i - 1];
    let step = 8;
    while i + step <= n {
        let v_hi = _mm512_loadu_pd(high.as_ptr().add(i));
        let v_lo = _mm512_loadu_pd(low.as_ptr().add(i));
        let v_cl_prev = _mm512_loadu_pd(close.as_ptr().add(i - 1));
        let v_lo_prev = _mm512_loadu_pd(low.as_ptr().add(i - 1));
        let v_hi_prev = _mm512_loadu_pd(high.as_ptr().add(i - 1));

        let hl = _mm512_sub_pd(v_hi, v_lo);
        let hc = abs512(_mm512_sub_pd(v_hi, v_cl_prev));
        let lc = abs512(_mm512_sub_pd(v_lo, v_cl_prev));
        let tr_v = _mm512_max_pd(hl, _mm512_max_pd(hc, lc));
        let vp_v = abs512(_mm512_sub_pd(v_hi, v_lo_prev));
        let vm_v = abs512(_mm512_sub_pd(v_lo, v_hi_prev));

        let mut tr_tmp = [0.0f64; 8];
        let mut vp_tmp = [0.0f64; 8];
        let mut vm_tmp = [0.0f64; 8];
        _mm512_storeu_pd(tr_tmp.as_mut_ptr(), tr_v);
        _mm512_storeu_pd(vp_tmp.as_mut_ptr(), vp_v);
        _mm512_storeu_pd(vm_tmp.as_mut_ptr(), vm_v);
        let mut k = 0;
        while k < step {
            carry_tr += tr_tmp[k];
            carry_vp += vp_tmp[k];
            carry_vm += vm_tmp[k];
            pfx_tr[i + k] = carry_tr;
            pfx_vp[i + k] = carry_vp;
            pfx_vm[i + k] = carry_vm;
            k += 1;
        }
        i += step;
    }
    while i < n {
        let hi = *high.get_unchecked(i);
        let lo = *low.get_unchecked(i);
        let prev_c = *close.get_unchecked(i - 1);
        let prev_l = *low.get_unchecked(i - 1);
        let prev_h = *high.get_unchecked(i - 1);
        let hl = hi - lo;
        let hc = (hi - prev_c).abs();
        let lc = (lo - prev_c).abs();
        let tr_i = hl.max(hc.max(lc));
        let vp_i = (hi - prev_l).abs();
        let vm_i = (lo - prev_h).abs();
        carry_tr += tr_i;
        carry_vp += vp_i;
        carry_vm += vm_i;
        pfx_tr[i] = carry_tr;
        pfx_vp[i] = carry_vp;
        pfx_vm[i] = carry_vm;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_vi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = ViParams { period: None };
        let input = ViInput::from_candles(&candles, default_params);
        let output = vi_with_kernel(&input, kernel)?;
        assert_eq!(output.plus.len(), candles.close.len());
        assert_eq!(output.minus.len(), candles.close.len());
        Ok(())
    }
    fn check_vi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ViInput::from_candles(&candles, ViParams::default());
        let result = vi_with_kernel(&input, kernel)?;
        let expected_last_five_plus = [
            0.9970238095238095,
            0.9871071716357775,
            0.9464453759945247,
            0.890897412369242,
            0.9206478557604156,
        ];
        let expected_last_five_minus = [
            1.0097117794486214,
            1.04174053182917,
            1.1152365471811105,
            1.181684712791338,
            1.1894672506875827,
        ];
        let n = result.plus.len();
        let plus_slice = &result.plus[n - 5..];
        let minus_slice = &result.minus[n - 5..];
        for (i, &val) in plus_slice.iter().enumerate() {
            let expected = expected_last_five_plus[i];
            assert!(
                (val - expected).abs() < 1e-8,
                "[{}] VI+ mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected
            );
        }
        for (i, &val) in minus_slice.iter().enumerate() {
            let expected = expected_last_five_minus[i];
            assert!(
                (val - expected).abs() < 1e-8,
                "[{}] VI- mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected
            );
        }
        Ok(())
    }
    fn check_vi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ViInput::with_default_candles(&candles);
        let output = vi_with_kernel(&input, kernel)?;
        assert_eq!(output.plus.len(), candles.close.len());
        assert_eq!(output.minus.len(), candles.close.len());
        Ok(())
    }
    fn check_vi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ViParams { period: Some(0) };
        let input = ViInput::from_slices(&input_data, &input_data, &input_data, params);
        let res = vi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VI should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_vi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ViParams { period: Some(10) };
        let input = ViInput::from_slices(&data_small, &data_small, &data_small, params);
        let res = vi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_vi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ViParams { period: Some(14) };
        let input = ViInput::from_slices(&single_point, &single_point, &single_point, params);
        let res = vi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VI should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_vi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = ViInput::from_candles(&candles, ViParams::default());
        let res = vi_with_kernel(&input, kernel)?;
        assert_eq!(res.plus.len(), candles.close.len());
        if res.plus.len() > 20 {
            for (i, &val) in res.plus[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_vi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            ViParams::default(),
            ViParams { period: Some(1) },
            ViParams { period: Some(2) },
            ViParams { period: Some(5) },
            ViParams { period: Some(7) },
            ViParams { period: Some(10) },
            ViParams { period: Some(20) },
            ViParams { period: Some(30) },
            ViParams { period: Some(50) },
            ViParams { period: Some(100) },
            ViParams { period: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = ViInput::from_candles(&candles, params.clone());
            let output = vi_with_kernel(&input, kernel)?;

            for (i, &val) in output.plus.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in plus array \
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
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in plus array \
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
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in plus array \
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

            for (i, &val) in output.minus.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in minus array \
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
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in minus array \
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
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in minus array \
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
    fn check_vi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=100).prop_flat_map(|period| {
            (period + 50..400).prop_flat_map(move |len| {
                (
                    prop::collection::vec(
                        (50.0f64..500.0f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    prop::collection::vec((0.001f64..0.05f64), len),
                    prop::collection::vec((0.0f64..1.0f64), len),
                    Just(period),
                )
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(base_prices, volatilities, close_positions, period)| {
                    let mut high = Vec::with_capacity(base_prices.len());
                    let mut low = Vec::with_capacity(base_prices.len());
                    let mut close = Vec::with_capacity(base_prices.len());

                    assert_eq!(base_prices.len(), volatilities.len());
                    assert_eq!(base_prices.len(), close_positions.len());

                    for i in 0..base_prices.len() {
                        let price = base_prices[i];
                        let vol = volatilities[i];
                        let close_pos = close_positions[i];

                        let range = price * vol;
                        let h = price + range * (0.3 + vol * 2.0);
                        let l = price - range * (0.3 + vol * 2.0);
                        let c = l + (h - l) * close_pos;

                        high.push(h);
                        low.push(l);
                        close.push(c);
                    }

                    let params = ViParams {
                        period: Some(period),
                    };
                    let input = ViInput::from_slices(&high, &low, &close, params.clone());

                    let ViOutput {
                        plus: out_plus,
                        minus: out_minus,
                    } = vi_with_kernel(&input, kernel).unwrap();
                    let ViOutput {
                        plus: ref_plus,
                        minus: ref_minus,
                    } = vi_with_kernel(&input, Kernel::Scalar).unwrap();

                    for i in 0..out_plus.len() {
                        if !out_plus[i].is_nan() {
                            prop_assert!(
                                out_plus[i] >= -1e-9,
                                "[{}] VI+ negative at idx {}: {}",
                                test_name,
                                i,
                                out_plus[i]
                            );
                        }
                        if !out_minus[i].is_nan() {
                            prop_assert!(
                                out_minus[i] >= -1e-9,
                                "[{}] VI- negative at idx {}: {}",
                                test_name,
                                i,
                                out_minus[i]
                            );
                        }
                    }

                    let first_valid = (0..high.len())
                        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
                        .unwrap_or(0);
                    let warmup_end = first_valid + period - 1;

                    for i in 0..warmup_end.min(out_plus.len()) {
                        prop_assert!(
                            out_plus[i].is_nan(),
                            "[{}] Expected NaN during warmup at idx {}, got {}",
                            test_name,
                            i,
                            out_plus[i]
                        );
                        prop_assert!(
                            out_minus[i].is_nan(),
                            "[{}] Expected NaN during warmup at idx {}, got {}",
                            test_name,
                            i,
                            out_minus[i]
                        );
                    }

                    for i in warmup_end..out_plus.len() {
                        let plus_bits = out_plus[i].to_bits();
                        let ref_plus_bits = ref_plus[i].to_bits();
                        let minus_bits = out_minus[i].to_bits();
                        let ref_minus_bits = ref_minus[i].to_bits();

                        if !out_plus[i].is_finite() || !ref_plus[i].is_finite() {
                            prop_assert!(
                                plus_bits == ref_plus_bits,
                                "[{}] VI+ finite/NaN mismatch at idx {}: {} vs {}",
                                test_name,
                                i,
                                out_plus[i],
                                ref_plus[i]
                            );
                        } else {
                            let ulp_diff = plus_bits.abs_diff(ref_plus_bits);
                            prop_assert!(
                                (out_plus[i] - ref_plus[i]).abs() <= 1e-9 || ulp_diff <= 4,
                                "[{}] VI+ mismatch at idx {}: {} vs {} (ULP={})",
                                test_name,
                                i,
                                out_plus[i],
                                ref_plus[i],
                                ulp_diff
                            );
                        }

                        if !out_minus[i].is_finite() || !ref_minus[i].is_finite() {
                            prop_assert!(
                                minus_bits == ref_minus_bits,
                                "[{}] VI- finite/NaN mismatch at idx {}: {} vs {}",
                                test_name,
                                i,
                                out_minus[i],
                                ref_minus[i]
                            );
                        } else {
                            let ulp_diff = minus_bits.abs_diff(ref_minus_bits);
                            prop_assert!(
                                (out_minus[i] - ref_minus[i]).abs() <= 1e-9 || ulp_diff <= 4,
                                "[{}] VI- mismatch at idx {}: {} vs {} (ULP={})",
                                test_name,
                                i,
                                out_minus[i],
                                ref_minus[i],
                                ulp_diff
                            );
                        }
                    }

                    if period == 1 {
                        if warmup_end < out_plus.len() {
                            prop_assert!(
                                out_plus[warmup_end].is_finite(),
                                "[{}] VI+ should be finite for period=1 at idx {}",
                                test_name,
                                warmup_end
                            );
                            prop_assert!(
                                out_minus[warmup_end].is_finite(),
                                "[{}] VI- should be finite for period=1 at idx {}",
                                test_name,
                                warmup_end
                            );
                        }
                    }

                    if period <= 5 && warmup_end + 5 < high.len() && warmup_end >= period {
                        let idx = warmup_end;

                        let mut tr_sum = 0.0;
                        let mut vp_sum = 0.0;
                        let mut vm_sum = 0.0;

                        let first_idx = idx + 1 - period;
                        tr_sum += high[first_idx] - low[first_idx];

                        for j in (first_idx + 1)..=idx {
                            let tr = (high[j] - low[j])
                                .max((high[j] - close[j - 1]).abs())
                                .max((low[j] - close[j - 1]).abs());
                            let vp = (high[j] - low[j - 1]).abs();
                            let vm = (low[j] - high[j - 1]).abs();

                            tr_sum += tr;
                            vp_sum += vp;
                            vm_sum += vm;
                        }

                        if tr_sum > 1e-10 {
                            let expected_plus = vp_sum / tr_sum;
                            let expected_minus = vm_sum / tr_sum;

                            prop_assert!(
                                (out_plus[idx] - expected_plus).abs() < 1e-6,
                                "[{}] VI+ formula verification failed at idx {}: {} vs {}",
                                test_name,
                                idx,
                                out_plus[idx],
                                expected_plus
                            );
                            prop_assert!(
                                (out_minus[idx] - expected_minus).abs() < 1e-6,
                                "[{}] VI- formula verification failed at idx {}: {} vs {}",
                                test_name,
                                idx,
                                out_minus[idx],
                                expected_minus
                            );
                        }
                    }

                    #[cfg(debug_assertions)]
                    {
                        for i in 0..out_plus.len() {
                            if !out_plus[i].is_nan() {
                                let bits = out_plus[i].to_bits();
                                prop_assert!(
                                    bits != 0x11111111_11111111
                                        && bits != 0x22222222_22222222
                                        && bits != 0x33333333_33333333,
                                    "[{}] Found poison value in VI+ at idx {}: 0x{:016X}",
                                    test_name,
                                    i,
                                    bits
                                );
                            }
                            if !out_minus[i].is_nan() {
                                let bits = out_minus[i].to_bits();
                                prop_assert!(
                                    bits != 0x11111111_11111111
                                        && bits != 0x22222222_22222222
                                        && bits != 0x33333333_33333333,
                                    "[{}] Found poison value in VI- at idx {}: 0x{:016X}",
                                    test_name,
                                    i,
                                    bits
                                );
                            }
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_vi_tests {
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
    generate_all_vi_tests!(
        check_vi_partial_params,
        check_vi_accuracy,
        check_vi_default_candles,
        check_vi_zero_period,
        check_vi_period_exceeds_length,
        check_vi_very_small_dataset,
        check_vi_nan_handling,
        check_vi_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vi_tests!(check_vi_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = ViBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = ViParams::default();
        let row = output.plus_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (5, 25, 5),
            (20, 50, 10),
            (2, 5, 1),
            (14, 14, 0),
            (30, 60, 15),
            (50, 100, 25),
            (100, 200, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = ViBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.plus.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in plus array with params: period={}",
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
						 at row {} col {} (flat index {}) in plus array with params: period={}",
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
						 at row {} col {} (flat index {}) in plus array with params: period={}",
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

            for (idx, &val) in output.minus.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in minus array with params: period={}",
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
						 at row {} col {} (flat index {}) in minus array with params: period={}",
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
						 at row {} col {} (flat index {}) in minus array with params: period={}",
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
    gen_batch_tests!(check_batch_no_poison);
}
