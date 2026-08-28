use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
#[cfg(target_arch = "wasm32")]
use crate::utilities::helpers::detect_wasm_kernel;
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
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DonchianData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[inline(always)]
fn donchian_slices<'a>(data: &'a DonchianData<'a>) -> (&'a [f64], &'a [f64]) {
    match data {
        DonchianData::Candles { candles } => (candles.high.as_slice(), candles.low.as_slice()),
        DonchianData::Slices { high, low } => (high, low),
    }
}

#[derive(Debug, Clone)]
pub struct DonchianOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DonchianParams {
    pub period: Option<usize>,
}

impl Default for DonchianParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct DonchianInput<'a> {
    pub data: DonchianData<'a>,
    pub params: DonchianParams,
}

impl<'a> DonchianInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DonchianParams) -> Self {
        Self {
            data: DonchianData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DonchianParams) -> Self {
        Self {
            data: DonchianData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DonchianParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DonchianBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DonchianBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DonchianBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<DonchianOutput, DonchianError> {
        let p = DonchianParams {
            period: self.period,
        };
        let i = DonchianInput::from_candles(c, p);
        donchian_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DonchianOutput, DonchianError> {
        let p = DonchianParams {
            period: self.period,
        };
        let i = DonchianInput::from_slices(high, low, p);
        donchian_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DonchianStream, DonchianError> {
        let p = DonchianParams {
            period: self.period,
        };
        DonchianStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DonchianError {
    #[error("donchian: Empty data provided.")]
    EmptyInputData,
    #[error("donchian: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("donchian: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("donchian: All values are NaN.")]
    AllValuesNaN,
    #[error("donchian: High/Low data slices have different lengths.")]
    MismatchedLength,
    #[error("donchian: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("donchian: invalid range expansion: start={start} end={end} step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("donchian: invalid input: {0}")]
    InvalidInput(String),
    #[error("donchian: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn donchian(input: &DonchianInput) -> Result<DonchianOutput, DonchianError> {
    donchian_with_kernel(input, Kernel::Auto)
}

pub fn donchian_with_kernel(
    input: &DonchianInput,
    kernel: Kernel,
) -> Result<DonchianOutput, DonchianError> {
    let (high, low) = donchian_slices(&input.data);

    if high.is_empty() || low.is_empty() {
        return Err(DonchianError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(DonchianError::MismatchedLength);
    }

    let first_valid_high = high.iter().position(|&x| !x.is_nan());
    let first_valid_low = low.iter().position(|&x| !x.is_nan());
    let first_valid_idx = match (first_valid_high, first_valid_low) {
        (Some(h), Some(l)) => h.max(l),
        _ => return Err(DonchianError::AllValuesNaN),
    };

    let len = high.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(DonchianError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first_valid_idx) < period {
        return Err(DonchianError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    let warmup_period = first_valid_idx + period - 1;
    let mut upperband = alloc_with_nan_prefix(len, warmup_period);
    let mut middleband = alloc_with_nan_prefix(len, warmup_period);
    let mut lowerband = alloc_with_nan_prefix(len, warmup_period);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => donchian_scalar(
                high,
                low,
                period,
                first_valid_idx,
                &mut upperband,
                &mut middleband,
                &mut lowerband,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => donchian_avx2(
                high,
                low,
                period,
                first_valid_idx,
                &mut upperband,
                &mut middleband,
                &mut lowerband,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch => donchian_scalar(
                high,
                low,
                period,
                first_valid_idx,
                &mut upperband,
                &mut middleband,
                &mut lowerband,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => donchian_avx512(
                high,
                low,
                period,
                first_valid_idx,
                &mut upperband,
                &mut middleband,
                &mut lowerband,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx512 | Kernel::Avx512Batch => donchian_scalar(
                high,
                low,
                period,
                first_valid_idx,
                &mut upperband,
                &mut middleband,
                &mut lowerband,
            ),
            _ => unreachable!(),
        }
    }

    Ok(DonchianOutput {
        upperband,
        middleband,
        lowerband,
    })
}

#[inline]
pub fn donchian_upper_with_kernel(
    input: &DonchianInput,
    kernel: Kernel,
) -> Result<Vec<f64>, DonchianError> {
    donchian_selected_with_kernel::<0>(input, kernel)
}

#[inline]
pub fn donchian_middle_with_kernel(
    input: &DonchianInput,
    kernel: Kernel,
) -> Result<Vec<f64>, DonchianError> {
    donchian_selected_with_kernel::<1>(input, kernel)
}

#[inline]
pub fn donchian_lower_with_kernel(
    input: &DonchianInput,
    kernel: Kernel,
) -> Result<Vec<f64>, DonchianError> {
    donchian_selected_with_kernel::<2>(input, kernel)
}

#[inline]
fn donchian_selected_with_kernel<const OUT: u8>(
    input: &DonchianInput,
    kernel: Kernel,
) -> Result<Vec<f64>, DonchianError> {
    let (high, low) = donchian_slices(&input.data);

    if high.is_empty() || low.is_empty() {
        return Err(DonchianError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(DonchianError::MismatchedLength);
    }

    let first_valid_high = high.iter().position(|&x| !x.is_nan());
    let first_valid_low = low.iter().position(|&x| !x.is_nan());
    let first_valid_idx = match (first_valid_high, first_valid_low) {
        (Some(h), Some(l)) => h.max(l),
        _ => return Err(DonchianError::AllValuesNaN),
    };

    let len = high.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(DonchianError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first_valid_idx) < period {
        return Err(DonchianError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let warmup_period = first_valid_idx + period - 1;
    let mut out = alloc_with_nan_prefix(len, warmup_period);

    if period <= 32 {
        donchian_selected_short::<OUT>(high, low, period, first_valid_idx, &mut out);
        return Ok(out);
    }

    let full = donchian_with_kernel(input, kernel)?;
    Ok(match OUT {
        0 => full.upperband,
        1 => full.middleband,
        2 => full.lowerband,
        _ => unreachable!(),
    })
}

#[inline(always)]
fn donchian_selected_short<const OUT: u8>(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let n = high.len();
    if n == 0 || period == 0 {
        return;
    }

    let warmup = first_valid + period - 1;
    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let op = out.as_mut_ptr();
        if period == 1 {
            for i in warmup..n {
                let h = *hp.add(i);
                let l = *lp.add(i);
                if h.is_nan() || l.is_nan() {
                    *op.add(i) = f64::NAN;
                } else {
                    *op.add(i) = match OUT {
                        0 => h,
                        1 => (h - l).mul_add(0.5, l),
                        2 => l,
                        _ => unreachable!(),
                    };
                }
            }
            return;
        }

        for i in warmup..n {
            let start = i + 1 - period;
            let mut maxv = f64::NEG_INFINITY;
            let mut minv = f64::INFINITY;
            let mut has_nan = false;
            for k in 0..period {
                let h = *hp.add(start + k);
                let l = *lp.add(start + k);
                if h.is_nan() || l.is_nan() {
                    has_nan = true;
                    break;
                }
                if OUT != 2 && h > maxv {
                    maxv = h;
                }
                if OUT != 0 && l < minv {
                    minv = l;
                }
            }
            if has_nan {
                *op.add(i) = f64::NAN;
            } else {
                *op.add(i) = match OUT {
                    0 => maxv,
                    1 => (maxv - minv).mul_add(0.5, minv),
                    2 => minv,
                    _ => unreachable!(),
                };
            }
        }
    }
}

#[inline]
pub fn donchian_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    let n = high.len();
    if n == 0 || period == 0 {
        return;
    }
    debug_assert_eq!(low.len(), n);
    debug_assert_eq!(upper.len(), n);
    debug_assert_eq!(middle.len(), n);
    debug_assert_eq!(lower.len(), n);

    let warmup = first_valid + period - 1;

    if period == 1 {
        let start = warmup;
        unsafe {
            let hp = high.as_ptr();
            let lp = low.as_ptr();
            let up = upper.as_mut_ptr();
            let mp = middle.as_mut_ptr();
            let lw = lower.as_mut_ptr();
            for i in start..n {
                let h = *hp.add(i);
                let l = *lp.add(i);
                if h.is_nan() || l.is_nan() {
                    *up.add(i) = f64::NAN;
                    *lw.add(i) = f64::NAN;
                    *mp.add(i) = f64::NAN;
                } else {
                    *up.add(i) = h;
                    *lw.add(i) = l;
                    *mp.add(i) = (h - l).mul_add(0.5, l);
                }
            }
        }
        return;
    }

    if period <= 32 {
        unsafe {
            let hp = high.as_ptr();
            let lp = low.as_ptr();
            let up = upper.as_mut_ptr();
            let mp = middle.as_mut_ptr();
            let lw = lower.as_mut_ptr();
            for i in warmup..n {
                let start = i + 1 - period;
                let mut maxv = f64::NEG_INFINITY;
                let mut minv = f64::INFINITY;
                let mut has_nan = false;
                for k in 0..period {
                    let h = *hp.add(start + k);
                    let l = *lp.add(start + k);
                    if h.is_nan() || l.is_nan() {
                        has_nan = true;
                        break;
                    }
                    if h > maxv {
                        maxv = h;
                    }
                    if l < minv {
                        minv = l;
                    }
                }
                if has_nan {
                    *up.add(i) = f64::NAN;
                    *lw.add(i) = f64::NAN;
                    *mp.add(i) = f64::NAN;
                } else {
                    *up.add(i) = maxv;
                    *lw.add(i) = minv;
                    *mp.add(i) = (maxv - minv).mul_add(0.5, minv);
                }
            }
        }
        return;
    }

    let mut g_max = AVec::<f64>::with_capacity(CACHELINE_ALIGN, n);
    let mut g_min = AVec::<f64>::with_capacity(CACHELINE_ALIGN, n);
    let mut valid: Vec<u8> = Vec::with_capacity(n);
    unsafe {
        g_max.set_len(n);
        g_min.set_len(n);
        valid.set_len(n);
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let gp_max = g_max.as_mut_ptr();
        let gp_min = g_min.as_mut_ptr();
        let vp = valid.as_mut_ptr();

        let mut acc_max = f64::NEG_INFINITY;
        let mut acc_min = f64::INFINITY;
        let mut k: usize = 0;

        for i in 0..n {
            let h = *hp.add(i);
            let l = *lp.add(i);
            let ok = h.is_finite() & l.is_finite();
            *vp.add(i) = ok as u8;
            let hv = if ok { h } else { f64::NEG_INFINITY };
            let lv = if ok { l } else { f64::INFINITY };
            if k == 0 {
                acc_max = hv;
                acc_min = lv;
            } else {
                if hv > acc_max {
                    acc_max = hv;
                }
                if lv < acc_min {
                    acc_min = lv;
                }
            }
            *gp_max.add(i) = acc_max;
            *gp_min.add(i) = acc_min;
            k += 1;
            if k == period {
                k = 0;
            }
        }
    }

    let mut ps: Vec<u32> = Vec::with_capacity(n + 1);
    unsafe {
        ps.set_len(n + 1);
        let psp = ps.as_mut_ptr();
        let vp = valid.as_ptr();
        *psp.add(0) = 0;
        for i in 0..n {
            let prev = *psp.add(i);
            let add = *vp.add(i) as u32;
            *psp.add(i + 1) = prev + add;
        }
    }

    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let up = upper.as_mut_ptr();
        let mp = middle.as_mut_ptr();
        let lw = lower.as_mut_ptr();
        let gp_max = g_max.as_ptr();
        let gp_min = g_min.as_ptr();
        let psp = ps.as_ptr();

        let mut acc_max = f64::NEG_INFINITY;
        let mut acc_min = f64::INFINITY;

        for j in (0..n).rev() {
            let h = *hp.add(j);
            let l = *lp.add(j);
            let ok = h.is_finite() & l.is_finite();
            let hv = if ok { h } else { f64::NEG_INFINITY };
            let lv = if ok { l } else { f64::INFINITY };

            if j == n - 1 || ((j + 1) % period) == 0 {
                acc_max = hv;
                acc_min = lv;
            } else {
                if hv > acc_max {
                    acc_max = hv;
                }
                if lv < acc_min {
                    acc_min = lv;
                }
            }

            let i = j + period - 1;
            if i >= n || i < warmup {
                continue;
            }

            let all_valid = {
                let vcnt = *psp.add(i + 1) - *psp.add(i + 1 - period);
                vcnt == period as u32
            };
            if all_valid {
                let gm = *gp_max.add(i);
                let gn = *gp_min.add(i);
                let maxv = if acc_max > gm { acc_max } else { gm };
                let minv = if acc_min < gn { acc_min } else { gn };
                *up.add(i) = maxv;
                *lw.add(i) = minv;
                *mp.add(i) = (maxv - minv).mul_add(0.5, minv);
            } else {
                *up.add(i) = f64::NAN;
                *lw.add(i) = f64::NAN;
                *mp.add(i) = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn donchian_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    donchian_avx2(high, low, period, first_valid, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn donchian_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    donchian_scalar(high, low, period, first_valid, upper, middle, lower)
}

pub fn donchian_into_slice(
    upper_dst: &mut [f64],
    middle_dst: &mut [f64],
    lower_dst: &mut [f64],
    input: &DonchianInput,
    kern: Kernel,
) -> Result<(), DonchianError> {
    let (high, low) = donchian_slices(&input.data);

    if high.is_empty() || low.is_empty() {
        return Err(DonchianError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(DonchianError::MismatchedLength);
    }
    if upper_dst.len() != high.len()
        || middle_dst.len() != high.len()
        || lower_dst.len() != high.len()
    {
        return Err(DonchianError::OutputLengthMismatch {
            expected: high.len(),
            got: upper_dst.len().max(middle_dst.len()).max(lower_dst.len()),
        });
    }

    let first_valid_high = high.iter().position(|&x| !x.is_nan());
    let first_valid_low = low.iter().position(|&x| !x.is_nan());
    let first_valid_idx = match (first_valid_high, first_valid_low) {
        (Some(h), Some(l)) => h.max(l),
        _ => return Err(DonchianError::AllValuesNaN),
    };

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(DonchianError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }
    if (high.len() - first_valid_idx) < period {
        return Err(DonchianError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    let chosen = match kern {
        #[cfg(target_arch = "wasm32")]
        Kernel::Auto => Kernel::Scalar,
        #[cfg(not(target_arch = "wasm32"))]
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    let warmup_period = first_valid_idx + period - 1;

    for i in 0..warmup_period {
        upper_dst[i] = f64::NAN;
        middle_dst[i] = f64::NAN;
        lower_dst[i] = f64::NAN;
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => donchian_scalar(
                high,
                low,
                period,
                first_valid_idx,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => donchian_avx2(
                high,
                low,
                period,
                first_valid_idx,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch => donchian_scalar(
                high,
                low,
                period,
                first_valid_idx,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => donchian_avx512(
                high,
                low,
                period,
                first_valid_idx,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx512 | Kernel::Avx512Batch => donchian_scalar(
                high,
                low,
                period,
                first_valid_idx,
                upper_dst,
                middle_dst,
                lower_dst,
            ),
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn donchian_into(
    input: &DonchianInput,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) -> Result<(), DonchianError> {
    donchian_into_slice(upper, middle, lower, input, Kernel::Auto)
}

#[inline(always)]
pub fn donchian_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianBatchRange,
    k: Kernel,
) -> Result<DonchianBatchOutput, DonchianError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DonchianError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    donchian_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DonchianBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DonchianBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 269, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DonchianBatchBuilder {
    range: DonchianBatchRange,
    kernel: Kernel,
}

impl DonchianBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<DonchianBatchOutput, DonchianError> {
        donchian_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<DonchianBatchOutput, DonchianError> {
        DonchianBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<DonchianBatchOutput, DonchianError> {
        self.apply_slices(&c.high, &c.low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DonchianBatchOutput, DonchianError> {
        DonchianBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct DonchianBatchOutput {
    pub upper: Vec<f64>,
    pub middle: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<DonchianParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DonchianBatchOutput {
    pub fn row_for_params(&self, p: &DonchianParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
    }
    pub fn upper_for(&self, p: &DonchianParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.upper[row * self.cols..][..self.cols])
    }
    pub fn middle_for(&self, p: &DonchianParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.middle[row * self.cols..][..self.cols])
    }
    pub fn lower_for(&self, p: &DonchianParams) -> Option<&[f64]> {
        self.row_for_params(p)
            .map(|row| &self.lower[row * self.cols..][..self.cols])
    }
}

#[inline(always)]
pub fn expand_grid(r: &DonchianBatchRange) -> Result<Vec<DonchianParams>, DonchianError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, DonchianError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            Ok((start..=end).step_by(step).collect())
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if let Some(next) = cur.checked_sub(step) {
                    cur = next;
                } else {
                    break;
                }
                if cur == usize::MAX {
                    break;
                }
            }
            if v.is_empty() {
                return Err(DonchianError::InvalidRange { start, end, step });
            }
            Ok(v)
        }
    }
    let periods = axis_usize(r.period)?;
    Ok(periods
        .into_iter()
        .map(|p| DonchianParams { period: Some(p) })
        .collect())
}

#[inline(always)]
pub fn donchian_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianBatchRange,
    kern: Kernel,
) -> Result<DonchianBatchOutput, DonchianError> {
    donchian_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn donchian_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianBatchRange,
    kern: Kernel,
) -> Result<DonchianBatchOutput, DonchianError> {
    donchian_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn donchian_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DonchianBatchOutput, DonchianError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(DonchianError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    if high.len() != low.len() {
        return Err(DonchianError::MismatchedLength);
    }
    let first = high
        .iter()
        .position(|x| !x.is_nan())
        .zip(low.iter().position(|x| !x.is_nan()))
        .map(|(a, b)| a.max(b));
    let first = match first {
        Some(idx) => idx,
        None => return Err(DonchianError::AllValuesNaN),
    };
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(DonchianError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();
    let _size = rows
        .checked_mul(cols)
        .ok_or_else(|| DonchianError::InvalidInput("rows*cols overflow".into()))?;

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut middle_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut upper_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut middle_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut lower_mu, cols, &warmup_periods);

    let mut upper_guard = core::mem::ManuallyDrop::new(upper_mu);
    let mut middle_guard = core::mem::ManuallyDrop::new(middle_mu);
    let mut lower_guard = core::mem::ManuallyDrop::new(lower_mu);

    let upper: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let middle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(middle_guard.as_mut_ptr() as *mut f64, middle_guard.len())
    };
    let lower: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };

    let do_row =
        |row: usize, out_upper: &mut [f64], out_middle: &mut [f64], out_lower: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match kern {
                Kernel::Scalar => {
                    donchian_row_scalar(high, low, first, period, out_upper, out_middle, out_lower)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => {
                    donchian_row_avx2(high, low, first, period, out_upper, out_middle, out_lower)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => {
                    donchian_row_avx512(high, low, first, period, out_upper, out_middle, out_lower)
                }
                _ => unreachable!(),
            }
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            upper
                .par_chunks_mut(cols)
                .zip(middle.par_chunks_mut(cols))
                .zip(lower.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((upper, middle), lower))| do_row(row, upper, middle, lower));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (((upper, middle), lower), row) in upper
                .chunks_mut(cols)
                .zip(middle.chunks_mut(cols))
                .zip(lower.chunks_mut(cols))
                .zip(0..)
            {
                do_row(row, upper, middle, lower);
            }
        }
    } else {
        for (((upper, middle), lower), row) in upper
            .chunks_mut(cols)
            .zip(middle.chunks_mut(cols))
            .zip(lower.chunks_mut(cols))
            .zip(0..)
        {
            do_row(row, upper, middle, lower);
        }
    }

    let upper = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            upper_guard.len(),
            upper_guard.capacity(),
        )
    };
    let middle = unsafe {
        Vec::from_raw_parts(
            middle_guard.as_mut_ptr() as *mut f64,
            middle_guard.len(),
            middle_guard.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            lower_guard.len(),
            lower_guard.capacity(),
        )
    };

    Ok(DonchianBatchOutput {
        upper,
        middle,
        lower,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn donchian_row_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    let n = high.len();
    if n == 0 || period == 0 {
        return;
    }
    let warmup = first + period - 1;

    if period == 1 {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let up = upper.as_mut_ptr();
        let mp = middle.as_mut_ptr();
        let lw = lower.as_mut_ptr();
        for i in warmup..n {
            let h = *hp.add(i);
            let l = *lp.add(i);
            if h.is_nan() || l.is_nan() {
                *up.add(i) = f64::NAN;
                *lw.add(i) = f64::NAN;
                *mp.add(i) = f64::NAN;
            } else {
                *up.add(i) = h;
                *lw.add(i) = l;
                *mp.add(i) = (h - l).mul_add(0.5, l);
            }
        }
        return;
    }

    if period <= 32 {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let up = upper.as_mut_ptr();
        let mp = middle.as_mut_ptr();
        let lw = lower.as_mut_ptr();
        for i in warmup..n {
            let start = i + 1 - period;
            let mut maxv = f64::NEG_INFINITY;
            let mut minv = f64::INFINITY;
            let mut has_nan = false;
            for k in 0..period {
                let h = *hp.add(start + k);
                let l = *lp.add(start + k);
                if h.is_nan() || l.is_nan() {
                    has_nan = true;
                    break;
                }
                if h > maxv {
                    maxv = h;
                }
                if l < minv {
                    minv = l;
                }
            }
            if has_nan {
                *up.add(i) = f64::NAN;
                *lw.add(i) = f64::NAN;
                *mp.add(i) = f64::NAN;
            } else {
                *up.add(i) = maxv;
                *lw.add(i) = minv;
                *mp.add(i) = (maxv - minv).mul_add(0.5, minv);
            }
        }
        return;
    }

    let mut g_max = AVec::<f64>::with_capacity(CACHELINE_ALIGN, n);
    let mut g_min = AVec::<f64>::with_capacity(CACHELINE_ALIGN, n);
    let mut valid: Vec<u8> = Vec::with_capacity(n);
    g_max.set_len(n);
    g_min.set_len(n);
    valid.set_len(n);
    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let gp_max = g_max.as_mut_ptr();
    let gp_min = g_min.as_mut_ptr();
    let vp = valid.as_mut_ptr();

    let mut acc_max = f64::NEG_INFINITY;
    let mut acc_min = f64::INFINITY;
    let mut k: usize = 0;
    for i in 0..n {
        let h = *hp.add(i);
        let l = *lp.add(i);
        let ok = h.is_finite() & l.is_finite();
        *vp.add(i) = ok as u8;
        let hv = if ok { h } else { f64::NEG_INFINITY };
        let lv = if ok { l } else { f64::INFINITY };
        if k == 0 {
            acc_max = hv;
            acc_min = lv;
        } else {
            if hv > acc_max {
                acc_max = hv;
            }
            if lv < acc_min {
                acc_min = lv;
            }
        }
        *gp_max.add(i) = acc_max;
        *gp_min.add(i) = acc_min;
        k += 1;
        if k == period {
            k = 0;
        }
    }

    let up = upper.as_mut_ptr();
    let mp = middle.as_mut_ptr();
    let lw = lower.as_mut_ptr();
    let gp_max = g_max.as_ptr();
    let gp_min = g_min.as_ptr();
    let vp = valid.as_ptr();

    acc_max = f64::NEG_INFINITY;
    acc_min = f64::INFINITY;
    let mut have_vcnt = false;
    let mut vcnt: u32 = 0;
    for j in (0..n).rev() {
        let h = *hp.add(j);
        let l = *lp.add(j);
        let ok = h.is_finite() & l.is_finite();
        let hv = if ok { h } else { f64::NEG_INFINITY };
        let lv = if ok { l } else { f64::INFINITY };

        if j == n - 1 || ((j + 1) % period) == 0 {
            acc_max = hv;
            acc_min = lv;
        } else {
            if hv > acc_max {
                acc_max = hv;
            }
            if lv < acc_min {
                acc_min = lv;
            }
        }

        let i = j + period - 1;
        if i < n {
            if !have_vcnt {
                let start = i + 1 - period;
                let mut sum: u32 = 0;
                for t in start..=i {
                    sum += *vp.add(t) as u32;
                }
                vcnt = sum;
                have_vcnt = true;
            } else {
                if i + 1 < n {
                    vcnt = vcnt - (*vp.add(i + 1) as u32) + (*vp.add(i + 1 - period) as u32);
                }
            }
        }

        if i >= n || i < warmup {
            continue;
        }

        let all_valid = vcnt == period as u32;
        if all_valid {
            let gm = *gp_max.add(i);
            let gn = *gp_min.add(i);
            let maxv = if acc_max > gm { acc_max } else { gm };
            let minv = if acc_min < gn { acc_min } else { gn };
            *up.add(i) = maxv;
            *lw.add(i) = minv;
            *mp.add(i) = (maxv - minv).mul_add(0.5, minv);
        } else {
            *up.add(i) = f64::NAN;
            *lw.add(i) = f64::NAN;
            *mp.add(i) = f64::NAN;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn donchian_row_avx2(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    donchian_row_scalar(high, low, first, period, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn donchian_row_avx512(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    if period <= 32 {
        donchian_row_avx512_short(high, low, first, period, upper, middle, lower)
    } else {
        donchian_row_avx512_long(high, low, first, period, upper, middle, lower)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn donchian_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    donchian_row_scalar(high, low, first, period, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn donchian_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    donchian_row_scalar(high, low, first, period, upper, middle, lower)
}

#[derive(Debug, Clone)]
pub struct DonchianStream {
    period: usize,

    valid_ring: Vec<u8>,
    head: usize,
    seen: usize,
    valid_count: usize,

    max_deque: VecDeque<(f64, usize)>,
    min_deque: VecDeque<(f64, usize)>,
}

impl DonchianStream {
    pub fn try_new(params: DonchianParams) -> Result<Self, DonchianError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(DonchianError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            valid_ring: vec![0; period],
            head: 0,
            seen: 0,
            valid_count: 0,
            max_deque: VecDeque::with_capacity(period),
            min_deque: VecDeque::with_capacity(period),
        })
    }

    #[inline(always)]
    fn evict_outdated(&mut self, window_start: usize) {
        while let Some(&(_, idx)) = self.max_deque.front() {
            if idx < window_start {
                self.max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(_, idx)) = self.min_deque.front() {
            if idx < window_start {
                self.min_deque.pop_front();
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64, f64)> {
        let ok = high.is_finite() & low.is_finite();

        let leaving = self.valid_ring[self.head] as usize;
        self.valid_ring[self.head] = ok as u8;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        self.valid_count = self.valid_count + (ok as usize) - leaving;

        let t = self.seen;
        self.seen = t + 1;
        let window_start = self.seen.saturating_sub(self.period);

        self.evict_outdated(window_start);

        if ok {
            while let Some(&(v, _)) = self.max_deque.back() {
                if v <= high {
                    self.max_deque.pop_back();
                } else {
                    break;
                }
            }
            self.max_deque.push_back((high, t));

            while let Some(&(v, _)) = self.min_deque.back() {
                if v >= low {
                    self.min_deque.pop_back();
                } else {
                    break;
                }
            }
            self.min_deque.push_back((low, t));
        }

        if self.seen < self.period {
            return None;
        }

        if self.valid_count != self.period {
            return Some((f64::NAN, f64::NAN, f64::NAN));
        }

        debug_assert!(!self.max_deque.is_empty() && !self.min_deque.is_empty());
        let maxv = self.max_deque.front().unwrap().0;
        let minv = self.min_deque.front().unwrap().0;

        let mid = (maxv - minv).mul_add(0.5, minv);
        Some((maxv, mid, minv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_donchian_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = DonchianParams { period: None };
        let input = DonchianInput::from_candles(&candles, default_params);
        let output = donchian_with_kernel(&input, kernel)?;
        assert_eq!(output.upperband.len(), candles.close.len());
        Ok(())
    }

    fn check_donchian_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = DonchianParams { period: Some(20) };
        let input = DonchianInput::from_candles(&candles, params);
        let result = donchian_with_kernel(&input, kernel)?;
        let expected_last_five_upper = [61290.0, 61290.0, 61290.0, 61290.0, 61290.0];
        let expected_last_five_middle = [59583.0, 59583.0, 59583.0, 59583.0, 59583.0];
        let expected_last_five_lower = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
        let start = result.upperband.len().saturating_sub(5);
        for i in 0..5 {
            assert!((result.upperband[start + i] - expected_last_five_upper[i]).abs() < 1e-1);
            assert!((result.middleband[start + i] - expected_last_five_middle[i]).abs() < 1e-1);
            assert!((result.lowerband[start + i] - expected_last_five_lower[i]).abs() < 1e-1);
        }
        Ok(())
    }

    fn check_donchian_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 3.0, 2.0];
        let params = DonchianParams { period: Some(0) };
        let input = DonchianInput::from_slices(&high, &low, params);
        let res = donchian_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_donchian_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 3.0, 2.0];
        let params = DonchianParams { period: Some(10) };
        let input = DonchianInput::from_slices(&high, &low, params);
        let res = donchian_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_donchian_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0];
        let low = [90.0];
        let params = DonchianParams { period: Some(20) };
        let input = DonchianInput::from_slices(&high, &low, params);
        let res = donchian_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_donchian_mismatched_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 3.0];
        let params = DonchianParams { period: Some(2) };
        let input = DonchianInput::from_slices(&high, &low, params);
        let res = donchian_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_donchian_all_nan_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let params = DonchianParams { period: Some(2) };
        let input = DonchianInput::from_slices(&high, &low, params);
        let res = donchian_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_donchian_partial_computation(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 3.0, 5.0, 8.0, 8.5, 9.0, 2.0, 1.0];
        let low = [f64::NAN, 2.0, 1.0, 4.0, 4.5, 1.0, 1.0, 0.5];
        let params = DonchianParams { period: Some(3) };
        let input = DonchianInput::from_slices(&high, &low, params);
        let output = donchian_with_kernel(&input, kernel)?;
        assert_eq!(output.upperband.len(), high.len());
        assert!(output.upperband[2].is_nan());
        assert!(!output.upperband[3].is_nan());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_donchian_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            DonchianParams::default(),
            DonchianParams { period: Some(2) },
            DonchianParams { period: Some(5) },
            DonchianParams { period: Some(10) },
            DonchianParams { period: Some(20) },
            DonchianParams { period: Some(50) },
            DonchianParams { period: Some(100) },
            DonchianParams { period: Some(200) },
            DonchianParams { period: Some(500) },
            DonchianParams { period: Some(14) },
            DonchianParams { period: Some(26) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DonchianInput::from_candles(&candles, params.clone());
            let output = donchian_with_kernel(&input, kernel)?;

            let bands = [
                ("upperband", &output.upperband),
                ("middleband", &output.middleband),
                ("lowerband", &output.lowerband),
            ];

            for (band_name, band_values) in &bands {
                for (i, &val) in band_values.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in {} with params: period={} (param set {})",
                            test_name,
                            val,
                            bits,
                            i,
                            band_name,
                            params.period.unwrap_or(20),
                            param_idx
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in {} with params: period={} (param set {})",
                            test_name,
                            val,
                            bits,
                            i,
                            band_name,
                            params.period.unwrap_or(20),
                            param_idx
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in {} with params: period={} (param set {})",
                            test_name,
                            val,
                            bits,
                            i,
                            band_name,
                            params.period.unwrap_or(20),
                            param_idx
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_donchian_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_donchian_tests {
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

    #[test]
    fn test_donchian_into_matches_api() {
        let n = 256usize;
        let mut high = vec![f64::NAN; n];
        let mut low = vec![f64::NAN; n];

        for i in 5..n {
            let base = (i as f64).sin() * 10.0 + 100.0;
            high[i] = base + 2.0 + ((i % 7) as f64) * 0.1;
            low[i] = base - 2.0 - ((i % 5) as f64) * 0.1;
        }
        for idx in [37usize, 88, 133, 210] {
            high[idx] = f64::NAN;
        }
        for idx in [59usize, 120, 178, 220] {
            low[idx] = f64::NAN;
        }

        let input = DonchianInput::from_slices(&high, &low, DonchianParams::default());
        let expected = donchian(&input).expect("baseline donchian() failed");

        let mut up = vec![0.0; n];
        let mut mid = vec![0.0; n];
        let mut lo = vec![0.0; n];

        {
            donchian_into(&input, &mut up, &mut mid, &mut lo).expect("donchian_into failed");
        }

        assert_eq!(expected.upperband.len(), up.len());
        assert_eq!(expected.middleband.len(), mid.len());
        assert_eq!(expected.lowerband.len(), lo.len());

        let eq = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a == b);
        for i in 0..n {
            assert!(eq(expected.upperband[i], up[i]), "upper mismatch at {}", i);
            assert!(
                eq(expected.middleband[i], mid[i]),
                "middle mismatch at {}",
                i
            );
            assert!(eq(expected.lowerband[i], lo[i]), "lower mismatch at {}", i);
        }
    }

    #[test]
    fn test_donchian_selected_outputs_match_api() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path).expect("failed to load candles");
        let input = DonchianInput::with_default_candles(&candles);
        let expected = donchian(&input).expect("baseline donchian failed");
        let upper = donchian_upper_with_kernel(&input, Kernel::Scalar).expect("upper failed");
        let middle = donchian_middle_with_kernel(&input, Kernel::Scalar).expect("middle failed");
        let lower = donchian_lower_with_kernel(&input, Kernel::Scalar).expect("lower failed");

        let eq = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a == b);
        assert_eq!(expected.upperband.len(), upper.len());
        assert_eq!(expected.middleband.len(), middle.len());
        assert_eq!(expected.lowerband.len(), lower.len());
        for i in 0..upper.len() {
            assert!(
                eq(expected.upperband[i], upper[i]),
                "upper mismatch at {}",
                i
            );
            assert!(
                eq(expected.middleband[i], middle[i]),
                "middle mismatch at {}",
                i
            );
            assert!(
                eq(expected.lowerband[i], lower[i]),
                "lower mismatch at {}",
                i
            );
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_donchian_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let random_strat = (2usize..=64)
            .prop_flat_map(|period| {
                (
                    prop::collection::vec((50f64..5000f64, 0.1f64..50f64), period..400),
                    Just(period),
                )
            })
            .prop_map(|(price_pairs, period)| {
                let mut high = Vec::with_capacity(price_pairs.len());
                let mut low = Vec::with_capacity(price_pairs.len());
                for (base, spread) in price_pairs {
                    low.push(base);
                    high.push(base + spread);
                }
                (high, low, period)
            });

        let constant_strat =
            (2usize..=64, 50f64..5000f64, 0f64..50f64).prop_map(|(period, base_price, spread)| {
                let len = period + 50;
                let high = vec![base_price + spread; len];
                let low = vec![base_price; len];
                (high, low, period)
            });

        let trending_strat = (2usize..=64).prop_map(|period| {
            let len = period + 100;
            let mut high = Vec::with_capacity(len);
            let mut low = Vec::with_capacity(len);
            for i in 0..len {
                let base = 100.0 + i as f64 * 10.0;
                low.push(base);
                high.push(base + 5.0);
            }
            (high, low, period)
        });

        let volatile_strat = (2usize..=64)
            .prop_flat_map(|period| {
                (
                    prop::collection::vec((10f64..10000f64, 0.1f64..500f64), period..200),
                    Just(period),
                )
            })
            .prop_map(|(price_pairs, period)| {
                let mut high = Vec::with_capacity(price_pairs.len());
                let mut low = Vec::with_capacity(price_pairs.len());
                for (i, (base, spread)) in price_pairs.iter().enumerate() {
                    let volatility = if i % 3 == 0 { 2.0 } else { 0.5 };
                    low.push(base - spread * 0.1);
                    high.push(base + spread * volatility);
                }
                (high, low, period)
            });

        let single_price_strat = (2usize..=64, 50f64..5000f64).prop_map(|(period, price)| {
            let len = period + 50;
            let high = vec![price; len];
            let low = vec![price; len];
            (high, low, period)
        });

        let combined_strat = prop_oneof![
            random_strat,
            constant_strat,
            trending_strat,
            volatile_strat,
            single_price_strat,
        ];

        proptest::test_runner::TestRunner::default()
            .run(&combined_strat, |(high, low, period)| {
                for i in 0..high.len() {
                    prop_assert!(
                        high[i] >= low[i],
                        "Invalid input data at index {}: high ({}) < low ({})",
                        i,
                        high[i],
                        low[i]
                    );
                }

                let params = DonchianParams {
                    period: Some(period),
                };
                let input = DonchianInput::from_slices(&high, &low, params.clone());

                let output = donchian_with_kernel(&input, kernel).unwrap();
                let ref_output = donchian_with_kernel(&input, Kernel::Scalar).unwrap();

                for i in 0..(period - 1) {
                    prop_assert!(
                        output.upperband[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {} (period={})",
                        i,
                        output.upperband[i],
                        period
                    );
                    prop_assert!(
                        output.middleband[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {} (period={})",
                        i,
                        output.middleband[i],
                        period
                    );
                    prop_assert!(
                        output.lowerband[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {} (period={})",
                        i,
                        output.lowerband[i],
                        period
                    );
                }

                for i in (period - 1)..high.len() {
                    let start = i + 1 - period;
                    let window_high = &high[start..=i];
                    let window_low = &low[start..=i];

                    let expected_max = window_high
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let expected_min = window_low.iter().cloned().fold(f64::INFINITY, f64::min);
                    let expected_mid = 0.5 * (expected_max + expected_min);

                    let upper = output.upperband[i];
                    let middle = output.middleband[i];
                    let lower = output.lowerband[i];

                    prop_assert!(
                        (upper - expected_max).abs() < 1e-9,
                        "Upperband mismatch at idx {}: got {}, expected {} (period={})",
                        i,
                        upper,
                        expected_max,
                        period
                    );

                    prop_assert!(
                        (lower - expected_min).abs() < 1e-9,
                        "Lowerband mismatch at idx {}: got {}, expected {} (period={})",
                        i,
                        lower,
                        expected_min,
                        period
                    );

                    prop_assert!(
                        (middle - expected_mid).abs() < 1e-9,
                        "Middleband mismatch at idx {}: got {}, expected {} (period={})",
                        i,
                        middle,
                        expected_mid,
                        period
                    );

                    prop_assert!(
						upper >= middle && middle >= lower,
						"Band ordering violated at idx {}: upper={}, middle={}, lower={} (period={})",
						i, upper, middle, lower, period
					);

                    let data_min = window_low.iter().cloned().fold(f64::INFINITY, f64::min);
                    let data_max = window_high
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    prop_assert!(
						upper <= data_max + 1e-9 && lower >= data_min - 1e-9,
						"Bands outside data range at idx {}: upper={}, lower={}, data_range=[{}, {}]",
						i, upper, lower, data_min, data_max
					);

                    if period == 1 {
                        prop_assert!(
                            (upper - high[i]).abs() < 1e-9,
                            "Period=1: upper should equal current high at idx {}: {} vs {}",
                            i,
                            upper,
                            high[i]
                        );
                        prop_assert!(
                            (lower - low[i]).abs() < 1e-9,
                            "Period=1: lower should equal current low at idx {}: {} vs {}",
                            i,
                            lower,
                            low[i]
                        );
                    }

                    let window_is_single_price = window_high
                        .iter()
                        .zip(window_low.iter())
                        .all(|(h, l)| (h - l).abs() < f64::EPSILON);

                    if window_is_single_price {
                        prop_assert!(
							(upper - lower).abs() < 1e-9,
							"Single price window: bands should converge at idx {}: upper={}, lower={}",
							i, upper, lower
						);
                        prop_assert!(
							(middle - upper).abs() < 1e-9,
							"Single price window: middle should equal upper/lower at idx {}: middle={}, upper={}",
							i, middle, upper
						);
                    }

                    let ref_upper = ref_output.upperband[i];
                    let ref_middle = ref_output.middleband[i];
                    let ref_lower = ref_output.lowerband[i];

                    if !upper.is_finite() || !ref_upper.is_finite() {
                        prop_assert!(
                            upper.to_bits() == ref_upper.to_bits(),
                            "Upper finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            upper,
                            ref_upper
                        );
                    } else {
                        let ulp_diff = upper.to_bits().abs_diff(ref_upper.to_bits());
                        prop_assert!(
                            (upper - ref_upper).abs() <= 1e-9 || ulp_diff <= 4,
                            "Upper kernel mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            upper,
                            ref_upper,
                            ulp_diff
                        );
                    }

                    if !middle.is_finite() || !ref_middle.is_finite() {
                        prop_assert!(
                            middle.to_bits() == ref_middle.to_bits(),
                            "Middle finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            middle,
                            ref_middle
                        );
                    } else {
                        let ulp_diff = middle.to_bits().abs_diff(ref_middle.to_bits());
                        prop_assert!(
                            (middle - ref_middle).abs() <= 1e-9 || ulp_diff <= 4,
                            "Middle kernel mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            middle,
                            ref_middle,
                            ulp_diff
                        );
                    }

                    if !lower.is_finite() || !ref_lower.is_finite() {
                        prop_assert!(
                            lower.to_bits() == ref_lower.to_bits(),
                            "Lower finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            lower,
                            ref_lower
                        );
                    } else {
                        let ulp_diff = lower.to_bits().abs_diff(ref_lower.to_bits());
                        prop_assert!(
                            (lower - ref_lower).abs() <= 1e-9 || ulp_diff <= 4,
                            "Lower kernel mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            lower,
                            ref_lower,
                            ulp_diff
                        );
                    }

                    for (band_name, val) in [("upper", upper), ("middle", middle), ("lower", lower)]
                    {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111,
                            "Found alloc_with_nan_prefix poison in {} at idx {}: {} (0x{:016X})",
                            band_name,
                            i,
                            val,
                            bits
                        );
                        prop_assert!(
                            bits != 0x22222222_22222222,
                            "Found init_matrix_prefixes poison in {} at idx {}: {} (0x{:016X})",
                            band_name,
                            i,
                            val,
                            bits
                        );
                        prop_assert!(
                            bits != 0x33333333_33333333,
                            "Found make_uninit_matrix poison in {} at idx {}: {} (0x{:016X})",
                            band_name,
                            i,
                            val,
                            bits
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_donchian_tests!(
        check_donchian_partial_params,
        check_donchian_accuracy,
        check_donchian_zero_period,
        check_donchian_period_exceeds_length,
        check_donchian_very_small_dataset,
        check_donchian_mismatched_length,
        check_donchian_all_nan_data,
        check_donchian_partial_computation,
        check_donchian_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_donchian_tests!(check_donchian_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = DonchianBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = DonchianParams::default();
        let row = output.upper_for(&def).expect("default row missing");
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (10, 50, 10),
            (20, 100, 20),
            (50, 150, 25),
            (2, 5, 1),
            (100, 300, 50),
            (14, 26, 4),
            (5, 20, 3),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = DonchianBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            let bands = [
                ("upper", &output.upper),
                ("middle", &output.middle),
                ("lower", &output.lower),
            ];

            for (band_name, band_values) in &bands {
                for (idx, &val) in band_values.iter().enumerate() {
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
							 in {} at row {} col {} (flat index {}) with params: period={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            band_name,
                            row,
                            col,
                            idx,
                            combo.period.unwrap_or(20)
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 in {} at row {} col {} (flat index {}) with params: period={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            band_name,
                            row,
                            col,
                            idx,
                            combo.period.unwrap_or(20)
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							 in {} at row {} col {} (flat index {}) with params: period={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            band_name,
                            row,
                            col,
                            idx,
                            combo.period.unwrap_or(20)
                        );
                    }
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
}

#[inline(always)]
fn donchian_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &DonchianBatchRange,
    kern: Kernel,
    parallel: bool,
    out_upper: &mut [f64],
    out_middle: &mut [f64],
    out_lower: &mut [f64],
) -> Result<Vec<DonchianParams>, DonchianError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(DonchianError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    if high.len() != low.len() {
        return Err(DonchianError::MismatchedLength);
    }

    let first = high
        .iter()
        .position(|x| !x.is_nan())
        .zip(low.iter().position(|x| !x.is_nan()))
        .map(|(a, b)| a.max(b));
    let first = match first {
        Some(idx) => idx,
        None => return Err(DonchianError::AllValuesNaN),
    };

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(DonchianError::NotEnoughValidData {
            needed: max_p,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();

    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warmup = first + period - 1;
        let row_start = row * cols;
        for i in 0..warmup {
            out_upper[row_start + i] = f64::NAN;
            out_middle[row_start + i] = f64::NAN;
            out_lower[row_start + i] = f64::NAN;
        }
    }

    let do_row =
        |row: usize, out_upper: &mut [f64], out_middle: &mut [f64], out_lower: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            match kern {
                Kernel::Scalar => {
                    donchian_row_scalar(high, low, first, period, out_upper, out_middle, out_lower)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => {
                    donchian_row_avx2(high, low, first, period, out_upper, out_middle, out_lower)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => {
                    donchian_row_avx512(high, low, first, period, out_upper, out_middle, out_lower)
                }
                _ => unreachable!(),
            }
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_upper
                .par_chunks_mut(cols)
                .zip(out_middle.par_chunks_mut(cols))
                .zip(out_lower.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((upper, middle), lower))| do_row(row, upper, middle, lower));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (((upper, middle), lower), row) in out_upper
                .chunks_mut(cols)
                .zip(out_middle.chunks_mut(cols))
                .zip(out_lower.chunks_mut(cols))
                .zip(0..)
            {
                do_row(row, upper, middle, lower);
            }
        }
    } else {
        for (((upper, middle), lower), row) in out_upper
            .chunks_mut(cols)
            .zip(out_middle.chunks_mut(cols))
            .zip(out_lower.chunks_mut(cols))
            .zip(0..)
        {
            do_row(row, upper, middle, lower);
        }
    }

    Ok(combos)
}
