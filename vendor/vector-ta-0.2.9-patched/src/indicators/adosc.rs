use crate::utilities::data_loader::Candles;
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
use thiserror::Error;

// External formula/edge authority (audit oracle only; never a runtime backend):
// https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADOSC.c

#[derive(Debug, Clone)]
pub enum AdoscData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AdoscParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AdoscParams {
    fn default() -> Self {
        Self {
            short_period: Some(3),
            long_period: Some(10),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdoscInput<'a> {
    pub data: AdoscData<'a>,
    pub params: AdoscParams,
}

impl<'a> AdoscInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AdoscParams) -> Self {
        Self {
            data: AdoscData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: AdoscParams,
    ) -> Self {
        Self {
            data: AdoscData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AdoscData::Candles { candles },
            params: AdoscParams::default(),
        }
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdoscBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}

impl Default for AdoscBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdoscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AdoscOutput, AdoscError> {
        let p = AdoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AdoscInput::from_candles(c, p);
        adosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AdoscOutput, AdoscError> {
        let p = AdoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AdoscInput::from_slices(high, low, close, volume, p);
        adosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AdoscStream, AdoscError> {
        let p = AdoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        AdoscStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AdoscError {
    #[error("adosc: input data is empty")]
    EmptyInputData,
    #[error("adosc: All values are NaN.")]
    AllValuesNaN,
    #[error("adosc: Invalid period: short={short}, long={long}, data length={data_len}")]
    InvalidPeriod {
        short: usize,
        long: usize,
        data_len: usize,
    },
    #[error("adosc: short_period must be less than long_period: short={short}, long={long}")]
    ShortPeriodGreaterThanLong { short: usize, long: usize },
    #[error(
        "adosc: At least one slice is empty: high={high}, low={low}, close={close}, volume={volume}"
    )]
    EmptySlices {
        high: usize,
        low: usize,
        close: usize,
        volume: usize,
    },
    #[error("adosc: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("adosc: invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("adosc: not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adosc: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("adosc: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn adosc(input: &AdoscInput) -> Result<AdoscOutput, AdoscError> {
    adosc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn adosc_prepare<'a>(
    input: &'a AdoscInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        usize,
        usize,
        Kernel,
    ),
    AdoscError,
> {
    let (high, low, close, volume) = match &input.data {
        AdoscData::Candles { candles } => {
            let n = candles.close.len();
            if n == 0 {
                return Err(AdoscError::EmptyInputData);
            }

            let (hh, ll, cc, vv) = (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                candles.volume.as_slice(),
            );
            let len = cc.len();
            if hh.len() != len || ll.len() != len || vv.len() != len {
                return Err(AdoscError::EmptySlices {
                    high: hh.len(),
                    low: ll.len(),
                    close: cc.len(),
                    volume: vv.len(),
                });
            }
            (hh, ll, cc, vv)
        }
        AdoscData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
                if high.is_empty() && low.is_empty() && close.is_empty() && volume.is_empty() {
                    return Err(AdoscError::EmptyInputData);
                }
                return Err(AdoscError::EmptySlices {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                    volume: volume.len(),
                });
            }
            let len = close.len();
            if high.len() != len || low.len() != len || volume.len() != len {
                return Err(AdoscError::EmptySlices {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                    volume: volume.len(),
                });
            }
            (*high, *low, *close, *volume)
        }
    };

    let len = close.len();
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 || long > len {
        return Err(AdoscError::InvalidPeriod {
            short,
            long,
            data_len: len,
        });
    }
    if short >= long {
        return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
    }

    let all_nan = |s: &[f64]| s.iter().all(|x| x.is_nan());
    if all_nan(high) && all_nan(low) && all_nan(close) && all_nan(volume) {
        return Err(AdoscError::AllValuesNaN);
    }

    let chosen = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Auto => detect_best_kernel(),
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    Ok((high, low, close, volume, short, long, 0, len, chosen))
}

pub fn adosc_with_kernel(input: &AdoscInput, kernel: Kernel) -> Result<AdoscOutput, AdoscError> {
    let (high, low, close, volume, short, long, first, len, chosen) = adosc_prepare(input, kernel)?;

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => unsafe {
            adosc_scalar(high, low, close, volume, short, long, first, len)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            adosc_scalar(high, low, close, volume, short, long, first, len)
        },
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            adosc_scalar(high, low, close, volume, short, long, first, len)
        },
        _ => unreachable!(),
    }
}

#[inline]
pub fn adosc_into(input: &AdoscInput, out: &mut [f64]) -> Result<(), AdoscError> {
    adosc_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
pub unsafe fn adosc_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    _first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    debug_assert!(len > 0);

    if short == 3 && long == 10 {
        return adosc_scalar_3_10(high, low, close, volume, len);
    }

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);
    let one_minus_alpha_short = 1.0 - alpha_short;
    let one_minus_alpha_long = 1.0 - alpha_long;

    let mut out = alloc_with_nan_prefix(len, 0);

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let vp = volume.as_ptr();
    let op = out.as_mut_ptr();

    let h0 = *hp;
    let l0 = *lp;
    let c0 = *cp;
    let v0 = *vp;
    let hl0 = h0 - l0;
    let mfm0 = if hl0 > 0.0 {
        ((c0 - l0) - (h0 - c0)) / hl0
    } else {
        0.0
    };
    let mfv0 = mfm0 * v0;
    let mut sum_ad = mfv0;
    let mut short_ema = sum_ad;
    let mut long_ema = sum_ad;
    *op = short_ema - long_ema;

    let mut i = 1usize;
    while i < len {
        let h = *hp.add(i);
        let l = *lp.add(i);
        let c = *cp.add(i);
        let v = *vp.add(i);

        let hl = h - l;
        let mfm = if hl > 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        let mfv = mfm * v;
        sum_ad += mfv;
        short_ema = alpha_short * sum_ad + one_minus_alpha_short * short_ema;
        long_ema = alpha_long * sum_ad + one_minus_alpha_long * long_ema;
        *op.add(i) = short_ema - long_ema;

        i += 1;
    }

    Ok(AdoscOutput { values: out })
}

#[inline(always)]
unsafe fn adosc_scalar_3_10(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    let mut out = alloc_with_nan_prefix(len, 0);

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let vp = volume.as_ptr();
    let op = out.as_mut_ptr();

    let h0 = *hp;
    let l0 = *lp;
    let c0 = *cp;
    let v0 = *vp;
    let hl0 = h0 - l0;
    let mfm0 = if hl0 > 0.0 {
        ((c0 - l0) - (h0 - c0)) / hl0
    } else {
        0.0
    };
    let mfv0 = mfm0 * v0;
    let mut sum_ad = mfv0;
    let mut short_ema = sum_ad;
    let mut long_ema = sum_ad;
    *op = short_ema - long_ema;

    let alpha_long = 2.0 / 11.0;
    let one_minus_alpha_long = 1.0 - alpha_long;

    let mut i = 1usize;
    while i < len {
        let h = *hp.add(i);
        let l = *lp.add(i);
        let c = *cp.add(i);
        let v = *vp.add(i);

        let hl = h - l;
        let mfm = if hl > 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        let mfv = mfm * v;
        sum_ad += mfv;
        short_ema = 0.5 * sum_ad + 0.5 * short_ema;
        long_ema = alpha_long * sum_ad + one_minus_alpha_long * long_ema;
        *op.add(i) = short_ema - long_ema;

        i += 1;
    }

    Ok(AdoscOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[derive(Clone, Debug)]
pub struct AdoscBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
}

impl Default for AdoscBatchRange {
    fn default() -> Self {
        Self {
            short_period: (3, 3, 0),
            long_period: (10, 259, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdoscBatchBuilder {
    range: AdoscBatchRange,
    kernel: Kernel,
}

impl AdoscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    #[inline]
    pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AdoscBatchOutput, AdoscError> {
        adosc_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<AdoscBatchOutput, AdoscError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct AdoscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdoscParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AdoscBatchOutput {
    pub fn row_for_params(&self, p: &AdoscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(3) == p.short_period.unwrap_or(3)
                && c.long_period.unwrap_or(10) == p.long_period.unwrap_or(10)
        })
    }
    pub fn values_for(&self, p: &AdoscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn expand_grid(r: &AdoscBatchRange) -> Vec<AdoscParams> {
    match expand_grid_checked(r) {
        Ok(v) => v,
        Err(_) => Vec::new(),
    }
}

fn expand_grid_checked(r: &AdoscBatchRange) -> Result<Vec<AdoscParams>, AdoscError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AdoscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<_> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(AdoscError::InvalidRange { start, end, step });
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur - end < step {
                    break;
                }
                cur -= step;
            }
            if v.is_empty() {
                return Err(AdoscError::InvalidRange { start, end, step });
            }
            Ok(v)
        }
    }
    let shorts = axis_usize(r.short_period)?;
    let longs = axis_usize(r.long_period)?;

    let mut out = Vec::new();
    for &short in &shorts {
        for &long in &longs {
            if short == 0 || long == 0 || short >= long {
                continue;
            }
            out.push(AdoscParams {
                short_period: Some(short),
                long_period: Some(long),
            });
        }
    }
    if out.is_empty() {
        return Err(AdoscError::InvalidRange {
            start: r.short_period.0,
            end: r.long_period.1,
            step: r.short_period.2.max(r.long_period.2),
        });
    }
    Ok(out)
}

pub fn adosc_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    k: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdoscError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    adosc_batch_par_slice(high, low, close, volume, sweep, simd)
}

pub fn adosc_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
    adosc_batch_inner(high, low, close, volume, sweep, kern, false)
}

pub fn adosc_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
    adosc_batch_inner(high, low, close, volume, sweep, kern, true)
}

fn adosc_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AdoscBatchOutput, AdoscError> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(AdoscError::EmptySlices {
            high: high.len(),
            low: low.len(),
            close: close.len(),
            volume: volume.len(),
        });
    }

    let combos = expand_grid_checked(sweep)?;
    let first = 0;
    let len = close.len();
    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| AdoscError::InvalidInput("rows*cols overflow".into()))?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    debug_assert_eq!(buf_mu.len(), expected);

    let warm: Vec<usize> = vec![0; rows];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let mut adl = vec![0.0f64; len];
    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let vp = volume.as_ptr();
        let ap = adl.as_mut_ptr();

        let h0 = *hp;
        let l0 = *lp;
        let c0 = *cp;
        let v0 = *vp;
        let hl0 = h0 - l0;
        let mfm0 = if hl0 > 0.0 {
            ((c0 - l0) - (h0 - c0)) / hl0
        } else {
            0.0
        };
        let mfv0 = mfm0 * v0;
        *ap = mfv0;

        let mut i = 1usize;
        while i < len {
            let h = *hp.add(i);
            let l = *lp.add(i);
            let c = *cp.add(i);
            let v = *vp.add(i);
            let prev = *ap.add(i - 1);
            let hl = h - l;
            let mfm = if hl > 0.0 {
                ((c - l) - (h - c)) / hl
            } else {
                0.0
            };
            let mfv = mfm * v;
            *ap.add(i) = prev + mfv;
            i += 1;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let short = prm.short_period.unwrap();
        let long = prm.long_period.unwrap();

        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);
        let one_minus_alpha_short = 1.0 - alpha_short;
        let one_minus_alpha_long = 1.0 - alpha_long;

        let ap = adl.as_ptr();
        let op = out_row.as_mut_ptr();

        let mut short_ema = *ap;
        let mut long_ema = *ap;
        *op = short_ema - long_ema;

        let mut i = 1usize;
        while i < cols {
            let s = *ap.add(i);
            short_ema = alpha_short * s + one_minus_alpha_short * short_ema;
            long_ema = alpha_long * s + one_minus_alpha_long * long_ema;
            *op.add(i) = short_ema - long_ema;
            i += 1;
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

    Ok(AdoscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn adosc_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AdoscParams>, AdoscError> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(AdoscError::EmptySlices {
            high: high.len(),
            low: low.len(),
            close: close.len(),
            volume: volume.len(),
        });
    }

    let combos = expand_grid_checked(sweep)?;
    let first = 0;
    let len = close.len();
    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| AdoscError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(AdoscError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let mut adl = vec![0.0f64; len];
    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let vp = volume.as_ptr();
        let ap = adl.as_mut_ptr();

        let h0 = *hp;
        let l0 = *lp;
        let c0 = *cp;
        let v0 = *vp;
        let hl0 = h0 - l0;
        let mfm0 = if hl0 > 0.0 {
            ((c0 - l0) - (h0 - c0)) / hl0
        } else {
            0.0
        };
        let mfv0 = mfm0 * v0;
        *ap = mfv0;

        let mut i = 1usize;
        while i < len {
            let h = *hp.add(i);
            let l = *lp.add(i);
            let c = *cp.add(i);
            let v = *vp.add(i);
            let prev = *ap.add(i - 1);
            let hl = h - l;
            let mfm = if hl > 0.0 {
                ((c - l) - (h - c)) / hl
            } else {
                0.0
            };
            let mfv = mfm * v;
            *ap.add(i) = prev + mfv;
            i += 1;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let short = prm.short_period.unwrap();
        let long = prm.long_period.unwrap();

        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);
        let one_minus_alpha_short = 1.0 - alpha_short;
        let one_minus_alpha_long = 1.0 - alpha_long;

        let ap = adl.as_ptr();
        let op = out_row.as_mut_ptr();
        let mut short_ema = *ap;
        let mut long_ema = *ap;
        *op = short_ema - long_ema;

        let mut i = 1usize;
        while i < cols {
            let s = *ap.add(i);
            short_ema = alpha_short * s + one_minus_alpha_short * short_ema;
            long_ema = alpha_long * s + one_minus_alpha_long * long_ema;
            *op.add(i) = short_ema - long_ema;
            i += 1;
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
pub unsafe fn adosc_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    _first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    let len = out.len();
    debug_assert!(len > 0);

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);
    let one_minus_alpha_short = 1.0 - alpha_short;
    let one_minus_alpha_long = 1.0 - alpha_long;

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let vp = volume.as_ptr();
    let op = out.as_mut_ptr();

    let h0 = *hp;
    let l0 = *lp;
    let c0 = *cp;
    let v0 = *vp;
    let hl0 = h0 - l0;
    let mfm0 = if hl0 > 0.0 {
        ((c0 - l0) - (h0 - c0)) / hl0
    } else {
        0.0
    };
    let mfv0 = mfm0 * v0;
    let mut sum_ad = mfv0;
    let mut short_ema = sum_ad;
    let mut long_ema = sum_ad;
    *op = short_ema - long_ema;

    let mut i = 1usize;
    while i < len {
        let h = *hp.add(i);
        let l = *lp.add(i);
        let c = *cp.add(i);
        let v = *vp.add(i);

        let hl = h - l;
        let mfm = if hl > 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        let mfv = mfm * v;
        sum_ad += mfv;
        short_ema = alpha_short * sum_ad + one_minus_alpha_short * short_ema;
        long_ema = alpha_long * sum_ad + one_minus_alpha_long * long_ema;
        *op.add(i) = short_ema - long_ema;

        i += 1;
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

pub struct AdoscStream {
    short_period: usize,
    long_period: usize,

    alpha_short: f64,
    alpha_long: f64,
    one_minus_alpha_short: f64,
    one_minus_alpha_long: f64,

    sum_ad: f64,
    short_ema: f64,
    long_ema: f64,
    initialized: bool,
}

impl AdoscStream {
    #[inline(always)]
    pub fn try_new(params: AdoscParams) -> Result<Self, AdoscError> {
        let short = params.short_period.unwrap_or(3);
        let long = params.long_period.unwrap_or(10);
        if short == 0 || long == 0 {
            return Err(AdoscError::InvalidPeriod {
                short,
                long,
                data_len: 0,
            });
        }
        if short >= long {
            return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
        }

        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);

        Ok(Self {
            short_period: short,
            long_period: long,
            alpha_short,
            alpha_long,
            one_minus_alpha_short: 1.0 - alpha_short,
            one_minus_alpha_long: 1.0 - alpha_long,
            sum_ad: 0.0,
            short_ema: 0.0,
            long_ema: 0.0,
            initialized: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        if volume != 0.0 {
            let hl = high - low;
            if hl > 0.0 {
                let mfm = ((close - low) - (high - close)) / hl;
                self.sum_ad += mfm * volume;
            }
        }

        if !self.initialized {
            self.short_ema = self.sum_ad;
            self.long_ema = self.sum_ad;
            self.initialized = true;
            return 0.0;
        }

        let x = self.sum_ad;
        self.short_ema = self.alpha_short * x + self.one_minus_alpha_short * self.short_ema;
        self.long_ema = self.alpha_long * x + self.one_minus_alpha_long * self.long_ema;

        self.short_ema - self.long_ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_adosc_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdoscInput::with_default_candles(&candles);
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772];
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];
        for (i, &actual) in result_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-1,
                "ADOSC value mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
        for (i, &val) in result.values.iter().enumerate() {
            assert!(
                val.is_finite(),
                "ADOSC output at index {} should be finite, got {}",
                i,
                val
            );
        }
        Ok(())
    }

    fn check_adosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let partial_params = AdoscParams {
            short_period: Some(2),
            long_period: None,
        };
        let input = AdoscInput::from_candles(&candles, partial_params);
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let missing_short = AdoscParams {
            short_period: None,
            long_period: Some(12),
        };
        let input_missing = AdoscInput::from_candles(&candles, missing_short);
        let result_missing = adosc_with_kernel(&input_missing, kernel)?;
        assert_eq!(result_missing.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adosc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdoscInput::with_default_candles(&candles);
        match input.data {
            AdoscData::Candles { .. } => {}
            _ => panic!("Expected AdoscData::Candles variant"),
        }
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adosc_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 10.0, 10.0];
        let low = [5.0, 5.0, 5.0];
        let close = [7.0, 7.0, 7.0];
        let volume = [1000.0, 1000.0, 1000.0];
        let zero_short = AdoscParams {
            short_period: Some(0),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, zero_short);
        let result = adosc_with_kernel(&input, kernel);
        assert!(result.is_err());
        let zero_long = AdoscParams {
            short_period: Some(3),
            long_period: Some(0),
        };
        let input2 = AdoscInput::from_slices(&high, &low, &close, &volume, zero_long);
        let result2 = adosc_with_kernel(&input2, kernel);
        assert!(result2.is_err());
        Ok(())
    }

    fn check_adosc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [5.0, 5.5, 6.0];
        let close = [7.0, 8.0, 9.0];
        let volume = [1000.0, 1000.0, 1000.0];
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_adosc_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0];
        let low = [5.0];
        let close = [7.0];
        let volume = [1000.0];
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_adosc_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let first_input = AdoscInput::from_candles(&candles, first_params);
        let first_result = adosc_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AdoscParams {
            short_period: Some(2),
            long_period: Some(6),
        };
        let second_input = AdoscInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = adosc_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_adosc_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdoscInput::from_candles(&candles, AdoscParams::default());
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for (i, &val) in result.values[240..].iter().enumerate() {
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

    fn check_adosc_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_candles(&candles, params.clone());
        let batch_output = adosc_with_kernel(&input, kernel)?.values;
        let mut stream = AdoscStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), (&c, &v)) in candles
            .high
            .iter()
            .zip(candles.low.iter())
            .zip(candles.close.iter().zip(candles.volume.iter()))
        {
            stream_values.push(stream.update(h, l, c, v));
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] ADOSC streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let batch = AdoscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;
        let def = AdoscParams::default();
        let row = batch.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), candles.close.len());
        Ok(())
    }

    macro_rules! generate_all_adosc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    fn check_adosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let len = candles.close.len();
        let mut high = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut low = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut close = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut volume = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);

        high.resize(len, f64::from_bits(0x11111111_11111111));
        low.resize(len, f64::from_bits(0x22222222_22222222));
        close.resize(len, f64::from_bits(0x33333333_33333333));
        volume.resize(len, f64::from_bits(0x11111111_11111111));

        high.copy_from_slice(&candles.high);
        low.copy_from_slice(&candles.low);
        close.copy_from_slice(&candles.close);
        volume.copy_from_slice(&candles.volume);

        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc_with_kernel(&input, kernel)?;

        for (i, &val) in result.values.iter().enumerate() {
            assert_ne!(
                val.to_bits(),
                0x11111111_11111111,
                "[{}] Poison value 0x11111111_11111111 found at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x22222222_22222222,
                "[{}] Poison value 0x22222222_22222222 found at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x33333333_33333333,
                "[{}] Poison value 0x33333333_33333333 found at index {}",
                test_name,
                i
            );
        }

        Ok(())
    }

    fn check_batch_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let slice_end = candles.close.len().min(1000);
        let high_slice = &candles.high[..slice_end];
        let low_slice = &candles.low[..slice_end];
        let close_slice = &candles.close[..slice_end];
        let volume_slice = &candles.volume[..slice_end];

        let batch_config = AdoscBatchRange {
            short_period: (2, 5, 1),
            long_period: (8, 12, 2),
        };

        let result = adosc_batch_with_kernel(
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            &batch_config,
            kernel,
        )?;

        for (i, &val) in result.values.iter().enumerate() {
            assert_ne!(
                val.to_bits(),
                0x11111111_11111111,
                "[{}] Poison value 0x11111111_11111111 found in batch output at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x22222222_22222222,
                "[{}] Poison value 0x22222222_22222222 found in batch output at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x33333333_33333333,
                "[{}] Poison value 0x33333333_33333333 found in batch output at index {}",
                test_name,
                i
            );
        }

        let expected_rows = result.combos.len();
        let expected_cols = slice_end;
        assert_eq!(
            result.values.len(),
            expected_rows * expected_cols,
            "[{}] Batch output size mismatch",
            test_name
        );

        let batch_config2 = AdoscBatchRange {
            short_period: (3, 7, 2),
            long_period: (10, 20, 5),
        };

        let result2 = adosc_batch_with_kernel(
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            &batch_config2,
            kernel,
        )?;

        for (i, &val) in result2.values.iter().enumerate() {
            assert_ne!(
                val.to_bits(),
                0x11111111_11111111,
                "[{}] Poison value found in second batch config at index {}",
                test_name,
                i
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_adosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=10, 11usize..=30).prop_flat_map(|(short_period, long_period)| {
            let len = long_period..400;
            (
                prop::collection::vec(
                    (1f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    len.clone(),
                )
                .prop_flat_map(move |base_prices| {
                    let len = base_prices.len();

                    let high_spreads = prop::collection::vec(
                        (0f64..100f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    );
                    let low_spreads = prop::collection::vec(
                        (0f64..100f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    );

                    let close_positions = prop::collection::vec(0f64..=1f64, len);

                    (
                        Just(base_prices),
                        high_spreads,
                        low_spreads,
                        close_positions,
                    )
                })
                .prop_map(|(base, high_spreads, low_spreads, close_positions)| {
                    let mut high = Vec::with_capacity(base.len());
                    let mut low = Vec::with_capacity(base.len());
                    let mut close = Vec::with_capacity(base.len());

                    for i in 0..base.len() {
                        let h = base[i] + high_spreads[i];
                        let l = base[i] - low_spreads[i];
                        let c = l + (h - l) * close_positions[i];

                        high.push(h);
                        low.push(l);
                        close.push(c);
                    }

                    (high, low, close)
                }),
                prop::collection::vec((0f64..1e6f64).prop_filter("finite", |x| x.is_finite()), len),
                Just(short_period),
                Just(long_period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |((high, low, close), volume, short_period, long_period)| {
                    let len = high.len();
                    prop_assert_eq!(low.len(), len);
                    prop_assert_eq!(close.len(), len);
                    prop_assert_eq!(volume.len(), len);

                    for i in 0..len {
                        prop_assert!(
                            high[i] >= low[i],
                            "High must be >= Low at index {}: {} < {}",
                            i,
                            high[i],
                            low[i]
                        );
                        prop_assert!(
                            close[i] >= low[i] && close[i] <= high[i],
                            "Close must be between Low and High at index {}: {} not in [{}, {}]",
                            i,
                            close[i],
                            low[i],
                            high[i]
                        );
                    }

                    let params = AdoscParams {
                        short_period: Some(short_period),
                        long_period: Some(long_period),
                    };
                    let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);

                    let result = adosc_with_kernel(&input, kernel);
                    prop_assert!(result.is_ok(), "ADOSC computation failed: {:?}", result);

                    let AdoscOutput { values: out } = result.unwrap();

                    prop_assert_eq!(out.len(), len, "Output length mismatch");

                    for (i, &val) in out.iter().enumerate() {
                        prop_assert!(
                            val.is_finite(),
                            "ADOSC output at index {} should be finite, got {}",
                            i,
                            val
                        );
                    }

                    prop_assert!(
                        out[0].abs() < 1e-10,
                        "First ADOSC value should be 0, got {}",
                        out[0]
                    );

                    if volume.iter().all(|&v| v == 0.0) {
                        for &val in out.iter() {
                            prop_assert!(
                                val.abs() < 1e-9,
                                "With zero volume, ADOSC should be ~0, got {}",
                                val
                            );
                        }
                    }

                    for i in 0..len {
                        let h = high[i];
                        let l = low[i];
                        let c = close[i];
                        let hl = h - l;
                        if hl > 0.0 {
                            let mfm = ((c - l) - (h - c)) / hl;
                            prop_assert!(
                                mfm >= -1.0 - 1e-10 && mfm <= 1.0 + 1e-10,
                                "MFM at index {} out of bounds: {}",
                                i,
                                mfm
                            );
                        }
                    }

                    let total_volume: f64 = volume.iter().sum();

                    let expected_bound = total_volume * 0.5;
                    for (i, &val) in out.iter().enumerate() {
                        prop_assert!(
                            val.abs() <= expected_bound,
                            "ADOSC at index {} exceeds reasonable bounds: {} > {}",
                            i,
                            val.abs(),
                            expected_bound
                        );
                    }

                    prop_assert!(
                        short_period < long_period,
                        "Short period must be less than long period"
                    );

                    if len >= 3 {
                        let alpha_short = 2.0 / (short_period as f64 + 1.0);
                        let alpha_long = 2.0 / (long_period as f64 + 1.0);

                        let h0 = high[0];
                        let l0 = low[0];
                        let c0 = close[0];
                        let v0 = volume[0];
                        let hl0 = h0 - l0;
                        let mfm0 = if hl0 > 0.0 {
                            ((c0 - l0) - (h0 - c0)) / hl0
                        } else {
                            0.0
                        };
                        let mfv0 = mfm0 * v0;
                        let sum_ad0 = mfv0;
                        let expected_first = 0.0;
                        prop_assert!(
                            (out[0] - expected_first).abs() < 1e-9,
                            "First value mismatch: expected {}, got {}",
                            expected_first,
                            out[0]
                        );

                        let h1 = high[1];
                        let l1 = low[1];
                        let c1 = close[1];
                        let v1 = volume[1];
                        let hl1 = h1 - l1;
                        let mfm1 = if hl1 > 0.0 {
                            ((c1 - l1) - (h1 - c1)) / hl1
                        } else {
                            0.0
                        };
                        let mfv1 = mfm1 * v1;
                        let sum_ad1 = sum_ad0 + mfv1;
                        let short_ema1 = alpha_short * sum_ad1 + (1.0 - alpha_short) * sum_ad0;
                        let long_ema1 = alpha_long * sum_ad1 + (1.0 - alpha_long) * sum_ad0;
                        let expected_second = short_ema1 - long_ema1;
                        prop_assert!(
                            (out[1] - expected_second).abs() < 1e-9,
                            "Second value mismatch: expected {}, got {}",
                            expected_second,
                            out[1]
                        );
                    }

                    let ref_output = adosc_with_kernel(&input, Kernel::Scalar);
                    prop_assert!(ref_output.is_ok(), "Reference scalar computation failed");
                    let AdoscOutput { values: ref_out } = ref_output.unwrap();

                    for (i, (&val, &ref_val)) in out.iter().zip(ref_out.iter()).enumerate() {
                        let val_bits = val.to_bits();
                        let ref_bits = ref_val.to_bits();

                        if !val.is_finite() || !ref_val.is_finite() {
                            prop_assert_eq!(
                                val_bits,
                                ref_bits,
                                "NaN/Inf mismatch at index {}: {} vs {}",
                                i,
                                val,
                                ref_val
                            );
                        } else {
                            let ulp_diff = val_bits.abs_diff(ref_bits);
                            prop_assert!(
                                (val - ref_val).abs() <= 1e-9 || ulp_diff <= 4,
                                "Kernel mismatch at index {}: {} vs {} (diff: {}, ULP: {})",
                                i,
                                val,
                                ref_val,
                                (val - ref_val).abs(),
                                ulp_diff
                            );
                        }
                    }

                    Ok(())
                },
            )
            .map_err(|e| e.into())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_adosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    generate_all_adosc_tests!(
        check_adosc_accuracy,
        check_adosc_partial_params,
        check_adosc_default_candles,
        check_adosc_zero_period,
        check_adosc_period_exceeds_length,
        check_adosc_very_small_dataset,
        check_adosc_reinput,
        check_adosc_nan_handling,
        check_adosc_streaming,
        check_adosc_no_poison,
        check_adosc_property
    );

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_adosc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let len = 512usize;
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);

        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.05 + ((i % 13) as f64) * 0.01;
            let spread = 0.5 + ((i % 7) as f64) * 0.03;
            let lo = base - spread;
            let hi = base + spread;

            let frac = ((i % 97) as f64) / 96.0;
            let cl = lo + (hi - lo) * frac;
            let vol = 1_000.0 + ((i * 37) % 10_000) as f64;

            low.push(lo);
            high.push(hi);
            close.push(cl);
            volume.push(vol);
        }

        let input = AdoscInput::from_slices(&high, &low, &close, &volume, AdoscParams::default());

        let baseline = adosc(&input)?.values;

        let mut out = vec![0.0; len];
        adosc_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || (a - b).abs() <= 1e-12
        }

        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "ADOSC parity mismatch at index {}: api={}, into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }
}

#[inline]
pub fn adosc_into_slice(
    dst: &mut [f64],
    input: &AdoscInput,
    kern: Kernel,
) -> Result<(), AdoscError> {
    let (high, low, close, volume, short, long, first, len, chosen) = adosc_prepare(input, kern)?;
    if dst.len() != len {
        return Err(AdoscError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                adosc_row_scalar(high, low, close, volume, short, long, first, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                adosc_row_scalar(high, low, close, volume, short, long, first, dst)?
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                adosc_row_scalar(high, low, close, volume, short, long, first, dst)?
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}
