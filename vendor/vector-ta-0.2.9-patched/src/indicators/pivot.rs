use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

const N_LEVELS: usize = 9;
const MAX_PIVOT_MODE: usize = 4;

#[inline(always)]
fn validate_pivot_mode(mode: usize) -> Result<(), PivotError> {
    if mode <= MAX_PIVOT_MODE {
        Ok(())
    } else {
        Err(PivotError::InvalidMode { mode })
    }
}

/// Whether output bar `index` has every source value required by its exact
/// published pivot formula. Pivot levels belong to the current period but are
/// derived from the previous period. Woodie alone additionally consumes the
/// current period's open.
#[inline(always)]
fn pivot_inputs_valid_at(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    index: usize,
) -> bool {
    if index == 0 || index >= high.len() {
        return false;
    }
    let previous = index - 1;
    match mode {
        0 | 1 | 3 => {
            high[previous].is_finite() && low[previous].is_finite() && close[previous].is_finite()
        }
        2 => {
            high[previous].is_finite()
                && low[previous].is_finite()
                && close[previous].is_finite()
                && open[previous].is_finite()
        }
        4 => high[previous].is_finite() && low[previous].is_finite() && open[index].is_finite(),
        _ => false,
    }
}

#[inline]
fn first_valid_pivot_output(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
) -> Option<usize> {
    (1..high.len()).find(|&index| pivot_inputs_valid_at(high, low, close, open, mode, index))
}

#[inline(always)]
fn pivot_levels_at(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    index: usize,
) -> Option<[f64; N_LEVELS]> {
    if !pivot_inputs_valid_at(high, low, close, open, mode, index) {
        return None;
    }
    let previous = index - 1;
    pivot_levels_from_period(
        high[previous],
        low[previous],
        close[previous],
        open[previous],
        open[index],
        mode,
    )
}

/// Published Pivot Points Standard formulas. The returned order is
/// `[R4, R3, R2, R1, P, S1, S2, S3, S4]`.
#[inline(always)]
fn pivot_levels_from_period(
    previous_high: f64,
    previous_low: f64,
    previous_close: f64,
    previous_open: f64,
    current_open: f64,
    mode: usize,
) -> Option<[f64; N_LEVELS]> {
    let nan = f64::NAN;
    let h = previous_high;
    let l = previous_low;
    let c = previous_close;
    let d = h - l;

    match mode {
        // Traditional. Unlike the old implementation, all four published
        // resistance/support levels are preserved.
        0 if h.is_finite() && l.is_finite() && c.is_finite() => {
            let p = (h + l + c) / 3.0;
            let two_p = 2.0 * p;
            let three_p = 3.0 * p;
            Some([
                three_p + h - 3.0 * l,
                two_p + h - 2.0 * l,
                p + d,
                two_p - l,
                p,
                two_p - h,
                p - d,
                two_p - 2.0 * h + l,
                three_p - 3.0 * h + l,
            ])
        }
        // Fibonacci.
        1 if h.is_finite() && l.is_finite() && c.is_finite() => {
            let p = (h + l + c) / 3.0;
            Some([
                nan,
                p + d,
                p + 0.618 * d,
                p + 0.382 * d,
                p,
                p - 0.382 * d,
                p - 0.618 * d,
                p - d,
                nan,
            ])
        }
        // DeMark. The branch is selected by the previous period's open/close.
        2 if h.is_finite() && l.is_finite() && c.is_finite() && previous_open.is_finite() => {
            let x = if c < previous_open {
                h + 2.0 * l + c
            } else if c > previous_open {
                2.0 * h + l + c
            } else {
                h + l + 2.0 * c
            };
            let p = x / 4.0;
            Some([nan, nan, nan, x / 2.0 - l, p, x / 2.0 - h, nan, nan, nan])
        }
        // Camarilla. Keep the published 1.1/12, 1.1/6, 1.1/4 and
        // 1.1/2 ratios instead of decimal truncations.
        3 if h.is_finite() && l.is_finite() && c.is_finite() => {
            let p = (h + l + c) / 3.0;
            let scaled_range = 1.1 * d;
            let d1 = scaled_range / 12.0;
            let d2 = scaled_range / 6.0;
            let d3 = scaled_range / 4.0;
            let d4 = scaled_range / 2.0;
            Some([
                c + d4,
                c + d3,
                c + d2,
                c + d1,
                p,
                c - d1,
                c - d2,
                c - d3,
                c - d4,
            ])
        }
        // Woodie uses previous H/L and the current period's open.
        4 if h.is_finite() && l.is_finite() && current_open.is_finite() => {
            let p = (h + l + 2.0 * current_open) / 4.0;
            let two_p = 2.0 * p;
            let r3 = h + 2.0 * (p - l);
            let s3 = l - 2.0 * (h - p);
            Some([
                r3 + d,
                r3,
                p + d,
                two_p - l,
                p,
                two_p - h,
                p - d,
                s3,
                s3 - d,
            ])
        }
        _ => None,
    }
}

#[inline(always)]
fn pivot_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    k: Kernel,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    match k {
        Kernel::Scalar | Kernel::ScalarBatch => pivot_scalar(
            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
        ),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
            pivot_avx2(
                high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
            )
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            pivot_avx512(
                high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
            )
        },
        // A requested SIMD kernel that is not compiled for this target uses
        // the same reviewed portable implementation; no alternate formula is
        // retained.
        _ => pivot_scalar(
            high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
        ),
    }
}

#[derive(Debug, Clone)]
pub enum PivotData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        open: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PivotParams {
    pub mode: Option<usize>,
}
impl Default for PivotParams {
    fn default() -> Self {
        Self { mode: Some(3) }
    }
}

#[derive(Debug, Clone)]
pub struct PivotInput<'a> {
    pub data: PivotData<'a>,
    pub params: PivotParams,
}
impl<'a> PivotInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: PivotParams) -> Self {
        Self {
            data: PivotData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        open: &'a [f64],
        params: PivotParams,
    ) -> Self {
        Self {
            data: PivotData::Slices {
                high,
                low,
                close,
                open,
            },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, PivotParams::default())
    }
    #[inline]
    pub fn get_mode(&self) -> usize {
        self.params
            .mode
            .unwrap_or_else(|| PivotParams::default().mode.unwrap())
    }
}
impl<'a> AsRef<PivotData<'a>> for PivotInput<'a> {
    fn as_ref(&self) -> &PivotData<'a> {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct PivotOutput {
    pub r4: Vec<f64>,
    pub r3: Vec<f64>,
    pub r2: Vec<f64>,
    pub r1: Vec<f64>,
    pub pp: Vec<f64>,
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    pub s3: Vec<f64>,
    pub s4: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum PivotError {
    #[error("pivot: One or more required fields is empty.")]
    EmptyData,
    #[error("pivot: All values are NaN.")]
    AllValuesNaN,
    #[error("pivot: Not enough valid data after the first valid index.")]
    NotEnoughValidData,
    #[error("pivot: Output slice length mismatch (expected {expected}, got {got}).")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("pivot: Invalid range: start={start}, end={end}, step={step}.")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("pivot: Invalid kernel for batch path: {0:?}.")]
    InvalidKernelForBatch(Kernel),
    #[error("pivot: Invalid formula mode {mode}; expected 0..={MAX_PIVOT_MODE}.")]
    InvalidMode { mode: usize },
}

#[derive(Copy, Clone, Debug)]
pub struct PivotBuilder {
    mode: Option<usize>,
    kernel: Kernel,
}
impl Default for PivotBuilder {
    fn default() -> Self {
        Self {
            mode: None,
            kernel: Kernel::Auto,
        }
    }
}
impl PivotBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn mode(mut self, mode: usize) -> Self {
        self.mode = Some(mode);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<PivotOutput, PivotError> {
        let params = PivotParams { mode: self.mode };
        let input = PivotInput::from_candles(candles, params);
        pivot_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        open: &[f64],
    ) -> Result<PivotOutput, PivotError> {
        let params = PivotParams { mode: self.mode };
        let input = PivotInput::from_slices(high, low, close, open, params);
        pivot_with_kernel(&input, self.kernel)
    }
}

#[inline]
pub fn pivot(input: &PivotInput) -> Result<PivotOutput, PivotError> {
    pivot_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn pivot_refs<'a>(input: &'a PivotInput<'a>) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
    match &input.data {
        PivotData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.open.as_slice(),
        ),
        PivotData::Slices {
            high,
            low,
            close,
            open,
        } => (*high, *low, *close, *open),
    }
}

pub fn pivot_with_kernel(input: &PivotInput, kernel: Kernel) -> Result<PivotOutput, PivotError> {
    let (high, low, close, open) = pivot_refs(input);
    let len = high.len();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(PivotError::EmptyData);
    }
    if low.len() != len || close.len() != len || open.len() != len {
        return Err(PivotError::EmptyData);
    }
    let mode = input.get_mode();
    validate_pivot_mode(mode)?;

    if len < 2 {
        return Err(PivotError::NotEnoughValidData);
    }
    let first_valid_idx =
        first_valid_pivot_output(high, low, close, open, mode).ok_or(PivotError::AllValuesNaN)?;

    let mut r4 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut r3 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut r2 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut r1 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut pp = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s1 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s2 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s3 = alloc_with_nan_prefix(len, first_valid_idx);
    let mut s4 = alloc_with_nan_prefix(len, first_valid_idx);

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };
    pivot_compute_into(
        high,
        low,
        close,
        open,
        mode,
        first_valid_idx,
        chosen,
        &mut r4,
        &mut r3,
        &mut r2,
        &mut r1,
        &mut pp,
        &mut s1,
        &mut s2,
        &mut s3,
        &mut s4,
    );
    Ok(PivotOutput {
        r4,
        r3,
        r2,
        r1,
        pp,
        s1,
        s2,
        s3,
        s4,
    })
}

#[inline]
pub fn pivot_into(
    input: &PivotInput,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) -> Result<(), PivotError> {
    let (high, low, close, open) = pivot_refs(input);

    let len = high.len();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(PivotError::EmptyData);
    }
    if low.len() != len || close.len() != len || open.len() != len {
        return Err(PivotError::EmptyData);
    }
    let expected = len;
    let first_mismatch = [
        r4.len(),
        r3.len(),
        r2.len(),
        r1.len(),
        pp.len(),
        s1.len(),
        s2.len(),
        s3.len(),
        s4.len(),
    ]
    .into_iter()
    .find(|&got| got != expected);
    if let Some(got) = first_mismatch {
        return Err(PivotError::OutputLengthMismatch { expected, got });
    }

    let mode = input.get_mode();
    validate_pivot_mode(mode)?;

    if len < 2 {
        return Err(PivotError::NotEnoughValidData);
    }
    let first_valid_idx =
        first_valid_pivot_output(high, low, close, open, mode).ok_or(PivotError::AllValuesNaN)?;

    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for i in 0..first_valid_idx {
        r4[i] = qnan;
        r3[i] = qnan;
        r2[i] = qnan;
        r1[i] = qnan;
        pp[i] = qnan;
        s1[i] = qnan;
        s2[i] = qnan;
        s3[i] = qnan;
        s4[i] = qnan;
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    let chosen = detect_best_kernel();
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    let chosen = Kernel::Scalar;
    pivot_compute_into(
        high,
        low,
        close,
        open,
        mode,
        first_valid_idx,
        chosen,
        r4,
        r3,
        r2,
        r1,
        pp,
        s1,
        s2,
        s3,
        s4,
    );

    Ok(())
}

#[inline]
pub fn pivot_into_slices(
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
    input: &PivotInput,
    kern: Kernel,
) -> Result<(), PivotError> {
    let (high, low, close, open) = pivot_refs(input);

    let len = high.len();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(PivotError::EmptyData);
    }
    if low.len() != len || close.len() != len || open.len() != len {
        return Err(PivotError::EmptyData);
    }
    let expected = len;
    let first_mismatch = [
        r4.len(),
        r3.len(),
        r2.len(),
        r1.len(),
        pp.len(),
        s1.len(),
        s2.len(),
        s3.len(),
        s4.len(),
    ]
    .into_iter()
    .find(|&got| got != expected);
    if let Some(got) = first_mismatch {
        return Err(PivotError::OutputLengthMismatch { expected, got });
    }

    let mode = input.get_mode();
    validate_pivot_mode(mode)?;

    if len < 2 {
        return Err(PivotError::NotEnoughValidData);
    }
    let first_valid_idx =
        first_valid_pivot_output(high, low, close, open, mode).ok_or(PivotError::AllValuesNaN)?;

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    pivot_compute_into(
        high,
        low,
        close,
        open,
        mode,
        first_valid_idx,
        chosen,
        r4,
        r3,
        r2,
        r1,
        pp,
        s1,
        s2,
        s3,
        s4,
    );

    for i in 0..first_valid_idx {
        r4[i] = f64::NAN;
        r3[i] = f64::NAN;
        r2[i] = f64::NAN;
        r1[i] = f64::NAN;
        pp[i] = f64::NAN;
        s1[i] = f64::NAN;
        s2[i] = f64::NAN;
        s3[i] = f64::NAN;
        s4[i] = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn pivot_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    let len = high.len();
    if first >= len {
        return;
    }

    for i in first.max(1)..len {
        let levels =
            pivot_levels_at(high, low, close, open, mode, i).unwrap_or([f64::NAN; N_LEVELS]);
        r4[i] = levels[0];
        r3[i] = levels[1];
        r2[i] = levels[2];
        r1[i] = levels[3];
        pp[i] = levels[4];
        s1[i] = levels[5];
        s2[i] = levels[6];
        s3[i] = levels[7];
        s4[i] = levels[8];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn pivot_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    // The target-feature wrapper lets LLVM vectorize the single reviewed
    // formula body. There is no second handwritten mathematical authority.
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn pivot_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    // The target-feature wrapper lets LLVM vectorize the single reviewed
    // formula body. There is no second handwritten mathematical authority.
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn pivot_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    // The target-feature wrapper lets LLVM vectorize the single reviewed
    // formula body. There is no second handwritten mathematical authority.
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
pub unsafe fn pivot_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    // The target-feature wrapper lets LLVM vectorize the single reviewed
    // formula body. There is no second handwritten mathematical authority.
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    );
}
#[inline(always)]
pub unsafe fn pivot_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx2(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512_short(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_avx512_long(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}

#[inline(always)]
fn pivot_rows_scalar_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    mode: usize,
    first: usize,
    r4: &mut [f64],
    r3: &mut [f64],
    r2: &mut [f64],
    r1: &mut [f64],
    pp: &mut [f64],
    s1: &mut [f64],
    s2: &mut [f64],
    s3: &mut [f64],
    s4: &mut [f64],
) {
    pivot_scalar(
        high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4,
    )
}

#[inline(always)]
fn pivot_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PivotParams>, PivotError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        let (start, end, step) = sweep.mode;
        return Err(PivotError::InvalidRange { start, end, step });
    }
    let cols = high.len();
    if cols == 0 || low.len() != cols || close.len() != cols || open.len() != cols {
        return Err(PivotError::EmptyData);
    }

    if cols < 2 {
        return Err(PivotError::NotEnoughValidData);
    }
    if !combos.iter().any(|params| {
        first_valid_pivot_output(high, low, close, open, params.mode.unwrap_or(3)).is_some()
    }) {
        return Err(PivotError::AllValuesNaN);
    }

    let rows = combos
        .len()
        .checked_mul(N_LEVELS)
        .ok_or(PivotError::InvalidRange {
            start: combos.len(),
            end: N_LEVELS,
            step: 0,
        })?;
    let expected_len = rows.checked_mul(cols).ok_or(PivotError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;
    if out.len() != expected_len {
        return Err(PivotError::OutputLengthMismatch {
            expected: expected_len,
            got: out.len(),
        });
    }

    // Every cell is initialized before parallel work. Invalid source periods
    // remain explicitly undefined rather than retaining uninitialized poison
    // or bridging over gaps.
    out.fill(f64::NAN);
    let _ = kern;
    let row_width = N_LEVELS * cols;

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(row_width)
                .zip(combos.par_iter())
                .for_each(|(chunk, params)| {
                    let mode = params.mode.unwrap_or(3);
                    let mut level_rows = chunk.chunks_mut(cols);
                    let r4 = level_rows.next().unwrap();
                    let r3 = level_rows.next().unwrap();
                    let r2 = level_rows.next().unwrap();
                    let r1 = level_rows.next().unwrap();
                    let pp = level_rows.next().unwrap();
                    let s1 = level_rows.next().unwrap();
                    let s2 = level_rows.next().unwrap();
                    let s3 = level_rows.next().unwrap();
                    let s4 = level_rows.next().unwrap();
                    pivot_rows_scalar_into(
                        high, low, close, open, mode, 1, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                    );
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (chunk, params) in out.chunks_mut(row_width).zip(combos.iter()) {
                let mode = params.mode.unwrap_or(3);
                let mut level_rows = chunk.chunks_mut(cols);
                let r4 = level_rows.next().unwrap();
                let r3 = level_rows.next().unwrap();
                let r2 = level_rows.next().unwrap();
                let r1 = level_rows.next().unwrap();
                let pp = level_rows.next().unwrap();
                let s1 = level_rows.next().unwrap();
                let s2 = level_rows.next().unwrap();
                let s3 = level_rows.next().unwrap();
                let s4 = level_rows.next().unwrap();
                pivot_rows_scalar_into(
                    high, low, close, open, mode, 1, r4, r3, r2, r1, pp, s1, s2, s3, s4,
                );
            }
        }
    } else {
        for (chunk, params) in out.chunks_mut(row_width).zip(combos.iter()) {
            let mode = params.mode.unwrap_or(3);
            let mut level_rows = chunk.chunks_mut(cols);
            let r4 = level_rows.next().unwrap();
            let r3 = level_rows.next().unwrap();
            let r2 = level_rows.next().unwrap();
            let r1 = level_rows.next().unwrap();
            let pp = level_rows.next().unwrap();
            let s1 = level_rows.next().unwrap();
            let s2 = level_rows.next().unwrap();
            let s3 = level_rows.next().unwrap();
            let s4 = level_rows.next().unwrap();
            pivot_rows_scalar_into(
                high, low, close, open, mode, 1, r4, r3, r2, r1, pp, s1, s2, s3, s4,
            );
        }
    }
    Ok(combos)
}

#[derive(Clone, Debug)]
pub struct PivotBatchRange {
    pub mode: (usize, usize, usize),
}
impl Default for PivotBatchRange {
    fn default() -> Self {
        Self { mode: (3, 3, 1) }
    }
}
#[derive(Clone, Debug, Default)]
pub struct PivotBatchBuilder {
    range: PivotBatchRange,
    kernel: Kernel,
}
impl PivotBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn mode_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.mode = (start, end, step);
        self
    }
    #[inline]
    pub fn mode_static(mut self, m: usize) -> Self {
        self.range.mode = (m, m, 1);
        self
    }
    pub fn apply_slice(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        open: &[f64],
    ) -> Result<PivotBatchOutput, PivotError> {
        pivot_batch_with_kernel(high, low, close, open, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<PivotBatchOutput, PivotError> {
        let high = source_type(candles, "high");
        let low = source_type(candles, "low");
        let close = source_type(candles, "close");
        let open = source_type(candles, "open");
        self.apply_slice(high, low, close, open)
    }
    pub fn with_default_candles(candles: &Candles) -> Result<PivotBatchOutput, PivotError> {
        PivotBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

pub fn pivot_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    k: Kernel,
) -> Result<PivotBatchOutput, PivotError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PivotError::InvalidKernelForBatch(k)),
    };
    pivot_batch_inner(high, low, close, open, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct PivotBatchOutput {
    pub levels: Vec<[Vec<f64>; 9]>,
    pub combos: Vec<PivotParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct PivotBatchFlatOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PivotParams>,
    pub rows: usize,
    pub cols: usize,
}

pub fn pivot_batch_flat_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    k: Kernel,
) -> Result<PivotBatchFlatOutput, PivotError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PivotError::InvalidKernelForBatch(k)),
    };
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        let (start, end, step) = sweep.mode;
        return Err(PivotError::InvalidRange { start, end, step });
    }
    let cols = high.len();
    let rows = combos
        .len()
        .checked_mul(N_LEVELS)
        .ok_or(PivotError::InvalidRange {
            start: combos.len(),
            end: N_LEVELS,
            step: 0,
        })?;
    let _ = rows.checked_mul(cols).ok_or(PivotError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    let mut values = vec![f64::NAN; rows * cols];
    pivot_batch_inner_into(high, low, close, open, sweep, kernel, true, &mut values)?;
    Ok(PivotBatchFlatOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn expand_grid(r: &PivotBatchRange) -> Result<Vec<PivotParams>, PivotError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, PivotError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                vals.push(cur);
                cur = cur
                    .checked_add(step)
                    .ok_or(PivotError::InvalidRange { start, end, step })?;
            }
        } else {
            let mut cur = start;
            while cur >= end {
                vals.push(cur);
                cur = cur
                    .checked_sub(step)
                    .ok_or(PivotError::InvalidRange { start, end, step })?;
                if cur == 0 && end > 0 {
                    break;
                }
                if let Some(&last) = vals.last() {
                    if last == cur {
                        break;
                    }
                }
            }
            if let Some(&last) = vals.last() {
                if last < end {
                    vals.pop();
                }
            }
        }
        if vals.is_empty() {
            return Err(PivotError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }

    let modes = axis_usize(r.mode)?;
    let mut v = Vec::with_capacity(modes.len());
    for m in modes {
        validate_pivot_mode(m)?;
        v.push(PivotParams { mode: Some(m) });
    }
    Ok(v)
}
fn pivot_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    open: &[f64],
    sweep: &PivotBatchRange,
    kernel: Kernel,
) -> Result<PivotBatchOutput, PivotError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        let (start, end, step) = sweep.mode;
        return Err(PivotError::InvalidRange { start, end, step });
    }
    let len = high.len();
    if len == 0 || low.len() != len || close.len() != len || open.len() != len {
        return Err(PivotError::EmptyData);
    }
    if len < 2 {
        return Err(PivotError::NotEnoughValidData);
    }
    if !combos.iter().any(|params| {
        first_valid_pivot_output(high, low, close, open, params.mode.unwrap_or(3)).is_some()
    }) {
        return Err(PivotError::AllValuesNaN);
    }
    let mut levels = Vec::with_capacity(combos.len());
    for p in &combos {
        let mode = p.mode.unwrap_or(3);
        let mut r4 = vec![f64::NAN; len];
        let mut r3 = vec![f64::NAN; len];
        let mut r2 = vec![f64::NAN; len];
        let mut r1 = vec![f64::NAN; len];
        let mut pp = vec![f64::NAN; len];
        let mut s1 = vec![f64::NAN; len];
        let mut s2 = vec![f64::NAN; len];
        let mut s3 = vec![f64::NAN; len];
        let mut s4 = vec![f64::NAN; len];
        let _ = kernel;
        pivot_rows_scalar_into(
            high, low, close, open, mode, 1, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1,
            &mut s2, &mut s3, &mut s4,
        );
        levels.push([r4, r3, r2, r1, pp, s1, s2, s3, s4]);
    }
    let rows = combos.len();
    let cols = high.len();
    Ok(PivotBatchOutput {
        levels,
        combos,
        rows,
        cols,
    })
}

pub struct PivotStream {
    mode: usize,
    previous: Option<[f64; 4]>,
}

impl PivotStream {
    pub fn new(mode: usize) -> Result<Self, PivotError> {
        validate_pivot_mode(mode)?;
        Ok(Self {
            mode,
            previous: None,
        })
    }

    pub fn try_new(params: PivotParams) -> Result<Self, PivotError> {
        let mode = params.mode.unwrap_or(3);
        Self::new(mode)
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        open: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        let current = [high, low, close, open];
        let previous = self.previous.replace(current)?;
        let levels = pivot_levels_from_period(
            previous[0],
            previous[1],
            previous[2],
            previous[3],
            open,
            self.mode,
        )?;
        Some((
            levels[0], levels[1], levels[2], levels[3], levels[4], levels[5], levels[6], levels[7],
            levels[8],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;
    use paste::paste;

    fn test_candles() -> Candles {
        let len = 512usize;
        let mut timestamp = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.25;
            let o = base + (i % 7) as f64 * 0.01;
            let c = base + (i % 5) as f64 * 0.015 - 0.02;
            timestamp.push(i as i64 * 300_000);
            open.push(o);
            high.push(o.max(c) + 0.75 + (i % 3) as f64 * 0.05);
            low.push(o.min(c) - 0.65 - (i % 4) as f64 * 0.04);
            close.push(c);
            volume.push(1_000.0 + i as f64);
        }
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn pivot_from_rows(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        open: &[f64],
        mode: usize,
    ) -> Result<PivotOutput, PivotError> {
        pivot(&PivotInput::from_slices(
            high,
            low,
            close,
            open,
            PivotParams { mode: Some(mode) },
        ))
    }

    #[test]
    fn pivot_standard_uses_the_previous_period_and_emits_all_four_levels() {
        let output = pivot_from_rows(
            &[10.0, 20.0, 30.0],
            &[6.0, 12.0, 18.0],
            &[8.0, 16.0, 24.0],
            &[7.0, 15.0, 21.0],
            0,
        )
        .unwrap();

        for value in [
            output.r4[0],
            output.r3[0],
            output.r2[0],
            output.r1[0],
            output.pp[0],
            output.s1[0],
            output.s2[0],
            output.s3[0],
            output.s4[0],
        ] {
            assert!(value.is_nan(), "the first period has no prior OHLC source");
        }

        // Official Traditional pivot formulas over the PREVIOUS bar:
        // H=10, L=6, C=8 -> P=8 and range=4.
        assert_eq!(output.pp[1], 8.0);
        assert_eq!(output.r1[1], 10.0);
        assert_eq!(output.r2[1], 12.0);
        assert_eq!(output.r3[1], 14.0);
        assert_eq!(output.r4[1], 16.0);
        assert_eq!(output.s1[1], 6.0);
        assert_eq!(output.s2[1], 4.0);
        assert_eq!(output.s3[1], 2.0);
        assert_eq!(output.s4[1], 0.0);
    }

    #[test]
    fn pivot_camarilla_uses_the_published_one_point_one_ratios() {
        let output =
            pivot_from_rows(&[1.7, 9.0], &[0.2, 8.0], &[1.1, 8.5], &[0.8, 8.25], 3).unwrap();
        let range: f64 = 1.7 - 0.2;
        let close: f64 = 1.1;

        assert_eq!(
            output.r1[1].to_bits(),
            (close + 1.1 * range / 12.0).to_bits()
        );
        assert_eq!(
            output.r2[1].to_bits(),
            (close + 1.1 * range / 6.0).to_bits()
        );
        assert_eq!(
            output.r3[1].to_bits(),
            (close + 1.1 * range / 4.0).to_bits()
        );
        assert_eq!(
            output.r4[1].to_bits(),
            (close + 1.1 * range / 2.0).to_bits()
        );
        assert_eq!(
            output.s1[1].to_bits(),
            (close - 1.1 * range / 12.0).to_bits()
        );
        assert_eq!(
            output.s2[1].to_bits(),
            (close - 1.1 * range / 6.0).to_bits()
        );
        assert_eq!(
            output.s3[1].to_bits(),
            (close - 1.1 * range / 4.0).to_bits()
        );
        assert_eq!(
            output.s4[1].to_bits(),
            (close - 1.1 * range / 2.0).to_bits()
        );
    }

    #[test]
    fn pivot_woodie_uses_previous_high_low_and_current_open() {
        let output =
            pivot_from_rows(&[10.0, 200.0], &[6.0, 120.0], &[8.0, 160.0], &[7.0, 9.0], 4).unwrap();
        let expected_pp = (10.0 + 6.0 + 2.0 * 9.0) / 4.0;
        assert_eq!(output.pp[1], expected_pp);
        assert_eq!(output.r1[1], 2.0 * expected_pp - 6.0);
        assert_eq!(output.s1[1], 2.0 * expected_pp - 10.0);
    }

    #[test]
    fn pivot_demark_uses_the_previous_period_branch() {
        let output = pivot_from_rows(
            &[10.0, 200.0],
            &[6.0, 120.0],
            &[8.0, 160.0],
            &[9.0, 90.0],
            2,
        )
        .unwrap();
        let x = 10.0 + 2.0 * 6.0 + 8.0;
        assert_eq!(output.pp[1], x / 4.0);
        assert_eq!(output.r1[1], x / 2.0 - 6.0);
        assert_eq!(output.s1[1], x / 2.0 - 10.0);
    }

    #[test]
    fn pivot_rejects_an_unknown_formula_mode() {
        let result = pivot_from_rows(&[10.0, 11.0], &[6.0, 7.0], &[8.0, 9.0], &[7.0, 8.0], 5);
        assert!(
            result.is_err(),
            "an unknown mode must not return plausible all-NaN data"
        );
    }

    #[test]
    fn pivot_validity_is_bound_to_the_exact_previous_period_inputs() {
        let output = pivot_from_rows(
            &[10.0, f64::NAN, 30.0],
            &[6.0, 12.0, 18.0],
            &[8.0, 16.0, 24.0],
            &[7.0, 15.0, 21.0],
            0,
        )
        .unwrap();

        assert!(output.pp[0].is_nan());
        assert_eq!(
            output.pp[1], 8.0,
            "current HLC is not a previous-period input yet"
        );
        assert!(
            output.pp[2].is_nan(),
            "an invalid previous period must not be bridged"
        );
    }

    #[test]
    fn pivot_stream_waits_for_the_previous_period_and_matches_the_slice_api() {
        let mut stream = PivotStream::try_new(PivotParams { mode: Some(0) }).unwrap();
        assert!(stream.update(10.0, 6.0, 8.0, 7.0).is_none());

        let streamed = stream.update(20.0, 12.0, 16.0, 15.0).unwrap();
        let batch =
            pivot_from_rows(&[10.0, 20.0], &[6.0, 12.0], &[8.0, 16.0], &[7.0, 15.0], 0).unwrap();
        let expected = (
            batch.r4[1],
            batch.r3[1],
            batch.r2[1],
            batch.r1[1],
            batch.pp[1],
            batch.s1[1],
            batch.s2[1],
            batch.s3[1],
            batch.s4[1],
        );
        assert_eq!(streamed, expected);
    }

    #[test]
    fn pivot_batch_matches_each_mode_specific_slice_oracle() {
        let high = [10.0, 20.0, 30.0];
        let low = [6.0, 12.0, 18.0];
        let close = [8.0, 16.0, 24.0];
        let open = [9.0, 15.0, 21.0];
        let batch = PivotBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .mode_range(0, MAX_PIVOT_MODE, 1)
            .apply_slice(&high, &low, &close, &open)
            .unwrap();

        for (row, params) in batch.combos.iter().enumerate() {
            let mode = params.mode.unwrap();
            let expected = pivot_from_rows(&high, &low, &close, &open, mode).unwrap();
            let expected_levels = [
                expected.r4,
                expected.r3,
                expected.r2,
                expected.r1,
                expected.pp,
                expected.s1,
                expected.s2,
                expected.s3,
                expected.s4,
            ];
            for (actual, expected) in batch.levels[row].iter().zip(expected_levels.iter()) {
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "mode {mode}");
                }
            }
        }
    }

    #[test]
    fn pivot_batch_rejects_an_unknown_formula_mode() {
        let result = PivotBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .mode_static(MAX_PIVOT_MODE + 1)
            .apply_slice(&[10.0, 11.0], &[6.0, 7.0], &[8.0, 9.0], &[7.0, 8.0]);
        assert!(matches!(result, Err(PivotError::InvalidMode { mode: 5 })));
    }

    fn check_pivot_default_mode_camarilla(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = test_candles();

        let params = PivotParams { mode: None };
        let input = PivotInput::from_candles(&candles, params);
        let result = pivot_with_kernel(&input, kernel)?;

        assert_eq!(result.r4.len(), candles.close.len());
        assert_eq!(result.r3.len(), candles.close.len());
        assert_eq!(result.r2.len(), candles.close.len());
        assert_eq!(result.r1.len(), candles.close.len());
        assert_eq!(result.pp.len(), candles.close.len());
        assert_eq!(result.s1.len(), candles.close.len());
        assert_eq!(result.s2.len(), candles.close.len());
        assert_eq!(result.s3.len(), candles.close.len());
        assert_eq!(result.s4.len(), candles.close.len());

        let start = result.r4.len().saturating_sub(5).max(1);
        for i in start..result.r4.len() {
            let range = candles.high[i - 1] - candles.low[i - 1];
            let expected = candles.close[i - 1] + 1.1 * range / 2.0;
            assert_eq!(
                result.r4[i].to_bits(),
                expected.to_bits(),
                "Camarilla R4 must be sourced from the previous period at index {i}"
            );
        }
        Ok(())
    }

    fn check_pivot_nan_values(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, f64::NAN, 30.0];
        let low = [9.0, 8.5, f64::NAN];
        let close = [9.5, 9.0, 29.0];
        let open = [9.1, 8.8, 28.5];

        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel)?;
        assert_eq!(result.pp.len(), high.len());
        Ok(())
    }

    fn check_pivot_no_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let open: [f64; 0] = [];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("One or more required fields"),
                "Expected 'EmptyData' error, got: {}",
                e
            );
        }
        Ok(())
    }

    fn check_pivot_all_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let open = [f64::NAN, f64::NAN];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'AllValuesNaN' error, got: {}",
                e
            );
        }
        Ok(())
    }

    fn check_pivot_fibonacci_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = test_candles();
        let params = PivotParams { mode: Some(1) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r3.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_standard_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = test_candles();
        let params = PivotParams { mode: Some(0) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r2.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_demark_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = test_candles();
        let params = PivotParams { mode: Some(2) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r1.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_woodie_mode(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = test_candles();
        let params = PivotParams { mode: Some(4) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r4.len(), candles.close.len());
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_pivot_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (10usize..=200).prop_flat_map(|len| {
            prop_oneof![
                prop::collection::vec(
                    (100f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                    len,
                )
                .prop_flat_map(move |base_prices| {
                    let ohlc_strat = prop::collection::vec(
                        (0f64..1f64, 0f64..1f64, 0f64..1f64, 0f64..1f64),
                        len,
                    );

                    (ohlc_strat, 0usize..=4).prop_map(move |(factors, mode)| {
                        let mut high_data = Vec::with_capacity(len);
                        let mut low_data = Vec::with_capacity(len);
                        let mut close_data = Vec::with_capacity(len);
                        let mut open_data = Vec::with_capacity(len);

                        for (i, base) in base_prices.iter().enumerate() {
                            let (high_factor, low_factor, close_factor, open_factor) = factors[i];

                            let range = base * 0.1;
                            let low = base - range * low_factor;
                            let high = base + range * high_factor;
                            let open = low + (high - low) * open_factor;
                            let close = low + (high - low) * close_factor;

                            high_data.push(high);
                            low_data.push(low);
                            open_data.push(open);
                            close_data.push(close);
                        }

                        (high_data, low_data, close_data, open_data, mode)
                    })
                }),
                (100f64..1000f64, 0usize..=4).prop_map(move |(price, mode)| {
                    let data = vec![price; len];
                    (data.clone(), data.clone(), data.clone(), data, mode)
                }),
                (100f64..1000f64, 0usize..=4).prop_map(move |(base, mode)| {
                    let mut high_data = Vec::with_capacity(len);
                    let mut low_data = Vec::with_capacity(len);
                    let mut close_data = Vec::with_capacity(len);
                    let mut open_data = Vec::with_capacity(len);

                    for _ in 0..len {
                        let epsilon = 1e-10;
                        let low = base;
                        let high = base + epsilon;
                        let open = base + epsilon * 0.3;
                        let close = base + epsilon * 0.7;

                        high_data.push(high);
                        low_data.push(low);
                        open_data.push(open);
                        close_data.push(close);
                    }

                    (high_data, low_data, close_data, open_data, mode)
                }),
            ]
        });

        let mut runner = proptest::test_runner::TestRunner::new(proptest::test_runner::Config {
            failure_persistence: None,
            ..proptest::test_runner::Config::default()
        });
        runner.run(&strat, |(high, low, close, open, mode)| {
            let params = PivotParams { mode: Some(mode) };
            let input = PivotInput::from_slices(&high, &low, &close, &open, params);

            let output = pivot_with_kernel(&input, kernel)?;
            let ref_output = pivot_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(output.pp.len(), high.len());
            prop_assert_eq!(output.r1.len(), high.len());
            prop_assert_eq!(output.s1.len(), high.len());

            let output_levels = [
                &output.r4, &output.r3, &output.r2, &output.r1, &output.pp, &output.s1, &output.s2,
                &output.s3, &output.s4,
            ];
            let reference_levels = [
                &ref_output.r4,
                &ref_output.r3,
                &ref_output.r2,
                &ref_output.r1,
                &ref_output.pp,
                &ref_output.s1,
                &ref_output.s2,
                &ref_output.s3,
                &ref_output.s4,
            ];
            let defined = match mode {
                0 | 3 | 4 => [true; N_LEVELS],
                1 => [false, true, true, true, true, true, true, true, false],
                2 => [false, false, false, true, true, true, false, false, false],
                _ => unreachable!(),
            };

            for i in 0..high.len() {
                let valid = pivot_inputs_valid_at(&high, &low, &close, &open, mode, i);
                for level in 0..N_LEVELS {
                    let actual = output_levels[level][i];
                    let reference = reference_levels[level][i];
                    prop_assert_eq!(
                        actual.to_bits(),
                        reference.to_bits(),
                        "kernel mismatch at output {}, index {}",
                        level,
                        i
                    );
                    if valid && defined[level] {
                        prop_assert!(
                            actual.is_finite(),
                            "defined output {} is non-finite at index {}",
                            level,
                            i
                        );
                    } else {
                        prop_assert!(
                            actual.is_nan(),
                            "undefined output {} became numeric at index {}",
                            level,
                            i
                        );
                    }

                    #[cfg(debug_assertions)]
                    if !actual.is_nan() {
                        let bits = actual.to_bits();
                        prop_assert_ne!(bits, 0x11111111_11111111);
                        prop_assert_ne!(bits, 0x22222222_22222222);
                        prop_assert_ne!(bits, 0x33333333_33333333);
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_pivot_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let candles = test_candles();

        let test_params = vec![
            PivotParams::default(),
            PivotParams { mode: Some(0) },
            PivotParams { mode: Some(1) },
            PivotParams { mode: Some(2) },
            PivotParams { mode: Some(3) },
            PivotParams { mode: Some(4) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = PivotInput::from_candles(&candles, params.clone());
            let output = pivot_with_kernel(&input, kernel)?;

            let arrays = vec![
                ("r4", &output.r4),
                ("r3", &output.r3),
                ("r2", &output.r2),
                ("r1", &output.r1),
                ("pp", &output.pp),
                ("s1", &output.s1),
                ("s2", &output.s2),
                ("s3", &output.s3),
                ("s4", &output.s4),
            ];

            for (array_name, values) in arrays {
                for (i, &val) in values.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in array {} with params: {:?} (param set {})",
                            test_name, val, bits, i, array_name, params, param_idx
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in array {} with params: {:?} (param set {})",
                            test_name, val, bits, i, array_name, params, param_idx
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in array {} with params: {:?} (param set {})",
                            test_name, val, bits, i, array_name, params, param_idx
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_pivot_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_pivot_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() { $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar).unwrap(); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() { $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2).unwrap(); }
                    #[test]
                    fn [<$test_fn _avx512>]() { $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512).unwrap(); }
                )*
                $(
                    #[test]
                    fn [<$test_fn _auto_detect>]() { $test_fn(stringify!([<$test_fn _auto_detect>]), Kernel::Auto).unwrap(); }
                )*
            }
        }
    }

    generate_all_pivot_tests!(
        check_pivot_default_mode_camarilla,
        check_pivot_nan_values,
        check_pivot_no_data,
        check_pivot_all_nan,
        check_pivot_fibonacci_mode,
        check_pivot_standard_mode,
        check_pivot_demark_mode,
        check_pivot_woodie_mode,
        check_pivot_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_pivot_tests!(check_pivot_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let candles = test_candles();

        let output = PivotBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;

        let def = PivotParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| p.mode == def.mode)
            .expect("default row missing");
        let levels = &output.levels[row];

        for arr in levels.iter() {
            assert_eq!(arr.len(), candles.close.len());
        }

        let r4 = &levels[0];
        let start = r4.len().saturating_sub(5).max(1);
        for i in start..r4.len() {
            let range = candles.high[i - 1] - candles.low[i - 1];
            let expected = candles.close[i - 1] + 1.1 * range / 2.0;
            assert_eq!(
                r4[i].to_bits(),
                expected.to_bits(),
                "[{test}] Camarilla R4 must use the previous period at index {i}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let c = test_candles();

        let test_configs = vec![
            (0, 2, 1),
            (0, 4, 1),
            (0, 4, 2),
            (1, 3, 1),
            (3, 4, 1),
            (2, 2, 1),
            (0, 0, 1),
        ];

        for (cfg_idx, &(mode_start, mode_end, mode_step)) in test_configs.iter().enumerate() {
            let output = PivotBatchBuilder::new()
                .kernel(kernel)
                .mode_range(mode_start, mode_end, mode_step)
                .apply_candles(&c)?;

            for (row_idx, levels) in output.levels.iter().enumerate() {
                let combo = &output.combos[row_idx];

                for (level_idx, level_array) in levels.iter().enumerate() {
                    let level_name = match level_idx {
                        0 => "r4",
                        1 => "r3",
                        2 => "r2",
                        3 => "r1",
                        4 => "pp",
                        5 => "s1",
                        6 => "s2",
                        7 => "s3",
                        8 => "s4",
                        _ => "unknown",
                    };

                    for (col, &val) in level_array.iter().enumerate() {
                        if val.is_nan() {
                            continue;
                        }

                        let bits = val.to_bits();

                        if bits == 0x11111111_11111111 {
                            panic!(
                                "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
								 at row {} col {} in array {} with params: {:?}",
                                test, cfg_idx, val, bits, row_idx, col, level_name, combo
                            );
                        }

                        if bits == 0x22222222_22222222 {
                            panic!(
                                "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
								 at row {} col {} in array {} with params: {:?}",
                                test, cfg_idx, val, bits, row_idx, col, level_name, combo
                            );
                        }

                        if bits == 0x33333333_33333333 {
                            panic!(
                                "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
								 at row {} col {} in array {} with params: {:?}",
                                test, cfg_idx, val, bits, row_idx, col, level_name, combo
                            );
                        }
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

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch).unwrap();
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch).unwrap();
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch).unwrap();
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto).unwrap();
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_pivot_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let candles = test_candles();
        let params = PivotParams::default();
        let input = PivotInput::from_candles(&candles, params);

        let base = pivot(&input)?;

        let len = candles.close.len();

        let mut r4 = vec![0.0; len];
        let mut r3 = vec![0.0; len];
        let mut r2 = vec![0.0; len];
        let mut r1 = vec![0.0; len];
        let mut pp = vec![0.0; len];
        let mut s1 = vec![0.0; len];
        let mut s2 = vec![0.0; len];
        let mut s3 = vec![0.0; len];
        let mut s4 = vec![0.0; len];

        {
            pivot_into(
                &input, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3,
                &mut s4,
            )?;

            assert_eq!(r4.len(), base.r4.len());
            assert_eq!(r3.len(), base.r3.len());
            assert_eq!(r2.len(), base.r2.len());
            assert_eq!(r1.len(), base.r1.len());
            assert_eq!(pp.len(), base.pp.len());
            assert_eq!(s1.len(), base.s1.len());
            assert_eq!(s2.len(), base.s2.len());
            assert_eq!(s3.len(), base.s3.len());
            assert_eq!(s4.len(), base.s4.len());

            fn eq_or_both_nan(a: f64, b: f64) -> bool {
                (a.is_nan() && b.is_nan()) || (a == b)
            }

            for i in 0..len {
                assert!(eq_or_both_nan(r4[i], base.r4[i]), "r4 mismatch at {i}");
                assert!(eq_or_both_nan(r3[i], base.r3[i]), "r3 mismatch at {i}");
                assert!(eq_or_both_nan(r2[i], base.r2[i]), "r2 mismatch at {i}");
                assert!(eq_or_both_nan(r1[i], base.r1[i]), "r1 mismatch at {i}");
                assert!(eq_or_both_nan(pp[i], base.pp[i]), "pp mismatch at {i}");
                assert!(eq_or_both_nan(s1[i], base.s1[i]), "s1 mismatch at {i}");
                assert!(eq_or_both_nan(s2[i], base.s2[i]), "s2 mismatch at {i}");
                assert!(eq_or_both_nan(s3[i], base.s3[i]), "s3 mismatch at {i}");
                assert!(eq_or_both_nan(s4[i], base.s4[i]), "s4 mismatch at {i}");
            }
        }

        Ok(())
    }
}
