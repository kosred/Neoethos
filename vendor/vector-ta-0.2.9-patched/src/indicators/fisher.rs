use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

const FISHER_QNAN_BITS_F64_V2: u64 = 0x7ff8_0000_0000_0000;
const FISHER_RANGE_FLOOR_F64_V2: f64 = 0.001;
pub const FISHER_CUDA_F64_MAX_PERIOD_V2: usize = 1024;

// Bounded-faithful audit receipts, not a universal RN or ULP guarantee:
// FISHER_F64_V2_FIXTURE_MAX_ULP=2
// FISHER_F64_V2_FIXTURE_MAX_ABS=8.881784197001252e-16
// FISHER_F64_V2_ADVERSARIAL_MAX_ABS=1.7763568394002505e-15
// The fixture bound is against a correctly-rounded transform while retaining
// the established binary64 coefficient/floor schedule: 24,195 primary cells,
// 1,327 nonzero differences, 28 above one ULP. Exact-real normalization is a
// separate authority question and is deliberately not claimed by this v2.
// The frozen 1M-row, six-period host microbenchmark requires the O(N) deque
// route to beat the former native-log direct scan in aggregate; the design
// receipt measured 2.32x.
#[cfg(test)]
const FISHER_F64_V2_ADVERSARIAL_MAX_ABS: f64 = 1.776_356_839_400_250_5e-15;

// Sun fdlibm/OpenLibm e_log binary64 constants. The operation order below is
// deliberately literal: it is mirrored with explicit RN intrinsics in CUDA.
// Immutable authority receipt:
// commit=82e90aef0657289192efe77be89791c07dea0775
// source=https://raw.githubusercontent.com/JuliaMath/openlibm/82e90aef0657289192efe77be89791c07dea0775/src/e_log.c
// license=https://raw.githubusercontent.com/JuliaMath/openlibm/82e90aef0657289192efe77be89791c07dea0775/LICENSE.md
// sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD
const FISHER_LOG_TWO54_F64_V2: f64 = 1.801_439_850_948_198_400_00e16;
const FISHER_LOG_LN2_HI_F64_V2: f64 = 6.931_471_803_691_238_164_90e-1;
const FISHER_LOG_LN2_LO_F64_V2: f64 = 1.908_214_929_270_587_700_02e-10;
const FISHER_LOG_LG1_F64_V2: f64 = 6.666_666_666_666_735_130e-1;
const FISHER_LOG_LG2_F64_V2: f64 = 3.999_999_999_940_941_908e-1;
const FISHER_LOG_LG3_F64_V2: f64 = 2.857_142_874_366_239_149e-1;
const FISHER_LOG_LG4_F64_V2: f64 = 2.222_219_843_214_978_396e-1;
const FISHER_LOG_LG5_F64_V2: f64 = 1.818_357_216_161_805_012e-1;
const FISHER_LOG_LG6_F64_V2: f64 = 1.531_383_769_920_937_332e-1;
const FISHER_LOG_LG7_F64_V2: f64 = 1.479_819_860_511_658_591e-1;

#[inline(always)]
fn fisher_qnan_f64_v2() -> f64 {
    f64::from_bits(FISHER_QNAN_BITS_F64_V2)
}

#[inline(always)]
fn fisher_with_high_word_f64_v2(value: f64, high: u32) -> f64 {
    f64::from_bits((value.to_bits() & 0x0000_0000_ffff_ffff) | (u64::from(high) << 32))
}

#[inline(always)]
fn fisher_log_f64_v2(mut value: f64) -> Option<f64> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }

    let mut high = (value.to_bits() >> 32) as i32;
    let low = value.to_bits() as u32;
    let mut exponent = 0i32;
    if high < 0x0010_0000 {
        if (((high as u32) & 0x7fff_ffff) | low) == 0 {
            return None;
        }
        exponent -= 54;
        value *= FISHER_LOG_TWO54_F64_V2;
        high = (value.to_bits() >> 32) as i32;
    }
    if high >= 0x7ff0_0000 {
        return None;
    }

    exponent += (high >> 20) - 1023;
    high &= 0x000f_ffff;
    let normalize = (high + 0x0009_5f64) & 0x0010_0000;
    value = fisher_with_high_word_f64_v2(value, (high | (normalize ^ 0x3ff0_0000)) as u32);
    exponent += normalize >> 20;

    let fraction = value - 1.0;
    if (0x000f_ffff & (2 + high)) < 3 {
        if fraction == 0.0 {
            if exponent == 0 {
                return Some(0.0);
            }
            let exponent_f64 = f64::from(exponent);
            return Some(
                exponent_f64 * FISHER_LOG_LN2_HI_F64_V2 + exponent_f64 * FISHER_LOG_LN2_LO_F64_V2,
            );
        }
        let remainder = fraction * fraction * (0.5 - 0.333_333_333_333_333_33 * fraction);
        if exponent == 0 {
            return Some(fraction - remainder);
        }
        let exponent_f64 = f64::from(exponent);
        return Some(
            exponent_f64 * FISHER_LOG_LN2_HI_F64_V2
                - ((remainder - exponent_f64 * FISHER_LOG_LN2_LO_F64_V2) - fraction),
        );
    }

    let scaled = fraction / (2.0 + fraction);
    let exponent_f64 = f64::from(exponent);
    let square = scaled * scaled;
    let selector = (high - 0x0006_147a) | (0x0006_b851 - high);
    let fourth = square * square;
    let even = fourth
        * (FISHER_LOG_LG2_F64_V2
            + fourth * (FISHER_LOG_LG4_F64_V2 + fourth * FISHER_LOG_LG6_F64_V2));
    let odd = square
        * (FISHER_LOG_LG1_F64_V2
            + fourth
                * (FISHER_LOG_LG3_F64_V2
                    + fourth * (FISHER_LOG_LG5_F64_V2 + fourth * FISHER_LOG_LG7_F64_V2)));
    let remainder = odd + even;
    let result = if selector > 0 {
        let half_square = 0.5 * fraction * fraction;
        if exponent == 0 {
            fraction - (half_square - scaled * (half_square + remainder))
        } else {
            exponent_f64 * FISHER_LOG_LN2_HI_F64_V2
                - ((half_square
                    - (scaled * (half_square + remainder)
                        + exponent_f64 * FISHER_LOG_LN2_LO_F64_V2))
                    - fraction)
        }
    } else if exponent == 0 {
        fraction - scaled * (fraction - remainder)
    } else {
        exponent_f64 * FISHER_LOG_LN2_HI_F64_V2
            - ((scaled * (fraction - remainder) - exponent_f64 * FISHER_LOG_LN2_LO_F64_V2)
                - fraction)
    };
    result.is_finite().then_some(result)
}

#[inline(always)]
fn fisher_midpoint_f64_v2(high: f64, low: f64) -> Option<f64> {
    if !high.is_finite() || !low.is_finite() {
        return None;
    }
    let midpoint = 0.5 * (high + low);
    midpoint.is_finite().then_some(midpoint)
}

#[inline(always)]
fn fisher_first_finite_midpoint_v2(high: &[f64], low: &[f64]) -> Option<usize> {
    high.iter().zip(low).position(|(&high_value, &low_value)| {
        fisher_midpoint_f64_v2(high_value, low_value).is_some()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FisherHostAdmissionV2 {
    data_len: usize,
    first: usize,
}

#[inline(always)]
fn fisher_admit_shape_v2(high: &[f64], low: &[f64]) -> Result<usize, FisherError> {
    if high.is_empty() || low.is_empty() {
        return Err(FisherError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(FisherError::MismatchedDataLength {
            high: high.len(),
            low: low.len(),
        });
    }
    Ok(high.len())
}

#[inline(always)]
fn fisher_admit_period_v2(period: usize, data_len: usize) -> Result<(), FisherError> {
    if period == 0 || period > data_len {
        return Err(FisherError::InvalidPeriod { period, data_len });
    }
    Ok(())
}

#[inline(always)]
fn fisher_admit_finite_tail_v2(
    data_len: usize,
    first: usize,
    period: usize,
) -> Result<(), FisherError> {
    let valid = data_len.saturating_sub(first);
    if valid < period {
        return Err(FisherError::NotEnoughValidData {
            needed: period,
            valid,
        });
    }
    Ok(())
}

#[inline(always)]
fn fisher_admit_host_v2(
    high: &[f64],
    low: &[f64],
    period: usize,
) -> Result<FisherHostAdmissionV2, FisherError> {
    let data_len = fisher_admit_shape_v2(high, low)?;
    fisher_admit_period_v2(period, data_len)?;
    let first = fisher_first_finite_midpoint_v2(high, low).ok_or(FisherError::AllValuesNaN)?;
    fisher_admit_finite_tail_v2(data_len, first, period)?;
    Ok(FisherHostAdmissionV2 { data_len, first })
}

#[inline(always)]
fn fisher_raw_into_is_admitted_v2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &[f64],
    signal_out: &[f64],
) -> bool {
    let data_len = high.len();
    !high.is_empty()
        && low.len() == data_len
        && fisher_out.len() == data_len
        && signal_out.len() == data_len
        && period != 0
        && first < data_len
        && period <= data_len - first
        && period.checked_add(1).is_some()
}

#[inline(always)]
fn fisher_stream_period_is_admitted_v2(period: usize) -> bool {
    period != 0
        && period
            .checked_add(1)
            .and_then(|capacity| capacity.checked_mul(core::mem::size_of::<(f64, usize)>()))
            .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

#[inline(always)]
fn reset_finite_segment_v2(
    min_queue: &mut VecDeque<(f64, usize)>,
    max_queue: &mut VecDeque<(f64, usize)>,
    finite_bars: &mut usize,
    value1: &mut f64,
    previous_fisher: &mut f64,
) {
    min_queue.clear();
    max_queue.clear();
    *finite_bars = 0;
    *value1 = 0.0;
    *previous_fisher = 0.0;
}

#[inline(always)]
fn fisher_admit_midpoint_v2(
    min_queue: &mut VecDeque<(f64, usize)>,
    max_queue: &mut VecDeque<(f64, usize)>,
    index: usize,
    midpoint: f64,
    period: usize,
) {
    while let Some(&(last, _)) = min_queue.back() {
        if last >= midpoint {
            min_queue.pop_back();
        } else {
            break;
        }
    }
    min_queue.push_back((midpoint, index));

    while let Some(&(last, _)) = max_queue.back() {
        if last <= midpoint {
            max_queue.pop_back();
        } else {
            break;
        }
    }
    max_queue.push_back((midpoint, index));

    let start = index.saturating_add(1).saturating_sub(period);
    while min_queue.front().is_some_and(|&(_, queued)| queued < start) {
        min_queue.pop_front();
    }
    while max_queue.front().is_some_and(|&(_, queued)| queued < start) {
        max_queue.pop_front();
    }
}

#[inline(always)]
fn fisher_transition_f64_v2(
    midpoint: f64,
    minimum: f64,
    maximum: f64,
    value1: &mut f64,
    previous_fisher: &mut f64,
) -> Option<(f64, f64)> {
    let range_delta = maximum - minimum;
    if !range_delta.is_finite() {
        return None;
    }
    let range = range_delta.max(FISHER_RANGE_FLOOR_F64_V2);
    let normalized = (midpoint - minimum) / range - 0.5;
    let weighted = 0.66 * normalized;
    if !normalized.is_finite() || !weighted.is_finite() {
        return None;
    }

    let mut next_value1 = 0.67f64.mul_add(*value1, weighted);
    if !next_value1.is_finite() {
        return None;
    }
    if next_value1 > 0.99 {
        next_value1 = 0.999;
    } else if next_value1 < -0.99 {
        next_value1 = -0.999;
    }

    let numerator = 1.0 + next_value1;
    let denominator = 1.0 - next_value1;
    let ratio = numerator / denominator;
    let logarithm = fisher_log_f64_v2(ratio)?;
    let signal = *previous_fisher;
    let next_fisher = 0.5f64.mul_add(logarithm, 0.5 * signal);
    if !next_fisher.is_finite() {
        return None;
    }

    *value1 = next_value1;
    *previous_fisher = next_fisher;
    Some((next_fisher, signal))
}

#[inline]
fn fisher_f64_into_v2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    if !fisher_raw_into_is_admitted_v2(high, low, period, first, fisher_out, signal_out) {
        return;
    }
    let len = high.len();
    fisher_out.fill(fisher_qnan_f64_v2());
    signal_out.fill(fisher_qnan_f64_v2());

    let mut min_queue = VecDeque::with_capacity(period + 1);
    let mut max_queue = VecDeque::with_capacity(period + 1);
    let mut finite_bars = 0usize;
    let mut previous_fisher = 0.0f64;
    let mut value1 = 0.0f64;

    for index in first..len {
        let Some(midpoint) = fisher_midpoint_f64_v2(high[index], low[index]) else {
            reset_finite_segment_v2(
                &mut min_queue,
                &mut max_queue,
                &mut finite_bars,
                &mut value1,
                &mut previous_fisher,
            );
            continue;
        };
        fisher_admit_midpoint_v2(&mut min_queue, &mut max_queue, index, midpoint, period);
        finite_bars = finite_bars.saturating_add(1).min(period);
        if finite_bars < period {
            continue;
        }

        let minimum = min_queue
            .front()
            .map(|&(value, _)| value)
            .unwrap_or(midpoint);
        let maximum = max_queue
            .front()
            .map(|&(value, _)| value)
            .unwrap_or(midpoint);
        if let Some((fisher, signal)) = fisher_transition_f64_v2(
            midpoint,
            minimum,
            maximum,
            &mut value1,
            &mut previous_fisher,
        ) {
            fisher_out[index] = fisher;
            signal_out[index] = signal;
        } else {
            reset_finite_segment_v2(
                &mut min_queue,
                &mut max_queue,
                &mut finite_bars,
                &mut value1,
                &mut previous_fisher,
            );
        }
    }
}

impl<'a> FisherInput<'a> {
    #[inline(always)]
    pub fn as_ref(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            FisherData::Candles { candles } => (&candles.high, &candles.low),
            FisherData::Slices { high, low } => (*high, *low),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FisherData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct FisherOutput {
    pub fisher: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FisherParams {
    pub period: Option<usize>,
}

impl Default for FisherParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct FisherInput<'a> {
    pub data: FisherData<'a>,
    pub params: FisherParams,
}

impl<'a> FisherInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FisherParams) -> Self {
        Self {
            data: FisherData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn get_high_low(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            FisherData::Candles { candles } => (&candles.high, &candles.low),
            FisherData::Slices { high, low } => (*high, *low),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: FisherParams) -> Self {
        Self {
            data: FisherData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FisherParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FisherBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for FisherBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FisherBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<FisherOutput, FisherError> {
        let p = FisherParams {
            period: self.period,
        };
        let i = FisherInput::from_candles(c, p);
        fisher_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<FisherOutput, FisherError> {
        let p = FisherParams {
            period: self.period,
        };
        let i = FisherInput::from_slices(high, low, p);
        fisher_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<FisherStream, FisherError> {
        let p = FisherParams {
            period: self.period,
        };
        FisherStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum FisherError {
    #[error("fisher: Empty data provided.")]
    EmptyData,

    #[error("fisher: Empty input data.")]
    EmptyInputData,
    #[error("fisher: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fisher: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fisher: All values are NaN.")]
    AllValuesNaN,

    #[error("fisher: Invalid output length: expected = {expected}, actual = {actual}")]
    InvalidLength { expected: usize, actual: usize },

    #[error("fisher: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("fisher: Mismatched data length: high={high}, low={low}")]
    MismatchedDataLength { high: usize, low: usize },

    #[error("fisher: Invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("fisher: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline(always)]
pub fn fisher(input: &FisherInput) -> Result<FisherOutput, FisherError> {
    fisher_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn fisher_with_kernel(
    input: &FisherInput,
    kernel: Kernel,
) -> Result<FisherOutput, FisherError> {
    let (high, low) = input.get_high_low();
    let period = input.get_period();
    let FisherHostAdmissionV2 { data_len, first } = fisher_admit_host_v2(high, low, period)?;

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let warmup = first + period - 1;
    let mut fisher_vals = alloc_with_nan_prefix(data_len, warmup);
    let mut signal_vals = alloc_with_nan_prefix(data_len, warmup);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                fisher_scalar_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_scalar_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                fisher_avx2_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                fisher_avx512_into(high, low, period, first, &mut fisher_vals, &mut signal_vals)
            }
            _ => unreachable!(),
        }
    }

    Ok(FisherOutput {
        fisher: fisher_vals,
        signal: signal_vals,
    })
}

#[inline]
pub fn fisher_into(
    input: &FisherInput,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), FisherError> {
    fisher_into_slice(fisher_out, signal_out, input, Kernel::Auto)
}

#[inline]
pub fn fisher_scalar_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    fisher_f64_into_v2(high, low, period, first, fisher_out, signal_out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx512_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    fisher_scalar_into(high, low, period, first, fisher_out, signal_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx2_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) {
    fisher_scalar_into(high, low, period, first, fisher_out, signal_out)
}

#[inline]
pub fn fisher_into_slice(
    fisher_dst: &mut [f64],
    signal_dst: &mut [f64],
    input: &FisherInput,
    kern: Kernel,
) -> Result<(), FisherError> {
    let (high, low) = input.as_ref();
    let period = input.params.period.unwrap_or(9);
    let FisherHostAdmissionV2 { data_len, first } = fisher_admit_host_v2(high, low, period)?;
    if fisher_dst.len() != data_len || signal_dst.len() != data_len {
        return Err(FisherError::OutputLengthMismatch {
            expected: data_len,
            got: fisher_dst.len().min(signal_dst.len()),
        });
    }

    let chosen = if kern == Kernel::Auto {
        Kernel::Scalar
    } else {
        kern
    };

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            fisher_scalar_into(high, low, period, first, fisher_dst, signal_dst)
        }
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            fisher_scalar_into(high, low, period, first, fisher_dst, signal_dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => {
            fisher_avx2_into(high, low, period, first, fisher_dst, signal_dst)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            fisher_avx512_into(high, low, period, first, fisher_dst, signal_dst)
        }
        _ => unreachable!(),
    }

    Ok(())
}

#[inline]
pub fn fisher_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    k: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(FisherError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    fisher_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct FisherBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for FisherBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 258, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FisherBatchBuilder {
    range: FisherBatchRange,
    kernel: Kernel,
}

impl FisherBatchBuilder {
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
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<FisherBatchOutput, FisherError> {
        fisher_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<FisherBatchOutput, FisherError> {
        FisherBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<FisherBatchOutput, FisherError> {
        self.apply_slices(&c.high, &c.low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FisherBatchOutput, FisherError> {
        FisherBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct FisherBatchOutput {
    pub fisher: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<FisherParams>,
    pub rows: usize,
    pub cols: usize,
}
impl FisherBatchOutput {
    pub fn row_for_params(&self, p: &FisherParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }
    pub fn fisher_for(&self, p: &FisherParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.fisher[start..start + self.cols]
        })
    }
    pub fn signal_for(&self, p: &FisherParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.signal[start..start + self.cols]
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FisherGridShapeV2 {
    count: usize,
    max_period: usize,
}

#[inline(always)]
fn fisher_grid_shape_v2(
    r: &FisherBatchRange,
    max_count: usize,
) -> Result<FisherGridShapeV2, FisherError> {
    let (start, end, step) = r.period;
    let invalid_range = || FisherError::InvalidRange { start, end, step };

    if step == 0 || start == end {
        if start == 0 {
            return Err(FisherError::InvalidPeriod {
                period: 0,
                data_len: max_count,
            });
        }
        if max_count == 0 {
            return Err(invalid_range());
        }
        return Ok(FisherGridShapeV2 {
            count: 1,
            max_period: start,
        });
    }

    let distance = if start < end {
        end - start
    } else {
        start - end
    };
    let count = (distance / step).checked_add(1).ok_or_else(invalid_range)?;
    if count == 0 || count > max_count {
        return Err(invalid_range());
    }
    let steps = count - 1;
    let traversed = steps.checked_mul(step).ok_or_else(invalid_range)?;
    let last = if start < end {
        start.checked_add(traversed).ok_or_else(invalid_range)?
    } else {
        start.checked_sub(traversed).ok_or_else(invalid_range)?
    };
    if start == 0 || last == 0 {
        return Err(FisherError::InvalidPeriod {
            period: 0,
            data_len: max_count,
        });
    }

    Ok(FisherGridShapeV2 {
        count,
        max_period: start.max(last),
    })
}

#[inline(always)]
fn expand_grid_checked_v2(
    r: &FisherBatchRange,
    shape: FisherGridShapeV2,
) -> Result<Vec<FisherParams>, FisherError> {
    let (start, end, step) = r.period;
    let invalid_range = || FisherError::InvalidRange { start, end, step };
    let mut out = Vec::new();
    out.try_reserve_exact(shape.count)
        .map_err(|_| invalid_range())?;
    let mut period = start;
    for index in 0..shape.count {
        out.push(FisherParams {
            period: Some(period),
        });
        if index + 1 == shape.count {
            break;
        }
        period = if start < end {
            period.checked_add(step).ok_or_else(invalid_range)?
        } else {
            period.checked_sub(step).ok_or_else(invalid_range)?
        };
    }
    Ok(out)
}

#[inline(always)]
pub fn fisher_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
    fisher_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn fisher_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
    fisher_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn fisher_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FisherBatchOutput, FisherError> {
    let data_len = fisher_admit_shape_v2(high, low)?;
    let grid = fisher_grid_shape_v2(sweep, data_len)?;
    fisher_admit_period_v2(grid.max_period, data_len)?;
    let first = fisher_first_finite_midpoint_v2(high, low).ok_or(FisherError::AllValuesNaN)?;
    fisher_admit_finite_tail_v2(data_len, first, grid.max_period)?;
    let combos = expand_grid_checked_v2(sweep, grid)?;
    let rows = combos.len();
    let cols = data_len;

    let _cell_count = rows.checked_mul(cols).ok_or(FisherError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    let mut fisher_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);

    let mut warmup_periods: Vec<usize> = Vec::with_capacity(combos.len());
    for c in &combos {
        let p = c.period.unwrap_or(0);
        let warm = first
            .checked_add(p.saturating_sub(1))
            .ok_or(FisherError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            })?;
        warmup_periods.push(warm);
    }

    init_matrix_prefixes(&mut fisher_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut signal_mu, cols, &warmup_periods);

    let mut fisher_guard = core::mem::ManuallyDrop::new(fisher_mu);
    let mut signal_guard = core::mem::ManuallyDrop::new(signal_mu);

    let fisher_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(fisher_guard.as_mut_ptr() as *mut f64, fisher_guard.len())
    };
    let signal_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    let do_row = |row: usize, out_fish: &mut [f64], out_signal: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let _route_label = kern;
        fisher_f64_into_v2(high, low, period, first, out_fish, out_signal);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            fisher_slice
                .par_chunks_mut(cols)
                .zip(signal_slice.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (fish, sig))| do_row(row, fish, sig));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (fish, sig)) in fisher_slice
                .chunks_mut(cols)
                .zip(signal_slice.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, fish, sig);
            }
        }
    } else {
        for (row, (fish, sig)) in fisher_slice
            .chunks_mut(cols)
            .zip(signal_slice.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, fish, sig);
        }
    }

    let fisher = unsafe {
        Vec::from_raw_parts(
            fisher_guard.as_mut_ptr() as *mut f64,
            fisher_guard.len(),
            fisher_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    core::mem::forget(fisher_guard);
    core::mem::forget(signal_guard);

    Ok(FisherBatchOutput {
        fisher,
        signal,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn fisher_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &FisherBatchRange,
    kern: Kernel,
    parallel: bool,
    fisher_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<FisherParams>, FisherError> {
    let data_len = fisher_admit_shape_v2(high, low)?;
    let grid = fisher_grid_shape_v2(sweep, data_len)?;
    fisher_admit_period_v2(grid.max_period, data_len)?;
    let first = fisher_first_finite_midpoint_v2(high, low).ok_or(FisherError::AllValuesNaN)?;
    fisher_admit_finite_tail_v2(data_len, first, grid.max_period)?;
    let rows = grid.count;
    let cols = data_len;
    let expected = rows.checked_mul(cols).ok_or(FisherError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    if fisher_out.len() != expected || signal_out.len() != expected {
        return Err(FisherError::OutputLengthMismatch {
            expected,
            got: fisher_out.len().min(signal_out.len()),
        });
    }
    let combos = expand_grid_checked_v2(sweep, grid)?;

    for (row, combo) in combos.iter().enumerate() {
        let p = combo.period.unwrap_or(0);
        let warmup = first
            .checked_add(p.saturating_sub(1))
            .ok_or(FisherError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            })?;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            fisher_out[row_start + i] = f64::NAN;
            signal_out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_fish: &mut [f64], out_signal: &mut [f64]| {
        let period = combos[row].period.unwrap();
        let _route_label = kern;
        fisher_f64_into_v2(high, low, period, first, out_fish, out_signal);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            fisher_out
                .par_chunks_mut(cols)
                .zip(signal_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (fish, sig))| do_row(row, fish, sig));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (fish, sig)) in fisher_out
                .chunks_mut(cols)
                .zip(signal_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, fish, sig);
            }
        }
    } else {
        for (row, (fish, sig)) in fisher_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, fish, sig);
        }
    }

    Ok(combos)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx2_direct(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out_fish: &mut [f64],
    out_signal: &mut [f64],
) {
    fisher_f64_into_v2(high, low, period, first, out_fish, out_signal)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx512_direct(
    high: &[f64],
    low: &[f64],
    first: usize,
    period: usize,
    out_fish: &mut [f64],
    out_signal: &mut [f64],
) {
    fisher_f64_into_v2(high, low, period, first, out_fish, out_signal)
}

#[derive(Debug, Clone)]
pub struct FisherStream {
    period: usize,

    idx: usize,
    finite_bars: usize,

    minq: VecDeque<(f64, usize)>,
    maxq: VecDeque<(f64, usize)>,
    prev_fish: f64,
    val1: f64,
}

impl FisherStream {
    pub fn try_new(params: FisherParams) -> Result<Self, FisherError> {
        let period = params.period.unwrap_or(9);
        if !fisher_stream_period_is_admitted_v2(period) {
            return Err(FisherError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            idx: 0,
            finite_bars: 0,
            minq: VecDeque::new(),
            maxq: VecDeque::new(),
            prev_fish: 0.0,
            val1: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        let index = self.idx;
        self.idx = self.idx.saturating_add(1);
        if !high.is_finite() || !low.is_finite() {
            reset_finite_segment_v2(
                &mut self.minq,
                &mut self.maxq,
                &mut self.finite_bars,
                &mut self.val1,
                &mut self.prev_fish,
            );
            return None;
        }
        let Some(midpoint) = fisher_midpoint_f64_v2(high, low) else {
            reset_finite_segment_v2(
                &mut self.minq,
                &mut self.maxq,
                &mut self.finite_bars,
                &mut self.val1,
                &mut self.prev_fish,
            );
            return None;
        };

        fisher_admit_midpoint_v2(&mut self.minq, &mut self.maxq, index, midpoint, self.period);
        self.finite_bars = self.finite_bars.saturating_add(1).min(self.period);
        if self.finite_bars < self.period {
            return None;
        }

        let minimum = self
            .minq
            .front()
            .map(|&(value, _)| value)
            .unwrap_or(midpoint);
        let maximum = self
            .maxq
            .front()
            .map(|&(value, _)| value)
            .unwrap_or(midpoint);
        let output = fisher_transition_f64_v2(
            midpoint,
            minimum,
            maximum,
            &mut self.val1,
            &mut self.prev_fish,
        );
        if output.is_none() {
            reset_finite_segment_v2(
                &mut self.minq,
                &mut self.maxq,
                &mut self.finite_bars,
                &mut self.val1,
                &mut self.prev_fish,
            );
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_fisher_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = FisherParams { period: None };
        let input = FisherInput::from_candles(&candles, default_params);
        let output = fisher_with_kernel(&input, kernel)?;
        assert_eq!(output.fisher.len(), candles.close.len());
        Ok(())
    }

    fn check_fisher_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = FisherInput::from_candles(&candles, FisherParams::default());
        let result = fisher_with_kernel(&input, kernel)?;
        let expected_last_five_fisher = [
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
            -0.2569225042442664,
        ];
        let start = result.fisher.len().saturating_sub(5);
        for (i, &val) in result.fisher[start..].iter().enumerate() {
            let diff = (val - expected_last_five_fisher[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] Fisher {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five_fisher[i]
            );
        }
        Ok(())
    }

    fn check_fisher_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = FisherParams { period: Some(0) };
        let input = FisherInput::from_slices(&high, &low, params);
        let res = fisher_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Fisher should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_fisher_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = FisherParams { period: Some(10) };
        let input = FisherInput::from_slices(&high, &low, params);
        let res = fisher_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Fisher should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_fisher_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0];
        let low = [5.0];
        let params = FisherParams { period: Some(9) };
        let input = FisherInput::from_slices(&high, &low, params);
        let res = fisher_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Fisher should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_fisher_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let low = [5.0, 7.0, 9.0, 10.0, 13.0, 15.0];
        let first_params = FisherParams { period: Some(3) };
        let first_input = FisherInput::from_slices(&high, &low, first_params);
        let first_result = fisher_with_kernel(&first_input, kernel)?;
        let second_params = FisherParams { period: Some(3) };
        let second_input =
            FisherInput::from_slices(&first_result.fisher, &first_result.signal, second_params);
        let second_result = fisher_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.fisher.len(), second_result.fisher.len());
        assert_eq!(first_result.signal.len(), second_result.signal.len());
        Ok(())
    }

    fn check_fisher_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = FisherInput::from_candles(&candles, FisherParams::default());
        let res = fisher_with_kernel(&input, kernel)?;
        assert_eq!(res.fisher.len(), candles.close.len());
        if res.fisher.len() > 240 {
            for (i, &val) in res.fisher[240..].iter().enumerate() {
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

    fn check_fisher_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 9;
        let input = FisherInput::from_candles(
            &candles,
            FisherParams {
                period: Some(period),
            },
        );
        let batch_output = fisher_with_kernel(&input, kernel)?.fisher;

        let highs = source_type(&candles, "high");
        let lows = source_type(&candles, "low");

        let mut stream = FisherStream::try_new(FisherParams {
            period: Some(period),
        })?;
        let mut stream_fisher = Vec::with_capacity(highs.len());
        for (&h, &l) in highs.iter().zip(lows.iter()) {
            match stream.update(h, l) {
                Some((fish, _sig)) => stream_fisher.push(fish),
                None => stream_fisher.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_fisher.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_fisher.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Fisher streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_fisher_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            FisherParams::default(),
            FisherParams { period: Some(1) },
            FisherParams { period: Some(2) },
            FisherParams { period: Some(3) },
            FisherParams { period: Some(5) },
            FisherParams { period: Some(10) },
            FisherParams { period: Some(20) },
            FisherParams { period: Some(30) },
            FisherParams { period: Some(50) },
            FisherParams { period: Some(100) },
            FisherParams { period: Some(200) },
            FisherParams { period: Some(240) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = FisherInput::from_candles(&candles, params.clone());
            let output = fisher_with_kernel(&input, kernel)?;

            for (i, &val) in output.fisher.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }
            }

            for (i, &val) in output.signal.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_fisher_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_fisher_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                (100f64..10000f64, 0.01f64..0.05f64, period + 10..400)
                    .prop_flat_map(move |(base_price, volatility, data_len)| {
                        (
                            Just(base_price),
                            Just(volatility),
                            Just(data_len),
                            prop::collection::vec((-1f64..1f64), data_len),
                            prop::collection::vec(prop::bool::ANY, data_len),
                        )
                    })
                    .prop_map(
                        move |(
                            base_price,
                            volatility,
                            data_len,
                            price_changes,
                            zero_spread_flags,
                        )| {
                            let mut high = Vec::with_capacity(data_len);
                            let mut low = Vec::with_capacity(data_len);
                            let mut current_price = base_price;

                            for i in 0..data_len {
                                let change = price_changes[i] * volatility * current_price;
                                current_price = (current_price + change).max(10.0);

                                if zero_spread_flags[i] && i % 5 == 0 {
                                    high.push(current_price);
                                    low.push(current_price);
                                } else {
                                    let spread =
                                        current_price * 0.01 * (0.1 + price_changes[i].abs());
                                    high.push(current_price + spread);
                                    low.push((current_price - spread).max(10.0));
                                }
                            }

                            (high, low)
                        },
                    ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |((high, low), period)| {
            let params = FisherParams {
                period: Some(period),
            };
            let input = FisherInput::from_slices(&high, &low, params);

            let FisherOutput {
                fisher: out,
                signal: sig,
            } = fisher_with_kernel(&input, kernel)?;
            let FisherOutput {
                fisher: ref_out,
                signal: ref_sig,
            } = fisher_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(
                out.len(),
                high.len(),
                "[{}] Fisher output length mismatch",
                test_name
            );
            prop_assert_eq!(
                sig.len(),
                high.len(),
                "[{}] Signal output length mismatch",
                test_name
            );

            let mut first_valid = None;
            for i in 0..high.len() {
                if !high[i].is_nan() && !low[i].is_nan() {
                    first_valid = Some(i);
                    break;
                }
            }

            if let Some(first) = first_valid {
                let warmup_end = first + period - 1;
                for i in 0..warmup_end.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN at index {} during warmup",
                        test_name,
                        i
                    );
                    prop_assert!(
                        sig[i].is_nan(),
                        "[{}] Expected NaN at signal index {} during warmup",
                        test_name,
                        i
                    );
                }

                if warmup_end < out.len() {
                    prop_assert!(
                        !out[warmup_end].is_nan(),
                        "[{}] Expected valid value at index {} after warmup",
                        test_name,
                        warmup_end
                    );
                }
            }

            if let Some(first) = first_valid {
                let warmup_end = first + period - 1;

                for window_start in warmup_end..out.len().saturating_sub(period * 2) {
                    let window_end = (window_start + period).min(out.len());

                    let mut is_constant = true;
                    let first_hl = (high[window_start] + low[window_start]) / 2.0;

                    for i in window_start..window_end {
                        let current_hl = (high[i] + low[i]) / 2.0;
                        if (current_hl - first_hl).abs() > 0.001 * first_hl {
                            is_constant = false;
                            break;
                        }
                    }

                    if is_constant && window_end > window_start + 3 {
                        let fisher_start = out[window_start].abs();
                        let fisher_end = out[window_end - 1].abs();

                        if fisher_start > 0.1 {
                            prop_assert!(
									fisher_end <= fisher_start * 1.1,
									"[{}] Fisher not trending to zero in constant period [{}, {}]: start={}, end={}",
									test_name, window_start, window_end, fisher_start, fisher_end
								);
                        }
                    }
                }
            }

            for i in 1..out.len() {
                if !out[i - 1].is_nan() && !sig[i].is_nan() {
                    prop_assert!(
                        (sig[i] - out[i - 1]).abs() < 1e-9,
                        "[{}] Signal at {} ({}) doesn't match previous Fisher ({})",
                        test_name,
                        i,
                        sig[i],
                        out[i - 1]
                    );
                }
            }

            if let Some(first) = first_valid {
                let warmup_end = first + period - 1;
                if warmup_end < out.len() && !out[warmup_end].is_nan() {
                    prop_assert!(
                        out[warmup_end].abs() < 5.0,
                        "[{}] First Fisher value {} seems incorrect (should start from zero state)",
                        test_name,
                        out[warmup_end]
                    );
                }
            }

            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];
                let s = sig[i];
                let rs = ref_sig[i];

                if y.is_nan() || r.is_nan() {
                    prop_assert_eq!(
                        y.is_nan(),
                        r.is_nan(),
                        "[{}] NaN mismatch at index {}",
                        test_name,
                        i
                    );
                    continue;
                }

                if s.is_nan() || rs.is_nan() {
                    prop_assert_eq!(
                        s.is_nan(),
                        rs.is_nan(),
                        "[{}] Signal NaN mismatch at index {}",
                        test_name,
                        i
                    );
                    continue;
                }

                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let s_bits = s.to_bits();
                let rs_bits = rs.to_bits();

                let ulp_diff_fisher: u64 = y_bits.abs_diff(r_bits);
                let ulp_diff_signal: u64 = s_bits.abs_diff(rs_bits);

                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff_fisher <= 4,
                    "[{}] Fisher mismatch idx {}: {} vs {} (ULP={})",
                    test_name,
                    i,
                    y,
                    r,
                    ulp_diff_fisher
                );

                prop_assert!(
                    (s - rs).abs() <= 1e-9 || ulp_diff_signal <= 4,
                    "[{}] Signal mismatch idx {}: {} vs {} (ULP={})",
                    test_name,
                    i,
                    s,
                    rs,
                    ulp_diff_signal
                );
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_fisher_tests {
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

    generate_all_fisher_tests!(
        check_fisher_partial_params,
        check_fisher_accuracy,
        check_fisher_zero_period,
        check_fisher_period_exceeds_length,
        check_fisher_very_small_dataset,
        check_fisher_reinput,
        check_fisher_nan_handling,
        check_fisher_streaming,
        check_fisher_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_fisher_tests!(check_fisher_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = FisherBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = FisherParams::default();
        let row = output.fisher_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected_last_five = [
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
            -0.2569225042442664,
        ];
        let start = row.len().saturating_sub(5);
        for (i, &val) in row[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {val} vs {expected_last_five:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (1, 10, 1),
            (2, 20, 2),
            (5, 50, 5),
            (10, 100, 10),
            (20, 240, 20),
            (9, 9, 0),
            (50, 200, 50),
            (1, 5, 1),
            (100, 240, 40),
            (3, 30, 3),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = FisherBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.fisher.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9)
                    );
                }
            }

            for (idx, &val) in output.signal.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in signal output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in signal output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in signal output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9)
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

    #[test]
    fn check_batch_kernel_dispatch() -> Result<(), Box<dyn Error>> {
        let high = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0];
        let low = vec![5.0, 7.0, 9.0, 10.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0];
        let sweep = FisherBatchRange { period: (3, 5, 1) };

        let scalar_result = fisher_batch_slice(&high, &low, &sweep, Kernel::Scalar)?;

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            let avx2_result = fisher_batch_slice(&high, &low, &sweep, Kernel::Avx2)?;

            for i in 0..scalar_result.fisher.len() {
                let diff = (scalar_result.fisher[i] - avx2_result.fisher[i]).abs();
                assert!(
                    diff < 1e-10
                        || (scalar_result.fisher[i].is_nan() && avx2_result.fisher[i].is_nan()),
                    "Fisher mismatch at {}: scalar={}, avx2={}",
                    i,
                    scalar_result.fisher[i],
                    avx2_result.fisher[i]
                );
            }
        }

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            let avx512_result = fisher_batch_slice(&high, &low, &sweep, Kernel::Avx512)?;

            for i in 0..scalar_result.fisher.len() {
                let diff = (scalar_result.fisher[i] - avx512_result.fisher[i]).abs();
                assert!(
                    diff < 1e-10
                        || (scalar_result.fisher[i].is_nan() && avx512_result.fisher[i].is_nan()),
                    "Fisher mismatch at {}: scalar={}, avx512={}",
                    i,
                    scalar_result.fisher[i],
                    avx512_result.fisher[i]
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_fisher_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut ts = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            ts.push(i as i64);
            let base = 1000.0 + (i as f64) * 0.1;
            let wiggle = ((i as f64) * 0.15).sin() * 2.0;
            let h = base + 5.0 + wiggle;
            let l = base - 5.0 - 0.5 * wiggle;
            let o = base - 1.0;
            let c = base + 1.0;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(100.0 + (i % 10) as f64);
        }

        let candles = crate::utilities::data_loader::Candles::new(
            ts,
            open,
            high.clone(),
            low.clone(),
            close,
            volume,
        );
        let input = FisherInput::from_candles(&candles, FisherParams::default());

        let base = fisher(&input)?;

        let mut out_fish = vec![0.0; n];
        let mut out_sig = vec![0.0; n];

        {
            fisher_into(&input, &mut out_fish, &mut out_sig)?;
        }

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        assert_eq!(out_fish.len(), base.fisher.len());
        assert_eq!(out_sig.len(), base.signal.len());
        for i in 0..n {
            assert!(
                eq_or_both_nan(out_fish[i], base.fisher[i]),
                "fisher mismatch at {}: {} vs {}",
                i,
                out_fish[i],
                base.fisher[i]
            );
            assert!(
                eq_or_both_nan(out_sig[i], base.signal[i]),
                "signal mismatch at {}: {} vs {}",
                i,
                out_sig[i],
                base.signal[i]
            );
        }

        Ok(())
    }

    fn reviewed_fixture_v3_high_low() -> (Vec<f64>, Vec<f64>) {
        const ROWS: usize = 4_096;
        let mut high = Vec::with_capacity(ROWS);
        let mut low = Vec::with_capacity(ROWS);
        let mut close = Vec::with_capacity(ROWS);
        for row in 0..ROWS {
            let drift = row as f64 * 0.000_000_7;
            let wave = match row % 11 {
                0 => 0.000_041,
                1 => -0.000_027,
                2 => 0.000_013,
                3 => -0.000_036,
                4 => 0.000_022,
                5 => -0.000_009,
                6 => 0.000_033,
                7 => -0.000_019,
                8 => 0.000_006,
                9 => -0.000_031,
                _ => 0.000_017,
            };
            let row_open = 1.075 + drift;
            let row_close = row_open + wave;
            high.push(row_open.max(row_close) + 0.000_08 + (row % 7) as f64 * 0.000_001);
            low.push(row_open.min(row_close) - 0.000_07 - (row % 5) as f64 * 0.000_001);
            close.push(row_close);
        }
        let final_row = ROWS - 1;
        close[final_row] = f64::from_bits(close[final_row].to_bits() ^ 1);
        high[final_row] = high[final_row].max(close[final_row] + 0.000_001);
        low[final_row] = low[final_row].min(close[final_row] - 0.000_001);
        (high, low)
    }

    fn fnv1a64_f64_bits(values: &[f64]) -> u64 {
        values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
            (hash ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn fisher_log_f64_v2_matches_frozen_sun_checkpoints() {
        let cases = [
            (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
            (0x0000_0000_0000_0001, 0xc087_4385_446d_71c3),
            (0x4000_0000_0000_0000, 0x3fe6_2e42_fefa_39ef),
            (0x3fe0_0000_0000_0000, 0xbfe6_2e42_fefa_39ef),
            (0x3f90_d8b0_1d6a_1591, 0xc010_6de8_9959_7cd8),
            (0x3fa8_6023_0080_6d1d, 0xc008_5ba3_1b96_26ee),
            (0x3fb8_4482_7417_a07c, 0xc002_d928_b548_dbe4),
            (0x3fc7_6c46_f3ca_9d14, 0xbffb_2c4a_e8c3_fca8),
        ];
        for (input_bits, expected_bits) in cases {
            let actual = fisher_log_f64_v2(f64::from_bits(input_bits))
                .expect("positive finite checkpoint must be in-domain");
            assert_eq!(actual.to_bits(), expected_bits, "input=0x{input_bits:016x}");
        }
        for invalid in [0.0, -0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(fisher_log_f64_v2(invalid).is_none());
        }
    }

    #[test]
    fn fisher_f64_v2_full_fixture_hashes_close_all_host_routes() -> Result<(), Box<dyn Error>> {
        let (high, low) = reviewed_fixture_v3_high_low();
        let expected = [
            (7, 0x9b5a_e551_a162_7b03, 0xc1c3_f13f_db8b_f8ed),
            (9, 0xc84a_1bc5_292a_9042, 0xeb5c_0c24_ec94_1076),
            (21, 0xebc3_f1d4_ef69_711b, 0xe661_29f8_343f_c31b),
            (50, 0x6fc8_7727_2dbb_cbda, 0x6df4_ac1c_7959_fed4),
            (100, 0x22b1_a232_c4ef_c58e, 0x77be_302a_66fe_8c89),
            (200, 0xd1e2_51ae_216a_896e, 0x53bd_a56d_b471_2b73),
        ];

        for (period, expected_fisher, expected_signal) in expected {
            let params = FisherParams {
                period: Some(period),
            };
            let input = FisherInput::from_slices(&high, &low, params.clone());
            for kernel in [Kernel::Scalar, Kernel::Auto, Kernel::Avx2, Kernel::Avx512] {
                let output = fisher_with_kernel(&input, kernel)?;
                assert_eq!(fnv1a64_f64_bits(&output.fisher), expected_fisher);
                assert_eq!(fnv1a64_f64_bits(&output.signal), expected_signal);
            }

            let sweep = FisherBatchRange {
                period: (period, period, 0),
            };
            for kernel in [Kernel::Scalar, Kernel::Auto, Kernel::Avx2, Kernel::Avx512] {
                let output = fisher_batch_slice(&high, &low, &sweep, kernel)?;
                assert_eq!(fnv1a64_f64_bits(&output.fisher), expected_fisher);
                assert_eq!(fnv1a64_f64_bits(&output.signal), expected_signal);
            }

            let mut stream = FisherStream::try_new(params)?;
            let mut fisher_values = Vec::with_capacity(high.len());
            let mut signal_values = Vec::with_capacity(high.len());
            for (&high_value, &low_value) in high.iter().zip(&low) {
                match stream.update(high_value, low_value) {
                    Some((fisher, signal)) => {
                        fisher_values.push(fisher);
                        signal_values.push(signal);
                    }
                    None => {
                        fisher_values.push(f64::from_bits(FISHER_QNAN_BITS_F64_V2));
                        signal_values.push(f64::from_bits(FISHER_QNAN_BITS_F64_V2));
                    }
                }
            }
            assert_eq!(fnv1a64_f64_bits(&fisher_values), expected_fisher);
            assert_eq!(fnv1a64_f64_bits(&signal_values), expected_signal);
        }
        Ok(())
    }

    #[test]
    fn fisher_f64_v2_resets_holes_warmup_and_first_signal_to_positive_zero()
    -> Result<(), Box<dyn Error>> {
        let mut high: Vec<f64> = (0..20).map(|row| 2.0 + row as f64 * 0.1).collect();
        let mut low: Vec<f64> = (0..20).map(|row| 1.0 + row as f64 * 0.1).collect();
        high[5] = f64::NAN;
        low[10] = f64::INFINITY;
        high[15] = f64::MAX;
        low[15] = f64::MAX;

        let input = FisherInput::from_slices(&high, &low, FisherParams { period: Some(3) });
        let output = fisher_with_kernel(&input, Kernel::Scalar)?;
        let emitted = [2usize, 3, 4, 8, 9, 13, 14, 18, 19];
        for row in 0..high.len() {
            if emitted.contains(&row) {
                assert!(output.fisher[row].is_finite(), "row {row} must emit");
                assert!(output.signal[row].is_finite(), "row {row} signal must emit");
            } else {
                assert_eq!(output.fisher[row].to_bits(), FISHER_QNAN_BITS_F64_V2);
                assert_eq!(output.signal[row].to_bits(), FISHER_QNAN_BITS_F64_V2);
            }
        }
        for first_emission in [2usize, 8, 13, 18] {
            assert_eq!(output.signal[first_emission].to_bits(), 0);
        }

        let mut stream = FisherStream::try_new(FisherParams { period: Some(3) })?;
        for (row, (&high_value, &low_value)) in high.iter().zip(&low).enumerate() {
            let streamed = stream.update(high_value, low_value);
            if emitted.contains(&row) {
                let (fisher, signal) = streamed.expect("stream must emit on the same finite row");
                assert_eq!(fisher.to_bits(), output.fisher[row].to_bits());
                assert_eq!(signal.to_bits(), output.signal[row].to_bits());
            } else {
                assert!(streamed.is_none(), "stream row {row} must be a hole/warmup");
            }
        }
        Ok(())
    }

    #[test]
    fn fisher_f64_v2_fails_closed_on_finite_range_overflow() -> Result<(), Box<dyn Error>> {
        let half_max = f64::MAX * 0.5;
        let high = [-half_max, half_max, 1.0, 1.1, 1.2];
        let low = high;
        let input = FisherInput::from_slices(&high, &low, FisherParams { period: Some(2) });
        let output = fisher_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(output.fisher[1].to_bits(), FISHER_QNAN_BITS_F64_V2);
        assert_eq!(output.signal[1].to_bits(), FISHER_QNAN_BITS_F64_V2);
        assert_eq!(output.fisher[2].to_bits(), FISHER_QNAN_BITS_F64_V2);
        assert!(output.fisher[3].is_finite());
        assert_eq!(output.signal[3].to_bits(), 0);
        Ok(())
    }

    #[test]
    fn fisher_f64_v2_leading_infinity_is_not_admitted_as_first_valid() {
        let high = [f64::INFINITY, 2.0, 3.0, 4.0];
        let low = [1.0, 1.0, 2.0, 3.0];
        let input = FisherInput::from_slices(&high, &low, FisherParams { period: Some(4) });
        assert!(matches!(
            fisher_with_kernel(&input, Kernel::Scalar),
            Err(FisherError::NotEnoughValidData {
                needed: 4,
                valid: 3
            })
        ));
    }

    #[test]
    fn fisher_f64_v2_cancellation_bound_is_absolute_not_universal_ulp() {
        let got = f64::from_bits(0xc012_5d40_b4e7_c082);
        let exact_fixed_schedule = f64::from_bits(0xc012_5d40_b4e7_c080);
        assert_eq!(
            (got - exact_fixed_schedule).abs(),
            1.776_356_839_400_250_5e-15
        );
        assert!(
            (got - exact_fixed_schedule).abs() <= FISHER_F64_V2_ADVERSARIAL_MAX_ABS,
            "the frozen adversarial corpus is an absolute-error claim only"
        );
    }

    #[test]
    fn fisher_f64_v2_admission_precedes_scan_grid_allocation_and_writes() {
        let high = [f64::NAN];
        let low = [f64::NAN, 1.0];
        let input = FisherInput::from_slices(&high, &low, FisherParams { period: Some(0) });
        assert!(matches!(
            fisher_with_kernel(&input, Kernel::Scalar),
            Err(FisherError::MismatchedDataLength { high: 1, low: 2 })
        ));

        let all_nan = [f64::NAN; 2];
        let zero = FisherInput::from_slices(&all_nan, &all_nan, FisherParams { period: Some(0) });
        assert!(matches!(
            fisher_with_kernel(&zero, Kernel::Scalar),
            Err(FisherError::InvalidPeriod {
                period: 0,
                data_len: 2
            })
        ));

        let sweep = FisherBatchRange { period: (0, 0, 0) };
        assert!(matches!(
            fisher_batch_slice(&high, &low, &sweep, Kernel::Scalar),
            Err(FisherError::MismatchedDataLength { high: 1, low: 2 })
        ));
    }

    #[test]
    fn fisher_f64_v2_grid_is_checked_and_count_bounded_before_materialization() {
        let descending = FisherBatchRange {
            period: (usize::MAX, usize::MAX - 1, usize::MAX),
        };
        let shape = fisher_grid_shape_v2(&descending, usize::MAX)
            .expect("descending endpoint arithmetic must not overflow");
        let combos = expand_grid_checked_v2(&descending, shape)
            .expect("the one-value descending grid is valid");
        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0].period, Some(usize::MAX));

        let enormous = FisherBatchRange {
            period: (1, usize::MAX, 1),
        };
        assert!(matches!(
            fisher_grid_shape_v2(&enormous, 4_096),
            Err(FisherError::InvalidRange { .. })
        ));
        let bounded = FisherBatchRange {
            period: (1, 1_000_000, 1),
        };
        assert!(matches!(
            fisher_grid_shape_v2(&bounded, 4_096),
            Err(FisherError::InvalidRange { .. })
        ));
    }

    #[test]
    fn fisher_f64_v2_extreme_raw_and_stream_periods_fail_without_capacity_panic_or_write() {
        let stream = std::panic::catch_unwind(|| {
            FisherStream::try_new(FisherParams {
                period: Some(usize::MAX),
            })
        });
        assert!(stream.is_ok(), "extreme stream admission must not panic");
        assert!(matches!(
            stream.unwrap(),
            Err(FisherError::InvalidPeriod {
                period: usize::MAX,
                data_len: 0
            })
        ));

        let mut fisher_out = [123.0];
        let mut signal_out = [456.0];
        fisher_scalar_into(
            &[1.0],
            &[1.0],
            usize::MAX,
            0,
            &mut fisher_out,
            &mut signal_out,
        );
        assert_eq!(fisher_out, [123.0]);
        assert_eq!(signal_out, [456.0]);
    }
}
