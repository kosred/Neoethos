use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

/// Versioned f64 identity. This is a bounded faithful-rounding contract, not a
/// claim of universally correctly-rounded binary64 arithmetic.
pub const EPMA_F64_AUTHORITY_V1: &str = "epma/jesse-period-minus-one-ramp/c0=2-offset/period-max=260/absolute-1024-segments/pow2-scaled-dd-rolling-dot2-fallback/compensated-quotient/bounded-faithful/f64/v1";
pub const EPMA_F64_MAX_CERTIFIED_PERIOD_V1: usize = 260;
pub const EPMA_F64_SEGMENT_OUTPUTS_V1: usize = 1024;
pub const EPMA_F64_CONDITION_REBASE_V1: f64 = 0.03125;
pub const EPMA_F64_FALLBACK_CONDITION_V1: f64 = 2.328_306_436_538_696_3e-10;
const EPMA_CANONICAL_QNAN_V1: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

impl<'a> AsRef<[f64]> for EpmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EpmaData::Slice(slice) => slice,
            EpmaData::Candles { candles, source } => epma_source(candles, source),
        }
    }
}

#[inline(always)]
fn epma_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "close" => &candles.close,
        "volume" => &candles.volume,
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum EpmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EpmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EpmaParams {
    pub period: Option<usize>,
    pub offset: Option<usize>,
}
impl Default for EpmaParams {
    fn default() -> Self {
        Self {
            period: Some(11),
            offset: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpmaInput<'a> {
    pub data: EpmaData<'a>,
    pub params: EpmaParams,
}

impl<'a> EpmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EpmaParams) -> Self {
        Self {
            data: EpmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EpmaParams) -> Self {
        Self {
            data: EpmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EpmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(11)
    }
    #[inline]
    pub fn get_offset(&self) -> usize {
        self.params.offset.unwrap_or(4)
    }
}

#[derive(Debug, Error)]
pub enum EpmaError {
    #[error("epma: Input data slice is empty.")]
    EmptyInputData,

    #[error("epma: All values are non-finite.")]
    AllValuesNaN,

    #[error("epma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("epma: f64 period {period} exceeds bounded-faithful v1 maximum {maximum}")]
    UncertifiedF64Period { period: usize, maximum: usize },

    #[error("epma: Invalid offset: {offset}")]
    InvalidOffset { offset: usize },

    #[error("epma: singular or non-finite integer weight sum for period={period}, offset={offset}")]
    InvalidWeightSum { period: usize, offset: usize },

    #[error("epma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("epma: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("epma: Invalid kernel for batch operation: got {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("epma: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("epma: size overflow computing rows*cols: rows={rows}, cols={cols}")]
    SizeOverflow { rows: usize, cols: usize },
}

#[inline(always)]
fn epma_weight_sum_f64(period: usize, offset: usize) -> Result<f64, EpmaError> {
    let width = period
        .checked_sub(1)
        .and_then(|value| i128::try_from(value).ok())
        .ok_or(EpmaError::InvalidWeightSum { period, offset })?;
    let offset_i128 =
        i128::try_from(offset).map_err(|_| EpmaError::InvalidWeightSum { period, offset })?;
    let c0 = 2_i128
        .checked_sub(offset_i128)
        .ok_or(EpmaError::InvalidWeightSum { period, offset })?;
    let twice_sum = width
        .checked_mul(
            c0.checked_mul(2)
                .and_then(|value| value.checked_add(width - 1))
                .ok_or(EpmaError::InvalidWeightSum { period, offset })?,
        )
        .ok_or(EpmaError::InvalidWeightSum { period, offset })?;
    if twice_sum & 1 != 0 {
        return Err(EpmaError::InvalidWeightSum { period, offset });
    }
    let integer_sum = twice_sum / 2;
    let sum = integer_sum as f64;
    if integer_sum == 0 || !sum.is_finite() {
        return Err(EpmaError::InvalidWeightSum { period, offset });
    }
    Ok(sum)
}

#[derive(Copy, Clone, Debug)]
pub struct EpmaBuilder {
    period: Option<usize>,
    offset: Option<usize>,
    kernel: Kernel,
}
impl Default for EpmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            offset: None,
            kernel: Kernel::Auto,
        }
    }
}
impl EpmaBuilder {
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
    pub fn offset(mut self, o: usize) -> Self {
        self.offset = Some(o);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EpmaOutput, EpmaError> {
        let p = EpmaParams {
            period: self.period,
            offset: self.offset,
        };
        let i = EpmaInput::from_candles(c, "close", p);
        epma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EpmaOutput, EpmaError> {
        let p = EpmaParams {
            period: self.period,
            offset: self.offset,
        };
        let i = EpmaInput::from_slice(d, p);
        epma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EpmaStream, EpmaError> {
        let p = EpmaParams {
            period: self.period,
            offset: self.offset,
        };
        EpmaStream::try_new(p)
    }
}

#[inline]
pub fn epma(input: &EpmaInput) -> Result<EpmaOutput, EpmaError> {
    epma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn epma_prepare<'a>(
    input: &'a EpmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, Kernel), EpmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EpmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(EpmaError::AllValuesNaN)?;
    let period = input.get_period();
    let offset = input.get_offset();

    if offset >= period {
        return Err(EpmaError::InvalidOffset { offset });
    }

    if period < 2 || period > len {
        return Err(EpmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period > EPMA_F64_MAX_CERTIFIED_PERIOD_V1 {
        return Err(EpmaError::UncertifiedF64Period {
            period,
            maximum: EPMA_F64_MAX_CERTIFIED_PERIOD_V1,
        });
    }
    epma_weight_sum_f64(period, offset)?;
    let needed = period + offset + 1;
    if (len - first) < needed {
        return Err(EpmaError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let warmup = first + period + offset + 1;

    Ok((data, period, offset, first, warmup, chosen))
}

#[inline(always)]
fn epma_compute_into(
    data: &[f64],
    period: usize,
    offset: usize,
    first: usize,
    _kernel: Kernel,
    out: &mut [f64],
) {
    // The f64 identity is intentionally operation-scheduled.  SIMD
    // reassociation would make Auto/AVX, stream, and strict CUDA disagree.
    epma_scalar(data, period, offset, first, out);
}

pub fn epma_with_kernel(input: &EpmaInput, kernel: Kernel) -> Result<EpmaOutput, EpmaError> {
    let (data, period, offset, first, warmup, chosen) = epma_prepare(input, kernel)?;

    let mut out = alloc_with_nan_prefix(data.len(), warmup);
    out[..warmup].fill(EPMA_CANONICAL_QNAN_V1);
    epma_compute_into(data, period, offset, first, chosen, &mut out);

    Ok(EpmaOutput { values: out })
}

#[inline]
pub fn epma_into(input: &EpmaInput, out: &mut [f64]) -> Result<(), EpmaError> {
    let (data, period, offset, first, warmup, chosen) = epma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(EpmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let w = warmup.min(out.len());
    for v in &mut out[..w] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    epma_compute_into(data, period, offset, first, chosen, out);
    Ok(())
}

#[inline]
pub fn epma_into_slice(dst: &mut [f64], input: &EpmaInput, kern: Kernel) -> Result<(), EpmaError> {
    let (data, period, offset, first, warmup, chosen) = epma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(EpmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    epma_compute_into(data, period, offset, first, chosen, dst);

    for v in &mut dst[..warmup] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    Ok(())
}

#[inline(always)]
pub fn epma_scalar(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period < 2 || offset >= period || period > EPMA_F64_MAX_CERTIFIED_PERIOD_V1 {
        let warmup = first_valid
            .saturating_add(period)
            .saturating_add(offset)
            .saturating_add(1)
            .min(out.len());
        out[warmup..].fill(EPMA_CANONICAL_QNAN_V1);
        return;
    }
    let Some(width) = period.checked_sub(1) else {
        return;
    };
    let Ok(weight_sum) = epma_weight_sum_f64(period, offset) else {
        let warmup = first_valid
            .saturating_add(period)
            .saturating_add(offset)
            .saturating_add(1)
            .min(out.len());
        out[warmup..].fill(EPMA_CANONICAL_QNAN_V1);
        return;
    };
    let warmup = first_valid
        .saturating_add(period)
        .saturating_add(offset)
        .saturating_add(1)
        .min(data.len());
    let Ok(offset_i128) = i128::try_from(offset) else {
        out[warmup..].fill(EPMA_CANONICAL_QNAN_V1);
        return;
    };
    let c0 = 2_i128 - offset_i128;

    let first_segment = warmup / EPMA_F64_SEGMENT_OUTPUTS_V1;
    let segment_count = data.len().div_ceil(EPMA_F64_SEGMENT_OUTPUTS_V1);
    for segment in first_segment..segment_count {
        let segment_start = segment * EPMA_F64_SEGMENT_OUTPUTS_V1;
        let output_start = warmup.max(segment_start);
        let output_end = data
            .len()
            .min(segment_start.saturating_add(EPMA_F64_SEGMENT_OUTPUTS_V1));
        if output_start < output_end {
            epma_f64_segment_v1(data, width, c0, weight_sum, output_start, output_end, out);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EpmaDdV1 {
    hi: f64,
    lo: f64,
}

#[inline(always)]
fn epma_two_sum_v1(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let recovered = sum - a;
    let error = (a - (sum - recovered)) + (b - recovered);
    (sum, error)
}

impl EpmaDdV1 {
    #[inline(always)]
    fn add(&mut self, value: f64) {
        let (sum, error) = epma_two_sum_v1(self.hi, value);
        let (tail, tail_error) = epma_two_sum_v1(error, self.lo);
        let (hi, lo) = epma_two_sum_v1(sum, tail);
        let (hi, lo) = epma_two_sum_v1(hi, lo + tail_error);
        self.hi = hi;
        self.lo = lo;
    }

    #[inline(always)]
    fn add_dd(&mut self, other: Self, sign: f64) {
        self.add(sign * other.hi);
        self.add(sign * other.lo);
    }

    #[inline(always)]
    fn add_product(&mut self, a: f64, b: f64) {
        let product = a * b;
        self.add(product);
        self.add(a.mul_add(b, -product));
    }

    #[inline(always)]
    fn scale(&mut self, ratio: f64) -> bool {
        let hi = self.hi * ratio;
        let lo = self.lo * ratio;
        if !hi.is_finite()
            || !lo.is_finite()
            || (self.hi != 0.0 && hi == 0.0)
            || (self.lo != 0.0 && lo == 0.0)
        {
            return false;
        }
        self.hi = hi;
        self.lo = lo;
        true
    }

    #[inline(always)]
    fn value(self) -> f64 {
        self.hi + self.lo
    }

    #[inline(always)]
    fn magnitude(self) -> f64 {
        self.hi.abs() + self.lo.abs()
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
}

#[derive(Clone, Copy, Debug)]
struct EpmaF64StateV1 {
    scale: f64,
    minimum_nonzero_abs: f64,
    sum: EpmaDdV1,
    weighted: EpmaDdV1,
}

#[inline(always)]
fn epma_floor_power_of_two_v1(value: f64) -> f64 {
    let magnitude_bits = value.abs().to_bits();
    let exponent_bits = magnitude_bits & 0x7ff0_0000_0000_0000;
    if exponent_bits != 0 {
        f64::from_bits(exponent_bits)
    } else if magnitude_bits == 0 {
        1.0
    } else {
        f64::from_bits(1_u64 << (63 - magnitude_bits.leading_zeros()))
    }
}

#[inline(always)]
fn epma_compensated_quotient_v1(numerator: EpmaDdV1, denominator: f64) -> Option<f64> {
    let quotient = numerator.hi / denominator;
    let numerator_nonzero = numerator.hi != 0.0 || numerator.lo != 0.0;
    if !quotient.is_finite() || quotient.is_subnormal() || (numerator_nonzero && quotient == 0.0) {
        return None;
    }
    let product_remainder = (-quotient).mul_add(denominator, numerator.hi);
    let remainder = product_remainder + numerator.lo;
    if !product_remainder.is_finite() || !remainder.is_finite() {
        return None;
    }
    let correction = remainder / denominator;
    if !correction.is_finite()
        || (remainder != 0.0 && (correction == 0.0 || correction.is_subnormal()))
    {
        return None;
    }
    let corrected = quotient + correction;
    (corrected.is_finite() && !corrected.is_subnormal() && !(numerator_nonzero && corrected == 0.0))
        .then_some(corrected)
}

#[inline(always)]
fn epma_rescale_result_v1(numerator: EpmaDdV1, denominator: f64, scale: f64) -> Option<f64> {
    let normalized = epma_compensated_quotient_v1(numerator, denominator)?;
    let result = normalized * scale;
    if !result.is_finite() || (normalized != 0.0 && result == 0.0) {
        return None;
    }
    Some(result)
}

#[inline(always)]
fn epma_weight_abs_sum_v1(width: usize, c0: i128) -> Option<f64> {
    let mut total = 0_i128;
    for index in 0..width {
        total = total.checked_add((c0 + index as i128).abs())?;
    }
    let total = total as f64;
    total.is_finite().then_some(total)
}

#[inline(always)]
fn epma_build_f64_state_v1(
    window: &[f64],
    c0: i128,
    weight_sum: f64,
) -> Option<(EpmaF64StateV1, f64)> {
    let mut maximum_abs = 0.0_f64;
    let mut minimum_nonzero_abs = f64::INFINITY;
    for &value in window {
        if !value.is_finite() {
            return None;
        }
        let magnitude = value.abs();
        maximum_abs = maximum_abs.max(magnitude);
        if magnitude != 0.0 {
            minimum_nonzero_abs = minimum_nonzero_abs.min(magnitude);
        }
    }
    let scale = epma_floor_power_of_two_v1(maximum_abs);
    let mut sum = EpmaDdV1::default();
    let mut weighted = EpmaDdV1::default();
    let mut absolute_products = EpmaDdV1::default();
    for (index, &value) in window.iter().enumerate() {
        let normalized = value / scale;
        if value != 0.0 && !normalized.is_normal() {
            return None;
        }
        sum.add(normalized);
        let weight = (c0 + index as i128) as f64;
        let product = normalized * weight;
        if value != 0.0 && weight != 0.0 && !product.is_normal() {
            return None;
        }
        weighted.add_product(normalized, weight);
        absolute_products.add(product.abs());
    }
    if !sum.is_finite() || !weighted.is_finite() || !absolute_products.is_finite() {
        return None;
    }
    let weighted_value = weighted.value();
    let absolute_value = absolute_products.value();
    if absolute_value != 0.0
        && weighted_value.abs() <= absolute_value * EPMA_F64_FALLBACK_CONDITION_V1
    {
        return None;
    }
    let result = if absolute_value == 0.0 {
        0.0
    } else {
        epma_rescale_result_v1(weighted, weight_sum, scale)?
    };
    Some((
        EpmaF64StateV1 {
            scale,
            minimum_nonzero_abs,
            sum,
            weighted,
        },
        result,
    ))
}

#[inline(always)]
fn epma_roll_f64_state_v1(
    mut rolling: EpmaF64StateV1,
    leaving: f64,
    entering: f64,
    width: usize,
    c0: i128,
    weight_sum: f64,
    absolute_weight_sum: Option<f64>,
) -> Option<(EpmaF64StateV1, f64)> {
    if !leaving.is_finite() || !entering.is_finite() {
        return None;
    }

    let entering_abs = entering.abs();
    let entering_scale = epma_floor_power_of_two_v1(entering_abs);
    if entering_scale > rolling.scale {
        let ratio = rolling.scale / entering_scale;
        if !ratio.is_normal() || !rolling.sum.scale(ratio) || !rolling.weighted.scale(ratio) {
            return None;
        }
        rolling.scale = entering_scale;
    }
    if entering_abs != 0.0 {
        rolling.minimum_nonzero_abs = rolling.minimum_nonzero_abs.min(entering_abs);
    }

    let leaving_normalized = leaving / rolling.scale;
    let entering_normalized = entering / rolling.scale;
    let minimum_normalized = rolling.minimum_nonzero_abs / rolling.scale;
    if (rolling.minimum_nonzero_abs.is_finite() && !minimum_normalized.is_normal())
        || (leaving != 0.0 && !leaving_normalized.is_normal())
        || (entering != 0.0 && !entering_normalized.is_normal())
    {
        return None;
    }

    let previous_sum_magnitude = rolling.sum.magnitude();
    let previous_weighted_magnitude = rolling.weighted.magnitude();
    let leaving_weight = (1_i128 - c0) as f64;
    let entering_weight = (c0 + width as i128 - 1) as f64;
    let leaving_product = leaving_weight * leaving_normalized;
    let entering_product = entering_weight * entering_normalized;
    if (leaving_weight != 0.0 && leaving_normalized != 0.0 && !leaving_product.is_normal())
        || (entering_weight != 0.0 && entering_normalized != 0.0 && !entering_product.is_normal())
    {
        return None;
    }

    rolling.weighted.add_dd(rolling.sum, -1.0);
    rolling
        .weighted
        .add_product(leaving_weight, leaving_normalized);
    rolling
        .weighted
        .add_product(entering_weight, entering_normalized);
    rolling.sum.add(-leaving_normalized);
    rolling.sum.add(entering_normalized);

    let weighted_bound = previous_weighted_magnitude
        + previous_sum_magnitude
        + leaving_product.abs()
        + entering_product.abs();
    let sum_bound = previous_sum_magnitude + leaving_normalized.abs() + entering_normalized.abs();
    let fallback_bound = absolute_weight_sum.map(|sum| 2.0 * sum);
    let uncertified = !rolling.sum.is_finite()
        || !rolling.weighted.is_finite()
        || rolling.weighted.value().abs() <= weighted_bound * EPMA_F64_CONDITION_REBASE_V1
        || rolling.sum.value().abs() <= sum_bound * EPMA_F64_CONDITION_REBASE_V1
        || fallback_bound.is_none_or(|bound| {
            rolling.weighted.value().abs() <= bound * EPMA_F64_FALLBACK_CONDITION_V1
        });
    if uncertified {
        return None;
    }

    let result = epma_rescale_result_v1(rolling.weighted, weight_sum, rolling.scale)?;
    Some((rolling, result))
}

#[inline(always)]
fn epma_f64_segment_v1(
    data: &[f64],
    width: usize,
    c0: i128,
    weight_sum: f64,
    output_start: usize,
    output_end: usize,
    out: &mut [f64],
) {
    let mut state: Option<EpmaF64StateV1> = None;
    let absolute_weight_sum = epma_weight_abs_sum_v1(width, c0);
    for output_index in output_start..output_end {
        let window_start = output_index + 1 - width;
        let window = &data[window_start..=output_index];
        let must_rebuild = state.is_none() || output_index == output_start;
        if must_rebuild {
            if let Some((rebuilt, result)) = epma_build_f64_state_v1(window, c0, weight_sum) {
                state = Some(rebuilt);
                out[output_index] = result;
            } else {
                state = None;
                out[output_index] = EPMA_CANONICAL_QNAN_V1;
            }
            continue;
        }

        if let Some((rolling, result)) = epma_roll_f64_state_v1(
            state.expect("state checked above"),
            data[window_start - 1],
            data[output_index],
            width,
            c0,
            weight_sum,
            absolute_weight_sum,
        ) {
            state = Some(rolling);
            out[output_index] = result;
            continue;
        }
        if let Some((rebuilt, result)) = epma_build_f64_state_v1(window, c0, weight_sum) {
            state = Some(rebuilt);
            out[output_index] = result;
        } else {
            state = None;
            out[output_index] = EPMA_CANONICAL_QNAN_V1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn epma_simd128(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    epma_scalar(data, period, offset, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn epma_avx2(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    epma_scalar(data, period, offset, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn epma_avx512(
    data: &[f64],
    period: usize,
    offset: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    epma_scalar(data, period, offset, first_valid, out);
}
#[derive(Debug, Clone)]
pub struct EpmaStream {
    period: usize,
    offset: usize,
    width: usize,
    c0: i128,
    weight_sum: f64,
    absolute_weight_sum: f64,
    buffer: Vec<f64>,
    head: usize,
    seen: usize,
    included: usize,
    first_valid: Option<usize>,
    state: Option<EpmaF64StateV1>,
    scratch: Vec<f64>,
}

impl EpmaStream {
    pub fn try_new(params: EpmaParams) -> Result<Self, EpmaError> {
        let period = params.period.unwrap_or(11);
        let offset = params.offset.unwrap_or(4);

        if period < 2 {
            return Err(EpmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if period > EPMA_F64_MAX_CERTIFIED_PERIOD_V1 {
            return Err(EpmaError::UncertifiedF64Period {
                period,
                maximum: EPMA_F64_MAX_CERTIFIED_PERIOD_V1,
            });
        }
        if offset >= period {
            return Err(EpmaError::InvalidOffset { offset });
        }

        let width = period - 1;
        let weight_sum = epma_weight_sum_f64(period, offset)?;
        let offset_i128 =
            i128::try_from(offset).map_err(|_| EpmaError::InvalidWeightSum { period, offset })?;
        let c0 = 2_i128 - offset_i128;
        let absolute_weight_sum = epma_weight_abs_sum_v1(width, c0)
            .ok_or(EpmaError::InvalidWeightSum { period, offset })?;

        Ok(Self {
            period,
            offset,
            width,
            c0,
            weight_sum,
            absolute_weight_sum,
            buffer: vec![0.0; width],
            head: 0,
            seen: 0,
            included: 0,
            first_valid: None,
            state: None,
            scratch: vec![0.0; width],
        })
    }

    #[inline(always)]
    fn copy_chronological_window(&mut self) {
        for index in 0..self.width {
            self.scratch[index] = self.buffer[(self.head + index) % self.width];
        }
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let index = self.seen;
        self.seen = self.seen.saturating_add(1);
        if self.first_valid.is_none() {
            if !value.is_finite() {
                return Some(EPMA_CANONICAL_QNAN_V1);
            }
            self.first_valid = Some(index);
        }

        let leaving = (self.included == self.width).then_some(self.buffer[self.head]);
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.width;
        if self.included < self.width {
            self.included += 1;
        }

        let warmup = self
            .first_valid
            .expect("first_valid set above")
            .saturating_add(self.period)
            .saturating_add(self.offset)
            .saturating_add(1);
        if index < warmup || self.included < self.width {
            return Some(EPMA_CANONICAL_QNAN_V1);
        }

        if index != warmup && index % EPMA_F64_SEGMENT_OUTPUTS_V1 != 0 {
            if let (Some(state), Some(leaving)) = (self.state, leaving) {
                if let Some((rolled, result)) = epma_roll_f64_state_v1(
                    state,
                    leaving,
                    value,
                    self.width,
                    self.c0,
                    self.weight_sum,
                    Some(self.absolute_weight_sum),
                ) {
                    self.state = Some(rolled);
                    return Some(result);
                }
            }
        }

        self.copy_chronological_window();
        if let Some((rebuilt, result)) =
            epma_build_f64_state_v1(&self.scratch, self.c0, self.weight_sum)
        {
            self.state = Some(rebuilt);
            Some(result)
        } else {
            self.state = None;
            Some(EPMA_CANONICAL_QNAN_V1)
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpmaBatchRange {
    pub period: (usize, usize, usize),
    pub offset: (usize, usize, usize),
}
impl Default for EpmaBatchRange {
    fn default() -> Self {
        Self {
            period: (11, 260, 1),
            offset: (4, 4, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct EpmaBatchBuilder {
    range: EpmaBatchRange,
    kernel: Kernel,
}
impl EpmaBatchBuilder {
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
    pub fn offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.offset = (start, end, step);
        self
    }
    #[inline]
    pub fn offset_static(mut self, o: usize) -> Self {
        self.range.offset = (o, o, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<EpmaBatchOutput, EpmaError> {
        epma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EpmaBatchOutput, EpmaError> {
        EpmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EpmaBatchOutput, EpmaError> {
        let slice = epma_source(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<EpmaBatchOutput, EpmaError> {
        EpmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct EpmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EpmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EpmaBatchOutput {
    pub fn row_for_params(&self, p: &EpmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(11) == p.period.unwrap_or(11)
                && c.offset.unwrap_or(4) == p.offset.unwrap_or(4)
        })
    }
    pub fn values_for(&self, p: &EpmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EpmaBatchRange) -> Vec<EpmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 {
            return vec![start];
        }
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        (lo..=hi).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let offsets = axis_usize(r.offset);
    let mut out = Vec::with_capacity(periods.len() * offsets.len());
    for &p in &periods {
        for &o in &offsets {
            out.push(EpmaParams {
                period: Some(p),
                offset: Some(o),
            });
        }
    }
    out
}

#[inline(always)]
fn epma_validate_batch_v1(
    data: &[f64],
    combos: &[EpmaParams],
) -> Result<(usize, Vec<usize>), EpmaError> {
    if data.is_empty() {
        return Err(EpmaError::EmptyInputData);
    }
    if combos.is_empty() {
        return Err(EpmaError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }

    for combo in combos {
        let period = combo.period.unwrap();
        let offset = combo.offset.unwrap();
        if period < 2 || period > data.len() {
            return Err(EpmaError::InvalidPeriod {
                period,
                data_len: data.len(),
            });
        }
        if period > EPMA_F64_MAX_CERTIFIED_PERIOD_V1 {
            return Err(EpmaError::UncertifiedF64Period {
                period,
                maximum: EPMA_F64_MAX_CERTIFIED_PERIOD_V1,
            });
        }
        if offset >= period {
            return Err(EpmaError::InvalidOffset { offset });
        }
        epma_weight_sum_f64(period, offset)?;
    }

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(EpmaError::AllValuesNaN)?;
    let valid = data.len() - first;
    let mut warmups = Vec::with_capacity(combos.len());
    for combo in combos {
        let period = combo.period.unwrap();
        let offset = combo.offset.unwrap();
        let needed = period
            .checked_add(offset)
            .and_then(|value| value.checked_add(1))
            .ok_or(EpmaError::NotEnoughValidData {
                needed: usize::MAX,
                valid,
            })?;
        if valid < needed {
            return Err(EpmaError::NotEnoughValidData { needed, valid });
        }
        warmups.push(first + needed);
    }
    Ok((first, warmups))
}

#[inline(always)]
pub fn epma_batch_with_kernel(
    data: &[f64],
    sweep: &EpmaBatchRange,
    k: Kernel,
) -> Result<EpmaBatchOutput, EpmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EpmaError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    epma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn epma_batch_slice(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
) -> Result<EpmaBatchOutput, EpmaError> {
    epma_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn epma_batch_par_slice(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
) -> Result<EpmaBatchOutput, EpmaError> {
    epma_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn epma_batch_inner(
    data: &[f64],
    sweep: &EpmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EpmaBatchOutput, EpmaError> {
    let combos = expand_grid(sweep);
    epma_validate_batch_v1(data, &combos)?;
    let rows = combos.len();
    let cols = data.len();

    let _total_cells = rows
        .checked_mul(cols)
        .ok_or(EpmaError::SizeOverflow { rows, cols })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let combos = epma_batch_inner_into_uninit(data, sweep, kern, parallel, &mut buf_mu)?;

    let values = unsafe {
        let mut buf_guard = ManuallyDrop::new(buf_mu);
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(EpmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn epma_batch_inner_into_uninit(
    data: &[f64],
    sweep: &EpmaBatchRange,
    _kern: Kernel,
    parallel: bool,
    buf_mu: &mut [MaybeUninit<f64>],
) -> Result<Vec<EpmaParams>, EpmaError> {
    let combos = expand_grid(sweep);
    let (first, warm) = epma_validate_batch_v1(data, &combos)?;
    let cols = data.len();
    init_matrix_prefixes(buf_mu, cols, &warm);

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let offset = combos[row].offset.unwrap();
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        epma_scalar(data, period, offset, first, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            buf_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    const CANONICAL_QNAN_BITS: u64 = 0x7ff8_0000_0000_0000;

    fn reviewed_close_fixture(len: usize) -> Vec<f64> {
        const WAVE: [f64; 11] = [
            0.000041, -0.000027, 0.000013, -0.000036, 0.000022, -0.000009, 0.000033, -0.000019,
            0.000006, -0.000031, 0.000017,
        ];
        (0..len)
            .map(|row| 1.075 + row as f64 * 0.000_000_7 + WAVE[row % WAVE.len()])
            .collect()
    }

    fn period_14_offset_4() -> EpmaParams {
        EpmaParams {
            period: Some(14),
            offset: Some(4),
        }
    }

    #[test]
    fn epma_v1_source_boundary_is_period_minus_one_and_c0_two_minus_offset() {
        let period = 14_usize;
        let offset = 4_usize;
        let width = period - 1;
        let c0 = 2_i128 - offset as i128;
        let weights = (0..width)
            .map(|index| c0 + index as i128)
            .collect::<Vec<_>>();

        assert_eq!(width, 13);
        assert_eq!(weights.first(), Some(&-2));
        assert_eq!(weights.last(), Some(&10));
        assert_eq!(weights.iter().sum::<i128>(), 52);
        assert_eq!(epma_weight_sum_f64(period, offset).unwrap(), 52.0);
    }

    #[test]
    fn epma_cuda_uses_header_independent_ieee_positive_infinity() {
        let kernel = include_str!("../../../kernels/cuda/moving_averages/epma_kernel.cu");
        assert!(
            !kernel.contains("CUDART_INF"),
            "standalone cubin compilation must not depend on an undeclared CUDART_INF macro"
        );
        assert!(
            kernel.contains("__longlong_as_double(0x7ff0000000000000ULL)"),
            "EPMA must pin the exact IEEE-754 positive-infinity bits used by its minimum sentinel"
        );
    }

    #[test]
    fn epma_reviewed_row19_uses_bounded_faithful_f64_schedule() {
        let data = reviewed_close_fixture(64);
        let input = EpmaInput::from_slice(&data, period_14_offset_4());
        let output = epma_with_kernel(&input, Kernel::Scalar).expect("valid EPMA fixture");

        assert_eq!(output.values[19].to_bits(), 0x3ff1_3342_36fd_3ee8);
    }

    #[test]
    fn epma_compensated_quotient_documents_one_ulp_bounded_faithful_limit() {
        let numerator = EpmaDdV1 {
            hi: f64::from_bits(0x6850_d172_5933_e1e3),
            lo: f64::from_bits(0xe4eb_1bff_ffff_ffff),
        };
        let observed = epma_compensated_quotient_v1(numerator, 3_913.0)
            .expect("the bounded period-92 corpus cell is finite");
        let exact_rational_rn_bits = 0x6791_9acc_e07b_6539_u64;

        assert_eq!(observed.to_bits(), 0x6791_9acc_e07b_6538);
        assert_eq!(observed.to_bits().abs_diff(exact_rational_rn_bits), 1);
    }

    #[test]
    fn epma_quotient_and_rescale_fail_closed_on_unsafe_exponents() {
        assert!(
            epma_compensated_quotient_v1(
                EpmaDdV1 {
                    hi: f64::MAX,
                    lo: 0.0,
                },
                0.5,
            )
            .is_none()
        );
        assert!(
            epma_compensated_quotient_v1(
                EpmaDdV1 {
                    hi: f64::MAX,
                    lo: f64::from_bits(1),
                },
                f64::MAX,
            )
            .is_none()
        );
        assert!(
            epma_compensated_quotient_v1(
                EpmaDdV1 {
                    hi: 1.0,
                    lo: f64::MAX,
                },
                f64::MIN_POSITIVE,
            )
            .is_none()
        );
        assert!(epma_rescale_result_v1(EpmaDdV1 { hi: 2.0, lo: 0.0 }, 1.0, f64::MAX,).is_none());
        assert_eq!(
            epma_rescale_result_v1(EpmaDdV1 { hi: 1.0, lo: 0.0 }, 1.0, f64::from_bits(1),)
                .expect("a representable subnormal result is valid")
                .to_bits(),
            1
        );
    }

    #[test]
    fn epma_rejects_every_singular_weight_sum_surface() {
        let data = reviewed_close_fixture(64);
        let singular = EpmaParams {
            period: Some(6),
            offset: Some(4),
        };
        assert!(epma(&EpmaInput::from_slice(&data, singular.clone())).is_err());
        assert!(EpmaStream::try_new(singular).is_err());

        let range = EpmaBatchRange {
            period: (6, 6, 0),
            offset: (4, 4, 0),
        };
        assert!(epma_batch_slice(&data, &range, Kernel::Scalar).is_err());
    }

    #[test]
    fn epma_rejects_periods_outside_the_v1_accuracy_corpus() {
        let data = reviewed_close_fixture(400);
        let params = EpmaParams {
            period: Some(261),
            offset: Some(4),
        };
        assert!(matches!(
            epma(&EpmaInput::from_slice(&data, params.clone())),
            Err(EpmaError::UncertifiedF64Period {
                period: 261,
                maximum: 260
            })
        ));
        assert!(matches!(
            EpmaStream::try_new(params),
            Err(EpmaError::UncertifiedF64Period {
                period: 261,
                maximum: 260
            })
        ));
        let range = EpmaBatchRange {
            period: (261, 261, 0),
            offset: (4, 4, 0),
        };
        assert!(matches!(
            epma_batch_slice(&data, &range, Kernel::Scalar),
            Err(EpmaError::UncertifiedF64Period {
                period: 261,
                maximum: 260
            })
        ));
    }

    #[test]
    fn epma_finite_first_valid_and_internal_gaps_recover_immediately() {
        let mut data = reviewed_close_fixture(80);
        data[0] = f64::NAN;
        data[1] = f64::INFINITY;
        data[30] = f64::NEG_INFINITY;
        data[45] = f64::NAN;
        let params = EpmaParams {
            period: Some(7),
            offset: Some(1),
        };
        let direct = epma_with_kernel(
            &EpmaInput::from_slice(&data, params.clone()),
            Kernel::Scalar,
        )
        .expect("finite data remains after the gaps")
        .values;

        assert_eq!(direct[10].to_bits(), CANONICAL_QNAN_BITS);
        assert!(
            direct[11].is_finite(),
            "warmup is relative to first finite close"
        );
        for value in &direct[30..=35] {
            assert_eq!(value.to_bits(), CANONICAL_QNAN_BITS);
        }
        assert!(direct[36].is_finite(), "the Inf left the six-value window");
        for value in &direct[45..=50] {
            assert_eq!(value.to_bits(), CANONICAL_QNAN_BITS);
        }
        assert!(direct[51].is_finite(), "the NaN left the six-value window");

        let mut stream = EpmaStream::try_new(params).expect("valid stream parameters");
        let streamed: Vec<f64> = data
            .iter()
            .map(|&value| stream.update(value).expect("EPMA emits one cell per input"))
            .collect();
        assert_eq!(
            direct
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            streamed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn epma_all_cpu_surfaces_share_exact_segment_checkpoint_bits() {
        let data = reviewed_close_fixture(2_200);
        let params = period_14_offset_4();
        let input = EpmaInput::from_slice(&data, params.clone());
        let scalar = epma_with_kernel(&input, Kernel::Scalar)
            .expect("scalar")
            .values;
        let auto = epma(&input).expect("auto").values;
        let avx2 = epma_with_kernel(&input, Kernel::Avx2)
            .expect("AVX2 identity")
            .values;
        let avx512 = epma_with_kernel(&input, Kernel::Avx512)
            .expect("AVX-512 identity")
            .values;

        let mut into = vec![0.0; data.len()];
        epma_into_slice(&mut into, &input, Kernel::Scalar).expect("into slice");
        let mut into_auto = vec![0.0; data.len()];
        epma_into(&input, &mut into_auto).expect("Auto into");

        let range = EpmaBatchRange {
            period: (14, 14, 0),
            offset: (4, 4, 0),
        };
        let batch = epma_batch_slice(&data, &range, Kernel::Scalar)
            .expect("batch")
            .values;
        let batch_auto = epma_batch_with_kernel(&data, &range, Kernel::Auto)
            .expect("Auto batch")
            .values;

        let mut stream = EpmaStream::try_new(params).expect("stream");
        let streamed: Vec<f64> = data
            .iter()
            .map(|&value| stream.update(value).expect("one output per input"))
            .collect();

        for index in [0, 18, 19, 20, 1_023, 1_024, 1_025, 2_047, 2_048, 2_049] {
            let expected = scalar[index].to_bits();
            assert_eq!(auto[index].to_bits(), expected, "Auto index {index}");
            assert_eq!(avx2[index].to_bits(), expected, "AVX2 index {index}");
            assert_eq!(avx512[index].to_bits(), expected, "AVX-512 index {index}");
            assert_eq!(into[index].to_bits(), expected, "into index {index}");
            assert_eq!(
                into_auto[index].to_bits(),
                expected,
                "Auto into index {index}"
            );
            assert_eq!(batch[index].to_bits(), expected, "batch index {index}");
            assert_eq!(
                batch_auto[index].to_bits(),
                expected,
                "Auto batch index {index}"
            );
            assert_eq!(streamed[index].to_bits(), expected, "stream index {index}");
        }
    }

    #[test]
    fn epma_fails_closed_on_uncertified_subnormal_window_then_recovers() {
        let mut data = vec![1.0; 64];
        data[7] = f64::MAX;
        data[8] = f64::from_bits(1);
        let output = epma_with_kernel(
            &EpmaInput::from_slice(&data, period_14_offset_4()),
            Kernel::Scalar,
        )
        .expect("parameters are valid")
        .values;

        assert_eq!(output[19].to_bits(), CANONICAL_QNAN_BITS);
        assert_eq!(output[20].to_bits(), CANONICAL_QNAN_BITS);
        assert!(output[21].is_finite());
    }

    #[test]
    fn epma_rejects_nonzero_result_that_underflows_during_rescale() {
        let mut data = vec![0.0; 64];
        data[10] = f64::from_bits(1);
        let output = epma_with_kernel(
            &EpmaInput::from_slice(&data, period_14_offset_4()),
            Kernel::Scalar,
        )
        .expect("parameters are valid")
        .values;

        assert_eq!(output[19].to_bits(), CANONICAL_QNAN_BITS);
        assert_eq!(output[20].to_bits(), 0);
        assert_eq!(output[21].to_bits(), CANONICAL_QNAN_BITS);
        assert_eq!(output[22].to_bits(), CANONICAL_QNAN_BITS);
        assert_eq!(output[23].to_bits(), 0);
    }

    #[test]
    fn epma_preserves_representable_constant_subnormal_output() {
        let data = vec![f64::from_bits(1); 64];
        let output = epma_with_kernel(
            &EpmaInput::from_slice(&data, period_14_offset_4()),
            Kernel::Scalar,
        )
        .expect("normalization makes the arithmetic certifiable")
        .values;

        assert_eq!(output[19].to_bits(), 1);
        assert_eq!(output[20].to_bits(), 1);
    }

    #[test]
    fn test_epma_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = Vec::with_capacity(256);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..253u32 {
            let v = (i as f64).sin() * 10.0 + (i as f64) * 0.01;
            data.push(v);
        }

        let input = EpmaInput::from_slice(&data, EpmaParams::default());

        let baseline = epma(&input)?.values;

        let mut out = vec![0.0; data.len()];
        {
            epma_into(&input, &mut out)?;
        }

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        assert_eq!(baseline.len(), out.len());
        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }

    fn check_epma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = EpmaParams {
            period: None,
            offset: None,
        };
        let input = EpmaInput::from_candles(&candles, "close", default_params);
        let output = epma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_epma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = EpmaParams::default();
        let input = EpmaInput::from_candles(&candles, "close", default_params);
        let result = epma_with_kernel(&input, kernel)?;
        let expected_last_five = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "[{}] EPMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                value,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_epma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = EpmaInput::with_default_candles(&candles);
        match input.data {
            EpmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EpmaData::Candles"),
        }
        let output = epma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_epma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EpmaParams {
            period: Some(0),
            offset: None,
        };
        let input = EpmaInput::from_slice(&input_data, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EPMA should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_epma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EpmaParams {
            period: Some(10),
            offset: None,
        };
        let input = EpmaInput::from_slice(&data_small, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EPMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_epma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EpmaParams {
            period: Some(9),
            offset: None,
        };
        let input = EpmaInput::from_slice(&single_point, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EPMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_epma_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EpmaInput::from_slice(&empty, EpmaParams::default());
        let res = epma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EpmaError::EmptyInputData)),
            "[{}] EPMA should fail with empty input",
            test_name
        );
        Ok(())
    }
    fn check_epma_invalid_offset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0];
        let params = EpmaParams {
            period: Some(3),
            offset: Some(3),
        };
        let input = EpmaInput::from_slice(&data, params);
        let res = epma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EpmaError::InvalidOffset { .. })),
            "[{}] EPMA should fail with invalid offset",
            test_name
        );
        Ok(())
    }
    fn check_epma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    (period * 2 + 10)..500,
                ),
                Just(period),
                0usize..period,
            )
        });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, offset)| {
				let params = EpmaParams {
					period: Some(period),
					offset: Some(offset),
				};
				let input = EpmaInput::from_slice(&data, params);
				if matches!(
					epma_weight_sum_f64(period, offset),
					Err(EpmaError::InvalidWeightSum { .. })
				) {
					let selected_rejected = matches!(
						epma_with_kernel(&input, kernel),
						Err(EpmaError::InvalidWeightSum { .. })
					);
					let scalar_rejected = matches!(
						epma_with_kernel(&input, Kernel::Scalar),
						Err(EpmaError::InvalidWeightSum { .. })
					);
					prop_assert!(selected_rejected, "selected kernel accepted singular EPMA");
					prop_assert!(scalar_rejected, "scalar kernel accepted singular EPMA");
					return Ok(());
				}


				let EpmaOutput { values: out } = epma_with_kernel(&input, kernel).unwrap();


				let EpmaOutput { values: ref_out } = epma_with_kernel(&input, Kernel::Scalar).unwrap();


				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);


				let warmup = first_valid + period + offset + 1;


				for i in 0..warmup.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"[{}] Expected NaN during warmup at index {}, got {}",
						test_name,
						i,
						out[i]
					);
				}


				if warmup < out.len() && data[warmup].is_finite() {

					let p1 = period - 1;
					let mut weight_sum = 0.0;
					for i in 0..p1 {
						let w = (period as i32 - i as i32 - offset as i32) as f64;
						weight_sum += w;
					}


					if weight_sum.abs() > 1e-10 {
						prop_assert!(
							!out[warmup].is_nan(),
							"[{}] Expected valid value at warmup index {}, got NaN",
							test_name,
							warmup
						);
					}
				}


				let p1 = period - 1;
				let mut weight_sum = 0.0;
				for i in 0..p1 {
					let w = (period as i32 - i as i32 - offset as i32) as f64;
					weight_sum += w;
				}

				if weight_sum.abs() > 1e-10 {

					for i in warmup..data.len() {
						let y = out[i];
						prop_assert!(
							y.is_finite(),
							"[{}] EPMA output at index {} is not finite: {} (period={}, offset={}, weight_sum={})",
							test_name,
							i,
							y,
							period,
							offset,
							weight_sum
						);
					}
				} else {


					for i in warmup..data.len() {
						let both_nan = out[i].is_nan() && ref_out[i].is_nan();
						let both_inf = out[i].is_infinite() && ref_out[i].is_infinite();
						prop_assert!(
							both_nan || both_inf,
							"[{}] With weight_sum=0, expected consistent NaN or Inf at index {} but got: kernel={}, scalar={} (period={}, offset={})",
							test_name,
							i,
							out[i],
							ref_out[i],
							period,
							offset
						);
					}
				}


				if period == 2 && offset == 0 && warmup < data.len() {

					for i in warmup..data.len() {
						if data[i].is_finite() {

							prop_assert!(
								(out[i] - data[i]).abs() < 1e-9,
								"[{}] Period=2,offset=0 mismatch at {}: got {}, expected {}",
								test_name,
								i,
								out[i],
								data[i]
							);
						}
					}
				}


				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && data.iter().any(|x| x.is_finite() && x.abs() > 1e-10) {
					let constant = *data.iter().find(|x| x.is_finite()).unwrap();

					let p1 = period - 1;
					let mut weight_sum = 0.0;
					for i in 0..p1 {
						let w = (period as i32 - i as i32 - offset as i32) as f64;
						weight_sum += w;
					}

					if weight_sum.abs() > 1e-10 {
						for i in warmup..data.len() {
							prop_assert!(
								(out[i] - constant).abs() < 1e-9,
								"[{}] Constant data mismatch at {}: got {}, expected {}",
								test_name,
								i,
								out[i],
								constant
							);
						}
					}
				}


				for i in (warmup.saturating_add(1))..data.len() {
					let y = out[i];
					let r = ref_out[i];

					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"[{}] finite/NaN mismatch at idx {}: {} vs {}",
							test_name,
							i,
							y,
							r
						);
						continue;
					}

					let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());

					let rel_error = if r.abs() > 1e-10 {
						(y - r).abs() / r.abs()
					} else {
						(y - r).abs()
					};


					prop_assert!(
						rel_error <= 1e-4 || (y - r).abs() <= 1e-9 || ulp_diff <= 100,
						"[{}] Kernel mismatch at idx {}: {} vs {} (ULP={}, rel_err={})",
						test_name,
						i,
						y,
						r,
						ulp_diff,
						rel_error
					);
				}


				if warmup + 5 < data.len() {

					let p1 = period - 1;
					let mut weights = Vec::with_capacity(p1);
					let mut weight_sum = 0.0;
					for i in 0..p1 {
						let w = (period as i32 - i as i32 - offset as i32) as f64;
						weights.push(w);
						weight_sum += w;
					}


					if weight_sum.abs() > 1e-10 {

						for idx in [warmup + 1, warmup + 2, data.len() - 1].iter().copied() {
							if idx > warmup && idx < data.len() {
								let start = idx + 1 - p1;
								let mut expected_sum = 0.0;
								for i in 0..p1 {
									expected_sum += data[start + i] * weights[p1 - 1 - i];
								}
								let expected = expected_sum / weight_sum;


								if out[idx].is_finite() && expected.is_finite() {

									let tolerance = if expected.abs() > 1000.0 {
										expected.abs() * 1e-12
									} else {
										1e-9
									};
									prop_assert!(
										(out[idx] - expected).abs() < tolerance,
										"[{}] EPMA formula mismatch at {}: got {}, expected {} (diff: {})",
										test_name,
										idx,
										out[idx],
										expected,
										(out[idx] - expected).abs()
									);
								} else {

									prop_assert!(
										out[idx].is_nan() == expected.is_nan() &&
										out[idx].is_infinite() == expected.is_infinite(),
										"[{}] EPMA formula NaN/Inf mismatch at {}: got {}, expected {}",
										test_name,
										idx,
										out[idx],
										expected
									);
								}
							}
						}
					}
				}


				if offset == period - 1 && warmup < data.len() && weight_sum.abs() > 1e-10 {

					for i in warmup..data.len() {
						prop_assert!(
							out[i].is_finite(),
							"[{}] Edge case offset={} produced non-finite at {}",
							test_name,
							offset,
							i
						);
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }
    fn check_epma_invalid_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let zero_weight_cases = vec![(4, 3), (5, 3), (6, 4), (8, 6)];

        for (period, offset) in zero_weight_cases {
            let p1 = period - 1;
            let mut weight_sum = 0.0;
            for i in 0..p1 {
                let w = (period as i32 - i as i32 - offset as i32) as f64;
                weight_sum += w;
            }

            let data = vec![1.0; period * 2];
            let params = EpmaParams {
                period: Some(period),
                offset: Some(offset),
            };
            let input = EpmaInput::from_slice(&data, params);

            if weight_sum.abs() < 1e-10 {
                assert!(matches!(
                    epma_with_kernel(&input, kernel),
                    Err(EpmaError::InvalidWeightSum { .. })
                ));
                assert!(matches!(
                    epma_with_kernel(&input, Kernel::Scalar),
                    Err(EpmaError::InvalidWeightSum { .. })
                ));
            }
        }

        Ok(())
    }

    fn check_epma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = EpmaParams {
            period: Some(9),
            offset: None,
        };
        let first_input = EpmaInput::from_candles(&candles, "close", first_params);
        let first_result = epma_with_kernel(&first_input, kernel)?;
        let second_params = EpmaParams {
            period: Some(3),
            offset: None,
        };
        let second_input = EpmaInput::from_slice(&first_result.values, second_params);
        let second_result = epma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_epma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = EpmaParams {
            period: Some(11),
            offset: Some(4),
        };
        let input = EpmaInput::from_candles(&candles, "close", params.clone());
        let res = epma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
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
    fn check_epma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 11;
        let offset = 4;
        let input = EpmaInput::from_candles(
            &candles,
            "close",
            EpmaParams {
                period: Some(period),
                offset: Some(offset),
            },
        );
        let batch_output = epma_with_kernel(&input, kernel)?.values;
        let mut stream = EpmaStream::try_new(EpmaParams {
            period: Some(period),
            offset: Some(offset),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(stream_values.iter())
            .enumerate()
            .skip(period + offset + 1)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EPMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_epma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_cases = vec![
            EpmaParams::default(),
            EpmaParams {
                period: Some(2),
                offset: Some(0),
            },
            EpmaParams {
                period: Some(5),
                offset: Some(1),
            },
            EpmaParams {
                period: Some(10),
                offset: Some(3),
            },
            EpmaParams {
                period: Some(10),
                offset: Some(9),
            },
            EpmaParams {
                period: Some(20),
                offset: Some(5),
            },
            EpmaParams {
                period: Some(30),
                offset: Some(10),
            },
            EpmaParams {
                period: Some(15),
                offset: Some(14),
            },
        ];

        for params in test_cases {
            let input = EpmaInput::from_candles(&candles, "close", params.clone());
            let output = epma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params period={:?}, offset={:?}",
                        test_name, val, bits, i, params.period, params.offset
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params period={:?}, offset={:?}",
                        test_name, val, bits, i, params.period, params.offset
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params period={:?}, offset={:?}",
                        test_name, val, bits, i, params.period, params.offset
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_epma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_epma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
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
    generate_all_epma_tests!(
        check_epma_partial_params,
        check_epma_accuracy,
        check_epma_default_candles,
        check_epma_zero_period,
        check_epma_period_exceeds_length,
        check_epma_very_small_dataset,
        check_epma_empty_input,
        check_epma_invalid_offset,
        check_epma_invalid_params,
        check_epma_reinput,
        check_epma_nan_handling,
        check_epma_streaming,
        check_epma_property,
        check_epma_no_poison
    );

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_epma_simd128_correctness() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let period = 5;
        let offset = 2;

        let params = EpmaParams {
            period: Some(period),
            offset: Some(offset),
        };
        let input = EpmaInput::from_slice(&data, params);

        let mut scalar_out = vec![0.0; data.len()];
        epma_scalar(&data, period, offset, 0, &mut scalar_out);

        let simd128_output = epma_with_kernel(&input, Kernel::Scalar).unwrap();

        let warmup = period + offset + 1;
        for i in warmup..data.len() {
            assert!(
                (scalar_out[i] - simd128_output.values[i]).abs() < 1e-10,
                "SIMD128 mismatch at index {}: scalar={}, simd128={}",
                i,
                scalar_out[i],
                simd128_output.values[i]
            );
        }
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = EpmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EpmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59174.48, 59201.04, 59167.60, 59200.32, 59117.04];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
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
            ((2, 5, 1), (0, 2, 1)),
            ((10, 20, 5), (0, 19, 3)),
            ((20, 30, 2), (5, 15, 5)),
            ((15, 25, 5), (10, 14, 2)),
            ((5, 10, 1), (0, 9, 1)),
        ];

        for (period_range, offset_range) in test_configs {
            let output = EpmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_range.0, period_range.1, period_range.2)
                .offset_range(offset_range.0, offset_range.1, offset_range.2)
                .apply_candles(&c, "close")?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, offset={:?})",
                        test, val, bits, row, col, params.period, params.offset
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, offset={:?})",
                        test, val, bits, row, col, params.period, params.offset
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, offset={:?})",
                        test, val, bits, row, col, params.period, params.offset
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
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_invalid_output_len_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = EpmaParams {
            period: Some(3),
            offset: Some(1),
        };
        let input = EpmaInput::from_slice(&data, params);
        let mut wrong_size_dst = vec![0.0; 3];

        let result = epma_into_slice(&mut wrong_size_dst, &input, Kernel::Scalar);
        assert!(result.is_err());

        match result {
            Err(EpmaError::OutputLengthMismatch { expected, got }) => {
                assert_eq!(expected, 5);
                assert_eq!(got, 3);
            }
            _ => panic!("Expected OutputLengthMismatch error"),
        }
    }

    #[test]
    fn test_invalid_kernel_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sweep = EpmaBatchRange {
            period: (3, 5, 1),
            offset: (1, 2, 1),
        };

        let result = epma_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(result.is_err());

        match result {
            Err(EpmaError::InvalidKernelForBatch(k)) => {
                assert_eq!(k, Kernel::Scalar);
            }
            _ => panic!("Expected InvalidKernelForBatch error"),
        }
    }
}
