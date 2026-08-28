use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

/// Largest period certified by the strict f64 host/CUDA authority.
pub const FWMA_F64_MAX_PERIOD: usize = 254;

/// Stable identity handed to the Classic semantic-v9 source-closure owner.
pub const FWMA_F64_SEMANTIC_IDENTITY: &str =
    "fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery";

const FWMA_F64_QNAN_BITS: u64 = 0x7ff8_0000_0000_0000;
const FWMA_F64_U32_BASE: f64 = 4_294_967_296.0;

#[derive(Clone, Copy, Debug, Default)]
struct FwmaDd {
    hi: f64,
    lo: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FwmaU192 {
    lo: u64,
    mid: u64,
    hi: u64,
}

impl FwmaU192 {
    const ZERO: Self = Self {
        lo: 0,
        mid: 0,
        hi: 0,
    };
    const ONE: Self = Self {
        lo: 1,
        mid: 0,
        hi: 0,
    };

    #[inline(always)]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let (lo, carry0) = self.lo.overflowing_add(rhs.lo);
        let (mid0, carry1) = self.mid.overflowing_add(rhs.mid);
        let (mid, carry2) = mid0.overflowing_add(carry0 as u64);
        let (hi0, carry3) = self.hi.overflowing_add(rhs.hi);
        let (hi, carry4) = hi0.overflowing_add((carry1 || carry2) as u64);
        if carry3 || carry4 {
            None
        } else {
            Some(Self { lo, mid, hi })
        }
    }
}

#[inline(always)]
fn fwma_qnan_f64_v2() -> f64 {
    f64::from_bits(FWMA_F64_QNAN_BITS)
}

#[inline(always)]
fn fwma_canonical_zero_f64_v2(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[inline(always)]
fn fwma_two_sum_f64_v2(a: f64, b: f64) -> FwmaDd {
    let hi = a + b;
    let b_virtual = hi - a;
    let lo = (a - (hi - b_virtual)) + (b - b_virtual);
    FwmaDd { hi, lo }
}

#[inline(always)]
fn fwma_dd_add_f64_v2(a: FwmaDd, b: FwmaDd) -> FwmaDd {
    let high = fwma_two_sum_f64_v2(a.hi, b.hi);
    let low = fwma_two_sum_f64_v2(a.lo, b.lo);
    let middle = fwma_two_sum_f64_v2(high.lo, low.hi);
    let normalized = fwma_two_sum_f64_v2(high.hi, middle.hi);
    fwma_two_sum_f64_v2(normalized.hi, normalized.lo + middle.lo + low.lo)
}

#[inline(always)]
fn fwma_dd_sub_f64_v2(a: FwmaDd, b: FwmaDd) -> FwmaDd {
    fwma_dd_add_f64_v2(
        a,
        FwmaDd {
            hi: -b.hi,
            lo: -b.lo,
        },
    )
}

#[inline(always)]
fn fwma_dd_mul_f64_v2(value: f64, weight: FwmaDd) -> FwmaDd {
    let product = value * weight.hi;
    let product_tail = value.mul_add(weight.hi, -product);
    fwma_two_sum_f64_v2(product, value.mul_add(weight.lo, product_tail))
}

#[inline(always)]
fn fwma_dd_mul_scalar_f64_v2(value: FwmaDd, scalar: f64) -> FwmaDd {
    let product = value.hi * scalar;
    let product_tail = value.hi.mul_add(scalar, -product);
    fwma_two_sum_f64_v2(product, value.lo.mul_add(scalar, product_tail))
}

#[inline(always)]
fn fwma_u192_to_dd_f64_v2(value: FwmaU192) -> FwmaDd {
    let chunks = [
        value.lo as u32,
        (value.lo >> 32) as u32,
        value.mid as u32,
        (value.mid >> 32) as u32,
        value.hi as u32,
        (value.hi >> 32) as u32,
    ];
    let mut result = FwmaDd::default();
    for &chunk in chunks.iter().rev() {
        result.hi *= FWMA_F64_U32_BASE;
        result.lo *= FWMA_F64_U32_BASE;
        result = fwma_dd_add_f64_v2(
            result,
            FwmaDd {
                hi: chunk as f64,
                lo: 0.0,
            },
        );
    }
    result
}

#[inline]
fn fwma_exact_fibonacci_dd_f64_v2(period: usize) -> Option<(Vec<FwmaDd>, FwmaDd)> {
    if !(1..=FWMA_F64_MAX_PERIOD).contains(&period) {
        return None;
    }
    let mut exact_weights = Vec::with_capacity(period);
    let mut previous = FwmaU192::ONE;
    let mut current = FwmaU192::ONE;
    let mut denominator = FwmaU192::ZERO;
    for index in 0..period {
        let weight = match index {
            0 | 1 => FwmaU192::ONE,
            _ => {
                let next = previous.checked_add(current)?;
                previous = current;
                current = next;
                next
            }
        };
        denominator = denominator.checked_add(weight)?;
        exact_weights.push(fwma_u192_to_dd_f64_v2(weight));
    }
    Some((exact_weights, fwma_u192_to_dd_f64_v2(denominator)))
}

#[inline(always)]
fn fwma_unbiased_exponent_f64_v2(value: f64) -> Option<i32> {
    let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    if bits == 0 {
        return None;
    }
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent != 0 {
        Some(exponent - 1023)
    } else {
        let fraction = bits & 0x000f_ffff_ffff_ffff;
        let highest = 63 - fraction.leading_zeros() as i32;
        Some(highest - 1074)
    }
}

#[inline(always)]
fn fwma_pow2_f64_v2(exponent: i32) -> f64 {
    debug_assert!((-512..=512).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

#[inline(always)]
fn fwma_scale_pow2_checked_f64_v2(mut value: f64, mut exponent: i32) -> Option<f64> {
    while exponent != 0 {
        let step = exponent.clamp(-512, 512);
        let scaled = value * fwma_pow2_f64_v2(step);
        if !scaled.is_finite() || (scaled == 0.0 && value != 0.0) {
            return None;
        }
        value = scaled;
        exponent -= step;
    }
    Some(value)
}

#[inline(always)]
fn fwma_compensated_quotient_f64_v2(numerator: FwmaDd, denominator: FwmaDd) -> Option<f64> {
    if denominator.hi == 0.0 || !denominator.hi.is_finite() || !numerator.hi.is_finite() {
        return None;
    }
    let q0 = numerator.hi / denominator.hi;
    if !q0.is_finite() {
        return None;
    }
    let residual0 = fwma_dd_sub_f64_v2(numerator, fwma_dd_mul_scalar_f64_v2(denominator, q0));
    let q1 = residual0.hi / denominator.hi;
    let residual1 = fwma_dd_sub_f64_v2(residual0, fwma_dd_mul_scalar_f64_v2(denominator, q1));
    let q2 = residual1.hi / denominator.hi;
    let quotient = fwma_dd_add_f64_v2(fwma_two_sum_f64_v2(q0, q1), FwmaDd { hi: q2, lo: 0.0 });
    let result = quotient.hi + quotient.lo;
    if !result.is_finite() || (result == 0.0 && (numerator.hi != 0.0 || numerator.lo != 0.0)) {
        None
    } else {
        Some(fwma_canonical_zero_f64_v2(result))
    }
}

#[inline]
fn fwma_f64_window_authority_v2<F>(
    period: usize,
    weights: &[FwmaDd],
    denominator: FwmaDd,
    mut value_at: F,
) -> Option<f64>
where
    F: FnMut(usize) -> f64,
{
    debug_assert_eq!(weights.len(), period);
    if period == 1 {
        let value = value_at(0);
        return value
            .is_finite()
            .then_some(fwma_canonical_zero_f64_v2(value));
    }

    let mut maximum_exponent = None;
    for index in 0..period {
        let value = value_at(index);
        if !value.is_finite() {
            return None;
        }
        if let Some(exponent) = fwma_unbiased_exponent_f64_v2(value) {
            maximum_exponent =
                Some(maximum_exponent.map_or(exponent, |old: i32| old.max(exponent)));
        }
    }
    let Some(maximum_exponent) = maximum_exponent else {
        return Some(0.0);
    };

    let mut numerator = FwmaDd::default();
    for (index, &weight) in weights.iter().enumerate() {
        let value = value_at(index);
        let scaled = fwma_scale_pow2_checked_f64_v2(value, -maximum_exponent)?;
        numerator = fwma_dd_add_f64_v2(numerator, fwma_dd_mul_f64_v2(scaled, weight));
    }
    let scaled_result = fwma_compensated_quotient_f64_v2(numerator, denominator)?;
    fwma_scale_pow2_checked_f64_v2(scaled_result, maximum_exponent).map(fwma_canonical_zero_f64_v2)
}

#[inline]
fn fwma_f64_apply_authority_v2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    assert!(
        out.len() >= data.len(),
        "out must be at least as long as data"
    );
    assert!(
        (1..=FWMA_F64_MAX_PERIOD).contains(&period),
        "period must be within the certified f64 domain"
    );
    out[..data.len()].fill(fwma_qnan_f64_v2());
    let (weights, denominator) = fwma_exact_fibonacci_dd_f64_v2(period)
        .expect("the certified p<=254 Fibonacci table fits exactly in U192");
    let Some(warm) = first.checked_add(period - 1) else {
        return;
    };
    for index in warm..data.len() {
        let start = index + 1 - period;
        if let Some(value) = fwma_f64_window_authority_v2(period, &weights, denominator, |offset| {
            data[start + offset]
        }) {
            out[index] = value;
        }
    }
}

impl<'a> AsRef<[f64]> for FwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FwmaData::Slice(slice) => slice,
            FwmaData::Candles { candles, source } => match *source {
                "close" => candles.close.as_slice(),
                "open" => candles.open.as_slice(),
                "high" => candles.high.as_slice(),
                "low" => candles.low.as_slice(),
                "volume" => candles.volume.as_slice(),
                "hl2" => candles.hl2.as_slice(),
                "hlc3" => candles.hlc3.as_slice(),
                "ohlc4" => candles.ohlc4.as_slice(),
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum FwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FwmaParams {
    pub period: Option<usize>,
}

impl Default for FwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FwmaInput<'a> {
    pub data: FwmaData<'a>,
    pub params: FwmaParams,
}

impl<'a> FwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: FwmaParams) -> Self {
        Self {
            data: FwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: FwmaParams) -> Self {
        Self {
            data: FwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", FwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for FwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<FwmaOutput, FwmaError> {
        let p = FwmaParams {
            period: self.period,
        };
        let i = FwmaInput::from_candles(c, "close", p);
        fwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<FwmaOutput, FwmaError> {
        let p = FwmaParams {
            period: self.period,
        };
        let i = FwmaInput::from_slice(d, p);
        fwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<FwmaStream, FwmaError> {
        let p = FwmaParams {
            period: self.period,
        };
        FwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum FwmaError {
    #[error("fwma: Input data slice is empty.")]
    EmptyInputData,
    #[error("fwma: All values are NaN.")]
    AllValuesNaN,
    #[error("fwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fwma: Fibonacci sum is zero. Cannot normalize weights.")]
    ZeroFibonacciSum,
    #[error("fwma: Output buffer length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("fwma: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("fwma: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("fwma: arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
}

#[inline]
pub fn fwma(input: &FwmaInput) -> Result<FwmaOutput, FwmaError> {
    fwma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn fwma_prepare<'a>(
    input: &'a FwmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), FwmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(FwmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(FwmaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len || period > FWMA_F64_MAX_PERIOD {
        return Err(FwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(FwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, chosen))
}

#[inline(always)]
fn fwma_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    match kernel {
        Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        _ => unreachable!(),
    }
    fwma_f64_apply_authority_v2(data, period, first, out);
}

pub fn fwma_with_kernel(input: &FwmaInput, kernel: Kernel) -> Result<FwmaOutput, FwmaError> {
    let (data, period, first, chosen) = fwma_prepare(input, kernel)?;

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    fwma_compute_into(data, period, first, chosen, &mut out);

    Ok(FwmaOutput { values: out })
}

#[inline]
pub fn fwma_into_slice(dst: &mut [f64], input: &FwmaInput, kern: Kernel) -> Result<(), FwmaError> {
    let (data, period, first, chosen) = fwma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(FwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let warmup_end = (first + period - 1).min(dst.len());
    for v in &mut dst[..warmup_end] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    fwma_compute_into(data, period, first, chosen, dst);

    Ok(())
}

#[inline]
pub fn fwma_into(input: &FwmaInput, out: &mut [f64]) -> Result<(), FwmaError> {
    let (data, period, first, chosen) = fwma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(FwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let warm = (first + period - 1).min(out.len());
    for v in &mut out[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    fwma_compute_into(data, period, first, chosen, out);
    Ok(())
}

#[inline(always)]
pub unsafe fn fwma_scalar(
    data: &[f64],
    fib: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    assert_eq!(fib.len(), period, "fib.len() must equal period");
    fwma_f64_apply_authority_v2(data, period, first_val, out);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn fwma_simd128(
    data: &[f64],
    fib: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    assert_eq!(fib.len(), period, "fib.len() must equal period");
    fwma_f64_apply_authority_v2(data, period, first_val, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn fwma_avx512(data: &[f64], fib: &[f64], period: usize, first: usize, out: &mut [f64]) {
    assert_eq!(fib.len(), period, "fib.len() must equal period");
    fwma_f64_apply_authority_v2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn fwma_avx2(
    data: &[f64],
    fib: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    assert_eq!(fib.len(), period, "fib.len() must equal period");
    fwma_f64_apply_authority_v2(data, period, first_valid, out);
}

#[derive(Debug, Clone)]
pub struct FwmaStream {
    period: usize,
    weights: Vec<FwmaDd>,
    denominator: FwmaDd,
    buffer: Vec<f64>,
    head: usize,
    samples: usize,
}

impl FwmaStream {
    pub fn try_new(params: FwmaParams) -> Result<Self, FwmaError> {
        let period = params.period.unwrap_or(5);
        if !(1..=FWMA_F64_MAX_PERIOD).contains(&period) {
            return Err(FwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let (weights, denominator) = fwma_exact_fibonacci_dd_f64_v2(period)
            .expect("the certified p<=254 Fibonacci table fits exactly in U192");

        Ok(Self {
            period,
            weights,
            denominator,
            buffer: vec![fwma_qnan_f64_v2(); period],
            head: 0,
            samples: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        self.samples = self.samples.saturating_add(1);
        if self.samples < self.period {
            return None;
        }
        fwma_f64_window_authority_v2(self.period, &self.weights, self.denominator, |offset| {
            self.buffer[(self.head + offset) % self.period]
        })
    }
}

#[derive(Clone, Debug)]
pub struct FwmaBatchRange {
    pub period: (usize, usize, usize),
}
impl Default for FwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FwmaBatchBuilder {
    range: FwmaBatchRange,
    kernel: Kernel,
}

impl FwmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<FwmaBatchOutput, FwmaError> {
        fwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<FwmaBatchOutput, FwmaError> {
        FwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<FwmaBatchOutput, FwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FwmaBatchOutput, FwmaError> {
        FwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn fwma_batch_with_kernel(
    data: &[f64],
    sweep: &FwmaBatchRange,
    k: Kernel,
) -> Result<FwmaBatchOutput, FwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(FwmaError::InvalidKernelForBatch(other));
        }
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    fwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct FwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl FwmaBatchOutput {
    pub fn row_for_params(&self, p: &FwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &FwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &FwmaBatchRange) -> Vec<FwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
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
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(FwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
fn fill_nan_prefixes_slice(
    rows: usize,
    cols: usize,
    warmup_periods: &[usize],
    out_slice: &mut [f64],
) {
    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        let row_end = row_start + warmup.min(cols);
        for i in row_start..row_end {
            out_slice[i] = f64::NAN;
        }
    }
}

#[inline(always)]
pub fn fwma_batch_slice(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
) -> Result<FwmaBatchOutput, FwmaError> {
    fwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn fwma_batch_par_slice(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
) -> Result<FwmaBatchOutput, FwmaError> {
    fwma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn fwma_batch_admission_v2(data: &[f64], combos: &[FwmaParams]) -> Result<usize, FwmaError> {
    let data_len = data.len();
    let mut max_period = 0usize;
    for combo in combos {
        let period = combo.period.unwrap();
        if period == 0 || period > data_len || period > FWMA_F64_MAX_PERIOD {
            return Err(FwmaError::InvalidPeriod { period, data_len });
        }
        max_period = max_period.max(period);
    }
    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(FwmaError::AllValuesNaN)?;
    let valid_tail = data_len - first;
    if valid_tail < max_period {
        return Err(FwmaError::NotEnoughValidData {
            needed: max_period,
            valid: valid_tail,
        });
    }
    Ok(first)
}

#[inline(always)]
fn fwma_batch_inner(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FwmaBatchOutput, FwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(FwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let first = fwma_batch_admission_v2(data, &combos)?;

    let _total = rows
        .checked_mul(cols)
        .ok_or(FwmaError::ArithmeticOverflow {
            context: "rows*cols in fwma_batch_inner",
        })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    {
        let values_slice: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(buf_mu.as_mut_ptr() as *mut f64, buf_mu.len())
        };
        fwma_batch_inner_into(data, sweep, kern, parallel, values_slice)?;
    }
    let mut buf_guard = ManuallyDrop::new(buf_mu);

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(FwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}
#[inline(always)]
fn fwma_batch_inner_into(
    data: &[f64],
    sweep: &FwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<FwmaParams>, FwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(FwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }

    let first = fwma_batch_admission_v2(data, &combos)?;

    let cols = data.len();

    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => fwma_row_avx512(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => fwma_row_avx2(data, first, period, dst),
            _ => fwma_row_scalar(data, first, period, dst),
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

#[inline]
unsafe fn fwma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fwma_f64_apply_authority_v2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fwma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fwma_f64_apply_authority_v2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
unsafe fn fwma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    fwma_f64_apply_authority_v2(data, period, first, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    const FWMA_RUST_SOURCE: &str = include_str!("fwma.rs");
    const FWMA_CUDA_SOURCE: &str =
        include_str!("../../../kernels/cuda/moving_averages/fwma_kernel.cu");

    #[test]
    fn fwma_f64_v2_source_contract_is_closed_over_one_authority() {
        let production = FWMA_RUST_SOURCE
            .split("#[cfg(test)]")
            .next()
            .expect("fwma.rs must retain a production section");

        assert_eq!(FWMA_F64_MAX_PERIOD, 254);
        assert!(production.contains("fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery"));
        assert!(production.contains("struct FwmaU192"));
        assert!(production.contains("fn fwma_f64_window_authority_v2"));
        assert!(production.contains("fn fwma_f64_apply_authority_v2"));
        assert!(production.contains("value.mul_add(weight.hi, -product)"));
        assert!(production.contains("fn fwma_canonical_zero_f64_v2"));
        assert!(production.contains("FWMA_F64_MAX_PERIOD: usize = 254"));
        assert!(!production.contains("quick_two_sum"));
        assert!(!production.contains("unsafe fn fwma_scalar_period5"));
        assert!(!production.contains("fn fwma_avx512_short"));
        assert!(!production.contains("fn fwma_avx512_long"));

        assert!(FWMA_CUDA_SOURCE.contains("#define FWMA_MAX_PERIOD_F64 254"));
        assert!(FWMA_CUDA_SOURCE.contains("fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery"));
        assert!(FWMA_CUDA_SOURCE.contains("struct fwma_u192_f64_v2"));
        assert!(FWMA_CUDA_SOURCE.contains("__fma_rn(value, weight.hi, -product)"));
        assert!(FWMA_CUDA_SOURCE.contains("fwma_scale_pow2_checked_f64_v2"));
        assert!(FWMA_CUDA_SOURCE.contains("fwma_canonical_zero_f64_v2"));
        assert!(FWMA_CUDA_SOURCE.contains("if (!isfinite(value))"));
        assert!(!FWMA_CUDA_SOURCE.contains("quick_two_sum"));
    }

    #[test]
    fn fwma_batch_admission_precedes_matrix_allocation_and_storage_ownership() {
        let production = FWMA_RUST_SOURCE
            .split("#[cfg(test)]")
            .next()
            .expect("production source");
        let batch = production
            .split_once("fn fwma_batch_inner(")
            .expect("batch function")
            .1
            .split_once("fn fwma_batch_inner_into(")
            .expect("batch function end")
            .0;
        let admission = batch
            .find("fwma_batch_admission_v2(data, &combos)?")
            .expect("pre-allocation admission");
        let allocation = batch
            .find("make_uninit_matrix(rows, cols)")
            .expect("matrix allocation");
        let compute = batch
            .find("fwma_batch_inner_into(data, sweep, kern, parallel, values_slice)?")
            .expect("batch computation");
        let storage_owner = batch
            .find("ManuallyDrop::new(buf_mu)")
            .expect("storage ownership transfer");
        assert!(admission < allocation);
        assert!(compute < storage_owner);
    }

    #[test]
    fn fwma_batch_late_finite_and_all_nonfinite_inputs_fail_closed() {
        let mut late_finite = vec![f64::NAN; 4096];
        late_finite[4095] = 1.0;
        for _ in 0..64 {
            assert!(matches!(
                FwmaBatchBuilder::new()
                    .kernel(Kernel::ScalarBatch)
                    .period_static(2)
                    .apply_slice(&late_finite),
                Err(FwmaError::NotEnoughValidData {
                    needed: 2,
                    valid: 1
                })
            ));
        }
        assert!(matches!(
            FwmaBatchBuilder::new()
                .kernel(Kernel::ScalarBatch)
                .period_static(2)
                .apply_slice(&vec![f64::NAN; 4096]),
            Err(FwmaError::AllValuesNaN)
        ));
    }

    #[test]
    fn fwma_f64_v2_rejects_periods_above_254_on_every_public_constructor() {
        let data = vec![1.0; FWMA_F64_MAX_PERIOD + 2];
        let input = FwmaInput::from_slice(
            &data,
            FwmaParams {
                period: Some(FWMA_F64_MAX_PERIOD + 1),
            },
        );
        assert!(matches!(
            fwma_with_kernel(&input, Kernel::Scalar),
            Err(FwmaError::InvalidPeriod { .. })
        ));
        assert!(matches!(
            FwmaStream::try_new(FwmaParams {
                period: Some(FWMA_F64_MAX_PERIOD + 1),
            }),
            Err(FwmaError::InvalidPeriod { .. })
        ));
        assert!(matches!(
            FwmaBatchBuilder::new()
                .kernel(Kernel::ScalarBatch)
                .period_static(FWMA_F64_MAX_PERIOD + 1)
                .apply_slice(&data),
            Err(FwmaError::InvalidPeriod { .. })
        ));
    }

    #[test]
    fn fwma_f64_v2_nonfinite_window_is_canonical_and_recovers_exactly() {
        const QNAN: u64 = 0x7ff8_0000_0000_0000;
        let period = 5;
        let mut data = (0..24).map(|i| 1.0 + i as f64 * 0.125).collect::<Vec<_>>();
        data[7] = f64::from_bits(QNAN | 0x55);
        data[15] = f64::INFINITY;
        let input = FwmaInput::from_slice(
            &data,
            FwmaParams {
                period: Some(period),
            },
        );
        let direct = fwma_with_kernel(&input, Kernel::Scalar).unwrap().values;
        let batch = FwmaBatchBuilder::new()
            .kernel(Kernel::ScalarBatch)
            .period_static(period)
            .apply_slice(&data)
            .unwrap();
        let batch_row = batch
            .values_for(&FwmaParams {
                period: Some(period),
            })
            .unwrap();
        let mut stream = FwmaStream::try_new(FwmaParams {
            period: Some(period),
        })
        .unwrap();

        for i in 0..data.len() {
            let window_is_finite =
                i + 1 >= period && data[i + 1 - period..=i].iter().all(|x| x.is_finite());
            if window_is_finite {
                assert_eq!(direct[i].to_bits(), batch_row[i].to_bits(), "row {i}");
                assert_eq!(
                    stream.update(data[i]).unwrap().to_bits(),
                    direct[i].to_bits()
                );
            } else {
                assert_eq!(direct[i].to_bits(), QNAN, "direct row {i}");
                assert_eq!(batch_row[i].to_bits(), QNAN, "batch row {i}");
                assert!(stream.update(data[i]).is_none(), "stream row {i}");
            }
        }
        assert!(direct[12].is_finite(), "first full finite window after NaN");
        assert!(
            direct[20].is_finite(),
            "first full finite window after +inf"
        );
    }

    #[test]
    fn fwma_f64_v2_all_route_labels_are_exact_bits_for_certified_periods() {
        for period in [1usize, 2, 5, 7, 14, 21, 50, 100, 200, 254] {
            let len = period + 19;
            let data = (0..len)
                .map(|i| {
                    let sign = if i % 7 == 0 { -1.0 } else { 1.0 };
                    sign * (0.75 + i as f64 * 0.03125)
                })
                .collect::<Vec<_>>();
            let input = FwmaInput::from_slice(
                &data,
                FwmaParams {
                    period: Some(period),
                },
            );
            let scalar = fwma_with_kernel(&input, Kernel::Scalar).unwrap().values;
            for label in [Kernel::Avx2, Kernel::Avx512] {
                let actual = fwma_with_kernel(&input, label).unwrap().values;
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "period {period}, label {label:?}"
                );
            }
            let batch = FwmaBatchBuilder::new()
                .kernel(Kernel::ScalarBatch)
                .period_static(period)
                .apply_slice(&data)
                .unwrap();
            assert_eq!(
                batch.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "period {period}, batch"
            );
            let mut stream = FwmaStream::try_new(FwmaParams {
                period: Some(period),
            })
            .unwrap();
            for (i, &value) in data.iter().enumerate() {
                match stream.update(value) {
                    Some(actual) => assert_eq!(actual.to_bits(), scalar[i].to_bits()),
                    None => assert_eq!(scalar[i].to_bits(), 0x7ff8_0000_0000_0000),
                }
            }
        }
    }

    #[test]
    fn fwma_f64_v2_frozen_exact_rational_fixture_points_are_exact_bits() {
        let waves = [
            0.000_041, -0.000_027, 0.000_013, -0.000_036, 0.000_022, -0.000_009, 0.000_033,
            -0.000_019, 0.000_006, -0.000_031, 0.000_017,
        ];
        let close = (0..=300)
            .map(|row| 1.075 + row as f64 * 0.000_000_7 + waves[row % waves.len()])
            .collect::<Vec<_>>();
        let cases = [
            (1usize, 0usize, 0x3ff1_335e_310d_bf05u64),
            (2, 1, 0x3ff1_333a_e833_5aa0),
            (5, 14, 0x3ff1_3330_a601_ce7e),
            (7, 17, 0x3ff1_334b_5548_5e7f),
            (14, 14, 0x3ff1_3330_ca0b_2113),
            (21, 23, 0x3ff1_3342_365a_b605),
            (50, 63, 0x3ff1_3362_6779_0fbd),
            (100, 102, 0x3ff1_3371_6b63_b31e),
            (200, 200, 0x3ff1_33c9_9930_9b87),
            (254, 300, 0x3ff1_3402_c08b_2ce0),
        ];
        for (period, row, expected_bits) in cases {
            let input = FwmaInput::from_slice(
                &close[..=row],
                FwmaParams {
                    period: Some(period),
                },
            );
            let actual = fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[row];
            assert_eq!(
                actual.to_bits(),
                expected_bits,
                "exact-rational oracle p={period} row={row}"
            );
        }
    }

    #[test]
    fn fwma_f64_v2_extremes_are_finite_or_fail_closed_never_silent_zero() {
        const QNAN: u64 = 0x7ff8_0000_0000_0000;
        for value in [f64::MAX, f64::from_bits(1)] {
            let data = vec![value; 7];
            let input = FwmaInput::from_slice(&data, FwmaParams { period: Some(7) });
            let actual = fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[6];
            assert_eq!(actual.to_bits(), value.to_bits());
        }

        let zeros = [0.0; 7];
        let input = FwmaInput::from_slice(&zeros, FwmaParams { period: Some(7) });
        assert_eq!(
            fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[6].to_bits(),
            0
        );

        let negative_zero = [-0.0];
        let input = FwmaInput::from_slice(&negative_zero, FwmaParams { period: Some(1) });
        assert_eq!(
            fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[0].to_bits(),
            0,
            "every accepted mathematical zero is canonical +0.0"
        );
        let mut stream = FwmaStream::try_new(FwmaParams { period: Some(1) }).unwrap();
        assert_eq!(
            stream.update(-0.0).unwrap().to_bits(),
            0,
            "streaming negative zero is canonical +0.0"
        );

        let ordinary_cancellation = [-1.0, -1.0, 1.0];
        let input = FwmaInput::from_slice(&ordinary_cancellation, FwmaParams { period: Some(3) });
        assert_eq!(
            fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[2].to_bits(),
            0,
            "non-extreme exact cancellation is canonical +0.0"
        );

        let balanced_max = [f64::MAX, -f64::MAX];
        let input = FwmaInput::from_slice(&balanced_max, FwmaParams { period: Some(2) });
        assert_eq!(
            fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[1].to_bits(),
            0,
            "representable mixed +/-MAX cancellation"
        );
        let finite_mixed_max = [f64::MAX, -f64::MAX, f64::MAX];
        let input = FwmaInput::from_slice(&finite_mixed_max, FwmaParams { period: Some(3) });
        assert_eq!(
            fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[2].to_bits(),
            (f64::MAX * 0.5).to_bits(),
            "representable mixed +/-MAX weighted result"
        );

        let underflow = [f64::from_bits(1), 0.0];
        let input = FwmaInput::from_slice(&underflow, FwmaParams { period: Some(2) });
        assert_eq!(
            fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[1].to_bits(),
            QNAN,
            "nonzero exact weighted result below the f64 range must fail closed"
        );

        let data = [f64::MAX, -f64::MAX, f64::from_bits(1), 1.0, -1.0];
        let input = FwmaInput::from_slice(&data, FwmaParams { period: Some(5) });
        let actual = fwma_with_kernel(&input, Kernel::Scalar).unwrap().values[4];
        assert_eq!(
            actual.to_bits(),
            QNAN,
            "uncertified scale loss must fail closed"
        );
    }

    fn check_fwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = FwmaParams { period: None };
        let input = FwmaInput::from_candles(&candles, "close", default_params);
        let output = fwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_fwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = FwmaInput::with_default_candles(&candles);
        let result = fwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] FWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_fwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = FwmaInput::with_default_candles(&candles);
        match input.data {
            FwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected FwmaData::Candles"),
        }
        let output = fwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_fwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = FwmaParams { period: Some(0) };
        let input = FwmaInput::from_slice(&input_data, params);
        let res = fwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_fwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = FwmaParams { period: Some(10) };
        let input = FwmaInput::from_slice(&data_small, params);
        let res = fwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_fwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = FwmaParams { period: Some(5) };
        let input = FwmaInput::from_slice(&single_point, params);
        let res = fwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_fwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = FwmaParams { period: Some(5) };
        let first_input = FwmaInput::from_candles(&candles, "close", first_params);
        let first_result = fwma_with_kernel(&first_input, kernel)?;

        let second_params = FwmaParams { period: Some(3) };
        let second_input = FwmaInput::from_slice(&first_result.values, second_params);
        let second_result = fwma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] NaN found at idx {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_fwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = FwmaInput::from_candles(&candles, "close", FwmaParams { period: Some(5) });
        let res = fwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_fwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let period = 5;

        let input = FwmaInput::from_candles(
            &candles,
            "close",
            FwmaParams {
                period: Some(period),
            },
        );
        let batch_output = fwma_with_kernel(&input, kernel)?.values;

        let mut stream = FwmaStream::try_new(FwmaParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] FWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_fwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_cases = vec![
            FwmaParams::default(),
            FwmaParams { period: Some(2) },
            FwmaParams { period: Some(3) },
            FwmaParams { period: Some(5) },
            FwmaParams { period: Some(8) },
            FwmaParams { period: Some(10) },
            FwmaParams { period: Some(15) },
            FwmaParams { period: Some(20) },
            FwmaParams { period: Some(30) },
            FwmaParams { period: Some(50) },
        ];

        for params in test_cases {
            let input = FwmaInput::from_candles(&candles, "close", params.clone());
            let output = fwma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_fwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_fwma_tests {
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


                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_fwma_tests!(
        check_fwma_partial_params,
        check_fwma_accuracy,
        check_fwma_default_candles,
        check_fwma_zero_period,
        check_fwma_period_exceeds_length,
        check_fwma_very_small_dataset,
        check_fwma_reinput,
        check_fwma_nan_handling,
        check_fwma_streaming,
        check_fwma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_fwma_tests!(check_fwma_property);

    #[test]
    fn test_fwma_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = FwmaInput::with_default_candles(&candles);
        let baseline = fwma(&input)?.values;

        let mut out = vec![0.0f64; baseline.len()];

        {
            fwma_into(&input, &mut out)?;
        }

        assert_eq!(out.len(), baseline.len());

        for (i, (&a, &b)) in out.iter().zip(baseline.iter()).enumerate() {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(
                equal,
                "into parity mismatch at idx {}: got {}, expected {}",
                i, a, b
            );
        }

        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = FwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = FwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
            (2, 5, 1),
            (5, 15, 2),
            (10, 30, 5),
            (3, 10, 1),
            (5, 50, 10),
            (2, 20, 1),
        ];

        for (start, end, step) in test_configs {
            let output = FwmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: period={:?})",
                        test, val, bits, row, col, params.period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: period={:?})",
                        test, val, bits, row, col, params.period
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: period={:?})",
                        test, val, bits, row, col, params.period
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_fwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(mut data, period)| {
                if data.len() > period && period > 1 {
                    if data.len() % 10 == 0 {
                        data.truncate(period);
                    }
                }

                let params = FwmaParams {
                    period: Some(period),
                };
                let input = FwmaInput::from_slice(&data, params);

                let FwmaOutput { values: out } = fwma_with_kernel(&input, kernel).unwrap();

                let FwmaOutput { values: ref_out } =
                    fwma_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), data.len());
                prop_assert_eq!(ref_out.len(), data.len());

                for i in 0..(period - 1).min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                if period == 2 && data.len() >= 2 {
                    let expected = (data[0] + data[1]) / 2.0;
                    if out[1].is_finite() && data[0].is_finite() && data[1].is_finite() {
                        prop_assert!(
                            (out[1] - expected).abs() <= 1e-9,
                            "Period=2: output {} should equal average {} at index 1",
                            out[1],
                            expected
                        );
                    }
                }

                for i in (period - 1)..data.len() {
                    let window = &data[i + 1 - period..=i];
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];
                    let r = ref_out[i];

                    prop_assert!(
                        y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                        "idx {}: {} ∉ [{}, {}]",
                        i,
                        y,
                        lo,
                        hi
                    );

                    if period == 1 {
                        prop_assert!(
                            (y - data[i]).abs() <= f64::EPSILON,
                            "Period=1: output {} should equal input {} at index {}",
                            y,
                            data[i],
                            i
                        );
                    }

                    if data.windows(2).all(|w| w[0] == w[1]) && !data.is_empty() {
                        prop_assert!(
                            (y - data[0]).abs() <= 1e-9,
                            "Constant data: output {} should equal constant {} at index {}",
                            y,
                            data[0],
                            i
                        );
                    }

                    if window.iter().any(|x| x.is_nan()) {
                        prop_assert!(
                            y.is_nan(),
                            "Window contains NaN but output {} is not NaN at index {}",
                            y,
                            i
                        );
                    }

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                let is_monotonic_inc = data.windows(2).all(|w| w[0] <= w[1]);
                let is_monotonic_dec = data.windows(2).all(|w| w[0] >= w[1]);

                if (is_monotonic_inc || is_monotonic_dec) && data.len() >= period + 1 {
                    for i in period..out.len() {
                        if out[i].is_finite() && out[i - 1].is_finite() {
                            if is_monotonic_inc {
                                prop_assert!(
									out[i] >= out[i-1] - 1e-9,
									"Monotonic increasing data but output decreases: {} < {} at index {}",
									out[i],
									out[i-1],
									i
								);
                            }
                            if is_monotonic_dec {
                                prop_assert!(
									out[i] <= out[i-1] + 1e-9,
									"Monotonic decreasing data but output increases: {} > {} at index {}",
									out[i],
									out[i-1],
									i
								);
                            }
                        }
                    }
                }

                if period >= 3 && data.len() >= period * 2 {
                    let test_start = period;
                    if test_start + period <= data.len() {
                        let all_ascending = (0..period).all(|j| {
                            let idx = test_start + j;
                            idx == 0
                                || !data[idx].is_finite()
                                || !data[idx - 1].is_finite()
                                || data[idx] >= data[idx - 1]
                        });

                        if all_ascending && out[test_start + period - 1].is_finite() {
                            let window = &data[test_start..test_start + period];
                            let window_avg = window.iter().sum::<f64>() / period as f64;
                            if window.iter().all(|x| x.is_finite()) {
                                prop_assert!(
                                    out[test_start + period - 1] >= window_avg - 1e-9,
                                    "FWMA {} should be >= average {} for ascending window",
                                    out[test_start + period - 1],
                                    window_avg
                                );
                            }
                        }
                    }
                }

                if period > 1 && data.len() >= period * 2 {
                    let data_range = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
                    for &val in &out[(period - 1)..] {
                        if val.is_finite() && data_range > 0.0 {
                            prop_assert!(
                                val.abs() <= data_range * 1.1,
                                "Output {} exceeds reasonable bounds for data range {}",
                                val,
                                data_range
                            );
                        }
                    }
                }

                if period == 3 && data.len() >= 3 {
                    let idx = period - 1;
                    if data[idx - 2].is_finite()
                        && data[idx - 1].is_finite()
                        && data[idx].is_finite()
                    {
                        let expected =
                            data[idx - 2] * 0.25 + data[idx - 1] * 0.25 + data[idx] * 0.5;
                        prop_assert!(
                            (out[idx] - expected).abs() <= 1e-9,
                            "Period=3: output {} should equal weighted avg {} at index {}",
                            out[idx],
                            expected,
                            idx
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        let nan_strat = (2usize..=10).prop_flat_map(|period| (Just(period), 1usize..10));

        proptest::test_runner::TestRunner::default()
            .run(&nan_strat, |(period, nan_pos)| {
                let mut data = vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ];
                if nan_pos < data.len() {
                    data[nan_pos] = f64::NAN;
                }

                let params = FwmaParams {
                    period: Some(period),
                };
                let input = FwmaInput::from_slice(&data, params);
                let FwmaOutput { values: out } = fwma_with_kernel(&input, kernel).unwrap();

                for i in (period - 1)..data.len() {
                    let window_start = i + 1 - period;
                    let window = &data[window_start..=i];
                    let has_nan = window.iter().any(|x| x.is_nan());

                    if has_nan {
                        prop_assert!(
                            out[i].is_nan(),
                            "Window [{}, {}] contains NaN but output {} is not NaN at index {}",
                            window_start,
                            i,
                            out[i],
                            i
                        );
                    }
                }
                Ok(())
            })
            .unwrap();

        let extreme_strat = (1usize..=10).prop_flat_map(|period| (Just(period), prop::bool::ANY));

        proptest::test_runner::TestRunner::default()
            .run(&extreme_strat, |(period, use_max)| {
                let extreme_val = if use_max { 1e308 } else { 1e-308 };
                let data = vec![extreme_val; period * 2];

                let params = FwmaParams {
                    period: Some(period),
                };
                let input = FwmaInput::from_slice(&data, params);
                let result = fwma_with_kernel(&input, kernel);

                prop_assert!(result.is_ok(), "Failed to handle extreme values");

                if let Ok(FwmaOutput { values: out }) = result {
                    for i in (period - 1)..data.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                (out[i] - extreme_val).abs() / extreme_val.abs() <= 1e-9,
                                "Extreme constant value {} doesn't match output {} at index {}",
                                extreme_val,
                                out[i],
                                i
                            );
                        }
                    }
                }
                Ok(())
            })
            .unwrap();

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
    fn print_actual_fwma_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path).unwrap();

        println!("\nCandle data info:");
        println!("Total candles: {}", candles.close.len());
        println!("Last 10 close prices:");
        let close_len = candles.close.len();
        for i in (close_len - 10)..close_len {
            println!("  [{}]: {}", i - (close_len - 10), candles.close[i]);
        }

        let input = FwmaInput::with_default_candles(&candles);
        let result = fwma(&input).unwrap();

        println!("\nFWMA results (period = 5):");
        println!("Last 5 values:");
        let result_len = result.values.len();
        for i in (result_len - 5)..result_len {
            println!("  [{}]: {:.12}", i - (result_len - 5), result.values[i]);
        }

        let expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];

        println!("\nExpected values:");
        for (i, val) in expected_last_five.iter().enumerate() {
            println!("  [{}]: {:.12}", i, val);
        }

        println!("\nDifferences:");
        for i in 0..5 {
            let actual = result.values[result_len - 5 + i];
            let expected = expected_last_five[i];
            println!(
                "  [{}]: {:.12} (diff: {:.2e})",
                i,
                actual - expected,
                (actual - expected).abs()
            );
        }
    }
}
