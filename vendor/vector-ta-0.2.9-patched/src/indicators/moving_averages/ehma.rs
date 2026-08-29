use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

pub const EHMA_F64_AUTHORITY_V2: &str =
    "ehma_hann_f64_msun_ddangle_symmetric_pow2_anchored_dot2_v2";

const EHMA_PI_HI_V2: f64 = f64::from_bits(0x4009_21fb_5444_2d18);
const EHMA_PI_LO_V2: f64 = f64::from_bits(0x3ca1_a626_3314_5c07);
const EHMA_CANONICAL_QNAN_V2: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

/* FreeBSD msun k_sin/k_cos and medium pi/2 reduction.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft. Permission to use, copy, modify, and
 * distribute this software is freely granted, provided this notice is
 * preserved.
 *
 * EHMA evaluates only finite half-angles in [0, pi/2]. The strict CUDA f64
 * row mirrors every constant, branch, operation, and parenthesisation below,
 * so neither host libm nor libdevice defines an f64 coefficient.
 */
#[inline(always)]
fn ehma_msun_k_cos_v2(x: f64, y: f64) -> f64 {
    let c1 = f64::from_bits(0x3fa5_5555_5555_554c);
    let c2 = f64::from_bits(0xbf56_c16c_16c1_5177);
    let c3 = f64::from_bits(0x3efa_01a0_19cb_1590);
    let c4 = f64::from_bits(0xbe92_7e4f_809c_52ad);
    let c5 = f64::from_bits(0x3e21_ee9e_bdb4_b1c4);
    let c6 = f64::from_bits(0xbda8_fae9_be88_38d4);
    let z = x * x;
    let w2 = z * z;
    let r = z * (c1 + z * (c2 + z * c3)) + w2 * w2 * (c4 + z * (c5 + z * c6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

#[inline(always)]
fn ehma_msun_k_sin_v2(x: f64, y: f64, has_tail: bool) -> f64 {
    let s1 = f64::from_bits(0xbfc5_5555_5555_5549);
    let s2 = f64::from_bits(0x3f81_1111_1110_f8a6);
    let s3 = f64::from_bits(0xbf2a_01a0_19c1_61d5);
    let s4 = f64::from_bits(0x3ec7_1de3_57b1_fe7d);
    let s5 = f64::from_bits(0xbe5a_e5e6_8a2b_9ceb);
    let s6 = f64::from_bits(0x3de5_d93a_5acf_d57c);
    let z = x * x;
    let w = z * z;
    let r = s2 + z * (s3 + z * s4) + z * w * (s5 + z * s6);
    let v = z * x;
    if has_tail {
        x - ((z * (0.5 * y - v * r) - y) - v * s1)
    } else {
        x + v * (s1 + z * r)
    }
}

#[inline(always)]
fn ehma_reduce_pio2_v2(x: f64) -> (i32, f64, f64) {
    let inv_pio2 = f64::from_bits(0x3fe4_5f30_6dc9_c883);
    let to_int = f64::from_bits(0x4338_0000_0000_0000);
    let pio2_1 = f64::from_bits(0x3ff9_21fb_5440_0000);
    let pio2_1t = f64::from_bits(0x3dd0_b461_1a62_6331);
    let pio2_2 = f64::from_bits(0x3dd0_b461_1a60_0000);
    let pio2_2t = f64::from_bits(0x3ba3_198a_2e03_7073);
    let pio2_3 = f64::from_bits(0x3ba3_198a_2e00_0000);
    let pio2_3t = f64::from_bits(0x397b_839a_2520_49c1);

    let tmp = x * inv_pio2 + to_int;
    let f_n = tmp - to_int;
    let n = f_n as i32;
    let mut r = x - f_n * pio2_1;
    let mut w = f_n * pio2_1t;
    let mut y0 = r - w;
    let ex = ((x.to_bits() >> 52) & 0x7ff) as i32;
    let mut ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
    if ex - ey > 16 {
        let t = r;
        w = f_n * pio2_2;
        r = t - w;
        w = f_n * pio2_2t - ((t - r) - w);
        y0 = r - w;
        ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
        if ex - ey > 49 {
            let t = r;
            w = f_n * pio2_3;
            r = t - w;
            w = f_n * pio2_3t - ((t - r) - w);
            y0 = r - w;
        }
    }
    (n, y0, (r - y0) - w)
}

#[inline(always)]
fn ehma_deterministic_sin_v2(x: f64) -> f64 {
    debug_assert!(x.is_finite() && (0.0..=EHMA_PI_HI_V2 * 0.5).contains(&x));
    let high = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if high <= 0x3fe9_21fb {
        return ehma_msun_k_sin_v2(x, 0.0, false);
    }

    let (quadrant, y0, y1) = ehma_reduce_pio2_v2(x);
    let sin = ehma_msun_k_sin_v2(y0, y1, true);
    let cos = ehma_msun_k_cos_v2(y0, y1);
    match quadrant & 3 {
        0 => sin,
        1 => cos,
        2 => -sin,
        _ => -cos,
    }
}

#[inline(always)]
fn ehma_half_angle_v2(period: usize, k: usize) -> f64 {
    let denominator = period as f64 + 1.0;
    let numerator = k as f64;
    let quotient = numerator / denominator;
    let quotient_remainder = (-quotient).mul_add(denominator, numerator);
    let product = quotient * EHMA_PI_HI_V2;
    let product_error = quotient.mul_add(EHMA_PI_HI_V2, -product);
    let correction = (product_error + quotient * EHMA_PI_LO_V2)
        + (quotient_remainder / denominator) * EHMA_PI_HI_V2;
    product + correction
}

#[derive(Clone, Copy, Debug, Default)]
struct EhmaDot2V2 {
    sum: f64,
    correction: f64,
}

impl EhmaDot2V2 {
    #[inline(always)]
    fn add_product(&mut self, left: f64, right: f64) {
        let product = left * right;
        let product_error = left.mul_add(right, -product);
        let updated = self.sum + product;
        let recovered = updated - self.sum;
        let addition_error = (self.sum - (updated - recovered)) + (product - recovered);
        self.sum = updated;
        self.correction += product_error + addition_error;
    }

    #[inline(always)]
    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

#[inline(always)]
fn build_hann_weights_v2(period: usize) -> (AVec<f64>, f64) {
    let _authority = EHMA_F64_AUTHORITY_V2;
    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
    weights.resize(period, 0.0);
    for k in 1..=((period + 1) / 2) {
        let sine = ehma_deterministic_sin_v2(ehma_half_angle_v2(period, k));
        let weight = 2.0 * (sine * sine);
        weights[k - 1] = weight;
        weights[period - k] = weight;
    }

    let mut coefficient = EhmaDot2V2::default();
    for &weight in weights.iter() {
        coefficient.add_product(1.0, weight);
    }
    (weights, coefficient.value())
}

#[inline(always)]
fn floor_power_of_two_scale_v2(max_abs_input: f64) -> f64 {
    let bits = max_abs_input.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    if exponent != 0 {
        return f64::from_bits(exponent << 52);
    }
    let fraction = bits & ((1_u64 << 52) - 1);
    let highest_bit = 63_u32 - fraction.leading_zeros();
    f64::from_bits(1_u64 << highest_bit)
}

#[inline(always)]
fn ehma_stable_window_indexed_v2<F>(
    period: usize,
    weights: &[f64],
    coefficient: f64,
    mut value_at: F,
) -> f64
where
    F: FnMut(usize) -> f64,
{
    let mut max_abs_input = 0.0_f64;
    let mut has_infinite = false;
    for index in 0..period {
        let value = value_at(index);
        if value.is_nan() {
            return EHMA_CANONICAL_QNAN_V2;
        }
        if value.is_infinite() {
            has_infinite = true;
        } else {
            max_abs_input = max_abs_input.max(value.abs());
        }
    }

    if has_infinite {
        let mut sum = 0.0_f64;
        for index in 0..period {
            sum = value_at(index).mul_add(weights[index], sum);
        }
        let result = sum / coefficient;
        return if result.is_nan() {
            EHMA_CANONICAL_QNAN_V2
        } else {
            result
        };
    }
    if max_abs_input == 0.0 {
        return 0.0;
    }

    let scale = floor_power_of_two_scale_v2(max_abs_input);
    let anchor = value_at(0) / scale;
    let mut shifted = EhmaDot2V2::default();
    for (index, &weight) in weights.iter().enumerate().take(period) {
        let normalized_value = value_at(index) / scale;
        shifted.add_product(normalized_value - anchor, weight);
    }
    let result = scale * (anchor + shifted.value() / coefficient);
    if result == 0.0 { 0.0 } else { result }
}

impl<'a> AsRef<[f64]> for EhmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhmaData::Slice(slice) => slice,
            EhmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EhmaParams {
    pub period: Option<usize>,
}

impl Default for EhmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct EhmaInput<'a> {
    pub data: EhmaData<'a>,
    pub params: EhmaParams,
}

impl<'a> EhmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhmaParams) -> Self {
        Self {
            data: EhmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EhmaParams) -> Self {
        Self {
            data: EhmaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhmaParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EhmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EhmaOutput, EhmaError> {
        let p = EhmaParams {
            period: self.period,
        };
        let i = EhmaInput::from_candles(c, "close", p);
        ehma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhmaOutput, EhmaError> {
        let p = EhmaParams {
            period: self.period,
        };
        let i = EhmaInput::from_slice(d, p);
        ehma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhmaStream, EhmaError> {
        let p = EhmaParams {
            period: self.period,
        };
        EhmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EhmaError {
    #[error("ehma: Input data slice is empty.")]
    EmptyInputData,

    #[error("ehma: All values are NaN.")]
    AllValuesNaN,

    #[error("ehma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("ehma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("ehma: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("ehma: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("ehma: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("ehma: size overflow while computing {what}")]
    SizeOverflow { what: &'static str },
}

#[inline]
pub fn ehma(input: &EhmaInput) -> Result<EhmaOutput, EhmaError> {
    ehma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ehma_prepare<'a>(
    input: &'a EhmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], AVec<f64>, usize, usize, f64, Kernel), EhmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EhmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhmaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(EhmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(EhmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let (weights, coefficient) = build_hann_weights_v2(period);

    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            k => k,
        },
        k => k,
    };

    Ok((data, weights, period, first, coefficient, chosen))
}

pub fn ehma_with_kernel(input: &EhmaInput, kernel: Kernel) -> Result<EhmaOutput, EhmaError> {
    let (data, weights, period, first, coefficient, chosen) = ehma_prepare(input, kernel)?;

    let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);

    ehma_compute_into(data, &weights, period, first, coefficient, chosen, &mut out);

    Ok(EhmaOutput { values: out })
}

#[inline]
pub fn ehma_into_slice(dst: &mut [f64], input: &EhmaInput, kern: Kernel) -> Result<(), EhmaError> {
    let (data, weights, period, first, coefficient, chosen) = ehma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(EhmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    ehma_compute_into(data, &weights, period, first, coefficient, chosen, dst);

    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn ehma_into(input: &EhmaInput, out: &mut [f64]) -> Result<(), EhmaError> {
    let (data, weights, period, first, coefficient, chosen) = ehma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(EhmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let warmup_end = first + period - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warm = warmup_end.min(out.len());
    for v in &mut out[..warm] {
        *v = qnan;
    }

    ehma_compute_into(data, &weights, period, first, coefficient, chosen, out);

    Ok(())
}

#[inline(always)]
fn ehma_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    coefficient: f64,
    _kernel: Kernel,
    out: &mut [f64],
) {
    for index in (first + period - 1)..data.len() {
        let start = index + 1 - period;
        out[index] = ehma_stable_window_indexed_v2(period, weights, coefficient, |offset| {
            data[start + offset]
        });
    }
}

#[inline]
pub fn ehma_scalar(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    coefficient: f64,
    out: &mut [f64],
) {
    assert_eq!(weights.len(), period, "weights.len() must equal `period`");
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    ehma_compute_into(
        data,
        weights,
        period,
        first_val,
        coefficient,
        Kernel::Scalar,
        out,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)] // Retained as the existing low-level API; V2 forbids reassociation.
unsafe fn ehma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    coefficient: f64,
    out: &mut [f64],
) {
    ehma_compute_into(
        data,
        weights,
        period,
        first_val,
        coefficient,
        Kernel::Avx2,
        out,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub fn ehma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    coefficient: f64,
    out: &mut [f64],
) {
    ehma_compute_into(
        data,
        weights,
        period,
        first_val,
        coefficient,
        Kernel::Avx512,
        out,
    );
}

#[derive(Debug, Clone)]
pub struct EhmaStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    weights: AVec<f64>,
    coefficient: f64,
}

impl EhmaStream {
    pub fn try_new(params: EhmaParams) -> Result<Self, EhmaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(EhmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let (weights, coefficient) = build_hann_weights_v2(period);

        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            weights,
            coefficient,
        })
    }

    #[inline(always)]
    fn recompute_full(&self) -> Option<f64> {
        let head = self.head;
        let period = self.period;
        Some(ehma_stable_window_indexed_v2(
            period,
            &self.weights,
            self.coefficient,
            |offset| self.buffer[(head + offset) % period],
        ))
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled {
            if self.head == 0 {
                self.filled = true;
                return self.recompute_full();
            } else {
                return None;
            }
        }
        self.recompute_full()
    }
}

#[derive(Clone, Debug)]
pub struct EhmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EhmaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EhmaBatchBuilder {
    range: EhmaBatchRange,
    kernel: Kernel,
}

impl EhmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<EhmaBatchOutput, EhmaError> {
        ehma_batch_with_kernel_slice(data, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EhmaBatchOutput, EhmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EhmaBatchOutput, EhmaError> {
        EhmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EhmaBatchOutput, EhmaError> {
        EhmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
}

#[derive(Clone, Debug)]
pub struct EhmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhmaBatchOutput {
    pub fn row_for_params(&self, p: &EhmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }

    pub fn values_for(&self, p: &EhmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid(r: &EhmaBatchRange) -> Vec<EhmaParams> {
    let (start, end, step) = r.period;

    if step == 0 {
        return vec![EhmaParams {
            period: Some(start),
        }];
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out = Vec::new();
    let mut p = lo;
    loop {
        out.push(EhmaParams { period: Some(p) });
        if p == hi {
            break;
        }
        match p.checked_add(step) {
            Some(next) if next > p && next <= hi => {
                p = next;
            }
            _ => {
                break;
            }
        }
    }
    out
}

#[inline(always)]
pub fn ehma_batch_slice(
    data: &[f64],
    sweep: &EhmaBatchRange,
    kern: Kernel,
) -> Result<EhmaBatchOutput, EhmaError> {
    ehma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ehma_batch_par_slice(
    data: &[f64],
    sweep: &EhmaBatchRange,
    kern: Kernel,
) -> Result<EhmaBatchOutput, EhmaError> {
    ehma_batch_inner(data, sweep, kern, true)
}

pub fn ehma_batch_with_kernel(
    data: &[f64],
    sweep: &EhmaBatchRange,
    k: Kernel,
) -> Result<EhmaBatchOutput, EhmaError> {
    let kernel = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx2Batch,
            other => other,
        },
        other if other.is_batch() => other,
        other => return Err(EhmaError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ehma_batch_inner(data, sweep, simd, true)
}

pub fn ehma_batch_with_kernel_slice(
    data: &[f64],
    sweep: &EhmaBatchRange,
    k: Kernel,
) -> Result<EhmaBatchOutput, EhmaError> {
    ehma_batch_with_kernel(data, sweep, k)
}

#[inline(always)]
fn ehma_batch_inner(
    data: &[f64],
    sweep: &EhmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EhmaBatchOutput, EhmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(EhmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }

    let cols = data.len();
    if cols == 0 {
        return Err(EhmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(EhmaError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let rows = combos.len();
    let _ = rows
        .checked_mul(cols)
        .ok_or(EhmaError::SizeOverflow { what: "rows*cols" })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let do_row = |row: usize, row_dst: &mut [f64]| {
        let period = combos[row].period.unwrap();

        let (w, coefficient) = build_hann_weights_v2(period);

        ehma_compute_into(data, &w, period, first, coefficient, kern, row_dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, dst)| do_row(row, dst));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                do_row(row, dst);
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            do_row(row, dst);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(EhmaBatchOutput {
        values,
        combos: combos.clone(),
        rows: combos.len(),
        cols,
    })
}

#[inline(always)]
pub fn ehma_batch_inner_into(
    data: &[f64],
    sweep: &EhmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EhmaParams>, EhmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(EhmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }

    let cols = data.len();
    if cols == 0 {
        return Err(EhmaError::EmptyInputData);
    }
    let rows = combos.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(EhmaError::SizeOverflow { what: "rows*cols" })?;
    if out.len() != expected {
        return Err(EhmaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhmaError::AllValuesNaN)?;
    if cols - first < combos.iter().map(|c| c.period.unwrap()).max().unwrap() {
        return Err(EhmaError::NotEnoughValidData {
            needed: combos.iter().map(|c| c.period.unwrap()).max().unwrap(),
            valid: cols - first,
        });
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let (weights, coefficient) = build_hann_weights_v2(p);
        let row_out = unsafe {
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        ehma_compute_into(data, &weights, p, first, coefficient, kern, row_out);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, ch)| do_row(r, ch));
        }
        #[cfg(target_arch = "wasm32")]
        for (r, ch) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, ch);
        }
    } else {
        for (r, ch) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, ch);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    use crate::skip_if_unsupported;

    macro_rules! generate_all_ehma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                mod avx_tests {
                    use super::*;
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                mod wasm_tests {
                    use super::*;
                    $(
                        #[test]
                        fn [<$test_fn _simd128_f64>]() {
                            let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                        }
                    )*
                }
            }
        };
    }

    const REVIEWED_ROUTEABLE_ROW_13_BITS_V2: u64 = 0x3ff1_3338_cd76_5d61;

    fn reviewed_routeable_row_13_v2() -> [f64; 14] {
        [
            0x3ff1_335e_310d_bf05,
            0x3ff1_3317_9f58_f63b,
            0x3ff1_3342_4cab_aa3d,
            0x3ff1_330f_a73c_f04b,
            0x3ff1_334d_3466_391a,
            0x3ff1_332d_6ece_13f5,
            0x3ff1_335a_34ff_bc0d,
            0x3ff1_3324_6a42_93f9,
            0x3ff1_333f_5d0d_2150,
            0x3ff1_3319_4cd8_1fe7,
            0x3ff1_334c_5da6_a444,
            0x3ff1_3366_4401_b790,
            0x3ff1_331f_b24c_eec6,
            0x3ff1_334a_5f9f_a2c8,
        ]
        .map(f64::from_bits)
    }

    fn assert_reviewed_row_13_v2(value: f64, route: &str) {
        assert_eq!(
            value.to_bits(),
            REVIEWED_ROUTEABLE_ROW_13_BITS_V2,
            "{route} must use the reviewed EHMA f64 v2 authority"
        );
    }

    #[test]
    fn ehma_f64_v2_pins_coefficients_and_exact_symmetry() {
        assert_eq!(
            EHMA_F64_AUTHORITY_V2,
            "ehma_hann_f64_msun_ddangle_symmetric_pow2_anchored_dot2_v2"
        );

        for period in 1..=512 {
            let (weights, coefficient) = build_hann_weights_v2(period);
            assert!(coefficient.is_finite() && coefficient > 0.0);
            for index in 0..period {
                assert_eq!(
                    weights[index].to_bits(),
                    weights[period - 1 - index].to_bits(),
                    "period={period}, index={index}"
                );
            }
        }

        for (period, first_bits, middle_bits, coefficient_bits) in [
            (
                1,
                0x4000_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x4000_0000_0000_0000,
            ),
            (
                2,
                0x3ff8_0000_0000_0001,
                0x3ff8_0000_0000_0001,
                0x4008_0000_0000_0001,
            ),
            (
                3,
                0x3fef_ffff_ffff_fffe,
                0x4000_0000_0000_0000,
                0x400f_ffff_ffff_ffff,
            ),
            (
                14,
                0x3fb6_21e2_8804_0356,
                0x3fff_a67e_193d_003f,
                0x402e_0000_0000_0000,
            ),
            (
                512,
                0x3f13_a97e_353f_b772,
                0x3fff_ffec_5675_b5eb,
                0x4080_0800_0000_0000,
            ),
        ] {
            let (weights, coefficient) = build_hann_weights_v2(period);
            assert_eq!(weights[0].to_bits(), first_bits, "period={period}");
            assert_eq!(
                weights[(period - 1) / 2].to_bits(),
                middle_bits,
                "period={period}"
            );
            assert_eq!(coefficient.to_bits(), coefficient_bits, "period={period}");
        }
    }

    #[test]
    fn ehma_f64_v2_reviewed_row_is_exact_on_every_cpu_route() -> Result<(), Box<dyn Error>> {
        let data = reviewed_routeable_row_13_v2();
        let input = EhmaInput::from_slice(&data, EhmaParams { period: Some(14) });

        for kernel in [Kernel::Scalar, Kernel::Auto, Kernel::Avx2, Kernel::Avx512] {
            let output = ehma_with_kernel(&input, kernel)?;
            assert_reviewed_row_13_v2(output.values[13], &format!("direct {kernel:?}"));

            let mut into = vec![0.0; data.len()];
            ehma_into_slice(&mut into, &input, kernel)?;
            assert_reviewed_row_13_v2(into[13], &format!("into {kernel:?}"));
        }

        let mut auto_into = vec![0.0; data.len()];
        ehma_into(&input, &mut auto_into)?;
        assert_reviewed_row_13_v2(auto_into[13], "auto into");

        let sweep = EhmaBatchRange {
            period: (14, 14, 0),
        };
        for kernel in [
            Kernel::ScalarBatch,
            Kernel::Auto,
            Kernel::Avx2Batch,
            Kernel::Avx512Batch,
        ] {
            let output = ehma_batch_with_kernel(&data, &sweep, kernel)?;
            assert_reviewed_row_13_v2(output.values[13], &format!("batch {kernel:?}"));
        }

        for parallel in [false, true] {
            let mut into = vec![0.0; data.len()];
            ehma_batch_inner_into(&data, &sweep, Kernel::Avx2, parallel, &mut into)?;
            assert_reviewed_row_13_v2(into[13], &format!("batch into parallel={parallel}"));
        }

        let mut stream = EhmaStream::try_new(EhmaParams { period: Some(14) })?;
        let mut streamed = None;
        for value in data {
            streamed = stream.update(value);
        }
        assert_reviewed_row_13_v2(streamed.expect("period-14 stream must be ready"), "stream");
        Ok(())
    }

    #[test]
    fn ehma_f64_v2_pins_constants_offsets_period_edges_and_gaps() -> Result<(), Box<dyn Error>> {
        for period in [1, 2, 3, 14, 31, 64, 127, 256, 512] {
            let constant = f64::from_bits(0x5f30_0000_0000_0000);
            let data = vec![constant; period];
            let input = EhmaInput::from_slice(
                &data,
                EhmaParams {
                    period: Some(period),
                },
            );
            assert_eq!(
                ehma_with_kernel(&input, Kernel::Scalar)?.values[period - 1].to_bits(),
                constant.to_bits()
            );
        }

        let subnormal = f64::from_bits(1);
        let subnormal_data = [subnormal; 14];
        let subnormal_input =
            EhmaInput::from_slice(&subnormal_data, EhmaParams { period: Some(14) });
        assert_eq!(
            ehma_with_kernel(&subnormal_input, Kernel::Scalar)?.values[13].to_bits(),
            1
        );

        let base_bits = 0x42b0_0000_0000_0000_u64;
        let offsets = [-7_i64, -2, 5, 1, -4, 8, -1, 3, -6, 7, 2, -3, 6, -5];
        let large_offset: Vec<f64> = offsets
            .iter()
            .map(|offset| f64::from_bits((base_bits as i64 + offset) as u64))
            .collect();
        let large_input = EhmaInput::from_slice(&large_offset, EhmaParams { period: Some(14) });
        let large = ehma_with_kernel(&large_input, Kernel::Scalar)?.values[13];
        assert_eq!(large.to_bits(), base_bits + 2);

        let reversed: Vec<f64> = large_offset.iter().copied().rev().collect();
        let reversed_input = EhmaInput::from_slice(&reversed, EhmaParams { period: Some(14) });
        assert_eq!(
            ehma_with_kernel(&reversed_input, Kernel::Scalar)?.values[13].to_bits(),
            large.to_bits()
        );

        let mut opposite_extremes = vec![f64::MAX; 14];
        opposite_extremes[7..].fill(-f64::MAX);
        let opposite_input =
            EhmaInput::from_slice(&opposite_extremes, EhmaParams { period: Some(14) });
        assert_eq!(
            ehma_with_kernel(&opposite_input, Kernel::Scalar)?.values[13].to_bits(),
            0
        );

        let positive_infinity = [f64::INFINITY; 14];
        let positive_infinity_input =
            EhmaInput::from_slice(&positive_infinity, EhmaParams { period: Some(14) });
        assert_eq!(
            ehma_with_kernel(&positive_infinity_input, Kernel::Scalar)?.values[13].to_bits(),
            f64::INFINITY.to_bits()
        );
        let mut mixed_infinity = positive_infinity;
        mixed_infinity[13] = f64::NEG_INFINITY;
        let mixed_infinity_input =
            EhmaInput::from_slice(&mixed_infinity, EhmaParams { period: Some(14) });
        assert_eq!(
            ehma_with_kernel(&mixed_infinity_input, Kernel::Scalar)?.values[13].to_bits(),
            EHMA_CANONICAL_QNAN_V2.to_bits()
        );

        let clean: Vec<f64> = (0..40)
            .map(|index| f64::from_bits(0x3ff0_0000_0000_0000 + index * 0x101))
            .collect();
        let clean_input = EhmaInput::from_slice(&clean, EhmaParams { period: Some(14) });
        let clean_output = ehma_with_kernel(&clean_input, Kernel::Scalar)?;
        let mut gapped = clean.clone();
        gapped[15] = f64::NAN;
        let gapped_input = EhmaInput::from_slice(&gapped, EhmaParams { period: Some(14) });
        let gapped_output = ehma_with_kernel(&gapped_input, Kernel::Scalar)?;
        for index in 15..=28 {
            assert_eq!(
                gapped_output.values[index].to_bits(),
                EHMA_CANONICAL_QNAN_V2.to_bits()
            );
        }
        for index in 29..clean.len() {
            assert_eq!(
                gapped_output.values[index].to_bits(),
                clean_output.values[index].to_bits()
            );
        }
        Ok(())
    }

    fn check_ehma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = EhmaInput::from_candles(&candles, "close", EhmaParams::default());
        let result = ehma_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());

        for i in 0..13 {
            assert!(
                result.values[i].is_nan(),
                "[{}] Value at {} should be NaN",
                test_name,
                i
            );
        }

        for i in 13..result.values.len().min(100) {
            assert!(
                !result.values[i].is_nan(),
                "[{}] Value at {} should not be NaN",
                test_name,
                i
            );
            assert!(
                result.values[i].is_finite(),
                "[{}] Value at {} should be finite",
                test_name,
                i
            );
        }

        Ok(())
    }

    fn check_ehma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let data: Vec<f64> = candles.close[0..18].to_vec();

        let params = EhmaParams { period: Some(14) };
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), data.len());

        for i in 0..13 {
            assert!(
                result.values[i].is_nan(),
                "Value at index {} should be NaN",
                i
            );
        }

        for i in 13..result.values.len() {
            assert!(
                !result.values[i].is_nan(),
                "Value at index {} should not be NaN",
                i
            );
            assert!(
                result.values[i].is_finite(),
                "Value at index {} should be finite",
                i
            );
        }

        let min_data = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_data = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        for i in 13..result.values.len() {
            let tolerance = (max_data - min_data) * 0.1;
            assert!(
                result.values[i] >= min_data - tolerance
                    && result.values[i] <= max_data + tolerance,
                "EHMA value {} at index {} is outside reasonable range [{}, {}]",
                result.values[i],
                i,
                min_data - tolerance,
                max_data + tolerance
            );
        }

        println!(
            "[{}] EHMA value at index 13: {}",
            test_name, result.values[13]
        );
        println!(
            "[{}] EHMA value at index 14: {}",
            test_name, result.values[14]
        );
        println!(
            "[{}] EHMA value at index 15: {}",
            test_name, result.values[15]
        );
        println!(
            "[{}] EHMA value at index 16: {}",
            test_name, result.values[16]
        );
        println!(
            "[{}] EHMA value at index 17: {}",
            test_name, result.values[17]
        );

        Ok(())
    }

    fn check_ehma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: Vec<f64> = vec![];
        let params = EhmaParams::default();
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma_with_kernel(&input, kernel);
        assert!(
            matches!(result, Err(EhmaError::EmptyInputData)),
            "[{}] EHMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ehma_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 20];
        let params = EhmaParams::default();
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma_with_kernel(&input, kernel);
        assert!(
            matches!(result, Err(EhmaError::AllValuesNaN)),
            "[{}] EHMA should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_ehma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = EhmaParams { period: Some(0) };
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma_with_kernel(&input, kernel);
        assert!(
            matches!(result, Err(EhmaError::InvalidPeriod { .. })),
            "[{}] EHMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_ehma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = EhmaParams { period: Some(10) };
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma_with_kernel(&input, kernel);
        assert!(
            matches!(result, Err(EhmaError::InvalidPeriod { .. })),
            "[{}] EHMA should fail when period exceeds data length",
            test_name
        );
        Ok(())
    }

    fn check_ehma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![42.0];
        let params = EhmaParams { period: Some(5) };
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma_with_kernel(&input, kernel);
        assert!(
            matches!(
                result,
                Err(EhmaError::InvalidPeriod { .. }) | Err(EhmaError::NotEnoughValidData { .. })
            ),
            "[{}] EHMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ehma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = EhmaParams { period: None };
        let input = EhmaInput::from_candles(&candles, "close", default_params);
        let output = ehma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = EhmaParams { period: Some(14) };
        let first_input = EhmaInput::from_candles(&candles, "close", first_params.clone());
        let first_result = ehma_with_kernel(&first_input, kernel)?;

        let second_input = EhmaInput::from_slice(&first_result.values, first_params);
        let second_result = ehma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        let valid_count = second_result
            .values
            .iter()
            .zip(first_result.values.iter())
            .filter(|(a, b)| !a.is_nan() && !b.is_nan() && (*a - *b).abs() > 1e-10)
            .count();

        assert!(
            valid_count > 0,
            "[{}] EHMA reinput should produce different values",
            test_name
        );

        Ok(())
    }

    fn check_ehma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = EhmaInput::from_candles(&candles, "close", EhmaParams { period: Some(14) });
        let res = ehma_with_kernel(&input, kernel)?;
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

    fn check_ehma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let period = 14;

        let input = EhmaInput::from_candles(
            &candles,
            "close",
            EhmaParams {
                period: Some(period),
            },
        );
        let batch_output = ehma_with_kernel(&input, kernel)?.values;

        let mut stream = EhmaStream::try_new(EhmaParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(y) => stream_values.push(y),
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
                "[{}] EHMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_ehma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            EhmaParams::default(),
            EhmaParams { period: Some(5) },
            EhmaParams { period: Some(10) },
            EhmaParams { period: Some(20) },
            EhmaParams { period: Some(50) },
            EhmaParams { period: Some(100) },
        ];

        for (_param_idx, params) in test_params.iter().enumerate() {
            let input = EhmaInput::from_candles(&candles, "close", params.clone());
            let output = ehma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ehma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_ehma_tests!(
        check_ehma_partial_params,
        check_ehma_accuracy,
        check_ehma_default_candles,
        check_ehma_zero_period,
        check_ehma_period_exceeds_length,
        check_ehma_very_small_dataset,
        check_ehma_empty_input,
        check_ehma_all_nan,
        check_ehma_reinput,
        check_ehma_nan_handling,
        check_ehma_streaming,
        check_ehma_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = EhmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = EhmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());
        assert!(
            row.iter()
                .skip(def.period.unwrap() - 1)
                .any(|v| v.is_finite())
        );

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = EhmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 20, 2)
            .apply_candles(&c, "close")?;

        assert_eq!(output.rows, 6);
        assert_eq!(output.cols, c.close.len());
        for (i, p) in output.combos.iter().enumerate() {
            assert_eq!(p.period.unwrap(), 10 + 2 * i);
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![(5, 15, 5), (10, 30, 10), (14, 14, 1), (20, 50, 15)];

        for (_cfg_idx, &(start, stop, step)) in test_configs.iter().enumerate() {
            let output = EhmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, stop, step)
                .apply_candles(&c, "close")?;

            for idx in 0..output.values.len() {
                let val = output.values[idx];
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at index {}",
                        test, val, bits, idx
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
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_ehma_simd128_correctness() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let period = 10;

        let params = EhmaParams {
            period: Some(period),
        };
        let input = EhmaInput::from_slice(&data, params);
        let scalar_output = ehma_with_kernel(&input, Kernel::Scalar).unwrap();

        let simd128_output = ehma_with_kernel(&input, Kernel::Scalar).unwrap();

        assert_eq!(scalar_output.values.len(), simd128_output.values.len());
        for (i, (scalar_val, simd_val)) in scalar_output
            .values
            .iter()
            .zip(simd128_output.values.iter())
            .enumerate()
        {
            assert!(
                (scalar_val - simd_val).abs() < 1e-10,
                "SIMD128 mismatch at index {}: scalar={}, simd128={}",
                i,
                scalar_val,
                simd_val
            );
        }
    }

    #[test]
    fn check_ehma_batch_inner_into_warm_and_no_poison() -> Result<(), Box<dyn std::error::Error>> {
        use crate::utilities::data_loader::read_candles_from_vortex;
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let sweep = EhmaBatchRange {
            period: (10, 14, 2),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = c.close.len();
        let mut out = vec![0.0f64; rows * cols];

        ehma_batch_inner_into(&c.close, &sweep, Kernel::Scalar, true, &mut out)?;

        for (r, p) in combos.iter().enumerate() {
            let warm = p.period.unwrap() - 1;
            assert!(out[r * cols..r * cols + warm].iter().all(|v| v.is_nan()));
        }

        #[cfg(debug_assertions)]
        for &v in &out {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                b != 0x11111111_11111111 && b != 0x22222222_22222222 && b != 0x33333333_33333333
            );
        }
        Ok(())
    }

    #[test]
    fn test_ehma_weight_debug() {
        let period = 14;
        let mut weights = vec![0.0; period];
        let mut coef_sum = 0.0;

        use std::f64::consts::PI;

        println!("Current weight calculation (i from 1 to period):");
        for i in 1..=period {
            let cosine = 1.0 - ((2.0 * PI * i as f64) / (period + 1) as f64).cos();
            weights[period - i] = cosine;
            coef_sum += cosine;
            println!(
                "  i={}, cosine={:.8}, stored at index {}",
                i,
                cosine,
                period - i
            );
        }

        println!("\nFinal weights array (index -> value):");
        for (idx, w) in weights.iter().enumerate() {
            println!("  weights[{}] = {:.8}", idx, w);
        }

        println!("\nSum of weights: {:.8}", coef_sum);
        println!("Normalization factor: {:.8}", 1.0 / coef_sum);

        let mut weights2 = vec![0.0; period];
        let mut coef_sum2 = 0.0;

        println!("\nAlternative weight calculation (reversed storage):");
        for i in 1..=period {
            let cosine = 1.0 - ((2.0 * PI * i as f64) / (period + 1) as f64).cos();
            weights2[i - 1] = cosine;
            coef_sum2 += cosine;
            println!("  i={}, cosine={:.8}, stored at index {}", i, cosine, i - 1);
        }

        println!("\nAlternative weights array:");
        for (idx, w) in weights2.iter().enumerate() {
            println!("  weights2[{}] = {:.8}", idx, w);
        }
    }

    #[test]
    fn test_ehma_reference_values() {
        let data = vec![
            59500.0, 59450.0, 59420.0, 59380.0, 59350.0, 59320.0, 59310.0, 59300.0, 59280.0,
            59260.0, 59250.0, 59240.0, 59230.0, 59220.0, 59210.0, 59200.0, 59190.0, 59180.0,
        ];

        let params = EhmaParams { period: Some(14) };
        let input = EhmaInput::from_slice(&data, params);
        let result = ehma(&input).expect("EHMA calculation failed");

        let expected_values = vec![
            59309.74802712,
            59291.69687546,
            59275.88831852,
            59261.82816317,
            59249.06571993,
        ];

        for (i, &expected) in expected_values.iter().enumerate() {
            let idx = 13 + i;
            assert!(
                (result.values[idx] - expected).abs() < 0.0001,
                "Value at index {} should be {:.8}, got {:.8}",
                idx,
                expected,
                result.values[idx]
            );
        }

        assert_eq!(result.values.len(), data.len());

        for i in 0..13.min(result.values.len()) {
            assert!(
                result.values[i].is_nan(),
                "Value at index {} should be NaN",
                i
            );
        }

        for i in 13..result.values.len() {
            assert!(
                !result.values[i].is_nan(),
                "Value at index {} should not be NaN",
                i
            );
            assert!(
                result.values[i].is_finite(),
                "Value at index {} should be finite",
                i
            );
        }
    }

    #[test]
    fn test_ehma_pinescript_parity() {
        println!("\n=== EHMA PineScript Parity Investigation ===\n");

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path).expect("failed to load Vortex fixture");

        let close: Vec<f64> = candles.close[0..100.min(candles.close.len())].to_vec();

        println!(
            "Using Vortex data - first 5 values: {:?}",
            &close[..5.min(close.len())]
        );
        println!("Total data points loaded: {}", close.len());

        let pine_refs = vec![
            59417.85296671,
            59307.66635431,
            59222.28072230,
            59171.41684053,
            59153.35666389,
        ];

        let period = 14;

        println!("Test 1: Standard EHMA with close prices");
        let params = EhmaParams {
            period: Some(period),
        };
        let input = EhmaInput::from_slice(&close, params.clone());
        let result1 = ehma(&input).expect("EHMA calculation failed");

        println!("  Values from index 13-30:");
        for i in 13..30.min(result1.values.len()) {
            if !result1.values[i].is_nan() {
                println!("    Index {}: {:.8}", i, result1.values[i]);

                for (ref_idx, &ref_val) in pine_refs.iter().enumerate() {
                    let diff = (result1.values[i] - ref_val).abs();
                    if diff < 1.0 {
                        println!(
                            "      -> Very close to Reference[{}]! (diff: {:.8})",
                            ref_idx, diff
                        );
                    }
                }
            }
        }

        println!("\nTest 2: EHMA with PineScript warmup (zero-padding)");
        let mut padded = Vec::with_capacity(period - 1 + close.len());
        padded.resize(period - 1, 0.0);
        padded.extend_from_slice(&close);

        let input2 = EhmaInput::from_slice(&padded, params.clone());
        let result2 = ehma(&input2).expect("EHMA calculation failed");
        let out2 = &result2.values[(period - 1)..];

        for i in 0..pine_refs.len().min(out2.len()) {
            println!("  Index {}: {:.8}", i, out2[i]);
        }

        println!("\nTest 3: EHMA with non-repaint shift (1-bar historical lag)");
        let hist = &close[..close.len().saturating_sub(1)];
        let input3 = EhmaInput::from_slice(hist, params.clone());
        let result3 = ehma(&input3).expect("EHMA calculation failed");

        let mut out3 = vec![f64::NAN; close.len()];
        out3[1..1 + result3.values.len()].copy_from_slice(&result3.values);

        for i in 14..(14 + pine_refs.len()).min(out3.len()) {
            if !out3[i].is_nan() {
                println!("  Index {}: {:.8}", i, out3[i]);
            }
        }

        println!("\nTest 4: Simulated HLCC4 source");
        let mut hlcc4 = vec![];
        for i in 0..close.len() {
            let high = close[i] * 1.001;
            let low = close[i] * 0.999;
            let hlcc4_val = (high + low + close[i] + close[i]) / 4.0;
            hlcc4.push(hlcc4_val);
        }

        let input4 = EhmaInput::from_slice(&hlcc4, params.clone());
        let result4 = ehma(&input4).expect("EHMA calculation failed");

        for i in 13..(13 + pine_refs.len()).min(result4.values.len()) {
            println!("  Index {}: {:.8}", i, result4.values[i]);
        }

        println!("\nTest 5: Zero-padded + non-repaint shift");
        let mut padded5 = Vec::with_capacity(period - 1 + close.len() - 1);
        padded5.resize(period - 1, 0.0);
        padded5.extend_from_slice(&close[..close.len() - 1]);

        let input5 = EhmaInput::from_slice(&padded5, params.clone());
        let result5 = ehma(&input5).expect("EHMA calculation failed");
        let mut out5 = vec![f64::NAN; close.len()];
        let tmp5 = &result5.values[(period - 1)..];
        if tmp5.len() > 0 {
            out5[1..1 + tmp5.len()].copy_from_slice(tmp5);
        }

        for i in 0..pine_refs.len().min(out5.len()) {
            if !out5[i].is_nan() {
                println!("  Index {}: {:.8}", i, out5[i]);
            }
        }

        println!("\n=== Comparison with PineScript Reference Values ===");
        for (i, ref_val) in pine_refs.iter().enumerate() {
            println!("Reference[{}]: {:.8}", i, ref_val);

            if 13 + i < result1.values.len() {
                let diff1 = (result1.values[13 + i] - ref_val).abs();
                println!("  Test 1 diff: {:.8}", diff1);
            }

            if i < out2.len() {
                let diff2 = (out2[i] - ref_val).abs();
                println!("  Test 2 diff: {:.8}", diff2);
            }

            if 14 + i < out3.len() && !out3[14 + i].is_nan() {
                let diff3 = (out3[14 + i] - ref_val).abs();
                println!("  Test 3 diff: {:.8}", diff3);
            }

            if 13 + i < result4.values.len() {
                let diff4 = (result4.values[13 + i] - ref_val).abs();
                println!("  Test 4 diff: {:.8}", diff4);
            }

            if i < out5.len() && !out5[i].is_nan() {
                let diff5 = (out5[i] - ref_val).abs();
                println!("  Test 5 diff: {:.8}", diff5);
            }
        }

        println!("\n=== Searching for exact matches ===");
        for (ref_idx, &ref_val) in pine_refs.iter().enumerate() {
            println!("Looking for Reference[{}] = {:.8}", ref_idx, ref_val);

            for (idx, &val) in result1.values.iter().enumerate() {
                if !val.is_nan() {
                    let diff = (val - ref_val).abs();
                    if diff < 0.01 {
                        println!(
                            "  Found close match in Test 1 at index {}: {} (diff: {})",
                            idx, val, diff
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_ehma_into_matches_api() {
        let n = 256usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64;

            data.push((x * 0.03125).sin() * 2.0 + (x * 0.001));
        }

        let input = EhmaInput::from_slice(&data, EhmaParams::default());

        let baseline = ehma(&input).expect("ehma baseline should succeed").values;

        let mut out = vec![0.0; data.len()];
        {
            ehma_into(&input, &mut out).expect("ehma_into should succeed");
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "divergence at index {}: baseline={}, into={}",
                i,
                a,
                b
            );
        }
    }
}
