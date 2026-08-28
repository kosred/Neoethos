use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[inline(always)]
fn correlation_cycle_auto_kernel() -> Kernel {
    Kernel::Scalar
}

const CORRELATION_CYCLE_HALF_PI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
const CORRELATION_CYCLE_TWO_PI: f64 = f64::from_bits(0x4019_21fb_5444_2d18);
const CORRELATION_CYCLE_RADIANS_TO_DEGREES: f64 = f64::from_bits(0x404c_a5dc_1a63_c1f8);

/* FreeBSD msun k_sin/k_cos, medium pi/2 reduction and s_atan.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft. Permission to use, copy, modify, and
 * distribute this software is freely granted, provided this notice is
 * preserved.
 *
 * Correlation Cycle evaluates sine/cosine only for finite positive weights in
 * (0, 2*pi], while atan must cover the full signed f64 ratio domain. The CUDA
 * f64 row mirrors every constant, branch, operation and parenthesisation below
 * so recursive rotation cannot amplify a host-libm/libdevice ULP split.
 */
#[inline(always)]
fn correlation_cycle_ms_k_cos(x: f64, y: f64) -> f64 {
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
fn correlation_cycle_ms_k_sin(x: f64, y: f64, has_tail: bool) -> f64 {
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
fn correlation_cycle_reduce_pio2(x: f64) -> (i32, f64, f64) {
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
fn correlation_cycle_deterministic_sin_cos(x: f64) -> (f64, f64) {
    debug_assert!(x.is_finite() && (0.0..=7.0).contains(&x));
    let high = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if high <= 0x3fe9_21fb {
        return (
            correlation_cycle_ms_k_sin(x, 0.0, false),
            correlation_cycle_ms_k_cos(x, 0.0),
        );
    }

    let (quadrant, y0, y1) = correlation_cycle_reduce_pio2(x);
    let sin = correlation_cycle_ms_k_sin(y0, y1, true);
    let cos = correlation_cycle_ms_k_cos(y0, y1);
    match quadrant & 3 {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    }
}

#[inline(always)]
fn correlation_cycle_deterministic_weight(w: f64, j: usize) -> (f64, f64) {
    let group_start = j & !3usize;
    let mut angle = w * ((group_start as f64) + 1.0);
    let mut offset = j & 3usize;
    while offset != 0 {
        angle += w;
        offset -= 1;
    }
    let (sin, cos) = correlation_cycle_deterministic_sin_cos(angle);
    (cos, -sin)
}

#[inline(always)]
fn correlation_cycle_deterministic_atan(input: f64) -> f64 {
    let atan_hi = [
        f64::from_bits(0x3fdd_ac67_0561_bb4f),
        f64::from_bits(0x3fe9_21fb_5444_2d18),
        f64::from_bits(0x3fef_730b_d281_f69b),
        f64::from_bits(0x3ff9_21fb_5444_2d18),
    ];
    let atan_lo = [
        f64::from_bits(0x3c7a_2b7f_222f_65e2),
        f64::from_bits(0x3c81_a626_3314_5c07),
        f64::from_bits(0x3c70_0788_7af0_cbbd),
        f64::from_bits(0x3c91_a626_3314_5c07),
    ];
    let coefficients = [
        f64::from_bits(0x3fd5_5555_5555_550d),
        f64::from_bits(0xbfc9_9999_9998_ebc4),
        f64::from_bits(0x3fc2_4924_9200_83ff),
        f64::from_bits(0xbfbc_71c6_fe23_1671),
        f64::from_bits(0x3fb7_45cd_c54c_206e),
        f64::from_bits(0xbfb3_b0f2_af74_9a6d),
        f64::from_bits(0x3fb1_0d66_a0d0_3d51),
        f64::from_bits(0xbfad_de2d_52de_fd9a),
        f64::from_bits(0x3fa9_7b4b_2476_0deb),
        f64::from_bits(0xbfa2_b444_2c6a_6c2f),
        f64::from_bits(0x3f90_ad3a_e322_da11),
    ];

    let mut x = input;
    let mut high = (x.to_bits() >> 32) as u32;
    let sign = high >> 31;
    high &= 0x7fff_ffff;
    if high >= 0x4410_0000 {
        if x.is_nan() {
            return x;
        }
        return if sign != 0 {
            -CORRELATION_CYCLE_HALF_PI
        } else {
            CORRELATION_CYCLE_HALF_PI
        };
    }

    let reduction = if high < 0x3fdc_0000 {
        if high < 0x3e40_0000 {
            return x;
        }
        -1
    } else {
        x = f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff);
        if high < 0x3ff3_0000 {
            if high < 0x3fe6_0000 {
                x = (2.0 * x - 1.0) / (2.0 + x);
                0
            } else {
                x = (x - 1.0) / (x + 1.0);
                1
            }
        } else if high < 0x4003_8000 {
            x = (x - 1.5) / (1.0 + 1.5 * x);
            2
        } else {
            x = -1.0 / x;
            3
        }
    };

    let z = x * x;
    let w = z * z;
    let s1 = z
        * (coefficients[0]
            + w * (coefficients[2]
                + w * (coefficients[4]
                    + w * (coefficients[6] + w * (coefficients[8] + w * coefficients[10])))));
    let s2 = w
        * (coefficients[1]
            + w * (coefficients[3]
                + w * (coefficients[5] + w * (coefficients[7] + w * coefficients[9]))));
    if reduction < 0 {
        return x - x * (s1 + s2);
    }

    let index = reduction as usize;
    let result = atan_hi[index] - ((x * (s1 + s2) - atan_lo[index]) - x);
    if sign != 0 { -result } else { result }
}

#[inline(always)]
fn correlation_cycle_deterministic_angle(real: f64, imag: f64) -> f64 {
    if imag == 0.0 {
        return 0.0;
    }
    let mut angle = (correlation_cycle_deterministic_atan(real / imag) + CORRELATION_CYCLE_HALF_PI)
        * CORRELATION_CYCLE_RADIANS_TO_DEGREES;
    if imag > 0.0 {
        angle -= 180.0;
    }
    angle
}

impl<'a> AsRef<[f64]> for CorrelationCycleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CorrelationCycleData::Slice(slice) => slice,
            CorrelationCycleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CorrelationCycleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleParams {
    pub period: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for CorrelationCycleParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            threshold: Some(9.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleInput<'a> {
    pub data: CorrelationCycleData<'a>,
    pub params: CorrelationCycleParams,
}

impl<'a> CorrelationCycleInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: CorrelationCycleParams,
    ) -> Self {
        Self {
            data: CorrelationCycleData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: CorrelationCycleParams) -> Self {
        Self {
            data: CorrelationCycleData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", CorrelationCycleParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }

    #[inline]
    pub fn get_threshold(&self) -> f64 {
        self.params.threshold.unwrap_or(9.0)
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleBuilder {
    period: Option<usize>,
    threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for CorrelationCycleBuilder {
    fn default() -> Self {
        Self {
            period: None,
            threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CorrelationCycleBuilder {
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
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        let i = CorrelationCycleInput::from_candles(c, "close", p);
        correlation_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        let i = CorrelationCycleInput::from_slice(d, p);
        correlation_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CorrelationCycleStream, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        CorrelationCycleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CorrelationCycleError {
    #[error("correlation_cycle: Empty data provided.")]
    EmptyInputData,
    #[error("correlation_cycle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("correlation_cycle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("correlation_cycle: All values are NaN.")]
    AllValuesNaN,
    #[error("correlation_cycle: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("correlation_cycle: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("correlation_cycle: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("correlation_cycle: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn correlation_cycle(
    input: &CorrelationCycleInput,
) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    correlation_cycle_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn correlation_cycle_with_kernel(
    input: &CorrelationCycleInput,
    kernel: Kernel,
) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let threshold = input.get_threshold();
    let chosen = match kernel {
        Kernel::Auto => correlation_cycle_auto_kernel(),
        k => k,
    };

    let warm_real_imag_angle = first + period;
    let warm_state = first + period + 1;

    let mut real = alloc_with_nan_prefix(len, warm_real_imag_angle);
    let mut imag = alloc_with_nan_prefix(len, warm_real_imag_angle);
    let mut angle = alloc_with_nan_prefix(len, warm_real_imag_angle);
    let mut state = alloc_with_nan_prefix(len, warm_state);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_compute_into(
                data, period, threshold, first, &mut real, &mut imag, &mut angle, &mut state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(
                data, period, threshold, first, &mut real, &mut imag, &mut angle, &mut state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(
                data, period, threshold, first, &mut real, &mut imag, &mut angle, &mut state,
            ),
            _ => unreachable!(),
        }
    }

    Ok(CorrelationCycleOutput {
        real,
        imag,
        angle,
        state,
    })
}

pub fn correlation_cycle_into(
    input: &CorrelationCycleInput,
    out_real: &mut [f64],
    out_imag: &mut [f64],
    out_angle: &mut [f64],
    out_state: &mut [f64],
) -> Result<(), CorrelationCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrelationCycleError::EmptyInputData);
    }

    if out_real.len() != len
        || out_imag.len() != len
        || out_angle.len() != len
        || out_state.len() != len
    {
        let got = *[
            out_real.len(),
            out_imag.len(),
            out_angle.len(),
            out_state.len(),
        ]
        .iter()
        .min()
        .unwrap_or(&0);
        return Err(CorrelationCycleError::OutputLengthMismatch { expected: len, got });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let threshold = input.get_threshold();
    let chosen = correlation_cycle_auto_kernel();

    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warm_ria = (first + period).min(len);
    let warm_s = (first + period + 1).min(len);
    for v in &mut out_real[..warm_ria] {
        *v = qnan;
    }
    for v in &mut out_imag[..warm_ria] {
        *v = qnan;
    }
    for v in &mut out_angle[..warm_ria] {
        *v = qnan;
    }
    for v in &mut out_state[..warm_s] {
        *v = qnan;
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_compute_into(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            _ => correlation_cycle_compute_into(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
        }
    }

    Ok(())
}

#[inline(always)]
pub fn correlation_cycle_into_slices(
    dst_real: &mut [f64],
    dst_imag: &mut [f64],
    dst_angle: &mut [f64],
    dst_state: &mut [f64],
    input: &CorrelationCycleInput,
    kernel: Kernel,
) -> Result<(), CorrelationCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    if dst_real.len() != len
        || dst_imag.len() != len
        || dst_angle.len() != len
        || dst_state.len() != len
    {
        let got = *[
            dst_real.len(),
            dst_imag.len(),
            dst_angle.len(),
            dst_state.len(),
        ]
        .iter()
        .min()
        .unwrap_or(&0);
        return Err(CorrelationCycleError::OutputLengthMismatch { expected: len, got });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let threshold = input.get_threshold();
    let chosen = match kernel {
        Kernel::Auto => correlation_cycle_auto_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_compute_into(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
            _ => correlation_cycle_compute_into(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
        }
    }

    let warm_ria = first + period;
    let warm_s = first + period + 1;
    for v in &mut dst_real[..warm_ria] {
        *v = f64::NAN;
    }
    for v in &mut dst_imag[..warm_ria] {
        *v = f64::NAN;
    }
    for v in &mut dst_angle[..warm_ria] {
        *v = f64::NAN;
    }
    for v in &mut dst_state[..warm_s] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn correlation_cycle_scalar(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    unsafe {
        correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
    }
}

#[inline(always)]
unsafe fn correlation_cycle_window_sums(
    dptr: *const f64,
    cptr: *const f64,
    sptr: *const f64,
    i: usize,
    period: usize,
) -> (f64, f64, f64, f64) {
    let mut sum_x = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_xc = 0.0f64;
    let mut sum_xs = 0.0f64;

    let mut j = 0usize;
    while j + 4 <= period {
        let idx0 = i - (j + 1);
        let idx1 = idx0 - 1;
        let idx2 = idx1 - 1;
        let idx3 = idx2 - 1;

        let mut x0 = *dptr.add(idx0);
        let mut x1 = *dptr.add(idx1);
        let mut x2 = *dptr.add(idx2);
        let mut x3 = *dptr.add(idx3);

        if x0 != x0 {
            x0 = 0.0;
        }
        if x1 != x1 {
            x1 = 0.0;
        }
        if x2 != x2 {
            x2 = 0.0;
        }
        if x3 != x3 {
            x3 = 0.0;
        }

        let c0 = *cptr.add(j);
        let s0 = *sptr.add(j);
        let c1 = *cptr.add(j + 1);
        let s1 = *sptr.add(j + 1);
        let c2 = *cptr.add(j + 2);
        let s2 = *sptr.add(j + 2);
        let c3 = *cptr.add(j + 3);
        let s3 = *sptr.add(j + 3);

        sum_x += x0 + x1 + x2 + x3;
        sum_x2 = x0.mul_add(x0, x1.mul_add(x1, x2.mul_add(x2, x3.mul_add(x3, sum_x2))));
        sum_xc = x0.mul_add(c0, x1.mul_add(c1, x2.mul_add(c2, x3.mul_add(c3, sum_xc))));
        sum_xs = x0.mul_add(s0, x1.mul_add(s1, x2.mul_add(s2, x3.mul_add(s3, sum_xs))));
        j += 4;
    }
    while j < period {
        let idx = i - (j + 1);
        let mut x = *dptr.add(idx);
        if x != x {
            x = 0.0;
        }
        let c = *cptr.add(j);
        let s = *sptr.add(j);
        sum_x += x;
        sum_x2 = x.mul_add(x, sum_x2);
        sum_xc = x.mul_add(c, sum_xc);
        sum_xs = x.mul_add(s, sum_xs);
        j += 1;
    }

    (sum_x, sum_x2, sum_xc, sum_xs)
}

#[inline(always)]
unsafe fn correlation_cycle_compute_into(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    let n = period as f64;
    let w = CORRELATION_CYCLE_TWO_PI / n;

    let mut cos_table = vec![0.0f64; period];
    let mut sin_table = vec![0.0f64; period];

    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;
    let mut sum_cos2 = 0.0f64;
    let mut sum_sin2 = 0.0f64;

    {
        let mut j = 0usize;
        while j + 4 <= period {
            let (c0, ys0) = correlation_cycle_deterministic_weight(w, j);
            *cos_table.get_unchecked_mut(j) = c0;
            *sin_table.get_unchecked_mut(j) = ys0;
            sum_cos += c0;
            sum_sin += ys0;
            sum_cos2 += c0 * c0;
            sum_sin2 += ys0 * ys0;

            let (c1, ys1) = correlation_cycle_deterministic_weight(w, j + 1);
            *cos_table.get_unchecked_mut(j + 1) = c1;
            *sin_table.get_unchecked_mut(j + 1) = ys1;
            sum_cos += c1;
            sum_sin += ys1;
            sum_cos2 += c1 * c1;
            sum_sin2 += ys1 * ys1;

            let (c2, ys2) = correlation_cycle_deterministic_weight(w, j + 2);
            *cos_table.get_unchecked_mut(j + 2) = c2;
            *sin_table.get_unchecked_mut(j + 2) = ys2;
            sum_cos += c2;
            sum_sin += ys2;
            sum_cos2 += c2 * c2;
            sum_sin2 += ys2 * ys2;

            let (c3, ys3) = correlation_cycle_deterministic_weight(w, j + 3);
            *cos_table.get_unchecked_mut(j + 3) = c3;
            *sin_table.get_unchecked_mut(j + 3) = ys3;
            sum_cos += c3;
            sum_sin += ys3;
            sum_cos2 += c3 * c3;
            sum_sin2 += ys3 * ys3;

            j += 4;
        }
        while j < period {
            let (c, ys) = correlation_cycle_deterministic_weight(w, j);
            *cos_table.get_unchecked_mut(j) = c;
            *sin_table.get_unchecked_mut(j) = ys;
            sum_cos += c;
            sum_sin += ys;
            sum_cos2 += c * c;
            sum_sin2 += ys * ys;
            j += 1;
        }
    }

    let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
    let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
    let has_t2 = t2_const > 0.0;
    let has_t4 = t4_const > 0.0;
    let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
    let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

    let start_ria = first + period;
    let start_s = start_ria + 1;

    let mut prev_angle = f64::NAN;

    let dptr = data.as_ptr();
    let cptr = cos_table.as_ptr();
    let sptr = sin_table.as_ptr();
    let len = data.len();

    if start_ria >= len {
        return;
    }

    let rebase_interval = if data[first..].iter().any(|x| x.is_infinite()) {
        1usize
    } else {
        256usize
    };
    let z_re = *cptr;
    let z_im = *sptr;
    let mut last_rebase = start_ria;
    let (mut sum_x, mut sum_x2, mut sum_xc, mut sum_xs) =
        correlation_cycle_window_sums(dptr, cptr, sptr, start_ria, period);

    let mut i = start_ria;
    while i < len {
        let t1 = n.mul_add(sum_x2, -(sum_x * sum_x));
        let mut r_val = 0.0;
        let mut i_val = 0.0;

        if t1 > 0.0 {
            let sqrt_t1 = t1.sqrt();
            if has_t2 {
                let denom = sqrt_t1 * sqrt_t2c;
                if denom > 0.0 {
                    r_val = (n.mul_add(sum_xc, -(sum_x * sum_cos))) / denom;
                }
            }
            if has_t4 {
                let denom = sqrt_t1 * sqrt_t4c;
                if denom > 0.0 {
                    i_val = (n.mul_add(sum_xs, -(sum_x * sum_sin))) / denom;
                }
            }
        }

        *real.get_unchecked_mut(i) = r_val;
        *imag.get_unchecked_mut(i) = i_val;

        let a = correlation_cycle_deterministic_angle(r_val, i_val);
        *angle.get_unchecked_mut(i) = a;

        if i >= start_s {
            let prev = prev_angle;
            let st = if !prev.is_nan() && (a - prev).abs() < threshold {
                if a >= 0.0 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            *state.get_unchecked_mut(i) = st;
        }

        prev_angle = a;
        let next_i = i + 1;
        if next_i < len {
            if next_i - last_rebase >= rebase_interval {
                let sums = correlation_cycle_window_sums(dptr, cptr, sptr, next_i, period);
                sum_x = sums.0;
                sum_x2 = sums.1;
                sum_xc = sums.2;
                sum_xs = sums.3;
                last_rebase = next_i;
            } else {
                let mut x_new = *dptr.add(i);
                let mut x_old = *dptr.add(i - period);
                if x_new != x_new {
                    x_new = 0.0;
                }
                if x_old != x_old {
                    x_old = 0.0;
                }
                let dx = x_new - x_old;
                sum_x += dx;
                sum_x2 += x_new.mul_add(x_new, -(x_old * x_old));
                let s = sum_xc + dx;
                let next_xc = z_re.mul_add(s, -z_im * sum_xs);
                let next_xs = z_im.mul_add(s, z_re * sum_xs);
                sum_xc = next_xc;
                sum_xs = next_xs;
            }
        }
        i = next_i;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn correlation_cycle_avx2(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = period as f64;
    let w = CORRELATION_CYCLE_TWO_PI / n;

    let mut cos_table = vec![0.0f64; period];
    let mut sin_table = vec![0.0f64; period];

    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;
    let mut sum_cos2 = 0.0f64;
    let mut sum_sin2 = 0.0f64;

    for j in 0..period {
        let (c, ys) = correlation_cycle_deterministic_weight(w, j);
        *cos_table.get_unchecked_mut(j) = c;
        *sin_table.get_unchecked_mut(j) = ys;
        sum_cos += c;
        sum_sin += ys;
        sum_cos2 += c * c;
        sum_sin2 += ys * ys;
    }

    let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
    let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
    let has_t2 = t2_const > 0.0;
    let has_t4 = t4_const > 0.0;
    let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
    let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

    let start_ria = first + period;
    let start_s = start_ria + 1;

    #[inline(always)]
    fn hsum256(v: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(v, 1);
            let lo = _mm256_castpd256_pd128(v);
            let sum128 = _mm_add_pd(hi, lo);
            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
        }
    }

    let dptr = data.as_ptr();
    let cptr = cos_table.as_ptr();
    let sptr = sin_table.as_ptr();

    let mut prev_angle = f64::NAN;

    for i in start_ria..data.len() {
        let mut vx = _mm256_setzero_pd();
        let mut vx2 = _mm256_setzero_pd();
        let mut vxc = _mm256_setzero_pd();
        let mut vxs = _mm256_setzero_pd();

        let mut j = 0usize;
        while j + 4 <= period {
            let idx0 = i - (j + 1);
            let x0 = *dptr.add(idx0);
            let x1 = *dptr.add(idx0 - 1);
            let x2 = *dptr.add(idx0 - 2);
            let x3 = *dptr.add(idx0 - 3);
            let mut vx0123 = _mm256_set_pd(x3, x2, x1, x0);

            let ord = _mm256_cmp_pd(vx0123, vx0123, _CMP_ORD_Q);
            vx0123 = _mm256_and_pd(vx0123, ord);

            let vc = _mm256_loadu_pd(cptr.add(j));
            let vs = _mm256_loadu_pd(sptr.add(j));

            vx = _mm256_add_pd(vx, vx0123);
            vx2 = _mm256_fmadd_pd(vx0123, vx0123, vx2);
            vxc = _mm256_fmadd_pd(vx0123, vc, vxc);
            vxs = _mm256_fmadd_pd(vx0123, vs, vxs);

            j += 4;
        }

        let mut sum_x = hsum256(vx);
        let mut sum_x2 = hsum256(vx2);
        let mut sum_xc = hsum256(vxc);
        let mut sum_xs = hsum256(vxs);

        while j < period {
            let idx = i - (j + 1);
            let mut x = *dptr.add(idx);
            if x != x {
                x = 0.0;
            }
            let c = *cptr.add(j);
            let s = *sptr.add(j);
            sum_x += x;
            sum_x2 = x.mul_add(x, sum_x2);
            sum_xc = x.mul_add(c, sum_xc);
            sum_xs = x.mul_add(s, sum_xs);
            j += 1;
        }

        let t1 = n.mul_add(sum_x2, -(sum_x * sum_x));
        let mut r_val = 0.0;
        let mut i_val = 0.0;
        if t1 > 0.0 {
            if has_t2 {
                let denom = t1.sqrt() * sqrt_t2c;
                if denom > 0.0 {
                    r_val = (n.mul_add(sum_xc, -(sum_x * sum_cos))) / denom;
                }
            }
            if has_t4 {
                let denom = t1.sqrt() * sqrt_t4c;
                if denom > 0.0 {
                    i_val = (n.mul_add(sum_xs, -(sum_x * sum_sin))) / denom;
                }
            }
        }

        *real.get_unchecked_mut(i) = r_val;
        *imag.get_unchecked_mut(i) = i_val;

        let a = correlation_cycle_deterministic_angle(r_val, i_val);
        *angle.get_unchecked_mut(i) = a;

        if i >= start_s {
            let prev = prev_angle;
            let st = if !prev.is_nan() && (a - prev).abs() < threshold {
                if a >= 0.0 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            *state.get_unchecked_mut(i) = st;
        }

        prev_angle = a;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn correlation_cycle_avx512(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = period as f64;
    let w = CORRELATION_CYCLE_TWO_PI / n;

    let mut cos_table = vec![0.0f64; period];
    let mut sin_table = vec![0.0f64; period];

    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;
    let mut sum_cos2 = 0.0f64;
    let mut sum_sin2 = 0.0f64;

    for j in 0..period {
        let (c, ys) = correlation_cycle_deterministic_weight(w, j);
        *cos_table.get_unchecked_mut(j) = c;
        *sin_table.get_unchecked_mut(j) = ys;
        sum_cos += c;
        sum_sin += ys;
        sum_cos2 += c * c;
        sum_sin2 += ys * ys;
    }

    let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
    let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
    let has_t2 = t2_const > 0.0;
    let has_t4 = t4_const > 0.0;
    let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
    let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

    let start_ria = first + period;
    let start_s = start_ria + 1;

    #[inline(always)]
    fn hsum512(v: __m512d) -> f64 {
        unsafe {
            let lo = _mm512_castpd512_pd256(v);
            let hi = _mm512_extractf64x4_pd(v, 1);
            let lohi = _mm256_add_pd(lo, hi);
            let hi128 = _mm256_extractf128_pd(lohi, 1);
            let lo128 = _mm256_castpd256_pd128(lohi);
            let sum128 = _mm_add_pd(hi128, lo128);
            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
        }
    }

    let dptr = data.as_ptr();
    let cptr = cos_table.as_ptr();
    let sptr = sin_table.as_ptr();

    let mut prev_angle = f64::NAN;

    for i in start_ria..data.len() {
        let mut vx = _mm512_setzero_pd();
        let mut vx2 = _mm512_setzero_pd();
        let mut vxc = _mm512_setzero_pd();
        let mut vxs = _mm512_setzero_pd();

        let mut j = 0usize;
        while j + 8 <= period {
            let idx0 = i - (j + 1);
            let x0 = *dptr.add(idx0);
            let x1 = *dptr.add(idx0 - 1);
            let x2 = *dptr.add(idx0 - 2);
            let x3 = *dptr.add(idx0 - 3);
            let x4 = *dptr.add(idx0 - 4);
            let x5 = *dptr.add(idx0 - 5);
            let x6 = *dptr.add(idx0 - 6);
            let x7 = *dptr.add(idx0 - 7);

            let mut vx01234567 = _mm512_setr_pd(x0, x1, x2, x3, x4, x5, x6, x7);

            let ordk = _mm512_cmp_pd_mask(vx01234567, vx01234567, _CMP_ORD_Q);
            vx01234567 = _mm512_maskz_mov_pd(ordk, vx01234567);

            let vc = _mm512_loadu_pd(cptr.add(j));
            let vs = _mm512_loadu_pd(sptr.add(j));

            vx = _mm512_add_pd(vx, vx01234567);
            vx2 = _mm512_fmadd_pd(vx01234567, vx01234567, vx2);
            vxc = _mm512_fmadd_pd(vx01234567, vc, vxc);
            vxs = _mm512_fmadd_pd(vx01234567, vs, vxs);

            j += 8;
        }

        let mut sum_x = hsum512(vx);
        let mut sum_x2 = hsum512(vx2);
        let mut sum_xc = hsum512(vxc);
        let mut sum_xs = hsum512(vxs);

        while j < period {
            let idx = i - (j + 1);
            let mut x = *dptr.add(idx);
            if x != x {
                x = 0.0;
            }
            let c = *cptr.add(j);
            let s = *sptr.add(j);
            sum_x += x;
            sum_x2 = x.mul_add(x, sum_x2);
            sum_xc = x.mul_add(c, sum_xc);
            sum_xs = x.mul_add(s, sum_xs);
            j += 1;
        }

        let t1 = n.mul_add(sum_x2, -(sum_x * sum_x));
        let mut r_val = 0.0;
        let mut i_val = 0.0;
        if t1 > 0.0 {
            if has_t2 {
                let denom = t1.sqrt() * sqrt_t2c;
                if denom > 0.0 {
                    r_val = (n.mul_add(sum_xc, -(sum_x * sum_cos))) / denom;
                }
            }
            if has_t4 {
                let denom = t1.sqrt() * sqrt_t4c;
                if denom > 0.0 {
                    i_val = (n.mul_add(sum_xs, -(sum_x * sum_sin))) / denom;
                }
            }
        }

        *real.get_unchecked_mut(i) = r_val;
        *imag.get_unchecked_mut(i) = i_val;

        let a = correlation_cycle_deterministic_angle(r_val, i_val);
        *angle.get_unchecked_mut(i) = a;

        if i >= start_s {
            let prev = prev_angle;
            let st = if !prev.is_nan() && (a - prev).abs() < threshold {
                if a >= 0.0 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            *state.get_unchecked_mut(i) = st;
        }

        prev_angle = a;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512_short(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512_long(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
}

#[inline(always)]
pub unsafe fn correlation_cycle_row_scalar_with_first(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx2_with_first(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx2(data, period, threshold, first, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx512_with_first(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx512(data, period, threshold, first, real, imag, angle, state)
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleStream {
    period: usize,
    threshold: f64,

    buffer: Vec<f64>,
    head: usize,
    filled: bool,

    last: Option<(f64, f64, f64, f64)>,

    sum_x: f64,
    sum_x2: f64,

    phasor_re: f64,
    phasor_im: f64,

    prev_angle: f64,

    n: f64,
    z_re: f64,
    z_im: f64,
    sum_cos: f64,
    sum_sin: f64,
    sqrt_t2c: f64,
    sqrt_t4c: f64,
    has_t2: bool,
    has_t4: bool,
}

impl CorrelationCycleStream {
    pub fn try_new(params: CorrelationCycleParams) -> Result<Self, CorrelationCycleError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(CorrelationCycleError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let threshold = params.threshold.unwrap_or(9.0);

        let n = period as f64;
        let w = CORRELATION_CYCLE_TWO_PI / n;

        let (z_re, z_im) = correlation_cycle_deterministic_weight(w, 0);

        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64;
        let mut sum_cos2 = 0.0f64;
        let mut sum_sin2 = 0.0f64;

        let mut j = 0usize;
        while j + 4 <= period {
            let (c0, ys0) = correlation_cycle_deterministic_weight(w, j);
            let (c1, ys1) = correlation_cycle_deterministic_weight(w, j + 1);
            let (c2, ys2) = correlation_cycle_deterministic_weight(w, j + 2);
            let (c3, ys3) = correlation_cycle_deterministic_weight(w, j + 3);

            sum_cos += c0 + c1 + c2 + c3;
            sum_sin += ys0 + ys1 + ys2 + ys3;
            sum_cos2 += c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
            sum_sin2 += ys0 * ys0 + ys1 * ys1 + ys2 * ys2 + ys3 * ys3;

            j += 4;
        }
        while j < period {
            let (c, ys) = correlation_cycle_deterministic_weight(w, j);
            sum_cos += c;
            sum_sin += ys;
            sum_cos2 += c * c;
            sum_sin2 += ys * ys;
            j += 1;
        }

        let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
        let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
        let has_t2 = t2_const > 0.0;
        let has_t4 = t4_const > 0.0;
        let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
        let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

        Ok(Self {
            period,
            threshold,
            buffer: vec![0.0; period],
            head: 0,
            filled: false,
            last: None,

            sum_x: 0.0,
            sum_x2: 0.0,
            phasor_re: 0.0,
            phasor_im: 0.0,
            prev_angle: f64::NAN,

            n,
            z_re,
            z_im,
            sum_cos,
            sum_sin,
            sqrt_t2c,
            sqrt_t4c,
            has_t2,
            has_t4,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        let x_new = if value.is_nan() { 0.0 } else { value };
        let x_old = self.buffer[self.head];
        self.buffer[self.head] = x_new;
        self.head = (self.head + 1) % self.period;

        self.sum_x += x_new - x_old;

        self.sum_x2 = (x_new * x_new) - (x_old * x_old) + self.sum_x2;

        let dx = x_new - x_old;
        let s = self.phasor_re + dx;

        let new_re = self.z_re.mul_add(s, -self.z_im * self.phasor_im);
        let new_im = self.z_im.mul_add(s, self.z_re * self.phasor_im);

        self.phasor_re = new_re;
        self.phasor_im = new_im;

        let first_wrap_now = !self.filled && self.head == 0;
        if first_wrap_now {
            self.filled = true;
        } else if !self.filled {
            return None;
        }

        let mut sum_x_exact = 0.0f64;
        let mut sum_x2_exact = 0.0f64;
        let mut k = 0usize;
        while k + 4 <= self.period {
            let idx0 = (self.head + k) % self.period;
            let idx1 = (self.head + k + 1) % self.period;
            let idx2 = (self.head + k + 2) % self.period;
            let idx3 = (self.head + k + 3) % self.period;
            let x0 = self.buffer[idx0];
            let x1 = self.buffer[idx1];
            let x2 = self.buffer[idx2];
            let x3 = self.buffer[idx3];
            sum_x_exact += x0 + x1 + x2 + x3;
            sum_x2_exact = x0.mul_add(x0, sum_x2_exact);
            sum_x2_exact = x1.mul_add(x1, sum_x2_exact);
            sum_x2_exact = x2.mul_add(x2, sum_x2_exact);
            sum_x2_exact = x3.mul_add(x3, sum_x2_exact);
            k += 4;
        }
        while k < self.period {
            let idx = (self.head + k) % self.period;
            let x = self.buffer[idx];
            sum_x_exact += x;
            sum_x2_exact = x.mul_add(x, sum_x2_exact);
            k += 1;
        }

        let t1 = self.n.mul_add(sum_x2_exact, -(sum_x_exact * sum_x_exact));

        let mut r_val = 0.0;
        let mut i_val = 0.0;

        if t1 > 0.0 {
            let sqrt_t1 = t1.sqrt();
            if self.has_t2 {
                let denom_r = sqrt_t1 * self.sqrt_t2c;
                if denom_r > 0.0 {
                    r_val = (self
                        .n
                        .mul_add(self.phasor_re, -(sum_x_exact * self.sum_cos)))
                        / denom_r;
                }
            }
            if self.has_t4 {
                let denom_i = sqrt_t1 * self.sqrt_t4c;
                if denom_i > 0.0 {
                    i_val = (self
                        .n
                        .mul_add(self.phasor_im, -(sum_x_exact * self.sum_sin)))
                        / denom_i;
                }
            }
        }

        let ang = correlation_cycle_deterministic_angle(r_val, i_val);

        let st = if self.prev_angle.is_finite() && (ang - self.prev_angle).abs() < self.threshold {
            if ang >= 0.0 { 1.0 } else { -1.0 }
        } else if self.prev_angle.is_finite() {
            0.0
        } else {
            f64::NAN
        };

        self.prev_angle = ang;

        let to_emit = self.last.take();
        self.last = Some((r_val, i_val, ang, st));

        if first_wrap_now { None } else { to_emit }
    }
}

#[derive(Clone, Debug)]
pub struct CorrelationCycleBatchRange {
    pub period: (usize, usize, usize),
    pub threshold: (f64, f64, f64),
}

impl Default for CorrelationCycleBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 269, 1),
            threshold: (9.0, 9.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CorrelationCycleBatchBuilder {
    range: CorrelationCycleBatchRange,
    kernel: Kernel,
}

impl CorrelationCycleBatchBuilder {
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
    pub fn threshold_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.threshold = (start, end, step);
        self
    }
    #[inline]
    pub fn threshold_static(mut self, x: f64) -> Self {
        self.range.threshold = (x, x, 0.0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        correlation_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        CorrelationCycleBatchBuilder::new()
            .kernel(k)
            .apply_slice(data)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        CorrelationCycleBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[inline(always)]
pub fn correlation_cycle_batch_with_kernel(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    k: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CorrelationCycleError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    correlation_cycle_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CorrelationCycleBatchOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
    pub combos: Vec<CorrelationCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CorrelationCycleBatchOutput {
    pub fn row_for_params(&self, p: &CorrelationCycleParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.threshold.unwrap_or(9.0) - p.threshold.unwrap_or(9.0)).abs() < 1e-12
        })
    }
    pub fn values_for(
        &self,
        p: &CorrelationCycleParams,
    ) -> Option<(&[f64], &[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.real[start..start + self.cols],
                &self.imag[start..start + self.cols],
                &self.angle[start..start + self.cols],
                &self.state[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(
    r: &CorrelationCycleBatchRange,
) -> Result<Vec<CorrelationCycleParams>, CorrelationCycleError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CorrelationCycleError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                v = match v.checked_add(step) {
                    Some(n) => n,
                    None => break,
                };
            }
        } else {
            let mut v = start;
            while v >= end {
                vals.push(v);
                v = match v.checked_sub(step) {
                    Some(n) => n,
                    None => break,
                };
            }
        }
        if vals.is_empty() {
            return Err(CorrelationCycleError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CorrelationCycleError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start <= end {
            let mut x = start;
            loop {
                vals.push(x);
                if x >= end {
                    break;
                }
                let next = x + step;
                if !next.is_finite() || next == x {
                    break;
                }
                x = next;
                if x > end + 1e-12 {
                    break;
                }
            }
        } else {
            let mut x = start;
            loop {
                vals.push(x);
                if x <= end {
                    break;
                }
                let next = x - step.abs();
                if !next.is_finite() || next == x {
                    break;
                }
                x = next;
                if x < end - 1e-12 {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(CorrelationCycleError::InvalidRange {
                start: start as usize,
                end: end as usize,
                step: step.abs() as usize,
            });
        }
        Ok(vals)
    }

    let periods = axis_usize(r.period)?;
    let thresholds = axis_f64(r.threshold)?;

    let cap = periods
        .len()
        .checked_mul(thresholds.len())
        .ok_or_else(|| CorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &t in &thresholds {
            out.push(CorrelationCycleParams {
                period: Some(p),
                threshold: Some(t),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn correlation_cycle_batch_slice(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    correlation_cycle_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn correlation_cycle_batch_par_slice(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    correlation_cycle_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn correlation_cycle_batch_inner(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    if data.is_empty() {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let _total = rows
        .checked_mul(cols)
        .ok_or_else(|| CorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;

    let mut real_mu = make_uninit_matrix(rows, cols);
    let mut imag_mu = make_uninit_matrix(rows, cols);
    let mut angle_mu = make_uninit_matrix(rows, cols);
    let mut state_mu = make_uninit_matrix(rows, cols);

    let ria_warm: Vec<usize> = combos
        .iter()
        .map(|c| first.checked_add(c.period.unwrap()).unwrap_or(usize::MAX))
        .collect();
    let st_warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            first
                .checked_add(c.period.unwrap())
                .and_then(|v| v.checked_add(1))
                .unwrap_or(usize::MAX)
        })
        .collect();

    init_matrix_prefixes(&mut real_mu, cols, &ria_warm);
    init_matrix_prefixes(&mut imag_mu, cols, &ria_warm);
    init_matrix_prefixes(&mut angle_mu, cols, &ria_warm);
    init_matrix_prefixes(&mut state_mu, cols, &st_warm);

    let mut real_guard = ManuallyDrop::new(real_mu);
    let mut imag_guard = ManuallyDrop::new(imag_mu);
    let mut angle_guard = ManuallyDrop::new(angle_mu);
    let mut state_guard = ManuallyDrop::new(state_mu);

    let real: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(real_guard.as_mut_ptr() as *mut f64, real_guard.len())
    };
    let imag: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(imag_guard.as_mut_ptr() as *mut f64, imag_guard.len())
    };
    let angle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(angle_guard.as_mut_ptr() as *mut f64, angle_guard.len())
    };
    let state: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(state_guard.as_mut_ptr() as *mut f64, state_guard.len())
    };

    let do_row = |row: usize,
                  out_real: &mut [f64],
                  out_imag: &mut [f64],
                  out_angle: &mut [f64],
                  out_state: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let threshold = combos[row].threshold.unwrap();
        match kern {
            Kernel::Scalar => correlation_cycle_row_scalar_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => correlation_cycle_row_avx2_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => correlation_cycle_row_avx512_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            _ => correlation_cycle_row_scalar_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            real.par_chunks_mut(cols)
                .zip(imag.par_chunks_mut(cols))
                .zip(angle.par_chunks_mut(cols))
                .zip(state.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (((r, im), an), st))| do_row(row, r, im, an, st));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((r, im), an), st)) in real
                .chunks_mut(cols)
                .zip(imag.chunks_mut(cols))
                .zip(angle.chunks_mut(cols))
                .zip(state.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, r, im, an, st);
            }
        }
    } else {
        for (row, (((r, im), an), st)) in real
            .chunks_mut(cols)
            .zip(imag.chunks_mut(cols))
            .zip(angle.chunks_mut(cols))
            .zip(state.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, r, im, an, st);
        }
    }

    let real = unsafe {
        Vec::from_raw_parts(
            real_guard.as_mut_ptr() as *mut f64,
            real_guard.len(),
            real_guard.capacity(),
        )
    };
    let imag = unsafe {
        Vec::from_raw_parts(
            imag_guard.as_mut_ptr() as *mut f64,
            imag_guard.len(),
            imag_guard.capacity(),
        )
    };
    let angle = unsafe {
        Vec::from_raw_parts(
            angle_guard.as_mut_ptr() as *mut f64,
            angle_guard.len(),
            angle_guard.capacity(),
        )
    };
    let state = unsafe {
        Vec::from_raw_parts(
            state_guard.as_mut_ptr() as *mut f64,
            state_guard.len(),
            state_guard.capacity(),
        )
    };

    Ok(CorrelationCycleBatchOutput {
        real,
        imag,
        angle,
        state,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn correlation_cycle_batch_inner_into(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
    parallel: bool,
    out_real: &mut [f64],
    out_imag: &mut [f64],
    out_angle: &mut [f64],
    out_state: &mut [f64],
) -> Result<Vec<CorrelationCycleParams>, CorrelationCycleError> {
    if data.is_empty() {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| CorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;
    if out_real.len() != expected
        || out_imag.len() != expected
        || out_angle.len() != expected
        || out_state.len() != expected
    {
        let got = *[
            out_real.len(),
            out_imag.len(),
            out_angle.len(),
            out_state.len(),
        ]
        .iter()
        .min()
        .unwrap_or(&0);
        return Err(CorrelationCycleError::OutputLengthMismatch { expected, got });
    }

    let do_row = |row: usize, r: &mut [f64], im: &mut [f64], an: &mut [f64], st: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let t = combos[row].threshold.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                correlation_cycle_row_scalar_with_first(data, p, t, first, r, im, an, st)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                correlation_cycle_row_avx2_with_first(data, p, t, first, r, im, an, st)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                correlation_cycle_row_avx512_with_first(data, p, t, first, r, im, an, st)
            }
            _ => correlation_cycle_row_scalar_with_first(data, p, t, first, r, im, an, st),
        }

        let ria = first + p;
        let stp = first + p + 1;
        for v in &mut r[..ria] {
            *v = f64::NAN;
        }
        for v in &mut im[..ria] {
            *v = f64::NAN;
        }
        for v in &mut an[..ria] {
            *v = f64::NAN;
        }
        for v in &mut st[..stp] {
            *v = f64::NAN;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_real
                .par_chunks_mut(cols)
                .zip(out_imag.par_chunks_mut(cols))
                .zip(out_angle.par_chunks_mut(cols))
                .zip(out_state.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (((r, im), an), st))| do_row(row, r, im, an, st));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((r, im), an), st)) in out_real
                .chunks_mut(cols)
                .zip(out_imag.chunks_mut(cols))
                .zip(out_angle.chunks_mut(cols))
                .zip(out_state.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, r, im, an, st);
            }
        }
    } else {
        for (row, (((r, im), an), st)) in out_real
            .chunks_mut(cols)
            .zip(out_imag.chunks_mut(cols))
            .zip(out_angle.chunks_mut(cols))
            .zip(out_state.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, r, im, an, st);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use crate::utilities::enums::Kernel;

    #[test]
    fn correlation_cycle_deterministic_sin_cos_pins_the_complete_weight_domain() {
        assert_eq!(CORRELATION_CYCLE_HALF_PI.to_bits(), 0x3ff9_21fb_5444_2d18);
        assert_eq!(CORRELATION_CYCLE_TWO_PI.to_bits(), 0x4019_21fb_5444_2d18);
        assert_eq!(
            CORRELATION_CYCLE_RADIANS_TO_DEGREES.to_bits(),
            0x404c_a5dc_1a63_c1f8
        );

        let cases = [
            (
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x3ff0_0000_0000_0000,
            ),
            (
                0x3fe0_c152_382d_7365,
                0x3fdf_ffff_ffff_ffff,
                0x3feb_b67a_e858_4cab,
            ),
            (
                0x3fe9_21fb_5444_2d18,
                0x3fe6_a09e_667f_3bcc,
                0x3fe6_a09e_667f_3bcd,
            ),
            (
                0x3ff9_21fb_5444_2d18,
                0x3ff0_0000_0000_0000,
                0x3c91_a626_3314_5c07,
            ),
            (
                0x4000_c152_382d_7365,
                0x3feb_b67a_e858_4cab,
                0xbfdf_ffff_ffff_fffc,
            ),
            (
                0x4009_21fb_5444_2d18,
                0x3ca1_a626_3314_5c07,
                0xbff0_0000_0000_0000,
            ),
            (
                0x4012_d97c_7f33_21d2,
                0xbff0_0000_0000_0000,
                0xbcaa_7939_4c9e_8a0a,
            ),
            (
                0x4019_21fb_5444_2d18,
                0xbcb1_a626_3314_5c07,
                0x3ff0_0000_0000_0000,
            ),
        ];

        for (input_bits, expected_sin_bits, expected_cos_bits) in cases {
            let (sin, cos) = correlation_cycle_deterministic_sin_cos(f64::from_bits(input_bits));
            assert_eq!(sin.to_bits(), expected_sin_bits, "sin({input_bits:#018x})");
            assert_eq!(cos.to_bits(), expected_cos_bits, "cos({input_bits:#018x})");
        }
    }

    #[test]
    fn correlation_cycle_deterministic_atan_pins_every_reduction_branch_and_sign() {
        let cases = [
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
            (0x3fd0_0000_0000_0000, 0x3fcf_5b75_f92c_80dd),
            (0xbfd0_0000_0000_0000, 0xbfcf_5b75_f92c_80dd),
            (0x3fe0_0000_0000_0000, 0x3fdd_ac67_0561_bb4f),
            (0xbfe0_0000_0000_0000, 0xbfdd_ac67_0561_bb4f),
            (0x3ff0_0000_0000_0000, 0x3fe9_21fb_5444_2d18),
            (0xbff0_0000_0000_0000, 0xbfe9_21fb_5444_2d18),
            (0x3ff8_0000_0000_0000, 0x3fef_730b_d281_f69b),
            (0xbff8_0000_0000_0000, 0xbfef_730b_d281_f69b),
            (0x4008_0000_0000_0000, 0x3ff3_fc17_6b7a_8560),
            (0xc008_0000_0000_0000, 0xbff3_fc17_6b7a_8560),
            (0x7ff0_0000_0000_0000, 0x3ff9_21fb_5444_2d18),
            (0xfff0_0000_0000_0000, 0xbff9_21fb_5444_2d18),
            (0x7ff8_1234_5678_9abc, 0x7ff8_1234_5678_9abc),
        ];

        for (input_bits, expected_bits) in cases {
            let actual = correlation_cycle_deterministic_atan(f64::from_bits(input_bits));
            assert_eq!(actual.to_bits(), expected_bits, "atan({input_bits:#018x})");
        }
    }

    fn check_cc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = CorrelationCycleParams {
            period: None,
            threshold: None,
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", default_params);
        let output = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.real.len(), candles.close.len());
        Ok(())
    }

    fn check_cc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = CorrelationCycleParams {
            period: Some(20),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", params);
        let result = correlation_cycle_with_kernel(&input, kernel)?;
        let expected_last_five_real = [
            -0.3348928030992766,
            -0.2908979303392832,
            -0.10648582811938148,
            -0.09118320471750277,
            0.0826798259258665,
        ];
        let expected_last_five_imag = [
            0.2902308064575494,
            0.4025192756952553,
            0.4704322460080054,
            0.5404405595224989,
            0.5418162415918566,
        ];
        let expected_last_five_angle = [
            -139.0865569687123,
            -125.8553823569915,
            -102.75438860700636,
            -99.576759208278,
            -81.32373697835556,
        ];
        let start = result.real.len().saturating_sub(5);
        for i in 0..5 {
            let diff_real = (result.real[start + i] - expected_last_five_real[i]).abs();
            let diff_imag = (result.imag[start + i] - expected_last_five_imag[i]).abs();
            let diff_angle = (result.angle[start + i] - expected_last_five_angle[i]).abs();
            assert!(
                diff_real < 1e-8,
                "[{}] CC {:?} real mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.real[start + i],
                expected_last_five_real[i]
            );
            assert!(
                diff_imag < 1e-8,
                "[{}] CC {:?} imag mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.imag[start + i],
                expected_last_five_imag[i]
            );
            assert!(
                diff_angle < 1e-8,
                "[{}] CC {:?} angle mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.angle[start + i],
                expected_last_five_angle[i]
            );
        }
        Ok(())
    }

    fn check_cc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = CorrelationCycleInput::with_default_candles(&candles);
        match input.data {
            CorrelationCycleData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CorrelationCycleData::Candles"),
        }
        let output = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.real.len(), candles.close.len());
        Ok(())
    }

    fn check_cc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(0),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&input_data, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CC should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(10),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&data_small, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CC should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CorrelationCycleParams {
            period: Some(9),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&single_point, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CC should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let params = CorrelationCycleParams {
            period: Some(4),
            threshold: Some(2.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params.clone());
        let first_result = correlation_cycle_with_kernel(&input, kernel)?;
        let second_input = CorrelationCycleInput::from_slice(&first_result.real, params);
        let second_result = correlation_cycle_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.real.len(), data.len());
        assert_eq!(second_result.real.len(), data.len());
        Ok(())
    }

    fn check_cc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams {
                period: Some(20),
                threshold: None,
            },
        );
        let res = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(res.real.len(), candles.close.len());
        if res.real.len() > 40 {
            for (i, &val) in res.real[40..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    40 + i
                );
            }
        }
        Ok(())
    }

    fn check_cc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 20;
        let threshold = 9.0;
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams {
                period: Some(period),
                threshold: Some(threshold),
            },
        );
        let batch_output = correlation_cycle_with_kernel(&input, kernel)?;
        let mut stream = CorrelationCycleStream::try_new(CorrelationCycleParams {
            period: Some(period),
            threshold: Some(threshold),
        })?;
        let mut stream_real = Vec::with_capacity(candles.close.len());
        let mut stream_imag = Vec::with_capacity(candles.close.len());
        let mut stream_angle = Vec::with_capacity(candles.close.len());
        let mut stream_state = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some((r, im, ang, st)) => {
                    stream_real.push(r);
                    stream_imag.push(im);
                    stream_angle.push(ang);
                    stream_state.push(st);
                }
                None => {
                    stream_real.push(f64::NAN);
                    stream_imag.push(f64::NAN);
                    stream_angle.push(f64::NAN);
                    stream_state.push(0.0);
                }
            }
        }
        assert_eq!(batch_output.real.len(), stream_real.len());
        for (i, (&b, &s)) in batch_output.real.iter().zip(stream_real.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] CC streaming real f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_cc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            CorrelationCycleParams {
                period: Some(20),
                threshold: Some(9.0),
            },
            CorrelationCycleParams {
                period: Some(10),
                threshold: Some(5.0),
            },
            CorrelationCycleParams {
                period: Some(30),
                threshold: Some(15.0),
            },
            CorrelationCycleParams {
                period: None,
                threshold: None,
            },
        ];

        for params in test_params {
            let input = CorrelationCycleInput::from_candles(&candles, "close", params.clone());
            let output = correlation_cycle_with_kernel(&input, kernel)?;

            let arrays = vec![
                ("real", &output.real),
                ("imag", &output.imag),
                ("angle", &output.angle),
                ("state", &output.state),
            ];

            for (array_name, values) in arrays {
                for (i, &val) in values.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in {} array with params {:?}",
                            test_name, val, bits, i, array_name, params
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in {} array with params {:?}",
                            test_name, val, bits, i, array_name, params
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in {} array with params {:?}",
                            test_name, val, bits, i, array_name, params
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_cc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cc_tests {
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

    generate_all_cc_tests!(
        check_cc_partial_params,
        check_cc_accuracy,
        check_cc_default_candles,
        check_cc_zero_period,
        check_cc_period_exceeds_length,
        check_cc_very_small_dataset,
        check_cc_reinput,
        check_cc_nan_handling,
        check_cc_streaming,
        check_cc_no_poison
    );

    #[test]
    fn test_correlation_cycle_into_matches_api_v2() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams::default(),
        );

        let baseline = correlation_cycle(&input)?;

        let len = candles.close.len();
        let mut out_real = vec![0.0f64; len];
        let mut out_imag = vec![0.0f64; len];
        let mut out_angle = vec![0.0f64; len];
        let mut out_state = vec![0.0f64; len];

        correlation_cycle_into(
            &input,
            &mut out_real,
            &mut out_imag,
            &mut out_angle,
            &mut out_state,
        )?;

        assert_eq!(baseline.real.len(), out_real.len());
        assert_eq!(baseline.imag.len(), out_imag.len());
        assert_eq!(baseline.angle.len(), out_angle.len());
        assert_eq!(baseline.state.len(), out_state.len());

        fn eq_or_both_nan_eps(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan_eps(baseline.real[i], out_real[i]),
                "real mismatch at {}: base={}, into={}",
                i,
                baseline.real[i],
                out_real[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.imag[i], out_imag[i]),
                "imag mismatch at {}: base={}, into={}",
                i,
                baseline.imag[i],
                out_imag[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.angle[i], out_angle[i]),
                "angle mismatch at {}: base={}, into={}",
                i,
                baseline.angle[i],
                out_angle[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.state[i], out_state[i]),
                "state mismatch at {}: base={}, into={}",
                i,
                baseline.state[i],
                out_state[i]
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_correlation_cycle_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (5usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                    period + 10..400,
                ),
                Just(period),
                1.0f64..20.0f64,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, threshold)| {
                let params = CorrelationCycleParams {
                    period: Some(period),
                    threshold: Some(threshold),
                };
                let input = CorrelationCycleInput::from_slice(&data, params);

                let output = correlation_cycle_with_kernel(&input, kernel).unwrap();

                let ref_output = correlation_cycle_with_kernel(&input, Kernel::Scalar).unwrap();

                let warmup_period = period;

                for i in 0..warmup_period {
                    prop_assert!(
                        output.real[i].is_nan(),
                        "[{}] real[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.real[i]
                    );
                    prop_assert!(
                        output.imag[i].is_nan(),
                        "[{}] imag[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.imag[i]
                    );
                    prop_assert!(
                        output.angle[i].is_nan(),
                        "[{}] angle[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.angle[i]
                    );
                }

                for i in 0..=period {
                    prop_assert!(
                        output.state[i].is_nan(),
                        "[{}] state[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.state[i]
                    );
                }

                for i in warmup_period..data.len() {
                    if !output.real[i].is_nan() {
                        prop_assert!(
                            output.real[i] >= -1.0 - 1e-9 && output.real[i] <= 1.0 + 1e-9,
                            "[{}] real[{}] = {} is outside [-1, 1] bounds",
                            test_name,
                            i,
                            output.real[i]
                        );
                    }
                    if !output.imag[i].is_nan() {
                        prop_assert!(
                            output.imag[i] >= -1.0 - 1e-9 && output.imag[i] <= 1.0 + 1e-9,
                            "[{}] imag[{}] = {} is outside [-1, 1] bounds",
                            test_name,
                            i,
                            output.imag[i]
                        );
                    }
                }

                for i in warmup_period..data.len() {
                    if !output.angle[i].is_nan() {
                        prop_assert!(
                            output.angle[i] >= -180.0 - 1e-9 && output.angle[i] <= 180.0 + 1e-9,
                            "[{}] angle[{}] = {} is outside [-180, 180] bounds",
                            test_name,
                            i,
                            output.angle[i]
                        );
                    }
                }

                for i in (period + 1)..data.len() {
                    if !output.state[i].is_nan() {
                        let state_val = output.state[i];
                        prop_assert!(
                            (state_val + 1.0).abs() < 1e-9
                                || state_val.abs() < 1e-9
                                || (state_val - 1.0).abs() < 1e-9,
                            "[{}] state[{}] = {} is not -1, 0, or 1",
                            test_name,
                            i,
                            state_val
                        );
                    }
                }

                for i in 0..data.len() {
                    let real_bits = output.real[i].to_bits();
                    let ref_real_bits = ref_output.real[i].to_bits();

                    if !output.real[i].is_finite() || !ref_output.real[i].is_finite() {
                        prop_assert!(
                            real_bits == ref_real_bits,
                            "[{}] real finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.real[i],
                            ref_output.real[i]
                        );
                    } else {
                        let ulp_diff = real_bits.abs_diff(ref_real_bits);
                        prop_assert!(
                            (output.real[i] - ref_output.real[i]).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] real mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            output.real[i],
                            ref_output.real[i],
                            ulp_diff
                        );
                    }

                    let imag_bits = output.imag[i].to_bits();
                    let ref_imag_bits = ref_output.imag[i].to_bits();

                    if !output.imag[i].is_finite() || !ref_output.imag[i].is_finite() {
                        prop_assert!(
                            imag_bits == ref_imag_bits,
                            "[{}] imag finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.imag[i],
                            ref_output.imag[i]
                        );
                    } else {
                        let ulp_diff = imag_bits.abs_diff(ref_imag_bits);
                        prop_assert!(
                            (output.imag[i] - ref_output.imag[i]).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] imag mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            output.imag[i],
                            ref_output.imag[i],
                            ulp_diff
                        );
                    }

                    let angle_bits = output.angle[i].to_bits();
                    let ref_angle_bits = ref_output.angle[i].to_bits();

                    if !output.angle[i].is_finite() || !ref_output.angle[i].is_finite() {
                        prop_assert!(
                            angle_bits == ref_angle_bits,
                            "[{}] angle finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.angle[i],
                            ref_output.angle[i]
                        );
                    } else {
                        let ulp_diff = angle_bits.abs_diff(ref_angle_bits);
                        prop_assert!(
                            (output.angle[i] - ref_output.angle[i]).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] angle mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            output.angle[i],
                            ref_output.angle[i],
                            ulp_diff
                        );
                    }

                    let state_bits = output.state[i].to_bits();
                    let ref_state_bits = ref_output.state[i].to_bits();

                    if !output.state[i].is_finite() || !ref_output.state[i].is_finite() {
                        prop_assert!(
                            state_bits == ref_state_bits,
                            "[{}] state finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.state[i],
                            ref_output.state[i]
                        );
                    } else {
                        prop_assert!(
                            (output.state[i] - ref_output.state[i]).abs() <= 1e-9,
                            "[{}] state mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.state[i],
                            ref_output.state[i]
                        );
                    }
                }

                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                    for i in warmup_period..data.len() {
                        if !output.real[i].is_nan() {
                            prop_assert!(
                                output.real[i].abs() < 1e-6,
                                "[{}] real[{}] = {} should be near 0 for constant data",
                                test_name,
                                i,
                                output.real[i]
                            );
                        }
                        if !output.imag[i].is_nan() {
                            prop_assert!(
                                output.imag[i].abs() < 1e-6,
                                "[{}] imag[{}] = {} should be near 0 for constant data",
                                test_name,
                                i,
                                output.imag[i]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_cc_tests!(check_correlation_cycle_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = CorrelationCycleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CorrelationCycleParams::default();
        let (row_real, row_imag, row_angle, row_state) =
            output.values_for(&def).expect("default row missing");
        assert_eq!(row_real.len(), c.close.len());
        assert_eq!(row_imag.len(), c.close.len());
        assert_eq!(row_angle.len(), c.close.len());
        assert_eq!(row_state.len(), c.close.len());
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = CorrelationCycleBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 40, 10)
            .threshold_range(5.0, 15.0, 5.0)
            .apply_candles(&c, "close")?;

        let matrices = vec![
            ("real", &output.real),
            ("imag", &output.imag),
            ("angle", &output.angle),
            ("state", &output.state),
        ];

        for (matrix_name, values) in matrices {
            for (idx, &val) in values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let period = output.combos[row].period.unwrap();
                let threshold = output.combos[row].threshold.unwrap();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} matrix, params: period={}, threshold={}",
                        test, val, bits, row, col, idx, matrix_name, period, threshold
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} matrix, params: period={}, threshold={}",
                        test, val, bits, row, col, idx, matrix_name, period, threshold
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} matrix, params: period={}, threshold={}",
                        test, val, bits, row, col, idx, matrix_name, period, threshold
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_correlation_cycle_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let x = 100.0 + (i as f64 * 0.07).sin() * 2.5 + (i as f64 * 0.011).cos() * 0.4;
            data.push(x);
        }

        let input = CorrelationCycleInput::from_slice(&data, CorrelationCycleParams::default());

        let base = correlation_cycle(&input)?;

        let mut out_r = vec![0.0; n];
        let mut out_i = vec![0.0; n];
        let mut out_a = vec![0.0; n];
        let mut out_s = vec![0.0; n];

        {
            correlation_cycle_into(&input, &mut out_r, &mut out_i, &mut out_a, &mut out_s)?;
        }

        assert_eq!(base.real.len(), n);
        assert_eq!(base.imag.len(), n);
        assert_eq!(base.angle.len(), n);
        assert_eq!(base.state.len(), n);

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(base.real[i], out_r[i]),
                "real mismatch at {}: base={}, into={}",
                i,
                base.real[i],
                out_r[i]
            );
            assert!(
                eq_or_both_nan(base.imag[i], out_i[i]),
                "imag mismatch at {}: base={}, into={}",
                i,
                base.imag[i],
                out_i[i]
            );
            assert!(
                eq_or_both_nan(base.angle[i], out_a[i]),
                "angle mismatch at {}: base={}, into={}",
                i,
                base.angle[i],
                out_a[i]
            );
            assert!(
                eq_or_both_nan(base.state[i], out_s[i]),
                "state mismatch at {}: base={}, into={}",
                i,
                base.state[i],
                out_s[i]
            );
        }

        Ok(())
    }
}
