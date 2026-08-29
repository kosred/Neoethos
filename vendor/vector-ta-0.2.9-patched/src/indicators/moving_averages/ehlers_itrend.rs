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
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EhlersITrendInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersITrendData::Candles { candles, source } => source_type(candles, source),
            EhlersITrendData::Slice(slice) => slice,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersITrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersITrendOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EhlersITrendParams {
    pub warmup_bars: Option<usize>,
    pub max_dc_period: Option<usize>,
}
impl Default for EhlersITrendParams {
    fn default() -> Self {
        Self {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersITrendInput<'a> {
    pub data: EhlersITrendData<'a>,
    pub params: EhlersITrendParams,
}
impl<'a> EhlersITrendInput<'a> {
    #[inline(always)]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersITrendParams) -> Self {
        Self {
            data: EhlersITrendData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline(always)]
    pub fn from_slice(sl: &'a [f64], p: EhlersITrendParams) -> Self {
        Self {
            data: EhlersITrendData::Slice(sl),
            params: p,
        }
    }
    #[inline(always)]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersITrendParams::default())
    }
    #[inline(always)]
    pub fn get_warmup_bars(&self) -> usize {
        self.params.warmup_bars.unwrap_or(12)
    }
    #[inline(always)]
    pub fn get_max_dc_period(&self) -> usize {
        self.params.max_dc_period.unwrap_or(50)
    }
}

#[derive(Debug, Error)]
pub enum EhlersITrendError {
    #[error("ehlers_itrend: Input data is empty.")]
    EmptyInputData,
    #[error("ehlers_itrend: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "ehlers_itrend: Not enough data for warmup. warmup_bars={warmup_bars} but data length={length}"
    )]
    NotEnoughDataForWarmup { warmup_bars: usize, length: usize },

    #[error("ehlers_itrend: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ehlers_itrend: Invalid warmup_bars: {warmup_bars}")]
    InvalidWarmupBars { warmup_bars: usize },
    #[error("ehlers_itrend: Invalid max_dc_period: {max_dc}")]
    InvalidMaxDcPeriod { max_dc: usize },

    #[allow(dead_code)]
    #[error("ehlers_itrend: Output length mismatch. expected={expected}, got={got}")]
    InvalidOutputLen { expected: usize, got: usize },
    #[error("ehlers_itrend: Invalid batch kernel")]
    InvalidBatchKernel,
    #[error("ehlers_itrend: Invalid batch range")]
    InvalidBatchRange,

    #[error("ehlers_itrend: Output length mismatch. expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ehlers_itrend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("ehlers_itrend: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("ehlers_itrend: arithmetic overflow while computing {context}")]
    SizeOverflow { context: &'static str },
}

#[inline]
pub fn ehlers_itrend(input: &EhlersITrendInput) -> Result<EhlersITrendOutput, EhlersITrendError> {
    ehlers_itrend_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ehlers_itrend_prepare<'a>(
    input: &'a EhlersITrendInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, Kernel), EhlersITrendError> {
    let data: &[f64] = match &input.data {
        EhlersITrendData::Candles { candles, source } => source_type(candles, source),
        EhlersITrendData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(EhlersITrendError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersITrendError::AllValuesNaN)?;
    let warmup_bars = input.get_warmup_bars();
    let max_dc = input.get_max_dc_period();
    if warmup_bars == 0 {
        return Err(EhlersITrendError::InvalidWarmupBars { warmup_bars });
    }
    if max_dc == 0 {
        return Err(EhlersITrendError::InvalidMaxDcPeriod { max_dc });
    }

    if len - first < warmup_bars {
        return Err(EhlersITrendError::NotEnoughDataForWarmup {
            warmup_bars,
            length: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let warm = first + warmup_bars;

    Ok((data, warmup_bars, max_dc, first, warm, chosen))
}

#[inline(always)]
fn warm_index(first: usize, warmup_bars: usize) -> usize {
    first + warmup_bars
}

/// NeoEthos f64 authority for the adaptive SMA inside Ehlers ITrend.
///
/// The creator's formula defines a simple mean over the rounded dominant-cycle
/// period followed by a 4-3-2-1 WMA. It does not define a floating-point
/// reduction tree. This version therefore fixes one accuracy-oriented tree for
/// every CPU/GPU f64 route: chronological, power-of-two-normalized Neumaier
/// summation for the adaptive window, and explicit 4-3-2-1 `/ 10` order.
pub const EHLERS_ITREND_F64_AUTHORITY_V1: &str =
    "ehlers_itrend_f64_pow2_scaled_chronological_neumaier_window_explicit_wma4_rn_v1";

#[derive(Clone, Copy, Default)]
struct ITrendNeumaierF64V1 {
    sum: f64,
    correction: f64,
}

impl ITrendNeumaierF64V1 {
    #[inline(always)]
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    #[inline(always)]
    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[inline(always)]
fn itrend_qnan_v1() -> f64 {
    f64::from_bits(0x7ff8_0000_0000_0000)
}

/// Exact power-of-two at or immediately below one finite positive binary64.
#[inline(always)]
fn itrend_floor_power_of_two_v1(max_abs: f64) -> f64 {
    debug_assert!(max_abs.is_finite() && max_abs > 0.0);
    let bits = max_abs.to_bits();
    let exponent = bits & 0x7ff0_0000_0000_0000;
    if exponent != 0 {
        f64::from_bits(exponent)
    } else {
        let highest_mantissa_bit = 63 - bits.leading_zeros();
        f64::from_bits(1_u64 << highest_mantissa_bit)
    }
}

/// Stable mean of one logical window supplied oldest-to-newest by `value_at`.
#[inline(always)]
fn itrend_stable_window_mean_v1<F>(count: usize, mut value_at: F) -> f64
where
    F: FnMut(usize) -> f64,
{
    debug_assert!(count > 0);
    let mut max_abs = 0.0_f64;
    for index in 0..count {
        let value = value_at(index);
        if !value.is_finite() {
            return itrend_qnan_v1();
        }
        max_abs = max_abs.max(value.abs());
    }
    if max_abs == 0.0 {
        return 0.0;
    }

    let scale = itrend_floor_power_of_two_v1(max_abs);
    let mut normalized = ITrendNeumaierF64V1::default();
    for index in 0..count {
        normalized.add(value_at(index) / scale);
    }
    let result = (normalized.total() / count as f64) * scale;
    if !result.is_finite() {
        itrend_qnan_v1()
    } else if result == 0.0 {
        0.0
    } else {
        result
    }
}

/// Explicitly rounded schedule shared by the input smoother and final ITrend.
#[inline(always)]
fn itrend_wma4_v1(current: f64, lag1: f64, lag2: f64, lag3: f64) -> f64 {
    let weighted_0 = 4.0 * current;
    let weighted_1 = weighted_0 + 3.0 * lag1;
    let weighted_2 = weighted_1 + 2.0 * lag2;
    let weighted_3 = weighted_2 + lag3;
    weighted_3 / 10.0
}

/// Explicit Hilbert-transform schedule shared by scalar, batch and stream.
#[inline(always)]
fn itrend_hilbert4_v1(current: f64, lag2: f64, lag4: f64, lag6: f64) -> f64 {
    let term_0 = 0.0962 * current;
    let term_1 = term_0 + 0.5769 * lag2;
    let term_2 = term_1 - 0.5769 * lag4;
    term_2 - 0.0962 * lag6
}

/// One explicit dominant-cycle update shared by every CPU f64 route.
#[inline(always)]
fn itrend_period_v1(
    re_smooth: f64,
    im_smooth: f64,
    prev_mesa: f64,
    prev_smooth: f64,
    max_dc: usize,
) -> (f64, f64, usize) {
    let mut new_mesa = 0.0;
    if re_smooth != 0.0 && im_smooth != 0.0 {
        let angle = im_smooth.atan2(re_smooth);
        if angle != 0.0 {
            new_mesa = (2.0 * core::f64::consts::PI) / angle;
        }
    }
    let up_lim = 1.5 * prev_mesa;
    if new_mesa > up_lim {
        new_mesa = up_lim;
    }
    let low_lim = 0.67 * prev_mesa;
    if new_mesa < low_lim {
        new_mesa = low_lim;
    }
    new_mesa = new_mesa.clamp(6.0, 50.0);

    let final_mesa = 0.2 * new_mesa + 0.8 * prev_mesa;
    let smooth_period = 0.33 * final_mesa + 0.67 * prev_smooth;
    let mut dcp = (smooth_period + 0.5).floor() as usize;
    dcp = dcp.clamp(1, max_dc);
    (final_mesa, smooth_period, dcp)
}

#[inline(always)]
fn itrend_ring_oldest_v1(next: usize, count: usize, capacity: usize) -> usize {
    debug_assert!(next < capacity && count <= capacity);
    if next >= count {
        next - count
    } else {
        capacity - (count - next)
    }
}

#[inline(always)]
pub fn ehlers_itrend_scalar_tail(
    src: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    _first_valid: usize,
    warm: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(src.len(), out.len());

    let length = src.len();
    let mut fir_buf = [0.0; 7];
    let mut det_buf = [0.0; 7];
    let mut i1_buf = [0.0; 7];
    let mut q1_buf = [0.0; 7];
    let (mut prev_i2, mut prev_q2) = (0.0, 0.0);
    let (mut prev_re, mut prev_im) = (0.0, 0.0);
    let (mut prev_mesa, mut prev_smooth) = (0.0, 0.0);
    let mut sum_ring = vec![0.0; max_dc];
    let mut sum_idx = 0usize;
    let (mut prev_it1, mut prev_it2, mut prev_it3) = (0.0, 0.0, 0.0);
    let mut ring_ptr = 0usize;
    let (mut src_l1, mut src_l2, mut src_l3) = (0.0, 0.0, 0.0);

    #[inline(always)]
    fn ring_get(buf: &[f64; 7], center: usize, off: usize) -> f64 {
        let mut idx = center + 7 - off;
        if idx >= 7 {
            idx -= 7;
        }
        buf[idx]
    }

    for i in 0..length {
        let x0 = src[i];

        let fir_val = itrend_wma4_v1(x0, src_l1, src_l2, src_l3);
        fir_buf[ring_ptr] = fir_val;

        let fir_0 = ring_get(&fir_buf, ring_ptr, 0);
        let fir_2 = ring_get(&fir_buf, ring_ptr, 2);
        let fir_4 = ring_get(&fir_buf, ring_ptr, 4);
        let fir_6 = ring_get(&fir_buf, ring_ptr, 6);

        let h_in = itrend_hilbert4_v1(fir_0, fir_2, fir_4, fir_6);
        let period_mult = 0.075 * prev_mesa + 0.54;
        let det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        let i1_val = ring_get(&det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        let det_0 = ring_get(&det_buf, ring_ptr, 0);
        let det_2 = ring_get(&det_buf, ring_ptr, 2);
        let det_4 = ring_get(&det_buf, ring_ptr, 4);
        let det_6 = ring_get(&det_buf, ring_ptr, 6);
        let h_in_q1 = itrend_hilbert4_v1(det_0, det_2, det_4, det_6);
        let q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        let i1_0 = ring_get(&i1_buf, ring_ptr, 0);
        let i1_2 = ring_get(&i1_buf, ring_ptr, 2);
        let i1_4 = ring_get(&i1_buf, ring_ptr, 4);
        let i1_6 = ring_get(&i1_buf, ring_ptr, 6);
        let j_i_val = itrend_hilbert4_v1(i1_0, i1_2, i1_4, i1_6) * period_mult;

        let q1_0 = ring_get(&q1_buf, ring_ptr, 0);
        let q1_2 = ring_get(&q1_buf, ring_ptr, 2);
        let q1_4 = ring_get(&q1_buf, ring_ptr, 4);
        let q1_6 = ring_get(&q1_buf, ring_ptr, 6);
        let j_q_val = itrend_hilbert4_v1(q1_0, q1_2, q1_4, q1_6) * period_mult;

        let mut i2_cur = 0.2 * (i1_val - j_q_val) + 0.8 * prev_i2;
        let mut q2_cur = 0.2 * (q1_val + j_i_val) + 0.8 * prev_q2;

        let re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        let im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        let re_smooth = 0.2 * re_val + 0.8 * prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * prev_im;
        prev_re = re_smooth;
        prev_im = im_smooth;

        let (final_mesa, sp_val, dcp) =
            itrend_period_v1(re_smooth, im_smooth, prev_mesa, prev_smooth, max_dc);
        prev_mesa = final_mesa;
        prev_smooth = sp_val;

        sum_ring[sum_idx] = x0;
        sum_idx += 1;
        if sum_idx == max_dc {
            sum_idx = 0;
        }
        let oldest = itrend_ring_oldest_v1(sum_idx, dcp, max_dc);
        let it_val =
            itrend_stable_window_mean_v1(dcp, |offset| sum_ring[(oldest + offset) % max_dc]);

        let eit_val = if i < warmup_bars {
            x0
        } else {
            itrend_wma4_v1(it_val, prev_it1, prev_it2, prev_it3)
        };

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if i >= warm {
            out[i] = eit_val;
        }
        src_l3 = src_l2;
        src_l2 = src_l1;
        src_l1 = x0;
        ring_ptr += 1;
        if ring_ptr == 7 {
            ring_ptr = 0;
        }
    }
}

#[inline(always)]
fn ehlers_itrend_compute_into(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    chosen: Kernel,
    out: &mut [f64],
) {
    let warm = warm_index(first, warmup_bars);
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            ehlers_itrend_scalar_tail(data, warmup_bars, max_dc, first, warm, out)
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
            ehlers_itrend_scalar_tail_avx2(data, warmup_bars, max_dc, first, warm, out)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            ehlers_itrend_scalar_tail_avx512(data, warmup_bars, max_dc, first, warm, out)
        },
        _ => unreachable!(),
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn ehlers_itrend_scalar_tail_avx2(
    src: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    warm: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar_tail(src, warmup_bars, max_dc, first_valid, warm, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn ehlers_itrend_scalar_tail_avx512(
    src: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    warm: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar_tail(src, warmup_bars, max_dc, first_valid, warm, out);
}

pub fn ehlers_itrend_with_kernel(
    input: &EhlersITrendInput,
    kernel: Kernel,
) -> Result<EhlersITrendOutput, EhlersITrendError> {
    let (data, warmup_bars, max_dc, first, warm, chosen) = ehlers_itrend_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), warm);
    ehlers_itrend_compute_into(data, warmup_bars, max_dc, first, chosen, &mut out);
    Ok(EhlersITrendOutput { values: out })
}

pub fn ehlers_itrend_into(
    input: &EhlersITrendInput,
    out: &mut [f64],
) -> Result<(), EhlersITrendError> {
    ehlers_itrend_into_slice(out, input, Kernel::Auto)
}

pub fn ehlers_itrend_into_slice(
    dst: &mut [f64],
    input: &EhlersITrendInput,
    kern: Kernel,
) -> Result<(), EhlersITrendError> {
    let (data, warmup_bars, max_dc, first, warm, chosen) = ehlers_itrend_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(EhlersITrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    for v in &mut dst[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }
    ehlers_itrend_compute_into(data, warmup_bars, max_dc, first, chosen, dst);
    Ok(())
}

#[inline]
pub fn ehlers_itrend_scalar(
    src: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    _first_valid: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(src.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("fma") {
            unsafe { ehlers_itrend_unsafe_scalar(src, warmup_bars, max_dc, out) }
        } else {
            ehlers_itrend_safe_scalar(src, warmup_bars, max_dc, out)
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        ehlers_itrend_safe_scalar(src, warmup_bars, max_dc, out)
    }
}

#[inline(always)]
fn ehlers_itrend_safe_scalar(src: &[f64], warmup_bars: usize, max_dc: usize, out: &mut [f64]) {
    let length = src.len();
    let mut fir_buf = [0.0; 7];
    let mut det_buf = [0.0; 7];
    let mut i1_buf = [0.0; 7];
    let mut q1_buf = [0.0; 7];
    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;
    let mut prev_mesa = 0.0;
    let mut prev_smooth = 0.0;
    let mut sum_ring = vec![0.0; max_dc];
    let mut sum_idx = 0_usize;
    let mut prev_it1 = 0.0;
    let mut prev_it2 = 0.0;
    let mut prev_it3 = 0.0;
    let mut ring_ptr = 0_usize;
    for i in 0..length {
        let x0 = src[i];
        let x1 = if i >= 1 { src[i - 1] } else { 0.0 };
        let x2 = if i >= 2 { src[i - 2] } else { 0.0 };
        let x3 = if i >= 3 { src[i - 3] } else { 0.0 };
        let fir_val = itrend_wma4_v1(x0, x1, x2, x3);
        fir_buf[ring_ptr] = fir_val;

        #[inline(always)]
        fn get_ring(buf: &[f64; 7], center: usize, offset: usize) -> f64 {
            buf[(7 + center - offset) % 7]
        }
        let fir_0 = get_ring(&fir_buf, ring_ptr, 0);
        let fir_2 = get_ring(&fir_buf, ring_ptr, 2);
        let fir_4 = get_ring(&fir_buf, ring_ptr, 4);
        let fir_6 = get_ring(&fir_buf, ring_ptr, 6);

        let h_in = itrend_hilbert4_v1(fir_0, fir_2, fir_4, fir_6);
        let period_mult = 0.075 * prev_mesa + 0.54;
        let det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        let i1_val = get_ring(&det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        let det_0 = get_ring(&det_buf, ring_ptr, 0);
        let det_2 = get_ring(&det_buf, ring_ptr, 2);
        let det_4 = get_ring(&det_buf, ring_ptr, 4);
        let det_6 = get_ring(&det_buf, ring_ptr, 6);
        let h_in_q1 = itrend_hilbert4_v1(det_0, det_2, det_4, det_6);
        let q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        let i1_0 = get_ring(&i1_buf, ring_ptr, 0);
        let i1_2 = get_ring(&i1_buf, ring_ptr, 2);
        let i1_4 = get_ring(&i1_buf, ring_ptr, 4);
        let i1_6 = get_ring(&i1_buf, ring_ptr, 6);
        let j_i_val = itrend_hilbert4_v1(i1_0, i1_2, i1_4, i1_6) * period_mult;

        let q1_0 = get_ring(&q1_buf, ring_ptr, 0);
        let q1_2 = get_ring(&q1_buf, ring_ptr, 2);
        let q1_4 = get_ring(&q1_buf, ring_ptr, 4);
        let q1_6 = get_ring(&q1_buf, ring_ptr, 6);
        let j_q_val = itrend_hilbert4_v1(q1_0, q1_2, q1_4, q1_6) * period_mult;

        let mut i2_cur = i1_val - j_q_val;
        let mut q2_cur = q1_val + j_i_val;
        i2_cur = 0.2 * i2_cur + 0.8 * prev_i2;
        q2_cur = 0.2 * q2_cur + 0.8 * prev_q2;

        let re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        let im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        let re_smooth = 0.2 * re_val + 0.8 * prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * prev_im;
        prev_re = re_smooth;
        prev_im = im_smooth;

        let (final_mesa, sp_val, dcp) =
            itrend_period_v1(re_smooth, im_smooth, prev_mesa, prev_smooth, max_dc);
        prev_mesa = final_mesa;
        prev_smooth = sp_val;

        sum_ring[sum_idx] = x0;
        sum_idx = (sum_idx + 1) % max_dc;
        let oldest = itrend_ring_oldest_v1(sum_idx, dcp, max_dc);
        let it_val =
            itrend_stable_window_mean_v1(dcp, |offset| sum_ring[(oldest + offset) % max_dc]);

        let eit_val = if i < warmup_bars {
            x0
        } else {
            itrend_wma4_v1(it_val, prev_it1, prev_it2, prev_it3)
        };
        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        out[i] = eit_val;

        ring_ptr = (ring_ptr + 1) % 7;
    }
}

#[inline(always)]
unsafe fn r7(buf: &[f64; 7], p: usize, off: usize) -> f64 {
    let idx = if p >= off { p - off } else { p + 7 - off };
    *buf.get_unchecked(idx)
}

#[inline(always)]
pub unsafe fn ehlers_itrend_unsafe_scalar(
    src: &[f64],
    warmup: usize,
    max_dc: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(src.len(), out.len());
    let len = src.len();

    let mut fir = [0.0; 7];
    let mut det = [0.0; 7];
    let mut i1 = [0.0; 7];
    let mut q1 = [0.0; 7];

    let mut sum: Vec<f64> = vec![0.0; max_dc];

    let (mut i2p, mut q2p, mut rep, mut imp) = (0.0, 0.0, 0.0, 0.0);
    let (mut mesa_p, mut sm_p) = (0.0, 0.0);
    let (mut it1p, mut it2p, mut it3p) = (0.0, 0.0, 0.0);

    let mut rp = 0;
    let mut sp = 0;

    for (idx, &x0) in src.iter().enumerate() {
        let x1 = if idx >= 1 {
            *src.get_unchecked(idx - 1)
        } else {
            0.0
        };
        let x2 = if idx >= 2 {
            *src.get_unchecked(idx - 2)
        } else {
            0.0
        };
        let x3 = if idx >= 3 {
            *src.get_unchecked(idx - 3)
        } else {
            0.0
        };
        let fir_val = itrend_wma4_v1(x0, x1, x2, x3);
        *fir.get_unchecked_mut(rp) = fir_val;

        let hp = itrend_hilbert4_v1(
            r7(&fir, rp, 0),
            r7(&fir, rp, 2),
            r7(&fir, rp, 4),
            r7(&fir, rp, 6),
        );
        let period_mult = 0.075 * mesa_p + 0.54;
        let det_val = hp * period_mult;
        *det.get_unchecked_mut(rp) = det_val;

        let i1v = r7(&det, rp, 3);
        let q1v = itrend_hilbert4_v1(
            r7(&det, rp, 0),
            r7(&det, rp, 2),
            r7(&det, rp, 4),
            r7(&det, rp, 6),
        ) * period_mult;
        *i1.get_unchecked_mut(rp) = i1v;
        *q1.get_unchecked_mut(rp) = q1v;

        let j_i = itrend_hilbert4_v1(
            r7(&i1, rp, 0),
            r7(&i1, rp, 2),
            r7(&i1, rp, 4),
            r7(&i1, rp, 6),
        ) * period_mult;
        let j_q = itrend_hilbert4_v1(
            r7(&q1, rp, 0),
            r7(&q1, rp, 2),
            r7(&q1, rp, 4),
            r7(&q1, rp, 6),
        ) * period_mult;

        let mut i2 = 0.2 * (i1v - j_q) + 0.8 * i2p;
        let mut q2 = 0.2 * (q1v + j_i) + 0.8 * q2p;

        let re = 0.2 * (i2 * i2p + q2 * q2p) + 0.8 * rep;
        let im = 0.2 * (i2 * q2p - q2 * i2p) + 0.8 * imp;
        i2p = i2;
        q2p = q2;
        rep = re;
        imp = im;

        let (mesa_f, sp_v, dcp) = itrend_period_v1(re, im, mesa_p, sm_p, max_dc);
        mesa_p = mesa_f;
        sm_p = sp_v;

        *sum.get_unchecked_mut(sp) = x0;
        sp += 1;
        if sp == max_dc {
            sp = 0;
        }

        let oldest = itrend_ring_oldest_v1(sp, dcp, max_dc);
        let it = itrend_stable_window_mean_v1(dcp, |offset| unsafe {
            *sum.get_unchecked((oldest + offset) % max_dc)
        });

        out[idx] = if idx < warmup {
            x0
        } else {
            itrend_wma4_v1(it, it1p, it2p, it3p)
        };

        it3p = it2p;
        it2p = it1p;
        it1p = it;

        rp += 1;
        if rp == 7 {
            rp = 0;
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ehlers_itrend_avx2(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first_valid, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn ehlers_itrend_avx512(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    ehlers_itrend_scalar(data, warmup_bars, max_dc, first_valid, out);
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersITrendBuilder {
    warmup_bars: Option<usize>,
    max_dc_period: Option<usize>,
    kernel: Kernel,
}
impl Default for EhlersITrendBuilder {
    fn default() -> Self {
        Self {
            warmup_bars: None,
            max_dc_period: None,
            kernel: Kernel::Auto,
        }
    }
}
impl EhlersITrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn warmup_bars(mut self, n: usize) -> Self {
        self.warmup_bars = Some(n);
        self
    }
    #[inline(always)]
    pub fn max_dc_period(mut self, n: usize) -> Self {
        self.max_dc_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EhlersITrendOutput, EhlersITrendError> {
        let p = EhlersITrendParams {
            warmup_bars: self.warmup_bars,
            max_dc_period: self.max_dc_period,
        };
        let i = EhlersITrendInput::from_candles(c, "close", p);
        ehlers_itrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersITrendOutput, EhlersITrendError> {
        let p = EhlersITrendParams {
            warmup_bars: self.warmup_bars,
            max_dc_period: self.max_dc_period,
        };
        let i = EhlersITrendInput::from_slice(d, p);
        ehlers_itrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersITrendStream, EhlersITrendError> {
        let p = EhlersITrendParams {
            warmup_bars: self.warmup_bars,
            max_dc_period: self.max_dc_period,
        };
        EhlersITrendStream::try_new(p)
    }
}

#[derive(Debug, Clone)]
pub struct EhlersITrendStream {
    warmup_bars: usize,
    max_dc: usize,
    fir_buf: [f64; 7],
    det_buf: [f64; 7],
    i1_buf: [f64; 7],
    q1_buf: [f64; 7],
    prev_i2: f64,
    prev_q2: f64,
    prev_re: f64,
    prev_im: f64,
    prev_mesa: f64,
    prev_smooth: f64,
    sum_ring: Vec<f64>,
    sum_idx: usize,

    wma_hist: [f64; 3],
    prev_it1: f64,
    prev_it2: f64,
    prev_it3: f64,
    ring_ptr: usize,
    bar: usize,
}
impl EhlersITrendStream {
    pub fn try_new(params: EhlersITrendParams) -> Result<Self, EhlersITrendError> {
        let warmup_bars = params.warmup_bars.unwrap_or(12);
        let max_dc = params.max_dc_period.unwrap_or(50);
        if warmup_bars == 0 {
            return Err(EhlersITrendError::InvalidWarmupBars { warmup_bars });
        }
        if max_dc == 0 {
            return Err(EhlersITrendError::InvalidMaxDcPeriod { max_dc });
        }
        Ok(Self {
            warmup_bars,
            max_dc,
            fir_buf: [0.0; 7],
            det_buf: [0.0; 7],
            i1_buf: [0.0; 7],
            q1_buf: [0.0; 7],
            prev_i2: 0.0,
            prev_q2: 0.0,
            prev_re: 0.0,
            prev_im: 0.0,
            prev_mesa: 0.0,
            prev_smooth: 0.0,
            sum_ring: vec![0.0; max_dc],
            sum_idx: 0,
            wma_hist: [0.0; 3],
            prev_it1: 0.0,
            prev_it2: 0.0,
            prev_it3: 0.0,
            ring_ptr: 0,
            bar: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x0: f64) -> Option<f64> {
        #[inline(always)]
        fn r7(buf: &[f64; 7], p: usize, off: usize) -> f64 {
            let mut idx = p + 7 - off;
            if idx >= 7 {
                idx -= 7;
            }
            buf[idx]
        }
        let fir_val = itrend_wma4_v1(x0, self.wma_hist[0], self.wma_hist[1], self.wma_hist[2]);

        self.wma_hist[2] = self.wma_hist[1];
        self.wma_hist[1] = self.wma_hist[0];
        self.wma_hist[0] = x0;

        self.sum_ring[self.sum_idx] = x0;
        self.sum_idx += 1;
        if self.sum_idx == self.max_dc {
            self.sum_idx = 0;
        }

        self.fir_buf[self.ring_ptr] = fir_val;

        let fir_0 = r7(&self.fir_buf, self.ring_ptr, 0);
        let fir_2 = r7(&self.fir_buf, self.ring_ptr, 2);
        let fir_4 = r7(&self.fir_buf, self.ring_ptr, 4);
        let fir_6 = r7(&self.fir_buf, self.ring_ptr, 6);

        let h_in = itrend_hilbert4_v1(fir_0, fir_2, fir_4, fir_6);
        let period_mult = 0.075 * self.prev_mesa + 0.54;

        let det_val = h_in * period_mult;
        self.det_buf[self.ring_ptr] = det_val;

        let i1_val = r7(&self.det_buf, self.ring_ptr, 3);
        self.i1_buf[self.ring_ptr] = i1_val;

        let det_0 = r7(&self.det_buf, self.ring_ptr, 0);
        let det_2 = r7(&self.det_buf, self.ring_ptr, 2);
        let det_4 = r7(&self.det_buf, self.ring_ptr, 4);
        let det_6 = r7(&self.det_buf, self.ring_ptr, 6);
        let h_in_q1 = itrend_hilbert4_v1(det_0, det_2, det_4, det_6);
        let q1_val = h_in_q1 * period_mult;
        self.q1_buf[self.ring_ptr] = q1_val;

        let i1_0 = r7(&self.i1_buf, self.ring_ptr, 0);
        let i1_2 = r7(&self.i1_buf, self.ring_ptr, 2);
        let i1_4 = r7(&self.i1_buf, self.ring_ptr, 4);
        let i1_6 = r7(&self.i1_buf, self.ring_ptr, 6);
        let j_i_val = itrend_hilbert4_v1(i1_0, i1_2, i1_4, i1_6) * period_mult;

        let q1_0 = r7(&self.q1_buf, self.ring_ptr, 0);
        let q1_2 = r7(&self.q1_buf, self.ring_ptr, 2);
        let q1_4 = r7(&self.q1_buf, self.ring_ptr, 4);
        let q1_6 = r7(&self.q1_buf, self.ring_ptr, 6);
        let j_q_val = itrend_hilbert4_v1(q1_0, q1_2, q1_4, q1_6) * period_mult;

        let mut i2_cur = 0.2 * (i1_val - j_q_val) + 0.8 * self.prev_i2;
        let mut q2_cur = 0.2 * (q1_val + j_i_val) + 0.8 * self.prev_q2;

        let re_val = i2_cur * self.prev_i2 + q2_cur * self.prev_q2;
        let im_val = i2_cur * self.prev_q2 - q2_cur * self.prev_i2;
        self.prev_i2 = i2_cur;
        self.prev_q2 = q2_cur;

        let re_smooth = 0.2 * re_val + 0.8 * self.prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * self.prev_im;
        self.prev_re = re_smooth;
        self.prev_im = im_smooth;

        let (final_mesa, sp_val, dcp) = itrend_period_v1(
            re_smooth,
            im_smooth,
            self.prev_mesa,
            self.prev_smooth,
            self.max_dc,
        );
        self.prev_mesa = final_mesa;
        self.prev_smooth = sp_val;

        let oldest = itrend_ring_oldest_v1(self.sum_idx, dcp, self.max_dc);
        let it_val = itrend_stable_window_mean_v1(dcp, |offset| {
            self.sum_ring[(oldest + offset) % self.max_dc]
        });

        let eit_val = if self.bar < self.warmup_bars {
            x0
        } else {
            itrend_wma4_v1(it_val, self.prev_it1, self.prev_it2, self.prev_it3)
        };

        self.prev_it3 = self.prev_it2;
        self.prev_it2 = self.prev_it1;
        self.prev_it1 = it_val;

        self.ring_ptr = (self.ring_ptr + 1) % 7;

        let result = if self.bar < self.warmup_bars {
            None
        } else {
            Some(eit_val)
        };
        self.bar += 1;
        result
    }
}

#[derive(Clone, Debug)]
pub struct EhlersITrendBatchRange {
    pub warmup_bars: (usize, usize, usize),
    pub max_dc_period: (usize, usize, usize),
}
impl Default for EhlersITrendBatchRange {
    fn default() -> Self {
        Self {
            warmup_bars: (12, 12, 0),
            max_dc_period: (50, 299, 1),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct EhlersITrendBatchBuilder {
    range: EhlersITrendBatchRange,
    kernel: Kernel,
}
impl EhlersITrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn warmup_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.warmup_bars = (start, end, step);
        self
    }
    #[inline]
    pub fn warmup_static(mut self, w: usize) -> Self {
        self.range.warmup_bars = (w, w, 0);
        self
    }
    #[inline]
    pub fn max_dc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.max_dc_period = (start, end, step);
        self
    }
    #[inline]
    pub fn max_dc_static(mut self, m: usize) -> Self {
        self.range.max_dc_period = (m, m, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
        ehlers_itrend_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
        Self::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn ehlers_itrend_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kernel: Kernel,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EhlersITrendError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    ehlers_itrend_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn ehlers_itrend_batch_inner(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(EhlersITrendError::InvalidBatchRange);
    }
    let rows = combos.len();
    let cols = data.len();

    rows.checked_mul(cols)
        .ok_or(EhlersITrendError::SizeOverflow {
            context: "rows*cols in batch output matrix",
        })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersITrendError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.warmup_bars.unwrap())
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let result_combos = ehlers_itrend_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(EhlersITrendBatchOutput {
        values,
        combos: result_combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ehlers_itrend_batch_inner_into(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EhlersITrendParams>, EhlersITrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(EhlersITrendError::InvalidBatchRange);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersITrendError::AllValuesNaN)?;
    let max_warmup = combos.iter().map(|c| c.warmup_bars.unwrap()).max().unwrap();
    if data.len() - first < max_warmup {
        return Err(EhlersITrendError::NotEnoughDataForWarmup {
            warmup_bars: max_warmup,
            length: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    rows.checked_mul(cols)
        .ok_or(EhlersITrendError::SizeOverflow {
            context: "rows*cols in batch output matrix (into)",
        })?;
    debug_assert_eq!(out.len(), rows * cols);

    let raw = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    #[inline(always)]
    fn precompute_wma4(src: &[f64]) -> Vec<f64> {
        let n = src.len();
        let mut fir = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = src[i];
            let x1 = if i >= 1 { src[i - 1] } else { 0.0 };
            let x2 = if i >= 2 { src[i - 2] } else { 0.0 };
            let x3 = if i >= 3 { src[i - 3] } else { 0.0 };
            fir.push(itrend_wma4_v1(x0, x1, x2, x3));
        }
        fir
    }

    let fir_series = precompute_wma4(data);

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let p = &combos[row];
        let warmup = p.warmup_bars.unwrap();
        let max_dc = p.max_dc_period.unwrap();
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        ehlers_itrend_row_scalar_tail_with_fir(data, &fir_series, warmup, max_dc, first, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[derive(Clone, Debug)]
pub struct EhlersITrendBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersITrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EhlersITrendBatchOutput {
    pub fn row_for_params(&self, p: &EhlersITrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.warmup_bars.unwrap_or(12) == p.warmup_bars.unwrap_or(12)
                && c.max_dc_period.unwrap_or(50) == p.max_dc_period.unwrap_or(50)
        })
    }
    pub fn values_for(&self, p: &EhlersITrendParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EhlersITrendBatchRange) -> Result<Vec<EhlersITrendParams>, EhlersITrendError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, EhlersITrendError> {
        if step == 0 {
            return if start == end {
                Ok(vec![start])
            } else {
                Err(EhlersITrendError::InvalidRange { start, end, step })
            };
        }
        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            loop {
                out.push(x);
                match x.checked_add(step) {
                    Some(n) if n > x && n <= end => x = n,
                    _ => break,
                }
            }
        } else {
            let mut x = start;
            loop {
                out.push(x);
                match x.checked_sub(step) {
                    Some(n) if n < x && n >= end => x = n,
                    _ => break,
                }
            }
        }
        if out.is_empty() {
            return Err(EhlersITrendError::InvalidRange { start, end, step });
        }
        Ok(out)
    }
    let warmups = axis(r.warmup_bars)?;
    let max_dcs = axis(r.max_dc_period)?;
    let cap = warmups
        .len()
        .checked_mul(max_dcs.len())
        .ok_or(EhlersITrendError::SizeOverflow {
            context: "expand_grid capacity warmups.len()*max_dcs.len()",
        })?;
    let mut out = Vec::with_capacity(cap);
    for &w in &warmups {
        for &m in &max_dcs {
            out.push(EhlersITrendParams {
                warmup_bars: Some(w),
                max_dc_period: Some(m),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn ehlers_itrend_per_slice(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    ehlers_itrend_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ehlers_itrend_batch_par_slice(
    data: &[f64],
    sweep: &EhlersITrendBatchRange,
    kern: Kernel,
) -> Result<EhlersITrendBatchOutput, EhlersITrendError> {
    ehlers_itrend_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn ehlers_itrend_row_scalar_tail(
    data: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    out: &mut [f64],
) {
    let warm = warm_index(first, warmup_bars);
    ehlers_itrend_scalar_tail(data, warmup_bars, max_dc, first, warm, out);
}

#[inline(always)]
fn ehlers_itrend_row_scalar_tail_with_fir(
    data: &[f64],
    fir: &[f64],
    warmup_bars: usize,
    max_dc: usize,
    first: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(data.len(), fir.len());
    debug_assert_eq!(data.len(), out.len());

    let warm = warm_index(first, warmup_bars);

    let length = data.len();
    let mut fir_buf = [0.0; 7];
    let mut det_buf = [0.0; 7];
    let mut i1_buf = [0.0; 7];
    let mut q1_buf = [0.0; 7];
    let (mut prev_i2, mut prev_q2) = (0.0, 0.0);
    let (mut prev_re, mut prev_im) = (0.0, 0.0);
    let (mut prev_mesa, mut prev_smooth) = (0.0, 0.0);
    let mut sum_ring = vec![0.0; max_dc];
    let mut sum_idx = 0usize;
    let (mut prev_it1, mut prev_it2, mut prev_it3) = (0.0, 0.0, 0.0);
    let mut ring_ptr = 0usize;

    #[inline(always)]
    fn ring_get(buf: &[f64; 7], center: usize, off: usize) -> f64 {
        let mut idx = center + 7 - off;
        if idx >= 7 {
            idx -= 7;
        }
        buf[idx]
    }

    for i in 0..length {
        let x0 = data[i];

        let fir_val = fir[i];
        fir_buf[ring_ptr] = fir_val;

        let fir_0 = ring_get(&fir_buf, ring_ptr, 0);
        let fir_2 = ring_get(&fir_buf, ring_ptr, 2);
        let fir_4 = ring_get(&fir_buf, ring_ptr, 4);
        let fir_6 = ring_get(&fir_buf, ring_ptr, 6);

        let h_in = itrend_hilbert4_v1(fir_0, fir_2, fir_4, fir_6);
        let period_mult = 0.075 * prev_mesa + 0.54;
        let det_val = h_in * period_mult;
        det_buf[ring_ptr] = det_val;

        let i1_val = ring_get(&det_buf, ring_ptr, 3);
        i1_buf[ring_ptr] = i1_val;

        let det_0 = ring_get(&det_buf, ring_ptr, 0);
        let det_2 = ring_get(&det_buf, ring_ptr, 2);
        let det_4 = ring_get(&det_buf, ring_ptr, 4);
        let det_6 = ring_get(&det_buf, ring_ptr, 6);
        let h_in_q1 = itrend_hilbert4_v1(det_0, det_2, det_4, det_6);
        let q1_val = h_in_q1 * period_mult;
        q1_buf[ring_ptr] = q1_val;

        let i1_0 = ring_get(&i1_buf, ring_ptr, 0);
        let i1_2 = ring_get(&i1_buf, ring_ptr, 2);
        let i1_4 = ring_get(&i1_buf, ring_ptr, 4);
        let i1_6 = ring_get(&i1_buf, ring_ptr, 6);
        let j_i_val = itrend_hilbert4_v1(i1_0, i1_2, i1_4, i1_6) * period_mult;

        let q1_0 = ring_get(&q1_buf, ring_ptr, 0);
        let q1_2 = ring_get(&q1_buf, ring_ptr, 2);
        let q1_4 = ring_get(&q1_buf, ring_ptr, 4);
        let q1_6 = ring_get(&q1_buf, ring_ptr, 6);
        let j_q_val = itrend_hilbert4_v1(q1_0, q1_2, q1_4, q1_6) * period_mult;

        let mut i2_cur = i1_val - j_q_val;
        let mut q2_cur = q1_val + j_i_val;
        i2_cur = 0.2 * i2_cur + 0.8 * prev_i2;
        q2_cur = 0.2 * q2_cur + 0.8 * prev_q2;

        let re_val = i2_cur * prev_i2 + q2_cur * prev_q2;
        let im_val = i2_cur * prev_q2 - q2_cur * prev_i2;
        prev_i2 = i2_cur;
        prev_q2 = q2_cur;

        let re_smooth = 0.2 * re_val + 0.8 * prev_re;
        let im_smooth = 0.2 * im_val + 0.8 * prev_im;
        prev_re = re_smooth;
        prev_im = im_smooth;

        let (final_mesa, sp_val, dcp) =
            itrend_period_v1(re_smooth, im_smooth, prev_mesa, prev_smooth, max_dc);
        prev_mesa = final_mesa;
        prev_smooth = sp_val;

        sum_ring[sum_idx] = x0;
        sum_idx += 1;
        if sum_idx == max_dc {
            sum_idx = 0;
        }
        let oldest = itrend_ring_oldest_v1(sum_idx, dcp, max_dc);
        let it_val =
            itrend_stable_window_mean_v1(dcp, |offset| sum_ring[(oldest + offset) % max_dc]);

        let eit_val = if i < warmup_bars {
            x0
        } else {
            itrend_wma4_v1(it_val, prev_it1, prev_it2, prev_it3)
        };

        prev_it3 = prev_it2;
        prev_it2 = prev_it1;
        prev_it1 = it_val;

        if i >= warm {
            out[i] = eit_val;
        }

        ring_ptr += 1;
        if ring_ptr == 7 {
            ring_ptr = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn reviewed_routeable_subset_close_v3() -> Vec<f64> {
        const WAVE: [f64; 11] = [
            0.000_041, -0.000_027, 0.000_013, -0.000_036, 0.000_022, -0.000_009, 0.000_033,
            -0.000_019, 0.000_006, -0.000_031, 0.000_017,
        ];
        (0..64)
            .map(|row| 1.075 + row as f64 * 0.000_000_7 + WAVE[row % WAVE.len()])
            .collect()
    }

    #[test]
    fn stable_window_mean_f64_authority_covers_offsets_extremes_subnormals_and_gaps() {
        fn assert_mean_bits(values: &[f64], expected_bits: u64) {
            let actual = itrend_stable_window_mean_v1(values.len(), |index| values[index]);
            assert_eq!(actual.to_bits(), expected_bits, "values={values:?}");
        }

        let fixture = reviewed_routeable_subset_close_v3();
        assert_mean_bits(&fixture[11..25], 0x3ff1_3342_d0f8_8cca);

        let offset = (0..37)
            .map(|index| {
                2.0_f64.powi(50)
                    + ((index % 17) as f64 - 8.0) * 0.25
                    + ((index * 7) % 5) as f64 * 0.125
            })
            .collect::<Vec<_>>();
        assert_mean_bits(&offset, 0x4310_0000_0000_0000);

        let huge = (0..43)
            .map(|index| 1e300 * (1.0 + ((index % 19) as f64 - 9.0) * 2.0_f64.powi(-49)))
            .collect::<Vec<_>>();
        assert_mean_bits(&huge, 0x7e37_e43c_8800_7592);

        let subnormal = (0..29)
            .map(|index| ((index % 13) as f64 - 6.0) * f64::from_bits(1))
            .collect::<Vec<_>>();
        assert_mean_bits(&subnormal, 0x8000_0000_0000_0001);

        let alternating = (0..31)
            .map(|index| {
                (if index % 2 == 0 { 1e250 } else { -1e250 }) + ((index % 7) as f64 - 3.0) * 1e235
            })
            .collect::<Vec<_>>();
        assert_mean_bits(&alternating, 0x7387_116f_249e_43fe);

        assert_mean_bits(&[0.0, -0.0, 0.0], 0);
        assert_eq!(
            itrend_stable_window_mean_v1(3, |index| [1.0, f64::NAN, 2.0][index]).to_bits(),
            0x7ff8_0000_0000_0000
        );
    }

    #[test]
    fn stable_f64_authority_is_bit_exact_across_scalar_batch_legacy_and_stream_routes()
    -> Result<(), Box<dyn Error>> {
        const ROW: usize = 24;
        const EXPECTED_BITS: u64 = 0x3ff1_3341_e8f7_49b5;
        let data = reviewed_routeable_subset_close_v3();
        let params = EhlersITrendParams {
            warmup_bars: Some(20),
            max_dc_period: Some(14),
        };
        let input = EhlersITrendInput::from_slice(&data, params.clone());

        let scalar = ehlers_itrend_with_kernel(&input, Kernel::Scalar)?.values;
        let auto = ehlers_itrend_with_kernel(&input, Kernel::Auto)?.values;
        let mut into = vec![0.0; data.len()];
        ehlers_itrend_into_slice(&mut into, &input, Kernel::Scalar)?;

        let sweep = EhlersITrendBatchRange {
            warmup_bars: (20, 20, 0),
            max_dc_period: (14, 14, 0),
        };
        let batch = ehlers_itrend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

        let mut legacy = vec![0.0; data.len()];
        ehlers_itrend_scalar(&data, 20, 14, 0, &mut legacy);

        let mut stream = EhlersITrendStream::try_new(params)?;
        let streamed = data
            .iter()
            .map(|&value| stream.update(value).unwrap_or(f64::NAN))
            .collect::<Vec<_>>();

        for (route, value) in [
            ("scalar", scalar[ROW]),
            ("auto", auto[ROW]),
            ("into", into[ROW]),
            ("batch", batch.values[ROW]),
            ("legacy", legacy[ROW]),
            ("stream", streamed[ROW]),
        ] {
            assert_eq!(
                value.to_bits(),
                EXPECTED_BITS,
                "{route} lost the stable f64 authority at reviewed fixture row {ROW}"
            );
        }
        Ok(())
    }

    #[test]
    fn nonfinite_gap_fails_closed_without_f64_route_drift() -> Result<(), Box<dyn Error>> {
        let mut data = reviewed_routeable_subset_close_v3();
        data[30] = f64::NAN;
        let params = EhlersITrendParams {
            warmup_bars: Some(20),
            max_dc_period: Some(14),
        };
        let input = EhlersITrendInput::from_slice(&data, params.clone());

        let authority = ehlers_itrend_with_kernel(&input, Kernel::Scalar)?.values;
        let sweep = EhlersITrendBatchRange {
            warmup_bars: (20, 20, 0),
            max_dc_period: (14, 14, 0),
        };
        let batch = ehlers_itrend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?.values;

        let mut safe = vec![0.0; data.len()];
        ehlers_itrend_safe_scalar(&data, 20, 14, &mut safe);
        let mut unchecked = vec![0.0; data.len()];
        unsafe { ehlers_itrend_unsafe_scalar(&data, 20, 14, &mut unchecked) };

        let mut stream = EhlersITrendStream::try_new(params)?;
        let streamed = data
            .iter()
            .map(|&value| stream.update(value).unwrap_or_else(itrend_qnan_v1))
            .collect::<Vec<_>>();

        for row in 20..data.len() {
            for (route, value) in [
                ("batch", batch[row]),
                ("safe", safe[row]),
                ("unchecked", unchecked[row]),
                ("stream", streamed[row]),
            ] {
                assert_eq!(
                    value.to_bits(),
                    authority[row].to_bits(),
                    "{route} drifted from the f64 authority at gapped row {row}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_ehlers_itrend_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64) * 0.01 + ((i as f64) * 0.1).sin())
            .collect();

        let input = EhlersITrendInput::from_slice(&data, EhlersITrendParams::default());

        let baseline = ehlers_itrend(&input)?.values;

        let mut out = vec![0.0; n];
        {
            ehlers_itrend_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());

        fn equal(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-12
        }

        for i in 0..n {
            assert!(
                equal(baseline[i], out[i]),
                "Mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_itrend_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = EhlersITrendParams {
            warmup_bars: None,
            max_dc_period: None,
        };
        let input = EhlersITrendInput::from_candles(&candles, "close", default_params);
        let output = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_itrend_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = EhlersITrendInput::with_default_candles(&candles);
        let result = ehlers_itrend_with_kernel(&input, kernel)?;

        let expected_last_five = [59638.12, 59497.26, 59431.08, 59391.23, 59372.19];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] EIT {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_itrend_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = EhlersITrendInput::with_default_candles(&candles);
        match input.data {
            EhlersITrendData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersITrendData::Candles"),
        }
        let output = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_itrend_no_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [];
        let params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&input_data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EIT should fail with empty data",
            test_name
        );
        Ok(())
    }

    fn check_itrend_all_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EIT should fail with all-NaN data",
            test_name
        );
        Ok(())
    }

    fn check_itrend_small_data_for_warmup(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0; 5];
        let params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EIT should fail if warmup_bars >= data length",
            test_name
        );
        Ok(())
    }

    fn check_itrend_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0, 43.0];
        let params = EhlersITrendParams {
            warmup_bars: Some(1),
            max_dc_period: Some(50),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let result = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        Ok(())
    }

    fn check_itrend_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = EhlersITrendParams {
            warmup_bars: Some(12),
            max_dc_period: Some(50),
        };
        let first_input = EhlersITrendInput::from_candles(&candles, "close", first_params);
        let first_result = ehlers_itrend_with_kernel(&first_input, kernel)?;
        let second_params = EhlersITrendParams {
            warmup_bars: Some(6),
            max_dc_period: Some(25),
        };
        let second_input = EhlersITrendInput::from_slice(&first_result.values, second_params);
        let second_result = ehlers_itrend_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "[{}] NaN found at index {} in EIT result",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_itrend_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = EhlersITrendInput::from_candles(
            &candles,
            "close",
            EhlersITrendParams {
                warmup_bars: Some(12),
                max_dc_period: Some(50),
            },
        );
        let result = ehlers_itrend_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] NaN found at index {} in EIT result",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_itrend_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let warmup_bars = 12;
        let max_dc = 50;
        let input = EhlersITrendInput::from_candles(
            &candles,
            "close",
            EhlersITrendParams {
                warmup_bars: Some(warmup_bars),
                max_dc_period: Some(max_dc),
            },
        );
        let batch_output = ehlers_itrend_with_kernel(&input, kernel)?.values;
        let mut stream = EhlersITrendStream::try_new(EhlersITrendParams {
            warmup_bars: Some(warmup_bars),
            max_dc_period: Some(max_dc),
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
                diff < 1e-7,
                "[{}] EIT streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_itrend_zero_warmup(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = EhlersITrendParams {
            warmup_bars: Some(0),
            max_dc_period: Some(10),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(matches!(
            res,
            Err(EhlersITrendError::InvalidWarmupBars { .. })
        ));
        Ok(())
    }

    fn check_itrend_invalid_max_dc(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = EhlersITrendParams {
            warmup_bars: Some(1),
            max_dc_period: Some(0),
        };
        let input = EhlersITrendInput::from_slice(&data, params);
        let res = ehlers_itrend_with_kernel(&input, kernel);
        assert!(matches!(
            res,
            Err(EhlersITrendError::InvalidMaxDcPeriod { .. })
        ));
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_itrend_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_warmup_bars = vec![1, 5, 12, 20, 30];
        let test_max_dc_periods = vec![35, 40, 50, 60];
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for warmup in &test_warmup_bars {
            for max_dc in &test_max_dc_periods {
                if *warmup >= *max_dc {
                    continue;
                }

                for source in &test_sources {
                    let input = EhlersITrendInput::from_candles(
                        &candles,
                        source,
                        EhlersITrendParams {
                            warmup_bars: Some(*warmup),
                            max_dc_period: Some(*max_dc),
                        },
                    );
                    let output = ehlers_itrend_with_kernel(&input, kernel)?;

                    for (i, &val) in output.values.iter().enumerate() {
                        if val.is_nan() {
                            continue;
                        }

                        let bits = val.to_bits();

                        if bits == 0x11111111_11111111 {
                            panic!(
                                "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with warmup={}, max_dc={}, source={}",
                                test_name, val, bits, i, warmup, max_dc, source
                            );
                        }

                        if bits == 0x22222222_22222222 {
                            panic!(
                                "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with warmup={}, max_dc={}, source={}",
                                test_name, val, bits, i, warmup, max_dc, source
                            );
                        }

                        if bits == 0x33333333_33333333 {
                            panic!(
                                "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with warmup={}, max_dc={}, source={}",
                                test_name, val, bits, i, warmup, max_dc, source
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_itrend_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[allow(clippy::float_cmp)]
    fn check_itrend_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (4usize..=15).prop_flat_map(|warmup| {
            ((warmup + 1)..=64).prop_flat_map(move |max_dc| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        (warmup + max_dc + 4)..400,
                    ),
                    Just(warmup),
                    Just(max_dc),
                    (1e-3f64..1e3f64).prop_filter("a>0", |a| a.is_finite()),
                    -1e3f64..1e3f64,
                )
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, warmup, max_dc, a, b)| {
                let params = EhlersITrendParams {
                    warmup_bars: Some(warmup),
                    max_dc_period: Some(max_dc),
                };
                let input = EhlersITrendInput::from_slice(&data, params.clone());

                let fast = ehlers_itrend_with_kernel(&input, kernel);
                let slow = ehlers_itrend_with_kernel(&input, Kernel::Scalar);

                match (fast, slow) {
                    (Err(e1), Err(e2))
                        if std::mem::discriminant(&e1) == std::mem::discriminant(&e2) =>
                    {
                        return Ok(());
                    }
                    (Err(e1), Err(e2)) => {
                        prop_assert!(false, "different errors: fast={:?} slow={:?}", e1, e2)
                    }

                    (Err(e), Ok(_)) => {
                        prop_assert!(false, "fast errored {e:?} but scalar succeeded")
                    }
                    (Ok(_), Err(e)) => {
                        prop_assert!(false, "scalar errored {e:?} but fast succeeded")
                    }

                    (Ok(fast), Ok(reference)) => {
                        let EhlersITrendOutput { values: out } = fast;
                        let EhlersITrendOutput { values: rref } = reference;

                        let mut stream = EhlersITrendStream::try_new(params.clone()).unwrap();
                        let mut s_out = Vec::with_capacity(data.len());
                        for &v in &data {
                            s_out.push(stream.update(v).unwrap_or(f64::NAN));
                        }

                        let transformed: Vec<f64> = data.iter().map(|x| a * x + b).collect();
                        let t_out =
                            ehlers_itrend(&EhlersITrendInput::from_slice(&transformed, params))?
                                .values;

                        let first = data.iter().position(|x| !x.is_nan()).unwrap();
                        let warm = first + warmup;
                        for i in warm..data.len() {
                            let y = out[i];
                            let yr = rref[i];
                            let ys = s_out[i];
                            let yt = t_out[i];

                            let start = (i + 1).saturating_sub(max_dc);
                            let look = &data[start..=i];
                            prop_assert!(
                                y.is_nan() || y.is_finite(),
                                "iTrend output at index {} is not finite: {}",
                                i,
                                y
                            );

                            if warmup == 1 && y.is_finite() {
                                prop_assert!((y - data[i]).abs() <= f64::EPSILON);
                            }

                            if look.iter().all(|v| *v == look[0]) {
                                prop_assert!((y - look[0]).abs() <= 1e-7);
                            }

                            let affine_start = warm + max_dc;
                            if i >= affine_start {
                                let expected = a * y + b;
                                let diff = (yt - expected).abs();
                                let tol = 1e-7_f64.max(expected.abs() * 1e-7);
                                let ulp = yt.to_bits().abs_diff(expected.to_bits());
                                prop_assert!(
                                    diff <= tol || ulp <= 8,
                                    "idx {i}: affine mismatch diff={diff:e}  ULP={ulp}"
                                );
                            }

                            let ulp = y.to_bits().abs_diff(yr.to_bits());
                            prop_assert!(
                                (y - yr).abs() <= 1e-7 || ulp <= 4,
                                "idx {i}: fast={y} ref={yr} ULP={ulp}"
                            );

                            prop_assert!(
                                (y - ys).abs() <= 1e-7 || (y.is_nan() && ys.is_nan()),
                                "idx {i}: stream mismatch"
                            );
                        }

                        for j in first..warm {
                            prop_assert!(
                                out[j].is_nan(),
                                "warm-up NaN expected at idx {j}: got {}",
                                out[j]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        assert!(
            ehlers_itrend(&EhlersITrendInput::from_slice(
                &[],
                EhlersITrendParams::default()
            ))
            .is_err()
        );
        assert!(
            ehlers_itrend(&EhlersITrendInput::from_slice(
                &[f64::NAN; 12],
                EhlersITrendParams::default()
            ))
            .is_err()
        );
        assert!(
            ehlers_itrend(&EhlersITrendInput::from_slice(
                &[1.0; 5],
                EhlersITrendParams {
                    warmup_bars: Some(8),
                    max_dc_period: Some(50)
                }
            ))
            .is_err()
        );
        assert!(
            ehlers_itrend(&EhlersITrendInput::from_slice(
                &[1.0; 5],
                EhlersITrendParams {
                    warmup_bars: Some(0),
                    max_dc_period: Some(10)
                }
            ))
            .is_err()
        );

        Ok(())
    }

    macro_rules! generate_all_itrend_tests {
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
            }
        }
    }

    generate_all_itrend_tests!(
        check_itrend_partial_params,
        check_itrend_accuracy,
        check_itrend_default_candles,
        check_itrend_no_data,
        check_itrend_all_nan_data,
        check_itrend_small_data_for_warmup,
        check_itrend_very_small_dataset,
        check_itrend_reinput,
        check_itrend_nan_handling,
        check_itrend_streaming,
        check_itrend_zero_warmup,
        check_itrend_invalid_max_dc,
        check_itrend_property,
        check_itrend_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = EhlersITrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EhlersITrendParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59638.12, 59497.26, 59431.08, 59391.23, 59372.19];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    fn check_batch_invalid_kernel(test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let data = [1.0, 2.0, 3.0];
        let sweep = EhlersITrendBatchRange::default();
        let res = ehlers_itrend_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(matches!(
            res,
            Err(EhlersITrendError::InvalidKernelForBatch(_))
        ));
        Ok(())
    }

    fn check_batch_invalid_range(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0];

        let sweep = EhlersITrendBatchRange {
            warmup_bars: (5, 1, 0),
            max_dc_period: (10, 10, 0),
        };
        let res = ehlers_itrend_batch_with_kernel(&data, &sweep, kernel);
        assert!(matches!(res, Err(EhlersITrendError::InvalidRange { .. })));
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for source in &test_sources {
            let output = EhlersITrendBatchBuilder::new()
                .kernel(kernel)
                .warmup_range(5, 30, 5)
                .max_dc_range(35, 60, 5)
                .apply_candles(&c, source)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }
            }
        }

        let edge_case_configs = vec![
            (5, 10, 1, 15, 50, 5),
            (20, 30, 5, 40, 60, 10),
            (5, 15, 2, 20, 40, 3),
        ];

        for (warmup_start, warmup_end, warmup_step, max_dc_start, max_dc_end, max_dc_step) in
            edge_case_configs
        {
            let output = EhlersITrendBatchBuilder::new()
                .kernel(kernel)
                .warmup_range(warmup_start, warmup_end, warmup_step)
                .max_dc_range(max_dc_start, max_dc_end, max_dc_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at row {} col {} with warmup_range({},{},{}) and max_dc_range({},{},{})",
                        test,
                        val,
                        bits,
                        row,
                        col,
                        warmup_start,
                        warmup_end,
                        warmup_step,
                        max_dc_start,
                        max_dc_end,
                        max_dc_step
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
    gen_batch_tests!(check_batch_invalid_kernel);
    gen_batch_tests!(check_batch_invalid_range);
    gen_batch_tests!(check_batch_no_poison);
}
