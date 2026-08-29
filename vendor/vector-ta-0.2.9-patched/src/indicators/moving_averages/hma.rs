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
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
impl<'a> AsRef<[f64]> for HmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HmaData::Slice(slice) => slice,
            HmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HmaParams {
    pub period: Option<usize>,
}

impl Default for HmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct HmaInput<'a> {
    pub data: HmaData<'a>,
    pub params: HmaParams,
}

impl<'a> HmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HmaParams) -> Self {
        Self {
            data: HmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HmaParams) -> Self {
        Self {
            data: HmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for HmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<HmaOutput, HmaError> {
        let p = HmaParams {
            period: self.period,
        };
        let i = HmaInput::from_candles(c, "close", p);
        hma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HmaOutput, HmaError> {
        let p = HmaParams {
            period: self.period,
        };
        let i = HmaInput::from_slice(d, p);
        hma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<HmaStream, HmaError> {
        let p = HmaParams {
            period: self.period,
        };
        HmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HmaError {
    #[error("hma: No data provided.")]
    NoData,

    #[error("hma: All values are NaN.")]
    AllValuesNaN,

    #[error("hma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("hma: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("hma: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("hma: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("hma: arithmetic overflow when computing {what}")]
    ArithmeticOverflow { what: &'static str },

    #[error("hma: Cannot calculate half of period: period = {period}")]
    ZeroHalf { period: usize },

    #[error("hma: Cannot calculate sqrt of period: period = {period}")]
    ZeroSqrtPeriod { period: usize },

    #[error("hma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn hma(input: &HmaInput) -> Result<HmaOutput, HmaError> {
    hma_with_kernel(input, Kernel::Auto)
}

pub fn hma_into(input: &HmaInput, out: &mut [f64]) -> Result<(), HmaError> {
    hma_into_internal(input, out)
}

#[inline]
fn hma_into_internal(input: &HmaInput, out: &mut [f64]) -> Result<(), HmaError> {
    hma_with_kernel_into(input, Kernel::Auto, out)
}

pub fn hma_with_kernel(input: &HmaInput, kernel: Kernel) -> Result<HmaOutput, HmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(HmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let half = period / 2;
    if half == 0 {
        return Err(HmaError::ZeroHalf { period });
    }
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    if sqrt_len == 0 {
        return Err(HmaError::ZeroSqrtPeriod { period });
    }
    if len - first < period + sqrt_len - 1 {
        return Err(HmaError::NotEnoughValidData {
            needed: period + sqrt_len - 1,
            valid: len - first,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let first_out = first + period + sqrt_len - 2;
    let mut out = alloc_with_nan_prefix(len, first_out);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch if period == 5 => {
                hma_scalar_period5(data, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch if period == 5 => {
                hma_scalar_period5(data, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch if period == 5 => {
                hma_scalar_period5(data, first, &mut out)
            }
            Kernel::Scalar | Kernel::ScalarBatch => hma_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(HmaOutput { values: out })
}

fn hma_with_kernel_into(input: &HmaInput, kernel: Kernel, out: &mut [f64]) -> Result<(), HmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }
    if out.len() != len {
        return Err(HmaError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(HmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let half = period / 2;
    if half == 0 {
        return Err(HmaError::ZeroHalf { period });
    }
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    if sqrt_len == 0 {
        return Err(HmaError::ZeroSqrtPeriod { period });
    }
    if len - first < period + sqrt_len - 1 {
        return Err(HmaError::NotEnoughValidData {
            needed: period + sqrt_len - 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch if period == 5 => {
                hma_scalar_period5(data, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch if period == 5 => hma_scalar_period5(data, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch if period == 5 => {
                hma_scalar_period5(data, first, out)
            }
            Kernel::Scalar | Kernel::ScalarBatch => hma_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }

    let first_out = first + period + sqrt_len - 2;
    for v in &mut out[..first_out] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
fn hma_scalar_period5(data: &[f64], first: usize, out: &mut [f64]) {
    let len = data.len();
    let first_out = first + 5;
    if first_out >= len {
        return;
    }
    let inv45 = 1.0 / 45.0;
    for i in first_out..len {
        let d2 = data[i - 2];
        out[i] = if d2.is_nan() {
            f64::NAN
        } else {
            (30.0 * data[i] + 27.0 * data[i - 1]
                - 7.0 * data[i - 3]
                - 4.0 * data[i - 4]
                - data[i - 5])
                * inv45
        };
    }
}

#[inline]
pub fn hma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    if period < 2 || first >= len || period > len - first {
        return;
    }
    let half = period / 2;
    if half == 0 {
        return;
    }
    let sq = (period as f64).sqrt().floor() as usize;
    if sq == 0 {
        return;
    }

    let first_out = first + period + sq - 2;
    if first_out >= len {
        return;
    }

    let ws_half = (half * (half + 1) / 2) as f64;
    let ws_full = (period * (period + 1) / 2) as f64;
    let ws_sqrt = (sq * (sq + 1) / 2) as f64;
    let half_f = half as f64;
    let period_f = period as f64;
    let sq_f = sq as f64;

    let (mut s_half, mut ws_half_acc) = (0.0, 0.0);
    let (mut s_full, mut ws_full_acc) = (0.0, 0.0);
    let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

    let mut x_buf = vec![0.0f64; sq];
    let mut x_sum = 0.0;
    let mut x_wsum = 0.0;
    let mut x_head = 0usize;

    let start = first;

    for j in 0..half {
        let v = data[start + j];
        let jf = j as f64 + 1.0;
        s_full += v;
        ws_full_acc = jf.mul_add(v, ws_full_acc);
        s_half += v;
        ws_half_acc = jf.mul_add(v, ws_half_acc);
    }
    wma_half = ws_half_acc / ws_half;

    if period > half + 1 {
        for j in half..(period - 1) {
            let idx = start + j;
            let v = data[idx];

            let jf = j as f64 + 1.0;
            s_full += v;
            ws_full_acc = jf.mul_add(v, ws_full_acc);

            let old_h = data[idx - half];
            let prev = s_half;
            s_half = prev + v - old_h;
            ws_half_acc = half_f.mul_add(v, ws_half_acc - prev);
            wma_half = ws_half_acc / ws_half;
        }
    }

    {
        let j = period - 1;
        let idx = start + j;
        let v = data[idx];

        let jf = j as f64 + 1.0;
        s_full += v;
        ws_full_acc = jf.mul_add(v, ws_full_acc);
        wma_full = ws_full_acc / ws_full;

        let old_h = data[idx - half];
        let prev = s_half;
        s_half = prev + v - old_h;
        ws_half_acc = half_f.mul_add(v, ws_half_acc - prev);
        wma_half = ws_half_acc / ws_half;

        let x = 2.0 * wma_half - wma_full;
        x_buf[0] = x;
        x_sum += x;
        x_wsum = 1.0f64.mul_add(x, x_wsum);

        if sq == 1 {
            out[first_out] = x_wsum / ws_sqrt;
        }
    }

    if sq > 1 {
        for j in period..(period + sq - 1) {
            let idx = start + j;
            let v = data[idx];

            let old_f = data[idx - period];
            let prev_f = s_full;
            s_full = prev_f + v - old_f;
            ws_full_acc = period_f.mul_add(v, ws_full_acc - prev_f);
            wma_full = ws_full_acc / ws_full;

            let old_h = data[idx - half];
            let prev_h = s_half;
            s_half = prev_h + v - old_h;
            ws_half_acc = half_f.mul_add(v, ws_half_acc - prev_h);
            wma_half = ws_half_acc / ws_half;

            let x = 2.0 * wma_half - wma_full;
            let pos = j + 1 - period;
            x_buf[pos] = x;
            x_sum += x;
            x_wsum = (pos as f64 + 1.0).mul_add(x, x_wsum);

            if pos + 1 == sq {
                out[first_out] = x_wsum / ws_sqrt;
            }
        }
    }

    let mut j = period + sq - 1;
    while j < len - start {
        let idx = start + j;
        let v = data[idx];

        let old_f = data[idx - period];
        let prev_f = s_full;
        s_full = prev_f + v - old_f;
        ws_full_acc = period_f.mul_add(v, ws_full_acc - prev_f);
        wma_full = ws_full_acc / ws_full;

        let old_h = data[idx - half];
        let prev_h = s_half;
        s_half = prev_h + v - old_h;
        ws_half_acc = half_f.mul_add(v, ws_half_acc - prev_h);
        wma_half = ws_half_acc / ws_half;

        let x = 2.0 * wma_half - wma_full;
        let old_x = x_buf[x_head];
        x_buf[x_head] = x;
        x_head += 1;
        if x_head == sq {
            x_head = 0;
        }

        let prev_sum = x_sum;
        x_sum = prev_sum + x - old_x;
        x_wsum = sq_f.mul_add(x, x_wsum - prev_sum);

        out[idx] = x_wsum / ws_sqrt;
        j += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn hma_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    hma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn hma_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use aligned_vec::AVec;
    use core::arch::x86_64::*;

    let len = data.len();
    if period < 2 || first >= len || period > len - first {
        return;
    }
    let half = period / 2;
    if half == 0 {
        return;
    }
    let sq = (period as f64).sqrt().floor() as usize;
    debug_assert!(
        sq > 0 && sq <= 65_535,
        "HMA: √period must fit in 16-bit to keep Σw < 2^53"
    );
    if sq == 0 {
        return;
    }
    let first_out = first + period + sq - 2;
    if first_out >= len {
        return;
    }

    let ws_half = (half * (half + 1) / 2) as f64;
    let ws_full = (period * (period + 1) / 2) as f64;
    let ws_sqrt = (sq * (sq + 1) / 2) as f64;
    let sq_f = sq as f64;

    let (mut s_half, mut ws_half_acc) = (0.0, 0.0);
    let (mut s_full, mut ws_full_acc) = (0.0, 0.0);
    let (mut wma_half, mut wma_full) = (f64::NAN, f64::NAN);

    let sq_aligned = (sq + 7) & !7;
    let mut x_buf: AVec<f64> = AVec::with_capacity(64, sq_aligned);
    x_buf.resize(sq_aligned, 0.0);

    let mut x_sum = 0.0;
    let mut x_wsum = 0.0;
    let mut x_head = 0usize;

    const W_RAMP_ARR: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let w_ramp: __m512d = _mm512_loadu_pd(W_RAMP_ARR.as_ptr());

    #[inline(always)]
    unsafe fn horiz_sum(z: __m512d) -> f64 {
        let hi = _mm512_extractf64x4_pd(z, 1);
        let lo = _mm512_castpd512_pd256(z);

        let sum256 = _mm256_add_pd(hi, lo);

        let sum128 = _mm256_hadd_pd(sum256, sum256);

        let hi128 = _mm256_extractf128_pd(sum128, 1);
        let lo128 = _mm256_castpd256_pd128(sum128);
        let final_sum = _mm_add_pd(hi128, lo128);

        _mm_cvtsd_f64(final_sum)
    }

    for j in 0..(period + sq - 1) {
        let idx = first + j;
        let val = *data.get_unchecked(idx);

        if j < period {
            s_full += val;
            ws_full_acc += (j as f64 + 1.0) * val;
        } else {
            let old = *data.get_unchecked(idx - period);
            let prev = s_full;
            s_full = prev + val - old;
            ws_full_acc = ws_full_acc - prev + (period as f64) * val;
        }

        if j < half {
            s_half += val;
            ws_half_acc += (j as f64 + 1.0) * val;
        } else {
            let old = *data.get_unchecked(idx - half);
            let prev = s_half;
            s_half = prev + val - old;
            ws_half_acc = ws_half_acc - prev + (half as f64) * val;
        }

        if j + 1 >= half {
            wma_half = ws_half_acc / ws_half;
        }
        if j + 1 >= period {
            wma_full = ws_full_acc / ws_full;
        }

        if j + 1 >= period {
            let x_val = 2.0 * wma_half - wma_full;
            let pos = (j + 1 - period) as usize;

            if pos < sq {
                *x_buf.get_unchecked_mut(pos) = x_val;
                x_sum += x_val;

                if pos + 1 == sq {
                    let mut acc = _mm512_setzero_pd();
                    let mut i = 0usize;
                    let mut off = 0.0;
                    while i + 8 <= sq {
                        let x = _mm512_loadu_pd(x_buf.as_ptr().add(i));

                        let w = _mm512_add_pd(w_ramp, _mm512_set1_pd(off + 1.0));
                        acc = _mm512_fmadd_pd(x, w, acc);
                        i += 8;
                        off += 8.0;
                    }
                    x_wsum = horiz_sum(acc);
                    for k in i..sq {
                        x_wsum += x_buf[k] * (k as f64 + 1.0);
                    }
                    *out.get_unchecked_mut(first_out) = x_wsum / ws_sqrt;
                }
            }
        }
    }

    for j in (period + sq - 1)..(len - first) {
        let idx = first + j;
        let val = *data.get_unchecked(idx);

        let old_f = *data.get_unchecked(idx - period);
        let old_h = *data.get_unchecked(idx - half);

        let sum_vec = _mm_set_pd(s_full, s_half);
        let old_vec = _mm_set_pd(old_f, old_h);
        let ws_vec = _mm_set_pd(ws_full_acc, ws_half_acc);
        let weights = _mm_set_pd(period as f64, half as f64);
        let v_val = _mm_set1_pd(val);

        let new_sum_vec = _mm_add_pd(_mm_sub_pd(sum_vec, old_vec), v_val);

        let diff = _mm_sub_pd(ws_vec, sum_vec);
        let new_ws_vec = _mm_fmadd_pd(v_val, weights, diff);

        s_full = _mm_cvtsd_f64(_mm_unpackhi_pd(new_sum_vec, new_sum_vec));
        s_half = _mm_cvtsd_f64(new_sum_vec);
        ws_full_acc = _mm_cvtsd_f64(_mm_unpackhi_pd(new_ws_vec, new_ws_vec));
        ws_half_acc = _mm_cvtsd_f64(new_ws_vec);

        wma_full = ws_full_acc / ws_full;
        wma_half = ws_half_acc / ws_half;
        let x_val = 2.0 * wma_half - wma_full;

        let old_x = *x_buf.get_unchecked(x_head);
        *x_buf.get_unchecked_mut(x_head) = x_val;
        x_head = (x_head + 1) % sq;

        let prev_sum = x_sum;
        x_sum = prev_sum + x_val - old_x;
        x_wsum = sq_f.mul_add(x_val, x_wsum - prev_sum);

        *out.get_unchecked_mut(idx) = x_wsum / ws_sqrt;

        let pf = core::cmp::min(idx + 128, len - 1);
        _mm_prefetch(data.as_ptr().add(pf) as *const i8, _MM_HINT_T1);
    }
}

#[derive(Debug, Clone)]
struct LinWma {
    period: usize,
    inv_norm: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    count: usize,

    sum: f64,
    wsum: f64,
    nan_count: usize,
    dirty: bool,
}

impl LinWma {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let norm = (period as f64) * ((period as f64) + 1.0) * 0.5;
        Self {
            period,
            inv_norm: 1.0 / norm,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            count: 0,
            sum: 0.0,
            wsum: 0.0,
            nan_count: 0,
            dirty: false,
        }
    }

    #[inline(always)]
    fn rebuild(&mut self) {
        self.sum = 0.0;
        self.wsum = 0.0;
        self.nan_count = 0;

        let mut idx = self.head;
        for i in 0..self.period {
            let v = self.buffer[idx];
            if v.is_nan() {
                self.nan_count += 1;
            } else {
                self.sum += v;
                self.wsum = (i as f64 + 1.0).mul_add(v, self.wsum);
            }
            idx = if idx + 1 == self.period { 0 } else { idx + 1 };
        }
        self.dirty = self.nan_count != 0;
        debug_assert!(self.nan_count == 0, "rebuild expected clean window");
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let n = self.period as f64;

        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = if self.head + 1 == self.period {
            0
        } else {
            self.head + 1
        };

        if !self.filled {
            self.count += 1;

            if value.is_nan() {
                self.nan_count += 1;
                self.dirty = true;
            } else {
                self.sum += value;
                self.wsum = (self.count as f64).mul_add(value, self.wsum);
            }

            if self.count == self.period {
                self.filled = true;

                return Some(if self.nan_count > 0 {
                    f64::NAN
                } else {
                    self.wsum * self.inv_norm
                });
            }
            return None;
        }

        if old.is_nan() {
            self.nan_count = self.nan_count.saturating_sub(1);
        }
        if value.is_nan() {
            self.nan_count += 1;
        }

        if self.nan_count > 0 {
            self.dirty = true;
            return Some(f64::NAN);
        }

        if self.dirty {
            self.rebuild();
            self.dirty = false;
            debug_assert_eq!(self.nan_count, 0);
            return Some(self.wsum * self.inv_norm);
        }

        let prev_sum = self.sum;
        self.sum = prev_sum + value - old;
        self.wsum = n.mul_add(value, self.wsum - prev_sum);

        Some(self.wsum * self.inv_norm)
    }
}

#[derive(Debug, Clone)]
pub struct HmaStream {
    wma_half: LinWma,
    wma_full: LinWma,
    wma_sqrt: LinWma,
}

impl HmaStream {
    pub fn try_new(params: HmaParams) -> Result<Self, HmaError> {
        let period = params.period.unwrap_or(5);
        if period < 2 {
            return Err(HmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let half = period / 2;
        if half == 0 {
            return Err(HmaError::ZeroHalf { period });
        }
        let sqrt_len = (period as f64).sqrt().floor() as usize;
        if sqrt_len == 0 {
            return Err(HmaError::ZeroSqrtPeriod { period });
        }

        Ok(Self {
            wma_half: LinWma::new(half),
            wma_full: LinWma::new(period),
            wma_sqrt: LinWma::new(sqrt_len),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let full = self.wma_full.update(value);
        let half = self.wma_half.update(value);

        if let (Some(f), Some(h)) = (full, half) {
            let x = 2.0f64.mul_add(h, -f);
            self.wma_sqrt.update(x)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct HmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for HmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HmaBatchBuilder {
    range: HmaBatchRange,
    kernel: Kernel,
}

impl HmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<HmaBatchOutput, HmaError> {
        hma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<HmaBatchOutput, HmaError> {
        HmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<HmaBatchOutput, HmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HmaBatchOutput, HmaError> {
        HmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn hma_batch_with_kernel(
    data: &[f64],
    sweep: &HmaBatchRange,
    k: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HmaError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    hma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HmaBatchOutput {
    pub fn row_for_params(&self, p: &HmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &HmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HmaBatchRange) -> Vec<HmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }

        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let mut v = Vec::new();
        let mut x = lo;
        while x <= hi {
            v.push(x);
            match x.checked_add(step) {
                Some(nx) => x = nx,
                None => break,
            }
        }
        v
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(HmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn hma_batch_slice(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    hma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn hma_batch_par_slice(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
) -> Result<HmaBatchOutput, HmaError> {
    hma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn hma_batch_inner(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HmaBatchOutput, HmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(HmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let s = (p as f64).sqrt().floor() as usize;
            first + p + s - 2
        })
        .collect();

    let _ = rows
        .checked_mul(cols)
        .ok_or(HmaError::ArithmeticOverflow { what: "rows*cols" })?;
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
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

    let rows = combos.len();
    let cols = data.len();
    let _ = rows
        .checked_mul(cols)
        .ok_or(HmaError::ArithmeticOverflow { what: "rows*cols" })?;

    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(HmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn hma_batch_inner_into(
    data: &[f64],
    sweep: &HmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(Vec<HmaParams>, usize, usize), HmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(HmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let expected = rows
        .checked_mul(cols)
        .ok_or(HmaError::ArithmeticOverflow { what: "rows*cols" })?;
    if out.len() != expected {
        return Err(HmaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let s = (p as f64).sqrt().floor() as usize;
            first + p + s - 2
        })
        .collect();

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => hma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => hma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => hma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
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

    Ok((combos, rows, cols))
}

#[inline(always)]
pub unsafe fn hma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period == 5 {
        hma_scalar_period5(data, first, out);
    } else {
        hma_scalar(data, period, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period == 5 {
        hma_scalar_period5(data, first, out);
    } else {
        hma_avx2(data, period, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period == 5 {
        hma_scalar_period5(data, first, out);
    } else {
        hma_avx512(data, period, first, out);
    }
}

#[inline(always)]
fn expand_grid_hma(r: &HmaBatchRange) -> Vec<HmaParams> {
    expand_grid(r)
}

#[inline]
pub fn hma_into_slice(dst: &mut [f64], input: &HmaInput, kern: Kernel) -> Result<(), HmaError> {
    let data: &[f64] = input.as_ref();

    if dst.len() != data.len() {
        return Err(HmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    hma_with_kernel_into(input, kern, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use proptest::prelude::*;

    #[test]
    fn test_hma_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data: Vec<f64> = (0..512)
            .map(|i| ((i as f64).sin() * 123.456789) + (i as f64) * 0.25)
            .collect();

        let input = HmaInput::from_slice(&data, HmaParams::default());

        let baseline = hma(&input)?.values;

        let mut out = vec![0.0; data.len()];
        hma_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());
        for (a, b) in baseline.iter().zip(out.iter()) {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(equal, "Mismatch: a={:?}, b={:?}", a, b);
        }

        Ok(())
    }

    fn check_hma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = HmaParams { period: None };
        let input_default = HmaInput::from_candles(&candles, "close", default_params);
        let output_default = hma_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    fn check_hma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = HmaInput::with_default_candles(&candles);
        let result = hma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(result.values.len(), candles.close.len());
        let start = result.values.len() - 5;
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-3,
                "[{}] idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_hma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(0) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] HMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_hma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(10) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] HMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_hma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = HmaParams { period: Some(5) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] HMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_hma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = HmaParams { period: Some(5) };
        let first_input = HmaInput::from_candles(&candles, "close", first_params);
        let first_result = hma_with_kernel(&first_input, kernel)?;
        let second_params = HmaParams { period: Some(3) };
        let second_input = HmaInput::from_slice(&first_result.values, second_params);
        let second_result = hma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_hma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = HmaParams::default();
        let period = params.period.unwrap_or(5) * 2;
        let input = HmaInput::from_candles(&candles, "close", params);
        let result = hma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_hma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = HmaInput::from_slice(&empty, HmaParams::default());
        let res = hma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HmaError::NoData)),
            "[{}] expected NoData",
            test_name
        );
        Ok(())
    }

    fn check_hma_not_enough_valid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, 1.0, 2.0];
        let params = HmaParams { period: Some(3) };
        let input = HmaInput::from_slice(&data, params);
        let res = hma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HmaError::NotEnoughValidData { .. })),
            "[{}] expected NotEnoughValidData",
            test_name
        );
        Ok(())
    }

    fn check_hma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 5;
        let input = HmaInput::from_candles(
            &candles,
            "close",
            HmaParams {
                period: Some(period),
            },
        );
        let batch_output = hma_with_kernel(&input, kernel)?.values;

        let mut stream = HmaStream::try_new(HmaParams {
            period: Some(period),
        })?;
        let mut stream_vals = Vec::with_capacity(candles.close.len());
        for &p in &candles.close {
            match stream.update(p) {
                Some(v) => stream_vals.push(v),
                None => stream_vals.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_vals.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_vals.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-4,
                "[{}] HMA streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_hma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close_data = &candles.close;

        let strat = (
            2usize..=100,
            0usize..close_data.len().saturating_sub(500),
            200usize..=500,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period + 10 {
                    return Ok(());
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = HmaParams {
                    period: Some(period),
                };
                let input = HmaInput::from_slice(data_slice, params);

                let result = hma_with_kernel(&input, kernel);

                let scalar_result = hma_with_kernel(&input, Kernel::Scalar);

                match (result, scalar_result) {
                    (Ok(HmaOutput { values: out }), Ok(HmaOutput { values: ref_out })) => {
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        let sqrt_period = (period as f64).sqrt().floor() as usize;
                        let expected_warmup = period + sqrt_period - 1;

                        let first_valid = out.iter().position(|x| !x.is_nan());
                        if let Some(first_idx) = first_valid {
                            prop_assert!(
                                first_idx >= expected_warmup - 1,
                                "First valid at {} but expected warmup is {}",
                                first_idx,
                                expected_warmup
                            );

                            for i in 0..first_idx {
                                prop_assert!(
                                    out[i].is_nan(),
                                    "Expected NaN at index {} during warmup, got {}",
                                    i,
                                    out[i]
                                );
                            }
                        }

                        for i in 0..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            if y.is_nan() {
                                prop_assert!(
                                    r.is_nan(),
                                    "Kernel mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                                continue;
                            }

                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();
                            let ulp_diff = y_bits.abs_diff(r_bits);

                            let ulp_tolerance = if matches!(kernel, Kernel::Avx512) {
                                20000
                            } else {
                                8
                            };
                            prop_assert!(
                                (y - r).abs() <= 1e-8 || ulp_diff <= ulp_tolerance,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );
                        }

                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            if y.is_nan() {
                                continue;
                            }

                            prop_assert!(y.is_finite(), "HMA output at {} is not finite: {}", i, y);

                            if i >= period * 2 {
                                let window_start = i.saturating_sub(period);
                                let window = &data_slice[window_start..=i];
                                let is_constant =
                                    window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                                if is_constant {
                                    let constant_val = window[0];
                                    prop_assert!(
										(y - constant_val).abs() <= 1e-6,
										"HMA should converge to {} for constant data, got {} at index {}",
										constant_val,
										y,
										i
									);
                                }
                            }
                        }

                        if period == 2 {
                            let min_valid_idx = expected_warmup;
                            if out.len() > min_valid_idx {
                                prop_assert!(
                                    out[min_valid_idx].is_finite(),
                                    "HMA with period=2 should produce valid output at index {}",
                                    min_valid_idx
                                );
                            }
                        }

                        Ok(())
                    }
                    (Err(e1), Err(_e2)) => {
                        prop_assert!(
                            format!("{:?}", e1).contains("NotEnoughValidData")
                                || format!("{:?}", e1).contains("InvalidPeriod"),
                            "Unexpected error type: {:?}",
                            e1
                        );
                        Ok(())
                    }
                    (Ok(_), Err(e)) | (Err(e), Ok(_)) => {
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed with {:?}",
                            e
                        );
                        Ok(())
                    }
                }
            })
            .unwrap();

        let edge_cases = vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0], 2),
            (vec![42.0; 100], 10),
            ((0..100).map(|i| i as f64).collect::<Vec<_>>(), 15),
            ((0..100).map(|i| 100.0 - i as f64).collect::<Vec<_>>(), 20),
        ];

        for (case_idx, (data, period)) in edge_cases.into_iter().enumerate() {
            let params = HmaParams {
                period: Some(period),
            };
            let input = HmaInput::from_slice(&data, params);

            match hma_with_kernel(&input, kernel) {
                Ok(out) => {
                    let has_valid = out.values.iter().any(|&x| x.is_finite() && !x.is_nan());
                    assert!(
                        has_valid || data.len() < period + 2,
                        "[{}] Edge case {} produced no valid output",
                        test_name,
                        case_idx
                    );
                }
                Err(_) => {}
            }
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_hma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            HmaParams::default(),
            HmaParams { period: Some(2) },
            HmaParams { period: Some(3) },
            HmaParams { period: Some(4) },
            HmaParams { period: Some(5) },
            HmaParams { period: Some(7) },
            HmaParams { period: Some(10) },
            HmaParams { period: Some(14) },
            HmaParams { period: Some(20) },
            HmaParams { period: Some(30) },
            HmaParams { period: Some(50) },
            HmaParams { period: Some(100) },
            HmaParams { period: Some(200) },
            HmaParams { period: Some(1) },
            HmaParams { period: Some(250) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = HmaInput::from_candles(&candles, "close", params.clone());
            let output = hma_with_kernel(&input, kernel)?;

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
                        params.period.unwrap_or(5)
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
                        params.period.unwrap_or(5)
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
                        params.period.unwrap_or(5)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_hma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_hma_tests {
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

    generate_all_hma_tests!(
        check_hma_partial_params,
        check_hma_accuracy,
        check_hma_zero_period,
        check_hma_period_exceeds_length,
        check_hma_very_small_dataset,
        check_hma_reinput,
        check_hma_nan_handling,
        check_hma_empty_input,
        check_hma_not_enough_valid,
        check_hma_streaming,
        check_hma_property,
        check_hma_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = HmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
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
            (5, 25, 5),
            (10, 50, 10),
            (2, 4, 1),
            (50, 150, 25),
            (10, 30, 2),
            (10, 30, 10),
            (100, 300, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = HmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
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
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
