use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use crate::utilities::math_functions::atan_fast;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MamaOutput {
    pub mama_values: Vec<f64>,
    pub fama_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MamaParams {
    pub fast_limit: Option<f64>,
    pub slow_limit: Option<f64>,
}

impl Default for MamaParams {
    fn default() -> Self {
        Self {
            fast_limit: Some(0.5),
            slow_limit: Some(0.05),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MamaInput<'a> {
    pub data: MamaData<'a>,
    pub params: MamaParams,
}

impl<'a> AsRef<[f64]> for MamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MamaData::Slice(slice) => slice,
            MamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> MamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MamaParams) -> Self {
        Self {
            data: MamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MamaParams) -> Self {
        Self {
            data: MamaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MamaParams::default())
    }
    #[inline]
    pub fn get_fast_limit(&self) -> f64 {
        self.params.fast_limit.unwrap_or(0.5)
    }
    #[inline]
    pub fn get_slow_limit(&self) -> f64 {
        self.params.slow_limit.unwrap_or(0.05)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MamaBuilder {
    fast_limit: Option<f64>,
    slow_limit: Option<f64>,
    kernel: Kernel,
}

impl Default for MamaBuilder {
    fn default() -> Self {
        Self {
            fast_limit: None,
            slow_limit: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MamaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast_limit(mut self, n: f64) -> Self {
        self.fast_limit = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_limit(mut self, x: f64) -> Self {
        self.slow_limit = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MamaOutput, MamaError> {
        let p = MamaParams {
            fast_limit: self.fast_limit,
            slow_limit: self.slow_limit,
        };
        let i = MamaInput::from_candles(c, "close", p);
        mama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MamaOutput, MamaError> {
        let p = MamaParams {
            fast_limit: self.fast_limit,
            slow_limit: self.slow_limit,
        };
        let i = MamaInput::from_slice(d, p);
        mama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MamaStream, MamaError> {
        let p = MamaParams {
            fast_limit: self.fast_limit,
            slow_limit: self.slow_limit,
        };
        MamaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MamaError {
    #[error("mama: empty input data")]
    EmptyInputData,
    #[error("mama: all values are NaN")]
    AllValuesNaN,
    #[error("mama: not enough valid data: needed {needed}, valid {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mama: Not enough data: needed at least {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },
    #[error("mama: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("mama: invalid range expansion start={start} end={end} step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("mama: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("mama: Invalid fast limit: {fast_limit}")]
    InvalidFastLimit { fast_limit: f64 },
    #[error("mama: Invalid slow limit: {slow_limit}")]
    InvalidSlowLimit { slow_limit: f64 },
}

#[inline]
pub fn mama(input: &MamaInput) -> Result<MamaOutput, MamaError> {
    mama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn mama_prepare<'a>(
    input: &'a MamaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], f64, f64, Kernel), MamaError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(MamaError::EmptyInputData);
    }
    if len < 10 {
        return Err(MamaError::NotEnoughData {
            needed: 10,
            found: len,
        });
    }

    let fast_limit = input.get_fast_limit();
    let slow_limit = input.get_slow_limit();
    if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
        return Err(MamaError::InvalidFastLimit { fast_limit });
    }
    if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
        return Err(MamaError::InvalidSlowLimit { slow_limit });
    }

    let chosen = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Auto => {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                Kernel::Avx2
            } else {
                Kernel::Scalar
            }
        }
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((data, fast_limit, slow_limit, chosen))
}

pub fn mama_with_kernel(input: &MamaInput, kernel: Kernel) -> Result<MamaOutput, MamaError> {
    let (data, fast_limit, slow_limit, chosen) = mama_prepare(input, kernel)?;
    let len = data.len();
    const WARM: usize = 10;

    let mut mama_values = alloc_with_nan_prefix(len, WARM);
    let mut fama_values = alloc_with_nan_prefix(len, WARM);

    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
                mama_simd128_inplace(
                    data,
                    fast_limit,
                    slow_limit,
                    &mut mama_values,
                    &mut fama_values,
                );

                for v in &mut mama_values[..WARM] {
                    *v = f64::NAN;
                }
                for v in &mut fama_values[..WARM] {
                    *v = f64::NAN;
                }
                return Ok(MamaOutput {
                    mama_values,
                    fama_values,
                });
            }
        }

        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                mama_scalar_inplace(
                    data,
                    fast_limit,
                    slow_limit,
                    &mut mama_values,
                    &mut fama_values,
                );
            }

            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mama_avx2_inplace(
                    data,
                    fast_limit,
                    slow_limit,
                    &mut mama_values,
                    &mut fama_values,
                );
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mama_scalar_inplace(
                    data,
                    fast_limit,
                    slow_limit,
                    &mut mama_values,
                    &mut fama_values,
                );
            }

            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mama_avx512_inplace(
                    data,
                    fast_limit,
                    slow_limit,
                    &mut mama_values,
                    &mut fama_values,
                );
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mama_scalar_inplace(
                    data,
                    fast_limit,
                    slow_limit,
                    &mut mama_values,
                    &mut fama_values,
                );
            }

            _ => unreachable!("unsupported kernel variant"),
        }
    }

    for v in &mut mama_values[..WARM] {
        *v = f64::NAN;
    }
    for v in &mut fama_values[..WARM] {
        *v = f64::NAN;
    }

    Ok(MamaOutput {
        mama_values,
        fama_values,
    })
}

pub fn mama_compute_into(
    input: &MamaInput,
    kernel: Kernel,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<(), MamaError> {
    let (data, fast_limit, slow_limit, chosen) = mama_prepare(input, kernel)?;

    if out_mama.len() != data.len() || out_fama.len() != data.len() {
        return Err(MamaError::OutputLengthMismatch {
            expected: data.len(),
            got: out_mama.len().min(out_fama.len()),
        });
    }

    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
                mama_simd128_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
                return Ok(());
            }
        }

        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                mama_scalar_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
            }

            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
            }

            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mama_avx512_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
            }

            _ => unreachable!("unsupported kernel variant"),
        }
    }

    Ok(())
}

#[inline]
pub fn mama_into(
    input: &MamaInput,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<(), MamaError> {
    let data = input.as_ref();
    if out_mama.len() != data.len() || out_fama.len() != data.len() {
        return Err(MamaError::OutputLengthMismatch {
            expected: data.len(),
            got: out_mama.len().min(out_fama.len()),
        });
    }

    mama_compute_into(input, Kernel::Auto, out_mama, out_fama)?;

    const WARM: usize = 10;
    let warm = WARM.min(data.len());
    for v in &mut out_mama[..warm] {
        *v = f64::NAN;
    }
    for v in &mut out_fama[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline]
pub fn mama_into_slice(
    dst_mama: &mut [f64],
    dst_fama: &mut [f64],
    input: &MamaInput,
    kern: Kernel,
) -> Result<(), MamaError> {
    let (data, _fast, _slow, _chosen) = mama_prepare(input, kern)?;
    if dst_mama.len() != data.len() || dst_fama.len() != data.len() {
        return Err(MamaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_mama.len().min(dst_fama.len()),
        });
    }
    mama_compute_into(input, kern, dst_mama, dst_fama)?;

    const WARM: usize = 10;
    let warm = WARM.min(data.len());
    for v in &mut dst_mama[..warm] {
        *v = f64::NAN;
    }
    for v in &mut dst_fama[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
pub fn mama_scalar(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<(), MamaError> {
    mama_scalar_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mama_avx2(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<(), MamaError> {
    mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
unsafe fn hilbert4_avx512(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    let v_x = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, x6, x4, x2, x0);

    const H3: f64 = -0.096_2;
    const H2: f64 = -0.576_9;
    const H1: f64 = 0.576_9;
    const H0: f64 = 0.096_2;
    let v_h = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, H3, H2, H1, H0);

    let v_mul = _mm512_mul_pd(v_x, v_h);
    _mm512_reduce_add_pd(v_mul)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
pub unsafe fn mama_avx512_inplace(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());

    const LEN: usize = 8;
    const MASK: usize = LEN - 1;

    #[repr(align(64))]
    struct A([f64; LEN]);
    let first = data[0];
    let mut smooth = A([first; LEN]).0;
    let mut detrender = A([first; LEN]).0;
    let mut i1_buf = A([first; LEN]).0;
    let mut q1_buf = A([first; LEN]).0;

    const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

    let (mut idx, mut prev_mesa, mut prev_phase) = (0usize, 0.0, 0.0);
    let (mut prev_mama, mut prev_fama) = (first, first);
    let (mut prev_i2, mut prev_q2) = (0.0, 0.0);
    let (mut prev_re, mut prev_im) = (0.0, 0.0);

    #[inline(always)]
    fn lag(buf: &[f64; LEN], p: usize, k: usize) -> f64 {
        unsafe { *buf.get_unchecked((p.wrapping_sub(k)) & MASK) }
    }

    for (i, &price) in data.iter().enumerate() {
        let s1 = if i >= 1 { data[i - 1] } else { price };
        let s2 = if i >= 2 { data[i - 2] } else { price };
        let s3 = if i >= 3 { data[i - 3] } else { price };
        let smooth_val =
            0.1 * (4.0_f64.mul_add(price, 3.0_f64.mul_add(s1, 2.0_f64.mul_add(s2, s3))));
        smooth[idx] = smooth_val;

        let amp = 0.075_f64.mul_add(prev_mesa, 0.54);
        let dt_val = amp
            * hilbert4_avx512(
                smooth[idx],
                lag(&smooth, idx, 2),
                lag(&smooth, idx, 4),
                lag(&smooth, idx, 6),
            );
        detrender[idx] = dt_val;

        let i1 = lag(&detrender, idx, 3);
        i1_buf[idx] = i1;

        let q1 = amp
            * hilbert4_avx512(
                detrender[idx],
                lag(&detrender, idx, 2),
                lag(&detrender, idx, 4),
                lag(&detrender, idx, 6),
            );
        q1_buf[idx] = q1;

        let j_i = amp
            * hilbert4_avx512(
                i1_buf[idx],
                lag(&i1_buf, idx, 2),
                lag(&i1_buf, idx, 4),
                lag(&i1_buf, idx, 6),
            );
        let j_q = amp
            * hilbert4_avx512(
                q1_buf[idx],
                lag(&q1_buf, idx, 2),
                lag(&q1_buf, idx, 4),
                lag(&q1_buf, idx, 6),
            );

        let i2 = i1 - j_q;
        let q2 = q1 + j_i;
        let old_i2 = prev_i2;
        let old_q2 = prev_q2;
        let i2s = 0.2_f64.mul_add(i2, 0.8 * old_i2);
        let q2s = 0.2_f64.mul_add(q2, 0.8 * old_q2);
        prev_i2 = i2s;
        prev_q2 = q2s;

        let re = 0.2_f64.mul_add(i2s * old_i2 + q2s * old_q2, 0.8 * prev_re);
        let im = 0.2_f64.mul_add(i2s * old_q2 - q2s * old_i2, 0.8 * prev_im);
        prev_re = re;
        prev_im = im;

        let mut mesa = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / atan_fast(im / re)
        } else {
            prev_mesa
        };
        mesa = mesa
            .min(1.5 * prev_mesa)
            .max(0.67 * prev_mesa)
            .max(6.0)
            .min(50.0);
        mesa = 0.2_f64.mul_add(mesa, 0.8 * prev_mesa);
        prev_mesa = mesa;

        let phase = if i1 != 0.0 {
            atan_fast(q1 / i1) * DEG_PER_RAD
        } else {
            prev_phase
        };
        let mut dp = prev_phase - phase;
        if dp < 1.0 {
            dp = 1.0;
        }
        prev_phase = phase;

        let mut alpha = fast_limit / dp;
        alpha = alpha.clamp(slow_limit, fast_limit);

        let cur_mama = alpha.mul_add(price, (1.0 - alpha) * prev_mama);
        let cur_fama = (0.5 * alpha).mul_add(cur_mama, (1.0 - 0.5 * alpha) * prev_fama);
        prev_mama = cur_mama;
        prev_fama = cur_fama;

        out_mama[i] = cur_mama;
        out_fama[i] = cur_fama;

        idx = (idx + 1) & MASK;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hilbert4_avx2(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    let v_x = _mm256_set_pd(x6, x4, x2, x0);

    const H3: f64 = -0.096_2;
    const H2: f64 = -0.576_9;
    const H1: f64 = 0.576_9;
    const H0: f64 = 0.096_2;
    let v_h = _mm256_set_pd(H3, H2, H1, H0);

    let v_mul = _mm256_mul_pd(v_x, v_h);
    let v_sum = _mm256_hadd_pd(v_mul, v_mul);

    let v_fold = _mm256_permute2f128_pd(v_sum, v_sum, 0x1);
    let v_res = _mm256_add_pd(v_sum, v_fold);
    _mm256_cvtsd_f64(v_res)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn mama_avx2_inplace(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());

    const RING_LEN: usize = 8;
    const MASK: usize = RING_LEN - 1;

    const W0: f64 = 4.0;
    const W1: f64 = 3.0;
    const W2: f64 = 2.0;
    const W3: f64 = 1.0;

    const H0: f64 = 0.096_2;
    const H1: f64 = 0.576_9;
    const H2: f64 = -0.576_9;
    const H3: f64 = -0.096_2;

    const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

    let first = data[0];
    let mut smooth = [first; RING_LEN];
    let mut detrender = [first; RING_LEN];
    let mut i1_buf = [first; RING_LEN];
    let mut q1_buf = [first; RING_LEN];

    let mut idx = 0usize;
    let mut prev_mesa = 0.0;
    let mut prev_phase = 0.0;
    let mut prev_mama = first;
    let mut prev_fama = first;
    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;

    #[inline(always)]
    fn lag(buf: &[f64; RING_LEN], p: usize, k: usize) -> f64 {
        buf[(p.wrapping_sub(k)) & MASK]
    }

    for (i, &price) in data.iter().enumerate() {
        let s1 = if i >= 1 { data[i - 1] } else { price };
        let s2 = if i >= 2 { data[i - 2] } else { price };
        let s3 = if i >= 3 { data[i - 3] } else { price };

        let smooth_val = W0.mul_add(price, W1.mul_add(s1, W2.mul_add(s2, s3))) * 0.1;
        smooth[idx] = smooth_val;

        let amp = 0.075_f64.mul_add(prev_mesa, 0.54);

        let dt_val = amp
            * hilbert4_avx2(
                smooth[idx],
                lag(&smooth, idx, 2),
                lag(&smooth, idx, 4),
                lag(&smooth, idx, 6),
            );
        detrender[idx] = dt_val;

        let i1 = lag(&detrender, idx, 3);
        i1_buf[idx] = i1;

        let q1 = amp
            * hilbert4_avx2(
                detrender[idx],
                lag(&detrender, idx, 2),
                lag(&detrender, idx, 4),
                lag(&detrender, idx, 6),
            );
        q1_buf[idx] = q1;

        let j_i = amp
            * hilbert4_avx2(
                i1_buf[idx],
                lag(&i1_buf, idx, 2),
                lag(&i1_buf, idx, 4),
                lag(&i1_buf, idx, 6),
            );
        let j_q = amp
            * hilbert4_avx2(
                q1_buf[idx],
                lag(&q1_buf, idx, 2),
                lag(&q1_buf, idx, 4),
                lag(&q1_buf, idx, 6),
            );

        let i2 = i1 - j_q;
        let q2 = q1 + j_i;
        let old_i2 = prev_i2;
        let old_q2 = prev_q2;
        let i2s = 0.2_f64.mul_add(i2, 0.8 * old_i2);
        let q2s = 0.2_f64.mul_add(q2, 0.8 * old_q2);
        prev_i2 = i2s;
        prev_q2 = q2s;

        let re = 0.2_f64.mul_add(i2s * old_i2 + q2s * old_q2, 0.8 * prev_re);
        let im = 0.2_f64.mul_add(i2s * old_q2 - q2s * old_i2, 0.8 * prev_im);
        prev_re = re;
        prev_im = im;

        let mut mesa = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / atan_fast(im / re)
        } else {
            prev_mesa
        };

        mesa = mesa
            .min(1.5 * prev_mesa)
            .max(0.67 * prev_mesa)
            .max(6.0)
            .min(50.0);
        mesa = 0.2_f64.mul_add(mesa, 0.8 * prev_mesa);
        prev_mesa = mesa;

        let phase = if i1 != 0.0 {
            atan_fast(q1 / i1) * DEG_PER_RAD
        } else {
            prev_phase
        };
        let mut dp = prev_phase - phase;
        if dp < 1.0 {
            dp = 1.0;
        }
        prev_phase = phase;

        let mut alpha = fast_limit / dp;
        alpha = alpha.clamp(slow_limit, fast_limit);

        let cur_mama = alpha.mul_add(price, (1.0 - alpha) * prev_mama);
        let cur_fama = (0.5 * alpha).mul_add(cur_mama, (1.0 - 0.5 * alpha) * prev_fama);
        prev_mama = cur_mama;
        prev_fama = cur_fama;

        out_mama[i] = cur_mama;
        out_fama[i] = cur_fama;

        idx = (idx + 1) & MASK;
    }
}
#[inline(always)]
fn hilbert(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
    0.0962 * x0 + 0.5769 * x2 - 0.5769 * x4 - 0.0962 * x6
}

#[inline]
pub fn mama_scalar_inplace(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());
    let len = data.len();

    const RING: usize = 8;
    const MASK: usize = RING - 1;

    const H0: f64 = 0.096_2;
    const H1: f64 = 0.576_9;
    const H2: f64 = -0.576_9;
    const H3: f64 = -0.096_2;
    const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

    #[inline(always)]
    fn hilbert4(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
        H0.mul_add(x0, H1.mul_add(x2, H2.mul_add(x4, H3 * x6)))
    }

    #[inline(always)]
    fn lag<const N: usize>(buf: &[f64; N], pos: usize, k: usize) -> f64 {
        buf[(pos.wrapping_sub(k)) & (N - 1)]
    }

    let first = data[0];

    let mut smooth = [first; RING];
    let mut detrender = [first; RING];
    let mut i1_buf = [first; RING];
    let mut q1_buf = [first; RING];

    let mut idx = 0usize;
    let mut prev_mesa = 0.0;
    let mut prev_phase = 0.0;
    let mut prev_mama = first;
    let mut prev_fama = first;
    let mut prev_i2 = 0.0;
    let mut prev_q2 = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;

    for (i, &price) in data.iter().enumerate() {
        let s1 = if i >= 1 { data[i - 1] } else { price };
        let s2 = if i >= 2 { data[i - 2] } else { price };
        let s3 = if i >= 3 { data[i - 3] } else { price };
        let smooth_val =
            0.1 * (4.0_f64.mul_add(price, 3.0_f64.mul_add(s1, 2.0_f64.mul_add(s2, s3))));
        smooth[idx] = smooth_val;

        let amp = 0.075_f64.mul_add(prev_mesa, 0.54);

        let dt = amp
            * hilbert4(
                smooth[idx],
                lag(&smooth, idx, 2),
                lag(&smooth, idx, 4),
                lag(&smooth, idx, 6),
            );
        detrender[idx] = dt;

        let i1 = lag(&detrender, idx, 3);
        i1_buf[idx] = i1;
        let q1 = amp
            * hilbert4(
                detrender[idx],
                lag(&detrender, idx, 2),
                lag(&detrender, idx, 4),
                lag(&detrender, idx, 6),
            );
        q1_buf[idx] = q1;

        let j_i = amp
            * hilbert4(
                i1_buf[idx],
                lag(&i1_buf, idx, 2),
                lag(&i1_buf, idx, 4),
                lag(&i1_buf, idx, 6),
            );
        let j_q = amp
            * hilbert4(
                q1_buf[idx],
                lag(&q1_buf, idx, 2),
                lag(&q1_buf, idx, 4),
                lag(&q1_buf, idx, 6),
            );

        let i2 = i1 - j_q;
        let q2 = q1 + j_i;
        let i2s = 0.2_f64.mul_add(i2, 0.8 * prev_i2);
        let q2s = 0.2_f64.mul_add(q2, 0.8 * prev_q2);
        let re = 0.2_f64.mul_add(i2s * prev_i2 + q2s * prev_q2, 0.8 * prev_re);
        let im = 0.2_f64.mul_add(i2s * prev_q2 - q2s * prev_i2, 0.8 * prev_im);
        prev_i2 = i2s;
        prev_q2 = q2s;
        prev_re = re;
        prev_im = im;

        let mut mesa = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / atan_fast(im / re)
        } else {
            prev_mesa
        };
        if mesa > 1.5 * prev_mesa {
            mesa = 1.5 * prev_mesa;
        }
        if mesa < 0.67 * prev_mesa {
            mesa = 0.67 * prev_mesa;
        }
        if mesa < 6.0 {
            mesa = 6.0;
        }
        if mesa > 50.0 {
            mesa = 50.0;
        }
        mesa = 0.2_f64.mul_add(mesa, 0.8 * prev_mesa);
        prev_mesa = mesa;

        let phase = if i1 != 0.0 {
            atan_fast(q1 / i1) * DEG_PER_RAD
        } else {
            prev_phase
        };
        let mut dphi = prev_phase - phase;
        if dphi < 1.0 {
            dphi = 1.0;
        }
        prev_phase = phase;

        let mut alpha = fast_limit / dphi;
        if alpha < slow_limit {
            alpha = slow_limit;
        }
        if alpha > fast_limit {
            alpha = fast_limit;
        }

        let mama = alpha.mul_add(price, (1.0 - alpha) * prev_mama);
        let fama = (0.5 * alpha).mul_add(mama, (1.0 - 0.5 * alpha) * prev_fama);
        prev_mama = mama;
        prev_fama = fama;

        out_mama[i] = mama;
        out_fama[i] = fama;

        idx = (idx + 1) & MASK;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn mama_simd128_inplace(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    use core::arch::wasm32::*;

    debug_assert_eq!(data.len(), out_mama.len());
    debug_assert_eq!(data.len(), out_fama.len());

    let len = data.len();

    let mut smooth_buf = [data[0]; 7];
    let mut detrender_buf = [data[0]; 7];
    let mut i1_buf = [data[0]; 7];
    let mut q1_buf = [data[0]; 7];

    let mut prev_mesa_period = 0.0;
    let mut prev_mama = data[0];
    let mut prev_fama = data[0];
    let mut prev_i2_sm = 0.0;
    let mut prev_q2_sm = 0.0;
    let mut prev_re = 0.0;
    let mut prev_im = 0.0;
    let mut prev_phase = 0.0;

    let hilbert_weights = f64x2(0.0962, 0.5769);
    let neg_hilbert_weights = f64x2(-0.5769, -0.0962);

    let smooth_weights = f64x2(4.0, 3.0);
    let smooth_weights2 = f64x2(2.0, 1.0);
    let smooth_div = f64x2_splat(0.1);

    #[inline(always)]
    fn hilbert_simd128(
        x0: f64,
        x2: f64,
        x4: f64,
        x6: f64,
        weights: v128,
        neg_weights: v128,
    ) -> f64 {
        let v1 = f64x2(x0, x2);
        let v2 = f64x2(x4, x6);

        let prod1 = f64x2_mul(v1, weights);
        let prod2 = f64x2_mul(v2, neg_weights);
        let sum = f64x2_add(prod1, prod2);

        f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum)
    }

    for i in 0..len {
        let price = data[i];

        let s1 = if i >= 1 { data[i - 1] } else { price };
        let s2 = if i >= 2 { data[i - 2] } else { price };
        let s3 = if i >= 3 { data[i - 3] } else { price };

        let v1 = f64x2(price, s1);
        let v2 = f64x2(s2, s3);
        let prod1 = f64x2_mul(v1, smooth_weights);
        let prod2 = f64x2_mul(v2, smooth_weights2);
        let sum = f64x2_add(prod1, prod2);
        let smooth_val = (f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum)) * 0.1;

        let idx = i % 7;
        smooth_buf[idx] = smooth_val;

        let x0 = smooth_buf[idx];
        let x2 = smooth_buf[(idx + 5) % 7];
        let x4 = smooth_buf[(idx + 3) % 7];
        let x6 = smooth_buf[(idx + 1) % 7];

        let mesa_mult = 0.075 * prev_mesa_period + 0.54;
        let dt_val =
            hilbert_simd128(x0, x2, x4, x6, hilbert_weights, neg_hilbert_weights) * mesa_mult;
        detrender_buf[idx] = dt_val;

        let i1_val = if i >= 3 {
            detrender_buf[(idx + 4) % 7]
        } else {
            dt_val
        };
        i1_buf[idx] = i1_val;

        let d0 = detrender_buf[idx];
        let d2 = detrender_buf[(idx + 5) % 7];
        let d4 = detrender_buf[(idx + 3) % 7];
        let d6 = detrender_buf[(idx + 1) % 7];
        let q1_val =
            hilbert_simd128(d0, d2, d4, d6, hilbert_weights, neg_hilbert_weights) * mesa_mult;
        q1_buf[idx] = q1_val;

        let j_i = {
            let i0 = i1_buf[idx];
            let i2 = i1_buf[(idx + 5) % 7];
            let i4 = i1_buf[(idx + 3) % 7];
            let i6 = i1_buf[(idx + 1) % 7];
            hilbert_simd128(i0, i2, i4, i6, hilbert_weights, neg_hilbert_weights) * mesa_mult
        };
        let j_q = {
            let q0 = q1_buf[idx];
            let q2 = q1_buf[(idx + 5) % 7];
            let q4 = q1_buf[(idx + 3) % 7];
            let q6 = q1_buf[(idx + 1) % 7];
            hilbert_simd128(q0, q2, q4, q6, hilbert_weights, neg_hilbert_weights) * mesa_mult
        };

        let i2 = i1_val - j_q;
        let q2 = q1_val + j_i;
        let i2_sm = 0.2 * i2 + 0.8 * prev_i2_sm;
        let q2_sm = 0.2 * q2 + 0.8 * prev_q2_sm;
        let re = 0.2 * (i2_sm * prev_i2_sm + q2_sm * prev_q2_sm) + 0.8 * prev_re;
        let im = 0.2 * (i2_sm * prev_q2_sm - q2_sm * prev_i2_sm) + 0.8 * prev_im;
        prev_i2_sm = i2_sm;
        prev_q2_sm = q2_sm;
        prev_re = re;
        prev_im = im;

        let mut mesa_period = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / atan_fast(im / re)
        } else {
            prev_mesa_period
        };

        if mesa_period > 1.5 * prev_mesa_period {
            mesa_period = 1.5 * prev_mesa_period;
        }
        if mesa_period < 0.67 * prev_mesa_period {
            mesa_period = 0.67 * prev_mesa_period;
        }
        if mesa_period < 6.0 {
            mesa_period = 6.0;
        }
        if mesa_period > 50.0 {
            mesa_period = 50.0;
        }

        let phase = if i1_val != 0.0 {
            atan_fast(q1_val / i1_val) * 180.0 / std::f64::consts::PI
        } else {
            prev_phase
        };

        let mut dp = prev_phase - phase;
        if dp < 1.0 {
            dp = 1.0;
        }
        prev_phase = phase;

        let mut alpha = fast_limit / dp;
        alpha = alpha.clamp(slow_limit, fast_limit);

        prev_mesa_period = mesa_period;

        let mama_val = alpha * price + (1.0 - alpha) * prev_mama;
        let fama_val = 0.5 * alpha * mama_val + (1.0 - 0.5 * alpha) * prev_fama;

        out_mama[i] = mama_val;
        out_fama[i] = fama_val;

        prev_mama = mama_val;
        prev_fama = fama_val;
    }
}

#[derive(Debug, Clone)]
pub struct MamaStream {
    fast_limit: f64,
    slow_limit: f64,

    smooth: [f64; 8],
    detrender: [f64; 8],
    i1_buf: [f64; 8],
    q1_buf: [f64; 8],
    idx: usize,

    prev_mesa: f64,
    prev_phase: f64,
    prev_mama: f64,
    prev_fama: f64,
    prev_i2: f64,
    prev_q2: f64,
    prev_re: f64,
    prev_im: f64,

    last1: f64,
    last2: f64,
    last3: f64,

    seeded: bool,
    seen: usize,
}

impl MamaStream {
    #[inline]
    pub fn try_new(params: MamaParams) -> Result<Self, MamaError> {
        let fast_limit = params.fast_limit.unwrap_or(0.5);
        let slow_limit = params.slow_limit.unwrap_or(0.05);
        if fast_limit <= 0.0 || !fast_limit.is_finite() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || !slow_limit.is_finite() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }

        Ok(Self {
            fast_limit,
            slow_limit,
            smooth: [f64::NAN; 8],
            detrender: [f64::NAN; 8],
            i1_buf: [f64::NAN; 8],
            q1_buf: [f64::NAN; 8],
            idx: 0,

            prev_mesa: 0.0,
            prev_phase: 0.0,
            prev_mama: f64::NAN,
            prev_fama: f64::NAN,
            prev_i2: 0.0,
            prev_q2: 0.0,
            prev_re: 0.0,
            prev_im: 0.0,

            last1: f64::NAN,
            last2: f64::NAN,
            last3: f64::NAN,

            seeded: false,
            seen: 0,
        })
    }

    #[inline]
    pub fn update(&mut self, price: f64) -> Option<(f64, f64)> {
        const RING: usize = 8;
        const MASK: usize = RING - 1;
        const H0: f64 = 0.096_2;
        const H1: f64 = 0.576_9;
        const H2: f64 = -0.576_9;
        const H3: f64 = -0.096_2;
        const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

        #[inline(always)]
        fn hilbert4(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
            H0.mul_add(x0, H1.mul_add(x2, H2.mul_add(x4, H3 * x6)))
        }
        #[inline(always)]
        fn lag<const N: usize>(buf: &[f64; N], pos: usize, k: usize) -> f64 {
            buf[(pos.wrapping_sub(k)) & (N - 1)]
        }

        if !self.seeded {
            self.smooth = [price; RING];
            self.detrender = [price; RING];
            self.i1_buf = [price; RING];
            self.q1_buf = [price; RING];
            self.idx = 0;

            self.prev_mesa = 0.0;
            self.prev_phase = 0.0;
            self.prev_mama = price;
            self.prev_fama = price;
            self.prev_i2 = 0.0;
            self.prev_q2 = 0.0;
            self.prev_re = 0.0;
            self.prev_im = 0.0;

            self.last1 = price;
            self.last2 = price;
            self.last3 = price;

            self.seeded = true;

            let _ = self.process_one(price, hilbert4, lag::<RING>, DEG_PER_RAD);

            return None;
        }

        let (mama, fama) = self.process_one(price, hilbert4, lag::<RING>, DEG_PER_RAD);

        self.seen += 1;
        if self.seen < 10 {
            return None;
        }
        Some((mama, fama))
    }

    #[inline(always)]
    fn process_one(
        &mut self,
        price: f64,
        hilbert4: impl Fn(f64, f64, f64, f64) -> f64,
        lag: impl Fn(&[f64; 8], usize, usize) -> f64,
        deg_per_rad: f64,
    ) -> (f64, f64) {
        const MASK: usize = 7;
        let i = self.idx;

        let s1 = if self.seen >= 1 { self.last1 } else { price };
        let s2 = if self.seen >= 2 { self.last2 } else { price };
        let s3 = if self.seen >= 3 { self.last3 } else { price };
        let smooth_val =
            0.1 * (4.0_f64.mul_add(price, 3.0_f64.mul_add(s1, 2.0_f64.mul_add(s2, s3))));
        self.smooth[i] = smooth_val;

        let amp = 0.075_f64.mul_add(self.prev_mesa, 0.54);

        let dt = amp
            * hilbert4(
                self.smooth[i],
                lag(&self.smooth, i, 2),
                lag(&self.smooth, i, 4),
                lag(&self.smooth, i, 6),
            );
        self.detrender[i] = dt;

        let i1 = lag(&self.detrender, i, 3);
        self.i1_buf[i] = i1;

        let q1 = amp
            * hilbert4(
                self.detrender[i],
                lag(&self.detrender, i, 2),
                lag(&self.detrender, i, 4),
                lag(&self.detrender, i, 6),
            );
        self.q1_buf[i] = q1;

        let j_i = amp
            * hilbert4(
                self.i1_buf[i],
                lag(&self.i1_buf, i, 2),
                lag(&self.i1_buf, i, 4),
                lag(&self.i1_buf, i, 6),
            );
        let j_q = amp
            * hilbert4(
                self.q1_buf[i],
                lag(&self.q1_buf, i, 2),
                lag(&self.q1_buf, i, 4),
                lag(&self.q1_buf, i, 6),
            );

        let i2 = i1 - j_q;
        let q2 = q1 + j_i;

        let old_i2 = self.prev_i2;
        let old_q2 = self.prev_q2;

        let i2s = 0.2_f64.mul_add(i2, 0.8 * old_i2);
        let q2s = 0.2_f64.mul_add(q2, 0.8 * old_q2);
        self.prev_i2 = i2s;
        self.prev_q2 = q2s;

        let re = 0.2_f64.mul_add(i2s * old_i2 + q2s * old_q2, 0.8 * self.prev_re);
        let im = 0.2_f64.mul_add(i2s * old_q2 - q2s * old_i2, 0.8 * self.prev_im);
        self.prev_re = re;
        self.prev_im = im;

        let mut mesa = if re != 0.0 && im != 0.0 {
            2.0 * std::f64::consts::PI / atan_fast(im / re)
        } else {
            self.prev_mesa
        };

        mesa = mesa
            .min(1.5 * self.prev_mesa)
            .max(0.67 * self.prev_mesa)
            .max(6.0)
            .min(50.0);
        mesa = 0.2_f64.mul_add(mesa, 0.8 * self.prev_mesa);
        self.prev_mesa = mesa;

        let phase = if i1 != 0.0 {
            atan_fast(q1 / i1) * deg_per_rad
        } else {
            self.prev_phase
        };
        let mut dphi = self.prev_phase - phase;
        if dphi < 1.0 {
            dphi = 1.0;
        }
        self.prev_phase = phase;

        let mut alpha = self.fast_limit / dphi;
        if alpha < self.slow_limit {
            alpha = self.slow_limit;
        }
        if alpha > self.fast_limit {
            alpha = self.fast_limit;
        }

        let one_minus_alpha = 1.0 - alpha;
        let mama = alpha.mul_add(price, one_minus_alpha * self.prev_mama);

        let half_alpha = 0.5 * alpha;
        let fama = half_alpha.mul_add(mama, (1.0 - half_alpha) * self.prev_fama);

        self.prev_mama = mama;
        self.prev_fama = fama;

        self.idx = (self.idx + 1) & MASK;
        self.last3 = self.last2;
        self.last2 = self.last1;
        self.last1 = price;

        (mama, fama)
    }
}

#[derive(Clone, Debug)]
pub struct MamaBatchRange {
    pub fast_limit: (f64, f64, f64),
    pub slow_limit: (f64, f64, f64),
}

impl Default for MamaBatchRange {
    fn default() -> Self {
        Self {
            fast_limit: (0.5, 0.749, 0.001),
            slow_limit: (0.05, 0.05, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MamaBatchBuilder {
    range: MamaBatchRange,
    kernel: Kernel,
}

impl MamaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn fast_limit_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fast_limit = (start, end, step);
        self
    }
    #[inline]
    pub fn fast_limit_static(mut self, x: f64) -> Self {
        self.range.fast_limit = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn slow_limit_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.slow_limit = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_limit_static(mut self, x: f64) -> Self {
        self.range.slow_limit = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<MamaBatchOutput, MamaError> {
        mama_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MamaBatchOutput, MamaError> {
        MamaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MamaBatchOutput, MamaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MamaBatchOutput, MamaError> {
        MamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MamaBatchOutput {
    pub mama_values: Vec<f64>,
    pub fama_values: Vec<f64>,
    pub combos: Vec<MamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MamaBatchOutput {
    pub fn row_for_params(&self, p: &MamaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.fast_limit.unwrap_or(0.5) - p.fast_limit.unwrap_or(0.5)).abs() < 1e-12
                && (c.slow_limit.unwrap_or(0.05) - p.slow_limit.unwrap_or(0.05)).abs() < 1e-12
        })
    }
    pub fn mama_for(&self, p: &MamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.mama_values[start..start + self.cols]
        })
    }
    pub fn fama_for(&self, p: &MamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.fama_values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid(r: &MamaBatchRange) -> Result<Vec<MamaParams>, MamaError> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, MamaError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut step_signed = step;
        if end < start && step_signed > 0.0 {
            step_signed = -step_signed;
        } else if end > start && step_signed < 0.0 {
            step_signed = -step_signed;
        }

        let mut v = Vec::new();
        let eps = 1e-12_f64;
        let mut x = start;
        if step_signed > 0.0 {
            while x <= end + eps {
                v.push(x);
                x += step_signed;
            }
        } else {
            while x >= end - eps {
                v.push(x);
                x += step_signed;
            }
        }

        if v.is_empty() {
            return Err(MamaError::InvalidRange { start, end, step });
        }
        Ok(v)
    }

    let fast_limits = axis_f64(r.fast_limit)?;
    let slow_limits = axis_f64(r.slow_limit)?;

    let cap = fast_limits
        .len()
        .checked_mul(slow_limits.len())
        .ok_or(MamaError::InvalidRange {
            start: r.fast_limit.0,
            end: r.fast_limit.1,
            step: r.fast_limit.2,
        })?;

    let mut out = Vec::with_capacity(cap);
    for &f in &fast_limits {
        for &s in &slow_limits {
            out.push(MamaParams {
                fast_limit: Some(f),
                slow_limit: Some(s),
            });
        }
    }
    Ok(out)
}

pub fn mama_batch_with_kernel(
    data: &[f64],
    sweep: &MamaBatchRange,
    k: Kernel,
) -> Result<MamaBatchOutput, MamaError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(MamaError::InvalidKernelForBatch(other)),
    };

    let simd = Kernel::Scalar;
    mama_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn mama_batch_slice(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
) -> Result<MamaBatchOutput, MamaError> {
    mama_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mama_batch_par_slice(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
) -> Result<MamaBatchOutput, MamaError> {
    mama_batch_inner(data, sweep, kern, true)
}

fn mama_batch_inner(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MamaBatchOutput, MamaError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(MamaError::InvalidRange {
            start: sweep.fast_limit.0,
            end: sweep.fast_limit.1,
            step: sweep.fast_limit.2,
        });
    }
    if data.len() < 10 {
        return Err(MamaError::NotEnoughData {
            needed: 10,
            found: data.len(),
        });
    }

    for combo in &combos {
        let fast_limit = combo.fast_limit.unwrap_or(0.5);
        let slow_limit = combo.slow_limit.unwrap_or(0.05);

        if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }
    }

    let rows = combos.len();
    let cols = data.len();

    let mut raw_mama = make_uninit_matrix(rows, cols);
    let mut raw_fama = make_uninit_matrix(rows, cols);

    let warm_prefixes = vec![10; rows];
    unsafe {
        init_matrix_prefixes(&mut raw_mama, cols, &warm_prefixes);
        init_matrix_prefixes(&mut raw_fama, cols, &warm_prefixes);
    }

    let delta_phase: Vec<f64> = {
        const RING: usize = 8;
        const MASK: usize = RING - 1;
        const H0: f64 = 0.096_2;
        const H1: f64 = 0.576_9;
        const H2: f64 = -0.576_9;
        const H3: f64 = -0.096_2;
        const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;

        #[inline(always)]
        fn hilbert4(x0: f64, x2: f64, x4: f64, x6: f64) -> f64 {
            H0.mul_add(x0, H1.mul_add(x2, H2.mul_add(x4, H3 * x6)))
        }
        #[inline(always)]
        fn lag<const N: usize>(buf: &[f64; N], pos: usize, k: usize) -> f64 {
            buf[(pos.wrapping_sub(k)) & (N - 1)]
        }

        let mut out = vec![1.0; cols];
        if cols == 0 {
            out
        } else {
            let first = data[0];
            let mut smooth = [first; RING];
            let mut detrender = [first; RING];
            let mut i1_buf = [first; RING];
            let mut q1_buf = [first; RING];

            let mut idx = 0usize;
            let mut prev_mesa = 0.0;
            let mut prev_phase = 0.0;
            let mut prev_i2 = 0.0;
            let mut prev_q2 = 0.0;
            let mut prev_re = 0.0;
            let mut prev_im = 0.0;

            for (i, &price) in data.iter().enumerate() {
                let s1 = if i >= 1 { data[i - 1] } else { price };
                let s2 = if i >= 2 { data[i - 2] } else { price };
                let s3 = if i >= 3 { data[i - 3] } else { price };
                let smooth_val =
                    0.1 * (4.0_f64.mul_add(price, 3.0_f64.mul_add(s1, 2.0_f64.mul_add(s2, s3))));
                smooth[idx] = smooth_val;

                let amp = 0.075_f64.mul_add(prev_mesa, 0.54);
                let dt = amp
                    * hilbert4(
                        smooth[idx],
                        lag(&smooth, idx, 2),
                        lag(&smooth, idx, 4),
                        lag(&smooth, idx, 6),
                    );
                detrender[idx] = dt;

                let i1 = lag(&detrender, idx, 3);
                i1_buf[idx] = i1;
                let q1 = amp
                    * hilbert4(
                        detrender[idx],
                        lag(&detrender, idx, 2),
                        lag(&detrender, idx, 4),
                        lag(&detrender, idx, 6),
                    );
                q1_buf[idx] = q1;

                let j_i = amp
                    * hilbert4(
                        i1_buf[idx],
                        lag(&i1_buf, idx, 2),
                        lag(&i1_buf, idx, 4),
                        lag(&i1_buf, idx, 6),
                    );
                let j_q = amp
                    * hilbert4(
                        q1_buf[idx],
                        lag(&q1_buf, idx, 2),
                        lag(&q1_buf, idx, 4),
                        lag(&q1_buf, idx, 6),
                    );

                let i2 = i1 - j_q;
                let q2 = q1 + j_i;
                let old_i2 = prev_i2;
                let old_q2 = prev_q2;
                let i2s = 0.2_f64.mul_add(i2, 0.8 * old_i2);
                let q2s = 0.2_f64.mul_add(q2, 0.8 * old_q2);
                prev_i2 = i2s;
                prev_q2 = q2s;
                let re = 0.2_f64.mul_add(i2s * old_i2 + q2s * old_q2, 0.8 * prev_re);
                let im = 0.2_f64.mul_add(i2s * old_q2 - q2s * old_i2, 0.8 * prev_im);
                prev_re = re;
                prev_im = im;

                let mut mesa = if re != 0.0 && im != 0.0 {
                    2.0 * std::f64::consts::PI / atan_fast(im / re)
                } else {
                    prev_mesa
                };
                if mesa > 1.5 * prev_mesa {
                    mesa = 1.5 * prev_mesa;
                }
                if mesa < 0.67 * prev_mesa {
                    mesa = 0.67 * prev_mesa;
                }
                if mesa < 6.0 {
                    mesa = 6.0;
                }
                if mesa > 50.0 {
                    mesa = 50.0;
                }
                mesa = 0.2_f64.mul_add(mesa, 0.8 * prev_mesa);
                prev_mesa = mesa;

                let phase = if i1 != 0.0 {
                    atan_fast(q1 / i1) * DEG_PER_RAD
                } else {
                    prev_phase
                };
                let mut dphi = prev_phase - phase;
                if dphi < 1.0 {
                    dphi = 1.0;
                }
                prev_phase = phase;
                out[i] = dphi;

                idx = (idx + 1) & MASK;
            }
            out
        }
    };

    let do_row = |row: usize, dst_m: &mut [MaybeUninit<f64>], dst_f: &mut [MaybeUninit<f64>]| unsafe {
        let prm = &combos[row];
        let fast = prm.fast_limit.unwrap_or(0.5);
        let slow = prm.slow_limit.unwrap_or(0.05);

        let out_m = core::slice::from_raw_parts_mut(dst_m.as_mut_ptr() as *mut f64, dst_m.len());
        let out_f = core::slice::from_raw_parts_mut(dst_f.as_mut_ptr() as *mut f64, dst_f.len());

        let mut prev_mama = data[0];
        let mut prev_fama = data[0];
        for i in 0..cols {
            let price = data[i];
            let mut alpha = fast / delta_phase[i];
            if alpha < slow {
                alpha = slow;
            }
            if alpha > fast {
                alpha = fast;
            }

            let mama = alpha.mul_add(price, (1.0 - alpha) * prev_mama);
            let fama = (0.5 * alpha).mul_add(mama, (1.0 - 0.5 * alpha) * prev_fama);
            prev_mama = mama;
            prev_fama = fama;
            out_m[i] = mama;
            out_f[i] = fama;
        }

        for j in 0..10.min(out_m.len()) {
            out_m[j] = f64::NAN;
            out_f[j] = f64::NAN;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw_mama
                .par_chunks_mut(cols)
                .zip(raw_fama.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (m_row, f_row))| do_row(row, m_row, f_row));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (m_row, f_row)) in raw_mama
                .chunks_mut(cols)
                .zip(raw_fama.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, m_row, f_row);
            }
        }
    } else {
        for (row, (m_row, f_row)) in raw_mama
            .chunks_mut(cols)
            .zip(raw_fama.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, m_row, f_row);
        }
    }

    let mut guard_m = core::mem::ManuallyDrop::new(raw_mama);
    let mut guard_f = core::mem::ManuallyDrop::new(raw_fama);

    let mama_values = unsafe {
        Vec::from_raw_parts(
            guard_m.as_mut_ptr() as *mut f64,
            guard_m.len(),
            guard_m.capacity(),
        )
    };
    let fama_values = unsafe {
        Vec::from_raw_parts(
            guard_f.as_mut_ptr() as *mut f64,
            guard_f.len(),
            guard_f.capacity(),
        )
    };

    Ok(MamaBatchOutput {
        mama_values,
        fama_values,
        combos,
        rows,
        cols,
    })
}

pub fn mama_batch_inner_into(
    data: &[f64],
    sweep: &MamaBatchRange,
    kern: Kernel,
    parallel: bool,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) -> Result<Vec<MamaParams>, MamaError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(MamaError::InvalidRange {
            start: sweep.fast_limit.0,
            end: sweep.fast_limit.1,
            step: sweep.fast_limit.2,
        });
    }
    if data.len() < 10 {
        return Err(MamaError::NotEnoughData {
            needed: 10,
            found: data.len(),
        });
    }

    for combo in &combos {
        let fast_limit = combo.fast_limit.unwrap_or(0.5);
        let slow_limit = combo.slow_limit.unwrap_or(0.05);

        if fast_limit <= 0.0 || fast_limit.is_nan() || fast_limit.is_infinite() {
            return Err(MamaError::InvalidFastLimit { fast_limit });
        }
        if slow_limit <= 0.0 || slow_limit.is_nan() || slow_limit.is_infinite() {
            return Err(MamaError::InvalidSlowLimit { slow_limit });
        }
    }

    let rows = combos.len();
    let cols = data.len();

    let expected = rows.checked_mul(cols).ok_or(MamaError::InvalidRange {
        start: sweep.fast_limit.0,
        end: sweep.fast_limit.1,
        step: sweep.fast_limit.2,
    })?;
    if out_mama.len() != expected || out_fama.len() != expected {
        return Err(MamaError::OutputLengthMismatch {
            expected,
            got: out_mama.len().min(out_fama.len()),
        });
    }

    let out_mama_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out_mama.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_mama.len(),
        )
    };
    let out_fama_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out_fama.as_mut_ptr() as *mut MaybeUninit<f64>,
            out_fama.len(),
        )
    };

    let warm_prefixes = vec![10; rows];
    unsafe {
        init_matrix_prefixes(out_mama_uninit, cols, &warm_prefixes);
        init_matrix_prefixes(out_fama_uninit, cols, &warm_prefixes);
    }

    let do_row = |row: usize, dst_m: &mut [MaybeUninit<f64>], dst_f: &mut [MaybeUninit<f64>]| unsafe {
        let prm = &combos[row];
        let fast = prm.fast_limit.unwrap_or(0.5);
        let slow = prm.slow_limit.unwrap_or(0.05);

        let out_m = core::slice::from_raw_parts_mut(dst_m.as_mut_ptr() as *mut f64, dst_m.len());
        let out_f = core::slice::from_raw_parts_mut(dst_f.as_mut_ptr() as *mut f64, dst_f.len());

        match kern {
            Kernel::Scalar => mama_row_scalar(data, fast, slow, out_m, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mama_row_avx2(data, fast, slow, out_m, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mama_row_avx512(data, fast, slow, out_m, out_f),
            _ => unreachable!(),
        }

        for j in 0..10.min(out_m.len()) {
            out_m[j] = f64::NAN;
            out_f[j] = f64::NAN;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mama_uninit
                .par_chunks_mut(cols)
                .zip(out_fama_uninit.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (m_row, f_row))| do_row(row, m_row, f_row));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (m_row, f_row)) in out_mama_uninit
                .chunks_mut(cols)
                .zip(out_fama_uninit.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, m_row, f_row);
            }
        }
    } else {
        for (row, (m_row, f_row)) in out_mama_uninit
            .chunks_mut(cols)
            .zip(out_fama_uninit.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, m_row, f_row);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn mama_row_scalar(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_scalar_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_row_avx2(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_avx2_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mama_row_avx512(
    data: &[f64],
    fast_limit: f64,
    slow_limit: f64,
    out_mama: &mut [f64],
    out_fama: &mut [f64],
) {
    mama_avx512_inplace(data, fast_limit, slow_limit, out_mama, out_fama);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use paste::paste;
    use proptest::prelude::*;

    fn check_mama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = MamaParams {
            fast_limit: None,
            slow_limit: None,
        };
        let input = MamaInput::from_candles(&candles, "close", default_params);
        let output = mama_with_kernel(&input, kernel)?;
        assert_eq!(output.mama_values.len(), candles.close.len());
        assert_eq!(output.fama_values.len(), candles.close.len());
        Ok(())
    }

    fn check_mama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MamaInput::from_candles(&candles, "close", MamaParams::default());
        let result = mama_with_kernel(&input, kernel)?;
        assert_eq!(result.mama_values.len(), candles.close.len());
        assert_eq!(result.fama_values.len(), candles.close.len());
        Ok(())
    }

    fn check_mama_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = MamaInput::with_default_candles(&candles);
        match input.data {
            MamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MamaData::Candles"),
        }
        let output = mama_with_kernel(&input, kernel)?;
        assert_eq!(output.mama_values.len(), candles.close.len());
        assert_eq!(output.fama_values.len(), candles.close.len());
        Ok(())
    }

    fn check_mama_with_insufficient_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [100.0; 9];
        let params = MamaParams::default();
        let input = MamaInput::from_slice(&input_data, params);
        let res = mama_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_mama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0; 10];
        let params = MamaParams::default();
        let input = MamaInput::from_slice(&input_data, params);
        let result = mama_with_kernel(&input, kernel)?;
        assert_eq!(result.mama_values.len(), input_data.len());
        assert_eq!(result.fama_values.len(), input_data.len());
        Ok(())
    }

    fn check_mama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = MamaParams::default();
        let first_input = MamaInput::from_candles(&candles, "close", first_params);
        let first_result = mama_with_kernel(&first_input, kernel)?;
        let second_params = MamaParams {
            fast_limit: Some(0.7),
            slow_limit: Some(0.1),
        };
        let second_input = MamaInput::from_slice(&first_result.mama_values, second_params);
        let second_result = mama_with_kernel(&second_input, kernel)?;
        assert_eq!(
            second_result.mama_values.len(),
            first_result.mama_values.len()
        );
        assert_eq!(
            second_result.fama_values.len(),
            first_result.mama_values.len()
        );
        Ok(())
    }

    fn check_mama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = MamaParams::default();
        let input = MamaInput::from_candles(&candles, "close", params);
        let result = mama_with_kernel(&input, kernel)?;
        for (i, &val) in result.mama_values.iter().enumerate() {
            if i > 20 {
                assert!(val.is_finite());
            }
        }
        for (i, &val) in result.fama_values.iter().enumerate() {
            if i > 20 {
                assert!(val.is_finite());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_mama_tests {
        ($($test_fn:ident),*) => {
            paste! {
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

    #[cfg(debug_assertions)]
    fn check_mama_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_cases = vec![
            MamaParams::default(),
            MamaParams {
                fast_limit: Some(0.3),
                slow_limit: Some(0.03),
            },
            MamaParams {
                fast_limit: Some(0.4),
                slow_limit: Some(0.04),
            },
            MamaParams {
                fast_limit: Some(0.5),
                slow_limit: Some(0.05),
            },
            MamaParams {
                fast_limit: Some(0.6),
                slow_limit: Some(0.06),
            },
            MamaParams {
                fast_limit: Some(0.7),
                slow_limit: Some(0.07),
            },
            MamaParams {
                fast_limit: Some(0.8),
                slow_limit: Some(0.01),
            },
            MamaParams {
                fast_limit: Some(0.2),
                slow_limit: Some(0.1),
            },
            MamaParams {
                fast_limit: Some(0.9),
                slow_limit: Some(0.02),
            },
        ];

        for params in test_cases {
            let input = MamaInput::from_candles(&candles, "close", params.clone());
            let output = mama_with_kernel(&input, kernel)?;

            for (i, &val) in output.mama_values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in mama_values with params fast_limit={:?}, slow_limit={:?}",
                        test_name, val, bits, i, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in mama_values with params fast_limit={:?}, slow_limit={:?}",
                        test_name, val, bits, i, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in mama_values with params fast_limit={:?}, slow_limit={:?}",
                        test_name, val, bits, i, params.fast_limit, params.slow_limit
                    );
                }
            }

            for (i, &val) in output.fama_values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in fama_values with params fast_limit={:?}, slow_limit={:?}",
                        test_name, val, bits, i, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in fama_values with params fast_limit={:?}, slow_limit={:?}",
                        test_name, val, bits, i, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in fama_values with params fast_limit={:?}, slow_limit={:?}",
                        test_name, val, bits, i, params.fast_limit, params.slow_limit
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_mama_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn check_mama_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let strat = (10usize..=200).prop_flat_map(|len| {
            (
                prop::collection::vec(
                    (-1e5f64..1e5f64).prop_filter("finite", |x| x.is_finite()),
                    len,
                ),
                (0.01f64..0.99f64).prop_filter("valid fast_limit", |x| x.is_finite() && *x > 0.0),
                (0.001f64..0.5f64).prop_filter("valid slow_limit", |x| x.is_finite() && *x > 0.0),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, fast_limit, slow_limit)| {
                let slow = slow_limit.min(fast_limit * 0.9);

                let params = MamaParams {
                    fast_limit: Some(fast_limit),
                    slow_limit: Some(slow),
                };
                let input = MamaInput::from_slice(&data, params);

                let result = mama_with_kernel(&input, kernel).unwrap();
                let mama_out = &result.mama_values;
                let fama_out = &result.fama_values;

                let ref_result = mama_with_kernel(&input, Kernel::Scalar).unwrap();
                let ref_mama = &ref_result.mama_values;
                let ref_fama = &ref_result.fama_values;

                prop_assert_eq!(mama_out.len(), data.len(), "MAMA output length mismatch");
                prop_assert_eq!(fama_out.len(), data.len(), "FAMA output length mismatch");

                const WARMUP: usize = 10;
                for i in 0..data.len() {
                    if i < WARMUP {
                        prop_assert!(
                            mama_out[i].is_nan(),
                            "MAMA should have NaN warmup at index {}, got {}",
                            i,
                            mama_out[i]
                        );
                        prop_assert!(
                            fama_out[i].is_nan(),
                            "FAMA should have NaN warmup at index {}, got {}",
                            i,
                            fama_out[i]
                        );
                    } else {
                        prop_assert!(
                            mama_out[i].is_finite(),
                            "MAMA should output finite values at index {}, got {}",
                            i,
                            mama_out[i]
                        );
                        prop_assert!(
                            fama_out[i].is_finite(),
                            "FAMA should output finite values at index {}, got {}",
                            i,
                            fama_out[i]
                        );
                    }
                }

                let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let data_range = data_max - data_min;

                let tolerance = data_range * 0.2 + 10.0;

                for i in WARMUP..data.len() {
                    prop_assert!(
                        mama_out[i] >= data_min - tolerance && mama_out[i] <= data_max + tolerance,
                        "MAMA at index {} ({}) outside bounds [{}, {}]",
                        i,
                        mama_out[i],
                        data_min - tolerance,
                        data_max + tolerance
                    );
                    prop_assert!(
                        fama_out[i] >= data_min - tolerance && fama_out[i] <= data_max + tolerance,
                        "FAMA at index {} ({}) outside bounds [{}, {}]",
                        i,
                        fama_out[i],
                        data_min - tolerance,
                        data_max + tolerance
                    );
                }

                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) {
                    let constant_val = data[0];

                    for i in 10..data.len() {
                        prop_assert!(
                            (mama_out[i] - constant_val).abs() < 1e-6,
                            "MAMA should converge to constant value {} at index {}, got {}",
                            constant_val,
                            i,
                            mama_out[i]
                        );
                        prop_assert!(
                            (fama_out[i] - constant_val).abs() < 1e-6,
                            "FAMA should converge to constant value {} at index {}, got {}",
                            constant_val,
                            i,
                            fama_out[i]
                        );
                    }
                }

                if data.len() > 30 {
                    let mama_variance = variance(&mama_out[10..]);
                    let fama_variance = variance(&fama_out[10..]);

                    prop_assert!(
                        mama_variance >= 0.0 && mama_variance.is_finite(),
                        "MAMA variance should be finite and non-negative: {}",
                        mama_variance
                    );
                    prop_assert!(
                        fama_variance >= 0.0 && fama_variance.is_finite(),
                        "FAMA variance should be finite and non-negative: {}",
                        fama_variance
                    );

                    let data_variance = variance(&data);
                    if data_variance > 1e-6 {
                        prop_assert!(
                            mama_variance < data_variance * 100.0,
                            "MAMA variance ({}) too large relative to data variance ({})",
                            mama_variance,
                            data_variance
                        );
                        prop_assert!(
                            fama_variance < data_variance * 100.0,
                            "FAMA variance ({}) too large relative to data variance ({})",
                            fama_variance,
                            data_variance
                        );
                    }
                }

                for i in WARMUP..data.len() {
                    prop_assert!(
                        mama_out[i].is_finite(),
                        "MAMA kernel {:?} produced non-finite value at idx {}: {}",
                        kernel,
                        i,
                        mama_out[i]
                    );
                    prop_assert!(
                        fama_out[i].is_finite(),
                        "FAMA kernel {:?} produced non-finite value at idx {}: {}",
                        kernel,
                        i,
                        fama_out[i]
                    );
                }

                if data.len() > 50 && fast_limit > slow * 2.0 && variance(&data) > 1e-6 {
                    let alt_params = MamaParams {
                        fast_limit: Some(fast_limit * 0.5),
                        slow_limit: Some(slow),
                    };
                    let alt_input = MamaInput::from_slice(&data, alt_params);
                    if let Ok(alt_result) = mama_with_kernel(&alt_input, kernel) {
                        let mama_var = variance(&mama_out[20..]);
                        let alt_var = variance(&alt_result.mama_values[20..]);

                        if mama_var > 1e-6 && alt_var > 1e-6 {
                            prop_assert!(
                                (mama_var - alt_var).abs() > 1e-12,
                                "MAMA should be sensitive to fast_limit parameter"
                            );
                        }
                    }
                }

                if (fast_limit - slow).abs() < 0.01 && data.len() > 20 {
                    for i in 10..data.len() {
                        prop_assert!(
                            mama_out[i].is_finite() && fama_out[i].is_finite(),
                            "MAMA/FAMA should remain finite even with close limits at idx {}",
                            i
                        );

                        prop_assert!(
                            mama_out[i].abs() < data_max.abs() * 100.0 + 1000.0,
                            "MAMA should not diverge with close limits"
                        );
                        prop_assert!(
                            fama_out[i].abs() < data_max.abs() * 100.0 + 1000.0,
                            "FAMA should not diverge with close limits"
                        );
                    }
                }

                let is_monotonic_inc = data.windows(2).all(|w| w[1] >= w[0] - 1e-9);
                let is_monotonic_dec = data.windows(2).all(|w| w[1] <= w[0] + 1e-9);

                if (is_monotonic_inc || is_monotonic_dec) && data.len() > 20 {
                    for i in 11..data.len() {
                        if is_monotonic_inc {
                            prop_assert!(
                                mama_out[i] >= mama_out[i - 10] - tolerance * 0.1,
                                "MAMA should follow increasing trend at idx {}",
                                i
                            );
                        }
                        if is_monotonic_dec {
                            prop_assert!(
                                mama_out[i] <= mama_out[i - 10] + tolerance * 0.1,
                                "MAMA should follow decreasing trend at idx {}",
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

    fn variance(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }

    generate_all_mama_tests!(
        check_mama_partial_params,
        check_mama_accuracy,
        check_mama_default_candles,
        check_mama_with_insufficient_data,
        check_mama_very_small_dataset,
        check_mama_reinput,
        check_mama_nan_handling,
        check_mama_no_poison,
        check_mama_property
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = MamaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MamaParams::default();
        let mama_row = output.mama_for(&def).expect("default row missing");
        assert_eq!(mama_row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
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

        let test_configs = vec![
            ((0.2, 0.4, 0.1), (0.02, 0.04, 0.01)),
            ((0.3, 0.7, 0.2), (0.03, 0.07, 0.02)),
            ((0.4, 0.9, 0.1), (0.01, 0.09, 0.02)),
            ((0.5, 0.8, 0.15), (0.01, 0.03, 0.01)),
            ((0.2, 0.6, 0.05), (0.02, 0.08, 0.01)),
        ];

        for (fast_range, slow_range) in test_configs {
            let output = MamaBatchBuilder::new()
                .kernel(kernel)
                .fast_limit_range(fast_range.0, fast_range.1, fast_range.2)
                .slow_limit_range(slow_range.0, slow_range.1, slow_range.2)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.mama_values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let params = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} in mama_values (params: fast_limit={:?}, slow_limit={:?})",
                        test, val, bits, row, col, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} in mama_values (params: fast_limit={:?}, slow_limit={:?})",
                        test, val, bits, row, col, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} in mama_values (params: fast_limit={:?}, slow_limit={:?})",
                        test, val, bits, row, col, params.fast_limit, params.slow_limit
                    );
                }
            }

            for (idx, &val) in output.fama_values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let params = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} in fama_values (params: fast_limit={:?}, slow_limit={:?})",
                        test, val, bits, row, col, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} in fama_values (params: fast_limit={:?}, slow_limit={:?})",
                        test, val, bits, row, col, params.fast_limit, params.slow_limit
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} in fama_values (params: fast_limit={:?}, slow_limit={:?})",
                        test, val, bits, row, col, params.fast_limit, params.slow_limit
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

    #[test]
    fn test_mama_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (t * 0.013).sin() * 10.0 + (t * 0.01)
            })
            .collect();

        let input = MamaInput::from_slice(&data, MamaParams::default());

        let baseline = mama(&input)?;

        let mut out_mama = vec![0.0; n];
        let mut out_fama = vec![0.0; n];
        #[allow(unused_variables)]
        {
            {
                super::mama_into(&input, &mut out_mama, &mut out_fama)?;
            }
        }

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        assert_eq!(baseline.mama_values.len(), out_mama.len());
        assert_eq!(baseline.fama_values.len(), out_fama.len());
        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline.mama_values[i], out_mama[i]),
                "mama mismatch at {}: left={} right={}",
                i,
                baseline.mama_values[i],
                out_mama[i]
            );
            assert!(
                eq_or_both_nan(baseline.fama_values[i], out_fama[i]),
                "fama mismatch at {}: left={} right={}",
                i,
                baseline.fama_values[i],
                out_fama[i]
            );
        }
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
