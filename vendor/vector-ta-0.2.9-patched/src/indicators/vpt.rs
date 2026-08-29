use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct VptParams;

#[derive(Debug, Clone)]
pub struct VptInput<'a> {
    pub data: VptData<'a>,
    pub params: VptParams,
}

impl<'a> VptInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: VptData::Candles { candles, source },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64]) -> Self {
        Self {
            data: VptData::Slices { price, volume },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VptData::Candles {
                candles,
                source: "close",
            },
            params: VptParams::default(),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VptBuilder {
    kernel: Kernel,
}

impl VptBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VptOutput, VptError> {
        let i = VptInput::with_default_candles(c);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
        let i = VptInput::from_slices(price, volume);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> VptStream {
        VptStream::default()
    }
}

#[derive(Debug, Error)]
pub enum VptError {
    #[error("vpt: Empty data provided.")]
    EmptyInputData,
    #[error("vpt: All values are NaN.")]
    AllValuesNaN,
    #[error("vpt: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vpt: Not enough valid data (needed = {needed}, valid = {valid}).")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vpt: Output length mismatch. expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vpt: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("vpt: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("vpt: size overflow computing rows*cols")]
    SizeOverflow,
}

#[inline]
fn vpt_first_valid(price: &[f64], volume: &[f64]) -> Option<usize> {
    for i in 1..price.len() {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
            return Some(i);
        }
    }
    None
}

#[inline]
pub fn vpt(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, Kernel::Auto)
}

pub fn vpt_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = match *source {
                "close" => candles.close.as_slice(),
                _ => source_type(candles, source),
            };
            (price, candles.volume.as_slice())
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }

    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;
    let mut values = alloc_with_nan_prefix(price.len(), first + 1);
    let _ = kernel;
    unsafe {
        vpt_row_scalar_from(price, volume, first + 1, &mut values);
    }
    Ok(VptOutput { values })
}

#[inline]
pub unsafe fn vpt_scalar(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    let n = price.len();
    if n == 0 || volume.len() != n {
        return Err(VptError::EmptyInputData);
    }
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;
    let mut res = alloc_with_nan_prefix(n, first + 1);

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = res.as_mut_ptr();

    let mut prev = {
        let p0 = *p_ptr.add(first - 1);
        let p1 = *p_ptr.add(first);
        let v1 = *v_ptr.add(first);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    };

    let mut i = first + 1;
    let mut p_prev = *p_ptr.add(i - 1);

    while i + 3 < n {
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur0 = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val0 = cur0 + prev;
        *o_ptr.add(i) = val0;
        prev = val0;
        p_prev = p1;

        let j1 = i + 1;
        let p2 = *p_ptr.add(j1);
        let v2 = *v_ptr.add(j1);
        let cur1 = if (p_prev != p_prev) || (p_prev == 0.0) || (p2 != p2) || (v2 != v2) {
            f64::NAN
        } else {
            v2 * ((p2 - p_prev) / p_prev)
        };
        let val1 = cur1 + prev;
        *o_ptr.add(j1) = val1;
        prev = val1;
        p_prev = p2;

        let j2 = i + 2;
        let p3 = *p_ptr.add(j2);
        let v3 = *v_ptr.add(j2);
        let cur2 = if (p_prev != p_prev) || (p_prev == 0.0) || (p3 != p3) || (v3 != v3) {
            f64::NAN
        } else {
            v3 * ((p3 - p_prev) / p_prev)
        };
        let val2 = cur2 + prev;
        *o_ptr.add(j2) = val2;
        prev = val2;
        p_prev = p3;

        let j3 = i + 3;
        let p4 = *p_ptr.add(j3);
        let v4 = *v_ptr.add(j3);
        let cur3 = if (p_prev != p_prev) || (p_prev == 0.0) || (p4 != p4) || (v4 != v4) {
            f64::NAN
        } else {
            v4 * ((p4 - p_prev) / p_prev)
        };
        let val3 = cur3 + prev;
        *o_ptr.add(j3) = val3;
        prev = val3;
        p_prev = p4;

        i += 4;
    }

    while i < n {
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val = cur + prev;
        *o_ptr.add(i) = val;
        prev = val;
        p_prev = p1;
        i += 1;
    }

    Ok(VptOutput { values: res })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx2(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    use core::arch::x86_64::*;

    let n = price.len();
    if n == 0 || volume.len() != n {
        return Err(VptError::EmptyInputData);
    }
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;
    let mut out = alloc_with_nan_prefix(n, first + 1);

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let mut prev = {
        let p0 = *p_ptr.add(first - 1);
        let p1 = *p_ptr.add(first);
        let v1 = *v_ptr.add(first);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    };

    let mut i = first + 1;
    let vzero = _mm256_set1_pd(0.0);
    let vnan = _mm256_set1_pd(f64::NAN);

    #[inline(always)]
    unsafe fn prefix4_pd(x: __m256d) -> __m256d {
        let lo = _mm256_castpd256_pd128(x);
        let hi = _mm256_extractf128_pd(x, 1);
        let z = _mm_setzero_pd();

        let tlo = _mm_add_pd(lo, _mm_shuffle_pd(z, lo, 0));
        let thi = _mm_add_pd(hi, _mm_shuffle_pd(z, hi, 0));

        let last_lo = _mm_unpackhi_pd(tlo, tlo);
        let thi2 = _mm_add_pd(thi, last_lo);

        _mm256_insertf128_pd(_mm256_castpd128_pd256(tlo), thi2, 1)
    }

    while i + 3 < n {
        let p0 = _mm256_loadu_pd(p_ptr.add(i - 1));
        let p1 = _mm256_loadu_pd(p_ptr.add(i));
        let vv = _mm256_loadu_pd(v_ptr.add(i));

        let m_nan_p0 = _mm256_cmp_pd(p0, p0, _CMP_UNORD_Q);
        let m_nan_p1 = _mm256_cmp_pd(p1, p1, _CMP_UNORD_Q);
        let m_nan_v = _mm256_cmp_pd(vv, vv, _CMP_UNORD_Q);
        let m_eq0_p0 = _mm256_cmp_pd(p0, vzero, _CMP_EQ_OQ);
        let invalid = _mm256_or_pd(
            _mm256_or_pd(m_nan_p0, m_nan_p1),
            _mm256_or_pd(m_nan_v, m_eq0_p0),
        );

        let diff = _mm256_sub_pd(p1, p0);
        let div = _mm256_div_pd(diff, p0);
        let mul = _mm256_mul_pd(vv, div);
        let cur = _mm256_blendv_pd(mul, vnan, invalid);

        let ps = prefix4_pd(cur);
        let cary = _mm256_set1_pd(prev);
        let outv = _mm256_add_pd(ps, cary);

        _mm256_storeu_pd(o_ptr.add(i), outv);

        let hi128 = _mm256_extractf128_pd(outv, 1);
        let last_hi = _mm_unpackhi_pd(hi128, hi128);
        let tmp: [f64; 2] = core::mem::transmute(last_hi);
        prev = tmp[0];

        i += 4;
    }

    if i < n {
        let mut p_prev = *p_ptr.add(i - 1);
        while i < n {
            let p1 = *p_ptr.add(i);
            let v1 = *v_ptr.add(i);
            let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
                f64::NAN
            } else {
                v1 * ((p1 - p_prev) / p_prev)
            };
            let val = cur + prev;
            *o_ptr.add(i) = val;
            prev = val;
            p_prev = p1;
            i += 1;
        }
    }

    Ok(VptOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    use core::arch::x86_64::*;

    let n = price.len();
    if n == 0 || volume.len() != n {
        return Err(VptError::EmptyInputData);
    }
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;
    let mut out = alloc_with_nan_prefix(n, first + 1);

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let mut prev = {
        let p0 = *p_ptr.add(first - 1);
        let p1 = *p_ptr.add(first);
        let v1 = *v_ptr.add(first);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    };

    let mut i = first + 1;

    #[inline(always)]
    unsafe fn prefix4_pd(x: __m256d) -> __m256d {
        use core::arch::x86_64::*;
        let lo = _mm256_castpd256_pd128(x);
        let hi = _mm256_extractf128_pd(x, 1);
        let z = _mm_setzero_pd();
        let tlo = _mm_add_pd(lo, _mm_shuffle_pd(z, lo, 0));
        let thi = _mm_add_pd(hi, _mm_shuffle_pd(z, hi, 0));
        let last_lo = _mm_unpackhi_pd(tlo, tlo);
        let thi2 = _mm_add_pd(thi, last_lo);
        _mm256_insertf128_pd(_mm256_castpd128_pd256(tlo), thi2, 1)
    }

    while i + 7 < n {
        let p0 = _mm512_loadu_pd(p_ptr.add(i - 1));
        let p1 = _mm512_loadu_pd(p_ptr.add(i));
        let vv = _mm512_loadu_pd(v_ptr.add(i));

        let m_nan_p0 = _mm512_cmp_pd_mask(p0, p0, _CMP_UNORD_Q);
        let m_nan_p1 = _mm512_cmp_pd_mask(p1, p1, _CMP_UNORD_Q);
        let m_nan_v = _mm512_cmp_pd_mask(vv, vv, _CMP_UNORD_Q);
        let m_eq0_p0 = _mm512_cmp_pd_mask(p0, _mm512_set1_pd(0.0), _CMP_EQ_OQ);
        let invalid = m_nan_p0 | m_nan_p1 | m_nan_v | m_eq0_p0;

        let diff = _mm512_sub_pd(p1, p0);
        let r0 = _mm512_rcp14_pd(p0);
        let two = _mm512_set1_pd(2.0);
        let e1 = _mm512_fnmadd_pd(p0, r0, two);
        let r1 = _mm512_mul_pd(r0, e1);
        let e2 = _mm512_fnmadd_pd(p0, r1, two);
        let r2 = _mm512_mul_pd(r1, e2);
        let div = _mm512_mul_pd(diff, r2);
        let mul = _mm512_mul_pd(vv, div);
        let cur = _mm512_mask_mov_pd(mul, invalid, _mm512_set1_pd(f64::NAN));

        let lo256 = _mm512_castpd512_pd256(cur);
        let hi256 = _mm512_extractf64x4_pd(cur, 1);
        let lo_ps = prefix4_pd(lo256);
        let mut hi_ps = prefix4_pd(hi256);

        let lo_hi128 = _mm256_extractf128_pd(lo_ps, 1);
        let lo_total = {
            let last_lo = _mm_unpackhi_pd(lo_hi128, lo_hi128);
            let tmp: [f64; 2] = core::mem::transmute(last_lo);
            tmp[0]
        };
        hi_ps = _mm256_add_pd(hi_ps, _mm256_set1_pd(lo_total));

        let ps512 = _mm512_insertf64x4(_mm512_castpd256_pd512(lo_ps), hi_ps, 1);

        let outv = _mm512_add_pd(ps512, _mm512_set1_pd(prev));
        _mm512_storeu_pd(o_ptr.add(i), outv);

        let hi2 = _mm512_extractf64x4_pd(outv, 1);
        let hi128 = _mm256_extractf128_pd(hi2, 1);
        let last_hi = _mm_unpackhi_pd(hi128, hi128);
        let tmp: [f64; 2] = core::mem::transmute(last_hi);
        prev = tmp[0];

        i += 8;
    }

    while i + 3 < n {
        use core::arch::x86_64::*;
        let p0 = _mm256_loadu_pd(p_ptr.add(i - 1));
        let p1 = _mm256_loadu_pd(p_ptr.add(i));
        let vv = _mm256_loadu_pd(v_ptr.add(i));
        let vzero = _mm256_set1_pd(0.0);
        let vnan = _mm256_set1_pd(f64::NAN);

        let m_nan_p0 = _mm256_cmp_pd(p0, p0, _CMP_UNORD_Q);
        let m_nan_p1 = _mm256_cmp_pd(p1, p1, _CMP_UNORD_Q);
        let m_nan_v = _mm256_cmp_pd(vv, vv, _CMP_UNORD_Q);
        let m_eq0_p0 = _mm256_cmp_pd(p0, vzero, _CMP_EQ_OQ);
        let invalid = _mm256_or_pd(
            _mm256_or_pd(m_nan_p0, m_nan_p1),
            _mm256_or_pd(m_nan_v, m_eq0_p0),
        );

        let diff = _mm256_sub_pd(p1, p0);
        let div = _mm256_div_pd(diff, p0);
        let mul = _mm256_mul_pd(vv, div);
        let cur = _mm256_blendv_pd(mul, vnan, invalid);

        let ps = {
            let lo = _mm256_castpd256_pd128(cur);
            let hi = _mm256_extractf128_pd(cur, 1);
            let z = _mm_setzero_pd();
            let tlo = _mm_add_pd(lo, _mm_shuffle_pd(z, lo, 0));
            let thi = _mm_add_pd(hi, _mm_shuffle_pd(z, hi, 0));
            let last_lo = _mm_unpackhi_pd(tlo, tlo);
            let thi2 = _mm_add_pd(thi, last_lo);
            _mm256_insertf128_pd(_mm256_castpd128_pd256(tlo), thi2, 1)
        };

        let outv = _mm256_add_pd(ps, _mm256_set1_pd(prev));
        _mm256_storeu_pd(o_ptr.add(i), outv);
        let hi128 = _mm256_extractf128_pd(outv, 1);
        let last_hi = _mm_unpackhi_pd(hi128, hi128);
        let tmp: [f64; 2] = core::mem::transmute(last_hi);
        prev = tmp[0];
        i += 4;
    }

    if i < n {
        let mut p_prev = *p_ptr.add(i - 1);
        while i < n {
            let p1 = *p_ptr.add(i);
            let v1 = *v_ptr.add(i);
            let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
                f64::NAN
            } else {
                v1 * ((p1 - p_prev) / p_prev)
            };
            let val = cur + prev;
            *o_ptr.add(i) = val;
            prev = val;
            p_prev = p1;
            i += 1;
        }
    }

    Ok(VptOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_short(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_long(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[inline]
pub fn vpt_indicator(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt(input)
}

#[inline]
pub fn vpt_indicator_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, kernel)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx2(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx2(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_short(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_short(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_long(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_long(price, volume)
    }
}

#[inline]
pub fn vpt_indicator_scalar(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_scalar(price, volume)
    }
}

#[inline]
pub fn vpt_expand_grid() -> Vec<VptParams> {
    vec![VptParams::default()]
}

pub fn vpt_into(input: &VptInput, out: &mut [f64]) -> Result<(), VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = match *source {
                "close" => candles.close.as_slice(),
                _ => source_type(candles, source),
            };
            (price, candles.volume.as_slice())
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    vpt_into_slice(out, price, volume, Kernel::Auto)
}

pub fn vpt_into_slice(
    dst: &mut [f64],
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }

    if dst.len() != price.len() {
        return Err(VptError::OutputLengthMismatch {
            expected: price.len(),
            got: dst.len(),
        });
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }

    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                vpt_row_scalar_from(price, volume, first + 1, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2_from(price, volume, first + 1, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_row_avx512_from(price, volume, first + 1, dst)
            }
            _ => vpt_row_scalar_from(price, volume, first + 1, dst),
        }
    }
    for v in &mut dst[..=first] {
        *v = f64::NAN;
    }
    Ok(())
}

pub fn vpt_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    _range: &VptBatchRange,
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VptParams>, VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }
    let combos = vec![VptParams::default()];
    let cols = price.len();
    if out.len() != cols {
        return Err(VptError::OutputLengthMismatch {
            expected: cols,
            got: out.len(),
        });
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;

    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                vpt_row_scalar_from(price, volume, first + 1, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2_from(price, volume, first + 1, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_row_avx512_from(price, volume, first + 1, out)
            }
            _ => vpt_row_scalar_from(price, volume, first + 1, out),
        }
    }
    Ok(combos)
}

#[derive(Clone, Debug, Default)]
pub struct VptStream {
    last_price: f64,

    carry_inc: f64,

    cum: f64,

    seeded: bool,

    sticky_nan: bool,
}

impl VptStream {
    #[inline(always)]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        if !self.seeded {
            self.last_price = price;
            self.seeded = true;
            self.carry_inc = f64::NAN;
            self.cum = f64::NAN;
            self.sticky_nan = false;
            return None;
        }

        if self.sticky_nan {
            self.last_price = price;
            return Some(f64::NAN);
        }

        if !(self.last_price.is_finite()
            && self.last_price != 0.0
            && price.is_finite()
            && volume.is_finite())
        {
            self.sticky_nan = true;
            self.last_price = price;
            self.carry_inc = f64::NAN;
            self.cum = f64::NAN;
            return Some(f64::NAN);
        }

        let inv = 1.0 / self.last_price;
        let scale = volume * inv;
        let dv = price - self.last_price;
        self.last_price = price;

        let cur_inc = dv.mul_add(scale, 0.0);

        if self.carry_inc.is_nan() {
            self.carry_inc = cur_inc;
            return Some(f64::NAN);
        }

        let base = if self.cum.is_finite() {
            self.cum
        } else {
            self.carry_inc
        };
        let new_cum = base + cur_inc;

        self.carry_inc = cur_inc;
        self.cum = new_cum;
        Some(new_cum)
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline(always)]
    pub fn restart_from(&mut self, price: f64) {
        self.last_price = price;
        self.carry_inc = f64::NAN;
        self.cum = f64::NAN;
        self.seeded = true;
        self.sticky_nan = false;
    }
}

#[derive(Clone, Debug, Default)]
pub struct VptBatchRange;

#[derive(Clone, Debug, Default)]
pub struct VptBatchBuilder {
    kernel: Kernel,
}

impl VptBatchBuilder {
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptBatchOutput, VptError> {
        vpt_batch_with_kernel(price, volume, self.kernel)
    }

    pub fn with_default_slices(
        price: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new().kernel(k).apply_slices(price, volume)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VptBatchOutput, VptError> {
        let price = match src {
            "close" => c.close.as_slice(),
            _ => source_type(c, src),
        };
        self.apply_slices(price, c.volume.as_slice())
    }

    pub fn with_default_candles(c: &Candles) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vpt_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    k: Kernel,
) -> Result<VptBatchOutput, VptError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VptError::InvalidKernelForBatch(other)),
    };
    vpt_batch_par_slice(price, volume, kernel)
}

#[derive(Clone, Debug)]
pub struct VptBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VptParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VptBatchOutput {
    pub fn row_for_params(&self, _p: &VptParams) -> Option<usize> {
        Some(0)
    }

    pub fn values_for(&self, _p: &VptParams) -> Option<&[f64]> {
        Some(&self.values[..])
    }
}

#[inline(always)]
pub fn vpt_batch_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, false)
}

#[inline(always)]
pub fn vpt_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, true)
}

#[inline(always)]
fn vpt_batch_inner(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<VptBatchOutput, VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }

    let combos = vpt_expand_grid();
    let rows = 1usize;
    let cols = price.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first_valid = vpt_first_valid(price, volume).ok_or(VptError::NotEnoughValidData {
        needed: 2,
        valid: valid_count,
    })?;
    let warm = vec![first_valid + 1];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    vpt_batch_inner_into(price, volume, &VptBatchRange, kern, _parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VptBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn vpt_row_scalar(price: &[f64], volume: &[f64], out: &mut [f64]) {
    let n = price.len();
    if let Some(first) = vpt_first_valid(price, volume) {
        for i in 0..=first {
            out[i] = f64::NAN;
        }

        vpt_row_scalar_from(price, volume, first + 1, out);
    } else {
        for i in 0..n {
            out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
pub unsafe fn vpt_row_scalar_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    let n = price.len();
    if start_i >= n {
        return;
    }

    assert!(start_i > 0, "vpt_row_scalar_from requires start_i >= 1");

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let mut prev = if start_i >= 2 {
        let k = start_i - 1;
        let p0 = *p_ptr.add(k - 1);
        let p1 = *p_ptr.add(k);
        let v1 = *v_ptr.add(k);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    } else {
        0.0
    };

    let mut i = start_i;
    let mut p_prev = *p_ptr.add(i - 1);

    while i + 3 < n {
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur0 = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val0 = cur0 + prev;
        *o_ptr.add(i) = val0;
        prev = val0;
        p_prev = p1;

        let j1 = i + 1;
        let p2 = *p_ptr.add(j1);
        let v2 = *v_ptr.add(j1);
        let cur1 = if (p_prev != p_prev) || (p_prev == 0.0) || (p2 != p2) || (v2 != v2) {
            f64::NAN
        } else {
            v2 * ((p2 - p_prev) / p_prev)
        };
        let val1 = cur1 + prev;
        *o_ptr.add(j1) = val1;
        prev = val1;
        p_prev = p2;

        let j2 = i + 2;
        let p3 = *p_ptr.add(j2);
        let v3 = *v_ptr.add(j2);
        let cur2 = if (p_prev != p_prev) || (p_prev == 0.0) || (p3 != p3) || (v3 != v3) {
            f64::NAN
        } else {
            v3 * ((p3 - p_prev) / p_prev)
        };
        let val2 = cur2 + prev;
        *o_ptr.add(j2) = val2;
        prev = val2;
        p_prev = p3;

        let j3 = i + 3;
        let p4 = *p_ptr.add(j3);
        let v4 = *v_ptr.add(j3);
        let cur3 = if (p_prev != p_prev) || (p_prev == 0.0) || (p4 != p4) || (v4 != v4) {
            f64::NAN
        } else {
            v4 * ((p4 - p_prev) / p_prev)
        };
        let val3 = cur3 + prev;
        *o_ptr.add(j3) = val3;
        prev = val3;
        p_prev = p4;

        i += 4;
    }

    while i < n {
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val = cur + prev;
        *o_ptr.add(i) = val;
        prev = val;
        p_prev = p1;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    vpt_row_scalar_from(price, volume, start_i, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    vpt_row_scalar_from(price, volume, start_i, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_short(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_long(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_vpt_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VptInput::from_candles(&candles, "close");

        let baseline = vpt_with_kernel(&input, Kernel::Scalar)?;

        let mut out = vec![0.0f64; candles.close.len()];
        vpt_into(&input, &mut out)?;

        assert_eq!(baseline.values.len(), out.len());

        fn eq_or_both_nan_eps(a: f64, b: f64, eps: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= eps
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan_eps(baseline.values[i], out[i], 1e-12),
                "Mismatch at index {}: baseline={} out={}",
                i,
                baseline.values[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_vpt_basic_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vpt_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [1.0, 1.1, 1.05, 1.2, 1.3];
        let volume = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let input = VptInput::from_slices(&price, &volume);
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), price.len());
        Ok(())
    }

    fn check_vpt_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [100.0];
        let volume = [500.0];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN, f64::NAN];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_accuracy_from_csv(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -18292.323972247592,
            -18292.510374716476,
            -18292.803266539282,
            -18292.62919783763,
            -18296.152568643138,
        ];

        assert!(output.values.len() >= 5);
        let start_index = output.values.len() - 5;
        for (i, &value) in output.values[start_index..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-9,
                "VPT mismatch at final bars, index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vpt_tests {
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

    #[cfg(debug_assertions)]
    fn check_vpt_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_sources = vec!["close", "open", "high", "low"];

        for (source_idx, &source) in test_sources.iter().enumerate() {
            let input = VptInput::from_candles(&candles, source);
            let output = vpt_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vpt_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vpt_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=400).prop_flat_map(|len| {
            (
                prop::collection::vec(
                    (0.0f64..1e6f64)
                        .prop_filter("finite non-negative price", |x| x.is_finite() && *x >= 0.0),
                    len,
                ),
                prop::collection::vec(
                    (0.0f64..1e9f64)
                        .prop_filter("finite non-negative volume", |x| x.is_finite() && *x >= 0.0),
                    len,
                ),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(price, volume)| {
            let input = VptInput::from_slices(&price, &volume);

            let VptOutput { values: out } = vpt_with_kernel(&input, kernel)?;

            let VptOutput { values: ref_out } = vpt_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(out.len(), price.len(), "Output length mismatch");
            prop_assert_eq!(
                ref_out.len(),
                price.len(),
                "Reference output length mismatch"
            );

            prop_assert!(
                out[0].is_nan(),
                "First VPT value should be NaN, got {}",
                out[0]
            );
            prop_assert!(
                ref_out[0].is_nan(),
                "First reference VPT value should be NaN, got {}",
                ref_out[0]
            );

            let mut expected_vpt = vec![f64::NAN; price.len()];
            let mut prev_vpt_val = f64::NAN;

            for i in 1..price.len() {
                let p0 = price[i - 1];
                let p1 = price[i];
                let v1 = volume[i];

                let vpt_val = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
                    f64::NAN
                } else {
                    v1 * ((p1 - p0) / p0)
                };

                expected_vpt[i] = if vpt_val.is_nan() || prev_vpt_val.is_nan() {
                    f64::NAN
                } else {
                    vpt_val + prev_vpt_val
                };

                prev_vpt_val = vpt_val;
            }

            for i in 0..price.len() {
                let y = out[i];
                let r = ref_out[i];
                let e = expected_vpt[i];

                if y.is_nan() && r.is_nan() {
                    continue;
                } else if !y.is_nan() && !r.is_nan() {
                    let diff = (y - r).abs();
                    prop_assert!(
                        diff < 1e-9,
                        "Kernel mismatch at idx {}: {} vs {} (diff: {})",
                        i,
                        y,
                        r,
                        diff
                    );

                    if !e.is_nan() {
                        let diff_expected = (y - e).abs();
                        prop_assert!(
                            diff_expected < 1e-9,
                            "Value mismatch at idx {}: got {} expected {} (diff: {})",
                            i,
                            y,
                            e,
                            diff_expected
                        );
                    }
                } else {
                    prop_assert!(
                        false,
                        "NaN mismatch at idx {}: kernel={}, scalar={}",
                        i,
                        y,
                        r
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_vpt_tests!(
        check_vpt_basic_candles,
        check_vpt_basic_slices,
        check_vpt_not_enough_data,
        check_vpt_empty_data,
        check_vpt_all_nan,
        check_vpt_accuracy_from_csv,
        check_vpt_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vpt_tests!(check_vpt_property);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_sources = vec!["close", "open", "high", "low"];

        for (src_idx, &source) in test_sources.iter().enumerate() {
            let output = VptBatchBuilder::new()
                .kernel(kernel)
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
                        "[{}] Source {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Source {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Source {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
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
                    let kernel = detect_best_batch_kernel();
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), kernel);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_no_poison);
}
