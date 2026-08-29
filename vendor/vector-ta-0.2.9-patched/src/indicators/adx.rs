use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdxData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdxOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AdxParams {
    pub period: Option<usize>,
}

impl Default for AdxParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AdxInput<'a> {
    pub data: AdxData<'a>,
    pub params: AdxParams,
}

impl<'a> AdxInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: AdxParams) -> Self {
        Self {
            data: AdxData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(h: &'a [f64], l: &'a [f64], c: &'a [f64], p: AdxParams) -> Self {
        Self {
            data: AdxData::Slices {
                high: h,
                low: l,
                close: c,
            },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, AdxParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdxBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for AdxBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdxBuilder {
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
    pub fn apply(self, candles: &Candles) -> Result<AdxOutput, AdxError> {
        let p = AdxParams {
            period: self.period,
        };
        let i = AdxInput::from_candles(candles, p);
        adx_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AdxOutput, AdxError> {
        let p = AdxParams {
            period: self.period,
        };
        let i = AdxInput::from_slices(high, low, close, p);
        adx_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AdxStream, AdxError> {
        let p = AdxParams {
            period: self.period,
        };
        AdxStream::try_new(p)
    }
}
#[derive(Debug, thiserror::Error)]
pub enum AdxError {
    #[error("adx: All values are NaN.")]
    AllValuesNaN,

    #[error("adx: Invalid period: period = {period}, data_len = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("adx: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("adx: Candle field error: {field}")]
    CandleFieldError { field: &'static str },

    #[error("adx: Input arrays must have the same length")]
    InconsistentLengths,

    #[error("adx: Input data slice is empty.")]
    EmptyInputData,

    #[error("adx: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("adx: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("adx: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_triple(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let fh = high.iter().position(|x| !x.is_nan()).unwrap_or(high.len());
    let fl = low.iter().position(|x| !x.is_nan()).unwrap_or(low.len());
    let fc = close
        .iter()
        .position(|x| !x.is_nan())
        .unwrap_or(close.len());
    fh.max(fl).max(fc)
}

#[inline(always)]
fn first_valid_triple_checked(high: &[f64], low: &[f64], close: &[f64]) -> Result<usize, AdxError> {
    let len = high.len();
    let mut fh = len;
    let mut fl = len;
    let mut fc = len;
    for i in 0..len {
        if fh == len && !high[i].is_nan() {
            fh = i;
        }
        if fl == len && !low[i].is_nan() {
            fl = i;
        }
        if fc == len && !close[i].is_nan() {
            fc = i;
        }
        if fh != len && fl != len && fc != len {
            return Ok(fh.max(fl).max(fc));
        }
    }
    if fh == len || fl == len || fc == len {
        Err(AdxError::AllValuesNaN)
    } else {
        Ok(fh.max(fl).max(fc))
    }
}

#[inline]
pub fn adx(input: &AdxInput) -> Result<AdxOutput, AdxError> {
    adx_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn adx_into(input: &AdxInput, out: &mut [f64]) -> Result<(), AdxError> {
    adx_into_slice(out, input, Kernel::Auto)
}

pub fn adx_with_kernel(input: &AdxInput, kernel: Kernel) -> Result<AdxOutput, AdxError> {
    let (high, low, close) = match &input.data {
        AdxData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        AdxData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.len() != low.len() || high.len() != close.len() {
        return Err(AdxError::InconsistentLengths);
    }
    let len = close.len();
    if len == 0 {
        return Err(AdxError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(AdxError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first = first_valid_triple_checked(high, low, close)?;
    if len - first < period + 1 {
        return Err(AdxError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let warm_end = first + (2 * period - 1);
    let mut out = alloc_with_nan_prefix(len, warm_end);

    let mut chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kernel, Kernel::Auto) && matches!(chosen, Kernel::Avx512 | Kernel::Avx512Batch) {
        chosen = Kernel::Avx2;
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => adx_scalar(
                &high[first..],
                &low[first..],
                &close[first..],
                period,
                &mut out[first..],
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => adx_avx2(
                &high[first..],
                &low[first..],
                &close[first..],
                period,
                &mut out[first..],
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => adx_avx512(
                &high[first..],
                &low[first..],
                &close[first..],
                period,
                &mut out[first..],
            ),
            _ => unreachable!(),
        }
    }

    Ok(AdxOutput { values: out })
}

#[inline]
pub fn adx_scalar(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    let len = close.len();
    if len <= period {
        return;
    }

    let period_f64 = period as f64;
    let reciprocal_period = 1.0 / period_f64;
    let one_minus_rp = 1.0 - reciprocal_period;
    let period_minus_one = period_f64 - 1.0;

    let mut tr_sum = 0.0f64;
    let mut plus_dm_sum = 0.0f64;
    let mut minus_dm_sum = 0.0f64;

    let mut prev_h = high[0];
    let mut prev_l = low[0];
    let mut prev_c = close[0];

    let mut i = 1usize;
    while i <= period {
        let ch = high[i];
        let cl = low[i];

        let hl = ch - cl;
        let hpc = (ch - prev_c).abs();
        let lpc = (cl - prev_c).abs();
        let tr = hl.max(hpc).max(lpc);

        let up = ch - prev_h;
        let down = prev_l - cl;
        if up > down && up > 0.0 {
            plus_dm_sum += up;
        }
        if down > up && down > 0.0 {
            minus_dm_sum += down;
        }
        tr_sum += tr;

        prev_h = ch;
        prev_l = cl;
        prev_c = close[i];
        i += 1;
    }

    let mut atr = tr_sum;
    let mut plus_dm_smooth = plus_dm_sum;
    let mut minus_dm_smooth = minus_dm_sum;

    let (plus_di_prev, minus_di_prev) = if atr != 0.0 {
        (
            (plus_dm_smooth / atr) * 100.0,
            (minus_dm_smooth / atr) * 100.0,
        )
    } else {
        (0.0, 0.0)
    };
    let sum_di_prev = plus_di_prev + minus_di_prev;
    let mut dx_sum = if sum_di_prev != 0.0 {
        ((plus_di_prev - minus_di_prev).abs() / sum_di_prev) * 100.0
    } else {
        0.0
    };
    let mut dx_count = 1usize;
    let mut last_adx = 0.0f64;

    let mut prev_h = high[period];
    let mut prev_l = low[period];
    let mut prev_c = close[period];

    let mut i = period + 1;
    while i < len {
        let ch = high[i];
        let cl = low[i];

        let hl = ch - cl;
        let hpc = (ch - prev_c).abs();
        let lpc = (cl - prev_c).abs();
        let tr = hl.max(hpc).max(lpc);

        let up = ch - prev_h;
        let down = prev_l - cl;
        let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
        let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };

        atr = atr * one_minus_rp + tr;
        plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
        minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

        let (plus_di, minus_di) = if atr != 0.0 {
            (
                (plus_dm_smooth / atr) * 100.0,
                (minus_dm_smooth / atr) * 100.0,
            )
        } else {
            (0.0, 0.0)
        };
        let sum_di = plus_di + minus_di;
        let dx = if sum_di != 0.0 {
            ((plus_di - minus_di).abs() / sum_di) * 100.0
        } else {
            0.0
        };

        if dx_count < period {
            dx_sum += dx;
            dx_count += 1;
            if dx_count == period {
                last_adx = dx_sum * reciprocal_period;
                out[i] = last_adx;
            }
        } else {
            last_adx = (last_adx * period_minus_one + dx) * reciprocal_period;
            out[i] = last_adx;
        }

        prev_h = ch;
        prev_l = cl;
        prev_c = close[i];
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx2(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    unsafe { adx_avx2_inner(high, low, close, period, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn adx_avx2_inner(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;
    let len = close.len();
    if len <= period {
        return;
    }

    let period_f64 = period as f64;
    let reciprocal_period = 1.0 / period_f64;
    let one_minus_rp = 1.0 - reciprocal_period;
    let period_minus_one = period_f64 - 1.0;

    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();

        let mut tr_sum = 0.0f64;
        let mut plus_dm_sum = 0.0f64;
        let mut minus_dm_sum = 0.0f64;

        let mut prev_h_scalar = *hp.add(0);
        let mut prev_l_scalar = *lp.add(0);
        let mut prev_c_scalar = *cp.add(0);

        let zero = _mm256_setzero_pd();
        let sign_mask = _mm256_set1_pd(-0.0f64);

        let mut i = 1usize;
        while i + 3 <= period {
            let ch = _mm256_loadu_pd(hp.add(i));
            let cl = _mm256_loadu_pd(lp.add(i));
            let pch = _mm256_loadu_pd(hp.add(i - 1));
            let pcl = _mm256_loadu_pd(lp.add(i - 1));
            let pcc = _mm256_loadu_pd(cp.add(i - 1));

            let hl = _mm256_sub_pd(ch, cl);
            let hpc = _mm256_andnot_pd(sign_mask, _mm256_sub_pd(ch, pcc));
            let lpc = _mm256_andnot_pd(sign_mask, _mm256_sub_pd(cl, pcc));
            let t0 = _mm256_max_pd(hl, hpc);
            let trv = _mm256_max_pd(t0, lpc);

            let up = _mm256_sub_pd(ch, pch);
            let down = _mm256_sub_pd(pcl, cl);
            let m_up_gt_down = _mm256_cmp_pd(up, down, _CMP_GT_OQ);
            let m_up_gt_zero = _mm256_cmp_pd(up, zero, _CMP_GT_OQ);
            let m_dn_gt_up = _mm256_cmp_pd(down, up, _CMP_GT_OQ);
            let m_dn_gt_zero = _mm256_cmp_pd(down, zero, _CMP_GT_OQ);
            let plus_mask = _mm256_and_pd(m_up_gt_down, m_up_gt_zero);
            let minus_mask = _mm256_and_pd(m_dn_gt_up, m_dn_gt_zero);
            let plus_v = _mm256_and_pd(plus_mask, up);
            let minus_v = _mm256_and_pd(minus_mask, down);

            let mut buf_tr = [0.0f64; 4];
            let mut buf_p = [0.0f64; 4];
            let mut buf_m = [0.0f64; 4];
            _mm256_storeu_pd(buf_tr.as_mut_ptr(), trv);
            _mm256_storeu_pd(buf_p.as_mut_ptr(), plus_v);
            _mm256_storeu_pd(buf_m.as_mut_ptr(), minus_v);

            tr_sum += buf_tr[0];
            plus_dm_sum += buf_p[0];
            minus_dm_sum += buf_m[0];
            tr_sum += buf_tr[1];
            plus_dm_sum += buf_p[1];
            minus_dm_sum += buf_m[1];
            tr_sum += buf_tr[2];
            plus_dm_sum += buf_p[2];
            minus_dm_sum += buf_m[2];
            tr_sum += buf_tr[3];
            plus_dm_sum += buf_p[3];
            minus_dm_sum += buf_m[3];

            prev_h_scalar = *hp.add(i + 3);
            prev_l_scalar = *lp.add(i + 3);
            prev_c_scalar = *cp.add(i + 3);

            i += 4;
        }
        while i <= period {
            let ch = *hp.add(i);
            let cl = *lp.add(i);
            let hl = ch - cl;
            let hpc = (ch - prev_c_scalar).abs();
            let lpc = (cl - prev_c_scalar).abs();
            let t0 = if hl > hpc { hl } else { hpc };
            let tr = if t0 > lpc { t0 } else { lpc };
            let up = ch - prev_h_scalar;
            let down = prev_l_scalar - cl;
            if up > down && up > 0.0 {
                plus_dm_sum += up;
            }
            if down > up && down > 0.0 {
                minus_dm_sum += down;
            }
            tr_sum += tr;
            prev_h_scalar = ch;
            prev_l_scalar = cl;
            prev_c_scalar = *cp.add(i);
            i += 1;
        }

        let mut atr = tr_sum;
        let mut plus_dm_smooth = plus_dm_sum;
        let mut minus_dm_smooth = minus_dm_sum;

        let (plus_di_prev, minus_di_prev) = if atr != 0.0 {
            (
                (plus_dm_smooth / atr) * 100.0,
                (minus_dm_smooth / atr) * 100.0,
            )
        } else {
            (0.0, 0.0)
        };
        let sum_di_prev = plus_di_prev + minus_di_prev;
        let mut dx_sum = if sum_di_prev != 0.0 {
            ((plus_di_prev - minus_di_prev).abs() / sum_di_prev) * 100.0
        } else {
            0.0
        };
        let mut dx_count = 1usize;
        let mut last_adx = 0.0f64;

        let mut prev_h = *hp.add(period);
        let mut prev_l = *lp.add(period);
        let mut prev_c = *cp.add(period);

        let mut i = period + 1;
        while i < len {
            let ch = *hp.add(i);
            let cl = *lp.add(i);

            let hl = ch - cl;
            let hpc = (ch - prev_c).abs();
            let lpc = (cl - prev_c).abs();
            let t0 = if hl > hpc { hl } else { hpc };
            let tr = if t0 > lpc { t0 } else { lpc };

            let up = ch - prev_h;
            let down = prev_l - cl;
            let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
            let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };

            atr = atr * one_minus_rp + tr;
            plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
            minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

            let (plus_di, minus_di) = if atr != 0.0 {
                (
                    (plus_dm_smooth / atr) * 100.0,
                    (minus_dm_smooth / atr) * 100.0,
                )
            } else {
                (0.0, 0.0)
            };
            let sum_di = plus_di + minus_di;
            let dx = if sum_di != 0.0 {
                ((plus_di - minus_di).abs() / sum_di) * 100.0
            } else {
                0.0
            };

            if dx_count < period {
                dx_sum += dx;
                dx_count += 1;
                if dx_count == period {
                    last_adx = dx_sum * reciprocal_period;
                    *out.get_unchecked_mut(i) = last_adx;
                }
            } else {
                last_adx = (last_adx * period_minus_one + dx) * reciprocal_period;
                *out.get_unchecked_mut(i) = last_adx;
            }

            prev_h = ch;
            prev_l = cl;
            prev_c = *cp.add(i);
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx512(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    unsafe { adx_avx512_inner(high, low, close, period, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn adx_avx512_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;
    let len = close.len();
    if len <= period {
        return;
    }

    let period_f64 = period as f64;
    let reciprocal_period = 1.0 / period_f64;
    let one_minus_rp = 1.0 - reciprocal_period;
    let period_minus_one = period_f64 - 1.0;

    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();

        let mut tr_sum = 0.0f64;
        let mut plus_dm_sum = 0.0f64;
        let mut minus_dm_sum = 0.0f64;

        let mut prev_h_scalar = *hp.add(0);
        let mut prev_l_scalar = *lp.add(0);
        let mut prev_c_scalar = *cp.add(0);

        let zero = _mm512_setzero_pd();
        let sign_mask = _mm512_set1_pd(-0.0f64);

        let mut i = 1usize;
        while i + 7 <= period {
            let ch = _mm512_loadu_pd(hp.add(i));
            let cl = _mm512_loadu_pd(lp.add(i));
            let pch = _mm512_loadu_pd(hp.add(i - 1));
            let pcl = _mm512_loadu_pd(lp.add(i - 1));
            let pcc = _mm512_loadu_pd(cp.add(i - 1));

            let hl = _mm512_sub_pd(ch, cl);
            let hpc = _mm512_andnot_pd(sign_mask, _mm512_sub_pd(ch, pcc));
            let lpc = _mm512_andnot_pd(sign_mask, _mm512_sub_pd(cl, pcc));
            let t0 = _mm512_max_pd(hl, hpc);
            let trv = _mm512_max_pd(t0, lpc);

            let up = _mm512_sub_pd(ch, pch);
            let down = _mm512_sub_pd(pcl, cl);
            let m_up_gt_down = _mm512_cmp_pd_mask(up, down, _CMP_GT_OQ);
            let m_up_gt_zero = _mm512_cmp_pd_mask(up, zero, _CMP_GT_OQ);
            let m_dn_gt_up = _mm512_cmp_pd_mask(down, up, _CMP_GT_OQ);
            let m_dn_gt_zero = _mm512_cmp_pd_mask(down, zero, _CMP_GT_OQ);
            let m_plus = m_up_gt_down & m_up_gt_zero;
            let m_minus = m_dn_gt_up & m_dn_gt_zero;
            let plus_v = _mm512_maskz_mov_pd(m_plus, up);
            let minus_v = _mm512_maskz_mov_pd(m_minus, down);

            let mut buf_tr = [0.0f64; 8];
            let mut buf_p = [0.0f64; 8];
            let mut buf_m = [0.0f64; 8];
            _mm512_storeu_pd(buf_tr.as_mut_ptr(), trv);
            _mm512_storeu_pd(buf_p.as_mut_ptr(), plus_v);
            _mm512_storeu_pd(buf_m.as_mut_ptr(), minus_v);

            tr_sum += buf_tr[0];
            plus_dm_sum += buf_p[0];
            minus_dm_sum += buf_m[0];
            tr_sum += buf_tr[1];
            plus_dm_sum += buf_p[1];
            minus_dm_sum += buf_m[1];
            tr_sum += buf_tr[2];
            plus_dm_sum += buf_p[2];
            minus_dm_sum += buf_m[2];
            tr_sum += buf_tr[3];
            plus_dm_sum += buf_p[3];
            minus_dm_sum += buf_m[3];
            tr_sum += buf_tr[4];
            plus_dm_sum += buf_p[4];
            minus_dm_sum += buf_m[4];
            tr_sum += buf_tr[5];
            plus_dm_sum += buf_p[5];
            minus_dm_sum += buf_m[5];
            tr_sum += buf_tr[6];
            plus_dm_sum += buf_p[6];
            minus_dm_sum += buf_m[6];
            tr_sum += buf_tr[7];
            plus_dm_sum += buf_p[7];
            minus_dm_sum += buf_m[7];

            prev_h_scalar = *hp.add(i + 7);
            prev_l_scalar = *lp.add(i + 7);
            prev_c_scalar = *cp.add(i + 7);

            i += 8;
        }
        while i <= period {
            let ch = *hp.add(i);
            let cl = *lp.add(i);
            let hl = ch - cl;
            let hpc = (ch - prev_c_scalar).abs();
            let lpc = (cl - prev_c_scalar).abs();
            let t0 = if hl > hpc { hl } else { hpc };
            let tr = if t0 > lpc { t0 } else { lpc };
            let up = ch - prev_h_scalar;
            let down = prev_l_scalar - cl;
            if up > down && up > 0.0 {
                plus_dm_sum += up;
            }
            if down > up && down > 0.0 {
                minus_dm_sum += down;
            }
            tr_sum += tr;
            prev_h_scalar = ch;
            prev_l_scalar = cl;
            prev_c_scalar = *cp.add(i);
            i += 1;
        }

        let mut atr = tr_sum;
        let mut plus_dm_smooth = plus_dm_sum;
        let mut minus_dm_smooth = minus_dm_sum;

        let (plus_di_prev, minus_di_prev) = if atr != 0.0 {
            (
                (plus_dm_smooth / atr) * 100.0,
                (minus_dm_smooth / atr) * 100.0,
            )
        } else {
            (0.0, 0.0)
        };
        let sum_di_prev = plus_di_prev + minus_di_prev;
        let mut dx_sum = if sum_di_prev != 0.0 {
            ((plus_di_prev - minus_di_prev).abs() / sum_di_prev) * 100.0
        } else {
            0.0
        };
        let mut dx_count = 1usize;
        let mut last_adx = 0.0f64;

        let mut prev_h = *hp.add(period);
        let mut prev_l = *lp.add(period);
        let mut prev_c = *cp.add(period);

        let mut i = period + 1;
        while i < len {
            let ch = *hp.add(i);
            let cl = *lp.add(i);

            let hl = ch - cl;
            let hpc = (ch - prev_c).abs();
            let lpc = (cl - prev_c).abs();
            let t0 = if hl > hpc { hl } else { hpc };
            let tr = if t0 > lpc { t0 } else { lpc };

            let up = ch - prev_h;
            let down = prev_l - cl;
            let plus_dm = if up > down && up > 0.0 { up } else { 0.0 };
            let minus_dm = if down > up && down > 0.0 { down } else { 0.0 };

            atr = atr * one_minus_rp + tr;
            plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
            minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

            let (plus_di, minus_di) = if atr != 0.0 {
                (
                    (plus_dm_smooth / atr) * 100.0,
                    (minus_dm_smooth / atr) * 100.0,
                )
            } else {
                (0.0, 0.0)
            };
            let sum_di = plus_di + minus_di;
            let dx = if sum_di != 0.0 {
                ((plus_di - minus_di).abs() / sum_di) * 100.0
            } else {
                0.0
            };

            if dx_count < period {
                dx_sum += dx;
                dx_count += 1;
                if dx_count == period {
                    last_adx = dx_sum * reciprocal_period;
                    *out.get_unchecked_mut(i) = last_adx;
                }
            } else {
                last_adx = (last_adx * period_minus_one + dx) * reciprocal_period;
                *out.get_unchecked_mut(i) = last_adx;
            }

            prev_h = ch;
            prev_l = cl;
            prev_c = *cp.add(i);
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx512_short(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    adx_avx512(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx512_long(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    adx_avx512(high, low, close, period, out)
}

#[inline]
pub fn adx_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxBatchRange,
    k: Kernel,
) -> Result<AdxBatchOutput, AdxError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        Kernel::Scalar => Kernel::ScalarBatch,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => Kernel::Avx2Batch,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => Kernel::Avx512Batch,
        _ => return Err(AdxError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    adx_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AdxBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for AdxBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

const ADX_SHARED_PRECOMP_THRESHOLD: usize = 16;

#[inline(always)]
fn precompute_streams_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let tail_len = high.len() - first;
    let mut tr = Vec::with_capacity(tail_len);
    let mut pdm = Vec::with_capacity(tail_len);
    let mut mdm = Vec::with_capacity(tail_len);
    tr.push(0.0);
    pdm.push(0.0);
    mdm.push(0.0);
    let mut prev_h = high[first];
    let mut prev_l = low[first];
    let mut prev_c = close[first];
    let mut j = 1usize;
    while first + j < high.len() {
        let ch = high[first + j];
        let cl = low[first + j];
        let hl = ch - cl;
        let hpc = (ch - prev_c).abs();
        let lpc = (cl - prev_c).abs();
        let trj = hl.max(hpc).max(lpc);
        let up = ch - prev_h;
        let down = prev_l - cl;
        let plus = if up > down && up > 0.0 { up } else { 0.0 };
        let minus = if down > up && down > 0.0 { down } else { 0.0 };
        tr.push(trj);
        pdm.push(plus);
        mdm.push(minus);
        prev_h = ch;
        prev_l = cl;
        prev_c = close[first + j];
        j += 1;
    }
    (tr, pdm, mdm)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn precompute_streams_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use core::arch::x86_64::*;
    let tail_len = high.len() - first;
    let mut tr = Vec::with_capacity(tail_len);
    let mut pdm = Vec::with_capacity(tail_len);
    let mut mdm = Vec::with_capacity(tail_len);
    tr.push(0.0);
    pdm.push(0.0);
    mdm.push(0.0);

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();

    let sign_mask = _mm256_set1_pd(-0.0f64);
    let zero = _mm256_setzero_pd();

    let mut prev_h_scalar = *hp.add(first);
    let mut prev_l_scalar = *lp.add(first);
    let mut prev_c_scalar = *cp.add(first);

    let mut j = 1usize;
    while j + 3 < tail_len {
        let base = first + j;
        let ch = _mm256_loadu_pd(hp.add(base));
        let cl = _mm256_loadu_pd(lp.add(base));
        let pch = _mm256_loadu_pd(hp.add(base - 1));
        let pcl = _mm256_loadu_pd(lp.add(base - 1));
        let pcc = _mm256_loadu_pd(cp.add(base - 1));

        let hl = _mm256_sub_pd(ch, cl);
        let hpc = _mm256_andnot_pd(sign_mask, _mm256_sub_pd(ch, pcc));
        let lpc = _mm256_andnot_pd(sign_mask, _mm256_sub_pd(cl, pcc));
        let t0 = _mm256_max_pd(hl, hpc);
        let trv = _mm256_max_pd(t0, lpc);

        let up = _mm256_sub_pd(ch, pch);
        let down = _mm256_sub_pd(pcl, cl);
        let m_up_gt_down = _mm256_cmp_pd(up, down, _CMP_GT_OQ);
        let m_up_gt_zero = _mm256_cmp_pd(up, zero, _CMP_GT_OQ);
        let m_dn_gt_up = _mm256_cmp_pd(down, up, _CMP_GT_OQ);
        let m_dn_gt_zero = _mm256_cmp_pd(down, zero, _CMP_GT_OQ);
        let plus_mask = _mm256_and_pd(m_up_gt_down, m_up_gt_zero);
        let minus_mask = _mm256_and_pd(m_dn_gt_up, m_dn_gt_zero);
        let plus_v = _mm256_and_pd(plus_mask, up);
        let minus_v = _mm256_and_pd(minus_mask, down);

        let mut buf_tr = [0.0f64; 4];
        let mut buf_p = [0.0f64; 4];
        let mut buf_m = [0.0f64; 4];
        _mm256_storeu_pd(buf_tr.as_mut_ptr(), trv);
        _mm256_storeu_pd(buf_p.as_mut_ptr(), plus_v);
        _mm256_storeu_pd(buf_m.as_mut_ptr(), minus_v);

        tr.push(buf_tr[0]);
        pdm.push(buf_p[0]);
        mdm.push(buf_m[0]);
        tr.push(buf_tr[1]);
        pdm.push(buf_p[1]);
        mdm.push(buf_m[1]);
        tr.push(buf_tr[2]);
        pdm.push(buf_p[2]);
        mdm.push(buf_m[2]);
        tr.push(buf_tr[3]);
        pdm.push(buf_p[3]);
        mdm.push(buf_m[3]);

        prev_h_scalar = *hp.add(base + 3);
        prev_l_scalar = *lp.add(base + 3);
        prev_c_scalar = *cp.add(base + 3);
        j += 4;
    }
    while j < tail_len {
        let ch = *hp.add(first + j);
        let cl = *lp.add(first + j);
        let hl = ch - cl;
        let hpc = (ch - prev_c_scalar).abs();
        let lpc = (cl - prev_c_scalar).abs();
        let trj = if hl > hpc { hl } else { hpc }.max(lpc);
        let up = ch - prev_h_scalar;
        let down = prev_l_scalar - cl;
        let plus = if up > down && up > 0.0 { up } else { 0.0 };
        let minus = if down > up && down > 0.0 { down } else { 0.0 };
        tr.push(trj);
        pdm.push(plus);
        mdm.push(minus);
        prev_h_scalar = ch;
        prev_l_scalar = cl;
        prev_c_scalar = *cp.add(first + j);
        j += 1;
    }
    (tr, pdm, mdm)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn precompute_streams_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use core::arch::x86_64::*;
    let tail_len = high.len() - first;
    let mut tr = Vec::with_capacity(tail_len);
    let mut pdm = Vec::with_capacity(tail_len);
    let mut mdm = Vec::with_capacity(tail_len);
    tr.push(0.0);
    pdm.push(0.0);
    mdm.push(0.0);

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();

    let sign_mask = _mm512_set1_pd(-0.0f64);
    let zero = _mm512_setzero_pd();

    let mut prev_h_scalar = *hp.add(first);
    let mut prev_l_scalar = *lp.add(first);
    let mut prev_c_scalar = *cp.add(first);

    let mut j = 1usize;
    while j + 7 < tail_len {
        let base = first + j;
        let ch = _mm512_loadu_pd(hp.add(base));
        let cl = _mm512_loadu_pd(lp.add(base));
        let pch = _mm512_loadu_pd(hp.add(base - 1));
        let pcl = _mm512_loadu_pd(lp.add(base - 1));
        let pcc = _mm512_loadu_pd(cp.add(base - 1));

        let hl = _mm512_sub_pd(ch, cl);
        let hpc = _mm512_andnot_pd(sign_mask, _mm512_sub_pd(ch, pcc));
        let lpc = _mm512_andnot_pd(sign_mask, _mm512_sub_pd(cl, pcc));
        let t0 = _mm512_max_pd(hl, hpc);
        let trv = _mm512_max_pd(t0, lpc);

        let up = _mm512_sub_pd(ch, pch);
        let down = _mm512_sub_pd(pcl, cl);
        let m_up_gt_down = _mm512_cmp_pd_mask(up, down, _CMP_GT_OQ);
        let m_up_gt_zero = _mm512_cmp_pd_mask(up, zero, _CMP_GT_OQ);
        let m_dn_gt_up = _mm512_cmp_pd_mask(down, up, _CMP_GT_OQ);
        let m_dn_gt_zero = _mm512_cmp_pd_mask(down, zero, _CMP_GT_OQ);
        let m_plus = m_up_gt_down & m_up_gt_zero;
        let m_minus = m_dn_gt_up & m_dn_gt_zero;
        let plus_v = _mm512_maskz_mov_pd(m_plus, up);
        let minus_v = _mm512_maskz_mov_pd(m_minus, down);

        let mut buf_tr = [0.0f64; 8];
        let mut buf_p = [0.0f64; 8];
        let mut buf_m = [0.0f64; 8];
        _mm512_storeu_pd(buf_tr.as_mut_ptr(), trv);
        _mm512_storeu_pd(buf_p.as_mut_ptr(), plus_v);
        _mm512_storeu_pd(buf_m.as_mut_ptr(), minus_v);

        for k in 0..8 {
            tr.push(buf_tr[k]);
            pdm.push(buf_p[k]);
            mdm.push(buf_m[k]);
        }
        prev_h_scalar = *hp.add(base + 7);
        prev_l_scalar = *lp.add(base + 7);
        prev_c_scalar = *cp.add(base + 7);
        j += 8;
    }
    while j < tail_len {
        let ch = *hp.add(first + j);
        let cl = *lp.add(first + j);
        let hl = ch - cl;
        let hpc = (ch - prev_c_scalar).abs();
        let lpc = (cl - prev_c_scalar).abs();
        let trj = if hl > hpc { hl } else { hpc }.max(lpc);
        let up = ch - prev_h_scalar;
        let down = prev_l_scalar - cl;
        let plus = if up > down && up > 0.0 { up } else { 0.0 };
        let minus = if down > up && down > 0.0 { down } else { 0.0 };
        tr.push(trj);
        pdm.push(plus);
        mdm.push(minus);
        prev_h_scalar = ch;
        prev_l_scalar = cl;
        prev_c_scalar = *cp.add(first + j);
        j += 1;
    }
    (tr, pdm, mdm)
}

#[derive(Clone, Debug, Default)]
pub struct AdxBatchBuilder {
    range: AdxBatchRange,
    kernel: Kernel,
}

impl AdxBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AdxBatchOutput, AdxError> {
        adx_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<AdxBatchOutput, AdxError> {
        let high = candles
            .select_candle_field("high")
            .map_err(|_| AdxError::CandleFieldError { field: "high" })?;
        let low = candles
            .select_candle_field("low")
            .map_err(|_| AdxError::CandleFieldError { field: "low" })?;
        let close = candles
            .select_candle_field("close")
            .map_err(|_| AdxError::CandleFieldError { field: "close" })?;
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AdxBatchOutput, AdxError> {
        AdxBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct AdxBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdxParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AdxBatchOutput {
    pub fn row_for_params(&self, p: &AdxParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }

    pub fn values_for(&self, p: &AdxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AdxBatchRange) -> Vec<AdxParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if start == end || step == 0 {
            return vec![start];
        }
        if start < end {
            return (start..=end).step_by(step.max(1)).collect();
        }

        let mut v = Vec::new();
        let mut cur = start;
        let s = step.max(1);
        while cur >= end {
            v.push(cur);
            if cur < s {
                break;
            }
            cur -= s;
            if cur == usize::MAX {
                break;
            }
        }
        v
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(AdxParams { period: Some(p) });
    }
    out
}

#[inline(always)]
fn adx_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AdxParams>, AdxError> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(AdxError::InconsistentLengths);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AdxError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let rows = combos.len();
    let cols = close.len();
    let expected = rows.checked_mul(cols).ok_or(AdxError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    if out.len() != expected {
        return Err(AdxError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let first = first_valid_triple(high, low, close);
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p + 1 {
        return Err(AdxError::NotEnoughValidData {
            needed: max_p + 1,
            valid: cols - first,
        });
    }

    let mut warms: Vec<usize> = Vec::with_capacity(combos.len());
    for c in &combos {
        let p = c.period.unwrap();
        let two_p = p.checked_mul(2).ok_or(AdxError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
        let warm = first
            .checked_add(two_p.saturating_sub(1))
            .ok_or(AdxError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            })?;
        warms.push(warm);
    }
    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    init_matrix_prefixes(&mut { out_mu }, cols, &warms);

    let use_shared = combos.len() >= ADX_SHARED_PRECOMP_THRESHOLD;

    if use_shared {
        let (tr_stream, plus_stream, minus_stream) = {
            match kern {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => unsafe { precompute_streams_avx512(high, low, close, first) },
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => unsafe { precompute_streams_avx2(high, low, close, first) },
                _ => precompute_streams_scalar(high, low, close, first),
            }
        };

        let do_row_shared = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
            let p = combos[row].period.unwrap();
            let row_f64 =
                core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
            let dst_tail = &mut row_f64[first..];

            let pf = p as f64;
            let rp = 1.0 / pf;
            let one_minus_rp = 1.0 - rp;
            let pm1 = pf - 1.0;

            let mut atr = 0.0f64;
            let mut plus_s = 0.0f64;
            let mut minus_s = 0.0f64;
            let mut j = 1usize;
            while j <= p {
                atr += tr_stream[j];
                plus_s += plus_stream[j];
                minus_s += minus_stream[j];
                j += 1;
            }
            let (plus_di_prev, minus_di_prev) = if atr != 0.0 {
                ((plus_s / atr) * 100.0, (minus_s / atr) * 100.0)
            } else {
                (0.0, 0.0)
            };
            let sum_di_prev = plus_di_prev + minus_di_prev;
            let mut dx_sum = if sum_di_prev != 0.0 {
                ((plus_di_prev - minus_di_prev).abs() / sum_di_prev) * 100.0
            } else {
                0.0
            };
            let mut dx_count = 1usize;
            let mut last_adx = 0.0f64;

            let tail_len = tr_stream.len();
            let mut j = p + 1;
            while j < tail_len {
                atr = atr * one_minus_rp + tr_stream[j];
                plus_s = plus_s * one_minus_rp + plus_stream[j];
                minus_s = minus_s * one_minus_rp + minus_stream[j];

                let (plus_di, minus_di) = if atr != 0.0 {
                    ((plus_s / atr) * 100.0, (minus_s / atr) * 100.0)
                } else {
                    (0.0, 0.0)
                };
                let sum_di = plus_di + minus_di;
                let dx = if sum_di != 0.0 {
                    ((plus_di - minus_di).abs() / sum_di) * 100.0
                } else {
                    0.0
                };

                if dx_count < p {
                    dx_sum += dx;
                    dx_count += 1;
                    if dx_count == p {
                        last_adx = dx_sum * rp;
                        dst_tail[j] = last_adx;
                    }
                } else {
                    last_adx = (last_adx * pm1 + dx) * rp;
                    dst_tail[j] = last_adx;
                }
                j += 1;
            }
        };

        let out_mu2 = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                out.len(),
            )
        };
        let rows_iter = (0..rows).zip(out_mu2.chunks_mut(cols));
        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            rows_iter
                .par_bridge()
                .for_each(|(r, s)| do_row_shared(r, s));
            #[cfg(target_arch = "wasm32")]
            for (r, s) in rows_iter {
                do_row_shared(r, s);
            }
        } else {
            for (r, s) in rows_iter {
                do_row_shared(r, s);
            }
        }
        return Ok(combos);
    }

    let do_row = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let p = combos[row].period.unwrap();
        let row_f64 =
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        let dst_tail = &mut row_f64[first..];
        match kern {
            Kernel::Scalar => {
                adx_row_scalar(&high[first..], &low[first..], &close[first..], p, dst_tail)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                adx_row_avx2(&high[first..], &low[first..], &close[first..], p, dst_tail)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                adx_row_avx512(&high[first..], &low[first..], &close[first..], p, dst_tail)
            }
            _ => adx_row_scalar(&high[first..], &low[first..], &close[first..], p, dst_tail),
        }
    };

    let out_mu2 = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    let rows_iter = (0..rows).zip(out_mu2.chunks_mut(cols));
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        rows_iter.par_bridge().for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in rows_iter {
            do_row(r, s);
        }
    } else {
        for (r, s) in rows_iter {
            do_row(r, s);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn adx_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxBatchRange,
    kern: Kernel,
) -> Result<AdxBatchOutput, AdxError> {
    let simd_kern = match kern {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    adx_batch_inner(high, low, close, sweep, simd_kern, false)
}

#[inline(always)]
pub fn adx_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxBatchRange,
    kern: Kernel,
) -> Result<AdxBatchOutput, AdxError> {
    adx_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn adx_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdxBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AdxBatchOutput, AdxError> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(AdxError::InconsistentLengths);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AdxError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(AdxError::EmptyInputData);
    }

    let first = first_valid_triple(high, low, close);
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p + 1 {
        return Err(AdxError::NotEnoughValidData {
            needed: max_p + 1,
            valid: cols - first,
        });
    }

    let _cap = rows.checked_mul(cols).ok_or(AdxError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let mut warm: Vec<usize> = Vec::with_capacity(combos.len());
    for c in &combos {
        let p = c.period.unwrap();
        let two_p = p.checked_mul(2).ok_or(AdxError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
        let w = first
            .checked_add(two_p.saturating_sub(1))
            .ok_or(AdxError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            })?;
        warm.push(w);
    }
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = ManuallyDrop::new(buf_mu);
    let values: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let use_shared = combos.len() >= ADX_SHARED_PRECOMP_THRESHOLD;

    if use_shared {
        let (tr_stream, plus_stream, minus_stream) = {
            match kern {
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => unsafe { precompute_streams_avx512(high, low, close, first) },
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => unsafe { precompute_streams_avx2(high, low, close, first) },
                _ => precompute_streams_scalar(high, low, close, first),
            }
        };

        let do_row = |row: usize, out_row: &mut [f64]| {
            let p = combos[row].period.unwrap();
            let pf = p as f64;
            let rp = 1.0 / pf;
            let one_minus_rp = 1.0 - rp;
            let pm1 = pf - 1.0;
            let dst_tail = &mut out_row[first..];

            let mut atr = 0.0f64;
            let mut plus_s = 0.0f64;
            let mut minus_s = 0.0f64;
            let mut j = 1usize;
            while j <= p {
                atr += tr_stream[j];
                plus_s += plus_stream[j];
                minus_s += minus_stream[j];
                j += 1;
            }
            let (plus_di_prev, minus_di_prev) = if atr != 0.0 {
                ((plus_s / atr) * 100.0, (minus_s / atr) * 100.0)
            } else {
                (0.0, 0.0)
            };
            let sum_di_prev = plus_di_prev + minus_di_prev;
            let mut dx_sum = if sum_di_prev != 0.0 {
                ((plus_di_prev - minus_di_prev).abs() / sum_di_prev) * 100.0
            } else {
                0.0
            };
            let mut dx_count = 1usize;
            let mut last_adx = 0.0f64;

            let tail_len = tr_stream.len();
            let mut j = p + 1;
            while j < tail_len {
                atr = atr * one_minus_rp + tr_stream[j];
                plus_s = plus_s * one_minus_rp + plus_stream[j];
                minus_s = minus_s * one_minus_rp + minus_stream[j];

                let (plus_di, minus_di) = if atr != 0.0 {
                    ((plus_s / atr) * 100.0, (minus_s / atr) * 100.0)
                } else {
                    (0.0, 0.0)
                };
                let sum_di = plus_di + minus_di;
                let dx = if sum_di != 0.0 {
                    ((plus_di - minus_di).abs() / sum_di) * 100.0
                } else {
                    0.0
                };

                if dx_count < p {
                    dx_sum += dx;
                    dx_count += 1;
                    if dx_count == p {
                        last_adx = dx_sum * rp;
                        dst_tail[j] = last_adx;
                    }
                } else {
                    last_adx = (last_adx * pm1 + dx) * rp;
                    dst_tail[j] = last_adx;
                }
                j += 1;
            }
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
            #[cfg(target_arch = "wasm32")]
            for (r, s) in values.chunks_mut(cols).enumerate() {
                do_row(r, s);
            }
        } else {
            for (r, s) in values.chunks_mut(cols).enumerate() {
                do_row(r, s);
            }
        }

        let values = unsafe {
            Vec::from_raw_parts(
                guard.as_mut_ptr() as *mut f64,
                guard.len(),
                guard.capacity(),
            )
        };

        return Ok(AdxBatchOutput {
            values,
            combos,
            rows,
            cols,
        });
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let pf = p as f64;
        let rp = 1.0 / pf;
        let one_minus_rp = 1.0 - rp;
        let pm1 = pf - 1.0;
        let dst_tail = &mut out_row[first..];
        match kern {
            Kernel::Scalar => {
                adx_row_scalar(&high[first..], &low[first..], &close[first..], p, dst_tail)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                adx_row_avx2(&high[first..], &low[first..], &close[first..], p, dst_tail)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                adx_row_avx512(&high[first..], &low[first..], &close[first..], p, dst_tail)
            }
            _ => adx_row_scalar(&high[first..], &low[first..], &close[first..], p, dst_tail),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in values.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in values.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(AdxBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn adx_row_scalar(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
    adx_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &mut [f64],
) {
    adx_avx2(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &mut [f64],
) {
    adx_avx512(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &mut [f64],
) {
    adx_avx512(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &mut [f64],
) {
    adx_avx512(high, low, close, period, out)
}

#[derive(Debug, Clone)]
pub struct AdxStream {
    period: usize,
    atr: f64,
    plus_dm_smooth: f64,
    minus_dm_smooth: f64,
    dx_sum: f64,
    dx_count: usize,
    last_adx: f64,
    count: usize,
    prev_high: f64,
    prev_low: f64,
    prev_close: f64,
}

impl AdxStream {
    pub fn try_new(params: AdxParams) -> Result<Self, AdxError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(AdxError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            atr: 0.0,
            plus_dm_smooth: 0.0,
            minus_dm_smooth: 0.0,
            dx_sum: 0.0,
            dx_count: 0,
            last_adx: 0.0,
            count: 0,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if self.count == 0 {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            self.count = 1;
            return None;
        }

        let prev_c = self.prev_close;
        let tr = high.max(prev_c) - low.min(prev_c);

        let up_move = high - self.prev_high;
        let down_move = self.prev_low - low;
        let plus_dm = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        let minus_dm = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };

        self.count += 1;

        if self.count <= self.period + 1 {
            self.atr += tr;
            self.plus_dm_smooth += plus_dm;
            self.minus_dm_smooth += minus_dm;

            if self.count == self.period + 1 {
                let inv_atr100 = if self.atr != 0.0 {
                    100.0 / self.atr
                } else {
                    0.0
                };
                let plus_di = self.plus_dm_smooth * inv_atr100;
                let minus_di = self.minus_dm_smooth * inv_atr100;
                let sum_di = plus_di + minus_di;

                self.dx_sum = if sum_di != 0.0 {
                    ((plus_di - minus_di).abs() / sum_di) * 100.0
                } else {
                    0.0
                };
                self.dx_count = 1;
            }

            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return None;
        }

        let rp = 1.0 / (self.period as f64);
        let one_minus_rp = 1.0 - rp;
        let period_minus_one = (self.period as f64) - 1.0;

        self.atr = self.atr * one_minus_rp + tr;
        self.plus_dm_smooth = self.plus_dm_smooth * one_minus_rp + plus_dm;
        self.minus_dm_smooth = self.minus_dm_smooth * one_minus_rp + minus_dm;

        let inv_atr100 = if self.atr != 0.0 {
            100.0 / self.atr
        } else {
            0.0
        };
        let plus_di = self.plus_dm_smooth * inv_atr100;
        let minus_di = self.minus_dm_smooth * inv_atr100;
        let sum_di = plus_di + minus_di;

        let dx = if sum_di != 0.0 {
            ((plus_di - minus_di).abs() / sum_di) * 100.0
        } else {
            0.0
        };

        let out = if self.dx_count < self.period {
            self.dx_sum += dx;
            self.dx_count += 1;
            if self.dx_count == self.period {
                self.last_adx = self.dx_sum * rp;
                Some(self.last_adx)
            } else {
                None
            }
        } else {
            self.last_adx = (self.last_adx * period_minus_one + dx) * rp;
            Some(self.last_adx)
        };

        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    #[test]
    fn test_adx_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let base = 100.0 + 0.5 * t + (t * 0.1).sin() * 0.7;
            let c = base;
            let h = c + 0.6 + (t * 0.05).cos() * 0.1;
            let l = c - 0.6 - (t * 0.07).sin() * 0.1;
            high.push(h);
            low.push(l);
            close.push(c);
        }

        let input = AdxInput::from_slices(&high, &low, &close, AdxParams::default());

        let AdxOutput { values: expected } = adx(&input)?;

        let mut got = vec![0.0; n];
        {
            adx_into(&input, &mut got)?;
        }

        assert_eq!(expected.len(), got.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }
        for i in 0..n {
            assert!(
                eq_or_both_nan(expected[i], got[i]),
                "mismatch at {}: expected {:?}, got {:?}",
                i,
                expected[i],
                got[i]
            );
        }
        Ok(())
    }

    fn check_adx_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = AdxParams { period: None };
        let input = AdxInput::from_candles(&candles, default_params);
        let output = adx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_adx_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = AdxInput::from_candles(&candles, AdxParams::default());
        let result = adx_with_kernel(&input, kernel)?;
        let expected_last_five = [36.14, 36.52, 37.01, 37.46, 38.47];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] ADX {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_adx_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = AdxInput::with_default_candles(&candles);
        let output = adx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_adx_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let close = [9.0, 19.0, 29.0];
        let params = AdxParams { period: Some(0) };
        let input = AdxInput::from_slices(&high, &low, &close, params);
        let res = adx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADX should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_adx_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let close = [9.0, 19.0, 29.0];
        let params = AdxParams { period: Some(10) };
        let input = AdxInput::from_slices(&high, &low, &close, params);
        let res = adx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADX should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_adx_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [41.0];
        let close = [40.5];
        let params = AdxParams { period: Some(14) };
        let input = AdxInput::from_slices(&high, &low, &close, params);
        let res = adx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ADX should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_adx_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = AdxParams { period: Some(14) };
        let first_input = AdxInput::from_candles(&candles, first_params);
        let first_result = adx_with_kernel(&first_input, kernel)?;

        let second_params = AdxParams { period: Some(5) };
        let second_input = AdxInput::from_slices(
            &candles.high,
            &candles.low,
            &first_result.values,
            second_params,
        );
        let second_result = adx_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), candles.close.len());
        for i in 100..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_adx_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = AdxInput::from_candles(&candles, AdxParams { period: Some(14) });
        let res = adx_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 100 {
            for (i, &val) in res.values[100..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    100 + i
                );
            }
        }
        Ok(())
    }

    fn check_adx_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let period = 14;

        let input = AdxInput::from_candles(
            &candles,
            AdxParams {
                period: Some(period),
            },
        );
        let batch_output = adx_with_kernel(&input, kernel)?.values;

        let mut stream = AdxStream::try_new(AdxParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), &c) in candles.high.iter().zip(&candles.low).zip(&candles.close) {
            match stream.update(h, l, c) {
                Some(adx_val) => stream_values.push(adx_val),
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
                diff < 1e-8,
                "[{}] ADX streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_adx_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            AdxParams::default(),
            AdxParams { period: Some(5) },
            AdxParams { period: Some(10) },
            AdxParams { period: Some(20) },
            AdxParams { period: Some(50) },
        ];

        for params in test_params {
            let input =
                AdxInput::from_slices(&candles.high, &candles.low, &candles.close, params.clone());
            let output = adx_with_kernel(&input, kernel)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
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
                        idx,
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
                        idx,
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
                        idx,
                        params.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_adx_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_adx_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=100).prop_flat_map(|period| {
            (
                (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                (0.01f64..0.2f64),
                period + 1..400,
            )
                .prop_flat_map(move |(base_price, volatility, len)| {
                    prop::collection::vec(
                        (0f64..1f64).prop_map(move |rand| {
                            let change = (rand - 0.5) * volatility * base_price.abs();
                            let open = base_price + change;
                            let close = open + (rand - 0.5) * volatility * base_price.abs() * 0.5;
                            let high = open.max(close) + rand * volatility * base_price.abs() * 0.3;
                            let low = open.min(close) - rand * volatility * base_price.abs() * 0.3;
                            (high, low, close)
                        }),
                        len,
                    )
                    .prop_map(move |bars| (bars, period))
                })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(bars, period)| {
                let mut highs = Vec::with_capacity(bars.len());
                let mut lows = Vec::with_capacity(bars.len());
                let mut closes = Vec::with_capacity(bars.len());

                for &(h, l, c) in &bars {
                    highs.push(h);
                    lows.push(l);
                    closes.push(c);
                }

                let params = AdxParams {
                    period: Some(period),
                };
                let input = AdxInput::from_slices(&highs, &lows, &closes, params.clone());

                let AdxOutput { values: out } = adx_with_kernel(&input, kernel).unwrap();
                let AdxOutput { values: ref_out } =
                    adx_with_kernel(&input, Kernel::Scalar).unwrap();

                let warmup_period = 2 * period - 1;
                for i in 0..warmup_period.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Property 1: Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                if out.len() > warmup_period + 10 {
                    for i in (warmup_period + 10)..out.len() {
                        prop_assert!(
                            !out[i].is_nan(),
                            "[{}] Property 2: Unexpected NaN after warmup at index {}",
                            test_name,
                            i
                        );
                    }
                }

                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val >= 0.0 && val <= 100.0,
                            "[{}] Property 3: ADX value {} at index {} outside [0, 100] range",
                            test_name,
                            val,
                            i
                        );
                    }
                }

                let const_price = 100.0;
                let const_highs = vec![const_price; closes.len()];
                let const_lows = vec![const_price; closes.len()];
                let const_closes = vec![const_price; closes.len()];
                let const_input =
                    AdxInput::from_slices(&const_highs, &const_lows, &const_closes, params.clone());

                if let Ok(AdxOutput { values: const_out }) = adx_with_kernel(&const_input, kernel) {
                    for i in warmup_period..const_out.len() {
                        if !const_out[i].is_nan() {
                            prop_assert!(
								const_out[i] <= 1.0,
								"[{}] Property 4: ADX should be near 0 for constant prices, got {} at index {}",
								test_name, const_out[i], i
							);
                        }
                    }
                }

                prop_assert_eq!(
                    out.len(),
                    ref_out.len(),
                    "[{}] Property 5: Kernel output length mismatch",
                    test_name
                );

                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] Property 5: NaN/Inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "[{}] Property 5: Kernel mismatch at index {}: {} vs {} (ULP={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                if period == 2 {
                    prop_assert!(
                        out.len() == closes.len(),
                        "[{}] Property 6: Output length mismatch with period=2",
                        test_name
                    );

                    if out.len() > 3 {
                        prop_assert!(
                            !out[3].is_nan(),
                            "[{}] Property 6: Should have valid ADX at index 3 with period=2",
                            test_name
                        );
                    }
                }

                let trend_len = closes.len();
                let mut trend_highs = Vec::with_capacity(trend_len);
                let mut trend_lows = Vec::with_capacity(trend_len);
                let mut trend_closes = Vec::with_capacity(trend_len);

                for i in 0..trend_len {
                    let base = 100.0 + (i as f64) * 2.0;
                    trend_lows.push(base - 0.5);
                    trend_highs.push(base + 0.5);
                    trend_closes.push(base);
                }

                let trend_input =
                    AdxInput::from_slices(&trend_highs, &trend_lows, &trend_closes, params.clone());

                if let Ok(AdxOutput { values: trend_out }) = adx_with_kernel(&trend_input, kernel) {
                    let last_valid_adx = trend_out
                        .iter()
                        .rposition(|&v| !v.is_nan())
                        .and_then(|i| Some(trend_out[i]));

                    if let Some(adx_val) = last_valid_adx {
                        prop_assert!(
                            adx_val > 20.0,
                            "[{}] Property 7: Strong trend should produce high ADX, got {}",
                            test_name,
                            adx_val
                        );
                    }
                }

                let doji_price = 100.0;
                let mut doji_highs = Vec::with_capacity(closes.len());
                let mut doji_lows = Vec::with_capacity(closes.len());
                let mut doji_closes = Vec::with_capacity(closes.len());

                for _ in 0..closes.len() {
                    doji_highs.push(doji_price + 0.01);
                    doji_lows.push(doji_price - 0.01);
                    doji_closes.push(doji_price);
                }

                let doji_input =
                    AdxInput::from_slices(&doji_highs, &doji_lows, &doji_closes, params.clone());

                if let Ok(AdxOutput { values: doji_out }) = adx_with_kernel(&doji_input, kernel) {
                    for i in warmup_period..doji_out.len() {
                        if !doji_out[i].is_nan() {
                            prop_assert!(
								doji_out[i] <= 30.0,
								"[{}] Property 8: Low movement should produce low ADX, got {} at index {}",
								test_name, doji_out[i], i
							);
                        }
                    }
                }

                if out.len() > warmup_period {
                    prop_assert!(
                        !out[warmup_period].is_nan(),
                        "[{}] Property 9: Should have valid ADX at index {} (warmup_period)",
                        test_name,
                        warmup_period
                    );
                    if warmup_period > 0 {
                        prop_assert!(
                            out[warmup_period - 1].is_nan(),
                            "[{}] Property 9: Should have NaN at index {} (before warmup_period)",
                            test_name,
                            warmup_period - 1
                        );
                    }
                }

                #[cfg(debug_assertions)]
                {
                    for (i, &val) in out.iter().enumerate() {
                        if val.is_finite() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "[{}] Property 10: Found poison value {} (0x{:016X}) at index {}",
                                test_name,
                                val,
                                bits,
                                i
                            );
                        }
                    }
                }

                for (i, &(h, l, c)) in bars.iter().enumerate() {
                    prop_assert!(
                        h >= l,
                        "[{}] Property 11: High {} < Low {} at index {}",
                        test_name,
                        h,
                        l,
                        i
                    );
                    prop_assert!(
                        c >= l && c <= h,
                        "[{}] Property 11: Close {} outside [Low {}, High {}] at index {}",
                        test_name,
                        c,
                        l,
                        h,
                        i
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_adx_tests {
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

    generate_all_adx_tests!(
        check_adx_partial_params,
        check_adx_accuracy,
        check_adx_default_candles,
        check_adx_zero_period,
        check_adx_period_exceeds_length,
        check_adx_very_small_dataset,
        check_adx_reinput,
        check_adx_nan_handling,
        check_adx_streaming,
        check_adx_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_adx_tests!(check_adx_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = AdxBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = AdxParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| p.period == def.period)
            .expect("default row missing");
        let slice = &output.values[row * output.cols..][..output.cols];

        assert_eq!(slice.len(), c.close.len());
        let expected = [36.14, 36.52, 37.01, 37.46, 38.47];
        let start = slice.len().saturating_sub(5);
        for (i, &v) in slice[start..].iter().enumerate() {
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
            (5, 20, 5),
            (10, 30, 10),
            (14, 14, 1),
            (20, 50, 15),
            (2, 10, 2),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = AdxBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
                    );
                }
            }

            let params = expand_grid(&AdxBatchRange {
                period: (p_start, p_end, p_step),
            });

            for p in &params {
                if let Some(slice) = output.values_for(p) {
                    for (idx, &val) in slice.iter().enumerate() {
                        if val.is_nan() || val.is_infinite() {
                            continue;
                        }

                        let bits = val.to_bits();
                        if bits == 0x11111111_11111111
                            || bits == 0x22222222_22222222
                            || bits == 0x33333333_33333333
                        {
                            panic!(
                                "[{}] Config {}: Found poison value {} (0x{:016X}) in sliced output \
								at index {} with params: period={}",
                                test,
                                cfg_idx,
                                val,
                                bits,
                                idx,
                                p.period.unwrap_or(14)
                            );
                        }
                    }
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
}

#[inline]
pub fn adx_into_slice(dst: &mut [f64], input: &AdxInput, kern: Kernel) -> Result<(), AdxError> {
    let (high, low, close) = match &input.data {
        AdxData::Candles { candles } => (&candles.high[..], &candles.low[..], &candles.close[..]),
        AdxData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.len() != low.len() || high.len() != close.len() {
        return Err(AdxError::InconsistentLengths);
    }
    let len = close.len();
    if dst.len() != len {
        return Err(AdxError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    if len == 0 {
        return Err(AdxError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(AdxError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first = first_valid_triple_checked(high, low, close)?;
    if len - first < period + 1 {
        return Err(AdxError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let warm_end = first + (2 * period - 1);
    for v in &mut dst[..warm_end.min(len)] {
        *v = f64::NAN;
    }

    let mut chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kern, Kernel::Auto) && matches!(chosen, Kernel::Avx512 | Kernel::Avx512Batch) {
        chosen = Kernel::Avx2;
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => adx_scalar(
                &high[first..],
                &low[first..],
                &close[first..],
                period,
                &mut dst[first..],
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => adx_avx2(
                &high[first..],
                &low[first..],
                &close[first..],
                period,
                &mut dst[first..],
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => adx_avx512(
                &high[first..],
                &low[first..],
                &close[first..],
                period,
                &mut dst[first..],
            ),
            _ => unreachable!(),
        }
    }
    Ok(())
}
