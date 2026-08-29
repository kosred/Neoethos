use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";

#[inline(always)]
fn source_slice<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        DEFAULT_SOURCE => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "hlcc4" | "hlcc" => &candles.hlcc4,
        _ => source_type(candles, source),
    }
}

impl<'a> AsRef<[f64]> for VidyaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VidyaData::Slice(slice) => slice,
            VidyaData::Candles { candles, source } => source_slice(candles, source),
        }
    }
}

#[cfg(test)]
mod tests_into_parity {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_vortex;

    #[test]
    fn test_vidya_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VidyaInput::with_default_candles(&candles);

        let baseline = vidya(&input)?.values;

        let mut out = vec![0.0; candles.close.len()];
        {
            vidya_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "Mismatch at index {}: baseline={} into={}",
                i,
                a,
                b
            );
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum VidyaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VidyaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VidyaParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
    pub alpha: Option<f64>,
}

impl Default for VidyaParams {
    fn default() -> Self {
        Self {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VidyaInput<'a> {
    pub data: VidyaData<'a>,
    pub params: VidyaParams,
}

impl<'a> VidyaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VidyaParams) -> Self {
        Self {
            data: VidyaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VidyaParams) -> Self {
        Self {
            data: VidyaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, DEFAULT_SOURCE, VidyaParams::default())
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(2)
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_alpha(&self) -> f64 {
        self.params.alpha.unwrap_or(0.2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VidyaBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for VidyaBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VidyaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = Some(a);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VidyaOutput, VidyaError> {
        let p = VidyaParams {
            short_period: self.short_period,
            long_period: self.long_period,
            alpha: self.alpha,
        };
        let i = VidyaInput::from_candles(c, "close", p);
        vidya_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VidyaOutput, VidyaError> {
        let p = VidyaParams {
            short_period: self.short_period,
            long_period: self.long_period,
            alpha: self.alpha,
        };
        let i = VidyaInput::from_slice(d, p);
        vidya_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VidyaStream, VidyaError> {
        let p = VidyaParams {
            short_period: self.short_period,
            long_period: self.long_period,
            alpha: self.alpha,
        };
        VidyaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VidyaError {
    #[error("vidya: Input data slice is empty.")]
    EmptyInputData,
    #[error("vidya: All values are NaN.")]
    AllValuesNaN,
    #[error("vidya: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vidya: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vidya: Invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error("vidya: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vidya: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("vidya: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn vidya(input: &VidyaInput) -> Result<VidyaOutput, VidyaError> {
    vidya_with_kernel(input, Kernel::Auto)
}

pub fn vidya_with_kernel(input: &VidyaInput, kernel: Kernel) -> Result<VidyaOutput, VidyaError> {
    let data: &[f64] = match &input.data {
        VidyaData::Candles { candles, source } => source_slice(candles, source),
        VidyaData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(VidyaError::EmptyInputData);
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();
    let alpha = input.get_alpha();

    if short_period < 2 {
        return Err(VidyaError::InvalidPeriod {
            period: short_period,
            data_len: data.len(),
        });
    }
    if long_period < short_period || long_period < 2 || long_period > data.len() {
        return Err(VidyaError::InvalidPeriod {
            period: long_period,
            data_len: data.len(),
        });
    }
    if !(0.0..=1.0).contains(&alpha) || alpha.is_nan() || alpha.is_infinite() {
        return Err(VidyaError::InvalidAlpha { alpha });
    }

    let first = data
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(VidyaError::AllValuesNaN)?;
    if (data.len() - first) < long_period {
        return Err(VidyaError::NotEnoughValidData {
            needed: long_period,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    let warmup_period = first + long_period - 2;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vidya_scalar(data, short_period, long_period, alpha, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vidya_avx2(data, short_period, long_period, alpha, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vidya_avx512(data, short_period, long_period, alpha, first, &mut out)
            }
            _ => vidya_scalar(data, short_period, long_period, alpha, first, &mut out),
        }
    }

    Ok(VidyaOutput { values: out })
}

#[inline]
pub fn vidya_into_slice(
    dst: &mut [f64],
    input: &VidyaInput,
    kern: Kernel,
) -> Result<(), VidyaError> {
    let data: &[f64] = match &input.data {
        VidyaData::Candles { candles, source } => source_slice(candles, source),
        VidyaData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(VidyaError::EmptyInputData);
    }

    if dst.len() != data.len() {
        return Err(VidyaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();
    let alpha = input.get_alpha();

    if short_period < 2 {
        return Err(VidyaError::InvalidPeriod {
            period: short_period,
            data_len: data.len(),
        });
    }
    if long_period < short_period || long_period < 2 || long_period > data.len() {
        return Err(VidyaError::InvalidPeriod {
            period: long_period,
            data_len: data.len(),
        });
    }
    if !(0.0..=1.0).contains(&alpha) || alpha.is_nan() || alpha.is_infinite() {
        return Err(VidyaError::InvalidAlpha { alpha });
    }

    let first = data
        .iter()
        .position(|&x| !x.is_nan())
        .ok_or(VidyaError::AllValuesNaN)?;
    if (data.len() - first) < long_period {
        return Err(VidyaError::NotEnoughValidData {
            needed: long_period,
            valid: data.len() - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    let warmup_period = first + long_period - 2;

    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
                vidya_simd128(data, short_period, long_period, alpha, first, dst);

                for v in &mut dst[..warmup_period] {
                    *v = f64::NAN;
                }
                return Ok(());
            }
        }

        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vidya_scalar(data, short_period, long_period, alpha, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vidya_avx2(data, short_period, long_period, alpha, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vidya_avx512(data, short_period, long_period, alpha, first, dst)
            }
            _ => vidya_scalar(data, short_period, long_period, alpha, first, dst),
        }
    }

    for v in &mut dst[..warmup_period] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn vidya_into(input: &VidyaInput, out: &mut [f64]) -> Result<(), VidyaError> {
    vidya_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub unsafe fn vidya_scalar(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();

    let mut long_sum = 0.0_f64;
    let mut long_sum2 = 0.0_f64;
    let mut short_sum = 0.0_f64;
    let mut short_sum2 = 0.0_f64;

    let sp_f = short_period as f64;
    let lp_f = long_period as f64;
    let short_inv = 1.0 / sp_f;
    let long_inv = 1.0 / lp_f;

    let warm_end = first + long_period;
    let short_head = warm_end - short_period;

    for i in first..short_head {
        let x = data[i];
        long_sum += x;

        long_sum2 = x.mul_add(x, long_sum2);
    }

    for i in short_head..warm_end {
        let x = data[i];
        long_sum += x;
        long_sum2 = x.mul_add(x, long_sum2);
        short_sum += x;
        short_sum2 = x.mul_add(x, short_sum2);
    }

    let idx_m2 = warm_end - 2;
    let idx_m1 = warm_end - 1;

    let mut val = data[idx_m2];
    out[idx_m2] = val;

    if idx_m1 < len {
        let short_mean = short_sum * short_inv;
        let long_mean = long_sum * long_inv;
        let short_var = short_sum2 * short_inv - (short_mean * short_mean);
        let long_var = long_sum2 * long_inv - (long_mean * long_mean);
        let short_std = short_var.sqrt();
        let long_std = long_var.sqrt();

        let mut k = short_std / long_std;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;

        let x = data[idx_m1];

        val = (x - val).mul_add(k, val);
        out[idx_m1] = val;
    }

    for t in warm_end..len {
        let x_new = data[t];
        let x_new2 = x_new * x_new;

        long_sum += x_new;
        long_sum2 += x_new2;
        short_sum += x_new;
        short_sum2 += x_new2;

        let x_long_out = data[t - long_period];
        let x_short_out = data[t - short_period];
        long_sum -= x_long_out;

        long_sum2 = (-x_long_out).mul_add(x_long_out, long_sum2);
        short_sum -= x_short_out;
        short_sum2 = (-x_short_out).mul_add(x_short_out, short_sum2);

        let short_mean = short_sum * short_inv;
        let long_mean = long_sum * long_inv;
        let short_var = short_sum2 * short_inv - (short_mean * short_mean);
        let long_var = long_sum2 * long_inv - (long_mean * long_mean);
        let short_std = short_var.sqrt();
        let long_std = long_var.sqrt();

        let mut k = short_std / long_std;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;

        val = (x_new - val).mul_add(k, val);
        out[t] = val;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn vidya_simd128(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;

    let len = data.len();
    let mut long_sum = 0.0;
    let mut long_sum2 = 0.0;
    let mut short_sum = 0.0;
    let mut short_sum2 = 0.0;

    for i in first..(first + long_period) {
        long_sum += data[i];
        long_sum2 += data[i] * data[i];
        if i >= (first + long_period - short_period) {
            short_sum += data[i];
            short_sum2 += data[i] * data[i];
        }
    }

    let mut val = data[first + long_period - 2];
    out[first + long_period - 2] = val;

    if first + long_period - 1 < data.len() {
        let sp = short_period as f64;
        let lp = long_period as f64;
        let short_div = 1.0 / sp;
        let long_div = 1.0 / lp;
        let short_stddev =
            (short_sum2 * short_div - (short_sum * short_div) * (short_sum * short_div)).sqrt();
        let long_stddev =
            (long_sum2 * long_div - (long_sum * long_div) * (long_sum * long_div)).sqrt();
        let mut k = short_stddev / long_stddev;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;
        val = (data[first + long_period - 1] - val) * k + val;
        out[first + long_period - 1] = val;
    }

    let alpha_v = f64x2_splat(alpha);
    let sp_v = f64x2_splat(short_period as f64);
    let lp_v = f64x2_splat(long_period as f64);
    let short_div_v = f64x2_splat(1.0 / short_period as f64);
    let long_div_v = f64x2_splat(1.0 / long_period as f64);

    for i in (first + long_period)..len {
        let new_val = data[i];
        long_sum += new_val;
        long_sum2 += new_val * new_val;
        short_sum += new_val;
        short_sum2 += new_val * new_val;

        let remove_long_idx = i - long_period;
        let remove_short_idx = i - short_period;
        let remove_long = data[remove_long_idx];
        let remove_short = data[remove_short_idx];

        long_sum -= remove_long;
        long_sum2 -= remove_long * remove_long;
        short_sum -= remove_short;
        short_sum2 -= remove_short * remove_short;

        let short_mean = short_sum / short_period as f64;
        let long_mean = long_sum / long_period as f64;

        let short_variance = short_sum2 / short_period as f64 - short_mean * short_mean;
        let long_variance = long_sum2 / long_period as f64 - long_mean * long_mean;

        let short_stddev = short_variance.sqrt();
        let long_stddev = long_variance.sqrt();

        let mut k = short_stddev / long_stddev;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;
        val = (new_val - val) * k + val;
        out[i] = val;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx2(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    vidya_avx2_experimental(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn vidya_avx2_experimental(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let ptr = data.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut long_sum = 0.0_f64;
    let mut long_sum2 = 0.0_f64;
    let mut short_sum = 0.0_f64;
    let mut short_sum2 = 0.0_f64;

    let sp_f = short_period as f64;
    let lp_f = long_period as f64;
    let short_inv = 1.0 / sp_f;
    let long_inv = 1.0 / lp_f;

    let warm_end = first + long_period;
    let short_head = warm_end - short_period;

    let mut i = first;
    while i < short_head {
        let x = *ptr.add(i);
        long_sum += x;
        long_sum2 = x.mul_add(x, long_sum2);
        i += 1;
    }

    while i < warm_end {
        let x = *ptr.add(i);
        long_sum += x;
        long_sum2 = x.mul_add(x, long_sum2);
        short_sum += x;
        short_sum2 = x.mul_add(x, short_sum2);
        i += 1;
    }

    let idx_m2 = warm_end - 2;
    let idx_m1 = warm_end - 1;

    let mut val = *ptr.add(idx_m2);
    *out_ptr.add(idx_m2) = val;

    if idx_m1 < len {
        let short_mean = short_sum * short_inv;
        let long_mean = long_sum * long_inv;
        let short_var = short_sum2 * short_inv - (short_mean * short_mean);
        let long_var = long_sum2 * long_inv - (long_mean * long_mean);
        let short_std = short_var.sqrt();
        let long_std = long_var.sqrt();

        let mut k = short_std / long_std;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;

        let x = *ptr.add(idx_m1);

        val = (x - val).mul_add(k, val);
        *out_ptr.add(idx_m1) = val;
    }

    let mut t = warm_end;
    while t < len {
        let x_new = *ptr.add(t);
        let x_new2 = x_new * x_new;

        long_sum += x_new;
        long_sum2 = x_new.mul_add(x_new, long_sum2);
        short_sum += x_new;
        short_sum2 = x_new.mul_add(x_new, short_sum2);

        let x_long_out = *ptr.add(t - long_period);
        let x_short_out = *ptr.add(t - short_period);
        long_sum -= x_long_out;
        long_sum2 = (-x_long_out).mul_add(x_long_out, long_sum2);
        short_sum -= x_short_out;
        short_sum2 = (-x_short_out).mul_add(x_short_out, short_sum2);

        let short_mean = short_sum * short_inv;
        let long_mean = long_sum * long_inv;
        let short_var = short_sum2 * short_inv - (short_mean * short_mean);
        let long_var = long_sum2 * long_inv - (long_mean * long_mean);
        let short_std = short_var.sqrt();
        let long_std = long_var.sqrt();

        let mut k = short_std / long_std;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;

        val = (x_new - val).mul_add(k, val);
        *out_ptr.add(t) = val;

        t += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx512(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    if long_period <= 32 {
        vidya_avx512_short(data, short_period, long_period, alpha, first, out)
    } else {
        vidya_avx512_long(data, short_period, long_period, alpha, first, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx512_short(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx512_long(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    alpha: f64,
    first: usize,
    out: &mut [f64],
) {
    vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[derive(Debug, Clone)]
pub struct VidyaStream {
    short_period: usize,
    long_period: usize,
    alpha: f64,
    long_buf: Vec<f64>,
    short_buf: Vec<f64>,
    long_sum: f64,
    long_sum2: f64,
    short_sum: f64,
    short_sum2: f64,
    head: usize,
    idx: usize,
    val: f64,
    filled: bool,
}

impl VidyaStream {
    pub fn try_new(params: VidyaParams) -> Result<Self, VidyaError> {
        let short_period = params.short_period.unwrap_or(2);
        let long_period = params.long_period.unwrap_or(5);
        let alpha = params.alpha.unwrap_or(0.2);

        if short_period < 2 || long_period < short_period || long_period < 2 {
            return Err(VidyaError::InvalidPeriod {
                period: long_period,
                data_len: 0,
            });
        }
        if !(0.0..=1.0).contains(&alpha) || alpha.is_nan() || alpha.is_infinite() {
            return Err(VidyaError::InvalidAlpha { alpha });
        }
        Ok(Self {
            short_period,
            long_period,
            alpha,
            long_buf: alloc_with_nan_prefix(long_period, long_period),
            short_buf: alloc_with_nan_prefix(short_period, short_period),
            long_sum: 0.0,
            long_sum2: 0.0,
            short_sum: 0.0,
            short_sum2: 0.0,
            head: 0,
            idx: 0,
            val: f64::NAN,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        let long_tail = self.long_buf[self.head];
        let short_idx = self.idx % self.short_period;
        let short_tail = self.short_buf[short_idx];

        let phase2_start = self.long_period - self.short_period;

        self.long_sum += x;
        self.long_sum2 = x.mul_add(x, self.long_sum2);

        if self.idx >= phase2_start {
            self.short_sum += x;
            self.short_sum2 = x.mul_add(x, self.short_sum2);
        }

        if self.idx >= self.long_period {
            self.long_sum -= long_tail;
            self.long_sum2 = (-long_tail).mul_add(long_tail, self.long_sum2);

            self.short_sum -= short_tail;
            self.short_sum2 = (-short_tail).mul_add(short_tail, self.short_sum2);
        }

        self.long_buf[self.head] = x;
        self.short_buf[short_idx] = x;

        let mut h = self.head + 1;
        if h == self.long_period {
            h = 0;
        }
        self.head = h;

        self.idx += 1;

        if self.idx < self.long_period - 1 {
            self.val = x;
            return None;
        }
        if self.idx == self.long_period - 1 {
            self.val = x;
            return Some(self.val);
        }

        let short_inv = 1.0 / (self.short_period as f64);
        let long_inv = 1.0 / (self.long_period as f64);

        let short_mean = self.short_sum * short_inv;
        let long_mean = self.long_sum * long_inv;

        let short_var = self.short_sum2 * short_inv - (short_mean * short_mean);
        let long_var = self.long_sum2 * long_inv - (long_mean * long_mean);

        let mut k = 0.0;
        if long_var > 0.0 && short_var > 0.0 {
            k = (short_var / long_var).sqrt() * self.alpha;
        }

        self.val = (x - self.val).mul_add(k, self.val);
        Some(self.val)
    }
}

#[derive(Clone, Debug)]
pub struct VidyaBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
    pub alpha: (f64, f64, f64),
}

impl Default for VidyaBatchRange {
    fn default() -> Self {
        Self {
            short_period: (2, 2, 0),
            long_period: (5, 254, 1),
            alpha: (0.2, 0.2, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VidyaBatchBuilder {
    range: VidyaBatchRange,
    kernel: Kernel,
}

impl VidyaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    #[inline]
    pub fn short_period_static(mut self, n: usize) -> Self {
        self.range.short_period = (n, n, 0);
        self
    }
    #[inline]
    pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    #[inline]
    pub fn long_period_static(mut self, n: usize) -> Self {
        self.range.long_period = (n, n, 0);
        self
    }
    #[inline]
    pub fn alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.alpha = (start, end, step);
        self
    }
    #[inline]
    pub fn alpha_static(mut self, a: f64) -> Self {
        self.range.alpha = (a, a, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VidyaBatchOutput, VidyaError> {
        vidya_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VidyaBatchOutput, VidyaError> {
        VidyaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VidyaBatchOutput, VidyaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VidyaBatchOutput, VidyaError> {
        VidyaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vidya_batch_with_kernel(
    data: &[f64],
    sweep: &VidyaBatchRange,
    k: Kernel,
) -> Result<VidyaBatchOutput, VidyaError> {
    if data.is_empty() {
        return Err(VidyaError::EmptyInputData);
    }
    let kernel = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx2Batch,
            other => other,
        },
        other if other.is_batch() => other,
        other => {
            return Err(VidyaError::InvalidKernelForBatch(other));
        }
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    vidya_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VidyaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VidyaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VidyaBatchOutput {
    pub fn row_for_params(&self, p: &VidyaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(2) == p.short_period.unwrap_or(2)
                && c.long_period.unwrap_or(5) == p.long_period.unwrap_or(5)
                && (c.alpha.unwrap_or(0.2) - p.alpha.unwrap_or(0.2)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &VidyaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VidyaBatchRange) -> Result<Vec<VidyaParams>, VidyaError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, VidyaError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.max(1);
            while x <= end {
                v.push(x);
                x = x.saturating_add(st);
            }
            if v.is_empty() {
                return Err(VidyaError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(VidyaError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, VidyaError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(VidyaError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(VidyaError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let short_periods = axis_usize(r.short_period)?;
    let long_periods = axis_usize(r.long_period)?;
    let alphas = axis_f64(r.alpha)?;

    let cap = short_periods
        .len()
        .checked_mul(long_periods.len())
        .and_then(|x| x.checked_mul(alphas.len()))
        .ok_or_else(|| VidyaError::InvalidRange {
            start: "cap".into(),
            end: "overflow".into(),
            step: "mul".into(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &sp in &short_periods {
        for &lp in &long_periods {
            for &a in &alphas {
                out.push(VidyaParams {
                    short_period: Some(sp),
                    long_period: Some(lp),
                    alpha: Some(a),
                });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn vidya_batch_slice(
    data: &[f64],
    sweep: &VidyaBatchRange,
    kern: Kernel,
) -> Result<VidyaBatchOutput, VidyaError> {
    vidya_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vidya_batch_par_slice(
    data: &[f64],
    sweep: &VidyaBatchRange,
    kern: Kernel,
) -> Result<VidyaBatchOutput, VidyaError> {
    vidya_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn vidya_batch_inner(
    data: &[f64],
    sweep: &VidyaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VidyaBatchOutput, VidyaError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(VidyaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VidyaError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(VidyaError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let _ = rows
        .checked_mul(cols)
        .ok_or_else(|| VidyaError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.long_period.unwrap() - 2)
        .collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };
    let mut values = out;

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        let sp = p.short_period.unwrap();
        let lp = p.long_period.unwrap();
        let a = p.alpha.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vidya_row_scalar(data, first, sp, lp, a, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vidya_row_avx2(data, first, sp, lp, a, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vidya_row_avx512(data, first, sp, lp, a, out_row)
            }
            _ => vidya_row_scalar(data, first, sp, lp, a, out_row),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(VidyaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn vidya_row_scalar(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    alpha: f64,
    out: &mut [f64],
) {
    vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx2(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    alpha: f64,
    out: &mut [f64],
) {
    vidya_avx2_experimental(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx512(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    alpha: f64,
    out: &mut [f64],
) {
    if long_period <= 32 {
        vidya_row_avx512_short(data, first, short_period, long_period, alpha, out)
    } else {
        vidya_row_avx512_long(data, first, short_period, long_period, alpha, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx512_short(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    alpha: f64,
    out: &mut [f64],
) {
    vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx512_long(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    alpha: f64,
    out: &mut [f64],
) {
    vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_vidya_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = VidyaParams {
            short_period: None,
            long_period: Some(10),
            alpha: None,
        };
        let input_default = VidyaInput::from_candles(&candles, "close", default_params);
        let output_default = vidya_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vidya_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close_prices = &candles.close;

        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_candles(&candles, "close", params);
        let vidya_result = vidya_with_kernel(&input, kernel)?;
        assert_eq!(vidya_result.values.len(), close_prices.len());

        if vidya_result.values.len() >= 5 {
            let expected_last_five = [
                59553.42785306692,
                59503.60445032524,
                59451.72283651444,
                59413.222561244685,
                59239.716526894175,
            ];
            let start_index = vidya_result.values.len() - 5;
            let result_last_five = &vidya_result.values[start_index..];
            for (i, &value) in result_last_five.iter().enumerate() {
                let expected_value = expected_last_five[i];
                assert!(
                    (value - expected_value).abs() < 1e-1,
                    "VIDYA mismatch at index {}: expected {}, got {}",
                    i,
                    expected_value,
                    value
                );
            }
        }
        Ok(())
    }

    fn check_vidya_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VidyaInput::with_default_candles(&candles);
        match input.data {
            VidyaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected VidyaData::Candles"),
        }
        let output = vidya_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vidya_invalid_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = VidyaParams {
            short_period: Some(0),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_slice(&data, params);
        let result = vidya_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for invalid short period");
        Ok(())
    }

    fn check_vidya_exceeding_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_slice(&data, params);
        let result = vidya_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for period > data.len()");
        Ok(())
    }

    fn check_vidya_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0, 43.0];
        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_slice(&data, params);
        let result = vidya_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for data smaller than long period"
        );
        Ok(())
    }

    fn check_vidya_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let first_input = VidyaInput::from_candles(&candles, "close", first_params);
        let first_result = vidya_with_kernel(&first_input, kernel)?;

        let second_params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let second_input = VidyaInput::from_slice(&first_result.values, second_params);
        let second_result = vidya_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_vidya_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_candles(&candles, "close", params);
        let vidya_result = vidya_with_kernel(&input, kernel)?;
        if vidya_result.values.len() > 10 {
            for i in 10..vidya_result.values.len() {
                assert!(!vidya_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_vidya_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let short_period = 2;
        let long_period = 5;
        let alpha = 0.2;

        let params = VidyaParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
            alpha: Some(alpha),
        };
        let input = VidyaInput::from_candles(&candles, "close", params.clone());
        let batch_output = vidya_with_kernel(&input, kernel)?.values;

        let mut stream = VidyaStream::try_new(params.clone())?;
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
                diff < 1e-3,
                "[{}] VIDYA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_vidya_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            VidyaParams::default(),
            VidyaParams {
                short_period: Some(1),
                long_period: Some(2),
                alpha: Some(0.1),
            },
            VidyaParams {
                short_period: Some(2),
                long_period: Some(3),
                alpha: Some(0.2),
            },
            VidyaParams {
                short_period: Some(2),
                long_period: Some(5),
                alpha: Some(0.5),
            },
            VidyaParams {
                short_period: Some(3),
                long_period: Some(7),
                alpha: Some(0.3),
            },
            VidyaParams {
                short_period: Some(4),
                long_period: Some(10),
                alpha: Some(0.2),
            },
            VidyaParams {
                short_period: Some(5),
                long_period: Some(20),
                alpha: Some(0.4),
            },
            VidyaParams {
                short_period: Some(10),
                long_period: Some(30),
                alpha: Some(0.2),
            },
            VidyaParams {
                short_period: Some(15),
                long_period: Some(50),
                alpha: Some(0.3),
            },
            VidyaParams {
                short_period: Some(20),
                long_period: Some(100),
                alpha: Some(0.2),
            },
            VidyaParams {
                short_period: Some(50),
                long_period: Some(200),
                alpha: Some(0.1),
            },
            VidyaParams {
                short_period: Some(2),
                long_period: Some(10),
                alpha: Some(0.8),
            },
            VidyaParams {
                short_period: Some(3),
                long_period: Some(15),
                alpha: Some(1.0),
            },
            VidyaParams {
                short_period: Some(1),
                long_period: Some(100),
                alpha: Some(0.01),
            },
            VidyaParams {
                short_period: Some(99),
                long_period: Some(100),
                alpha: Some(0.99),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = VidyaInput::from_candles(&candles, "close", params.clone());
            let output = vidya_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vidya_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vidya_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=20).prop_flat_map(|short_period| {
            let long_min = (short_period + 1).max(2);
            let long_max = 100.min(long_min + 50);

            (long_min..=long_max).prop_flat_map(move |long_period| {
                let data_len = long_period.max(10)..400;

                (
                    prop::collection::vec(
                        (-0.05f64..0.05f64).prop_filter("finite", |x| x.is_finite()),
                        data_len,
                    )
                    .prop_map(|returns| {
                        let mut prices = Vec::with_capacity(returns.len());
                        let mut price = 100.0;

                        for (i, ret) in returns.iter().enumerate() {
                            let volatility_factor = if (i / 20) % 2 == 0 { 0.5 } else { 2.0 };
                            price *= 1.0 + (ret * volatility_factor);
                            prices.push(price);
                        }
                        prices
                    }),
                    Just(short_period),
                    Just(long_period),
                    0.01f64..1.0f64,
                )
            })
        });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, short_period, long_period, alpha)| {
				let params = VidyaParams {
					short_period: Some(short_period),
					long_period: Some(long_period),
					alpha: Some(alpha),
				};
				let input = VidyaInput::from_slice(&data, params.clone());


				let VidyaOutput { values: out } = vidya_with_kernel(&input, kernel).unwrap();
				let VidyaOutput { values: ref_out } = vidya_with_kernel(&input, Kernel::Scalar).unwrap();


                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(y.to_bits() == r.to_bits(),
                            "[{}] finite/NaN mismatch at idx {}: {} vs {}", test_name, i, y, r);
                        continue;
                    }

	                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
	                    prop_assert!(


	                        (y - r).abs() <= 5e-8 || ulp_diff <= 4,
	                        "[{}] kernel mismatch at idx {}: {} vs {} (ULP={})",
	                        test_name, i, y, r, ulp_diff
	                    );
	                }


				let first = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
				let first_valid_idx = if first + long_period >= 2 {
					first + long_period - 2
				} else {
					0
				};


				for i in 0..first_valid_idx.min(data.len()) {
					prop_assert!(out[i].is_nan(),
						"[{}] Expected NaN during warmup at idx {}, got {}", test_name, i, out[i]);
				}


				if first_valid_idx < data.len() {
					prop_assert!(!out[first_valid_idx].is_nan(),
						"[{}] Expected valid value at first_valid_idx {}, got NaN", test_name, first_valid_idx);
				}


				let warmup_end = first + long_period - 2;


				if data.len() > warmup_end + 1 {
					let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
					let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

					let range = data_max - data_min;


					let alpha_factor = 1.0 + alpha * 2.0;
					let margin = if range < 1.0 {

						let avg_magnitude = (data_max.abs() + data_min.abs()) / 2.0;
						avg_magnitude * 0.3 * alpha_factor
					} else {


						range * 0.5 * alpha_factor
					};

					for i in (warmup_end + 1)..data.len() {
						let y = out[i];
						if y.is_finite() {
							prop_assert!(
								y >= data_min - margin && y <= data_max + margin,
								"[{}] Output {} at idx {} outside reasonable range [{}, {}] (alpha={:.3})",
								test_name, y, i, data_min - margin, data_max + margin, alpha
							);
						}
					}
				}


				if alpha < 0.05 && data.len() > warmup_end + 10 {

					let vidya_section = &out[(warmup_end + 1)..];
					if vidya_section.len() > 2 {

						for window in vidya_section.windows(2) {
							let change_ratio = (window[1] - window[0]).abs() / window[0].abs().max(1e-10);
							prop_assert!(
								change_ratio < 0.1,
								"[{}] With alpha={}, VIDYA should be stable but found large change ratio {}",
								test_name, alpha, change_ratio
							);
						}
					}
				}


				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > warmup_end + 1 {

					let constant_value = data[0];
					for i in (warmup_end + 1)..data.len() {
						prop_assert!(
							(out[i] - constant_value).abs() <= 1e-9,
							"[{}] Constant input should produce constant output, got {} expected {}",
							test_name, out[i], constant_value
						);
					}
				}


				if alpha >= 0.05 && data.len() > warmup_end + 20 {

					let mut same_direction_count = 0;
					let mut total_movements = 0;
					let mut frozen_periods = 0;

					for i in (warmup_end + 1)..data.len() {
						let price_change = data[i] - data[i - 1];
						let vidya_change = out[i] - out[i - 1];


						if price_change.abs() > 1e-6 && vidya_change.abs() <= 1e-10 {
							frozen_periods += 1;
						}

						else if price_change.abs() > 1e-6 && vidya_change.abs() > 1e-10 {
							total_movements += 1;
							if price_change.signum() == vidya_change.signum() {
								same_direction_count += 1;
							}
						}
					}


					if total_movements > 10 && frozen_periods < (data.len() - warmup_end) / 2 {
						let direction_ratio = same_direction_count as f64 / total_movements as f64;
						prop_assert!(
							direction_ratio >= 0.40,
							"[{}] VIDYA should generally follow price direction when moving, but only followed {:.1}% of the time (frozen for {} periods)",
							test_name, direction_ratio * 100.0, frozen_periods
						);
					}
				}


				for (i, &val) in out.iter().enumerate() {
					if val.is_finite() {
						let bits = val.to_bits();
						prop_assert!(
							bits != 0x11111111_11111111 &&
							bits != 0x22222222_22222222 &&
							bits != 0x33333333_33333333,
							"[{}] Found poison value {} (0x{:016X}) at index {}",
							test_name, val, bits, i
						);
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }

    macro_rules! generate_all_vidya_tests {
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

    generate_all_vidya_tests!(
        check_vidya_partial_params,
        check_vidya_accuracy,
        check_vidya_default_candles,
        check_vidya_invalid_params,
        check_vidya_exceeding_data_length,
        check_vidya_very_small_data_set,
        check_vidya_reinput,
        check_vidya_nan_handling,
        check_vidya_streaming,
        check_vidya_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vidya_tests!(check_vidya_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = VidyaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = VidyaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59553.42785306692,
            59503.60445032524,
            59451.72283651444,
            59413.222561244685,
            59239.716526894175,
        ];
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
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 5, 1, 5, 10, 1, 0.2, 0.2, 0.0),
            (2, 10, 2, 10, 30, 5, 0.1, 0.5, 0.1),
            (5, 20, 5, 30, 60, 10, 0.2, 0.2, 0.0),
            (10, 30, 10, 50, 100, 25, 0.3, 0.3, 0.0),
            (2, 2, 0, 5, 50, 5, 0.1, 0.9, 0.2),
            (5, 15, 5, 20, 20, 0, 0.2, 0.8, 0.3),
            (1, 3, 1, 4, 8, 2, 0.5, 0.5, 0.0),
            (20, 50, 15, 100, 200, 50, 0.1, 0.3, 0.1),
            (2, 2, 0, 3, 3, 0, 0.1, 1.0, 0.1),
        ];

        for (cfg_idx, &(s_start, s_end, s_step, l_start, l_end, l_step, a_start, a_end, a_step)) in
            test_configs.iter().enumerate()
        {
            let output = VidyaBatchBuilder::new()
                .kernel(kernel)
                .short_period_range(s_start, s_end, s_step)
                .long_period_range(l_start, l_end, l_step)
                .alpha_range(a_start, a_end, a_step)
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
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[inline(always)]
pub fn vidya_batch_inner_into(
    data: &[f64],
    sweep: &VidyaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VidyaParams>, VidyaError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(VidyaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VidyaError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(VidyaError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| VidyaError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;
    if out.len() < expected {
        return Err(VidyaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.long_period.unwrap() - 2)
        .collect();

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        out[row_start..row_start + warmup].fill(f64::NAN);
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        let sp = p.short_period.unwrap();
        let lp = p.long_period.unwrap();
        let a = p.alpha.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vidya_row_scalar(data, first, sp, lp, a, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vidya_row_avx2(data, first, sp, lp, a, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vidya_row_avx512(data, first, sp, lp, a, out_row)
            }
            _ => vidya_row_scalar(data, first, sp, lp, a, out_row),
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
