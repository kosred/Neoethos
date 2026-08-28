use crate::indicators::moving_averages::ema::{EmaError, EmaInput, EmaParams, ema};
use crate::indicators::moving_averages::sma::{SmaError, SmaInput, SmaParams, sma};
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
use thiserror::Error;

impl<'a> AsRef<[f64]> for WavetrendInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WavetrendData::Slice(slice) => slice,
            WavetrendData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WavetrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WavetrendOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavetrendOutputField {
    Wt1,
    Wt2,
    WtDiff,
}

#[derive(Debug, Clone)]
pub struct WavetrendParams {
    pub channel_length: Option<usize>,
    pub average_length: Option<usize>,
    pub ma_length: Option<usize>,
    pub factor: Option<f64>,
}

impl Default for WavetrendParams {
    fn default() -> Self {
        Self {
            channel_length: Some(9),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WavetrendInput<'a> {
    pub data: WavetrendData<'a>,
    pub params: WavetrendParams,
}

impl<'a> WavetrendInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hlc3", WavetrendParams::default())
    }
    #[inline]
    pub fn get_channel_length(&self) -> usize {
        self.params.channel_length.unwrap_or(9)
    }
    #[inline]
    pub fn get_average_length(&self) -> usize {
        self.params.average_length.unwrap_or(12)
    }
    #[inline]
    pub fn get_ma_length(&self) -> usize {
        self.params.ma_length.unwrap_or(3)
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.015)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WavetrendBuilder {
    channel_length: Option<usize>,
    average_length: Option<usize>,
    ma_length: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for WavetrendBuilder {
    fn default() -> Self {
        Self {
            channel_length: None,
            average_length: None,
            ma_length: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WavetrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn channel_length(mut self, n: usize) -> Self {
        self.channel_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn average_length(mut self, n: usize) -> Self {
        self.average_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_length(mut self, n: usize) -> Self {
        self.ma_length = Some(n);
        self
    }
    #[inline(always)]
    pub fn factor(mut self, f: f64) -> Self {
        self.factor = Some(f);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<WavetrendOutput, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        let i = WavetrendInput::from_candles(c, "hlc3", p);
        wavetrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WavetrendOutput, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        let i = WavetrendInput::from_slice(d, p);
        wavetrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WavetrendStream, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        WavetrendStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum WavetrendError {
    #[error("wavetrend: Empty data provided.")]
    EmptyInputData,
    #[error("wavetrend: Empty data provided.")]
    EmptyData,
    #[error("wavetrend: All values are NaN.")]
    AllValuesNaN,
    #[error("wavetrend: Invalid channel_length = {channel_length}, data length = {data_len}")]
    InvalidChannelLen {
        channel_length: usize,
        data_len: usize,
    },
    #[error("wavetrend: Invalid average_length = {average_length}, data length = {data_len}")]
    InvalidAverageLen {
        average_length: usize,
        data_len: usize,
    },
    #[error("wavetrend: Invalid ma_length = {ma_length}, data length = {data_len}")]
    InvalidMaLen { ma_length: usize, data_len: usize },
    #[error("wavetrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wavetrend: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("wavetrend: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputSliceLengthMismatch { expected: usize, got: usize },
    #[error("wavetrend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("wavetrend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("wavetrend: EMA error {0}")]
    EmaError(#[from] EmaError),
    #[error("wavetrend: SMA error {0}")]
    SmaError(#[from] SmaError),
}

#[inline]
pub fn wavetrend(input: &WavetrendInput) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_with_kernel(input, Kernel::Auto)
}

pub fn wavetrend_with_kernel(
    input: &WavetrendInput,
    kernel: Kernel,
) -> Result<WavetrendOutput, WavetrendError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(WavetrendError::EmptyInputData);
    }
    let channel_len = input.get_channel_length();
    let average_len = input.get_average_length();
    let ma_len = input.get_ma_length();
    let factor = input.get_factor();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;
    let needed = *[channel_len, average_len, ma_len].iter().max().unwrap();
    let valid = data.len() - first;

    if channel_len == 0 || channel_len > data.len() {
        return Err(WavetrendError::InvalidChannelLen {
            channel_length: channel_len,
            data_len: data.len(),
        });
    }
    if average_len == 0 || average_len > data.len() {
        return Err(WavetrendError::InvalidAverageLen {
            average_length: average_len,
            data_len: data.len(),
        });
    }
    if ma_len == 0 || ma_len > data.len() {
        return Err(WavetrendError::InvalidMaLen {
            ma_length: ma_len,
            data_len: data.len(),
        });
    }
    if valid < needed {
        return Err(WavetrendError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),

        Kernel::Avx2 | Kernel::Avx512 => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                wavetrend_avx2(data, channel_len, average_len, ma_len, factor, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wavetrend_avx512(data, channel_len, average_len, ma_len, factor, first)
            }
            _ => unreachable!(),
        }
    }
}

fn wavetrend_kernel_dispatch(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    kernel: Kernel,
) -> Result<WavetrendOutput, WavetrendError> {
    let warmup_period = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    let mut wt1_final = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut wt2_final = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut diff_final = alloc_with_nan_prefix(data.len(), warmup_period);

    wavetrend_compute_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        &mut wt1_final,
        &mut wt2_final,
        &mut diff_final,
        kernel,
    )?;

    Ok(WavetrendOutput {
        wt1: wt1_final,
        wt2: wt2_final,
        wt_diff: diff_final,
    })
}

pub fn wavetrend_scalar(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_kernel_dispatch(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        Kernel::Scalar,
    )
}

use std::collections::VecDeque;

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx2(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    let warmup_period = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    let mut wt1_out = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut wt2_out = alloc_with_nan_prefix(data.len(), warmup_period);
    let mut diff_out = alloc_with_nan_prefix(data.len(), warmup_period);

    wavetrend_fused_avx2_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        &mut wt1_out,
        &mut wt2_out,
        &mut diff_out,
    );

    Ok(WavetrendOutput {
        wt1: wt1_out,
        wt2: wt2_out,
        wt_diff: diff_out,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn wavetrend_fused_avx2_into(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    warmup_period: usize,
    dst_wt1: &mut [f64],
    dst_wt2: &mut [f64],
    dst_wt_diff: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let alpha_ch = 2.0 / (channel_len as f64 + 1.0);
    let beta_ch = 1.0 - alpha_ch;
    let alpha_avg = 2.0 / (average_len as f64 + 1.0);
    let beta_avg = 1.0 - alpha_avg;

    let mut esa_state: f64 = f64::NAN;
    let mut de_state: f64 = f64::NAN;
    let mut wt1_state: f64 = f64::NAN;
    let mut esa_seeded = false;
    let mut de_seeded = false;
    let mut wt1_seeded = false;

    let mut ring_vals = vec![f64::NAN; ma_len];
    let mut ring_mask = vec![0u8; ma_len];
    let mut head = 0usize;
    let mut sma_sum = 0.0f64;
    let mut sma_count = 0usize;
    let inv_ma = 1.0 / (ma_len as f64);

    for idx in first..n {
        let x = data[idx];
        let mut wt1_i = f64::NAN;
        let mut wt2_i = f64::NAN;

        if x.is_finite() {
            if !esa_seeded {
                esa_state = x;
                esa_seeded = true;
            } else {
                esa_state = x.mul_add(alpha_ch, beta_ch * esa_state);
            }

            let abs_diff = (x - esa_state).abs();
            if !de_seeded {
                de_state = abs_diff;
                de_seeded = true;
            } else {
                de_state = abs_diff.mul_add(alpha_ch, beta_ch * de_state);
            }

            let den = factor * de_state;
            if den != 0.0 && den.is_finite() && esa_state.is_finite() {
                let ci = (x - esa_state) / den;
                if ci.is_finite() {
                    if !wt1_seeded {
                        wt1_state = ci;
                        wt1_seeded = true;
                    } else {
                        wt1_state = ci.mul_add(alpha_avg, beta_avg * wt1_state);
                    }
                    wt1_i = wt1_state;
                }
            }
        }

        if ring_mask[head] != 0 {
            sma_sum -= ring_vals[head];
            sma_count -= 1;
        }
        if wt1_i.is_finite() {
            ring_vals[head] = wt1_i;
            ring_mask[head] = 1;
            sma_sum += wt1_i;
            sma_count += 1;
        } else {
            ring_vals[head] = f64::NAN;
            ring_mask[head] = 0;
        }
        head += 1;
        if head == ma_len {
            head = 0;
        }
        if sma_count == ma_len {
            wt2_i = sma_sum * inv_ma;
        }

        if idx >= warmup_period {
            dst_wt1[idx] = wt1_i;
            dst_wt2[idx] = wt2_i;
            dst_wt_diff[idx] = if wt1_i.is_finite() && wt2_i.is_finite() {
                wt2_i - wt1_i
            } else {
                f64::NAN
            };
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    if channel_len <= 32 {
        wavetrend_avx512_short(data, channel_len, average_len, ma_len, factor, first)
    } else {
        wavetrend_avx512_long(data, channel_len, average_len, ma_len, factor, first)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512_short(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_kernel_dispatch(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        Kernel::Avx512,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512_long(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_kernel_dispatch(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        Kernel::Avx512,
    )
}

#[inline(always)]
fn wavetrend_prepare<'a>(
    input: &'a WavetrendInput,
) -> Result<(&'a [f64], usize, usize, usize, f64, usize, usize), WavetrendError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(WavetrendError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;
    let channel_len = input.get_channel_length();
    let average_len = input.get_average_length();
    let ma_len = input.get_ma_length();
    let factor = input.get_factor();

    if channel_len == 0 || channel_len > data.len() {
        return Err(WavetrendError::InvalidChannelLen {
            channel_length: channel_len,
            data_len: data.len(),
        });
    }
    if average_len == 0 || average_len > data.len() {
        return Err(WavetrendError::InvalidAverageLen {
            average_length: average_len,
            data_len: data.len(),
        });
    }
    if ma_len == 0 || ma_len > data.len() {
        return Err(WavetrendError::InvalidMaLen {
            ma_length: ma_len,
            data_len: data.len(),
        });
    }

    let max_period = channel_len.max(average_len).max(ma_len);
    if data.len() - first < max_period {
        return Err(WavetrendError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }

    let warmup_period = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    Ok((
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
    ))
}

#[inline(always)]
fn wavetrend_compute_into(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    warmup_period: usize,
    dst_wt1: &mut [f64],
    dst_wt2: &mut [f64],
    dst_wt_diff: &mut [f64],
    kernel: Kernel,
) -> Result<(), WavetrendError> {
    if matches!(kernel.to_non_batch(), Kernel::Scalar) {
        let n = data.len();
        if n == 0 {
            return Ok(());
        }

        let alpha_ch = 2.0 / (channel_len as f64 + 1.0);
        let beta_ch = 1.0 - alpha_ch;
        let alpha_avg = 2.0 / (average_len as f64 + 1.0);
        let beta_avg = 1.0 - alpha_avg;

        let mut esa_state: f64 = f64::NAN;
        let mut de_state: f64 = f64::NAN;
        let mut wt1_state: f64 = f64::NAN;
        let mut esa_seeded = false;
        let mut de_seeded = false;
        let mut wt1_seeded = false;

        let mut ring_vals = vec![f64::NAN; ma_len];
        let mut ring_mask = vec![0u8; ma_len];
        let mut head = 0usize;
        let mut sma_sum = 0.0f64;
        let mut sma_count = 0usize;

        for idx in first..n {
            let x = data[idx];

            let mut wt1_i = f64::NAN;
            let mut wt2_i = f64::NAN;

            if x.is_finite() {
                if !esa_seeded {
                    esa_state = x;
                    esa_seeded = true;
                } else {
                    esa_state = alpha_ch * x + beta_ch * esa_state;
                }

                let abs_diff = (x - esa_state).abs();
                if !de_seeded {
                    de_state = abs_diff;
                    de_seeded = true;
                } else {
                    de_state = alpha_ch * abs_diff + beta_ch * de_state;
                }

                let den = factor * de_state;
                if den != 0.0 && den.is_finite() && esa_state.is_finite() {
                    let ci = (x - esa_state) / den;
                    if ci.is_finite() {
                        if !wt1_seeded {
                            wt1_state = ci;
                            wt1_seeded = true;
                        } else {
                            wt1_state = alpha_avg * ci + beta_avg * wt1_state;
                        }
                        wt1_i = wt1_state;
                    }
                }
            }

            if ma_len > 0 {
                if ring_mask[head] != 0 {
                    sma_sum -= ring_vals[head];
                    sma_count -= 1;
                }

                if wt1_i.is_finite() {
                    ring_vals[head] = wt1_i;
                    ring_mask[head] = 1;
                    sma_sum += wt1_i;
                    sma_count += 1;
                } else {
                    ring_vals[head] = f64::NAN;
                    ring_mask[head] = 0;
                }
                head += 1;
                if head == ma_len {
                    head = 0;
                }

                if sma_count == ma_len {
                    wt2_i = sma_sum / (ma_len as f64);
                }
            }

            if idx >= warmup_period {
                dst_wt1[idx] = wt1_i;
                dst_wt2[idx] = wt2_i;
                dst_wt_diff[idx] = if wt1_i.is_finite() && wt2_i.is_finite() {
                    wt2_i - wt1_i
                } else {
                    f64::NAN
                };
            }
        }

        return Ok(());
    }

    let data_valid = &data[first..];
    let simd_kernel = kernel.to_non_batch();

    if data_valid.len() <= STACK_LIMIT {
        let mut esa_buf = [0.0f64; STACK_LIMIT];
        let mut de_buf = [0.0f64; STACK_LIMIT];
        let mut ci_buf = [0.0f64; STACK_LIMIT];
        let mut wt1_buf = [0.0f64; STACK_LIMIT];
        let mut wt2_buf = [0.0f64; STACK_LIMIT];

        let esa = &mut esa_buf[..data_valid.len()];
        let de = &mut de_buf[..data_valid.len()];
        let ci = &mut ci_buf[..data_valid.len()];
        let wt1 = &mut wt1_buf[..data_valid.len()];
        let wt2 = &mut wt2_buf[..data_valid.len()];

        wavetrend_core_computation(
            data_valid,
            channel_len,
            average_len,
            ma_len,
            factor,
            esa,
            de,
            ci,
            wt1,
            wt2,
            simd_kernel,
        )?;

        for i in 0..data_valid.len() {
            let out_idx = i + first;
            if out_idx >= warmup_period {
                dst_wt1[out_idx] = wt1[i];
                dst_wt2[out_idx] = wt2[i];
                if !wt1[i].is_nan() && !wt2[i].is_nan() {
                    dst_wt_diff[out_idx] = wt2[i] - wt1[i];
                } else {
                    dst_wt_diff[out_idx] = f64::NAN;
                }
            }
        }
    } else {
        let mut esa = vec![0.0; data_valid.len()];
        let mut de = vec![0.0; data_valid.len()];
        let mut ci = vec![0.0; data_valid.len()];
        let mut wt1 = vec![0.0; data_valid.len()];
        let mut wt2 = vec![0.0; data_valid.len()];

        wavetrend_core_computation(
            data_valid,
            channel_len,
            average_len,
            ma_len,
            factor,
            &mut esa,
            &mut de,
            &mut ci,
            &mut wt1,
            &mut wt2,
            simd_kernel,
        )?;

        for i in 0..data_valid.len() {
            let out_idx = i + first;
            if out_idx >= warmup_period {
                dst_wt1[out_idx] = wt1[i];
                dst_wt2[out_idx] = wt2[i];
                if !wt1[i].is_nan() && !wt2[i].is_nan() {
                    dst_wt_diff[out_idx] = wt2[i] - wt1[i];
                } else {
                    dst_wt_diff[out_idx] = f64::NAN;
                }
            }
        }
    }

    Ok(())
}

#[inline(always)]
fn wavetrend_compute_output_into(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    first: usize,
    warmup_period: usize,
    dst: &mut [f64],
    field: WavetrendOutputField,
) {
    let n = data.len();
    for i in 0..warmup_period.min(n) {
        dst[i] = f64::NAN;
    }
    if n == 0 {
        return;
    }

    let alpha_ch = 2.0 / (channel_len as f64 + 1.0);
    let beta_ch = 1.0 - alpha_ch;
    let alpha_avg = 2.0 / (average_len as f64 + 1.0);
    let beta_avg = 1.0 - alpha_avg;

    let mut esa_state: f64 = f64::NAN;
    let mut de_state: f64 = f64::NAN;
    let mut wt1_state: f64 = f64::NAN;
    let mut esa_seeded = false;
    let mut de_seeded = false;
    let mut wt1_seeded = false;

    if matches!(field, WavetrendOutputField::Wt1) {
        for idx in first..n {
            let x = data[idx];
            let mut wt1_i = f64::NAN;

            if x.is_finite() {
                if !esa_seeded {
                    esa_state = x;
                    esa_seeded = true;
                } else {
                    esa_state = alpha_ch * x + beta_ch * esa_state;
                }

                let abs_diff = (x - esa_state).abs();
                if !de_seeded {
                    de_state = abs_diff;
                    de_seeded = true;
                } else {
                    de_state = alpha_ch * abs_diff + beta_ch * de_state;
                }

                let den = factor * de_state;
                if den != 0.0 && den.is_finite() && esa_state.is_finite() {
                    let ci = (x - esa_state) / den;
                    if ci.is_finite() {
                        if !wt1_seeded {
                            wt1_state = ci;
                            wt1_seeded = true;
                        } else {
                            wt1_state = alpha_avg * ci + beta_avg * wt1_state;
                        }
                        wt1_i = wt1_state;
                    }
                }
            }

            if idx >= warmup_period {
                dst[idx] = wt1_i;
            }
        }
        return;
    }

    let mut ring_vals = vec![f64::NAN; ma_len];
    let mut ring_mask = vec![0u8; ma_len];
    let mut head = 0usize;
    let mut sma_sum = 0.0f64;
    let mut sma_count = 0usize;

    for idx in first..n {
        let x = data[idx];
        let mut wt1_i = f64::NAN;
        let mut wt2_i = f64::NAN;

        if x.is_finite() {
            if !esa_seeded {
                esa_state = x;
                esa_seeded = true;
            } else {
                esa_state = alpha_ch * x + beta_ch * esa_state;
            }

            let abs_diff = (x - esa_state).abs();
            if !de_seeded {
                de_state = abs_diff;
                de_seeded = true;
            } else {
                de_state = alpha_ch * abs_diff + beta_ch * de_state;
            }

            let den = factor * de_state;
            if den != 0.0 && den.is_finite() && esa_state.is_finite() {
                let ci = (x - esa_state) / den;
                if ci.is_finite() {
                    if !wt1_seeded {
                        wt1_state = ci;
                        wt1_seeded = true;
                    } else {
                        wt1_state = alpha_avg * ci + beta_avg * wt1_state;
                    }
                    wt1_i = wt1_state;
                }
            }
        }

        if ring_mask[head] != 0 {
            sma_sum -= ring_vals[head];
            sma_count -= 1;
        }

        if wt1_i.is_finite() {
            ring_vals[head] = wt1_i;
            ring_mask[head] = 1;
            sma_sum += wt1_i;
            sma_count += 1;
        } else {
            ring_vals[head] = f64::NAN;
            ring_mask[head] = 0;
        }
        head += 1;
        if head == ma_len {
            head = 0;
        }

        if sma_count == ma_len {
            wt2_i = sma_sum / (ma_len as f64);
        }

        if idx >= warmup_period {
            dst[idx] = match field {
                WavetrendOutputField::Wt2 => wt2_i,
                WavetrendOutputField::WtDiff => {
                    if wt1_i.is_finite() && wt2_i.is_finite() {
                        wt2_i - wt1_i
                    } else {
                        f64::NAN
                    }
                }
                WavetrendOutputField::Wt1 => unreachable!(),
            };
        }
    }
}

const STACK_LIMIT: usize = 512;

#[inline(always)]
fn wavetrend_core_computation(
    data: &[f64],
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    esa: &mut [f64],
    de: &mut [f64],
    ci: &mut [f64],
    wt1: &mut [f64],
    wt2: &mut [f64],
    kernel: Kernel,
) -> Result<(), WavetrendError> {
    ema_compute_into(data, channel_len, esa);

    if data.len() <= STACK_LIMIT {
        let mut abs_diff_buf = [0.0f64; STACK_LIMIT];
        let abs_diff = &mut abs_diff_buf[..data.len()];
        compute_abs_diff(abs_diff, data, esa, kernel);
        ema_compute_into(abs_diff, channel_len, de);
    } else {
        let mut abs_diff = vec![0.0; data.len()];
        compute_abs_diff(&mut abs_diff, data, esa, kernel);
        ema_compute_into(&abs_diff, channel_len, de);
    }

    compute_ci(ci, data, esa, de, factor, kernel);

    ema_compute_into(ci, average_len, wt1);

    sma_compute_into(wt1, ma_len, wt2);

    Ok(())
}

#[inline(always)]
fn compute_abs_diff(out: &mut [f64], data: &[f64], esa: &[f64], kernel: Kernel) {
    let simd = kernel.to_non_batch();
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        match simd {
            Kernel::Avx512 => unsafe {
                absdiff_vec_avx512(out, data, esa);
                return;
            },
            Kernel::Avx2 => unsafe {
                absdiff_vec_avx2(out, data, esa);
                return;
            },
            _ => {}
        }
    }

    for i in 0..out.len() {
        out[i] = (data[i] - esa[i]).abs();
    }
}

#[inline(always)]
fn compute_ci(out: &mut [f64], data: &[f64], esa: &[f64], de: &[f64], factor: f64, kernel: Kernel) {
    let simd = kernel.to_non_batch();
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        match simd {
            Kernel::Avx512 => unsafe {
                ci_vec_avx512(out, data, esa, de, factor);
                return;
            },
            Kernel::Avx2 => unsafe {
                ci_vec_avx2(out, data, esa, de, factor);
                return;
            },
            _ => {}
        }
    }

    for i in 0..out.len() {
        let den = factor * de[i];
        if den != 0.0 && !data[i].is_nan() && !esa[i].is_nan() && !de[i].is_nan() {
            out[i] = (data[i] - esa[i]) / den;
        } else {
            out[i] = f64::NAN;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn absdiff_vec_avx2(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let n = dst.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let pd = dst.as_mut_ptr();
    let sign = _mm256_set1_pd(-0.0f64);
    let mut i = 0usize;
    while i + 4 <= n {
        let va = _mm256_loadu_pd(pa.add(i));
        let vb = _mm256_loadu_pd(pb.add(i));
        let vd = _mm256_sub_pd(va, vb);
        let vabs = _mm256_andnot_pd(sign, vd);
        _mm256_storeu_pd(pd.add(i), vabs);
        i += 4;
    }
    while i < n {
        *pd.add(i) = (*pa.add(i) - *pb.add(i)).abs();
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ci_vec_avx2(dst: &mut [f64], data: &[f64], esa: &[f64], de: &[f64], factor: f64) {
    let n = dst.len();
    let px = data.as_ptr();
    let pe = esa.as_ptr();
    let pd = de.as_ptr();
    let pr = dst.as_mut_ptr();

    let vf = _mm256_set1_pd(factor);
    let vzero = _mm256_set1_pd(0.0);
    let vnan = _mm256_set1_pd(f64::NAN);

    let mut i = 0usize;
    while i + 4 <= n {
        let vx = _mm256_loadu_pd(px.add(i));
        let ve = _mm256_loadu_pd(pe.add(i));
        let vd = _mm256_loadu_pd(pd.add(i));

        let vnum = _mm256_sub_pd(vx, ve);
        let vden = _mm256_mul_pd(vf, vd);
        let vci = _mm256_div_pd(vnum, vden);

        let ord_x = _mm256_cmp_pd(vx, vx, _CMP_ORD_Q);
        let ord_e = _mm256_cmp_pd(ve, ve, _CMP_ORD_Q);
        let ord_d = _mm256_cmp_pd(vd, vd, _CMP_ORD_Q);
        let ord_all = _mm256_and_pd(ord_x, _mm256_and_pd(ord_e, ord_d));
        let den_zero = _mm256_cmp_pd(vden, vzero, _CMP_EQ_OQ);
        let valid = _mm256_andnot_pd(den_zero, ord_all);

        let vres = _mm256_blendv_pd(vnan, vci, valid);
        _mm256_storeu_pd(pr.add(i), vres);
        i += 4;
    }
    while i < n {
        let x = *px.add(i);
        let e = *pe.add(i);
        let d = *pd.add(i);
        let den = factor * d;
        *pr.add(i) = if den != 0.0 && x.is_finite() && e.is_finite() && d.is_finite() {
            (x - e) / den
        } else {
            f64::NAN
        };
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn absdiff_vec_avx512(dst: &mut [f64], a: &[f64], b: &[f64]) {
    let n = dst.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let pd = dst.as_mut_ptr();
    let sign = _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64);
    let sign_pd = _mm512_castsi512_pd(sign);

    let mut i = 0usize;
    while i + 8 <= n {
        let va = _mm512_loadu_pd(pa.add(i));
        let vb = _mm512_loadu_pd(pb.add(i));
        let vd = _mm512_sub_pd(va, vb);
        let vabs = _mm512_andnot_pd(sign_pd, vd);
        _mm512_storeu_pd(pd.add(i), vabs);
        i += 8;
    }
    while i < n {
        *pd.add(i) = (*pa.add(i) - *pb.add(i)).abs();
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ci_vec_avx512(dst: &mut [f64], data: &[f64], esa: &[f64], de: &[f64], factor: f64) {
    let n = dst.len();
    let px = data.as_ptr();
    let pe = esa.as_ptr();
    let pdv = de.as_ptr();
    let pr = dst.as_mut_ptr();

    let vf = _mm512_set1_pd(factor);
    let vzero = _mm512_set1_pd(0.0);
    let vnan = _mm512_set1_pd(f64::NAN);

    let mut i = 0usize;
    while i + 8 <= n {
        let vx = _mm512_loadu_pd(px.add(i));
        let ve = _mm512_loadu_pd(pe.add(i));
        let vd = _mm512_loadu_pd(pdv.add(i));

        let vnum = _mm512_sub_pd(vx, ve);
        let vden = _mm512_mul_pd(vf, vd);
        let vci = _mm512_div_pd(vnum, vden);

        let ord_x = _mm512_cmp_pd_mask(vx, vx, _CMP_ORD_Q);
        let ord_e = _mm512_cmp_pd_mask(ve, ve, _CMP_ORD_Q);
        let ord_d = _mm512_cmp_pd_mask(vd, vd, _CMP_ORD_Q);
        let ord_all = ord_x & ord_e & ord_d;
        let den_zero = _mm512_cmp_pd_mask(vden, vzero, _CMP_EQ_OQ);
        let valid = ord_all & (!den_zero);

        let vres = _mm512_mask_mov_pd(vnan, valid, vci);
        _mm512_storeu_pd(pr.add(i), vres);
        i += 8;
    }
    while i < n {
        let x = *px.add(i);
        let e = *pe.add(i);
        let d = *pdv.add(i);
        let den = factor * d;
        *pr.add(i) = if den != 0.0 && x.is_finite() && e.is_finite() && d.is_finite() {
            (x - e) / den
        } else {
            f64::NAN
        };
        i += 1;
    }
}

#[inline(always)]
fn ema_compute_into(data: &[f64], period: usize, out: &mut [f64]) {
    if period == 0 || data.is_empty() {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;

    let mut ema_val = f64::NAN;
    for i in 0..data.len() {
        if !data[i].is_nan() {
            if ema_val.is_nan() {
                ema_val = data[i];
            } else {
                ema_val = alpha * data[i] + beta * ema_val;
            }
            out[i] = ema_val;
        } else {
            out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn sma_compute_into(data: &[f64], period: usize, out: &mut [f64]) {
    if period == 0 || data.is_empty() {
        return;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..out.len() {
        out[i] = f64::NAN;
    }

    for i in 0..data.len() {
        if !data[i].is_nan() {
            sum += data[i];
            count += 1;

            if i >= period {
                if !data[i - period].is_nan() {
                    sum -= data[i - period];
                    count -= 1;
                }
            }

            if count >= period {
                out[i] = sum / period as f64;
            }
        }
    }
}

#[inline]
pub fn wavetrend_into_slice(
    dst_wt1: &mut [f64],
    dst_wt2: &mut [f64],
    dst_wt_diff: &mut [f64],
    input: &WavetrendInput,
    kern: Kernel,
) -> Result<(), WavetrendError> {
    let (data, channel_len, average_len, ma_len, factor, first, warmup_period) =
        wavetrend_prepare(input)?;

    if dst_wt1.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_wt1.len(),
        });
    }
    if dst_wt2.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_wt2.len(),
        });
    }
    if dst_wt_diff.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_wt_diff.len(),
        });
    }

    for i in 0..warmup_period.min(data.len()) {
        dst_wt1[i] = f64::NAN;
        dst_wt2[i] = f64::NAN;
        dst_wt_diff[i] = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    };

    wavetrend_compute_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        dst_wt1,
        dst_wt2,
        dst_wt_diff,
        chosen,
    )?;

    Ok(())
}

#[inline]
pub fn wavetrend_output_into_slice(
    dst: &mut [f64],
    input: &WavetrendInput,
    kern: Kernel,
    field: WavetrendOutputField,
) -> Result<(), WavetrendError> {
    let _ = kern;
    let (data, channel_len, average_len, ma_len, factor, first, warmup_period) =
        wavetrend_prepare(input)?;

    if dst.len() != data.len() {
        return Err(WavetrendError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    wavetrend_compute_output_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup_period,
        dst,
        field,
    );

    Ok(())
}

#[derive(Clone, Debug)]
pub struct WavetrendStream {
    pub channel_length: usize,
    pub average_length: usize,
    pub ma_length: usize,
    pub factor: f64,

    esa_buf: VecDeque<f64>,
    last_esa: Option<f64>,
    alpha_ch: f64,

    beta_ch: f64,

    de_buf: VecDeque<f64>,
    last_de: Option<f64>,

    ci_buf: VecDeque<f64>,
    last_wt1: Option<f64>,
    alpha_avg: f64,

    beta_avg: f64,

    wt1_buf: VecDeque<f64>,
    running_sum: f64,

    sma_count: usize,

    inv_ma: f64,

    pub history: Vec<f64>,
}

impl WavetrendStream {
    pub fn try_new(p: WavetrendParams) -> Result<Self, WavetrendError> {
        let channel_length = p.channel_length.unwrap_or(9);
        let average_length = p.average_length.unwrap_or(12);
        let ma_length = p.ma_length.unwrap_or(3);
        let factor = p.factor.unwrap_or(0.015);

        if channel_length == 0 {
            return Err(WavetrendError::InvalidChannelLen {
                channel_length,
                data_len: 0,
            });
        }
        if average_length == 0 {
            return Err(WavetrendError::InvalidAverageLen {
                average_length,
                data_len: 0,
            });
        }
        if ma_length == 0 {
            return Err(WavetrendError::InvalidMaLen {
                ma_length,
                data_len: 0,
            });
        }

        let alpha_ch = 2.0 / (channel_length as f64 + 1.0);
        let alpha_avg = 2.0 / (average_length as f64 + 1.0);

        Ok(Self {
            channel_length,
            average_length,
            ma_length,
            factor,

            esa_buf: VecDeque::with_capacity(channel_length),
            last_esa: None,
            alpha_ch,
            beta_ch: 1.0 - alpha_ch,

            de_buf: VecDeque::with_capacity(channel_length),
            last_de: None,

            ci_buf: VecDeque::with_capacity(average_length),
            last_wt1: None,
            alpha_avg,
            beta_avg: 1.0 - alpha_avg,

            wt1_buf: VecDeque::with_capacity(ma_length),
            running_sum: 0.0,
            sma_count: 0,
            inv_ma: 1.0 / (ma_length as f64),

            history: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<(f64, f64, f64)> {
        self.history.push(price);

        let mut wt1_val = f64::NAN;

        if price.is_finite() {
            if let Some(prev) = self.last_esa {
                let new_esa = ema_step(prev, price, self.alpha_ch, self.beta_ch);
                self.last_esa = Some(new_esa);
            } else {
                self.last_esa = Some(price);
            }

            if let Some(esa_now) = self.last_esa {
                let abs_diff = fast_abs_f64(price - esa_now);
                if let Some(prev_de) = self.last_de {
                    let new_de = ema_step(prev_de, abs_diff, self.alpha_ch, self.beta_ch);
                    self.last_de = Some(new_de);
                } else {
                    self.last_de = Some(abs_diff);
                }
            }

            if let (Some(esa_now), Some(de_now)) = (self.last_esa, self.last_de) {
                let den = self.factor * de_now;
                if den != 0.0 && den.is_finite() && esa_now.is_finite() {
                    let ci = (price - esa_now) / den;
                    if ci.is_finite() {
                        if let Some(prev_wt1) = self.last_wt1 {
                            let new_wt1 = ema_step(prev_wt1, ci, self.alpha_avg, self.beta_avg);
                            self.last_wt1 = Some(new_wt1);
                        } else {
                            self.last_wt1 = Some(ci);
                        }
                        if let Some(v) = self.last_wt1 {
                            wt1_val = v;
                        }
                    }
                }
            }
        }

        if self.wt1_buf.len() == self.ma_length {
            if let Some(leaving) = self.wt1_buf.pop_front() {
                if leaving.is_finite() {
                    self.running_sum -= leaving;
                    if self.sma_count > 0 {
                        self.sma_count -= 1;
                    }
                }
            }
        }

        self.wt1_buf.push_back(wt1_val);
        if wt1_val.is_finite() {
            self.running_sum += wt1_val;
            self.sma_count += 1;
        }

        if self.wt1_buf.len() == self.ma_length && self.sma_count == self.ma_length {
            let wt1 = wt1_val;
            let wt2 = self.running_sum * self.inv_ma;
            let diff = wt2 - wt1;
            Some((wt1, wt2, diff))
        } else {
            None
        }
    }
}

#[inline(always)]
fn ema_step(prev: f64, x: f64, alpha: f64, beta: f64) -> f64 {
    x.mul_add(alpha, beta * prev)
}

#[inline(always)]
fn fast_abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
}

#[derive(Clone, Debug)]
pub struct WavetrendBatchRange {
    pub channel_length: (usize, usize, usize),
    pub average_length: (usize, usize, usize),
    pub ma_length: (usize, usize, usize),
    pub factor: (f64, f64, f64),
}

impl Default for WavetrendBatchRange {
    fn default() -> Self {
        Self {
            channel_length: (9, 9, 0),
            average_length: (12, 261, 1),
            ma_length: (3, 3, 0),
            factor: (0.015, 0.015, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WavetrendBatchBuilder {
    range: WavetrendBatchRange,
    kernel: Kernel,
}

impl WavetrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn channel_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.channel_length = (start, end, step);
        self
    }
    pub fn channel_static(mut self, x: usize) -> Self {
        self.range.channel_length = (x, x, 0);
        self
    }
    pub fn avg_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.average_length = (start, end, step);
        self
    }
    pub fn avg_static(mut self, x: usize) -> Self {
        self.range.average_length = (x, x, 0);
        self
    }
    pub fn ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_length = (start, end, step);
        self
    }
    pub fn ma_static(mut self, x: usize) -> Self {
        self.range.ma_length = (x, x, 0);
        self
    }
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<WavetrendBatchOutput, WavetrendError> {
        wavetrend_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<WavetrendBatchOutput, WavetrendError> {
        WavetrendBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<WavetrendBatchOutput, WavetrendError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WavetrendBatchOutput, WavetrendError> {
        WavetrendBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hlc3")
    }
}

pub fn wavetrend_batch_with_kernel(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    k: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,

        Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(WavetrendError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    wavetrend_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WavetrendBatchOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
    pub combos: Vec<WavetrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WavetrendBatchOutput {
    pub fn row_for_params(&self, p: &WavetrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.channel_length.unwrap_or(9) == p.channel_length.unwrap_or(9)
                && c.average_length.unwrap_or(12) == p.average_length.unwrap_or(12)
                && c.ma_length.unwrap_or(3) == p.ma_length.unwrap_or(3)
                && (c.factor.unwrap_or(0.015) - p.factor.unwrap_or(0.015)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &WavetrendParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.wt1[start..start + self.cols],
                &self.wt2[start..start + self.cols],
                &self.wt_diff[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &WavetrendBatchRange) -> Result<Vec<WavetrendParams>, WavetrendError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, WavetrendError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            return Ok((start..=end).step_by(st).collect());
        }

        let st = step.max(1) as isize;
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(WavetrendError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, WavetrendError> {
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
                return Err(WavetrendError::InvalidRange {
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
            return Err(WavetrendError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let chs = axis_usize(r.channel_length)?;
    let avgs = axis_usize(r.average_length)?;
    let mas = axis_usize(r.ma_length)?;
    let factors = axis_f64(r.factor)?;

    let cap = chs
        .len()
        .checked_mul(avgs.len())
        .and_then(|x| x.checked_mul(mas.len()))
        .and_then(|x| x.checked_mul(factors.len()))
        .ok_or_else(|| WavetrendError::InvalidRange {
            start: "cap".into(),
            end: "overflow".into(),
            step: "mul".into(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &c in &chs {
        for &a in &avgs {
            for &m in &mas {
                for &f in &factors {
                    out.push(WavetrendParams {
                        channel_length: Some(c),
                        average_length: Some(a),
                        ma_length: Some(m),
                        factor: Some(f),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn wavetrend_batch_slice(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    wavetrend_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn wavetrend_batch_par_slice(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    wavetrend_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wavetrend_batch_inner(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(WavetrendError::InvalidRange {
            start: "range".into(),
            end: "range".into(),
            step: "empty".into(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;

    let mut max_p = 0usize;
    let mut warmup_periods = Vec::with_capacity(combos.len());
    for c in combos.iter() {
        let channel_length = c.channel_length.unwrap();
        if channel_length == 0 {
            return Err(WavetrendError::InvalidChannelLen {
                channel_length,
                data_len: data.len(),
            });
        }
        let average_length = c.average_length.unwrap();
        if average_length == 0 {
            return Err(WavetrendError::InvalidAverageLen {
                average_length,
                data_len: data.len(),
            });
        }
        let ma_length = c.ma_length.unwrap();
        if ma_length == 0 {
            return Err(WavetrendError::InvalidMaLen {
                ma_length,
                data_len: data.len(),
            });
        }

        max_p = max_p.max(channel_length).max(average_length).max(ma_length);
        warmup_periods.push(first + channel_length - 1 + average_length - 1 + ma_length - 1);
    }
    if data.len() - first < max_p {
        return Err(WavetrendError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let _ = rows
        .checked_mul(cols)
        .ok_or_else(|| WavetrendError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;

    let mut wt1_mu = make_uninit_matrix(rows, cols);
    let mut wt2_mu = make_uninit_matrix(rows, cols);
    let mut wt_diff_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut wt1_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut wt2_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut wt_diff_mu, cols, &warmup_periods);

    let mut wt1_guard = core::mem::ManuallyDrop::new(wt1_mu);
    let mut wt2_guard = core::mem::ManuallyDrop::new(wt2_mu);
    let mut wt_diff_guard = core::mem::ManuallyDrop::new(wt_diff_mu);

    let wt1: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt1_guard.as_mut_ptr() as *mut f64, wt1_guard.len())
    };
    let wt2: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt2_guard.as_mut_ptr() as *mut f64, wt2_guard.len())
    };
    let wt_diff: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(wt_diff_guard.as_mut_ptr() as *mut f64, wt_diff_guard.len())
    };

    let do_row = |row: usize, w1: &mut [f64], w2: &mut [f64], wd: &mut [f64]| unsafe {
        let p = &combos[row];
        let row_kernel = match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => wavetrend_row_avx512(
                data,
                first,
                p.channel_length.unwrap(),
                p.average_length.unwrap(),
                p.ma_length.unwrap(),
                p.factor.unwrap_or(0.015),
                w1,
                w2,
                wd,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => wavetrend_row_avx2(
                data,
                first,
                p.channel_length.unwrap(),
                p.average_length.unwrap(),
                p.ma_length.unwrap(),
                p.factor.unwrap_or(0.015),
                w1,
                w2,
                wd,
            ),
            _ => wavetrend_row_scalar(
                data,
                first,
                p.channel_length.unwrap(),
                p.average_length.unwrap(),
                p.ma_length.unwrap(),
                p.factor.unwrap_or(0.015),
                w1,
                w2,
                wd,
            ),
        };
        if let Err(e) = row_kernel {
            panic!("wavetrend row error: {:?}", e);
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            wt1.par_chunks_mut(cols)
                .zip(wt2.par_chunks_mut(cols))
                .zip(wt_diff.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((w1, w2), wd))| do_row(row, w1, w2, wd));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((w1, w2), wd))) in wt1
                .chunks_mut(cols)
                .zip(wt2.chunks_mut(cols))
                .zip(wt_diff.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, w1, w2, wd);
            }
        }
    } else {
        for (row, (((w1, w2), wd))) in wt1
            .chunks_mut(cols)
            .zip(wt2.chunks_mut(cols))
            .zip(wt_diff.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, w1, w2, wd);
        }
    }

    let wt1_vec = unsafe {
        Vec::from_raw_parts(
            wt1_guard.as_mut_ptr() as *mut f64,
            wt1_guard.len(),
            wt1_guard.capacity(),
        )
    };
    let wt2_vec = unsafe {
        Vec::from_raw_parts(
            wt2_guard.as_mut_ptr() as *mut f64,
            wt2_guard.len(),
            wt2_guard.capacity(),
        )
    };
    let wt_diff_vec = unsafe {
        Vec::from_raw_parts(
            wt_diff_guard.as_mut_ptr() as *mut f64,
            wt_diff_guard.len(),
            wt_diff_guard.capacity(),
        )
    };

    Ok(WavetrendBatchOutput {
        wt1: wt1_vec,
        wt2: wt2_vec,
        wt_diff: wt_diff_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn wavetrend_batch_inner_into(
    data: &[f64],
    sweep: &WavetrendBatchRange,
    kern: Kernel,
    parallel: bool,
    out_wt1: &mut [f64],
    out_wt2: &mut [f64],
    out_wt_diff: &mut [f64],
) -> Result<Vec<WavetrendParams>, WavetrendError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(WavetrendError::InvalidRange {
            start: "range".into(),
            end: "range".into(),
            step: "empty".into(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WavetrendError::AllValuesNaN)?;

    let mut max_p = 0usize;
    for c in combos.iter() {
        let channel_length = c.channel_length.unwrap();
        if channel_length == 0 {
            return Err(WavetrendError::InvalidChannelLen {
                channel_length,
                data_len: data.len(),
            });
        }
        let average_length = c.average_length.unwrap();
        if average_length == 0 {
            return Err(WavetrendError::InvalidAverageLen {
                average_length,
                data_len: data.len(),
            });
        }
        let ma_length = c.ma_length.unwrap();
        if ma_length == 0 {
            return Err(WavetrendError::InvalidMaLen {
                ma_length,
                data_len: data.len(),
            });
        }

        max_p = max_p.max(channel_length).max(average_length).max(ma_length);
    }
    if data.len() - first < max_p {
        return Err(WavetrendError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| WavetrendError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;
    if out_wt1.len() != total {
        return Err(WavetrendError::OutputSliceLengthMismatch {
            expected: total,
            got: out_wt1.len(),
        });
    }
    if out_wt2.len() != total {
        return Err(WavetrendError::OutputSliceLengthMismatch {
            expected: total,
            got: out_wt2.len(),
        });
    }
    if out_wt_diff.len() != total {
        return Err(WavetrendError::OutputSliceLengthMismatch {
            expected: total,
            got: out_wt_diff.len(),
        });
    }

    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.channel_length.unwrap() - 1 + combo.average_length.unwrap() - 1
            + combo.ma_length.unwrap()
            - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out_wt1[row_start + i] = f64::NAN;
            out_wt2[row_start + i] = f64::NAN;
            out_wt_diff[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, w1: &mut [f64], w2: &mut [f64], wd: &mut [f64]| unsafe {
        let p = &combos[row];
        let r = wavetrend_row_scalar(
            data,
            first,
            p.channel_length.unwrap(),
            p.average_length.unwrap(),
            p.ma_length.unwrap(),
            p.factor.unwrap_or(0.015),
            w1,
            w2,
            wd,
        );
        if let Err(e) = r {
            panic!("wavetrend row error: {:?}", e);
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_wt1
                .par_chunks_mut(cols)
                .zip(out_wt2.par_chunks_mut(cols))
                .zip(out_wt_diff.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((w1, w2), wd))| do_row(row, w1, w2, wd));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((w1, w2), wd))) in out_wt1
                .chunks_mut(cols)
                .zip(out_wt2.chunks_mut(cols))
                .zip(out_wt_diff.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, w1, w2, wd);
            }
        }
    } else {
        for (row, (((w1, w2), wd))) in out_wt1
            .chunks_mut(cols)
            .zip(out_wt2.chunks_mut(cols))
            .zip(out_wt_diff.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, w1, w2, wd);
        }
    }
    Ok(combos)
}

#[inline(always)]
unsafe fn wavetrend_row_scalar(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Scalar,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx2(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx2,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx512,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512_short(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx512,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512_long(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
) -> Result<(), WavetrendError> {
    wavetrend_row_with_kernel(
        data,
        first,
        channel_len,
        average_len,
        ma_len,
        factor,
        wt1,
        wt2,
        wd,
        Kernel::Avx512,
    )
}

#[inline(always)]
unsafe fn wavetrend_row_with_kernel(
    data: &[f64],
    first: usize,
    channel_len: usize,
    average_len: usize,
    ma_len: usize,
    factor: f64,
    wt1: &mut [f64],
    wt2: &mut [f64],
    wd: &mut [f64],
    kernel: Kernel,
) -> Result<(), WavetrendError> {
    debug_assert_eq!(wt1.len(), data.len());
    debug_assert_eq!(wt2.len(), data.len());
    debug_assert_eq!(wd.len(), data.len());

    let warmup = first + channel_len - 1 + average_len - 1 + ma_len - 1;

    wavetrend_compute_into(
        data,
        channel_len,
        average_len,
        ma_len,
        factor,
        first,
        warmup,
        wt1,
        wt2,
        wd,
        kernel,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use crate::utilities::enums::Kernel;

    fn check_wavetrend_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = WavetrendParams {
            channel_length: None,
            average_length: None,
            ma_length: None,
            factor: None,
        };
        let input = WavetrendInput::from_candles(&candles, "hlc3", default_params);
        let output = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(output.wt1.len(), candles.close.len());
        Ok(())
    }

    fn check_wavetrend_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = WavetrendInput::from_candles(&candles, "hlc3", WavetrendParams::default());
        let result = wavetrend_with_kernel(&input, kernel)?;
        let len = result.wt1.len();
        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];
        for (i, &val) in result.wt1[len - 5..].iter().enumerate() {
            let diff = (val - expected_wt1[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT1 mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_wt1[i]
            );
        }
        for (i, &val) in result.wt2[len - 5..].iter().enumerate() {
            let diff = (val - expected_wt2[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT2 mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_wt2[i]
            );
        }
        let last_five_diff = &result.wt_diff[len - 5..];
        for i in 0..5 {
            let expected = expected_wt2[i] - expected_wt1[i];
            let diff = (last_five_diff[i] - expected).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT_DIFF mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                last_five_diff[i],
                expected
            );
        }
        Ok(())
    }

    fn check_wavetrend_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = WavetrendInput::with_default_candles(&candles);
        match input.data {
            WavetrendData::Candles { source, .. } => assert_eq!(source, "hlc3"),
            _ => panic!("Expected WavetrendData::Candles"),
        }
        let output = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(output.wt1.len(), candles.close.len());
        Ok(())
    }

    fn check_wavetrend_zero_channel(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(0),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with zero channel_length",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_channel_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(10),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        };
        let input = WavetrendInput::from_slice(&data_small, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with channel_length exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_slice(&single_point, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = WavetrendInput::from_candles(
            &candles,
            "hlc3",
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(0.015),
            },
        );
        let res = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(res.wt1.len(), candles.close.len());
        if res.wt1.len() > 240 {
            for (i, &val) in res.wt1[240..].iter().enumerate() {
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

    #[cfg(debug_assertions)]
    fn check_wavetrend_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            WavetrendParams::default(),
            WavetrendParams {
                channel_length: Some(1),
                average_length: Some(1),
                ma_length: Some(1),
                factor: Some(0.001),
            },
            WavetrendParams {
                channel_length: Some(2),
                average_length: Some(2),
                ma_length: Some(2),
                factor: Some(0.005),
            },
            WavetrendParams {
                channel_length: Some(5),
                average_length: Some(7),
                ma_length: Some(3),
                factor: Some(0.01),
            },
            WavetrendParams {
                channel_length: Some(10),
                average_length: Some(15),
                ma_length: Some(5),
                factor: Some(0.02),
            },
            WavetrendParams {
                channel_length: Some(20),
                average_length: Some(25),
                ma_length: Some(7),
                factor: Some(0.025),
            },
            WavetrendParams {
                channel_length: Some(30),
                average_length: Some(40),
                ma_length: Some(10),
                factor: Some(0.03),
            },
            WavetrendParams {
                channel_length: Some(50),
                average_length: Some(60),
                ma_length: Some(15),
                factor: Some(0.04),
            },
            WavetrendParams {
                channel_length: Some(100),
                average_length: Some(120),
                ma_length: Some(20),
                factor: Some(0.05),
            },
            WavetrendParams {
                channel_length: Some(7),
                average_length: Some(11),
                ma_length: Some(3),
                factor: Some(0.013),
            },
            WavetrendParams {
                channel_length: Some(13),
                average_length: Some(17),
                ma_length: Some(5),
                factor: Some(0.017),
            },
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(3),
                ma_length: Some(12),
                factor: Some(0.015),
            },
            WavetrendParams {
                channel_length: Some(15),
                average_length: Some(15),
                ma_length: Some(15),
                factor: Some(0.015),
            },
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(0.0001),
            },
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(1.0),
            },
            WavetrendParams {
                channel_length: Some(3),
                average_length: Some(5),
                ma_length: Some(1),
                factor: Some(0.008),
            },
            WavetrendParams {
                channel_length: Some(8),
                average_length: Some(13),
                ma_length: Some(2),
                factor: Some(0.021),
            },
            WavetrendParams {
                channel_length: Some(21),
                average_length: Some(34),
                ma_length: Some(8),
                factor: Some(0.034),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = WavetrendInput::from_candles(&candles, "hlc3", params.clone());
            let output = wavetrend_with_kernel(&input, kernel)?;

            for (i, &val) in output.wt1.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }
            }

            for (i, &val) in output.wt2.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }
            }

            for (i, &val) in output.wt_diff.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.channel_length.unwrap_or(9),
                        params.average_length.unwrap_or(12),
                        params.ma_length.unwrap_or(3),
                        params.factor.unwrap_or(0.015),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_wavetrend_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn check_wavetrend_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let channel_length = 9;
        let average_length = 12;
        let ma_length = 3;
        let factor = 0.015;

        let input = WavetrendInput::from_candles(
            &candles,
            "hlc3",
            WavetrendParams {
                channel_length: Some(channel_length),
                average_length: Some(average_length),
                ma_length: Some(ma_length),
                factor: Some(factor),
            },
        );
        let full_output = wavetrend_with_kernel(&input, kernel)?;

        let mut stream = WavetrendStream::try_new(WavetrendParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
            ma_length: Some(ma_length),
            factor: Some(factor),
        })?;

        let mut wt1_stream = Vec::with_capacity(candles.hlc3.len());
        let mut wt2_stream = Vec::with_capacity(candles.hlc3.len());
        let mut diff_stream = Vec::with_capacity(candles.hlc3.len());
        for &price in &candles.hlc3 {
            match stream.update(price) {
                Some((wt1, wt2, diff)) => {
                    wt1_stream.push(wt1);
                    wt2_stream.push(wt2);
                    diff_stream.push(diff);
                }
                None => {
                    wt1_stream.push(f64::NAN);
                    wt2_stream.push(f64::NAN);
                    diff_stream.push(f64::NAN);
                }
            }
        }

        let mut first_non_nan = None;
        for (i, &b) in full_output.wt1.iter().enumerate() {
            if !b.is_nan() {
                first_non_nan = Some(i);
                break;
            }
        }
        let start = first_non_nan.unwrap_or(0);
        assert_eq!(full_output.wt1.len(), wt1_stream.len());
        for (i, (&b, &s)) in full_output
            .wt1
            .iter()
            .zip(wt1_stream.iter())
            .enumerate()
            .skip(start)
        {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt1 f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in full_output.wt2.iter().zip(wt2_stream.iter()).enumerate() {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt2 f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in full_output
            .wt_diff
            .iter()
            .zip(diff_stream.iter())
            .enumerate()
        {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt_diff f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_wavetrend_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=30, 2usize..=30, 1usize..=10, 0.001f64..1.0f64).prop_flat_map(
            |(channel_len, average_len, ma_len, factor)| {
                let min_len = channel_len + average_len + ma_len + 20;
                (min_len..400).prop_flat_map(move |data_len| {
                    (
                        prop::collection::vec(
                            (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                            data_len,
                        ),
                        Just(channel_len),
                        Just(average_len),
                        Just(ma_len),
                        Just(factor),
                    )
                })
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(data, channel_len, average_len, ma_len, factor)| {
                    let params = WavetrendParams {
                        channel_length: Some(channel_len),
                        average_length: Some(average_len),
                        ma_length: Some(ma_len),
                        factor: Some(factor),
                    };
                    let input = WavetrendInput::from_slice(&data, params);

                    let output = wavetrend_with_kernel(&input, kernel).unwrap();
                    let ref_output = wavetrend_with_kernel(&input, Kernel::Scalar).unwrap();

                    let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                    let expected_warmup =
                        first_valid + channel_len - 1 + average_len - 1 + ma_len - 1;

                    for i in expected_warmup.min(data.len())..data.len() {
                        if output.wt1[i].is_finite() && output.wt2[i].is_finite() {
                            let expected_diff = output.wt2[i] - output.wt1[i];
                            let actual_diff = output.wt_diff[i];
                            prop_assert!(
                                (actual_diff - expected_diff).abs() <= 1e-9,
                                "WT_DIFF mismatch at idx {}: expected {}, got {}",
                                i,
                                expected_diff,
                                actual_diff
                            );
                        }
                    }

                    let valid_start = expected_warmup.min(data.len());
                    let valid_wt1: Vec<f64> = output.wt1[valid_start..]
                        .iter()
                        .filter(|&&x| x.is_finite())
                        .copied()
                        .collect();
                    let valid_wt2: Vec<f64> = output.wt2[valid_start..]
                        .iter()
                        .filter(|&&x| x.is_finite())
                        .copied()
                        .collect();

                    if valid_wt1.len() > 10 && valid_wt2.len() > 10 && ma_len > 1 {
                        let mut wt1_changes = 0.0;
                        let mut wt2_changes = 0.0;
                        for i in 1..valid_wt1.len().min(valid_wt2.len()) {
                            wt1_changes += (valid_wt1[i] - valid_wt1[i - 1]).abs();
                            wt2_changes += (valid_wt2[i] - valid_wt2[i - 1]).abs();
                        }

                        if wt1_changes > 1e-6 {
                            prop_assert!(
                                wt2_changes <= wt1_changes * 1.1,
                                "WT2 should be smoother: wt1_changes={}, wt2_changes={}",
                                wt1_changes,
                                wt2_changes
                            );
                        }
                    }

                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9)
                        && data.len() > valid_start + 10
                    {
                        let last_10_wt1: Vec<f64> = output.wt1[output.wt1.len() - 10..]
                            .iter()
                            .filter(|&&x| x.is_finite())
                            .copied()
                            .collect();
                        if last_10_wt1.len() >= 5 {
                            let avg_wt1: f64 =
                                last_10_wt1.iter().sum::<f64>() / last_10_wt1.len() as f64;
                            prop_assert!(
                                avg_wt1.abs() <= 1.0,
                                "Constant price should give near-zero oscillator: avg_wt1={}",
                                avg_wt1
                            );
                        }
                    }

                    if factor < 0.5 && valid_start < data.len() {
                        let params_double = WavetrendParams {
                            channel_length: Some(channel_len),
                            average_length: Some(average_len),
                            ma_length: Some(ma_len),
                            factor: Some(factor * 2.0),
                        };
                        let input_double = WavetrendInput::from_slice(&data, params_double);
                        let output_double = wavetrend_with_kernel(&input_double, kernel).unwrap();

                        let check_end = data.len().min(valid_start + 20);
                        let mut checked_count = 0;
                        for i in valid_start..check_end {
                            if output.wt1[i].is_finite()
                                && output_double.wt1[i].is_finite()
                                && output.wt1[i].abs() > 0.1
                            {
                                let ratio = output_double.wt1[i] / output.wt1[i];

                                prop_assert!(
								(ratio - 0.5).abs() <= 0.35,
								"Factor doubling should roughly halve WT1 at idx {}: original={}, doubled={}, ratio={}",
								i, output.wt1[i], output_double.wt1[i], ratio
							);
                                checked_count += 1;
                                if checked_count >= 5 {
                                    break;
                                }
                            }
                        }
                    }

                    if ma_len == 1 {
                        for i in valid_start..data.len() {
                            if output.wt1[i].is_finite() && output.wt2[i].is_finite() {
                                prop_assert!(
                                    (output.wt1[i] - output.wt2[i]).abs() <= 1e-9,
                                    "When ma_len=1, WT2 should equal WT1 at idx {}: wt1={}, wt2={}",
                                    i,
                                    output.wt1[i],
                                    output.wt2[i]
                                );
                            }
                        }
                    }

                    for i in 0..data.len() {
                        let wt1 = output.wt1[i];
                        let wt1_ref = ref_output.wt1[i];
                        let wt2 = output.wt2[i];
                        let wt2_ref = ref_output.wt2[i];
                        let diff = output.wt_diff[i];
                        let diff_ref = ref_output.wt_diff[i];

                        if wt1.is_nan() || wt1_ref.is_nan() {
                            prop_assert!(
                                wt1.is_nan() && wt1_ref.is_nan(),
                                "NaN mismatch for WT1 at idx {}: kernel={:?}, ref={:?}",
                                i,
                                wt1,
                                wt1_ref
                            );
                        } else {
                            let wt1_bits = wt1.to_bits();
                            let wt1_ref_bits = wt1_ref.to_bits();
                            let ulp_diff = wt1_bits.abs_diff(wt1_ref_bits);
                            prop_assert!(
                                (wt1 - wt1_ref).abs() <= 1e-9 || ulp_diff <= 4,
                                "WT1 mismatch at idx {}: kernel={}, ref={} (ULP={})",
                                i,
                                wt1,
                                wt1_ref,
                                ulp_diff
                            );
                        }

                        if wt2.is_nan() || wt2_ref.is_nan() {
                            prop_assert!(
                                wt2.is_nan() && wt2_ref.is_nan(),
                                "NaN mismatch for WT2 at idx {}: kernel={:?}, ref={:?}",
                                i,
                                wt2,
                                wt2_ref
                            );
                        } else {
                            let wt2_bits = wt2.to_bits();
                            let wt2_ref_bits = wt2_ref.to_bits();
                            let ulp_diff = wt2_bits.abs_diff(wt2_ref_bits);
                            prop_assert!(
                                (wt2 - wt2_ref).abs() <= 1e-9 || ulp_diff <= 4,
                                "WT2 mismatch at idx {}: kernel={}, ref={} (ULP={})",
                                i,
                                wt2,
                                wt2_ref,
                                ulp_diff
                            );
                        }

                        if diff.is_nan() || diff_ref.is_nan() {
                            prop_assert!(
                                diff.is_nan() && diff_ref.is_nan(),
                                "NaN mismatch for WT_DIFF at idx {}: kernel={:?}, ref={:?}",
                                i,
                                diff,
                                diff_ref
                            );
                        } else {
                            let diff_bits = diff.to_bits();
                            let diff_ref_bits = diff_ref.to_bits();
                            let ulp_diff = diff_bits.abs_diff(diff_ref_bits);
                            prop_assert!(
                                (diff - diff_ref).abs() <= 1e-9 || ulp_diff <= 4,
                                "WT_DIFF mismatch at idx {}: kernel={}, ref={} (ULP={})",
                                i,
                                diff,
                                diff_ref,
                                ulp_diff
                            );
                        }
                    }

                    for i in 0..expected_warmup.min(data.len()) {
                        prop_assert!(
                            output.wt1[i].is_nan(),
                            "WT1 should be NaN during warmup at idx {}: got {}",
                            i,
                            output.wt1[i]
                        );
                        prop_assert!(
                            output.wt2[i].is_nan(),
                            "WT2 should be NaN during warmup at idx {}: got {}",
                            i,
                            output.wt2[i]
                        );
                        prop_assert!(
                            output.wt_diff[i].is_nan(),
                            "WT_DIFF should be NaN during warmup at idx {}: got {}",
                            i,
                            output.wt_diff[i]
                        );
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_wavetrend_tests {
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

    generate_all_wavetrend_tests!(
        check_wavetrend_partial_params,
        check_wavetrend_accuracy,
        check_wavetrend_default_candles,
        check_wavetrend_zero_channel,
        check_wavetrend_channel_exceeds_length,
        check_wavetrend_very_small_dataset,
        check_wavetrend_nan_handling,
        check_wavetrend_streaming,
        check_wavetrend_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_wavetrend_tests!(check_wavetrend_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = WavetrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hlc3")?;

        let def = WavetrendParams::default();
        let (wt1_row, wt2_row, diff_row) = output.values_for(&def).expect("default row missing");

        assert_eq!(wt1_row.len(), c.close.len());
        assert_eq!(wt2_row.len(), c.close.len());
        assert_eq!(diff_row.len(), c.close.len());

        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];

        let start = wt1_row.len().saturating_sub(5);
        for (i, &v) in wt1_row[start..].iter().enumerate() {
            assert!(
                (v - expected_wt1[i]).abs() < 1e-8,
                "[{test}] default-row WT1 mismatch at idx {i}: {v} vs {expected}",
                test = test,
                i = i,
                v = v,
                expected = expected_wt1[i]
            );
        }
        for (i, &v) in wt2_row[start..].iter().enumerate() {
            assert!(
                (v - expected_wt2[i]).abs() < 1e-6,
                "[{test}] default-row WT2 mismatch at idx {i}: {v} vs {expected}",
                test = test,
                i = i,
                v = v,
                expected = expected_wt2[i]
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
            (2, 10, 2, 3, 12, 3, 1, 5, 1, 0.005, 0.015, 0.005),
            (5, 25, 5, 10, 30, 5, 2, 8, 2, 0.01, 0.03, 0.01),
            (20, 60, 10, 25, 75, 10, 5, 15, 5, 0.02, 0.05, 0.015),
            (2, 5, 1, 2, 5, 1, 1, 3, 1, 0.001, 0.005, 0.001),
            (10, 30, 10, 15, 45, 15, 3, 9, 3, 0.015, 0.045, 0.015),
            (50, 100, 25, 60, 120, 30, 10, 20, 5, 0.03, 0.06, 0.03),
            (9, 9, 0, 12, 12, 0, 3, 3, 0, 0.015, 0.015, 0.0),
            (1, 3, 1, 1, 3, 1, 1, 2, 1, 0.001, 0.003, 0.001),
        ];

        for (cfg_idx, config) in test_configs.iter().enumerate() {
            let output = WavetrendBatchBuilder::new()
                .kernel(kernel)
                .channel_range(config.0, config.1, config.2)
                .avg_range(config.3, config.4, config.5)
                .ma_range(config.6, config.7, config.8)
                .factor_range(config.9, config.10, config.11)
                .apply_candles(&c, "hlc3")?;

            for (idx, &val) in output.wt1.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt1 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }
            }

            for (idx, &val) in output.wt2.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt2 output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }
            }

            for (idx, &val) in output.wt_diff.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in wt_diff output with params: channel_length={}, average_length={}, ma_length={}, factor={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.channel_length.unwrap_or(9),
                        combo.average_length.unwrap_or(12),
                        combo.ma_length.unwrap_or(3),
                        combo.factor.unwrap_or(0.015)
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
