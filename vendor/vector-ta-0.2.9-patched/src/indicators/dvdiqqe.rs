use crate::indicators::moving_averages::ema::{EmaInput, EmaParams, ema_with_kernel};
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

#[derive(Debug, Clone)]
pub enum DvdiqqeData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: Option<&'a [f64]>,
    },
}

#[derive(Debug, Clone)]
pub struct DvdiqqeOutput {
    pub dvdi: Vec<f64>,
    pub fast_tl: Vec<f64>,
    pub slow_tl: Vec<f64>,
    pub center_line: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub enum DvdiqqeOutputField {
    Dvdi,
    FastTl,
    SlowTl,
    CenterLine,
}

#[derive(Debug, Clone)]
pub struct DvdiqqeParams {
    pub period: Option<usize>,
    pub smoothing_period: Option<usize>,
    pub fast_multiplier: Option<f64>,
    pub slow_multiplier: Option<f64>,
    pub volume_type: Option<String>,
    pub center_type: Option<String>,
    pub tick_size: Option<f64>,
}

impl Default for DvdiqqeParams {
    fn default() -> Self {
        Self {
            period: Some(13),
            smoothing_period: Some(6),
            fast_multiplier: Some(2.618),
            slow_multiplier: Some(4.236),
            volume_type: Some("default".to_string()),
            center_type: Some("dynamic".to_string()),
            tick_size: Some(0.01),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DvdiqqeInput<'a> {
    pub data: DvdiqqeData<'a>,
    pub params: DvdiqqeParams,
}

impl<'a> AsRef<[f64]> for DvdiqqeInput<'a> {
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DvdiqqeData::Candles { candles } => &candles.close,
            DvdiqqeData::Slices { close, .. } => close,
        }
    }
}

impl<'a> DvdiqqeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: DvdiqqeParams) -> Self {
        Self {
            data: DvdiqqeData::Candles { candles: c },
            params: p,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: Option<&'a [f64]>,
        p: DvdiqqeParams,
    ) -> Self {
        Self {
            data: DvdiqqeData::Slices {
                open,
                high,
                low,
                close,
                volume,
            },
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, DvdiqqeParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(13)
    }

    #[inline]
    pub fn get_smoothing_period(&self) -> usize {
        self.params.smoothing_period.unwrap_or(6)
    }

    #[inline]
    pub fn get_fast_multiplier(&self) -> f64 {
        self.params.fast_multiplier.unwrap_or(2.618)
    }

    #[inline]
    pub fn get_slow_multiplier(&self) -> f64 {
        self.params.slow_multiplier.unwrap_or(4.236)
    }

    #[inline]
    pub fn get_volume_type(&self) -> &str {
        self.params.volume_type.as_deref().unwrap_or("default")
    }

    #[inline]
    pub fn get_center_type(&self) -> &str {
        self.params.center_type.as_deref().unwrap_or("dynamic")
    }

    #[inline]
    pub fn get_tick_size(&self) -> f64 {
        self.params.tick_size.unwrap_or(0.01)
    }
}

#[derive(Debug, Error)]
pub enum DvdiqqeError {
    #[error("dvdiqqe: Empty input data")]
    EmptyInputData,

    #[error("dvdiqqe: All values are NaN")]
    AllValuesNaN,

    #[error("dvdiqqe: Invalid period: {period}, data length: {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("dvdiqqe: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("Input arrays must have the same length")]
    MissingData,

    #[error("dvdiqqe: Invalid smoothing period: {smoothing}")]
    InvalidSmoothing { smoothing: usize },

    #[error("dvdiqqe: Invalid tick size: {tick}")]
    InvalidTick { tick: f64 },

    #[error("Invalid multiplier: {which} multiplier must be positive (got {multiplier})")]
    InvalidMultiplier { multiplier: f64, which: String },

    #[error("dvdiqqe: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("dvdiqqe: Invalid range (usize): start={start} end={end} step={step}")]
    InvalidRangeUsize {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("dvdiqqe: Invalid range (f64): start={start} end={end} step={step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },

    #[error("dvdiqqe: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),

    #[error("dvdiqqe: {0}")]
    InvalidInput(String),

    #[error("dvdiqqe: EMA computation failed: {0}")]
    EmaError(String),
}

#[inline(always)]
fn dvdiqqe_prepare<'a>(
    input: &'a DvdiqqeInput,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        Option<&'a [f64]>,
        usize,
        usize,
        f64,
        f64,
        &'a str,
        &'a str,
        f64,
        usize,
    ),
    DvdiqqeError,
> {
    let (o, h, l, c, v) = match &input.data {
        DvdiqqeData::Candles { candles } => (
            &candles.open[..],
            &candles.high[..],
            &candles.low[..],
            &candles.close[..],
            Some(&candles.volume[..]),
        ),
        DvdiqqeData::Slices {
            open,
            high,
            low,
            close,
            volume,
        } => (*open, *high, *low, *close, *volume),
    };

    let len = c.len();
    if len == 0 {
        return Err(DvdiqqeError::EmptyInputData);
    }
    if o.len() != len || h.len() != len || l.len() != len {
        return Err(DvdiqqeError::MissingData);
    }
    if let Some(vs) = v {
        if vs.len() != len {
            return Err(DvdiqqeError::MissingData);
        }
    }

    let first = c
        .iter()
        .position(|x| x.is_finite())
        .ok_or(DvdiqqeError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(DvdiqqeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let smoothing = input.get_smoothing_period();
    if smoothing == 0 {
        return Err(DvdiqqeError::InvalidSmoothing { smoothing });
    }

    let fast_mult = input.get_fast_multiplier();
    let slow_mult = input.get_slow_multiplier();
    if fast_mult <= 0.0 || !fast_mult.is_finite() {
        return Err(DvdiqqeError::InvalidMultiplier {
            multiplier: fast_mult,
            which: "fast".to_string(),
        });
    }
    if slow_mult <= 0.0 || !slow_mult.is_finite() {
        return Err(DvdiqqeError::InvalidMultiplier {
            multiplier: slow_mult,
            which: "slow".to_string(),
        });
    }

    if len - first < period {
        return Err(DvdiqqeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let tick = input.get_tick_size();
    if !tick.is_finite() || tick <= 0.0 {
        return Err(DvdiqqeError::InvalidTick { tick });
    }

    Ok((
        o,
        h,
        l,
        c,
        v,
        period,
        smoothing,
        fast_mult,
        slow_mult,
        input.get_volume_type(),
        input.get_center_type(),
        tick,
        first,
    ))
}

#[inline(always)]
fn dvdiqqe_compute_into(
    open: &[f64],
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    volume_opt: Option<&[f64]>,
    period: usize,
    smoothing_period: usize,
    fast_mult: f64,
    slow_mult: f64,
    volume_type: &str,
    center_type: &str,
    tick: f64,
    first_valid: usize,
    kernel: Kernel,
    dvdi_out: &mut [f64],
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    center_out: &mut [f64],
) -> Result<(), DvdiqqeError> {
    let len = close.len();
    assert_eq!(dvdi_out.len(), len);
    assert_eq!(fast_out.len(), len);
    assert_eq!(slow_out.len(), len);
    assert_eq!(center_out.len(), len);

    if len == 0 {
        return Ok(());
    }

    let wper = (period * 2) - 1;
    let warmup = first_valid + wper;

    let mut pvi = alloc_with_nan_prefix(len, 0);
    let mut nvi = alloc_with_nan_prefix(len, 0);

    let mut pvi_prev = 0.0f64;
    let mut nvi_prev = 0.0f64;
    let mut prev_vol = 0.0f64;
    let mut prev_close = 0.0f64;
    let mut tickrng_prev = tick;
    let use_tick_only = volume_type.eq_ignore_ascii_case("tick");

    for i in 0..len {
        let oi = open[i];
        let ci = close[i];

        let rng = ci - oi;
        let tickrng = if rng.abs() < tick { tickrng_prev } else { rng };
        let tick_vol = (tickrng.abs() / tick).max(0.0);

        let sel_vol = if use_tick_only {
            tick_vol
        } else if let Some(vs) = volume_opt {
            let vv = vs[i];
            if vv.is_finite() { vv } else { tick_vol }
        } else {
            tick_vol
        };

        let d_close = ci - prev_close;
        if sel_vol > prev_vol {
            pvi_prev += d_close;
        }
        if sel_vol < prev_vol {
            nvi_prev -= d_close;
        }

        pvi[i] = pvi_prev;
        nvi[i] = nvi_prev;
        prev_close = ci;
        prev_vol = sel_vol;
        tickrng_prev = tickrng;
    }

    let pvi_ema = {
        let prm = EmaParams {
            period: Some(period),
        };
        let inp = EmaInput::from_slice(&pvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let nvi_ema = {
        let prm = EmaParams {
            period: Some(period),
        };
        let inp = EmaInput::from_slice(&nvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    for i in 0..len {
        pvi[i] = pvi[i] - pvi_ema.values[i];
        nvi[i] = nvi[i] - nvi_ema.values[i];
    }

    let pdiv_ema = {
        let prm = EmaParams {
            period: Some(smoothing_period),
        };
        let inp = EmaInput::from_slice(&pvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let ndiv_ema = {
        let prm = EmaParams {
            period: Some(smoothing_period),
        };
        let inp = EmaInput::from_slice(&nvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    let mut ranges = alloc_with_nan_prefix(len, 1);
    dvdi_out[0] = pdiv_ema.values[0] - ndiv_ema.values[0];
    for i in 1..len {
        let dvdi_i = pdiv_ema.values[i] - ndiv_ema.values[i];
        ranges[i] = (dvdi_i - dvdi_out[i - 1]).abs();
        dvdi_out[i] = dvdi_i;
    }

    let avg_range = {
        let prm = EmaParams { period: Some(wper) };
        let inp = EmaInput::from_slice(&ranges, prm);
        ema_with_kernel(&inp, Kernel::Auto).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let smooth_range = {
        let prm = EmaParams { period: Some(wper) };
        let inp = EmaInput::from_slice(&avg_range.values, prm);
        ema_with_kernel(&inp, Kernel::Auto).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    for i in 0..warmup.min(len) {
        dvdi_out[i] = f64::NAN;
        fast_out[i] = f64::NAN;
        slow_out[i] = f64::NAN;
    }

    if warmup < len {
        fast_out[warmup] = dvdi_out[warmup];
        slow_out[warmup] = dvdi_out[warmup];

        for i in (warmup + 1)..len {
            let fr = smooth_range.values[i] * fast_mult;
            let sr = smooth_range.values[i] * slow_mult;

            if dvdi_out[i] > fast_out[i - 1] {
                let nv = dvdi_out[i] - fr;
                fast_out[i] = if nv < fast_out[i - 1] {
                    fast_out[i - 1]
                } else {
                    nv
                };
            } else {
                let nv = dvdi_out[i] + fr;
                fast_out[i] = if nv > fast_out[i - 1] {
                    fast_out[i - 1]
                } else {
                    nv
                };
            }

            if dvdi_out[i] > slow_out[i - 1] {
                let nv = dvdi_out[i] - sr;
                slow_out[i] = if nv < slow_out[i - 1] {
                    slow_out[i - 1]
                } else {
                    nv
                };
            } else {
                let nv = dvdi_out[i] + sr;
                slow_out[i] = if nv > slow_out[i - 1] {
                    slow_out[i - 1]
                } else {
                    nv
                };
            }
        }
    }

    for i in 0..warmup.min(len) {
        center_out[i] = f64::NAN;
    }
    if center_type.eq_ignore_ascii_case("dynamic") {
        let mut sum = 0.0f64;
        let mut cnt = 0.0f64;
        for i in warmup..len {
            let v = dvdi_out[i];
            if v.is_finite() {
                sum += v;
                cnt += 1.0;
            }
            center_out[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
        }
    } else {
        for i in warmup..len {
            center_out[i] = 0.0;
        }
    }

    Ok(())
}

#[inline(always)]
fn dvdiqqe_build_raw_dvdi(
    open: &[f64],
    close: &[f64],
    volume_opt: Option<&[f64]>,
    period: usize,
    smoothing_period: usize,
    volume_type: &str,
    tick: f64,
    kernel: Kernel,
) -> Result<Vec<f64>, DvdiqqeError> {
    let len = close.len();
    let mut pvi = alloc_with_nan_prefix(len, 0);
    let mut nvi = alloc_with_nan_prefix(len, 0);

    let mut pvi_prev = 0.0f64;
    let mut nvi_prev = 0.0f64;
    let mut prev_vol = 0.0f64;
    let mut prev_close = 0.0f64;
    let mut tickrng_prev = tick;
    let use_tick_only = volume_type.eq_ignore_ascii_case("tick");

    for i in 0..len {
        let oi = open[i];
        let ci = close[i];

        let rng = ci - oi;
        let tickrng = if rng.abs() < tick { tickrng_prev } else { rng };
        let tick_vol = (tickrng.abs() / tick).max(0.0);

        let sel_vol = if use_tick_only {
            tick_vol
        } else if let Some(vs) = volume_opt {
            let vv = vs[i];
            if vv.is_finite() { vv } else { tick_vol }
        } else {
            tick_vol
        };

        let d_close = ci - prev_close;
        if sel_vol > prev_vol {
            pvi_prev += d_close;
        }
        if sel_vol < prev_vol {
            nvi_prev -= d_close;
        }

        pvi[i] = pvi_prev;
        nvi[i] = nvi_prev;
        prev_close = ci;
        prev_vol = sel_vol;
        tickrng_prev = tickrng;
    }

    let pvi_ema = {
        let prm = EmaParams {
            period: Some(period),
        };
        let inp = EmaInput::from_slice(&pvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let nvi_ema = {
        let prm = EmaParams {
            period: Some(period),
        };
        let inp = EmaInput::from_slice(&nvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    for i in 0..len {
        pvi[i] = pvi[i] - pvi_ema.values[i];
        nvi[i] = nvi[i] - nvi_ema.values[i];
    }

    let pdiv_ema = {
        let prm = EmaParams {
            period: Some(smoothing_period),
        };
        let inp = EmaInput::from_slice(&pvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };
    let ndiv_ema = {
        let prm = EmaParams {
            period: Some(smoothing_period),
        };
        let inp = EmaInput::from_slice(&nvi, prm);
        ema_with_kernel(&inp, kernel).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
    };

    let mut dvdi = alloc_with_nan_prefix(len, 0);
    for i in 0..len {
        dvdi[i] = pdiv_ema.values[i] - ndiv_ema.values[i];
    }

    Ok(dvdi)
}

#[inline]
pub fn dvdiqqe_output_into_slice(
    dst: &mut [f64],
    input: &DvdiqqeInput,
    kernel: Kernel,
    field: DvdiqqeOutputField,
) -> Result<(), DvdiqqeError> {
    let (o, _h, _l, c, v, period, smoothing, fast, slow, vt, ct, tick, first) =
        dvdiqqe_prepare(input)?;

    let len = c.len();
    if dst.len() != len {
        return Err(DvdiqqeError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let wper = (period * 2) - 1;
    let warmup = first + wper;
    let dvdi = dvdiqqe_build_raw_dvdi(o, c, v, period, smoothing, vt, tick, kernel)?;

    for i in 0..warmup.min(len) {
        dst[i] = f64::NAN;
    }

    match field {
        DvdiqqeOutputField::Dvdi => {
            if warmup < len {
                dst[warmup..].copy_from_slice(&dvdi[warmup..]);
            }
        }
        DvdiqqeOutputField::CenterLine => {
            if ct.eq_ignore_ascii_case("dynamic") {
                let mut sum = 0.0f64;
                let mut cnt = 0.0f64;
                for i in warmup..len {
                    let value = dvdi[i];
                    if value.is_finite() {
                        sum += value;
                        cnt += 1.0;
                    }
                    dst[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
                }
            } else {
                for i in warmup..len {
                    dst[i] = 0.0;
                }
            }
        }
        DvdiqqeOutputField::FastTl | DvdiqqeOutputField::SlowTl => {
            let mut ranges = alloc_with_nan_prefix(len, 1);
            for i in 1..len {
                ranges[i] = (dvdi[i] - dvdi[i - 1]).abs();
            }

            let avg_range = {
                let prm = EmaParams { period: Some(wper) };
                let inp = EmaInput::from_slice(&ranges, prm);
                ema_with_kernel(&inp, Kernel::Auto)
                    .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
            };
            let smooth_range = {
                let prm = EmaParams { period: Some(wper) };
                let inp = EmaInput::from_slice(&avg_range.values, prm);
                ema_with_kernel(&inp, Kernel::Auto)
                    .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
            };

            if warmup < len {
                let multiplier = match field {
                    DvdiqqeOutputField::FastTl => fast,
                    DvdiqqeOutputField::SlowTl => slow,
                    _ => unreachable!(),
                };
                dst[warmup] = dvdi[warmup];
                for i in (warmup + 1)..len {
                    let range = smooth_range.values[i] * multiplier;
                    if dvdi[i] > dst[i - 1] {
                        let nv = dvdi[i] - range;
                        dst[i] = if nv < dst[i - 1] { dst[i - 1] } else { nv };
                    } else {
                        let nv = dvdi[i] + range;
                        dst[i] = if nv > dst[i - 1] { dst[i - 1] } else { nv };
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn dvdiqqe(input: &DvdiqqeInput) -> Result<DvdiqqeOutput, DvdiqqeError> {
    dvdiqqe_with_kernel(input, Kernel::Auto)
}

pub fn dvdiqqe_with_kernel(
    input: &DvdiqqeInput,
    kernel: Kernel,
) -> Result<DvdiqqeOutput, DvdiqqeError> {
    let (_, _, _, c, _, period, _, _, _, _, _, _, first) = dvdiqqe_prepare(input)?;
    let len = c.len();
    let wper = (period * 2) - 1;
    let warmup = first + wper;
    let mut dvdi = alloc_with_nan_prefix(len, warmup);
    let mut fast = alloc_with_nan_prefix(len, warmup);
    let mut slow = alloc_with_nan_prefix(len, warmup);
    let mut center = alloc_with_nan_prefix(len, warmup);

    let actual_kernel = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    match actual_kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            dvdiqqe_avx512(&mut dvdi, &mut fast, &mut slow, &mut center, input)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            dvdiqqe_avx2(&mut dvdi, &mut fast, &mut slow, &mut center, input)
        },
        _ => dvdiqqe_into_slices(
            &mut dvdi,
            &mut fast,
            &mut slow,
            &mut center,
            input,
            actual_kernel,
        ),
    }?;

    Ok(DvdiqqeOutput {
        dvdi,
        fast_tl: fast,
        slow_tl: slow,
        center_line: center,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dvdiqqe_avx2(
    dvdi_dst: &mut [f64],
    fast_dst: &mut [f64],
    slow_dst: &mut [f64],
    center_dst: &mut [f64],
    input: &DvdiqqeInput,
) -> Result<(), DvdiqqeError> {
    dvdiqqe_into_slices(
        dvdi_dst,
        fast_dst,
        slow_dst,
        center_dst,
        input,
        Kernel::Avx2,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dvdiqqe_avx512(
    dvdi_dst: &mut [f64],
    fast_dst: &mut [f64],
    slow_dst: &mut [f64],
    center_dst: &mut [f64],
    input: &DvdiqqeInput,
) -> Result<(), DvdiqqeError> {
    dvdiqqe_into_slices(
        dvdi_dst,
        fast_dst,
        slow_dst,
        center_dst,
        input,
        Kernel::Avx512,
    )
}

#[inline]
fn calculate_tick_volume_pine_like(open: &[f64], close: &[f64], tick: f64) -> Vec<f64> {
    let len = open.len();
    let mut out = alloc_with_nan_prefix(len, 0);
    let mut tickrng_prev = tick;

    for i in 0..len {
        let rng = close[i] - open[i];
        let tickrng = if rng.abs() < tick { tickrng_prev } else { rng };
        out[i] = (tickrng.abs() / tick).max(0.0);
        tickrng_prev = tickrng;
    }
    out
}

#[inline]
fn select_volume_pine_like(
    vol_opt: Option<&[f64]>,
    tick_vol: &[f64],
    volume_type: &str,
) -> Vec<f64> {
    let len = tick_vol.len();
    let mut out = alloc_with_nan_prefix(len, 0);
    match (volume_type.eq_ignore_ascii_case("tick"), vol_opt) {
        (true, _) => {
            for i in 0..len {
                out[i] = tick_vol[i];
            }
        }
        (false, Some(v)) => {
            for i in 0..len {
                out[i] = if v[i].is_finite() { v[i] } else { tick_vol[i] };
            }
        }
        (false, None) => {
            for i in 0..len {
                out[i] = tick_vol[i];
            }
        }
    }
    out
}

#[inline]
fn build_pvi_nvi_pine_like(close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut pvi = alloc_with_nan_prefix(len, 0);
    let mut nvi = alloc_with_nan_prefix(len, 0);
    let mut pvi_prev = 0.0;
    let mut nvi_prev = 0.0;
    let mut prev_vol = 0.0;
    let mut prev_x = 0.0;

    for i in 0..len {
        if volume[i] > prev_vol {
            pvi_prev += close[i] - prev_x;
        }
        if volume[i] < prev_vol {
            nvi_prev -= close[i] - prev_x;
        }
        pvi[i] = pvi_prev;
        nvi[i] = nvi_prev;
        prev_vol = volume[i];
        prev_x = close[i];
    }
    (pvi, nvi)
}

fn calculate_dvdi(
    close: &[f64],
    volume: &[f64],
    period: usize,
    smoothing_period: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, DvdiqqeError> {
    let len = close.len();

    let (pvi, nvi) = build_pvi_nvi_pine_like(close, volume);

    let pvi_ema_params = EmaParams {
        period: Some(period),
    };
    let pvi_ema_input = EmaInput::from_slice(&pvi, pvi_ema_params);
    let pvi_ema = ema_with_kernel(&pvi_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let nvi_ema_params = EmaParams {
        period: Some(period),
    };
    let nvi_ema_input = EmaInput::from_slice(&nvi, nvi_ema_params);
    let nvi_ema = ema_with_kernel(&nvi_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let mut pvi_div = alloc_with_nan_prefix(len, 0);
    let mut nvi_div = alloc_with_nan_prefix(len, 0);

    for i in 0..len {
        pvi_div[i] = pvi[i] - pvi_ema.values[i];
        nvi_div[i] = nvi[i] - nvi_ema.values[i];
    }

    let pdiv_ema_params = EmaParams {
        period: Some(smoothing_period),
    };
    let pdiv_ema_input = EmaInput::from_slice(&pvi_div, pdiv_ema_params);
    let pdiv_ema = ema_with_kernel(&pdiv_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let ndiv_ema_params = EmaParams {
        period: Some(smoothing_period),
    };
    let ndiv_ema_input = EmaInput::from_slice(&nvi_div, ndiv_ema_params);
    let ndiv_ema = ema_with_kernel(&ndiv_ema_input, kernel)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let mut dvdi = alloc_with_nan_prefix(len, 0);
    for i in 0..len {
        dvdi[i] = pdiv_ema.values[i] - ndiv_ema.values[i];
    }

    Ok(dvdi)
}

fn calculate_trailing_levels(
    dvdi: &[f64],
    period: usize,
    fast_mult: f64,
    slow_mult: f64,
) -> Result<(Vec<f64>, Vec<f64>), DvdiqqeError> {
    let len = dvdi.len();
    let wper = (period * 2) - 1;

    let mut ranges = alloc_with_nan_prefix(len, 1);
    for i in 1..len {
        ranges[i] = (dvdi[i] - dvdi[i - 1]).abs();
    }

    let range_ema_params = EmaParams { period: Some(wper) };
    let range_ema_input = EmaInput::from_slice(&ranges, range_ema_params);
    let avg_range = ema_with_kernel(&range_ema_input, Kernel::Auto)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let smooth_range_params = EmaParams { period: Some(wper) };
    let smooth_range_input = EmaInput::from_slice(&avg_range.values, smooth_range_params);
    let smooth_range = ema_with_kernel(&smooth_range_input, Kernel::Auto)
        .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?;

    let first_valid = dvdi.iter().position(|&x| x.is_finite()).unwrap_or(len);

    let mut fast_tl = alloc_with_nan_prefix(len, first_valid);
    let mut slow_tl = alloc_with_nan_prefix(len, first_valid);

    if first_valid < len {
        fast_tl[first_valid] = dvdi[first_valid];
        slow_tl[first_valid] = dvdi[first_valid];

        for i in (first_valid + 1)..len {
            let fast_range = smooth_range.values[i] * fast_mult;
            let slow_range = smooth_range.values[i] * slow_mult;

            if dvdi[i] > fast_tl[i - 1] {
                let new_val = dvdi[i] - fast_range;
                fast_tl[i] = if new_val < fast_tl[i - 1] {
                    fast_tl[i - 1]
                } else {
                    new_val
                };
            } else {
                let new_val = dvdi[i] + fast_range;
                fast_tl[i] = if new_val > fast_tl[i - 1] {
                    fast_tl[i - 1]
                } else {
                    new_val
                };
            }

            if dvdi[i] > slow_tl[i - 1] {
                let new_val = dvdi[i] - slow_range;
                slow_tl[i] = if new_val < slow_tl[i - 1] {
                    slow_tl[i - 1]
                } else {
                    new_val
                };
            } else {
                let new_val = dvdi[i] + slow_range;
                slow_tl[i] = if new_val > slow_tl[i - 1] {
                    slow_tl[i - 1]
                } else {
                    new_val
                };
            }
        }
    }

    Ok((fast_tl, slow_tl))
}

fn calculate_cumulative_mean(dvdi: &[f64]) -> Vec<f64> {
    let len = dvdi.len();
    let first_valid = dvdi.iter().position(|&x| x.is_finite()).unwrap_or(len);
    let mut center = alloc_with_nan_prefix(len, first_valid);
    let mut sum = 0.0;
    let mut cnt = 0.0;

    for i in first_valid..len {
        if dvdi[i].is_finite() {
            sum += dvdi[i];
            cnt += 1.0;
        }
        center[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
    }
    center
}

pub fn dvdiqqe_into_slices(
    dvdi_dst: &mut [f64],
    fast_tl_dst: &mut [f64],
    slow_tl_dst: &mut [f64],
    center_dst: &mut [f64],
    input: &DvdiqqeInput,
    kernel: Kernel,
) -> Result<(), DvdiqqeError> {
    let (o, h, l, c, v, period, smoothing, fast, slow, vt, ct, tick, first) =
        dvdiqqe_prepare(input)?;

    let len = c.len();
    if dvdi_dst.len() != len {
        return Err(DvdiqqeError::OutputLengthMismatch {
            expected: len,
            got: dvdi_dst.len(),
        });
    }
    if fast_tl_dst.len() != len {
        return Err(DvdiqqeError::OutputLengthMismatch {
            expected: len,
            got: fast_tl_dst.len(),
        });
    }
    if slow_tl_dst.len() != len {
        return Err(DvdiqqeError::OutputLengthMismatch {
            expected: len,
            got: slow_tl_dst.len(),
        });
    }
    if center_dst.len() != len {
        return Err(DvdiqqeError::OutputLengthMismatch {
            expected: len,
            got: center_dst.len(),
        });
    }

    dvdiqqe_compute_into(
        o,
        h,
        l,
        c,
        v,
        period,
        smoothing,
        fast,
        slow,
        vt,
        ct,
        tick,
        first,
        kernel,
        dvdi_dst,
        fast_tl_dst,
        slow_tl_dst,
        center_dst,
    )
}

pub fn dvdiqqe_into(
    input: &DvdiqqeInput,
    dvdi_out: &mut [f64],
    fast_tl_out: &mut [f64],
    slow_tl_out: &mut [f64],
    center_out: &mut [f64],
) -> Result<(), DvdiqqeError> {
    dvdiqqe_into_slices(
        dvdi_out,
        fast_tl_out,
        slow_tl_out,
        center_out,
        input,
        Kernel::Auto,
    )
}

#[inline]
pub fn dvdiqqe_into_flat(
    dst_4xlen: &mut [f64],
    input: &DvdiqqeInput,
    k: Kernel,
) -> Result<(), DvdiqqeError> {
    let (_, _, _, c, _, _, _, _, _, _, _, _, _) = dvdiqqe_prepare(input)?;
    let len = c.len();
    let expected = len
        .checked_mul(4)
        .ok_or_else(|| DvdiqqeError::InvalidInput("4*len overflow".into()))?;
    if dst_4xlen.len() != expected {
        return Err(DvdiqqeError::OutputLengthMismatch {
            expected,
            got: dst_4xlen.len(),
        });
    }
    let (dvdi, rest) = dst_4xlen.split_at_mut(len);
    let (fast, rest) = rest.split_at_mut(len);
    let (slow, cent) = rest.split_at_mut(len);
    dvdiqqe_into_slices(dvdi, fast, slow, cent, input, k)
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DvdiqqeBuilder {
    period: Option<usize>,
    smoothing: Option<usize>,
    fast: Option<f64>,
    slow: Option<f64>,
    volume_type: Option<&'static str>,
    center_type: Option<&'static str>,
    tick: Option<f64>,
    kernel: Kernel,
}

impl DvdiqqeBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }

    pub fn smoothing(mut self, n: usize) -> Self {
        self.smoothing = Some(n);
        self
    }

    pub fn fast(mut self, mult: f64) -> Self {
        self.fast = Some(mult);
        self
    }

    pub fn slow(mut self, mult: f64) -> Self {
        self.slow = Some(mult);
        self
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn volume_type(mut self, vt: &'static str) -> Self {
        self.volume_type = Some(vt);
        self
    }

    pub fn center_type(mut self, ct: &'static str) -> Self {
        self.center_type = Some(ct);
        self
    }

    pub fn tick_size(mut self, ts: f64) -> Self {
        self.tick = Some(ts);
        self
    }

    pub fn apply_slice(
        self,
        o: &[f64],
        h: &[f64],
        l: &[f64],
        c: &[f64],
        v: Option<&[f64]>,
    ) -> Result<DvdiqqeOutput, DvdiqqeError> {
        let p = DvdiqqeParams {
            period: self.period,
            smoothing_period: self.smoothing,
            fast_multiplier: self.fast,
            slow_multiplier: self.slow,
            volume_type: self.volume_type.map(String::from),
            center_type: self.center_type.map(String::from),
            tick_size: self.tick,
        };
        let i = DvdiqqeInput::from_slices(o, h, l, c, v, p);
        dvdiqqe_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(self, c: &Candles) -> Result<DvdiqqeOutput, DvdiqqeError> {
        let p = DvdiqqeParams {
            period: self.period,
            smoothing_period: self.smoothing,
            fast_multiplier: self.fast,
            slow_multiplier: self.slow,
            volume_type: self.volume_type.map(str::to_string),
            center_type: self.center_type.map(str::to_string),
            tick_size: self.tick,
        };
        let i = DvdiqqeInput::from_candles(c, p);
        dvdiqqe_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DvdiqqeStream, DvdiqqeError> {
        let p = DvdiqqeParams {
            period: self.period,
            smoothing_period: self.smoothing,
            fast_multiplier: self.fast,
            slow_multiplier: self.slow,
            volume_type: self.volume_type.map(str::to_string),
            center_type: self.center_type.map(str::to_string),
            tick_size: self.tick,
        };
        DvdiqqeStream::try_new(p)
    }
}

#[derive(Clone, Debug)]
pub struct DvdiqqeBatchRange {
    pub period: (usize, usize, usize),
    pub smoothing_period: (usize, usize, usize),
    pub fast_multiplier: (f64, f64, f64),
    pub slow_multiplier: (f64, f64, f64),
}

impl Default for DvdiqqeBatchRange {
    fn default() -> Self {
        Self {
            period: (13, 262, 1),
            smoothing_period: (6, 6, 0),
            fast_multiplier: (2.618, 2.618, 0.0),
            slow_multiplier: (4.236, 4.236, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DvdiqqeBatchOutput {
    pub dvdi_values: Vec<f64>,
    pub fast_tl_values: Vec<f64>,
    pub slow_tl_values: Vec<f64>,
    pub center_values: Vec<f64>,
    pub combos: Vec<DvdiqqeParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DvdiqqeBatchOutput {
    pub fn values_for(&self, params: &DvdiqqeParams) -> Option<DvdiqqeBatchValues> {
        self.combos
            .iter()
            .position(|p| {
                p.period == params.period
                    && p.smoothing_period == params.smoothing_period
                    && p.fast_multiplier == params.fast_multiplier
                    && p.slow_multiplier == params.slow_multiplier
            })
            .map(|idx| {
                let start = idx * self.cols;
                let end = start + self.cols;
                DvdiqqeBatchValues {
                    dvdi: &self.dvdi_values[start..end],
                    fast_tl: &self.fast_tl_values[start..end],
                    slow_tl: &self.slow_tl_values[start..end],
                    center: &self.center_values[start..end],
                }
            })
    }

    pub fn row_for_params(&self, params: &DvdiqqeParams) -> Option<Vec<f64>> {
        self.combos
            .iter()
            .position(|p| {
                p.period == params.period
                    && p.smoothing_period == params.smoothing_period
                    && p.fast_multiplier == params.fast_multiplier
                    && p.slow_multiplier == params.slow_multiplier
            })
            .map(|idx| {
                let start = idx * self.cols;
                let end = start + self.cols;

                let mut row = Vec::with_capacity(self.cols * 4);
                row.extend_from_slice(&self.dvdi_values[start..end]);
                row.extend_from_slice(&self.fast_tl_values[start..end]);
                row.extend_from_slice(&self.slow_tl_values[start..end]);
                row.extend_from_slice(&self.center_values[start..end]);
                row
            })
    }
}

#[derive(Debug)]
pub struct DvdiqqeBatchValues<'a> {
    pub dvdi: &'a [f64],
    pub fast_tl: &'a [f64],
    pub slow_tl: &'a [f64],
    pub center: &'a [f64],
}

#[derive(Clone, Debug)]
pub struct DvdiqqeBatchOutputFlat {
    pub values: Vec<f64>,
    pub combos: Vec<DvdiqqeParams>,
    pub rows: usize,
    pub cols: usize,
    pub series: usize,
}

impl DvdiqqeBatchOutputFlat {
    #[inline]
    pub fn slice_series(&self, s: usize) -> &[f64] {
        assert!(s < self.series);
        let plane = self.rows * self.cols;
        &self.values[s * plane..(s + 1) * plane]
    }
}

#[derive(Clone, Debug)]
pub struct DvdiqqeBatchBuilder {
    range: DvdiqqeBatchRange,
    kernel: Kernel,
    volume_type: String,
    center_type: String,
    tick_size: f64,
}

impl Default for DvdiqqeBatchBuilder {
    fn default() -> Self {
        Self {
            range: DvdiqqeBatchRange::default(),
            kernel: Kernel::Auto,
            volume_type: "default".to_string(),
            center_type: "dynamic".to_string(),
            tick_size: 0.01,
        }
    }
}

impl DvdiqqeBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }

    pub fn smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_period = (start, end, step);
        self
    }

    pub fn smoothing_static(mut self, s: usize) -> Self {
        self.range.smoothing_period = (s, s, 0);
        self
    }

    pub fn fast_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fast_multiplier = (start, end, step);
        self
    }

    pub fn fast_static(mut self, f: f64) -> Self {
        self.range.fast_multiplier = (f, f, 0.0);
        self
    }

    pub fn slow_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.slow_multiplier = (start, end, step);
        self
    }

    pub fn slow_static(mut self, s: f64) -> Self {
        self.range.slow_multiplier = (s, s, 0.0);
        self
    }

    pub fn volume_type(mut self, vt: &str) -> Self {
        self.volume_type = vt.to_string();
        self
    }

    pub fn center_type(mut self, ct: &str) -> Self {
        self.center_type = ct.to_string();
        self
    }

    pub fn tick_size(mut self, ts: f64) -> Self {
        self.tick_size = ts;
        self
    }

    pub fn apply_candles(self, candles: &Candles) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        dvdiqqe_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            Some(&candles.volume),
            &self.range,
            self.kernel,
            &self.volume_type,
            &self.center_type,
            self.tick_size,
        )
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        dvdiqqe_batch_with_kernel(
            open,
            high,
            low,
            close,
            volume,
            &self.range,
            self.kernel,
            &self.volume_type,
            &self.center_type,
            self.tick_size,
        )
    }

    pub fn with_default_candles(candles: &Candles) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        let builder = Self::default();
        dvdiqqe_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            Some(&candles.volume),
            &builder.range,
            builder.kernel,
            &builder.volume_type,
            &builder.center_type,
            builder.tick_size,
        )
    }

    pub fn with_default_slice(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: Option<&[f64]>,
    ) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
        let builder = Self::default();
        dvdiqqe_batch_with_kernel(
            open,
            high,
            low,
            close,
            volume,
            &builder.range,
            builder.kernel,
            &builder.volume_type,
            &builder.center_type,
            builder.tick_size,
        )
    }
}

#[inline(always)]
fn expand_grid(r: &DvdiqqeBatchRange) -> Result<Vec<DvdiqqeParams>, DvdiqqeError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, DvdiqqeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(n) => cur = n,
                    None => break,
                }
            }
        } else {
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur < step {
                    break;
                }
                cur -= step;
                if cur == usize::MAX {
                    break;
                }
                if cur == 0 && end > 0 {
                    break;
                }
            }
        }
        if v.is_empty() {
            return Err(DvdiqqeError::InvalidRangeUsize { start, end, step });
        }
        Ok(v)
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, DvdiqqeError> {
        if step == 0.0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut curr = start;
            while curr <= end + 1e-12 {
                v.push(curr);
                curr += step;
            }
        } else {
            let mut curr = start;
            let step_abs = step.abs();
            while curr >= end - 1e-12 {
                v.push(curr);
                curr -= step_abs;
                if !curr.is_finite() {
                    break;
                }
            }
        }
        if v.is_empty() {
            return Err(DvdiqqeError::InvalidRangeF64 { start, end, step });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let smoothings = axis_usize(r.smoothing_period)?;
    let fasts = axis_f64(r.fast_multiplier)?;
    let slows = axis_f64(r.slow_multiplier)?;

    let mut combos = Vec::new();
    for &p in &periods {
        for &s in &smoothings {
            for &f in &fasts {
                for &sl in &slows {
                    combos.push(DvdiqqeParams {
                        period: Some(p),
                        smoothing_period: Some(s),
                        fast_multiplier: Some(f),
                        slow_multiplier: Some(sl),
                        volume_type: None,
                        center_type: None,
                        tick_size: None,
                    });
                }
            }
        }
    }
    if combos.is_empty() {
        return Err(DvdiqqeError::InvalidInput("empty sweep".into()));
    }
    Ok(combos)
}

pub fn dvdiqqe_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    k: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    if !matches!(k, Kernel::Auto) && !k.is_batch() {
        return Err(DvdiqqeError::InvalidKernelForBatch(k));
    }
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => kernel,
    };
    dvdiqqe_batch_par_slice(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        simd,
        volume_type,
        center_type,
        tick_size,
    )
}

pub fn dvdiqqe_batch_with_kernel_flat(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    k: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutputFlat, DvdiqqeError> {
    if !matches!(k, Kernel::Auto) && !k.is_batch() {
        return Err(DvdiqqeError::InvalidKernelForBatch(k));
    }
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => kernel,
    };
    dvdiqqe_batch_inner_flat(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        simd,
        true,
        volume_type,
        center_type,
        tick_size,
    )
}

#[inline(always)]
pub fn dvdiqqe_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    dvdiqqe_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kern,
        false,
        volume_type,
        center_type,
        tick_size,
    )
}

#[inline(always)]
pub fn dvdiqqe_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    dvdiqqe_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kern,
        true,
        volume_type,
        center_type,
        tick_size,
    )
}

fn dvdiqqe_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    parallel: bool,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutput, DvdiqqeError> {
    let flat = dvdiqqe_batch_inner_flat(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kern,
        parallel,
        volume_type,
        center_type,
        tick_size,
    )?;

    let plane = flat.rows * flat.cols;
    Ok(DvdiqqeBatchOutput {
        dvdi_values: flat.values[0..plane].to_vec(),
        fast_tl_values: flat.values[plane..2 * plane].to_vec(),
        slow_tl_values: flat.values[2 * plane..3 * plane].to_vec(),
        center_values: flat.values[3 * plane..4 * plane].to_vec(),
        combos: flat.combos,
        rows: flat.rows,
        cols: flat.cols,
    })
}

fn dvdiqqe_batch_inner_flat(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    parallel: bool,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
) -> Result<DvdiqqeBatchOutputFlat, DvdiqqeError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(DvdiqqeError::EmptyInputData);
    }

    let series = 4usize;

    let rows_cols = rows
        .checked_mul(cols)
        .ok_or_else(|| DvdiqqeError::InvalidInput("rows*cols overflow".into()))?;
    let _ = series
        .checked_mul(rows_cols)
        .ok_or_else(|| DvdiqqeError::InvalidInput("series*rows*cols overflow".into()))?;
    let mut buf_mu = make_uninit_matrix(series * rows, cols);

    let first = close
        .iter()
        .position(|x| x.is_finite())
        .ok_or(DvdiqqeError::AllValuesNaN)?;
    let warms: Vec<usize> = combos
        .iter()
        .map(|p| first + p.period.unwrap_or(13) - 1)
        .collect();

    for s in 0..series {
        let off = s * rows * cols;
        let plane = &mut buf_mu[off..off + rows * cols];
        init_matrix_prefixes(plane, cols, &warms);
    }

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let tick_vol_once = calculate_tick_volume_pine_like(open, close, tick_size);
    let sel_vol_once = select_volume_pine_like(volume, &tick_vol_once, volume_type);
    let (pvi_stream, nvi_stream) = build_pvi_nvi_pine_like(close, &sel_vol_once);

    let process_row = |row: usize, out_slice: &mut [f64]| -> Result<(), DvdiqqeError> {
        let prm = &combos[row];
        let params = DvdiqqeParams {
            period: prm.period,
            smoothing_period: prm.smoothing_period,
            fast_multiplier: prm.fast_multiplier,
            slow_multiplier: prm.slow_multiplier,
            volume_type: Some(volume_type.to_string()),
            center_type: Some(center_type.to_string()),
            tick_size: Some(tick_size),
        };
        let input = DvdiqqeInput::from_slices(open, high, low, close, volume, params);
        let (_o, _h, _l, c, _v, period, smoothing, fast, slow, _vt, ct, _tick, first_local) =
            dvdiqqe_prepare(&input)?;

        let plane = rows * cols;
        let (dvdi_plane, rest) = out_slice.split_at_mut(plane);
        let (fast_plane, rest) = rest.split_at_mut(plane);
        let (slow_plane, center_plane) = rest.split_at_mut(plane);

        let dvdi_dst = &mut dvdi_plane[row * cols..(row + 1) * cols];
        let fast_dst = &mut fast_plane[row * cols..(row + 1) * cols];
        let slow_dst = &mut slow_plane[row * cols..(row + 1) * cols];
        let center_dst = &mut center_plane[row * cols..(row + 1) * cols];

        let pvi_ema = {
            let prm = EmaParams {
                period: Some(period),
            };
            let inp = EmaInput::from_slice(&pvi_stream, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };
        let nvi_ema = {
            let prm = EmaParams {
                period: Some(period),
            };
            let inp = EmaInput::from_slice(&nvi_stream, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };

        let mut pdiv = alloc_with_nan_prefix(cols, 0);
        let mut ndiv = alloc_with_nan_prefix(cols, 0);
        for i in 0..cols {
            pdiv[i] = pvi_stream[i] - pvi_ema.values[i];
            ndiv[i] = nvi_stream[i] - nvi_ema.values[i];
        }
        let pdiv_ema = {
            let prm = EmaParams {
                period: Some(smoothing),
            };
            let inp = EmaInput::from_slice(&pdiv, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };
        let ndiv_ema = {
            let prm = EmaParams {
                period: Some(smoothing),
            };
            let inp = EmaInput::from_slice(&ndiv, prm);
            ema_with_kernel(&inp, kern).map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };

        let wper = (period * 2) - 1;
        let warmup = first_local + wper;

        for i in 0..warmup.min(cols) {
            dvdi_dst[i] = f64::NAN;
            fast_dst[i] = f64::NAN;
            slow_dst[i] = f64::NAN;
        }

        let mut ranges = alloc_with_nan_prefix(cols, 1);
        if cols > 0 {
            dvdi_dst[0] = pdiv_ema.values[0] - ndiv_ema.values[0];
            for i in 1..cols {
                let dvdi_i = pdiv_ema.values[i] - ndiv_ema.values[i];
                ranges[i] = (dvdi_i - dvdi_dst[i - 1]).abs();
                dvdi_dst[i] = dvdi_i;
            }
        }

        let avg_range = {
            let prm = EmaParams { period: Some(wper) };
            let inp = EmaInput::from_slice(&ranges, prm);
            ema_with_kernel(&inp, Kernel::Auto)
                .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };
        let smooth_range = {
            let prm = EmaParams { period: Some(wper) };
            let inp = EmaInput::from_slice(&avg_range.values, prm);
            ema_with_kernel(&inp, Kernel::Auto)
                .map_err(|e| DvdiqqeError::EmaError(e.to_string()))?
        };

        if warmup < cols {
            fast_dst[warmup] = dvdi_dst[warmup];
            slow_dst[warmup] = dvdi_dst[warmup];
            for i in (warmup + 1)..cols {
                let fr = smooth_range.values[i] * fast;
                let sr = smooth_range.values[i] * slow;
                if dvdi_dst[i] > fast_dst[i - 1] {
                    let nv = dvdi_dst[i] - fr;
                    fast_dst[i] = if nv < fast_dst[i - 1] {
                        fast_dst[i - 1]
                    } else {
                        nv
                    };
                } else {
                    let nv = dvdi_dst[i] + fr;
                    fast_dst[i] = if nv > fast_dst[i - 1] {
                        fast_dst[i - 1]
                    } else {
                        nv
                    };
                }

                if dvdi_dst[i] > slow_dst[i - 1] {
                    let nv = dvdi_dst[i] - sr;
                    slow_dst[i] = if nv < slow_dst[i - 1] {
                        slow_dst[i - 1]
                    } else {
                        nv
                    };
                } else {
                    let nv = dvdi_dst[i] + sr;
                    slow_dst[i] = if nv > slow_dst[i - 1] {
                        slow_dst[i - 1]
                    } else {
                        nv
                    };
                }
            }
        }

        for i in 0..warmup.min(cols) {
            center_dst[i] = f64::NAN;
        }
        if ct.eq_ignore_ascii_case("dynamic") {
            let mut sum = 0.0f64;
            let mut cnt = 0.0f64;
            for i in warmup..cols {
                let v = dvdi_dst[i];
                if v.is_finite() {
                    sum += v;
                    cnt += 1.0;
                }
                center_dst[i] = if cnt > 0.0 { sum / cnt } else { f64::NAN };
            }
        } else {
            for i in warmup..cols {
                center_dst[i] = 0.0;
            }
        }

        Ok(())
    };

    for r in 0..rows {
        process_row(r, out)?;
    }

    let _ = parallel;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(DvdiqqeBatchOutputFlat {
        values,
        combos,
        rows,
        cols,
        series,
    })
}

fn dvdiqqe_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    sweep: &DvdiqqeBatchRange,
    kern: Kernel,
    parallel: bool,
    volume_type: &str,
    center_type: &str,
    tick_size: f64,
    dvdi_out: &mut [f64],
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    center_out: &mut [f64],
) -> Result<Vec<DvdiqqeParams>, DvdiqqeError> {
    let combos = expand_grid(sweep)?;
    let cols = close.len();

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            use rayon::prelude::*;

            let results: Result<Vec<_>, DvdiqqeError> = combos
                .par_iter()
                .map(|params| {
                    let mut full_params = params.clone();
                    full_params.volume_type = Some(volume_type.to_string());
                    full_params.center_type = Some(center_type.to_string());
                    full_params.tick_size = Some(tick_size);

                    let input =
                        DvdiqqeInput::from_slices(open, high, low, close, volume, full_params);
                    dvdiqqe_with_kernel(&input, kern)
                })
                .collect();

            let results = results?;

            for (row, output) in results.iter().enumerate() {
                let start = row * cols;
                let end = start + cols;
                dvdi_out[start..end].copy_from_slice(&output.dvdi);
                fast_out[start..end].copy_from_slice(&output.fast_tl);
                slow_out[start..end].copy_from_slice(&output.slow_tl);
                center_out[start..end].copy_from_slice(&output.center_line);
            }
        } else {
            for (row, params) in combos.iter().enumerate() {
                let mut full_params = params.clone();
                full_params.volume_type = Some(volume_type.to_string());
                full_params.center_type = Some(center_type.to_string());
                full_params.tick_size = Some(tick_size);

                let input = DvdiqqeInput::from_slices(open, high, low, close, volume, full_params);
                let output = dvdiqqe_with_kernel(&input, kern)?;

                let start = row * cols;
                let end = start + cols;
                dvdi_out[start..end].copy_from_slice(&output.dvdi);
                fast_out[start..end].copy_from_slice(&output.fast_tl);
                slow_out[start..end].copy_from_slice(&output.slow_tl);
                center_out[start..end].copy_from_slice(&output.center_line);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (row, params) in combos.iter().enumerate() {
            let mut full_params = params.clone();
            full_params.volume_type = Some(volume_type.to_string());
            full_params.center_type = Some(center_type.to_string());
            full_params.tick_size = Some(tick_size);

            let input = DvdiqqeInput::from_slices(open, high, low, close, volume, full_params);
            let output = dvdiqqe_with_kernel(&input, kern)?;

            let start = row * cols;
            let end = start + cols;
            dvdi_out[start..end].copy_from_slice(&output.dvdi);
            fast_out[start..end].copy_from_slice(&output.fast_tl);
            slow_out[start..end].copy_from_slice(&output.slow_tl);
            center_out[start..end].copy_from_slice(&output.center_line);
        }
    }

    Ok(combos)
}

pub struct DvdiqqeStream {
    period: usize,
    smoothing_period: usize,
    fast_mult: f64,
    slow_mult: f64,
    volume_type: String,
    center_type: String,
    tick_size: f64,

    alpha_pvi: f64,
    alpha_div: f64,
    alpha_rng: f64,
    inv_tick: f64,
    use_tick_only: bool,
    warmup_needed: usize,

    prev_close: f64,
    prev_sel_vol: f64,
    tickrng_prev: f64,

    pvi: f64,
    nvi: f64,
    pvi_ema: f64,
    nvi_ema: f64,
    ema_pvi_inited: bool,

    pdiv_ema: f64,
    ndiv_ema: f64,
    ema_div_inited: bool,

    dvdi_prev: f64,
    rng_ema1: f64,
    rng_ema2: f64,
    ema_rng_inited: bool,

    fast_tl_prev: f64,
    slow_tl_prev: f64,
    tl_seeded: bool,

    count: usize,
    center_sum: f64,
    center_count: f64,
}

impl DvdiqqeStream {
    pub fn try_new(params: DvdiqqeParams) -> Result<Self, DvdiqqeError> {
        let period = params.period.unwrap_or(13);
        let smoothing_period = params.smoothing_period.unwrap_or(6);
        if period == 0 {
            return Err(DvdiqqeError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if smoothing_period == 0 {
            return Err(DvdiqqeError::InvalidSmoothing { smoothing: 0 });
        }

        let fast_mult = params.fast_multiplier.unwrap_or(2.618);
        let slow_mult = params.slow_multiplier.unwrap_or(4.236);
        if !(fast_mult.is_finite() && fast_mult > 0.0) {
            return Err(DvdiqqeError::InvalidMultiplier {
                multiplier: fast_mult,
                which: "fast".into(),
            });
        }
        if !(slow_mult.is_finite() && slow_mult > 0.0) {
            return Err(DvdiqqeError::InvalidMultiplier {
                multiplier: slow_mult,
                which: "slow".into(),
            });
        }

        let volume_type = params.volume_type.unwrap_or_else(|| "default".to_string());
        let center_type = params.center_type.unwrap_or_else(|| "dynamic".to_string());
        let tick_size = params.tick_size.unwrap_or(0.01);
        if !(tick_size.is_finite() && tick_size > 0.0) {
            return Err(DvdiqqeError::InvalidTick { tick: tick_size });
        }

        let alpha_pvi = 2.0 / (period as f64 + 1.0);
        let alpha_div = 2.0 / (smoothing_period as f64 + 1.0);

        let alpha_rng = 1.0 / (period as f64);

        Ok(Self {
            period,
            smoothing_period,
            fast_mult,
            slow_mult,
            use_tick_only: volume_type.eq_ignore_ascii_case("tick"),
            volume_type,
            center_type,
            tick_size,
            inv_tick: 1.0 / tick_size,
            alpha_pvi,
            alpha_div,
            alpha_rng,
            warmup_needed: period * 2,

            prev_close: 0.0,
            prev_sel_vol: 0.0,
            tickrng_prev: tick_size,

            pvi: 0.0,
            nvi: 0.0,
            pvi_ema: 0.0,
            nvi_ema: 0.0,
            ema_pvi_inited: false,

            pdiv_ema: 0.0,
            ndiv_ema: 0.0,
            ema_div_inited: false,

            dvdi_prev: 0.0,
            rng_ema1: 0.0,
            rng_ema2: 0.0,
            ema_rng_inited: false,

            fast_tl_prev: f64::NAN,
            slow_tl_prev: f64::NAN,
            tl_seeded: false,

            count: 0,
            center_sum: 0.0,
            center_count: 0.0,
        })
    }

    pub fn update(
        &mut self,
        open: f64,
        _high: f64,
        _low: f64,
        close: f64,
        volume: f64,
    ) -> Option<DvdiqqeStreamOutput> {
        let rng = close - open;
        let tickrng = if rng.abs() < self.tick_size {
            self.tickrng_prev
        } else {
            rng
        };
        let tick_vol = (tickrng.abs() * self.inv_tick).max(0.0);
        self.tickrng_prev = tickrng;

        let sel_vol = if self.use_tick_only {
            tick_vol
        } else if volume.is_finite() {
            volume
        } else {
            tick_vol
        };

        let d_close = close - self.prev_close;
        if sel_vol > self.prev_sel_vol {
            self.pvi += d_close;
        } else if sel_vol < self.prev_sel_vol {
            self.nvi -= d_close;
        }
        self.prev_sel_vol = sel_vol;
        self.prev_close = close;

        if !self.ema_pvi_inited {
            self.pvi_ema = self.pvi;
            self.nvi_ema = self.nvi;
            self.ema_pvi_inited = true;
        } else {
            self.pvi_ema += self.alpha_pvi * (self.pvi - self.pvi_ema);
            self.nvi_ema += self.alpha_pvi * (self.nvi - self.nvi_ema);
        }

        let pdiv = self.pvi - self.pvi_ema;
        let ndiv = self.nvi - self.nvi_ema;

        if !self.ema_div_inited {
            self.pdiv_ema = pdiv;
            self.ndiv_ema = ndiv;
            self.ema_div_inited = true;
        } else {
            self.pdiv_ema += self.alpha_div * (pdiv - self.pdiv_ema);
            self.ndiv_ema += self.alpha_div * (ndiv - self.ndiv_ema);
        }

        let dvdi = self.pdiv_ema - self.ndiv_ema;

        let step_rng = (dvdi - self.dvdi_prev).abs();
        if !self.ema_rng_inited {
            self.rng_ema1 = step_rng;
            self.rng_ema2 = self.rng_ema1;
            self.ema_rng_inited = true;
        } else {
            self.rng_ema1 += self.alpha_rng * (step_rng - self.rng_ema1);
            self.rng_ema2 += self.alpha_rng * (self.rng_ema1 - self.rng_ema2);
        }
        self.dvdi_prev = dvdi;

        self.count = self.count.saturating_add(1);
        if self.count < self.warmup_needed {
            return None;
        }

        let smooth_rng = self.rng_ema2;
        let fr = smooth_rng * self.fast_mult;
        let sr = smooth_rng * self.slow_mult;

        if !self.tl_seeded {
            self.fast_tl_prev = dvdi;
            self.slow_tl_prev = dvdi;
            self.tl_seeded = true;

            if self.center_type.eq_ignore_ascii_case("dynamic") && dvdi.is_finite() {
                self.center_sum = dvdi;
                self.center_count = 1.0;
            }
            return Some(DvdiqqeStreamOutput {
                dvdi,
                fast_tl: self.fast_tl_prev,
                slow_tl: self.slow_tl_prev,
                center_line: if self.center_type.eq_ignore_ascii_case("static") {
                    0.0
                } else if self.center_count > 0.0 {
                    self.center_sum / self.center_count
                } else {
                    f64::NAN
                },
            });
        }

        let fast_tl = if dvdi > self.fast_tl_prev {
            let nv = dvdi - fr;
            if nv < self.fast_tl_prev {
                self.fast_tl_prev
            } else {
                nv
            }
        } else {
            let nv = dvdi + fr;
            if nv > self.fast_tl_prev {
                self.fast_tl_prev
            } else {
                nv
            }
        };

        let slow_tl = if dvdi > self.slow_tl_prev {
            let nv = dvdi - sr;
            if nv < self.slow_tl_prev {
                self.slow_tl_prev
            } else {
                nv
            }
        } else {
            let nv = dvdi + sr;
            if nv > self.slow_tl_prev {
                self.slow_tl_prev
            } else {
                nv
            }
        };

        self.fast_tl_prev = fast_tl;
        self.slow_tl_prev = slow_tl;

        let center_val = if self.center_type.eq_ignore_ascii_case("static") {
            0.0
        } else {
            if dvdi.is_finite() {
                self.center_sum += dvdi;
                self.center_count += 1.0;
            }
            if self.center_count > 0.0 {
                self.center_sum / self.center_count
            } else {
                f64::NAN
            }
        };

        Some(DvdiqqeStreamOutput {
            dvdi,
            fast_tl,
            slow_tl,
            center_line: center_val,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DvdiqqeStreamOutput {
    pub dvdi: f64,
    pub fast_tl: f64,
    pub slow_tl: f64,
    pub center_line: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::{Candles, read_candles_from_vortex};
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    #[test]
    fn test_dvdiqqe_accuracy_scalar() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_scalar", Kernel::Scalar)
    }

    #[test]
    fn test_dvdiqqe_accuracy_auto() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_auto", Kernel::Auto)
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_dvdiqqe_accuracy_avx2() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_avx2", Kernel::Avx2)
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn test_dvdiqqe_accuracy_avx512() -> Result<(), Box<dyn Error>> {
        check_dvdiqqe_accuracy("test_dvdiqqe_accuracy_avx512", Kernel::Avx512)
    }

    #[test]
    fn test_dvdiqqe_with_csv_data() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input)?;

        let len = candles.close.len();
        assert_eq!(result.dvdi.len(), len);
        assert_eq!(result.fast_tl.len(), len);
        assert_eq!(result.slow_tl.len(), len);
        assert_eq!(result.center_line.len(), len);

        let warmup = 25;
        for i in 0..warmup.min(len) {
            assert!(
                result.dvdi[i].is_nan(),
                "Expected NaN in warmup at index {}",
                i
            );
        }
        for i in warmup..len {
            assert!(
                result.dvdi[i].is_finite(),
                "Expected finite value after warmup at index {}",
                i
            );
        }

        Ok(())
    }

    #[test]
    fn test_dvdiqqe_into_matches_api_v2() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = DvdiqqeInput::with_default_candles(&candles);
        let baseline = dvdiqqe(&input)?;

        let len = candles.close.len();
        let mut dvdi = vec![0.0; len];
        let mut fast = vec![0.0; len];
        let mut slow = vec![0.0; len];
        let mut center = vec![0.0; len];

        {
            dvdiqqe_into(&input, &mut dvdi, &mut fast, &mut slow, &mut center)?;
        }

        assert_eq!(baseline.dvdi.len(), dvdi.len());
        assert_eq!(baseline.fast_tl.len(), fast.len());
        assert_eq!(baseline.slow_tl.len(), slow.len());
        assert_eq!(baseline.center_line.len(), center.len());

        fn eq_or_both_nan_eps(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan_eps(baseline.dvdi[i], dvdi[i]),
                "dvdi mismatch at {}: baseline={}, into={}",
                i,
                baseline.dvdi[i],
                dvdi[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.fast_tl[i], fast[i]),
                "fast_tl mismatch at {}: baseline={}, into={}",
                i,
                baseline.fast_tl[i],
                fast[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.slow_tl[i], slow[i]),
                "slow_tl mismatch at {}: baseline={}, into={}",
                i,
                baseline.slow_tl[i],
                slow[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.center_line[i], center[i]),
                "center_line mismatch at {}: baseline={}, into={}",
                i,
                baseline.center_line[i],
                center[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_dvdiqqe_empty_input() {
        let candles = Candles::new(vec![], vec![], vec![], vec![], vec![], vec![]);
        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dvdiqqe_all_nan() {
        let nan_vec = vec![f64::NAN; 10];
        let candles = Candles::new(
            vec![0; 10],
            nan_vec.clone(),
            nan_vec.clone(),
            nan_vec.clone(),
            nan_vec.clone(),
            nan_vec.clone(),
        );
        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dvdiqqe_period_validation() {
        let data = vec![1.0, 2.0, 3.0];
        let candles = Candles::new(
            vec![0, 1, 2],
            data.clone(),
            data.clone(),
            data.clone(),
            data.clone(),
            vec![100.0, 200.0, 300.0],
        );

        let params = DvdiqqeParams {
            period: Some(10),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dvdiqqe_into_matches_api() -> Result<(), Box<dyn Error>> {
        let len = 256usize;
        let mut ts = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);

        for i in 0..len {
            ts.push(i as i64);
            let base = 100.0 + (i as f64) * 0.1;
            let noise = ((i * 17) % 13) as f64 * 0.01;
            let o = base + noise;
            let c = base + (noise * 1.5) - 0.03;
            let h = o.max(c) + 0.5;
            let l = o.min(c) - 0.5;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(1000.0 + ((i * 37) % 23) as f64);
        }

        let candles = Candles::new(ts, open, high, low, close.clone(), volume);
        let input = DvdiqqeInput::with_default_candles(&candles);

        let baseline = dvdiqqe(&input)?;

        let mut dvdi = vec![0.0; len];
        let mut fast = vec![0.0; len];
        let mut slow = vec![0.0; len];
        let mut center = vec![0.0; len];

        dvdiqqe_into(&input, &mut dvdi, &mut fast, &mut slow, &mut center)?;

        assert_eq!(baseline.dvdi.len(), len);
        assert_eq!(baseline.fast_tl.len(), len);
        assert_eq!(baseline.slow_tl.len(), len);
        assert_eq!(baseline.center_line.len(), len);

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline.dvdi[i], dvdi[i]),
                "dvdi mismatch at {}: api={} into={}",
                i,
                baseline.dvdi[i],
                dvdi[i]
            );
            assert!(
                eq_or_both_nan(baseline.fast_tl[i], fast[i]),
                "fast_tl mismatch at {}: api={} into={}",
                i,
                baseline.fast_tl[i],
                fast[i]
            );
            assert!(
                eq_or_both_nan(baseline.slow_tl[i], slow[i]),
                "slow_tl mismatch at {}: api={} into={}",
                i,
                baseline.slow_tl[i],
                slow[i]
            );
            assert!(
                eq_or_both_nan(baseline.center_line[i], center[i]),
                "center_line mismatch at {}: api={} into={}",
                i,
                baseline.center_line[i],
                center[i]
            );
        }

        Ok(())
    }

    fn check_dvdiqqe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe_with_kernel(&input, kernel)?;

        let expected_dvdi = vec![
            -304.41010224,
            -279.48152664,
            -287.58723437,
            -252.40349484,
            -343.00922595,
        ];
        let expected_slow_tl = vec![
            -990.21769695,
            -955.69385266,
            -951.82562405,
            -903.39071943,
            -903.39071943,
        ];
        let expected_fast_tl = vec![
            -728.26380454,
            -697.40500858,
            -697.40500858,
            -654.73695895,
            -654.73695895,
        ];

        let expected_center = vec![
            21.98929919135097,
            21.969910753134442,
            21.950003541229705,
            21.932361363982043,
            21.908895469736102,
        ];

        let start = result.dvdi.len().saturating_sub(5);

        for i in 0..5 {
            let diff_dvdi = (result.dvdi[start + i] - expected_dvdi[i]).abs();
            let diff_slow = (result.slow_tl[start + i] - expected_slow_tl[i]).abs();
            let diff_fast = (result.fast_tl[start + i] - expected_fast_tl[i]).abs();
            let diff_center = (result.center_line[start + i] - expected_center[i]).abs();

            assert!(
                diff_dvdi < 1e-6,
                "[{}] DVDI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.dvdi[start + i],
                expected_dvdi[i]
            );
            assert!(
                diff_slow < 1e-6,
                "[{}] Slow TL {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.slow_tl[start + i],
                expected_slow_tl[i]
            );
            assert!(
                diff_fast < 1e-6,
                "[{}] Fast TL {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.fast_tl[start + i],
                expected_fast_tl[i]
            );
            assert!(
                diff_center < 1e-6,
                "[{}] Center line {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.center_line[start + i],
                expected_center[i]
            );
        }

        Ok(())
    }

    fn check_dvdiqqe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = DvdiqqeParams {
            period: None,
            smoothing_period: None,
            fast_multiplier: None,
            slow_multiplier: None,
            volume_type: None,
            center_type: None,
            tick_size: None,
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let output = dvdiqqe_with_kernel(&input, kernel)?;
        assert_eq!(output.dvdi.len(), candles.close.len());

        Ok(())
    }

    fn check_dvdiqqe_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = DvdiqqeInput::with_default_candles(&candles);
        let output = dvdiqqe_with_kernel(&input, kernel)?;
        assert_eq!(output.dvdi.len(), candles.close.len());

        Ok(())
    }

    fn check_dvdiqqe_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![100.0; 50];
        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            data.clone(),
            data.iter().map(|x| x + 1.0).collect(),
            data.iter().map(|x| x - 1.0).collect(),
            data.clone(),
            vec![1000.0; 50],
        );

        let params = DvdiqqeParams {
            period: Some(0),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with zero period",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![100.0; 5];
        let candles = Candles::new(
            vec![0, 1, 2, 3, 4],
            data.clone(),
            data.iter().map(|x| x + 1.0).collect(),
            data.iter().map(|x| x - 1.0).collect(),
            data.clone(),
            vec![1000.0; 5],
        );

        let params = DvdiqqeParams {
            period: Some(20),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with period exceeding length",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = Candles::new(
            vec![0],
            vec![100.0],
            vec![101.0],
            vec![99.0],
            vec![100.0],
            vec![1000.0],
        );

        let params = DvdiqqeParams::default();
        let input = DvdiqqeInput::from_candles(&candles, params);
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with insufficient data",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = Candles::new(vec![], vec![], vec![], vec![], vec![], vec![]);
        let input = DvdiqqeInput::from_candles(&candles, DvdiqqeParams::default());
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with empty input",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = vec![f64::NAN; 50];
        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            nan_data.clone(),
            nan_data.clone(),
            nan_data.clone(),
            nan_data.clone(),
            nan_data.clone(),
        );

        let input = DvdiqqeInput::from_candles(&candles, DvdiqqeParams::default());
        let res = dvdiqqe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DVDIQQE should fail with all NaN input",
            test_name
        );

        Ok(())
    }

    fn check_dvdiqqe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let mut close = vec![100.0; 50];
        close[10] = f64::NAN;
        close[11] = f64::NAN;

        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            close.clone(),
            close
                .iter()
                .map(|x| if x.is_nan() { f64::NAN } else { x + 1.0 })
                .collect(),
            close
                .iter()
                .map(|x| if x.is_nan() { f64::NAN } else { x - 1.0 })
                .collect(),
            close.clone(),
            vec![1000.0; 50],
        );

        let input = DvdiqqeInput::from_candles(&candles, DvdiqqeParams::default());
        let res = dvdiqqe_with_kernel(&input, kernel)?;

        assert_eq!(res.dvdi.len(), 50);

        if res.dvdi.len() > 30 {
            assert!(
                res.dvdi[30..].iter().any(|x| x.is_finite()),
                "[{}] DVDIQQE should recover after NaN values",
                test_name
            );
        }

        Ok(())
    }

    fn check_dvdiqqe_with_tick_volume(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = DvdiqqeParams {
            volume_type: Some("tick".to_string()),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe_with_kernel(&input, kernel)?;

        assert_eq!(result.dvdi.len(), candles.close.len());

        Ok(())
    }

    fn check_dvdiqqe_static_center(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![100.0; 50];
        let candles = Candles::new(
            (0..50).map(|i| i as i64).collect(),
            data.clone(),
            data.iter().map(|x| x + 1.0).collect(),
            data.iter().map(|x| x - 1.0).collect(),
            data.clone(),
            vec![1000.0; 50],
        );

        let params = DvdiqqeParams {
            center_type: Some("static".to_string()),
            ..Default::default()
        };

        let input = DvdiqqeInput::from_candles(&candles, params);
        let result = dvdiqqe_with_kernel(&input, kernel)?;

        let warmup = 25;
        for i in warmup..result.center_line.len() {
            assert!(
                result.center_line[i] == 0.0,
                "[{}] Static center line should be 0.0 at index {}, got {}",
                test_name,
                i,
                result.center_line[i]
            );
        }

        Ok(())
    }

    macro_rules! generate_all_dvdiqqe_tests {
        ($($test_fn:ident),* $(,)?) => {
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

    macro_rules! gen_dvdiqqe_batch_tests {
        ($fnn:ident) => {
            paste::paste!{
                #[test] fn [<$fnn _scalar>]() { let _ = $fnn(stringify!([<$fnn _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fnn _avx2>]()   { let _ = $fnn(stringify!([<$fnn _avx2>]),   Kernel::Avx2Batch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fnn _avx512>]() { let _ = $fnn(stringify!([<$fnn _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fnn _auto>]()   { let _ = $fnn(stringify!([<$fnn _auto>]),   Kernel::Auto); }
            }
        }
    }

    fn check_batch_default_row(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out = DvdiqqeBatchBuilder::new().kernel(k).apply_candles(&c)?;
        assert_eq!(out.dvdi_values.len(), out.rows * out.cols);
        assert_eq!(out.fast_tl_values.len(), out.rows * out.cols);
        assert_eq!(out.slow_tl_values.len(), out.rows * out.cols);
        assert_eq!(out.center_values.len(), out.rows * out.cols);
        Ok(())
    }

    fn check_batch_sweep(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out = DvdiqqeBatchBuilder::new()
            .kernel(k)
            .period_range(13, 16, 1)
            .smoothing_range(5, 7, 1)
            .fast_range(2.6, 2.6, 0.0)
            .slow_range(4.2, 4.2, 0.0)
            .apply_candles(&c)?;
        let expected = 4 * 3;
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    gen_dvdiqqe_batch_tests!(check_batch_default_row);
    gen_dvdiqqe_batch_tests!(check_batch_sweep);

    fn check_dvdiqqe_batch_default_row_old(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let batch_output = DvdiqqeBatchBuilder::new()
            .kernel(kernel)
            .period_static(13)
            .smoothing_static(6)
            .fast_static(2.618)
            .slow_static(4.236)
            .apply_candles(&c)?;

        let def_params = DvdiqqeParams::default();
        let batch_values = batch_output
            .values_for(&def_params)
            .expect("default row missing");

        let single_input = DvdiqqeInput::with_default_candles(&c);
        let single_output = dvdiqqe_with_kernel(&single_input, kernel)?;

        assert_eq!(batch_values.dvdi.len(), single_output.dvdi.len());

        for i in 0..batch_values.dvdi.len() {
            if batch_values.dvdi[i].is_finite() && single_output.dvdi[i].is_finite() {
                assert!(
                    (batch_values.dvdi[i] - single_output.dvdi[i]).abs() < 1e-10,
                    "[{}] DVDI mismatch at index {}: batch={}, single={}",
                    test_name,
                    i,
                    batch_values.dvdi[i],
                    single_output.dvdi[i]
                );
            }
        }

        Ok(())
    }

    fn check_dvdiqqe_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let test_data = vec![
            (100.0, 102.0, 99.0, 101.0, 1000.0),
            (101.0, 103.0, 100.0, 102.0, 1100.0),
            (102.0, 104.0, 101.0, 103.0, 1200.0),
            (103.0, 105.0, 102.0, 104.0, 1150.0),
            (104.0, 106.0, 103.0, 105.0, 1250.0),
        ];

        let mut full_data = test_data.clone();
        for i in 0..50 {
            let base = 105.0 + i as f64 * 0.5;
            full_data.push((
                base,
                base + 2.0,
                base - 1.0,
                base + 1.0,
                1000.0 + (i as f64 * 50.0),
            ));
        }

        let params = DvdiqqeParams::default();
        let mut stream = DvdiqqeStream::try_new(params.clone())?;

        let mut stream_results = Vec::new();
        for (open, high, low, close, volume) in &full_data {
            if let Some(output) = stream.update(*open, *high, *low, *close, *volume) {
                stream_results.push((
                    output.dvdi,
                    output.fast_tl,
                    output.slow_tl,
                    output.center_line,
                ));
            }
        }

        assert!(
            !stream_results.is_empty(),
            "[{}] Stream should produce outputs",
            test_name
        );

        if stream_results.len() > 0 {
            let (opens, highs, lows, closes, volumes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
                full_data.iter().cloned().unzip_n_tuple();

            let batch_input =
                DvdiqqeInput::from_slices(&opens, &highs, &lows, &closes, Some(&volumes), params);

            if let Ok(batch_output) = dvdiqqe(&batch_input) {
                let last_idx = batch_output.dvdi.len() - 1;
                let last_stream = stream_results.last().unwrap();

                if batch_output.dvdi[last_idx].is_finite() && last_stream.0.is_finite() {
                    assert!(
                        (batch_output.dvdi[last_idx] - last_stream.0).abs() < 1.0,
                        "[{}] Stream DVDI doesn't match batch: stream={}, batch={}",
                        test_name,
                        last_stream.0,
                        batch_output.dvdi[last_idx]
                    );
                }
            }
        }

        Ok(())
    }

    fn check_dvdiqqe_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let n = 100;
        let mut opens = vec![100.0; n];
        let mut highs = vec![102.0; n];
        let mut lows = vec![98.0; n];
        let mut closes = vec![100.0; n];
        let mut volumes = vec![1000.0; n];

        for i in 0..n {
            let base = 100.0 + (i as f64 * 0.1);
            opens[i] = base;
            highs[i] = base + 2.0;
            lows[i] = base - 2.0;
            closes[i] = base + 0.5;
            volumes[i] = 1000.0 + (i as f64 * 10.0);
        }

        let batch_output = DvdiqqeBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 15, 2)
            .smoothing_range(4, 8, 2)
            .fast_range(2.0, 3.0, 0.5)
            .slow_range(4.0, 5.0, 0.5)
            .apply_slices(&opens, &highs, &lows, &closes, Some(&volumes))?;

        let expected_periods = 3;
        let expected_smoothings = 3;
        let expected_fasts = 3;
        let expected_slows = 3;
        let expected_rows =
            expected_periods * expected_smoothings * expected_fasts * expected_slows;

        assert_eq!(
            batch_output.rows, expected_rows,
            "[{}] Wrong number of parameter combinations",
            test_name
        );
        assert_eq!(
            batch_output.cols, n,
            "[{}] Wrong number of data points",
            test_name
        );

        let test_params = DvdiqqeParams {
            period: Some(12),
            smoothing_period: Some(6),
            fast_multiplier: Some(2.5),
            slow_multiplier: Some(4.5),
            volume_type: None,
            center_type: None,
            tick_size: None,
        };

        assert!(
            batch_output.values_for(&test_params).is_some(),
            "[{}] Should find test parameter combination",
            test_name
        );

        Ok(())
    }

    trait UnzipN<A, B, C, D, E> {
        fn unzip_n_tuple(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>);
    }

    impl<A, B, C, D, E, I> UnzipN<A, B, C, D, E> for I
    where
        I: Iterator<Item = (A, B, C, D, E)>,
    {
        fn unzip_n_tuple(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>, Vec<E>) {
            let mut a_vec = Vec::new();
            let mut b_vec = Vec::new();
            let mut c_vec = Vec::new();
            let mut d_vec = Vec::new();
            let mut e_vec = Vec::new();

            for (a, b, c, d, e) in self {
                a_vec.push(a);
                b_vec.push(b);
                c_vec.push(c);
                d_vec.push(d);
                e_vec.push(e);
            }

            (a_vec, b_vec, c_vec, d_vec, e_vec)
        }
    }

    fn check_dvdiqqe_batch_sweep_old(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = DvdiqqeBatchBuilder::new()
            .kernel(kernel)
            .period_range(13, 16, 1)
            .smoothing_range(5, 7, 1)
            .fast_range(2.6, 2.6, 0.0)
            .slow_range(4.2, 4.2, 0.0)
            .apply_candles(&c)?;

        let expected = 4 * 3;
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());

        Ok(())
    }

    fn check_dvdiqqe_reinput(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out1 = dvdiqqe_with_kernel(&DvdiqqeInput::with_default_candles(&c), k)?;

        let i2 = DvdiqqeInput::from_slices(
            &c.open,
            &c.high,
            &c.low,
            &out1.dvdi,
            Some(&c.volume),
            DvdiqqeParams::default(),
        );
        let out2 = dvdiqqe_with_kernel(&i2, k)?;
        assert_eq!(out2.dvdi.len(), out1.dvdi.len());
        Ok(())
    }

    generate_all_dvdiqqe_tests!(
        check_dvdiqqe_accuracy,
        check_dvdiqqe_partial_params,
        check_dvdiqqe_default_candles,
        check_dvdiqqe_zero_period,
        check_dvdiqqe_period_exceeds_length,
        check_dvdiqqe_very_small_dataset,
        check_dvdiqqe_empty_input,
        check_dvdiqqe_all_nan,
        check_dvdiqqe_nan_handling,
        check_dvdiqqe_with_tick_volume,
        check_dvdiqqe_static_center,
        check_dvdiqqe_batch_default_row_old,
        check_dvdiqqe_streaming,
        check_dvdiqqe_batch_sweep_old,
        check_dvdiqqe_reinput
    );

    #[cfg(debug_assertions)]
    #[test]
    fn dvdiqqe_no_poison_in_outputs() -> Result<(), Box<dyn Error>> {
        use crate::utilities::data_loader::read_candles_from_vortex;
        let c = read_candles_from_vortex("src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex")?;

        let out = dvdiqqe_with_kernel(&DvdiqqeInput::with_default_candles(&c), Kernel::Scalar)?;

        for &v in out
            .dvdi
            .iter()
            .chain(&out.fast_tl)
            .chain(&out.slow_tl)
            .chain(&out.center_line)
        {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x2222_2222_2222_2222,
                "init_matrix_prefixes poison in single output"
            );
            assert_ne!(
                b, 0x3333_3333_3333_3333,
                "make_uninit_matrix poison in single output"
            );
        }

        let sweep = DvdiqqeBatchRange::default();
        let flat_out = dvdiqqe_batch_with_kernel_flat(
            &c.open,
            &c.high,
            &c.low,
            &c.close,
            Some(&c.volume),
            &sweep,
            Kernel::Auto,
            "default",
            "dynamic",
            0.01,
        )?;

        for (i, &v) in flat_out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x2222_2222_2222_2222,
                "init_matrix_prefixes poison at {}",
                i
            );
            assert_ne!(
                b, 0x3333_3333_3333_3333,
                "make_uninit_matrix poison at {}",
                i
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_dvdiqqe_output_length_matches_input(
                data in prop::collection::vec(prop::num::f64::NORMAL | prop::num::f64::POSITIVE, 50..200)
            ) {
                let len = data.len();
                let timestamps: Vec<i64> = (0..len as i64).collect();
                let mut open = Vec::with_capacity(len);
                let mut high = Vec::with_capacity(len);
                let mut low = Vec::with_capacity(len);
                let mut close = Vec::with_capacity(len);
                let mut volume = Vec::with_capacity(len);

                for &val in &data {
                    open.push(val - 0.5);
                    high.push(val + 1.0);
                    low.push(val - 1.0);
                    close.push(val);
                    volume.push(1000.0);
                }

                let candles = Candles::new(timestamps, open, high, low, close, volume);
                let input = DvdiqqeInput::with_default_candles(&candles);

                match dvdiqqe(&input) {
                    Ok(output) => {
                        prop_assert_eq!(output.dvdi.len(), len);
                        prop_assert_eq!(output.fast_tl.len(), len);
                        prop_assert_eq!(output.slow_tl.len(), len);
                        prop_assert_eq!(output.center_line.len(), len);
                    }
                    Err(DvdiqqeError::NotEnoughValidData { .. }) => {

                    }
                    Err(e) => {
                        prop_assert!(false, "Unexpected error: {:?}", e);
                    }
                }
            }

            #[test]
            fn test_dvdiqqe_nan_propagation(
                valid_data in prop::collection::vec(100.0f64..200.0, 50..100),
                nan_positions in prop::collection::vec(0usize..50, 0..10)
            ) {
                let len = valid_data.len();
                let mut data = valid_data.clone();


                for &pos in &nan_positions {
                    if pos < len {
                        data[pos] = f64::NAN;
                    }
                }

                let timestamps: Vec<i64> = (0..len as i64).collect();
                let open = data.iter().map(|&v| v - 0.5).collect();
                let high = data.iter().map(|&v| v + 1.0).collect();
                let low = data.iter().map(|&v| v - 1.0).collect();
                let volume = vec![1000.0; len];

                let candles = Candles::new(timestamps, open, high, low, data.clone(), volume);
                let input = DvdiqqeInput::with_default_candles(&candles);

                match dvdiqqe(&input) {
                    Ok(output) => {


                        prop_assert_eq!(output.dvdi.len(), len);
                        prop_assert_eq!(output.fast_tl.len(), len);
                        prop_assert_eq!(output.slow_tl.len(), len);
                        prop_assert_eq!(output.center_line.len(), len);


                        let expected_warmup = 25;
                        for i in 0..expected_warmup.min(len) {

                            prop_assert!(output.dvdi[i].is_nan() || output.dvdi[i].is_finite(),
                                "Position {} should be either NaN (warmup) or finite", i);
                        }


                        for &v in output.dvdi.iter()
                            .chain(&output.fast_tl)
                            .chain(&output.slow_tl)
                            .chain(&output.center_line) {
                            if !v.is_nan() {
                                let bits = v.to_bits();
                                prop_assert_ne!(bits, 0x2222_2222_2222_2222);
                                prop_assert_ne!(bits, 0x3333_3333_3333_3333);
                            }
                        }
                    }
                    Err(DvdiqqeError::AllValuesNaN) => {

                        prop_assert!(nan_positions.len() >= len / 2);
                    }
                    Err(_) => {

                    }
                }
            }

            #[test]
            fn test_dvdiqqe_parameter_bounds(
                period in 1usize..50,
                smoothing in 1usize..20,
                fast_mult in 0.1f64..10.0,
                slow_mult in 0.1f64..10.0
            ) {
                let len = 100;
                let data: Vec<f64> = (0..len).map(|i| 100.0 + i as f64).collect();
                let timestamps: Vec<i64> = (0..len as i64).collect();
                let open = data.iter().map(|&v| v - 0.5).collect();
                let high = data.iter().map(|&v| v + 1.0).collect();
                let low = data.iter().map(|&v| v - 1.0).collect();
                let volume = vec![1000.0; len];

                let candles = Candles::new(timestamps, open, high, low, data, volume);
                let params = DvdiqqeParams {
                    period: Some(period),
                    smoothing_period: Some(smoothing),
                    fast_multiplier: Some(fast_mult),
                    slow_multiplier: Some(slow_mult),
                    ..Default::default()
                };
                let input = DvdiqqeInput::from_candles(&candles, params);

                match dvdiqqe(&input) {
                    Ok(output) => {

                        for i in 30..len {
                            prop_assert!(output.dvdi[i].is_finite() || output.dvdi[i].is_nan());
                            prop_assert!(output.fast_tl[i].is_finite() || output.fast_tl[i].is_nan());
                            prop_assert!(output.slow_tl[i].is_finite() || output.slow_tl[i].is_nan());
                        }
                    }
                    Err(DvdiqqeError::InvalidPeriod { .. }) => {
                        prop_assert!(period > len || period == 0);
                    }
                    Err(_) => {

                    }
                }
            }
        }
    }
}
