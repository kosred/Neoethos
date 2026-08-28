use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::f64::consts::PI;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_DELTA: f64 = 0.1;
const DEFAULT_ALPHA: f64 = 0.07;
const MIN_VALID_SAMPLES: usize = 12;
const IN_PHASE_WARMUP: usize = MIN_VALID_SAMPLES - 1;
const LEAD_WARMUP: usize = MIN_VALID_SAMPLES;
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for AdaptiveBandpassTriggerOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdaptiveBandpassTriggerOscillatorData::Slice(slice) => slice,
            AdaptiveBandpassTriggerOscillatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdaptiveBandpassTriggerOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorOutput {
    pub in_phase: Vec<f64>,
    pub lead: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveBandpassTriggerOscillatorOutputField {
    InPhase,
    Lead,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveBandpassTriggerOscillatorParams {
    pub delta: Option<f64>,
    pub alpha: Option<f64>,
}

impl Default for AdaptiveBandpassTriggerOscillatorParams {
    fn default() -> Self {
        Self {
            delta: Some(DEFAULT_DELTA),
            alpha: Some(DEFAULT_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorInput<'a> {
    pub data: AdaptiveBandpassTriggerOscillatorData<'a>,
    pub params: AdaptiveBandpassTriggerOscillatorParams,
}

impl<'a> AdaptiveBandpassTriggerOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AdaptiveBandpassTriggerOscillatorParams,
    ) -> Self {
        Self {
            data: AdaptiveBandpassTriggerOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AdaptiveBandpassTriggerOscillatorParams) -> Self {
        Self {
            data: AdaptiveBandpassTriggerOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            AdaptiveBandpassTriggerOscillatorParams::default(),
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptiveBandpassTriggerOscillatorBuilder {
    delta: Option<f64>,
    alpha: Option<f64>,
    kernel: Kernel,
}

impl AdaptiveBandpassTriggerOscillatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = Some(delta);
        self
    }

    #[inline]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
}

#[derive(Debug, Error)]
pub enum AdaptiveBandpassTriggerOscillatorError {
    #[error("adaptive_bandpass_trigger_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("adaptive_bandpass_trigger_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("adaptive_bandpass_trigger_oscillator: Invalid delta: {delta}")]
    InvalidDelta { delta: f64 },
    #[error("adaptive_bandpass_trigger_oscillator: Invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error(
        "adaptive_bandpass_trigger_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "adaptive_bandpass_trigger_oscillator: Output length mismatch: expected = {expected}, in_phase = {in_phase_got}, lead = {lead_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        in_phase_got: usize,
        lead_got: usize,
    },
    #[error(
        "adaptive_bandpass_trigger_oscillator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("adaptive_bandpass_trigger_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    delta: f64,
    alpha: f64,
}

#[inline(always)]
fn resolve_params(
    params: &AdaptiveBandpassTriggerOscillatorParams,
) -> Result<ResolvedParams, AdaptiveBandpassTriggerOscillatorError> {
    let delta = params.delta.unwrap_or(DEFAULT_DELTA);
    let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidDelta { delta });
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidAlpha { alpha });
    }
    Ok(ResolvedParams { delta, alpha })
}

#[inline(always)]
fn median3(x: f64, y: f64, z: f64) -> f64 {
    (x + y + z) - x.min(y.min(z)) - x.max(y.max(z))
}

/* FreeBSD msun k_cos/k_sin and the small-argument s_cos reduction.
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro/SunSoft. Permission to use, copy, modify, and
 * distribute this software is freely granted, provided this notice is
 * preserved.
 *
 * Adaptive Bandpass only calls cosine with finite positive arguments below
 * 2*pi/3 because length >= 6 and 0 < delta < 1. Keeping this bounded copy in
 * both the Rust scalar and CUDA translation unit prevents a host-libm versus
 * CUDA-libdevice ULP split from being amplified by the recursive filter.
 */
#[inline(always)]
fn abto_ms_k_cos(x: f64, y: f64) -> f64 {
    let c1 = f64::from_bits(0x3fa555555555554c);
    let c2 = f64::from_bits(0xbf56c16c16c15177);
    let c3 = f64::from_bits(0x3efa01a019cb1590);
    let c4 = f64::from_bits(0xbe927e4f809c52ad);
    let c5 = f64::from_bits(0x3e21ee9ebdb4b1c4);
    let c6 = f64::from_bits(0xbda8fae9be8838d4);
    let z = x * x;
    let w2 = z * z;
    let r = z * (c1 + z * (c2 + z * c3)) + w2 * w2 * (c4 + z * (c5 + z * c6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

#[inline(always)]
fn abto_ms_k_sin(x: f64, y: f64) -> f64 {
    let s1 = f64::from_bits(0xbfc5555555555549);
    let s2 = f64::from_bits(0x3f8111111110f8a6);
    let s3 = f64::from_bits(0xbf2a01a019c161d5);
    let s4 = f64::from_bits(0x3ec71de357b1fe7d);
    let s5 = f64::from_bits(0xbe5ae5e68a2b9ceb);
    let s6 = f64::from_bits(0x3de5d93a5acfd57c);
    let z = x * x;
    let w = z * z;
    let r = s2 + z * (s3 + z * s4) + z * w * (s5 + z * s6);
    let v = z * x;
    x - ((z * (0.5 * y - v * r) - y) - v * s1)
}

#[inline(always)]
fn abto_reduce_pio2_near_half_pi(x: f64, high: u32) -> (f64, f64) {
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
    debug_assert_eq!(f_n, 1.0);
    let mut r = x - f_n * pio2_1;
    let mut w = f_n * pio2_1t;
    let mut y0 = r - w;
    let ex = (high >> 20) as i32;
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
    let y1 = (r - y0) - w;
    (y0, y1)
}

#[inline(always)]
fn abto_deterministic_cos(x: f64) -> f64 {
    debug_assert!(x.is_finite() && x >= 0.0 && x < 2.0 * PI / 3.0);
    let high = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if high <= 0x3fe9_21fb {
        return abto_ms_k_cos(x, 0.0);
    }
    let (y0, y1) = if high & 0x000f_ffff == 0x0009_21fb {
        // Near pi/2 the direct two-term subtraction loses too many bits.
        // Mirror msun's medium argument reduction for this cancellation case.
        abto_reduce_pio2_near_half_pi(x, high)
    } else {
        let pio2_1 = f64::from_bits(0x3ff9_21fb_5440_0000);
        let pio2_1t = f64::from_bits(0x3dd0_b461_1a62_6331);
        let z = x - pio2_1;
        let y0 = z - pio2_1t;
        (y0, (z - y0) - pio2_1t)
    };
    -abto_ms_k_sin(y0, y1)
}

#[inline(always)]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|value| value.is_finite()).count()
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    data.iter()
        .position(|value| value.is_finite())
        .unwrap_or(data.len())
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorStream {
    params: ResolvedParams,
    price: [f64; 4],
    smooth_hist: [f64; 2],
    c_hist: [f64; 6],
    dp_hist: [f64; 4],
    q1_prev: f64,
    i1_prev: f64,
    ip_prev: f64,
    p_prev: f64,
    bp_prev1: f64,
    bp_prev2: f64,
    valid_count: usize,
}

impl AdaptiveBandpassTriggerOscillatorStream {
    pub fn try_new(
        params: AdaptiveBandpassTriggerOscillatorParams,
    ) -> Result<Self, AdaptiveBandpassTriggerOscillatorError> {
        let params = resolve_params(&params)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            price: [0.0; 4],
            smooth_hist: [0.0; 2],
            c_hist: [0.0; 6],
            dp_hist: [0.0; 4],
            q1_prev: 0.0,
            i1_prev: 0.0,
            ip_prev: 0.0,
            p_prev: 0.0,
            bp_prev1: 0.0,
            bp_prev2: 0.0,
            valid_count: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        IN_PHASE_WARMUP
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        self.price[3] = self.price[2];
        self.price[2] = self.price[1];
        self.price[1] = self.price[0];
        self.price[0] = value;

        let index = self.valid_count;
        self.valid_count += 1;

        let smooth = if index >= 3 {
            (self.price[0] + 2.0 * self.price[1] + 2.0 * self.price[2] + self.price[3]) / 6.0
        } else {
            0.0
        };

        let alpha = self.params.alpha;
        let c = if index < 2 {
            0.0
        } else if index < 7 {
            (self.price[0] - 2.0 * self.price[1] + self.price[2]) * 0.25
        } else {
            let smooth_gain = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
            smooth_gain * (smooth - 2.0 * self.smooth_hist[0] + self.smooth_hist[1])
                + 2.0 * (1.0 - alpha) * self.c_hist[0]
                - (1.0 - alpha) * (1.0 - alpha) * self.c_hist[1]
        };

        let q1 = if index >= 6 {
            (0.0962 * c + 0.5769 * self.c_hist[1]
                - 0.5769 * self.c_hist[3]
                - 0.0962 * self.c_hist[5])
                * (0.5 + 0.08 * self.ip_prev)
        } else {
            0.0
        };
        let i1 = if index >= 3 { self.c_hist[2] } else { 0.0 };

        let dp_raw = if q1.abs() > FLOAT_TOL && self.q1_prev.abs() > FLOAT_TOL {
            let denominator = 1.0 + (i1 * self.i1_prev) / (q1 * self.q1_prev);
            if denominator.abs() > FLOAT_TOL {
                ((i1 / q1) - (self.i1_prev / self.q1_prev)) / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };
        let dp = dp_raw.clamp(0.1, 1.1);

        let md = if index >= 10 {
            median3(
                dp,
                self.dp_hist[0],
                median3(self.dp_hist[1], self.dp_hist[2], self.dp_hist[3]),
            )
        } else {
            0.0
        };
        let dc = if md.abs() <= FLOAT_TOL {
            15.0
        } else {
            (2.0 * PI) / md + 0.5
        };

        let ip = 0.33 * dc + 0.67 * self.ip_prev;
        let p = 0.15 * ip + 0.85 * self.p_prev;

        let mut in_phase = f64::NAN;
        let mut lead = f64::NAN;
        if index >= IN_PHASE_WARMUP {
            let length = p.max(6.0);
            // Host libm and CUDA libdevice are both accurate but are not
            // bit-identical. This recursive filter amplifies a 1-ULP cosine
            // difference, so both lanes use the same bounded msun routine.
            let beta = abto_deterministic_cos(2.0 * PI / length);
            let cos_angle = abto_deterministic_cos(4.0 * PI * self.params.delta / length);
            let denom = if cos_angle.abs() < FLOAT_TOL {
                if cos_angle.is_sign_negative() {
                    -FLOAT_TOL
                } else {
                    FLOAT_TOL
                }
            } else {
                cos_angle
            };
            let gamma = 1.0 / denom;
            let alpha_bp = gamma - (gamma * gamma - 1.0).max(0.0).sqrt();

            in_phase = 0.5 * (1.0 - alpha_bp) * (self.price[0] - self.price[2])
                + beta * (1.0 + alpha_bp) * self.bp_prev1
                - alpha_bp * self.bp_prev2;
            if index >= LEAD_WARMUP && self.bp_prev1.is_finite() {
                let quadrature = (in_phase - self.bp_prev1) * length / (2.0 * PI);
                lead = 0.5 * in_phase + 0.866 * quadrature;
            }
        }

        self.smooth_hist[1] = self.smooth_hist[0];
        self.smooth_hist[0] = smooth;

        self.c_hist[5] = self.c_hist[4];
        self.c_hist[4] = self.c_hist[3];
        self.c_hist[3] = self.c_hist[2];
        self.c_hist[2] = self.c_hist[1];
        self.c_hist[1] = self.c_hist[0];
        self.c_hist[0] = c;

        self.dp_hist[3] = self.dp_hist[2];
        self.dp_hist[2] = self.dp_hist[1];
        self.dp_hist[1] = self.dp_hist[0];
        self.dp_hist[0] = dp;

        self.q1_prev = q1;
        self.i1_prev = i1;
        self.ip_prev = ip;
        self.p_prev = p;

        if in_phase.is_finite() {
            self.bp_prev2 = self.bp_prev1;
            self.bp_prev1 = in_phase;
            Some((in_phase, lead))
        } else {
            None
        }
    }
}

impl AdaptiveBandpassTriggerOscillatorBuilder {
    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError>
    {
        let input = AdaptiveBandpassTriggerOscillatorInput::from_candles(
            candles,
            "close",
            AdaptiveBandpassTriggerOscillatorParams {
                delta: self.delta,
                alpha: self.alpha,
            },
        );
        adaptive_bandpass_trigger_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError>
    {
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            data,
            AdaptiveBandpassTriggerOscillatorParams {
                delta: self.delta,
                alpha: self.alpha,
            },
        );
        adaptive_bandpass_trigger_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<AdaptiveBandpassTriggerOscillatorStream, AdaptiveBandpassTriggerOscillatorError>
    {
        AdaptiveBandpassTriggerOscillatorStream::try_new(AdaptiveBandpassTriggerOscillatorParams {
            delta: self.delta,
            alpha: self.alpha,
        })
    }
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator(
    input: &AdaptiveBandpassTriggerOscillatorInput,
) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn adaptive_bandpass_trigger_oscillator_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
) {
    let mut stream = AdaptiveBandpassTriggerOscillatorStream::new_resolved(params);
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((in_phase, lead)) => {
                in_phase_out[i] = in_phase;
                lead_out[i] = lead;
            }
            None => {
                in_phase_out[i] = f64::NAN;
                lead_out[i] = f64::NAN;
            }
        }
    }
}

#[inline(always)]
fn adaptive_bandpass_trigger_oscillator_output_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    field: AdaptiveBandpassTriggerOscillatorOutputField,
    out: &mut [f64],
) {
    let mut stream = AdaptiveBandpassTriggerOscillatorStream::new_resolved(params);
    match field {
        AdaptiveBandpassTriggerOscillatorOutputField::InPhase => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((in_phase, _)) => in_phase,
                    None => f64::NAN,
                };
            }
        }
        AdaptiveBandpassTriggerOscillatorOutputField::Lead => {
            for i in 0..data.len() {
                out[i] = match stream.update(data[i]) {
                    Some((_, lead)) => lead,
                    None => f64::NAN,
                };
            }
        }
    }
}

#[inline(always)]
fn adaptive_bandpass_trigger_oscillator_prepare<'a>(
    input: &'a AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), AdaptiveBandpassTriggerOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AdaptiveBandpassTriggerOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AdaptiveBandpassTriggerOscillatorError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    let valid = count_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(AdaptiveBandpassTriggerOscillatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

pub fn adaptive_bandpass_trigger_oscillator_with_kernel(
    input: &AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorOutput, AdaptiveBandpassTriggerOscillatorError> {
    let (data, first, params, _chosen) =
        adaptive_bandpass_trigger_oscillator_prepare(input, kernel)?;
    let mut in_phase = alloc_with_nan_prefix(data.len(), (first + IN_PHASE_WARMUP).min(data.len()));
    let mut lead = alloc_with_nan_prefix(data.len(), (first + LEAD_WARMUP).min(data.len()));
    adaptive_bandpass_trigger_oscillator_row_from_slice(data, params, &mut in_phase, &mut lead);
    Ok(AdaptiveBandpassTriggerOscillatorOutput { in_phase, lead })
}

pub fn adaptive_bandpass_trigger_oscillator_into_slices(
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
    input: &AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
) -> Result<(), AdaptiveBandpassTriggerOscillatorError> {
    let expected = input.as_ref().len();
    if in_phase_out.len() != expected || lead_out.len() != expected {
        return Err(
            AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
                expected,
                in_phase_got: in_phase_out.len(),
                lead_got: lead_out.len(),
            },
        );
    }
    let (data, _first, params, _chosen) =
        adaptive_bandpass_trigger_oscillator_prepare(input, kernel)?;
    adaptive_bandpass_trigger_oscillator_row_from_slice(data, params, in_phase_out, lead_out);
    Ok(())
}

pub fn adaptive_bandpass_trigger_oscillator_output_into_slice(
    out: &mut [f64],
    input: &AdaptiveBandpassTriggerOscillatorInput,
    kernel: Kernel,
    field: AdaptiveBandpassTriggerOscillatorOutputField,
) -> Result<(), AdaptiveBandpassTriggerOscillatorError> {
    let expected = input.as_ref().len();
    if out.len() != expected {
        return Err(
            AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
                expected,
                in_phase_got: out.len(),
                lead_got: out.len(),
            },
        );
    }
    let (data, _first, params, _chosen) =
        adaptive_bandpass_trigger_oscillator_prepare(input, kernel)?;
    adaptive_bandpass_trigger_oscillator_output_row_from_slice(data, params, field, out);
    Ok(())
}

pub fn adaptive_bandpass_trigger_oscillator_into(
    input: &AdaptiveBandpassTriggerOscillatorInput,
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
) -> Result<(), AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_into_slices(in_phase_out, lead_out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorBatchRange {
    pub delta: (f64, f64, f64),
    pub alpha: (f64, f64, f64),
}

impl Default for AdaptiveBandpassTriggerOscillatorBatchRange {
    fn default() -> Self {
        Self {
            delta: (DEFAULT_DELTA, DEFAULT_DELTA, 0.0),
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBandpassTriggerOscillatorBatchOutput {
    pub in_phase: Vec<f64>,
    pub lead: Vec<f64>,
    pub combos: Vec<AdaptiveBandpassTriggerOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AdaptiveBandpassTriggerOscillatorBatchBuilder {
    sweep: AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
}

impl AdaptiveBandpassTriggerOscillatorBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn delta(mut self, start: f64, end: f64, step: f64) -> Self {
        self.sweep.delta = (start, end, step);
        self
    }

    #[inline]
    pub fn alpha(mut self, start: f64, end: f64, step: f64) -> Self {
        self.sweep.alpha = (start, end, step);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError>
    {
        adaptive_bandpass_trigger_oscillator_batch_with_kernel(data, &self.sweep, self.kernel)
    }
}

#[inline]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, AdaptiveBandpassTriggerOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    let mut values = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        values.push(value.min(end));
        value += step;
    }
    if (values.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(AdaptiveBandpassTriggerOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(values)
}

#[inline]
fn expand_grid_adaptive_bandpass_trigger_oscillator(
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
) -> Result<Vec<AdaptiveBandpassTriggerOscillatorParams>, AdaptiveBandpassTriggerOscillatorError> {
    let deltas = expand_axis_f64(sweep.delta.0, sweep.delta.1, sweep.delta.2)?;
    let alphas = expand_axis_f64(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?;
    let mut combos = Vec::with_capacity(deltas.len() * alphas.len());
    for &delta in &deltas {
        for &alpha in &alphas {
            let combo = AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(delta),
                alpha: Some(alpha),
            };
            let _ = resolve_params(&combo)?;
            combos.push(combo);
        }
    }
    Ok(combos)
}

impl AdaptiveBandpassTriggerOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(
        &self,
        params: &AdaptiveBandpassTriggerOscillatorParams,
    ) -> Option<usize> {
        self.combos.iter().position(|combo| {
            (combo.delta.unwrap_or(DEFAULT_DELTA) - params.delta.unwrap_or(DEFAULT_DELTA)).abs()
                < FLOAT_TOL
                && (combo.alpha.unwrap_or(DEFAULT_ALPHA) - params.alpha.unwrap_or(DEFAULT_ALPHA))
                    .abs()
                    < FLOAT_TOL
        })
    }

    #[inline]
    pub fn row_slices(&self, row: usize) -> Option<(&[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some((&self.in_phase[start..end], &self.lead[start..end]))
    }
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdaptiveBandpassTriggerOscillatorError::InvalidKernelForBatch(other)),
    };
    adaptive_bandpass_trigger_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator_batch_slice(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn adaptive_bandpass_trigger_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    adaptive_bandpass_trigger_oscillator_batch_inner(data, sweep, kernel, true)
}

pub fn adaptive_bandpass_trigger_oscillator_batch_inner(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<AdaptiveBandpassTriggerOscillatorBatchOutput, AdaptiveBandpassTriggerOscillatorError> {
    let combos = expand_grid_adaptive_bandpass_trigger_oscillator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= cols {
        return Err(AdaptiveBandpassTriggerOscillatorError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(AdaptiveBandpassTriggerOscillatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    let mut in_phase_mu = make_uninit_matrix(rows, cols);
    let mut lead_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(
        &mut in_phase_mu,
        cols,
        &vec![(first + IN_PHASE_WARMUP).min(cols); rows],
    );
    init_matrix_prefixes(
        &mut lead_mu,
        cols,
        &vec![(first + LEAD_WARMUP).min(cols); rows],
    );

    let mut in_phase_guard = ManuallyDrop::new(in_phase_mu);
    let mut lead_guard = ManuallyDrop::new(lead_mu);
    let in_phase_out = unsafe {
        std::slice::from_raw_parts_mut(
            in_phase_guard.as_mut_ptr() as *mut f64,
            in_phase_guard.len(),
        )
    };
    let lead_out = unsafe {
        std::slice::from_raw_parts_mut(lead_guard.as_mut_ptr() as *mut f64, lead_guard.len())
    };

    let combos = adaptive_bandpass_trigger_oscillator_batch_inner_into(
        data,
        sweep,
        _kernel,
        parallel,
        in_phase_out,
        lead_out,
    )?;

    let in_phase = unsafe {
        Vec::from_raw_parts(
            in_phase_guard.as_mut_ptr() as *mut f64,
            in_phase_guard.len(),
            in_phase_guard.capacity(),
        )
    };
    let lead = unsafe {
        Vec::from_raw_parts(
            lead_guard.as_mut_ptr() as *mut f64,
            lead_guard.len(),
            lead_guard.capacity(),
        )
    };

    Ok(AdaptiveBandpassTriggerOscillatorBatchOutput {
        in_phase,
        lead,
        combos,
        rows,
        cols,
    })
}

pub fn adaptive_bandpass_trigger_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &AdaptiveBandpassTriggerOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    in_phase_out: &mut [f64],
    lead_out: &mut [f64],
) -> Result<Vec<AdaptiveBandpassTriggerOscillatorParams>, AdaptiveBandpassTriggerOscillatorError> {
    let combos = expand_grid_adaptive_bandpass_trigger_oscillator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(AdaptiveBandpassTriggerOscillatorError::EmptyInputData);
    }
    let total = rows.checked_mul(cols).ok_or(
        AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            in_phase_got: in_phase_out.len(),
            lead_got: lead_out.len(),
        },
    )?;
    if in_phase_out.len() != total || lead_out.len() != total {
        return Err(
            AdaptiveBandpassTriggerOscillatorError::OutputLengthMismatch {
                expected: total,
                in_phase_got: in_phase_out.len(),
                lead_got: lead_out.len(),
            },
        );
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(AdaptiveBandpassTriggerOscillatorError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    if valid < MIN_VALID_SAMPLES {
        return Err(AdaptiveBandpassTriggerOscillatorError::NotEnoughValidData {
            needed: MIN_VALID_SAMPLES,
            valid,
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        in_phase_out
            .par_chunks_mut(cols)
            .zip(lead_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (in_phase_row, lead_row))| {
                let params = resolve_params(&combos[row]).unwrap();
                adaptive_bandpass_trigger_oscillator_row_from_slice(
                    data,
                    params,
                    in_phase_row,
                    lead_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, (in_phase_row, lead_row)) in in_phase_out
            .chunks_mut(cols)
            .zip(lead_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            adaptive_bandpass_trigger_oscillator_row_from_slice(
                data,
                params,
                in_phase_row,
                lead_row,
            );
        }
    } else {
        for (row, (in_phase_row, lead_row)) in in_phase_out
            .chunks_mut(cols)
            .zip(lead_out.chunks_mut(cols))
            .enumerate()
        {
            let params = resolve_params(&combos[row]).unwrap();
            adaptive_bandpass_trigger_oscillator_row_from_slice(
                data,
                params,
                in_phase_row,
                lead_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn sample_close(length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; length];
        let mut prev = 100.0;
        for i in 2..length {
            let x = i as f64;
            let value = prev + (x * 0.07).sin() * 1.3 + (x * 0.03).cos() * 0.6 + x * 0.02;
            out[i] = value;
            prev = value;
        }
        out
    }

    fn assert_series_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!(
                (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() < 1e-12,
                "series mismatch: left={lhs:?}, right={rhs:?}"
            );
        }
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_output_contract() {
        let close = sample_close(512);
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            &close,
            AdaptiveBandpassTriggerOscillatorParams::default(),
        );
        let out = adaptive_bandpass_trigger_oscillator(&input).unwrap();
        assert_eq!(out.in_phase.len(), close.len());
        assert_eq!(out.lead.len(), close.len());
        assert!(out.in_phase.iter().position(|v| v.is_finite()).unwrap() >= 13);
        assert!(out.lead.iter().position(|v| v.is_finite()).unwrap() >= 14);
        assert!(out.in_phase.last().unwrap().is_finite());
        assert!(out.lead.last().unwrap().is_finite());
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_rejects_invalid_parameters() {
        let close = sample_close(64);
        let err = adaptive_bandpass_trigger_oscillator(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(0.0),
                    alpha: Some(0.07),
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AdaptiveBandpassTriggerOscillatorError::InvalidDelta { .. }
        ));

        let err = adaptive_bandpass_trigger_oscillator(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(0.1),
                    alpha: Some(1.0),
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AdaptiveBandpassTriggerOscillatorError::InvalidAlpha { .. }
        ));
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_builder_supports_candles() {
        let candles =
            read_candles_from_vortex("src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex").unwrap();
        let out = AdaptiveBandpassTriggerOscillatorBuilder::new()
            .delta(0.1)
            .alpha(0.07)
            .apply(&candles)
            .unwrap();
        assert_eq!(out.in_phase.len(), candles.close.len());
        assert_eq!(out.lead.len(), candles.close.len());
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_stream_matches_batch_with_reset() {
        let mut close = sample_close(256);
        close[120] = f64::NAN;
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            &close,
            AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(0.12),
                alpha: Some(0.08),
            },
        );
        let batch = adaptive_bandpass_trigger_oscillator(&input).unwrap();
        let mut stream = AdaptiveBandpassTriggerOscillatorStream::try_new(
            AdaptiveBandpassTriggerOscillatorParams {
                delta: Some(0.12),
                alpha: Some(0.08),
            },
        )
        .unwrap();
        let mut in_phase = Vec::with_capacity(close.len());
        let mut lead = Vec::with_capacity(close.len());
        for value in close {
            match stream.update(value) {
                Some((bp, ld)) => {
                    in_phase.push(bp);
                    lead.push(ld);
                }
                None => {
                    in_phase.push(f64::NAN);
                    lead.push(f64::NAN);
                }
            }
        }
        assert_series_eq(&batch.in_phase, &in_phase);
        assert_series_eq(&batch.lead, &lead);
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_into_matches_api() {
        let close = sample_close(192);
        let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
            &close,
            AdaptiveBandpassTriggerOscillatorParams::default(),
        );
        let direct = adaptive_bandpass_trigger_oscillator(&input).unwrap();
        let mut in_phase = vec![0.0; close.len()];
        let mut lead = vec![0.0; close.len()];
        adaptive_bandpass_trigger_oscillator_into(&input, &mut in_phase, &mut lead).unwrap();
        assert_series_eq(&direct.in_phase, &in_phase);
        assert_series_eq(&direct.lead, &lead);
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_batch_single_param_matches_single() {
        let close = sample_close(192);
        let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
            delta: (0.1, 0.1, 0.0),
            alpha: (0.07, 0.07, 0.0),
        };
        let batch =
            adaptive_bandpass_trigger_oscillator_batch_with_kernel(&close, &sweep, Kernel::Auto)
                .unwrap();
        let single = adaptive_bandpass_trigger_oscillator(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams::default(),
            ),
        )
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.in_phase[..close.len()], single.in_phase.as_slice());
        assert_series_eq(&batch.lead[..close.len()], single.lead.as_slice());
    }

    #[test]
    fn adaptive_bandpass_trigger_oscillator_batch_metadata() {
        let close = sample_close(160);
        let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
            delta: (0.08, 0.12, 0.04),
            alpha: (0.05, 0.09, 0.02),
        };
        let batch =
            adaptive_bandpass_trigger_oscillator_batch_with_kernel(&close, &sweep, Kernel::Auto)
                .unwrap();
        assert_eq!(batch.rows, 6);
        assert_eq!(batch.cols, close.len());
        assert_eq!(batch.in_phase.len(), 6 * close.len());
        assert_eq!(batch.lead.len(), 6 * close.len());
    }

    #[test]
    fn deterministic_cos_matches_high_precision_golden_domain() {
        // Correctly rounded f64 values generated from a 200-decimal reference
        // over the complete cosine domain reachable by this indicator:
        // beta in (0, pi/3] and cos_angle in (0, 2*pi/3).
        let cases = [
            (f64::from_bits(0x0000000000000000), 0x3ff0000000000000),
            (f64::from_bits(0x3e10000000000000), 0x3ff0000000000000),
            (f64::from_bits(0x3fb999999999999a), 0x3fefd712f9a817c1),
            (f64::from_bits(0x3fd61cba4abb22be), 0x3fee1be4fb35bbbd),
            (f64::from_bits(0x3fdba3e8dd69eb6e), 0x3fed0fd10e88ff77),
            (f64::from_bits(0x3fe921fb54442d17), 0x3fe6a09e667f3bcd),
            (f64::from_bits(0x3fe921fb54442d18), 0x3fe6a09e667f3bcd),
            (f64::from_bits(0x3fe921fb54442d19), 0x3fe6a09e667f3bcc),
            (f64::from_bits(0x3ff0000000000000), 0x3fe14a280fb5068c),
            (f64::from_bits(0x3ff0c152382d7365), 0x3fe0000000000001),
            (f64::from_bits(0x3ff8000000000000), 0x3fb21bd54fc5f9a7),
            (f64::from_bits(0x3ff921fb54442d18), 0x3c91a62633145c07),
            (f64::from_bits(0x4000000000000000), 0xbfdaa22657537205),
            (f64::from_bits(0x4000c152382d7364), 0xbfdffffffffffff5),
        ];
        let ordered_bits = |bits: u64| {
            if bits >> 63 == 0 {
                bits | (1_u64 << 63)
            } else {
                !bits
            }
        };
        for (input, correctly_rounded_bits) in cases {
            let actual_bits = abto_deterministic_cos(input).to_bits();
            let ulp = ordered_bits(actual_bits).abs_diff(ordered_bits(correctly_rounded_bits));
            assert!(
                ulp <= 1,
                "pinned cosine exceeded one ULP from the correctly-rounded reference for \
                 input={input:?}: actual={:?} reference={:?} ulp={ulp}",
                f64::from_bits(actual_bits),
                f64::from_bits(correctly_rounded_bits),
            );
        }
    }
}
