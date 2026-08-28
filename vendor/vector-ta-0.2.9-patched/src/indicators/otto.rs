use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
};

use crate::indicators::cmo::{CmoData, CmoInput, CmoParams, cmo};
use crate::indicators::moving_averages::dema::{DemaData, DemaInput, DemaParams, dema};
use crate::indicators::moving_averages::ema::{EmaData, EmaInput, EmaParams, ema};
use crate::indicators::moving_averages::hma::{HmaData, HmaInput, HmaParams, hma};
use crate::indicators::moving_averages::linreg::{LinRegData, LinRegInput, LinRegParams, linreg};
use crate::indicators::moving_averages::sma::{SmaData, SmaInput, SmaParams, sma};
use crate::indicators::moving_averages::trima::{TrimaData, TrimaInput, TrimaParams, trima};
use crate::indicators::moving_averages::wma::{WmaData, WmaInput, WmaParams, wma};
use crate::indicators::moving_averages::zlema::{ZlemaData, ZlemaInput, ZlemaParams, zlema};
use crate::indicators::tsf::{TsfData, TsfInput, TsfParams, tsf};

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::alloc::{Layout, alloc, dealloc};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for OttoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            OttoData::Slice(slice) => slice,
            OttoData::Candles { candles, source } => match *source {
                "close" => candles.close.as_slice(),
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum OttoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct OttoOutput {
    pub hott: Vec<f64>,
    pub lott: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OttoParams {
    pub ott_period: Option<usize>,
    pub ott_percent: Option<f64>,
    pub fast_vidya_length: Option<usize>,
    pub slow_vidya_length: Option<usize>,
    pub correcting_constant: Option<f64>,
    pub ma_type: Option<String>,
}

impl Default for OttoParams {
    fn default() -> Self {
        Self {
            ott_period: Some(2),
            ott_percent: Some(0.6),
            fast_vidya_length: Some(10),
            slow_vidya_length: Some(25),
            correcting_constant: Some(100000.0),
            ma_type: Some("VAR".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OttoInput<'a> {
    pub data: OttoData<'a>,
    pub params: OttoParams,
}

impl<'a> OttoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: OttoParams) -> Self {
        Self {
            data: OttoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: OttoParams) -> Self {
        Self {
            data: OttoData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", OttoParams::default())
    }

    #[inline]
    pub fn get_ott_period(&self) -> usize {
        self.params.ott_period.unwrap_or(2)
    }

    #[inline]
    pub fn get_ott_percent(&self) -> f64 {
        self.params.ott_percent.unwrap_or(0.6)
    }

    #[inline]
    pub fn get_fast_vidya_length(&self) -> usize {
        self.params.fast_vidya_length.unwrap_or(10)
    }

    #[inline]
    pub fn get_slow_vidya_length(&self) -> usize {
        self.params.slow_vidya_length.unwrap_or(25)
    }

    #[inline]
    pub fn get_correcting_constant(&self) -> f64 {
        self.params.correcting_constant.unwrap_or(100000.0)
    }

    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("VAR")
    }
}

#[derive(Debug, Error)]
pub enum OttoError {
    #[error("otto: Input data slice is empty.")]
    EmptyInputData,
    #[error("otto: All values are NaN.")]
    AllValuesNaN,
    #[error("otto: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("otto: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("otto: Invalid moving average type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error("otto: CMO calculation failed: {0}")]
    CmoError(String),
    #[error("otto: Moving average calculation failed: {0}")]
    MaError(String),
    #[error("otto: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("otto: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("otto: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("otto: Invalid input: {0}")]
    InvalidInput(String),
}

#[derive(Copy, Clone, Debug)]
pub struct OttoBuilder {
    ott_period: Option<usize>,
    ott_percent: Option<f64>,
    fast_vidya_length: Option<usize>,
    slow_vidya_length: Option<usize>,
    correcting_constant: Option<f64>,
    ma_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for OttoBuilder {
    fn default() -> Self {
        Self {
            ott_period: None,
            ott_percent: None,
            fast_vidya_length: None,
            slow_vidya_length: None,
            correcting_constant: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl OttoBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn ott_period(mut self, p: usize) -> Self {
        self.ott_period = Some(p);
        self
    }

    #[inline]
    pub fn ott_percent(mut self, p: f64) -> Self {
        self.ott_percent = Some(p);
        self
    }

    #[inline]
    pub fn fast_vidya_length(mut self, l: usize) -> Self {
        self.fast_vidya_length = Some(l);
        self
    }

    #[inline]
    pub fn slow_vidya_length(mut self, l: usize) -> Self {
        self.slow_vidya_length = Some(l);
        self
    }

    #[inline]
    pub fn correcting_constant(mut self, c: f64) -> Self {
        self.correcting_constant = Some(c);
        self
    }

    #[inline]
    pub fn ma_type(mut self, m: &'static str) -> Self {
        self.ma_type = Some(m);
        self
    }

    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply(self, c: &Candles) -> Result<OttoOutput, OttoError> {
        let params = OttoParams {
            ott_period: self.ott_period,
            ott_percent: self.ott_percent,
            fast_vidya_length: self.fast_vidya_length,
            slow_vidya_length: self.slow_vidya_length,
            correcting_constant: self.correcting_constant,
            ma_type: self.ma_type.map(|s| s.to_string()),
        };
        let input = OttoInput::from_candles(c, "close", params);
        otto_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(self, data: &[f64]) -> Result<OttoOutput, OttoError> {
        let params = OttoParams {
            ott_period: self.ott_period,
            ott_percent: self.ott_percent,
            fast_vidya_length: self.fast_vidya_length,
            slow_vidya_length: self.slow_vidya_length,
            correcting_constant: self.correcting_constant,
            ma_type: self.ma_type.map(|s| s.to_string()),
        };
        let input = OttoInput::from_slice(data, params);
        otto_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<OttoStream, OttoError> {
        let params = OttoParams {
            ott_period: self.ott_period,
            ott_percent: self.ott_percent,
            fast_vidya_length: self.fast_vidya_length,
            slow_vidya_length: self.slow_vidya_length,
            correcting_constant: self.correcting_constant,
            ma_type: self.ma_type.map(|s| s.to_string()),
        };
        OttoStream::try_new(params)
    }
}

#[derive(Debug, Clone)]
pub struct OttoStream {
    ott_period: usize,
    ott_percent: f64,
    fast_vidya_length: usize,
    slow_vidya_length: usize,
    correcting_constant: f64,
    ma_type: String,

    required_len: usize,
    idx: usize,

    a1_base: f64,
    a2_base: f64,
    a3_base: f64,

    a_ott_base: f64,

    fark: f64,
    scale_up: f64,
    scale_dn: f64,

    ring_up_in: [f64; 9],
    ring_dn_in: [f64; 9],
    sum_up_in: f64,
    sum_dn_in: f64,
    head_in: usize,
    prev_x_in: f64,
    have_prev_in: bool,

    v1: f64,
    v2: f64,
    v3: f64,

    last_lott: f64,

    ring_up_lott: [f64; 9],
    ring_dn_lott: [f64; 9],
    sum_up_lott: f64,
    sum_dn_lott: f64,
    head_lott: usize,
    prev_lott: f64,
    have_prev_lott: bool,
    ma_prev: f64,

    ema_alpha: f64,
    ema_init: bool,

    dema_alpha: f64,
    dema_ema1: f64,
    dema_ema2: f64,
    dema_init: bool,

    sma_sum: f64,
    sma_buf: Vec<f64>,
    sma_head: usize,
    sma_count: usize,

    wma_buf: Vec<f64>,
    wma_head: usize,
    wma_count: usize,
    wma_sumx: f64,
    wma_sumwx: f64,
    wma_denom: f64,

    tma_p1: usize,
    tma_p2: usize,
    tma_ring1: Vec<f64>,
    tma_head1: usize,
    tma_sum1: f64,
    tma_count1: usize,
    tma_ring2: Vec<f64>,
    tma_head2: usize,
    tma_sum2: f64,
    tma_count2: usize,

    zlema_alpha: f64,
    zlema_prev: f64,
    zlema_init: bool,
    zlema_lag: usize,
    zlema_ring: Vec<f64>,
    zlema_head: usize,
    zlema_count: usize,

    long_stop_prev: f64,
    short_stop_prev: f64,
    dir_prev: i32,
    ott_init: bool,
}

impl OttoStream {
    pub fn try_new(params: OttoParams) -> Result<Self, OttoError> {
        let ott_period = params.ott_period.unwrap_or(2);
        let slow = params.slow_vidya_length.unwrap_or(25);
        let fast = params.fast_vidya_length.unwrap_or(10);
        let correcting_constant = params.correcting_constant.unwrap_or(100000.0);
        let ma_type = params.ma_type.unwrap_or_else(|| "VAR".to_string());
        let ott_percent = params.ott_percent.unwrap_or(0.6);

        if ott_period == 0 {
            return Err(OttoError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }

        let p1 = slow / 2;
        let p2 = slow;
        let p3 = slow.saturating_mul(fast);
        if p1 == 0 || p2 == 0 || p3 == 0 {
            return Err(OttoError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }

        let a1_base = 2.0 / (p1 as f64 + 1.0);
        let a2_base = 2.0 / (p2 as f64 + 1.0);
        let a3_base = 2.0 / (p3 as f64 + 1.0);
        let a_ott_base = 2.0 / (ott_period as f64 + 1.0);

        let required_len = p3 + 10;

        let fark = ott_percent * 0.01;
        let scale_up = (200.0 + ott_percent) / 200.0;
        let scale_dn = (200.0 - ott_percent) / 200.0;

        let sma_buf = vec![0.0; ott_period];
        let wma_buf = vec![0.0; ott_period];
        let wma_denom = (ott_period as f64) * (ott_period as f64 + 1.0) * 0.5;

        let tma_p1 = (ott_period + 1) / 2;
        let tma_p2 = ott_period / 2 + 1;
        let tma_ring1 = vec![0.0; tma_p1.max(1)];
        let tma_ring2 = vec![0.0; tma_p2.max(1)];

        let zlema_lag = (ott_period.saturating_sub(1)) / 2;
        let zlema_ring = vec![0.0; zlema_lag + 1];

        Ok(Self {
            ott_period,
            ott_percent,
            fast_vidya_length: fast,
            slow_vidya_length: slow,
            correcting_constant,
            ma_type,

            required_len,
            idx: 0,

            a1_base,
            a2_base,
            a3_base,
            a_ott_base,

            fark,
            scale_up,
            scale_dn,

            ring_up_in: [0.0; 9],
            ring_dn_in: [0.0; 9],
            sum_up_in: 0.0,
            sum_dn_in: 0.0,
            head_in: 0,
            prev_x_in: 0.0,
            have_prev_in: false,

            v1: 0.0,
            v2: 0.0,
            v3: 0.0,

            last_lott: 0.0,

            ring_up_lott: [0.0; 9],
            ring_dn_lott: [0.0; 9],
            sum_up_lott: 0.0,
            sum_dn_lott: 0.0,
            head_lott: 0,
            prev_lott: 0.0,
            have_prev_lott: false,
            ma_prev: 0.0,

            ema_alpha: 2.0 / (ott_period as f64 + 1.0),
            ema_init: false,

            dema_alpha: 2.0 / (ott_period as f64 + 1.0),
            dema_ema1: 0.0,
            dema_ema2: 0.0,
            dema_init: false,

            sma_sum: 0.0,
            sma_buf,
            sma_head: 0,
            sma_count: 0,

            wma_buf,
            wma_head: 0,
            wma_count: 0,
            wma_sumx: 0.0,
            wma_sumwx: 0.0,
            wma_denom,

            tma_p1,
            tma_p2,
            tma_ring1,
            tma_head1: 0,
            tma_sum1: 0.0,
            tma_count1: 0,
            tma_ring2,
            tma_head2: 0,
            tma_sum2: 0.0,
            tma_count2: 0,

            zlema_alpha: 2.0 / (ott_period as f64 + 1.0),
            zlema_prev: 0.0,
            zlema_init: false,
            zlema_lag,
            zlema_ring,
            zlema_head: 0,
            zlema_count: 0,

            long_stop_prev: f64::NAN,
            short_stop_prev: f64::NAN,
            dir_prev: 1,
            ott_init: false,
        })
    }

    #[inline]
    fn cmo_abs_from_ring(sum_up: f64, sum_dn: f64) -> f64 {
        let denom = sum_up + sum_dn;
        if denom != 0.0 {
            ((sum_up - sum_dn) / denom).abs()
        } else {
            0.0
        }
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let i = self.idx;
        self.idx = self.idx.wrapping_add(1);

        let x = if value.is_nan() { 0.0 } else { value };

        if self.have_prev_in {
            let mut d = value - self.prev_x_in;
            if !value.is_finite() || !self.prev_x_in.is_finite() {
                d = 0.0;
            }
            if i >= 9 {
                self.sum_up_in -= self.ring_up_in[self.head_in];
                self.sum_dn_in -= self.ring_dn_in[self.head_in];
            }
            let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
            self.ring_up_in[self.head_in] = up;
            self.ring_dn_in[self.head_in] = dn;
            self.sum_up_in += up;
            self.sum_dn_in += dn;
            self.head_in += 1;
            if self.head_in == 9 {
                self.head_in = 0;
            }
        } else {
            self.have_prev_in = true;
        }
        self.prev_x_in = value;

        let c_abs = if i >= 9 {
            Self::cmo_abs_from_ring(self.sum_up_in, self.sum_dn_in)
        } else {
            0.0
        };

        let a1 = self.a1_base * c_abs;
        let a2 = self.a2_base * c_abs;
        let a3 = self.a3_base * c_abs;

        self.v1 = a1.mul_add(x, (1.0 - a1) * self.v1);
        self.v2 = a2.mul_add(x, (1.0 - a2) * self.v2);
        self.v3 = a3.mul_add(x, (1.0 - a3) * self.v3);

        let denom_l = (self.v2 - self.v3) + self.correcting_constant;
        let lott = self.v1 / denom_l;
        self.last_lott = lott;

        let ma_opt = match self.ma_type.as_str() {
            "VAR" => {
                if self.have_prev_lott {
                    let mut d = lott - self.prev_lott;
                    if !lott.is_finite() || !self.prev_lott.is_finite() {
                        d = 0.0;
                    }
                    if i >= 9 {
                        self.sum_up_lott -= self.ring_up_lott[self.head_lott];
                        self.sum_dn_lott -= self.ring_dn_lott[self.head_lott];
                    }
                    let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
                    self.ring_up_lott[self.head_lott] = up;
                    self.ring_dn_lott[self.head_lott] = dn;
                    self.sum_up_lott += up;
                    self.sum_dn_lott += dn;
                    self.head_lott += 1;
                    if self.head_lott == 9 {
                        self.head_lott = 0;
                    }
                } else {
                    self.have_prev_lott = true;
                }
                self.prev_lott = lott;

                let c2 = if i >= 9 {
                    Self::cmo_abs_from_ring(self.sum_up_lott, self.sum_dn_lott)
                } else {
                    0.0
                };
                let a = self.a_ott_base * c2;
                self.ma_prev = a.mul_add(lott, (1.0 - a) * self.ma_prev);
                Some(self.ma_prev)
            }

            "EMA" => {
                if !self.ema_init {
                    self.ma_prev = lott;
                    self.ema_init = true;
                } else {
                    let a = self.ema_alpha;
                    self.ma_prev = a.mul_add(lott, (1.0 - a) * self.ma_prev);
                }
                Some(self.ma_prev)
            }

            "WWMA" => {
                let a = 1.0 / (self.ott_period as f64);
                if !self.ema_init {
                    self.ma_prev = lott;
                    self.ema_init = true;
                } else {
                    self.ma_prev = a.mul_add(lott, (1.0 - a) * self.ma_prev);
                }
                Some(self.ma_prev)
            }

            "DEMA" => {
                let a = self.dema_alpha;
                if !self.dema_init {
                    self.dema_ema1 = lott;
                    self.dema_ema2 = lott;
                    self.dema_init = true;
                } else {
                    self.dema_ema1 = a.mul_add(lott, (1.0 - a) * self.dema_ema1);
                    self.dema_ema2 = a.mul_add(self.dema_ema1, (1.0 - a) * self.dema_ema2);
                }
                Some(2.0 * self.dema_ema1 - self.dema_ema2)
            }

            "SMA" => {
                let p = self.ott_period;
                let _old = if self.sma_count < p {
                    self.sma_count += 1;
                    0.0
                } else {
                    let o = self.sma_buf[self.sma_head];
                    self.sma_sum -= o;
                    o
                };
                self.sma_buf[self.sma_head] = lott;
                self.sma_sum += lott;
                self.sma_head += 1;
                if self.sma_head == p {
                    self.sma_head = 0;
                }
                if self.sma_count >= p {
                    Some(self.sma_sum / p as f64)
                } else {
                    None
                }
            }

            "WMA" => {
                let p = self.ott_period;
                let x_old = if self.wma_count < p {
                    self.wma_count += 1;
                    0.0
                } else {
                    self.wma_buf[self.wma_head]
                };

                self.wma_buf[self.wma_head] = lott;
                self.wma_head += 1;
                if self.wma_head == p {
                    self.wma_head = 0;
                }

                self.wma_sumwx = self.wma_sumwx - self.wma_sumx + (p as f64) * lott;
                self.wma_sumx = self.wma_sumx + lott - x_old;
                if self.wma_count >= p {
                    Some(self.wma_sumwx / self.wma_denom)
                } else {
                    None
                }
            }

            "TMA" => {
                let p1 = self.tma_p1;
                let _o1 = if self.tma_count1 < p1 {
                    self.tma_count1 += 1;
                    0.0
                } else {
                    let o = self.tma_ring1[self.tma_head1];
                    self.tma_sum1 -= o;
                    o
                };
                self.tma_ring1[self.tma_head1] = lott;
                self.tma_sum1 += lott;
                self.tma_head1 += 1;
                if self.tma_head1 == p1 {
                    self.tma_head1 = 0;
                }
                let stage1 = if self.tma_count1 >= p1 {
                    self.tma_sum1 / p1 as f64
                } else {
                    return None;
                };

                let p2 = self.tma_p2;
                let _o2 = if self.tma_count2 < p2 {
                    self.tma_count2 += 1;
                    0.0
                } else {
                    let o = self.tma_ring2[self.tma_head2];
                    self.tma_sum2 -= o;
                    o
                };
                self.tma_ring2[self.tma_head2] = stage1;
                self.tma_sum2 += stage1;
                self.tma_head2 += 1;
                if self.tma_head2 == p2 {
                    self.tma_head2 = 0;
                }
                if self.tma_count2 >= p2 {
                    Some(self.tma_sum2 / p2 as f64)
                } else {
                    None
                }
            }

            "ZLEMA" => {
                let lag = self.zlema_lag;
                let x_lag = if self.zlema_count <= lag {
                    0.0
                } else {
                    self.zlema_ring[(self.zlema_head + self.zlema_ring.len() - lag - 1)
                        % self.zlema_ring.len()]
                };
                let x_adj = 2.0 * lott - x_lag;

                if self.zlema_count < self.zlema_ring.len() {
                    self.zlema_count += 1;
                }
                self.zlema_ring[self.zlema_head] = lott;
                self.zlema_head += 1;
                if self.zlema_head == self.zlema_ring.len() {
                    self.zlema_head = 0;
                }

                let a = self.zlema_alpha;
                if !self.zlema_init {
                    self.zlema_prev = x_adj;
                    self.zlema_init = true;
                } else {
                    self.zlema_prev = a.mul_add(x_adj, (1.0 - a) * self.zlema_prev);
                }
                Some(self.zlema_prev)
            }

            _ => None,
        };

        if self.idx < self.required_len {
            return None;
        }

        let ma = match ma_opt {
            Some(v) => v,
            None => return None,
        };

        if !self.ott_init {
            self.long_stop_prev = ma * (1.0 - self.fark);
            self.short_stop_prev = ma * (1.0 + self.fark);
            let mt = self.long_stop_prev;
            let hott0 = if ma > mt {
                mt * self.scale_up
            } else {
                mt * self.scale_dn
            };
            self.ott_init = true;
            return Some((hott0, lott));
        }

        let ls = ma * (1.0 - self.fark);
        let ss = ma * (1.0 + self.fark);
        let long_stop = if ma > self.long_stop_prev {
            ls.max(self.long_stop_prev)
        } else {
            ls
        };
        let short_stop = if ma < self.short_stop_prev {
            ss.min(self.short_stop_prev)
        } else {
            ss
        };
        let dir = if self.dir_prev == -1 && ma > self.short_stop_prev {
            1
        } else if self.dir_prev == 1 && ma < self.long_stop_prev {
            -1
        } else {
            self.dir_prev
        };
        let mt = if dir == 1 { long_stop } else { short_stop };
        let hott = if ma > mt {
            mt * self.scale_up
        } else {
            mt * self.scale_dn
        };

        self.long_stop_prev = long_stop;
        self.short_stop_prev = short_stop;
        self.dir_prev = dir;

        Some((hott, lott))
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::try_new(OttoParams {
            ott_period: Some(self.ott_period),
            ott_percent: Some(self.ott_percent),
            fast_vidya_length: Some(self.fast_vidya_length),
            slow_vidya_length: Some(self.slow_vidya_length),
            correcting_constant: Some(self.correcting_constant),
            ma_type: Some(self.ma_type.clone()),
        })
        .expect("OttoStream::reset: params should remain valid");
    }
}

fn cmo_sum_based(data: &[f64], period: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; data.len()];

    if data.len() < period + 1 {
        return output;
    }

    for i in period..data.len() {
        let mut sum_up = 0.0;
        let mut sum_down = 0.0;

        for j in 1..=period {
            let idx = i - period + j;
            if idx > 0 {
                let diff = data[idx] - data[idx - 1];
                if diff > 0.0 {
                    sum_up += diff;
                } else {
                    sum_down += diff.abs();
                }
            }
        }

        let sum_total = sum_up + sum_down;
        if sum_total != 0.0 {
            output[i] = (sum_up - sum_down) / sum_total;
        } else {
            output[i] = 0.0;
        }
    }

    output
}

fn vidya(data: &[f64], period: usize) -> Result<Vec<f64>, OttoError> {
    if data.is_empty() {
        return Err(OttoError::EmptyInputData);
    }

    if period == 0 || period > data.len() {
        return Err(OttoError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let mut output = vec![f64::NAN; data.len()];

    let cmo_values = cmo_sum_based(data, 9);

    let mut var_prev = 0.0;

    for i in 0..data.len() {
        let current_value = if data[i].is_nan() { 0.0 } else { data[i] };
        let current_cmo = if cmo_values[i].is_nan() {
            0.0
        } else {
            cmo_values[i]
        };

        if i == 0 {
            let abs_cmo = current_cmo.abs();
            let adaptive_alpha = alpha * abs_cmo;
            var_prev = adaptive_alpha * current_value + (1.0 - adaptive_alpha) * 0.0;
            output[i] = var_prev;
        } else {
            let abs_cmo = current_cmo.abs();
            let adaptive_alpha = alpha * abs_cmo;
            var_prev = adaptive_alpha * current_value + (1.0 - adaptive_alpha) * var_prev;
            output[i] = var_prev;
        }
    }

    Ok(output)
}

fn tma_custom(data: &[f64], period: usize) -> Result<Vec<f64>, OttoError> {
    if period <= 0 || period > data.len() {
        return Err(OttoError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_period = (period + 1) / 2;
    let second_period = period / 2 + 1;

    let params1 = SmaParams {
        period: Some(first_period),
    };
    let input1 = SmaInput::from_slice(data, params1);
    let sma1 = sma(&input1).map_err(|e| OttoError::MaError(e.to_string()))?;

    let params2 = SmaParams {
        period: Some(second_period),
    };
    let input2 = SmaInput::from_slice(&sma1.values, params2);
    let sma2 = sma(&input2).map_err(|e| OttoError::MaError(e.to_string()))?;

    Ok(sma2.values)
}

fn wwma(data: &[f64], period: usize) -> Result<Vec<f64>, OttoError> {
    if data.is_empty() {
        return Err(OttoError::EmptyInputData);
    }

    if period == 0 || period > data.len() {
        return Err(OttoError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let alpha = 1.0 / period as f64;
    let mut output = vec![f64::NAN; data.len()];

    let first_valid = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);

    let mut sum = 0.0;
    let mut count = 0;
    for i in first_valid..first_valid.saturating_add(period).min(data.len()) {
        if !data[i].is_nan() {
            sum += data[i];
            count += 1;
        }
    }

    if count > 0 {
        let mut wwma_prev = sum / count as f64;
        output[first_valid + period - 1] = wwma_prev;

        for i in first_valid + period..data.len() {
            if !data[i].is_nan() {
                wwma_prev = alpha * data[i] + (1.0 - alpha) * wwma_prev;
                output[i] = wwma_prev;
            } else {
                output[i] = wwma_prev;
            }
        }
    }

    Ok(output)
}

fn calculate_ma(data: &[f64], period: usize, ma_type: &str) -> Result<Vec<f64>, OttoError> {
    match ma_type {
        "SMA" => {
            let params = SmaParams {
                period: Some(period),
            };
            let input = SmaInput::from_slice(data, params);
            sma(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        "EMA" => {
            let params = EmaParams {
                period: Some(period),
            };
            let input = EmaInput::from_slice(data, params);
            ema(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        "WMA" => {
            let params = WmaParams {
                period: Some(period),
            };
            let input = WmaInput::from_slice(data, params);
            wma(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        "WWMA" => wwma(data, period),
        "DEMA" => {
            let params = DemaParams {
                period: Some(period),
            };
            let input = DemaInput::from_slice(data, params);
            dema(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        "TMA" => tma_custom(data, period),
        "VAR" => vidya(data, period),
        "ZLEMA" => {
            let params = ZlemaParams {
                period: Some(period),
            };
            let input = ZlemaInput::from_slice(data, params);
            zlema(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        "TSF" => {
            let params = TsfParams {
                period: Some(period),
            };
            let input = TsfInput::from_slice(data, params);
            tsf(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        "HULL" => {
            let params = HmaParams {
                period: Some(period),
            };
            let input = HmaInput::from_slice(data, params);
            hma(&input)
                .map(|o| o.values)
                .map_err(|e| OttoError::MaError(e.to_string()))
        }
        _ => Err(OttoError::InvalidMaType {
            ma_type: ma_type.to_string(),
        }),
    }
}

#[inline]
fn resolve_single_kernel(k: Kernel) -> Kernel {
    match k {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    }
}

#[inline]
fn resolve_batch_kernel(k: Kernel) -> Result<Kernel, OttoError> {
    Ok(match k {
        Kernel::Auto => detect_best_batch_kernel(),
        b if b.is_batch() => b,
        other => {
            return Err(OttoError::InvalidKernelForBatch(other));
        }
    })
}

#[inline]
fn first_valid_idx_and_all_finite(d: &[f64]) -> Result<(usize, bool), OttoError> {
    let mut first = None;
    let mut all_finite = true;
    for (i, &value) in d.iter().enumerate() {
        if !value.is_finite() {
            all_finite = false;
        }
        if first.is_none() && !value.is_nan() {
            first = Some(i);
        }
    }
    first
        .map(|idx| (idx, all_finite))
        .ok_or(OttoError::AllValuesNaN)
}

#[inline]
fn otto_prepare<'a>(
    input: &'a OttoInput,
) -> Result<(&'a [f64], usize, usize, usize, f64, &'a str, bool), OttoError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(OttoError::EmptyInputData);
    }

    let (first, all_finite) = first_valid_idx_and_all_finite(data)?;
    let ott_period = input.get_ott_period();
    if ott_period == 0 || ott_period > data.len() {
        return Err(OttoError::InvalidPeriod {
            period: ott_period,
            data_len: data.len(),
        });
    }

    let ott_percent = input.get_ott_percent();
    let ma_type = input.get_ma_type();

    let slow = input.get_slow_vidya_length();
    let fast = input.get_fast_vidya_length();

    let needed = (slow * fast).max(10);
    let valid = data.len() - first;
    if valid < needed {
        return Err(OttoError::NotEnoughValidData { needed, valid });
    }

    Ok((
        data,
        first,
        ott_period,
        needed,
        ott_percent,
        ma_type,
        all_finite,
    ))
}

#[inline]
pub fn otto_into_slices(
    hott_dst: &mut [f64],
    lott_dst: &mut [f64],
    input: &OttoInput,
    _kern: Kernel,
) -> Result<(), OttoError> {
    let (data, _first, ott_p, _needed, ott_percent, ma_type, all_finite) = otto_prepare(input)?;
    let n = data.len();
    if hott_dst.len() != n || lott_dst.len() != n {
        let expected = n;
        let got = hott_dst.len().max(lott_dst.len());
        return Err(OttoError::OutputLengthMismatch { expected, got });
    }

    let slow = input.get_slow_vidya_length();
    let fast = input.get_fast_vidya_length();
    let p1 = slow / 2;
    let p2 = slow;
    let p3 = slow.saturating_mul(fast);

    if p1 == 0 || p2 == 0 || p3 == 0 {
        return Err(OttoError::InvalidPeriod {
            period: 0,
            data_len: n,
        });
    }

    let coco = input.get_correcting_constant();

    if ma_type == "VAR" && all_finite {
        otto_var_clean_two_pass_into_slices(
            data,
            hott_dst,
            lott_dst,
            p1,
            p2,
            p3,
            coco,
            ott_p,
            ott_percent,
        );
        return Ok(());
    }

    let a1_base = 2.0 / (p1 as f64 + 1.0);
    let a2_base = 2.0 / (p2 as f64 + 1.0);
    let a3_base = 2.0 / (p3 as f64 + 1.0);

    const CMO_P: usize = 9;
    let mut ring_up = [0.0f64; CMO_P];
    let mut ring_dn = [0.0f64; CMO_P];
    let mut sum_up = 0.0f64;
    let mut sum_dn = 0.0f64;
    let mut head = 0usize;

    let mut v1 = 0.0f64;
    let mut v2 = 0.0f64;
    let mut v3 = 0.0f64;

    let mut prev_x = if n > 0 { data[0] } else { f64::NAN };

    for i in 0..n {
        let x = data[i];
        let val = if x.is_nan() { 0.0 } else { x };

        if i > 0 {
            let mut d = x - prev_x;
            if !x.is_finite() || !prev_x.is_finite() {
                d = 0.0;
            }

            if i >= CMO_P {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }

            let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;

            head += 1;
            if head == CMO_P {
                head = 0;
            }

            prev_x = x;
        }

        let cmo_abs = if i >= CMO_P {
            let denom = sum_up + sum_dn;
            if denom != 0.0 {
                ((sum_up - sum_dn) / denom).abs()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let a1 = a1_base * cmo_abs;
        let a2 = a2_base * cmo_abs;
        let a3 = a3_base * cmo_abs;
        v1 = a1 * val + (1.0 - a1) * v1;
        v2 = a2 * val + (1.0 - a2) * v2;
        v3 = a3 * val + (1.0 - a3) * v3;

        let denom_l = (v2 - v3) + coco;
        lott_dst[i] = v1 / denom_l;
    }

    let fark = ott_percent * 0.01;
    let scale_up = (200.0 + ott_percent) / 200.0;
    let scale_dn = (200.0 - ott_percent) / 200.0;

    if ma_type == "VAR" {
        const CMO_P2: usize = 9;
        let mut ring_up2 = [0.0f64; CMO_P2];
        let mut ring_dn2 = [0.0f64; CMO_P2];
        let mut sum_up2 = 0.0f64;
        let mut sum_dn2 = 0.0f64;
        let mut head2 = 0usize;
        let mut prev_lott = lott_dst[0];

        let a_base = 2.0 / (ott_p as f64 + 1.0);
        let mut ma_prev = 0.0f64;

        let mut long_stop_prev = f64::NAN;
        let mut short_stop_prev = f64::NAN;
        let mut dir_prev: i32 = 1;

        for i in 0..n {
            if i > 0 {
                let x = lott_dst[i];
                let mut d = x - prev_lott;
                if !x.is_finite() || !prev_lott.is_finite() {
                    d = 0.0;
                }
                if i >= CMO_P2 {
                    sum_up2 -= ring_up2[head2];
                    sum_dn2 -= ring_dn2[head2];
                }
                let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
                ring_up2[head2] = up;
                ring_dn2[head2] = dn;
                sum_up2 += up;
                sum_dn2 += dn;
                head2 += 1;
                if head2 == CMO_P2 {
                    head2 = 0;
                }
                prev_lott = x;
            }

            let c_abs = if i >= CMO_P2 {
                let denom = sum_up2 + sum_dn2;
                if denom != 0.0 {
                    ((sum_up2 - sum_dn2) / denom).abs()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let a = a_base * c_abs;
            let ma = a * lott_dst[i] + (1.0 - a) * ma_prev;
            ma_prev = ma;

            if i == 0 {
                long_stop_prev = ma * (1.0 - fark);
                short_stop_prev = ma * (1.0 + fark);
                let mt = long_stop_prev;
                hott_dst[i] = if ma > mt {
                    mt * scale_up
                } else {
                    mt * scale_dn
                };
            } else {
                let ls = ma * (1.0 - fark);
                let ss = ma * (1.0 + fark);
                let long_stop = if ma > long_stop_prev {
                    ls.max(long_stop_prev)
                } else {
                    ls
                };
                let short_stop = if ma < short_stop_prev {
                    ss.min(short_stop_prev)
                } else {
                    ss
                };
                let dir = if dir_prev == -1 && ma > short_stop_prev {
                    1
                } else if dir_prev == 1 && ma < long_stop_prev {
                    -1
                } else {
                    dir_prev
                };
                let mt = if dir == 1 { long_stop } else { short_stop };
                hott_dst[i] = if ma > mt {
                    mt * scale_up
                } else {
                    mt * scale_dn
                };
                long_stop_prev = long_stop;
                short_stop_prev = short_stop;
                dir_prev = dir;
            }
        }
    } else {
        let mavg = calculate_ma(lott_dst, ott_p, ma_type)?;

        let mut long_stop_prev = f64::NAN;
        let mut short_stop_prev = f64::NAN;
        let mut dir_prev: i32 = 1;

        let start = mavg.iter().position(|&x| !x.is_nan()).unwrap_or(n);
        for i in 0..start.min(n) {
            hott_dst[i] = f64::NAN;
        }
        if start < n {
            let ma0 = mavg[start];
            long_stop_prev = ma0 * (1.0 - fark);
            short_stop_prev = ma0 * (1.0 + fark);
            let mt0 = long_stop_prev;
            hott_dst[start] = if ma0 > mt0 {
                mt0 * scale_up
            } else {
                mt0 * scale_dn
            };
            for i in (start + 1)..n {
                let ma = mavg[i];
                if ma.is_nan() {
                    hott_dst[i] = hott_dst[i - 1];
                    continue;
                }
                let ls = ma * (1.0 - fark);
                let ss = ma * (1.0 + fark);
                let long_stop = if ma > long_stop_prev {
                    ls.max(long_stop_prev)
                } else {
                    ls
                };
                let short_stop = if ma < short_stop_prev {
                    ss.min(short_stop_prev)
                } else {
                    ss
                };
                let dir = if dir_prev == -1 && ma > short_stop_prev {
                    1
                } else if dir_prev == 1 && ma < long_stop_prev {
                    -1
                } else {
                    dir_prev
                };
                let mt = if dir == 1 { long_stop } else { short_stop };
                hott_dst[i] = if ma > mt {
                    mt * scale_up
                } else {
                    mt * scale_dn
                };
                long_stop_prev = long_stop;
                short_stop_prev = short_stop;
                dir_prev = dir;
            }
        }
    }

    Ok(())
}

#[inline(always)]
fn otto_var_clean_two_pass_into_slices(
    data: &[f64],
    hott_dst: &mut [f64],
    lott_dst: &mut [f64],
    p1: usize,
    p2: usize,
    p3: usize,
    coco: f64,
    ott_p: usize,
    ott_percent: f64,
) {
    let n = data.len();

    let a1_base = 2.0 / (p1 as f64 + 1.0);
    let a2_base = 2.0 / (p2 as f64 + 1.0);
    let a3_base = 2.0 / (p3 as f64 + 1.0);

    const CMO_P: usize = 9;
    let mut ring_up = [0.0f64; CMO_P];
    let mut ring_dn = [0.0f64; CMO_P];
    let mut sum_up = 0.0f64;
    let mut sum_dn = 0.0f64;
    let mut head = 0usize;

    let mut v1 = 0.0f64;
    let mut v2 = 0.0f64;
    let mut v3 = 0.0f64;
    let mut prev_x = data[0];

    for i in 0..n {
        let x = data[i];

        if i > 0 {
            let d = x - prev_x;

            if i >= CMO_P {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }

            let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;

            head += 1;
            if head == CMO_P {
                head = 0;
            }

            prev_x = x;
        }

        let cmo_abs = if i >= CMO_P {
            let denom = sum_up + sum_dn;
            if denom != 0.0 {
                ((sum_up - sum_dn) / denom).abs()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let a1 = a1_base * cmo_abs;
        let a2 = a2_base * cmo_abs;
        let a3 = a3_base * cmo_abs;
        v1 = a1 * x + (1.0 - a1) * v1;
        v2 = a2 * x + (1.0 - a2) * v2;
        v3 = a3 * x + (1.0 - a3) * v3;

        let denom_l = (v2 - v3) + coco;
        lott_dst[i] = v1 / denom_l;
    }

    let fark = ott_percent * 0.01;
    let scale_up = (200.0 + ott_percent) / 200.0;
    let scale_dn = (200.0 - ott_percent) / 200.0;

    const CMO_P2: usize = 9;
    let mut ring_up2 = [0.0f64; CMO_P2];
    let mut ring_dn2 = [0.0f64; CMO_P2];
    let mut sum_up2 = 0.0f64;
    let mut sum_dn2 = 0.0f64;
    let mut head2 = 0usize;
    let mut prev_lott = lott_dst[0];

    let a_base = 2.0 / (ott_p as f64 + 1.0);
    let mut ma_prev = 0.0f64;

    let mut long_stop_prev = f64::NAN;
    let mut short_stop_prev = f64::NAN;
    let mut dir_prev = 1i32;

    for i in 0..n {
        if i > 0 {
            let x = lott_dst[i];
            let d = x - prev_lott;
            if i >= CMO_P2 {
                sum_up2 -= ring_up2[head2];
                sum_dn2 -= ring_dn2[head2];
            }
            let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
            ring_up2[head2] = up;
            ring_dn2[head2] = dn;
            sum_up2 += up;
            sum_dn2 += dn;
            head2 += 1;
            if head2 == CMO_P2 {
                head2 = 0;
            }
            prev_lott = x;
        }

        let c_abs = if i >= CMO_P2 {
            let denom = sum_up2 + sum_dn2;
            if denom != 0.0 {
                ((sum_up2 - sum_dn2) / denom).abs()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let a = a_base * c_abs;
        let ma = a * lott_dst[i] + (1.0 - a) * ma_prev;
        ma_prev = ma;

        if i == 0 {
            long_stop_prev = ma * (1.0 - fark);
            short_stop_prev = ma * (1.0 + fark);
            let mt = long_stop_prev;
            hott_dst[i] = if ma > mt {
                mt * scale_up
            } else {
                mt * scale_dn
            };
        } else {
            let ls = ma * (1.0 - fark);
            let ss = ma * (1.0 + fark);
            let long_stop = if ma > long_stop_prev {
                ls.max(long_stop_prev)
            } else {
                ls
            };
            let short_stop = if ma < short_stop_prev {
                ss.min(short_stop_prev)
            } else {
                ss
            };
            let dir = if dir_prev == -1 && ma > short_stop_prev {
                1
            } else if dir_prev == 1 && ma < long_stop_prev {
                -1
            } else {
                dir_prev
            };
            let mt = if dir == 1 { long_stop } else { short_stop };
            hott_dst[i] = if ma > mt {
                mt * scale_up
            } else {
                mt * scale_dn
            };
            long_stop_prev = long_stop;
            short_stop_prev = short_stop;
            dir_prev = dir;
        }
    }
}

pub fn otto_with_kernel(input: &OttoInput, kern: Kernel) -> Result<OttoOutput, OttoError> {
    let chosen = resolve_single_kernel(kern);
    let data = input.as_ref();
    if data.is_empty() {
        return Err(OttoError::EmptyInputData);
    }

    let mut hott = alloc_with_nan_prefix(data.len(), 0);
    let mut lott = alloc_with_nan_prefix(data.len(), 0);

    otto_into_slices(&mut hott, &mut lott, input, chosen)?;
    Ok(OttoOutput { hott, lott })
}

#[inline(always)]
pub fn otto(input: &OttoInput) -> Result<OttoOutput, OttoError> {
    otto_with_kernel(input, Kernel::Scalar)
}

#[inline]
pub fn otto_into(
    input: &OttoInput,
    hott_out: &mut [f64],
    lott_out: &mut [f64],
) -> Result<(), OttoError> {
    otto_into_slices(hott_out, lott_out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct OttoBatchRange {
    pub ott_period: (usize, usize, usize),
    pub ott_percent: (f64, f64, f64),
    pub fast_vidya: (usize, usize, usize),
    pub slow_vidya: (usize, usize, usize),
    pub correcting_constant: (f64, f64, f64),
    pub ma_types: Vec<String>,
}

impl Default for OttoBatchRange {
    fn default() -> Self {
        Self {
            ott_period: (2, 251, 1),
            ott_percent: (0.6, 0.6, 0.0),
            fast_vidya: (10, 10, 0),
            slow_vidya: (25, 25, 0),
            correcting_constant: (100000.0, 100000.0, 0.0),
            ma_types: vec!["VAR".into()],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OttoBatchBuilder {
    range: OttoBatchRange,
    kernel: Kernel,
}

impl OttoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn ott_period_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.ott_period = (s, e, step);
        self
    }
    pub fn ott_percent_range(mut self, s: f64, e: f64, step: f64) -> Self {
        self.range.ott_percent = (s, e, step);
        self
    }
    pub fn fast_vidya_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.fast_vidya = (s, e, step);
        self
    }
    pub fn slow_vidya_range(mut self, s: usize, e: usize, step: usize) -> Self {
        self.range.slow_vidya = (s, e, step);
        self
    }
    pub fn correcting_constant_range(mut self, s: f64, e: f64, step: f64) -> Self {
        self.range.correcting_constant = (s, e, step);
        self
    }
    pub fn ma_types(mut self, v: Vec<String>) -> Self {
        self.range.ma_types = v;
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<OttoBatchOutput, OttoError> {
        otto_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<OttoBatchOutput, OttoError> {
        self.apply_slice(source_type(c, src))
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<OttoBatchOutput, OttoError> {
        OttoBatchBuilder::new().kernel(k).apply_slice(data)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OttoBatchCombo(pub OttoParams);

#[derive(Clone, Debug)]
pub struct OttoBatchOutput {
    pub hott: Vec<f64>,
    pub lott: Vec<f64>,
    pub combos: Vec<OttoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline]
fn axis_usize(a: (usize, usize, usize)) -> Result<Vec<usize>, OttoError> {
    let (start, end, step) = a;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    if start < end {
        let v: Vec<_> = (start..=end).step_by(step).collect();
        if v.is_empty() {
            return Err(OttoError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    } else {
        let mut v = Vec::new();
        let mut cur = start;
        while cur >= end {
            v.push(cur);
            if cur - end < step {
                break;
            }
            cur -= step;
        }
        if v.is_empty() {
            return Err(OttoError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
}

#[inline]
fn axis_f64(a: (f64, f64, f64)) -> Result<Vec<f64>, OttoError> {
    let (start, end, step) = a;
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let st = if step > 0.0 { step } else { -step };
        let mut x = start;
        while x <= end + 1e-12 {
            out.push(x);
            x += st;
        }
    } else {
        let st = if step > 0.0 { -step } else { step };
        if st.abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut x = start;
        while x >= end - 1e-12 {
            out.push(x);
            x += st;
        }
    }
    if out.is_empty() {
        return Err(OttoError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_otto(r: &OttoBatchRange) -> Result<Vec<OttoParams>, OttoError> {
    let p = axis_usize(r.ott_period)?;
    let op = axis_f64(r.ott_percent)?;
    let fv = axis_usize(r.fast_vidya)?;
    let sv = axis_usize(r.slow_vidya)?;
    let cc = axis_f64(r.correcting_constant)?;
    let mt = &r.ma_types;

    if mt.is_empty() {
        return Err(OttoError::InvalidRange {
            start: "ma_types".into(),
            end: "ma_types".into(),
            step: "0".into(),
        });
    }

    let mut v = Vec::with_capacity(
        p.len()
            .saturating_mul(op.len())
            .saturating_mul(fv.len())
            .saturating_mul(sv.len())
            .saturating_mul(cc.len())
            .saturating_mul(mt.len()),
    );
    for &pp in &p {
        for &oo in &op {
            for &ff in &fv {
                for &ss in &sv {
                    for &ccv in &cc {
                        for m in mt {
                            v.push(OttoParams {
                                ott_period: Some(pp),
                                ott_percent: Some(oo),
                                fast_vidya_length: Some(ff),
                                slow_vidya_length: Some(ss),
                                correcting_constant: Some(ccv),
                                ma_type: Some(m.clone()),
                            });
                        }
                    }
                }
            }
        }
    }
    if v.is_empty() {
        return Err(OttoError::InvalidRange {
            start: "otto_batch".into(),
            end: "otto_batch".into(),
            step: "0".into(),
        });
    }
    Ok(v)
}

#[inline]
fn cmo_abs9_stream(data: &[f64]) -> Vec<f64> {
    const CMO_P: usize = 9;
    let n = data.len();
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }

    let mut ring_up = [0.0f64; CMO_P];
    let mut ring_dn = [0.0f64; CMO_P];
    let mut sum_up = 0.0f64;
    let mut sum_dn = 0.0f64;
    let mut head = 0usize;
    let mut prev_x = data[0];

    for i in 0..n {
        let x = data[i];
        if i > 0 {
            let mut d = x - prev_x;
            if !x.is_finite() || !prev_x.is_finite() {
                d = 0.0;
            }
            if i >= CMO_P {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }
            let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;
            head += 1;
            if head == CMO_P {
                head = 0;
            }
            prev_x = x;
        }
        if i >= CMO_P {
            let denom = sum_up + sum_dn;
            out[i] = if denom != 0.0 {
                ((sum_up - sum_dn) / denom).abs()
            } else {
                0.0
            };
        } else {
            out[i] = 0.0;
        }
    }
    out
}

pub fn otto_batch_with_kernel(
    data: &[f64],
    sweep: &OttoBatchRange,
    k: Kernel,
) -> Result<OttoBatchOutput, OttoError> {
    if data.is_empty() {
        return Err(OttoError::EmptyInputData);
    }
    let kernel = resolve_batch_kernel(k)?;

    let combos = expand_grid_otto(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| OttoError::InvalidInput("rows*cols overflow".into()))?;

    let mut hott = vec![f64::NAN; total];
    let mut lott = vec![f64::NAN; total];

    let cmo_abs = cmo_abs9_stream(data);

    for (row, prm) in combos.iter().enumerate() {
        let input = OttoInput::from_slice(data, prm.clone());

        let (_d, _first, ott_p, _needed, ott_percent, ma_type, _all_finite) = otto_prepare(&input)?;

        let n = data.len();
        let slow = input.get_slow_vidya_length();
        let fast = input.get_fast_vidya_length();
        let p1 = slow / 2;
        let p2 = slow;
        let p3 = slow.saturating_mul(fast);
        if p1 == 0 || p2 == 0 || p3 == 0 {
            return Err(OttoError::InvalidPeriod {
                period: 0,
                data_len: n,
            });
        }

        let a1_base = 2.0 / (p1 as f64 + 1.0);
        let a2_base = 2.0 / (p2 as f64 + 1.0);
        let a3_base = 2.0 / (p3 as f64 + 1.0);
        let coco = input.get_correcting_constant();

        let row_l = &mut lott[row * cols..(row + 1) * cols];
        let row_h = &mut hott[row * cols..(row + 1) * cols];

        let mut v1 = 0.0f64;
        let mut v2 = 0.0f64;
        let mut v3 = 0.0f64;
        for i in 0..n {
            let x = data[i];
            let val = if x.is_nan() { 0.0 } else { x };
            let c = cmo_abs[i];
            let a1 = a1_base * c;
            let a2 = a2_base * c;
            let a3 = a3_base * c;
            v1 = a1 * val + (1.0 - a1) * v1;
            v2 = a2 * val + (1.0 - a2) * v2;
            v3 = a3 * val + (1.0 - a3) * v3;
            row_l[i] = v1 / ((v2 - v3) + coco);
        }

        let mavg = calculate_ma(row_l, ott_p, &ma_type)?;
        let fark = ott_percent * 0.01;
        let scale_up = (200.0 + ott_percent) / 200.0;
        let scale_dn = (200.0 - ott_percent) / 200.0;

        let mut long_stop_prev = f64::NAN;
        let mut short_stop_prev = f64::NAN;
        let mut dir_prev: i32 = 1;

        let start = mavg.iter().position(|&x| !x.is_nan()).unwrap_or(n);
        for i in 0..start.min(n) {
            row_h[i] = f64::NAN;
        }

        if start < n {
            let ma0 = mavg[start];
            long_stop_prev = ma0 * (1.0 - fark);
            short_stop_prev = ma0 * (1.0 + fark);
            let mt0 = long_stop_prev;
            row_h[start] = if ma0 > mt0 {
                mt0 * scale_up
            } else {
                mt0 * scale_dn
            };
            for i in (start + 1)..n {
                let ma = mavg[i];
                if ma.is_nan() {
                    row_h[i] = row_h[i - 1];
                    continue;
                }
                let ls = ma * (1.0 - fark);
                let ss = ma * (1.0 + fark);
                let long_stop = if ma > long_stop_prev {
                    ls.max(long_stop_prev)
                } else {
                    ls
                };
                let short_stop = if ma < short_stop_prev {
                    ss.min(short_stop_prev)
                } else {
                    ss
                };
                let dir = if dir_prev == -1 && ma > short_stop_prev {
                    1
                } else if dir_prev == 1 && ma < long_stop_prev {
                    -1
                } else {
                    dir_prev
                };
                let mt = if dir == 1 { long_stop } else { short_stop };
                row_h[i] = if ma > mt {
                    mt * scale_up
                } else {
                    mt * scale_dn
                };
                long_stop_prev = long_stop;
                short_stop_prev = short_stop;
                dir_prev = dir;
            }
        }
    }

    Ok(OttoBatchOutput {
        hott,
        lott,
        combos,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn generate_otto_test_data(n: usize) -> Vec<f64> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(0.612 - (i as f64 * 0.00001));
        }
        data
    }

    fn check_otto_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = OttoParams {
            ott_period: None,
            ott_percent: Some(0.8),
            fast_vidya_length: None,
            slow_vidya_length: Some(20),
            correcting_constant: None,
            ma_type: None,
        };

        let input = OttoInput::from_candles(&candles, "close", params);
        let output = otto_with_kernel(&input, kernel)?;

        assert_eq!(output.hott.len(), candles.close.len());
        assert_eq!(output.lott.len(), candles.close.len());

        Ok(())
    }

    fn check_otto_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = OttoParams::default();
        let input = OttoInput::from_candles(&candles, "close", params);
        let result = otto_with_kernel(&input, kernel)?;

        let expected_hott = [
            0.6137310801679211,
            0.6136758137211143,
            0.6135129389965592,
            0.6133345015018311,
            0.6130191362868016,
        ];
        let expected_lott = [
            0.6118478692473065,
            0.6118237221582352,
            0.6116076875101266,
            0.6114220222840161,
            0.6110393343841534,
        ];

        let start = result.hott.len().saturating_sub(5);
        for (i, &expected) in expected_hott.iter().enumerate() {
            let actual = result.hott[start + i];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "[{}] OTTO HOTT {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                actual,
                expected
            );
        }

        for (i, &expected) in expected_lott.iter().enumerate() {
            let actual = result.lott[start + i];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "[{}] OTTO LOTT {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                actual,
                expected
            );
        }

        Ok(())
    }

    fn check_otto_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = OttoInput::with_default_candles(&candles);
        let output = otto_with_kernel(&input, kernel)?;

        assert_eq!(output.hott.len(), candles.close.len());
        assert_eq!(output.lott.len(), candles.close.len());

        Ok(())
    }

    fn check_otto_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = OttoParams {
            ott_period: Some(0),
            ..Default::default()
        };

        let input = OttoInput::from_candles(&candles, "close", params);
        let result = otto_with_kernel(&input, kernel);

        assert!(
            result.is_err(),
            "[{}] Expected error for zero period",
            test_name
        );

        Ok(())
    }

    fn check_otto_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let small_data = &candles.close[0..3];

        let params = OttoParams {
            ott_period: Some(10),
            ..Default::default()
        };

        let input = OttoInput::from_slice(small_data, params);
        let result = otto_with_kernel(&input, kernel);

        assert!(
            result.is_err(),
            "[{}] Expected error when period exceeds length",
            test_name
        );

        Ok(())
    }

    fn check_otto_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let small_data = &candles.close[0..15];

        let params = OttoParams {
            ott_period: Some(1),
            ott_percent: Some(0.5),
            fast_vidya_length: Some(1),
            slow_vidya_length: Some(2),
            correcting_constant: Some(1.0),
            ma_type: Some("SMA".to_string()),
        };

        let input = OttoInput::from_slice(small_data, params);
        let result = otto_with_kernel(&input, kernel);

        assert!(
            result.is_ok(),
            "[{}] Should handle very small dataset: {:?}",
            test_name,
            result
        );

        Ok(())
    }

    fn check_otto_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data: Vec<f64> = vec![];
        let params = OttoParams::default();

        let input = OttoInput::from_slice(&data, params);
        let result = otto_with_kernel(&input, kernel);

        assert!(
            result.is_err(),
            "[{}] Expected error for empty input",
            test_name
        );

        Ok(())
    }

    fn check_otto_invalid_ma_type(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = OttoParams {
            ma_type: Some("INVALID_MA".to_string()),
            ..Default::default()
        };

        let input = OttoInput::from_candles(&candles, "close", params);
        let result = otto_with_kernel(&input, kernel);

        assert!(
            result.is_err(),
            "[{}] Expected error for invalid MA type",
            test_name
        );

        Ok(())
    }

    fn check_otto_all_ma_types(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let ma_types = [
            "SMA", "EMA", "WMA", "DEMA", "TMA", "VAR", "ZLEMA", "TSF", "HULL",
        ];

        for ma_type in &ma_types {
            let params = OttoParams {
                ma_type: Some(ma_type.to_string()),
                ..Default::default()
            };

            let input = OttoInput::from_candles(&candles, "close", params);
            let result = otto_with_kernel(&input, kernel)?;

            assert_eq!(
                result.hott.len(),
                candles.close.len(),
                "[{}] MA type {} output length mismatch",
                test_name,
                ma_type
            );
        }

        Ok(())
    }

    fn check_otto_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = OttoParams::default();
        let input = OttoInput::from_candles(&candles, "close", params);

        let result1 = otto_with_kernel(&input, kernel)?;
        let result2 = otto_with_kernel(&input, kernel)?;

        for i in 0..result1.hott.len() {
            if result1.hott[i].is_finite() && result2.hott[i].is_finite() {
                assert!(
                    (result1.hott[i] - result2.hott[i]).abs() < 1e-10,
                    "[{}] Reinput produced different HOTT at index {}",
                    test_name,
                    i
                );
            }
            if result1.lott[i].is_finite() && result2.lott[i].is_finite() {
                assert!(
                    (result1.lott[i] - result2.lott[i]).abs() < 1e-10,
                    "[{}] Reinput produced different LOTT at index {}",
                    test_name,
                    i
                );
            }
        }

        Ok(())
    }

    fn check_otto_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let mut data = candles.close.clone();

        data[100] = f64::NAN;
        data[150] = f64::NAN;
        data[200] = f64::NAN;

        let params = OttoParams::default();
        let input = OttoInput::from_slice(&data, params);
        let result = otto_with_kernel(&input, kernel)?;

        assert_eq!(result.hott.len(), data.len());
        assert_eq!(result.lott.len(), data.len());

        let valid_count = result
            .hott
            .iter()
            .skip(250)
            .filter(|&&x| x.is_finite())
            .count();
        assert!(
            valid_count > 0,
            "[{}] Should produce some valid values despite NaNs",
            test_name
        );

        Ok(())
    }

    fn check_otto_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = OttoParams::default();

        let input = OttoInput::from_candles(&candles, "close", params.clone());
        let batch_output = otto_with_kernel(&input, kernel)?;

        let mut stream = OttoStream::try_new(params)?;
        let mut stream_hott = Vec::new();
        let mut stream_lott = Vec::new();

        for &value in &candles.close {
            match stream.update(value) {
                Some((h, l)) => {
                    stream_hott.push(h);
                    stream_lott.push(l);
                }
                None => {
                    stream_hott.push(f64::NAN);
                    stream_lott.push(f64::NAN);
                }
            }
        }

        let start = stream_hott.len() - 10;
        for i in start..stream_hott.len() {
            if stream_hott[i].is_finite() && batch_output.hott[i].is_finite() {
                let diff = (stream_hott[i] - batch_output.hott[i]).abs();

                assert!(
                    diff < 0.2,
                    "[{}] Stream HOTT mismatch at {}: {} vs {} (diff: {})",
                    test_name,
                    i,
                    stream_hott[i],
                    batch_output.hott[i],
                    diff
                );
            }
        }

        Ok(())
    }

    fn check_otto_builder(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let output = OttoBuilder::new()
            .ott_period(3)
            .ott_percent(0.8)
            .fast_vidya_length(12)
            .slow_vidya_length(30)
            .correcting_constant(50000.0)
            .ma_type("EMA")
            .kernel(kernel)
            .apply(&candles)?;

        assert_eq!(output.hott.len(), candles.close.len());
        assert_eq!(output.lott.len(), candles.close.len());

        Ok(())
    }

    macro_rules! generate_all_otto_tests {
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

    generate_all_otto_tests!(
        check_otto_partial_params,
        check_otto_accuracy,
        check_otto_default_candles,
        check_otto_zero_period,
        check_otto_period_exceeds_length,
        check_otto_very_small_dataset,
        check_otto_empty_input,
        check_otto_invalid_ma_type,
        check_otto_all_ma_types,
        check_otto_reinput,
        check_otto_nan_handling,
        check_otto_streaming,
        check_otto_builder
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let output = OttoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles, "close")?;

        let def = OttoParams::default();
        let default_idx = output
            .combos
            .iter()
            .position(|c| {
                c.ott_period == def.ott_period
                    && c.ott_percent == def.ott_percent
                    && c.fast_vidya_length == def.fast_vidya_length
                    && c.slow_vidya_length == def.slow_vidya_length
                    && c.correcting_constant == def.correcting_constant
                    && c.ma_type == def.ma_type
            })
            .expect("default params not found in batch output");

        let hott_row = &output.hott[default_idx * output.cols..(default_idx + 1) * output.cols];
        let lott_row = &output.lott[default_idx * output.cols..(default_idx + 1) * output.cols];

        assert_eq!(hott_row.len(), candles.close.len());
        assert_eq!(lott_row.len(), candles.close.len());

        let non_nan_hott = hott_row.iter().filter(|&&x| !x.is_nan()).count();
        let non_nan_lott = lott_row.iter().filter(|&&x| !x.is_nan()).count();
        assert!(
            non_nan_hott > 0,
            "[{}] Expected some non-NaN HOTT values",
            test
        );
        assert!(
            non_nan_lott > 0,
            "[{}] Expected some non-NaN LOTT values",
            test
        );

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let output = OttoBatchBuilder::new()
            .kernel(kernel)
            .ott_period_range(2, 4, 1)
            .ott_percent_range(0.5, 0.7, 0.1)
            .fast_vidya_range(10, 12, 1)
            .slow_vidya_range(20, 22, 1)
            .correcting_constant_range(100000.0, 100000.0, 0.0)
            .ma_types(vec!["VAR".into(), "EMA".into()])
            .apply_candles(&candles, "close")?;

        let expected_combos = 3 * 3 * 3 * 3 * 1 * 2;
        assert_eq!(
            output.combos.len(),
            expected_combos,
            "[{}] Expected {} combos",
            test,
            expected_combos
        );
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, candles.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_no_poison_single(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use crate::utilities::data_loader::read_candles_from_vortex;
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out = OttoBuilder::new().kernel(kernel).apply(&c)?;
        for &v in out.hott.iter().chain(out.lott.iter()) {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x1111_1111_1111_1111,
                "[{test}] alloc_with_nan_prefix poison seen"
            );
            assert_ne!(
                b, 0x2222_2222_2222_2222,
                "[{test}] init_matrix_prefixes poison seen"
            );
            assert_ne!(
                b, 0x3333_3333_3333_3333,
                "[{test}] make_uninit_matrix poison seen"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_no_poison_batch(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = (0..300)
            .map(|i| (i as f64).cos() * 2.0 + 10.0)
            .collect::<Vec<_>>();
        let out = OttoBatchBuilder::new().kernel(kernel).apply_slice(&data)?;
        for &v in out.hott.iter().chain(out.lott.iter()) {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x1111_1111_1111_1111,
                "[{}] alloc_with_nan_prefix poison seen",
                test
            );
            assert_ne!(
                b, 0x2222_2222_2222_2222,
                "[{}] init_matrix_prefixes poison seen",
                test
            );
            assert_ne!(
                b, 0x3333_3333_3333_3333,
                "[{}] make_uninit_matrix poison seen",
                test
            );
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
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
    gen_batch_tests!(check_batch_sweep);
    #[cfg(debug_assertions)]
    gen_batch_tests!(check_no_poison_batch);

    #[cfg(debug_assertions)]
    generate_all_otto_tests!(check_no_poison_single);

    #[test]
    fn test_otto_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 512usize;
        let data: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.013).sin() * 0.5 + 1.0)
            .collect();

        let input = super::OttoInput::from_slice(&data, super::OttoParams::default());

        let baseline = super::otto(&input)?;

        let mut hott_out = vec![0.0f64; n];
        let mut lott_out = vec![0.0f64; n];

        super::otto_into(&input, &mut hott_out, &mut lott_out)?;

        assert_eq!(baseline.hott.len(), n);
        assert_eq!(baseline.lott.len(), n);
        assert_eq!(hott_out.len(), n);
        assert_eq!(lott_out.len(), n);

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline.hott[i], hott_out[i]),
                "HOTT mismatch at {i}: got {}, expected {}",
                hott_out[i],
                baseline.hott[i]
            );
            assert!(
                eq_or_both_nan(baseline.lott[i], lott_out[i]),
                "LOTT mismatch at {i}: got {}, expected {}",
                lott_out[i],
                baseline.lott[i]
            );
        }

        Ok(())
    }
}
