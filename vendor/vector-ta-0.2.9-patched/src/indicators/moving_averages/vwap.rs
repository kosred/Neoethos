use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use chrono::{Datelike, NaiveDate, NaiveDateTime};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VwapData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    CandlesPlusPrices {
        candles: &'a Candles,
        prices: &'a [f64],
    },
    Slice {
        timestamps: &'a [i64],
        volumes: &'a [f64],
        prices: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VwapParams {
    pub anchor: Option<String>,
}

impl Default for VwapParams {
    fn default() -> Self {
        Self {
            anchor: Some("1d".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwapInput<'a> {
    pub data: VwapData<'a>,
    pub params: VwapParams,
}

impl<'a> VwapInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VwapParams) -> Self {
        Self {
            data: VwapData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_candles_plus_prices(
        candles: &'a Candles,
        prices: &'a [f64],
        params: VwapParams,
    ) -> Self {
        Self {
            data: VwapData::CandlesPlusPrices { candles, prices },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        timestamps: &'a [i64],
        volumes: &'a [f64],
        prices: &'a [f64],
        params: VwapParams,
    ) -> Self {
        Self {
            data: VwapData::Slice {
                timestamps,
                volumes,
                prices,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VwapData::Candles {
                candles,
                source: "hlc3",
            },
            params: VwapParams::default(),
        }
    }

    #[inline]
    pub fn get_anchor(&self) -> &str {
        self.params.anchor.as_deref().unwrap_or("1d")
    }
}

#[derive(Debug, Clone)]
pub struct VwapOutput {
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct VwapBuilder {
    anchor: Option<String>,
    kernel: Kernel,
}

impl Default for VwapBuilder {
    fn default() -> Self {
        Self {
            anchor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwapBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn anchor(mut self, s: impl Into<String>) -> Self {
        self.anchor = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<VwapOutput, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        let input = VwapInput::from_candles(candles, "hlc3", params);
        vwap_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
    ) -> Result<VwapOutput, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        let input = VwapInput::from_slice(timestamps, volumes, prices, params);
        vwap_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VwapStream, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        VwapStream::try_new(params)
    }

    #[inline(always)]
    pub fn apply_candles_with_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<VwapOutput, VwapError> {
        let params = VwapParams {
            anchor: self.anchor,
        };
        let input = VwapInput::from_candles(candles, source, params);
        vwap_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &Candles) -> Result<VwapOutput, VwapError> {
        VwapBuilder::new().apply_candles_with_source(candles, "hlc3")
    }
}

#[derive(Debug, Error)]
pub enum VwapError {
    #[error("vwap: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "vwap: Mismatch in length of timestamps ({timestamps}), prices ({prices}), or volumes ({volumes})."
    )]
    MismatchTimestampsPricesVolumes {
        timestamps: usize,
        prices: usize,
        volumes: usize,
    },
    #[error("vwap: No data for VWAP calculation.")]
    NoData,
    #[error("vwap: Mismatch in length of prices ({prices}) and volumes ({volumes}).")]
    MismatchPricesVolumes { prices: usize, volumes: usize },
    #[error("vwap: Error parsing anchor: {msg}")]
    ParseAnchorError { msg: String },
    #[error("vwap: Unsupported anchor unit '{unit_char}'.")]
    UnsupportedAnchorUnit { unit_char: char },
    #[error("vwap: Error converting timestamp {ts_ms} to month-based anchor.")]
    MonthConversionError { ts_ms: i64 },

    #[error("vwap: Output length mismatch (expected {expected}, got {got}).")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vwap: Invalid kernel for batch path: {0:?}.")]
    InvalidKernelForBatch(Kernel),
    #[error("vwap: Invalid range expansion (start='{start}', end='{end}', step={step}).")]
    InvalidRange {
        start: String,
        end: String,
        step: u32,
    },
    #[error("vwap: Not enough valid data (needed {needed}, have {valid}).")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vwap: All input values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn vwap(input: &VwapInput) -> Result<VwapOutput, VwapError> {
    vwap_with_kernel(input, Kernel::Auto)
}

pub fn vwap_with_kernel(input: &VwapInput, kernel: Kernel) -> Result<VwapOutput, VwapError> {
    let (timestamps, volumes, prices) = match &input.data {
        VwapData::Candles { candles, source } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let prices: &[f64] = source_type(candles, source);
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, prices)
        }
        VwapData::CandlesPlusPrices { candles, prices } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, *prices)
        }
        VwapData::Slice {
            timestamps,
            volumes,
            prices,
        } => (*timestamps, *volumes, *prices),
    };

    let n = prices.len();
    if timestamps.len() != n || volumes.len() != n {
        return Err(VwapError::MismatchTimestampsPricesVolumes {
            timestamps: timestamps.len(),
            prices: n,
            volumes: volumes.len(),
        });
    }
    if n == 0 {
        return Err(VwapError::NoData);
    }

    let (count, unit_char) = parse_anchor(input.get_anchor())
        .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut values = alloc_with_nan_prefix(n, 0);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vwap_scalar(timestamps, volumes, prices, count, unit_char, &mut values)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vwap_avx2(timestamps, volumes, prices, count, unit_char, &mut values)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vwap_avx512(timestamps, volumes, prices, count, unit_char, &mut values)?
            }
            _ => unreachable!(),
        }
    }

    Ok(VwapOutput { values })
}

#[inline]
pub fn vwap_into_slice(dst: &mut [f64], input: &VwapInput, kern: Kernel) -> Result<(), VwapError> {
    let (timestamps, volumes, prices) = match &input.data {
        VwapData::Candles { candles, source } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let prices: &[f64] = source_type(candles, source);
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, prices)
        }
        VwapData::CandlesPlusPrices { candles, prices } => {
            let timestamps: &[i64] = candles
                .get_timestamp()
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            let vols: &[f64] = candles
                .select_candle_field("volume")
                .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
            (timestamps, vols, *prices)
        }
        VwapData::Slice {
            timestamps,
            volumes,
            prices,
        } => (*timestamps, *volumes, *prices),
    };

    let n = prices.len();
    if dst.len() != n {
        return Err(VwapError::OutputLengthMismatch {
            expected: n,
            got: dst.len(),
        });
    }
    if timestamps.len() != n || volumes.len() != n {
        return Err(VwapError::MismatchTimestampsPricesVolumes {
            timestamps: timestamps.len(),
            prices: n,
            volumes: volumes.len(),
        });
    }
    if n == 0 {
        return Err(VwapError::NoData);
    }

    let (count, unit_char) = parse_anchor(input.get_anchor())
        .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vwap_scalar(timestamps, volumes, prices, count, unit_char, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vwap_avx2(timestamps, volumes, prices, count, unit_char, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vwap_avx512(timestamps, volumes, prices, count, unit_char, dst)?
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn vwap_into(input: &VwapInput, out: &mut [f64]) -> Result<(), VwapError> {
    vwap_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
pub fn vwap_scalar(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) -> Result<(), VwapError> {
    debug_assert_eq!(out.len(), prices.len(), "output slice length mismatch");

    let mut current_group_id: i64 = i64::MIN;
    let mut volume_sum: f64 = 0.0;
    let mut vol_price_sum: f64 = 0.0;

    let n = prices.len();
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let ts_ptr = timestamps.as_ptr();
        let vol_ptr = volumes.as_ptr();
        let pr_ptr = prices.as_ptr();
        let out_ptr = out.as_mut_ptr();

        if unit_char == 'm' || unit_char == 'h' || unit_char == 'd' {
            const MINUTE_MS: i64 = 60_000;
            const HOUR_MS: i64 = 3_600_000;
            const DAY_MS: i64 = 86_400_000;

            let unit_ms: i64 = match unit_char {
                'm' => MINUTE_MS,
                'h' => HOUR_MS,
                _ => DAY_MS,
            };
            let bucket_ms: i64 = (count as i64) * unit_ms;

            let mut i: usize = 0;
            let unroll_end = n & !1usize;
            let mut window_start: i64 = 0;
            let mut next_cutoff: i64 = i64::MIN;

            while i < unroll_end {
                let ts0 = *ts_ptr.add(i);
                if ts0 >= 0 {
                    if ts0 >= next_cutoff || ts0 < window_start {
                        let gid0 = ts0 / bucket_ms;
                        current_group_id = gid0;
                        window_start = gid0.saturating_mul(bucket_ms);
                        next_cutoff = window_start.saturating_add(bucket_ms);
                        volume_sum = 0.0;
                        vol_price_sum = 0.0;
                    }
                } else {
                    let gid0 = ts0 / bucket_ms;
                    if gid0 != current_group_id {
                        current_group_id = gid0;
                        window_start = gid0.saturating_mul(bucket_ms);
                        next_cutoff = window_start.saturating_add(bucket_ms);
                        volume_sum = 0.0;
                        vol_price_sum = 0.0;
                    }
                }
                let v0 = *vol_ptr.add(i);
                let p0 = *pr_ptr.add(i);
                volume_sum += v0;
                vol_price_sum = p0.mul_add(v0, vol_price_sum);
                *out_ptr.add(i) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };

                let idx1 = i + 1;
                let ts1 = *ts_ptr.add(idx1);
                if ts1 >= 0 {
                    if ts1 >= next_cutoff || ts1 < window_start {
                        let gid1 = ts1 / bucket_ms;
                        current_group_id = gid1;
                        window_start = gid1.saturating_mul(bucket_ms);
                        next_cutoff = window_start.saturating_add(bucket_ms);
                        volume_sum = 0.0;
                        vol_price_sum = 0.0;
                    }
                } else {
                    let gid1 = ts1 / bucket_ms;
                    if gid1 != current_group_id {
                        current_group_id = gid1;
                        window_start = gid1.saturating_mul(bucket_ms);
                        next_cutoff = window_start.saturating_add(bucket_ms);
                        volume_sum = 0.0;
                        vol_price_sum = 0.0;
                    }
                }
                let v1 = *vol_ptr.add(idx1);
                let p1 = *pr_ptr.add(idx1);
                volume_sum += v1;
                vol_price_sum = p1.mul_add(v1, vol_price_sum);
                *out_ptr.add(idx1) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };

                i += 2;
            }

            if unroll_end != n {
                let ts = *ts_ptr.add(unroll_end);
                if ts >= 0 {
                    if ts >= next_cutoff || ts < window_start {
                        let gid = ts / bucket_ms;
                        current_group_id = gid;
                        window_start = gid.saturating_mul(bucket_ms);
                        next_cutoff = window_start.saturating_add(bucket_ms);
                        volume_sum = 0.0;
                        vol_price_sum = 0.0;
                    }
                } else {
                    let gid = ts / bucket_ms;
                    if gid != current_group_id {
                        current_group_id = gid;
                        window_start = gid.saturating_mul(bucket_ms);
                        next_cutoff = window_start.saturating_add(bucket_ms);
                        volume_sum = 0.0;
                        vol_price_sum = 0.0;
                    }
                }
                let v = *vol_ptr.add(unroll_end);
                let p = *pr_ptr.add(unroll_end);
                volume_sum += v;
                vol_price_sum = p.mul_add(v, vol_price_sum);
                *out_ptr.add(unroll_end) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };
            }

            return Ok(());
        }

        if unit_char == 'M' {
            let mut i: usize = 0;
            while i < n {
                let ts = *ts_ptr.add(i);
                let gid = match floor_to_month(ts, count) {
                    Ok(g) => g,
                    Err(_) => return Err(VwapError::MonthConversionError { ts_ms: ts }),
                };
                if gid != current_group_id {
                    current_group_id = gid;
                    volume_sum = 0.0;
                    vol_price_sum = 0.0;
                }
                let v = *vol_ptr.add(i);
                let p = *pr_ptr.add(i);
                volume_sum += v;
                vol_price_sum = p.mul_add(v, vol_price_sum);
                *out_ptr.add(i) = if volume_sum > 0.0 {
                    vol_price_sum / volume_sum
                } else {
                    f64::NAN
                };
                i += 1;
            }
            return Ok(());
        }
    }

    Err(VwapError::UnsupportedAnchorUnit { unit_char })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_avx2(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) -> Result<(), VwapError> {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_avx512(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) -> Result<(), VwapError> {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out)
}

#[derive(Debug, Clone)]
pub struct VwapStream {
    anchor: String,
    count: u32,
    unit_char: char,

    bucket_ms: i64,
    next_cutoff: i64,
    current_group_id: i64,

    volume_sum: f64,
    vol_price_sum: f64,
}

impl VwapStream {
    #[inline]
    pub fn try_new(params: VwapParams) -> Result<Self, VwapError> {
        let anchor = params.anchor.unwrap_or_else(|| "1d".to_string());
        let (count, unit_char) = parse_anchor(&anchor)
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;

        const MINUTE_MS: i64 = 60_000;
        const HOUR_MS: i64 = 3_600_000;
        const DAY_MS: i64 = 86_400_000;

        let bucket_ms = match unit_char {
            'm' => (count as i64).saturating_mul(MINUTE_MS),
            'h' => (count as i64).saturating_mul(HOUR_MS),
            'd' => (count as i64).saturating_mul(DAY_MS),
            'M' => 0,
            _ => return Err(VwapError::UnsupportedAnchorUnit { unit_char }),
        };

        Ok(Self {
            anchor,
            count,
            unit_char,
            bucket_ms,
            next_cutoff: i64::MIN,
            current_group_id: i64::MIN,
            volume_sum: 0.0,
            vol_price_sum: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, timestamp: i64, price: f64, volume: f64) -> Option<f64> {
        match self.unit_char {
            'm' | 'h' | 'd' => self.update_linear(timestamp, price, volume),
            'M' => self.update_month(timestamp, price, volume),
            _ => None,
        }
    }

    #[inline(always)]
    fn init_linear(&mut self, ts: i64) {
        let gid = ts / self.bucket_ms;
        self.current_group_id = gid;

        self.next_cutoff = gid.saturating_add(1).saturating_mul(self.bucket_ms);
        self.volume_sum = 0.0;
        self.vol_price_sum = 0.0;
    }

    #[inline(always)]
    fn roll_linear_to(&mut self, ts: i64) {
        let delta = ts.saturating_sub(self.next_cutoff);
        let k = (delta / self.bucket_ms).saturating_add(1);
        self.current_group_id = self.current_group_id.saturating_add(k);
        self.next_cutoff = self
            .next_cutoff
            .saturating_add(self.bucket_ms.saturating_mul(k));
        self.volume_sum = 0.0;
        self.vol_price_sum = 0.0;
    }

    #[inline(always)]
    fn update_linear(&mut self, ts: i64, price: f64, volume: f64) -> Option<f64> {
        debug_assert!(self.bucket_ms > 0);
        if self.current_group_id == i64::MIN {
            self.init_linear(ts);
        } else if ts >= self.next_cutoff {
            self.roll_linear_to(ts);
        }

        self.volume_sum += volume;
        self.vol_price_sum = price.mul_add(volume, self.vol_price_sum);

        if self.volume_sum > 0.0 {
            Some(self.vol_price_sum / self.volume_sum)
        } else {
            None
        }
    }

    #[inline]
    fn month_gid_and_next_cutoff(&self, ts_ms: i64) -> Result<(i64, i64), VwapError> {
        let seconds = ts_ms / 1000;
        let nanos = ((ts_ms % 1000) * 1_000_000) as u32;
        let dt = NaiveDateTime::from_timestamp_opt(seconds, nanos)
            .ok_or_else(|| VwapError::MonthConversionError { ts_ms })?;

        let year = dt.year();
        let month = dt.month() as i32;
        let total_months = (year - 1970) as i64 * 12 + (month - 1) as i64;
        let gid = total_months / (self.count as i64);

        let next_bucket_months = gid.saturating_add(1).saturating_mul(self.count as i64);

        let next_year = 1970 + next_bucket_months.div_euclid(12);
        let next_month0 = next_bucket_months.rem_euclid(12);
        let next_date = NaiveDate::from_ymd_opt(next_year as i32, (next_month0 + 1) as u32, 1)
            .ok_or_else(|| VwapError::MonthConversionError { ts_ms })?;
        let next_dt = next_date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| VwapError::MonthConversionError { ts_ms })?;

        let next_ms = next_dt.timestamp().saturating_mul(1000);

        Ok((gid, next_ms))
    }

    #[inline(always)]
    fn init_month(&mut self, ts: i64) -> Option<()> {
        match self.month_gid_and_next_cutoff(ts) {
            Ok((gid, next_ms)) => {
                self.current_group_id = gid;
                self.next_cutoff = next_ms;
                self.volume_sum = 0.0;
                self.vol_price_sum = 0.0;
                Some(())
            }
            Err(_) => None,
        }
    }

    #[inline(always)]
    fn update_month(&mut self, ts: i64, price: f64, volume: f64) -> Option<f64> {
        if self.current_group_id == i64::MIN {
            if self.init_month(ts).is_none() {
                return None;
            }
        } else if ts >= self.next_cutoff {
            match self.month_gid_and_next_cutoff(ts) {
                Ok((gid, next_ms)) => {
                    self.current_group_id = gid;
                    self.next_cutoff = next_ms;
                    self.volume_sum = 0.0;
                    self.vol_price_sum = 0.0;
                }
                Err(_) => return None,
            }
        }

        self.volume_sum += volume;
        self.vol_price_sum = price.mul_add(volume, self.vol_price_sum);

        if self.volume_sum > 0.0 {
            Some(self.vol_price_sum / self.volume_sum)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct VwapBatchRange {
    pub anchor: (String, String, u32),
}

impl Default for VwapBatchRange {
    fn default() -> Self {
        Self {
            anchor: ("1d".to_string(), "250d".to_string(), 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VwapBatchBuilder {
    range: VwapBatchRange,
    kernel: Kernel,
}

impl VwapBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn anchor_range(
        mut self,
        start: impl Into<String>,
        end: impl Into<String>,
        step: u32,
    ) -> Self {
        self.range.anchor = (start.into(), end.into(), step);
        self
    }
    #[inline]
    pub fn anchor_static(mut self, val: impl Into<String>) -> Self {
        let s = val.into();
        self.range.anchor = (s.clone(), s, 0);
        self
    }
    pub fn apply_slice(
        self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
    ) -> Result<VwapBatchOutput, VwapError> {
        vwap_batch_with_kernel(timestamps, volumes, prices, &self.range, self.kernel)
    }
}

pub fn vwap_batch_with_kernel(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    k: Kernel,
) -> Result<VwapBatchOutput, VwapError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VwapError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    vwap_batch_inner(timestamps, volumes, prices, sweep, simd, true)
}

#[derive(Clone, Debug)]
pub struct VwapBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VwapParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VwapBatchOutput {
    pub fn row_for_anchor(&self, a: &str) -> Option<usize> {
        self.combos
            .iter()
            .position(|p| p.anchor.as_deref() == Some(a))
    }

    pub fn values_for(&self, a: &str) -> Option<&[f64]> {
        self.row_for_anchor(a).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub(crate) fn expand_grid_vwap(r: &VwapBatchRange) -> Vec<VwapParams> {
    if r.anchor.2 == 0 || r.anchor.0 == r.anchor.1 {
        return vec![VwapParams {
            anchor: Some(r.anchor.0.clone()),
        }];
    }

    let step = r.anchor.2.max(1);
    let start = anchor_to_num_and_unit(&r.anchor.0);
    let end = anchor_to_num_and_unit(&r.anchor.1);
    let mut result = Vec::new();
    if let (Some((mut a, unit_a)), Some((b, unit_b))) = (start, end) {
        if unit_a != unit_b {
            return result;
        }
        if a <= b {
            while a <= b {
                result.push(VwapParams {
                    anchor: Some(format!("{}{}", a, unit_a)),
                });
                a = a.saturating_add(step);
            }
        } else {
            while a >= b {
                result.push(VwapParams {
                    anchor: Some(format!("{}{}", a, unit_a)),
                });
                if a < step {
                    break;
                }
                a -= step;
            }
        }
    }
    result
}

fn anchor_to_num_and_unit(anchor: &str) -> Option<(u32, char)> {
    let mut idx = 0;
    for (pos, ch) in anchor.char_indices() {
        if !ch.is_ascii_digit() {
            idx = pos;
            break;
        }
    }
    if idx == 0 {
        return None;
    }
    let num = anchor[..idx].parse::<u32>().ok()?;
    let unit = anchor[idx..].chars().next()?;
    Some((num, unit))
}

#[inline]
pub(crate) fn first_valid_vwap_index(
    timestamps: &[i64],
    volumes: &[f64],
    count: u32,
    unit: char,
) -> usize {
    if timestamps.is_empty() {
        return 0;
    }
    let mut cur_gid = i64::MIN;
    let mut vsum = 0.0;
    for i in 0..timestamps.len() {
        let ts = timestamps[i];
        let gid = match unit {
            'm' => ts / ((count as i64) * 60_000),
            'h' => ts / ((count as i64) * 3_600_000),
            'd' => ts / ((count as i64) * 86_400_000),
            'M' => floor_to_month(ts, count).unwrap_or(i64::MIN),
            _ => i64::MIN,
        };
        if gid != cur_gid {
            cur_gid = gid;
            vsum = 0.0;
        }
        vsum += volumes[i];
        if vsum > 0.0 {
            return i;
        }
    }
    0
}

#[inline(always)]
pub fn vwap_batch_slice(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
) -> Result<VwapBatchOutput, VwapError> {
    vwap_batch_inner(timestamps, volumes, prices, sweep, kern, false)
}

#[inline(always)]
pub fn vwap_batch_par_slice(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
) -> Result<VwapBatchOutput, VwapError> {
    vwap_batch_inner(timestamps, volumes, prices, sweep, kern, true)
}

fn vwap_batch_inner(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VwapBatchOutput, VwapError> {
    let combos = expand_grid_vwap(sweep);
    if combos.is_empty() {
        return Err(VwapError::InvalidRange {
            start: sweep.anchor.0.clone(),
            end: sweep.anchor.1.clone(),
            step: sweep.anchor.2,
        });
    }

    let rows = combos.len();
    let cols = prices.len();

    let mut raw = make_uninit_matrix(rows, cols);

    let mut warm: Vec<usize> = Vec::with_capacity(rows);
    for prm in &combos {
        let (cnt, unit) = parse_anchor(prm.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })?;
        let w = first_valid_vwap_index(timestamps, volumes, cnt, unit);
        warm.push(w);
    }
    init_matrix_prefixes(&mut raw, cols, &warm);

    // The `pv` precompute that used to live here is gone with
    // `vwap_row_scalar_pv`: it existed only to feed that arm, and it was the
    // extra rounding. See the `Kernel::Scalar` arm below.

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let params = combos.get(row).unwrap();
        let (count, unit_char) = parse_anchor(params.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })
            .unwrap();

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            // WAS `vwap_row_scalar_pv(.., &pv, ..)`. That path precomputed
            // `pv[i] = p * v` (one rounding) and then did `vol_price_sum += pv`
            // (a second), where `vwap_scalar` -- which EVERY other arm of this
            // match reaches, and which the single-series public `vwap()` uses --
            // does `vol_price_sum = p.mul_add(v, vol_price_sum)`, ONE rounding.
            // So `Kernel::Scalar` disagreed with `Kernel::Avx2` by 1 ULP, and
            // both disagreed with `vwap()`, inside one crate. That is the whole
            // of the `vwap` entry in `WITHHELD_PENDING_CPU_SELF_CONSISTENCY`:
            // there was no single CPU answer for a device result to be in
            // parity with.
            //
            // Fixed in the CPU, not worked around. The mul_add form wins on
            // three independent counts: it has FEWER roundings, it is what four
            // of the five arms here already ran, and it is what the public
            // single-series entry point produces -- so it is the number every
            // caller outside this one batch arm already saw.
            Kernel::Scalar => {
                vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwap_row_avx2(timestamps, volumes, prices, count, unit_char, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                vwap_row_avx512(timestamps, volumes, prices, count, unit_char, out_row)
            }
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

    let mut raw_guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            raw_guard.as_mut_ptr() as *mut f64,
            raw_guard.len(),
            raw_guard.capacity(),
        )
    };

    Ok(VwapBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn vwap_batch_inner_into(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    sweep: &VwapBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VwapParams>, VwapError> {
    if timestamps.len() != volumes.len() || volumes.len() != prices.len() {
        return Err(VwapError::MismatchTimestampsPricesVolumes {
            timestamps: timestamps.len(),
            prices: prices.len(),
            volumes: volumes.len(),
        });
    }
    let combos = expand_grid_vwap(sweep);
    if combos.is_empty() {
        return Err(VwapError::InvalidRange {
            start: sweep.anchor.0.clone(),
            end: sweep.anchor.1.clone(),
            step: sweep.anchor.2,
        });
    }

    let rows = combos.len();
    let cols = prices.len();

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| VwapError::InvalidRange {
            start: sweep.anchor.0.clone(),
            end: sweep.anchor.1.clone(),
            step: sweep.anchor.2,
        })?;
    if out.len() != expected {
        return Err(VwapError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // No `pv` precompute here either -- same reason as `vwap_batch_inner`:
    // it was the second rounding that made `Kernel::Scalar` disagree with
    // `Kernel::Avx2` and with `vwap()` by 1 ULP.

    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        let params = combos.get(row).unwrap();
        let (count, unit_char) = parse_anchor(params.anchor.as_deref().unwrap_or("1d"))
            .map_err(|e| VwapError::ParseAnchorError { msg: e.to_string() })
            .unwrap();

        let out_row = dst;

        match kern {
            Kernel::Scalar => {
                vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwap_row_avx2(timestamps, volumes, prices, count, unit_char, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                vwap_row_avx512(timestamps, volumes, prices, count, unit_char, out_row)
            }
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

    Ok(combos)
}

#[inline]
pub(crate) fn parse_anchor(anchor: &str) -> Result<(u32, char), Box<dyn std::error::Error>> {
    let mut idx = 0;
    for (pos, ch) in anchor.char_indices() {
        if !ch.is_ascii_digit() {
            idx = pos;
            break;
        }
    }
    if idx == 0 {
        return Err(format!("No numeric portion found in anchor '{}'", anchor).into());
    }
    let num_part = &anchor[..idx];
    let unit_part = &anchor[idx..];
    let count = num_part
        .parse::<u32>()
        .map_err(|_| format!("Failed parsing numeric portion '{}'", num_part))?;
    if unit_part.len() != 1 {
        return Err(format!("Anchor unit must be 1 char (found '{}')", unit_part).into());
    }
    let mut unit_char = unit_part.chars().next().unwrap();
    unit_char = match unit_char {
        'H' => 'h',
        'D' => 'd',
        c => c,
    };
    match unit_char {
        'm' | 'h' | 'd' | 'M' => Ok((count, unit_char)),
        _ => Err(format!("Unsupported unit '{}'", unit_char).into()),
    }
}

#[inline]
pub(crate) fn floor_to_month(ts_ms: i64, count: u32) -> Result<i64, Box<dyn Error>> {
    let seconds = ts_ms / 1000;
    let nanos = ((ts_ms % 1000) * 1_000_000) as u32;

    let dt = NaiveDateTime::from_timestamp_opt(seconds, nanos)
        .ok_or_else(|| format!("Invalid timestamp: {}", ts_ms))?;

    let year = dt.year();
    let month = dt.month() as i32;
    let total_months = (year - 1970) as i64 * 12 + (month - 1) as i64;

    if count == 1 {
        Ok(total_months)
    } else {
        Ok(total_months / (count as i64))
    }
}

#[inline(always)]
pub unsafe fn vwap_row_scalar(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_scalar(timestamps, volumes, prices, count, unit_char, out);
}

// `vwap_row_scalar_pv` WAS HERE AND IS DELETED.
//
// It was a second implementation of vwap that differed from `vwap_scalar` by
// one rounding: it consumed a precomputed `pv[i] = price * volume` and did
// `vol_price_sum += pv[i]`, where `vwap_scalar` does
// `vol_price_sum = p.mul_add(v, vol_price_sum)`. Only the `Kernel::Scalar` arm
// of the two batch paths ever reached it, so the crate answered the same
// question two ways depending on which kernel the caller asked for -- and the
// f64 CUDA lane could not be in parity with either, which is why `vwap` sat in
// `WITHHELD_PENDING_CPU_SELF_CONSISTENCY`. Deleting it, rather than teaching
// the device to reproduce it, is the fix: the crate now has ONE vwap
// recurrence and `neoethos_vwap_batch_f64` is in parity with it.

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx2(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx512(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx512_short(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwap_row_avx512_long(
    timestamps: &[i64],
    volumes: &[f64],
    prices: &[f64],
    count: u32,
    unit_char: char,
    out: &mut [f64],
) {
    vwap_row_scalar(timestamps, volumes, prices, count, unit_char, out);
}

#[inline(always)]
fn expand_grid(r: &VwapBatchRange) -> Vec<VwapParams> {
    expand_grid_vwap(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_vwap_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params_default = VwapParams { anchor: None };
        let input_default = VwapInput::from_candles(&candles, "hlc3", params_default);
        let output_default = vwap_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_vwap_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let timestamps = candles.get_timestamp().map_err(|e| e.to_string())?;
        let volumes = candles
            .select_candle_field("volume")
            .map_err(|e| e.to_string())?;
        let prices = source_type(&candles, "hlc3");

        let take = 512usize.min(prices.len());
        let start = prices.len().saturating_sub(take);
        let ts_slice = &timestamps[start..start + take];
        let vol_slice = &volumes[start..start + take];
        let price_slice = &prices[start..start + take];

        let params = VwapParams { anchor: None };
        let input = VwapInput::from_slice(ts_slice, vol_slice, price_slice, params);

        let baseline = vwap(&input)?.values;

        let mut out = vec![0.0; price_slice.len()];
        vwap_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_vwap_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five_vwap = [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0,
        ];
        let file_path: &str = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = VwapParams {
            anchor: Some("1D".to_string()),
        };
        let input = VwapInput::from_candles(&candles, "hlc3", params);
        let result = vwap_with_kernel(&input, kernel)?;
        assert!(result.values.len() >= 5, "Not enough data points for test");
        let start_idx = result.values.len() - 5;
        for (i, &vwap_val) in result.values[start_idx..].iter().enumerate() {
            let exp_val = expected_last_five_vwap[i];
            assert!(
                (vwap_val - exp_val).abs() < 1e-5,
                "[{}] VWAP mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                exp_val,
                vwap_val
            );
        }
        Ok(())
    }

    fn check_vwap_candles_plus_prices(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let source_prices = candles.get_calculated_field("hl2").unwrap();
        let params = VwapParams {
            anchor: Some("1d".to_string()),
        };
        let input = VwapInput::from_candles_plus_prices(&candles, source_prices, params);
        let result = vwap_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vwap_anchor_parsing_error(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = VwapParams {
            anchor: Some("xyz".to_string()),
        };
        let input = VwapInput::from_candles(&candles, "hlc3", params);
        let result = vwap_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vwap_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = VwapParams {
            anchor: Some("1d".to_string()),
        };
        let first_input = VwapInput::from_candles(&candles, "close", first_params);
        let first_result = vwap_with_kernel(&first_input, kernel)?;
        let second_params = VwapParams {
            anchor: Some("1h".to_string()),
        };
        let source_prices = &first_result.values;
        let second_input =
            VwapInput::from_candles_plus_prices(&candles, source_prices, second_params);
        let second_result = vwap_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_vwap_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VwapInput::with_default_candles(&candles);
        let result = vwap_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        for &val in &result.values {
            if !val.is_nan() {
                assert!(val.is_finite());
            }
        }
        Ok(())
    }

    fn check_vwap_with_default_candles(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = VwapInput::with_default_candles(&candles);
        match input.data {
            VwapData::Candles { source, .. } => {
                assert_eq!(source, "hlc3");
            }
            _ => panic!("Expected VwapData::Candles"),
        }
        let anchor = input.get_anchor();
        assert_eq!(anchor, "1d");
        Ok(())
    }

    fn check_vwap_with_default_params(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let default_params = VwapParams::default();
        assert_eq!(default_params.anchor, Some("1d".to_string()));
        Ok(())
    }

    macro_rules! generate_all_vwap_tests {
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
    fn check_vwap_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_anchors = vec!["1m", "5m", "15m", "30m", "1h", "4h", "1d", "3d"];
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for anchor in test_anchors {
            for source in &test_sources {
                let params = VwapParams {
                    anchor: Some(anchor.to_string()),
                };
                let input = VwapInput::from_candles(&candles, source, params);
                let output = vwap_with_kernel(&input, kernel)?;

                for (i, &val) in output.values.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (anchor={}, source={})",
                            test_name, val, bits, i, anchor, source
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (anchor={}, source={})",
                            test_name, val, bits, i, anchor, source
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (anchor={}, source={})",
                            test_name, val, bits, i, anchor, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vwap_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_vwap_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let timestamps = candles.get_timestamp().unwrap();
        let volumes = candles.select_candle_field("volume").unwrap();
        let close_prices = &candles.close;
        let high_prices = &candles.high;
        let low_prices = &candles.low;

        let anchor_periods = vec!["30m", "1h", "4h", "12h", "1d", "2d", "3d"];

        let strat = (
            0usize..anchor_periods.len(),
            0usize..timestamps.len().saturating_sub(200),
            100usize..=200,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(anchor_idx, start_idx, slice_len)| {
                let end_idx = (start_idx + slice_len).min(timestamps.len());
                if end_idx <= start_idx || end_idx - start_idx < 10 {
                    return Ok(());
                }

                let anchor = anchor_periods[anchor_idx];
                let ts_slice = &timestamps[start_idx..end_idx];
                let vol_slice = &volumes[start_idx..end_idx];
                let price_slice = &close_prices[start_idx..end_idx];
                let high_slice = &high_prices[start_idx..end_idx];
                let low_slice = &low_prices[start_idx..end_idx];

                let params = VwapParams {
                    anchor: Some(anchor.to_string()),
                };
                let input = VwapInput::from_slice(ts_slice, vol_slice, price_slice, params.clone());

                let result = vwap_with_kernel(&input, kernel);

                let scalar_input =
                    VwapInput::from_slice(ts_slice, vol_slice, price_slice, params.clone());
                let scalar_result = vwap_with_kernel(&scalar_input, Kernel::Scalar);

                match (result, scalar_result) {
                    (Ok(VwapOutput { values: out }), Ok(VwapOutput { values: ref_out })) => {
                        prop_assert_eq!(out.len(), price_slice.len());
                        prop_assert_eq!(ref_out.len(), price_slice.len());

                        let (count, unit_char) = parse_anchor(anchor).unwrap();

                        for i in 0..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            if y.is_finite() && r.is_finite() {
                                let y_bits = y.to_bits();
                                let r_bits = r.to_bits();
                                let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                                prop_assert!(
                                    (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                    "Kernel mismatch at {}: {} vs {} (ULP={})",
                                    i,
                                    y,
                                    r,
                                    ulp_diff
                                );
                            } else if y.is_nan() && r.is_nan() {
                                continue;
                            } else {
                                prop_assert_eq!(
                                    y.is_nan(),
                                    r.is_nan(),
                                    "NaN mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                            }

                            if y.is_finite() && vol_slice[i] > 0.0 && price_slice[i] > 0.0 {
                                prop_assert!(
									y > 0.0,
									"VWAP should be positive at {} for positive price {} and volume {}",
									i, price_slice[i], vol_slice[i]
								);
                            }
                        }

                        let mut current_group_id = -1_i64;
                        let mut group_start = 0;

                        for i in 0..ts_slice.len() {
                            let ts_ms = ts_slice[i];
                            let group_id = match unit_char {
                                'm' => ts_ms / ((count as i64) * 60_000),
                                'h' => ts_ms / ((count as i64) * 3_600_000),
                                'd' => ts_ms / ((count as i64) * 86_400_000),
                                'M' => floor_to_month(ts_ms, count).unwrap_or(-1),
                                _ => -1,
                            };

                            if group_id != current_group_id {
                                if current_group_id != -1 && i > group_start {
                                    let mut vol_sum = 0.0;
                                    let mut vol_price_sum = 0.0;
                                    let mut min_price = f64::INFINITY;
                                    let mut max_price = f64::NEG_INFINITY;

                                    for j in group_start..i {
                                        vol_sum += vol_slice[j];
                                        vol_price_sum += vol_slice[j] * price_slice[j];
                                        min_price = min_price.min(low_slice[j]);
                                        max_price = max_price.max(high_slice[j]);
                                    }

                                    if vol_sum > 0.0 {
                                        let expected_vwap = vol_price_sum / vol_sum;
                                        let actual_vwap = out[i - 1];

                                        prop_assert!(
											(actual_vwap - expected_vwap).abs() < 1e-6,
											"VWAP formula mismatch at group ending at {}: {} vs expected {}",
											i - 1, actual_vwap, expected_vwap
										);

                                        prop_assert!(
                                            actual_vwap >= min_price - 1e-9
                                                && actual_vwap <= max_price + 1e-9,
                                            "VWAP {} outside price bounds [{}, {}] at {}",
                                            actual_vwap,
                                            min_price,
                                            max_price,
                                            i - 1
                                        );
                                    }
                                }

                                current_group_id = group_id;
                                group_start = i;
                            }
                        }

                        if price_slice.len() <= 50 {
                            let mut stream = VwapStream::try_new(params).unwrap();
                            let mut stream_values = Vec::with_capacity(price_slice.len());

                            for i in 0..price_slice.len() {
                                match stream.update(ts_slice[i], price_slice[i], vol_slice[i]) {
                                    Some(val) => stream_values.push(val),
                                    None => stream_values.push(f64::NAN),
                                }
                            }

                            for (i, (&batch_val, &stream_val)) in
                                out.iter().zip(stream_values.iter()).enumerate()
                            {
                                if batch_val.is_nan() && stream_val.is_nan() {
                                    continue;
                                }
                                if batch_val.is_finite() && stream_val.is_finite() {
                                    prop_assert!(
                                        (batch_val - stream_val).abs() < 1e-9,
                                        "Streaming mismatch at {}: batch={} vs stream={}",
                                        i,
                                        batch_val,
                                        stream_val
                                    );
                                }
                            }
                        }

                        {
                            let base_ts = 1609459200000_i64;
                            let test_ts = vec![base_ts, base_ts + 3600000, base_ts + 7200000];
                            let test_prices = vec![100.0, 200.0, 300.0];
                            let test_volumes = vec![1.0, 2.0, 3.0];

                            let test_params = VwapParams {
                                anchor: Some("1d".to_string()),
                            };
                            let test_input = VwapInput::from_slice(
                                &test_ts,
                                &test_volumes,
                                &test_prices,
                                test_params,
                            );

                            if let Ok(VwapOutput { values: test_out }) =
                                vwap_with_kernel(&test_input, kernel)
                            {
                                if test_out.len() >= 3 {
                                    if test_out[0].is_finite() {
                                        prop_assert!(
                                            (test_out[0] - 100.0).abs() < 1e-9,
                                            "VWAP at index 0 should be 100, got {}",
                                            test_out[0]
                                        );
                                    }
                                    if test_out[1].is_finite() {
                                        let expected_1 = 500.0 / 3.0;
                                        prop_assert!(
                                            (test_out[1] - expected_1).abs() < 1e-9,
                                            "VWAP at index 1 should be {}, got {}",
                                            expected_1,
                                            test_out[1]
                                        );
                                    }
                                    if test_out[2].is_finite() {
                                        let expected_2 = 1400.0 / 6.0;
                                        prop_assert!(
                                            (test_out[2] - expected_2).abs() < 1e-9,
                                            "VWAP at index 2 should be {}, got {}",
                                            expected_2,
                                            test_out[2]
                                        );
                                    }
                                }
                            }
                        }
                    }
                    (Err(e1), Err(e2)) => {
                        prop_assert_eq!(
                            std::mem::discriminant(&e1),
                            std::mem::discriminant(&e2),
                            "Different error types: {:?} vs {:?}",
                            e1,
                            e2
                        );
                    }
                    _ => {
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed"
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_vwap_tests!(
        check_vwap_partial_params,
        check_vwap_accuracy,
        check_vwap_candles_plus_prices,
        check_vwap_anchor_parsing_error,
        check_vwap_slice_data_reinput,
        check_vwap_nan_handling,
        check_vwap_with_default_candles,
        check_vwap_with_default_params,
        check_vwap_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vwap_tests!(check_vwap_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let timestamps = c.get_timestamp().unwrap();
        let prices = c.get_calculated_field("hlc3").unwrap();
        let volumes = c.select_candle_field("volume").unwrap();

        let output = VwapBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(timestamps, volumes, prices)?;

        let def = VwapParams::default();
        let row = output
            .combos
            .iter()
            .position(|p| p.anchor == def.anchor)
            .expect("default row missing");
        let row_values = &output.values[row * output.cols..(row + 1) * output.cols];

        assert_eq!(row_values.len(), c.close.len());

        let expected = [
            59353.05963230107,
            59330.15815713043,
            59289.94649532547,
            59274.6155462414,
            58730.0,
        ];
        let start = row_values.len() - 5;
        for (i, &v) in row_values[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    fn check_batch_anchor_grid(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let timestamps = c.get_timestamp().unwrap();
        let prices = c.get_calculated_field("hlc3").unwrap();
        let volumes = c.select_candle_field("volume").unwrap();

        let batch = VwapBatchBuilder::new()
            .kernel(kernel)
            .anchor_range("1d", "3d", 1)
            .apply_slice(timestamps, volumes, prices)?;

        assert_eq!(batch.cols, c.close.len());
        assert!(batch.rows >= 1 && batch.rows <= 3);

        let anchors: Vec<_> = batch
            .combos
            .iter()
            .map(|p| p.anchor.clone().unwrap())
            .collect();
        assert_eq!(
            anchors,
            vec!["1d".to_string(), "2d".to_string(), "3d".to_string()]
        );
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let timestamps = c.get_timestamp().unwrap();
        let volumes = c.select_candle_field("volume").unwrap();

        let anchor_ranges = vec![
            ("1m", "5m", 1),
            ("1h", "6h", 1),
            ("1d", "5d", 1),
            ("7d", "14d", 7),
        ];

        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for (start, end, step) in anchor_ranges {
            for source in &test_sources {
                let prices = match *source {
                    "close" => c.select_candle_field("close").unwrap(),
                    "open" => c.select_candle_field("open").unwrap(),
                    "high" => c.select_candle_field("high").unwrap(),
                    "low" => c.select_candle_field("low").unwrap(),
                    _ => c.get_calculated_field(source).unwrap(),
                };

                let output = VwapBatchBuilder::new()
                    .kernel(kernel)
                    .anchor_range(start, end, step)
                    .apply_slice(timestamps, volumes, prices)?;

                for (idx, &val) in output.values.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();
                    let row = idx / output.cols;
                    let col = idx % output.cols;

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with anchor_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with anchor_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with anchor_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, start, end, step, source
                        );
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_anchor_grid);
    gen_batch_tests!(check_batch_no_poison);
}
