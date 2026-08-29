use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DmaData::Slice(slice) => slice,
            DmaData::Candles { candles, source } => dma_source_slice(candles, source),
        }
    }
}

#[inline(always)]
fn dma_source_slice<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => candles.open.as_slice(),
        "high" => candles.high.as_slice(),
        "low" => candles.low.as_slice(),
        "close" => candles.close.as_slice(),
        "volume" => candles.volume.as_slice(),
        "hl2" => candles.hl2.as_slice(),
        "hlc3" => candles.hlc3.as_slice(),
        "ohlc4" => candles.ohlc4.as_slice(),
        "hlcc4" | "hlcc" => candles.hlcc4.as_slice(),
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum DmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DmaParams {
    pub hull_length: Option<usize>,
    pub ema_length: Option<usize>,
    pub ema_gain_limit: Option<usize>,
    pub hull_ma_type: Option<String>,
}

impl Default for DmaParams {
    fn default() -> Self {
        Self {
            hull_length: Some(7),
            ema_length: Some(20),
            ema_gain_limit: Some(50),
            hull_ma_type: Some("WMA".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DmaInput<'a> {
    pub data: DmaData<'a>,
    pub params: DmaParams,
}

impl<'a> DmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DmaParams) -> Self {
        Self {
            data: DmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DmaParams) -> Self {
        Self {
            data: DmaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DmaParams::default())
    }

    #[inline]
    pub fn get_hull_length(&self) -> usize {
        self.params.hull_length.unwrap_or(7)
    }

    #[inline]
    pub fn get_ema_length(&self) -> usize {
        self.params.ema_length.unwrap_or(20)
    }

    #[inline]
    pub fn get_ema_gain_limit(&self) -> usize {
        self.params.ema_gain_limit.unwrap_or(50)
    }

    #[inline]
    pub fn get_hull_ma_type(&self) -> String {
        self.params
            .hull_ma_type
            .clone()
            .unwrap_or_else(|| "WMA".to_string())
    }

    #[inline]
    pub fn hull_ma_type_str(&self) -> &str {
        self.params.hull_ma_type.as_deref().unwrap_or("WMA")
    }
}

#[derive(Clone, Debug)]
pub struct DmaBuilder {
    hull_length: Option<usize>,
    ema_length: Option<usize>,
    ema_gain_limit: Option<usize>,
    hull_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for DmaBuilder {
    fn default() -> Self {
        Self {
            hull_length: None,
            ema_length: None,
            ema_gain_limit: None,
            hull_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn hull_length(mut self, val: usize) -> Self {
        self.hull_length = Some(val);
        self
    }

    #[inline(always)]
    pub fn ema_length(mut self, val: usize) -> Self {
        self.ema_length = Some(val);
        self
    }

    #[inline(always)]
    pub fn ema_gain_limit(mut self, val: usize) -> Self {
        self.ema_gain_limit = Some(val);
        self
    }

    #[inline(always)]
    pub fn hull_ma_type(mut self, val: String) -> Self {
        self.hull_ma_type = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DmaOutput, DmaError> {
        let p = DmaParams {
            hull_length: self.hull_length,
            ema_length: self.ema_length,
            ema_gain_limit: self.ema_gain_limit,
            hull_ma_type: self.hull_ma_type,
        };
        let i = DmaInput::from_candles(c, "close", p);
        dma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DmaOutput, DmaError> {
        let p = DmaParams {
            hull_length: self.hull_length,
            ema_length: self.ema_length,
            ema_gain_limit: self.ema_gain_limit,
            hull_ma_type: self.hull_ma_type,
        };
        let i = DmaInput::from_slice(d, p);
        dma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DmaStream, DmaError> {
        let p = DmaParams {
            hull_length: self.hull_length,
            ema_length: self.ema_length,
            ema_gain_limit: self.ema_gain_limit,
            hull_ma_type: self.hull_ma_type,
        };
        DmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DmaError {
    #[error("dma: Input data slice is empty.")]
    EmptyInputData,

    #[error("dma: All values are NaN.")]
    AllValuesNaN,

    #[error("dma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("dma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("dma: Invalid Hull MA type: {value}. Must be 'WMA' or 'EMA'.")]
    InvalidHullMAType { value: String },

    #[error("dma: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("dma: Invalid range expansion: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("dma: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
pub fn dma(input: &DmaInput) -> Result<DmaOutput, DmaError> {
    dma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn dma_with_kernel(input: &DmaInput, kernel: Kernel) -> Result<DmaOutput, DmaError> {
    let (data, hull_len, ema_len, ema_gain_limit, hull_ma_type, first, chosen) =
        dma_prepare(input, kernel)?;

    let sqrt_len = (hull_len as f64).sqrt().round() as usize;
    let warmup_end = first + hull_len.max(ema_len) + sqrt_len - 1;

    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);
    dma_compute_into(
        data,
        hull_len,
        ema_len,
        ema_gain_limit,
        &hull_ma_type,
        first,
        chosen,
        &mut out,
    );
    Ok(DmaOutput { values: out })
}

#[inline(always)]
pub fn dma_into(input: &DmaInput, out: &mut [f64]) -> Result<(), DmaError> {
    let (data, hull_len, ema_len, ema_gain_limit, hull_ma_type, first, chosen) =
        dma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(DmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let sqrt_len = (hull_len as f64).sqrt().round() as usize;
    let warmup_end = first + hull_len.max(ema_len) + sqrt_len - 1;
    let end = warmup_end.min(out.len());
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..end] {
        *v = qnan;
    }

    dma_compute_into(
        data,
        hull_len,
        ema_len,
        ema_gain_limit,
        &hull_ma_type,
        first,
        chosen,
        out,
    );
    Ok(())
}

#[inline(always)]
pub fn dma_into_slice(dst: &mut [f64], input: &DmaInput, kern: Kernel) -> Result<(), DmaError> {
    let (data, hull_len, ema_len, ema_gain_limit, hull_ma_type, first, chosen) =
        dma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(DmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    dma_compute_into(
        data,
        hull_len,
        ema_len,
        ema_gain_limit,
        &hull_ma_type,
        first,
        chosen,
        dst,
    );

    let sqrt_len = (hull_len as f64).sqrt().round() as usize;
    let warmup_end = first + hull_len.max(ema_len) + sqrt_len - 1;
    let end = warmup_end.min(dst.len());
    for v in &mut dst[..end] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
fn dma_prepare<'a>(
    input: &'a DmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, &'a str, usize, Kernel), DmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(DmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DmaError::AllValuesNaN)?;
    let hull_length = input.get_hull_length();
    let ema_length = input.get_ema_length();
    let ema_gain_limit = input.get_ema_gain_limit();
    let hull_ma_type = input.hull_ma_type_str();

    if hull_length == 0 || hull_length > len {
        return Err(DmaError::InvalidPeriod {
            period: hull_length,
            data_len: len,
        });
    }
    if ema_length == 0 || ema_length > len {
        return Err(DmaError::InvalidPeriod {
            period: ema_length,
            data_len: len,
        });
    }

    let sqrt_len = (hull_length as f64).sqrt().round() as usize;
    let needed = hull_length.max(ema_length) + sqrt_len;
    if len - first < needed {
        return Err(DmaError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }
    if hull_ma_type != "WMA" && hull_ma_type != "EMA" {
        return Err(DmaError::InvalidHullMAType {
            value: hull_ma_type.to_string(),
        });
    }
    let chosen = match kernel {
        Kernel::Auto => dma_auto_kernel(len),
        k => k,
    };
    Ok((
        data,
        hull_length,
        ema_length,
        ema_gain_limit,
        hull_ma_type,
        first,
        chosen,
    ))
}

#[inline(always)]
fn dma_auto_kernel(_len: usize) -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Kernel::Avx2;
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return Kernel::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Kernel::Avx2;
        }
    }

    Kernel::Scalar
}

#[inline(always)]
fn dma_compute_into(
    data: &[f64],
    hull_length: usize,
    ema_length: usize,
    ema_gain_limit: usize,
    hull_ma_type: &str,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                dma_simd128(
                    data,
                    hull_length,
                    ema_length,
                    ema_gain_limit,
                    hull_ma_type,
                    first,
                    out,
                );
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => dma_scalar(
                data,
                hull_length,
                ema_length,
                ema_gain_limit,
                hull_ma_type,
                first,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => dma_avx2(
                data,
                hull_length,
                ema_length,
                ema_gain_limit,
                hull_ma_type,
                first,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    dma_avx2(
                        data,
                        hull_length,
                        ema_length,
                        ema_gain_limit,
                        hull_ma_type,
                        first,
                        out,
                    )
                } else {
                    dma_avx512(
                        data,
                        hull_length,
                        ema_length,
                        ema_gain_limit,
                        hull_ma_type,
                        first,
                        out,
                    )
                }
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => dma_scalar(
                data,
                hull_length,
                ema_length,
                ema_gain_limit,
                hull_ma_type,
                first,
                out,
            ),
            _ => unreachable!(),
        }
    }
}

/// Canonical f64 reduction order for DMA seeds.
///
/// DMA is a stateful recurrence, so even a one-ULP seed reassociation changes
/// later bars. Keep this single accumulator in oldest-to-newest order across
/// Scalar, Auto/AVX, batch, stream, and the strict CUDA f64 lane. In
/// particular, an AVX horizontal sum is not an equivalent implementation of
/// this f64 contract.
#[inline]
fn dma_sum_chronological_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    for &value in values {
        sum += value;
    }
    sum
}

/// Return the unweighted and linearly weighted sums in the canonical DMA f64
/// order. `values` is oldest-to-newest and therefore receives weights 1..=N.
#[inline]
fn dma_seed_wma_chronological_f64(values: &[f64]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut weighted_sum = 0.0;
    for (index, &value) in values.iter().enumerate() {
        sum += value;
        let weighted_value = (index + 1) as f64 * value;
        weighted_sum += weighted_value;
    }
    (sum, weighted_sum)
}

/// Exact minimizer of Dickson's quantized gain search without evaluating all
/// `2 * limit + 1` candidates.
///
/// Nathan Dickson's primary definition searches integer tenths from
/// `-ema_gain_limit` through `+ema_gain_limit`. For `t != 0`, minimizing
/// `abs(r - t * gain)` over that grid is the nearest clamped point to `r / t`;
/// evaluating `floor(target)` and its next grid point preserves the creator's
/// lower-gain tie break while remaining O(1).
#[inline]
fn dma_quantized_best_gain_f64(r: f64, t: f64, ema_gain_limit: usize) -> f64 {
    if t == 0.0 {
        return 0.0;
    }

    let limit_i = i64::try_from(ema_gain_limit).unwrap_or(i64::MAX);
    let lower_i = -limit_i;
    let target = (r / t) * 10.0;
    let mut i0 = target.floor() as i64;
    if i0 < lower_i {
        i0 = lower_i;
    } else if i0 > limit_i {
        i0 = limit_i;
    }
    let i1 = if i0 < limit_i { i0 + 1 } else { i0 };
    // Preserve Dickson's `value1 / 10` construction. Multiplying by the
    // already-rounded binary64 literal 0.1 is a different f64 schedule.
    let g0 = (i0 as f64) / 10.0;
    let g1 = (i1 as f64) / 10.0;
    let e0 = (r - t * g0).abs();
    let e1 = (r - t * g1).abs();
    if e0 <= e1 { g0 } else { g1 }
}

#[inline]
pub fn dma_scalar(
    data: &[f64],
    hull_length: usize,
    ema_length: usize,
    ema_gain_limit: usize,
    hull_ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let alpha_e = 2.0 / (ema_length as f64 + 1.0);
    let one_minus_alpha_e = 1.0 - alpha_e;
    let i0_e = first + ema_length.saturating_sub(1);
    let mut e0_prev = 0.0;
    let mut e0_init_done = false;
    let mut ec_prev = 0.0;
    let mut ec_init_done = false;

    let half = hull_length / 2;
    let sqrt_len = (hull_length as f64).sqrt().round() as usize;

    let mut hull_val = f64::NAN;

    let wsum = |p: usize| -> f64 { (p * (p + 1)) as f64 / 2.0 };
    let i0_half = first + half.saturating_sub(1);
    let i0_full = first + hull_length.saturating_sub(1);

    let mut a_half = 0.0;
    let mut s_half = 0.0;
    let mut half_ready = false;

    let mut a_full = 0.0;
    let mut s_full = 0.0;
    let mut full_ready = false;

    let mut diff_ring: Vec<f64> = Vec::with_capacity(sqrt_len.max(1));
    let mut diff_pos: usize = 0;
    let mut diff_filled = 0usize;

    let mut a_diff = 0.0;
    let mut s_diff = 0.0;
    let mut diff_wma_init_done = false;

    let alpha_sqrt = if sqrt_len > 0 {
        2.0 / (sqrt_len as f64 + 1.0)
    } else {
        0.0
    };
    let mut diff_ema = 0.0;
    let mut diff_ema_init_done = false;
    let mut diff_sum_seed = 0.0;

    let mut e_half_prev = 0.0;
    let mut e_half_init_done = false;
    let mut e_full_prev = 0.0;
    let mut e_full_init_done = false;
    let alpha_half = if half > 0 {
        2.0 / (half as f64 + 1.0)
    } else {
        0.0
    };
    let alpha_full = if hull_length > 0 {
        2.0 / (hull_length as f64 + 1.0)
    } else {
        0.0
    };

    let is_wma = hull_ma_type == "WMA";

    for i in first..n {
        let x = data[i];

        if !e0_init_done {
            if i >= i0_e {
                let start = i + 1 - ema_length;
                let sum = dma_sum_chronological_f64(&data[start..=i]);
                e0_prev = sum / ema_length as f64;
                e0_init_done = true;
            }
        } else {
            e0_prev = x.mul_add(alpha_e, one_minus_alpha_e * e0_prev);
        }

        let mut diff_now = f64::NAN;

        if is_wma {
            if half > 0 {
                if !half_ready {
                    if i >= i0_half {
                        let start = i + 1 - half;
                        let (sum, wsum_local) = dma_seed_wma_chronological_f64(&data[start..=i]);
                        a_half = sum;
                        s_half = wsum_local;
                        half_ready = true;
                    }
                } else {
                    let a_prev = a_half;
                    a_half = a_prev + x - data[i - half];
                    s_half = s_half + (half as f64) * x - a_prev;
                }
            }

            if hull_length > 0 {
                if !full_ready {
                    if i >= i0_full {
                        let start = i + 1 - hull_length;
                        let (sum, wsum_local) = dma_seed_wma_chronological_f64(&data[start..=i]);
                        a_full = sum;
                        s_full = wsum_local;
                        full_ready = true;
                    }
                } else {
                    let a_prev = a_full;
                    a_full = a_prev + x - data[i - hull_length];
                    s_full = s_full + (hull_length as f64) * x - a_prev;
                }
            }

            if half_ready && full_ready {
                let w_half = s_half / wsum(half).max(1.0);
                let w_full = s_full / wsum(hull_length).max(1.0);
                diff_now = 2.0 * w_half - w_full;
            }
        } else {
            if half > 0 {
                if !e_half_init_done {
                    if i >= i0_half {
                        let start = i + 1 - half;
                        let sum = dma_sum_chronological_f64(&data[start..=i]);
                        e_half_prev = sum / half as f64;
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = x.mul_add(alpha_half, (1.0 - alpha_half) * e_half_prev);
                }
            }

            if hull_length > 0 {
                if !e_full_init_done {
                    if i >= i0_full {
                        let start = i + 1 - hull_length;
                        let sum = dma_sum_chronological_f64(&data[start..=i]);
                        e_full_prev = sum / hull_length as f64;
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = x.mul_add(alpha_full, (1.0 - alpha_full) * e_full_prev);
                }
            }

            if e_half_init_done && e_full_init_done {
                diff_now = 2.0 * e_half_prev - e_full_prev;
            }
        }

        if diff_now.is_finite() && sqrt_len > 0 {
            if diff_filled < sqrt_len {
                diff_ring.push(diff_now);
                diff_sum_seed += diff_now;
                diff_filled += 1;

                if diff_filled == sqrt_len {
                    if is_wma {
                        (a_diff, s_diff) = dma_seed_wma_chronological_f64(&diff_ring);
                        diff_wma_init_done = true;
                        hull_val = s_diff / wsum(sqrt_len).max(1.0);
                    } else {
                        diff_ema = diff_sum_seed / sqrt_len as f64;
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                let old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos = (diff_pos + 1) % sqrt_len;

                if is_wma {
                    let a_prev = a_diff;
                    a_diff = a_prev + diff_now - old;
                    s_diff = s_diff + (sqrt_len as f64) * diff_now - a_prev;
                    hull_val = s_diff / wsum(sqrt_len).max(1.0);
                } else {
                    diff_ema = diff_now.mul_add(alpha_sqrt, (1.0 - alpha_sqrt) * diff_ema);
                    hull_val = diff_ema;
                }
            }
        }

        let mut ec_now = f64::NAN;
        if e0_init_done {
            if !ec_init_done {
                ec_prev = e0_prev;
                ec_init_done = true;
                ec_now = ec_prev;
            } else {
                let dx = x - ec_prev;
                let t = alpha_e * dx;
                let base = e0_prev.mul_add(alpha_e, one_minus_alpha_e * ec_prev);
                let r = x - base;
                let g_sel = dma_quantized_best_gain_f64(r, t, ema_gain_limit);

                ec_now = (e0_prev + g_sel * dx).mul_add(alpha_e, one_minus_alpha_e * ec_prev);
                ec_prev = ec_now;
            }
        }

        if hull_val.is_finite() && ec_now.is_finite() {
            out[i] = 0.5 * (hull_val + ec_now);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn dma_simd128(
    data: &[f64],
    hull_length: usize,
    ema_length: usize,
    ema_gain_limit: usize,
    hull_ma_type: &str,
    first_val: usize,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;
    dma_scalar(
        data,
        hull_length,
        ema_length,
        ema_gain_limit,
        hull_ma_type,
        first_val,
        out,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dma_avx2(
    data: &[f64],
    hull_length: usize,
    ema_length: usize,
    ema_gain_limit: usize,
    hull_ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let alpha_e = 2.0 / (ema_length as f64 + 1.0);
    let i0_e = first + ema_length.saturating_sub(1);
    let mut e0_prev = 0.0;
    let mut e0_init_done = false;
    let mut ec_prev = 0.0;
    let mut ec_init_done = false;

    let half = hull_length / 2;
    let sqrt_len = (hull_length as f64).sqrt().round() as usize;

    let mut hull_val = f64::NAN;

    let wsum = |p: usize| -> f64 { (p * (p + 1)) as f64 / 2.0 };
    let i0_half = first + half.saturating_sub(1);
    let i0_full = first + hull_length.saturating_sub(1);

    let mut a_half = 0.0;
    let mut s_half = 0.0;
    let mut half_ready = false;

    let mut a_full = 0.0;
    let mut s_full = 0.0;
    let mut full_ready = false;

    let mut diff_ring: Vec<f64> = Vec::with_capacity(sqrt_len.max(1));
    let mut diff_pos: usize = 0;
    let mut diff_filled = 0usize;

    let mut a_diff = 0.0;
    let mut s_diff = 0.0;
    let mut diff_wma_init_done = false;

    let alpha_sqrt = if sqrt_len > 0 {
        2.0 / (sqrt_len as f64 + 1.0)
    } else {
        0.0
    };
    let mut diff_ema = 0.0;
    let mut diff_ema_init_done = false;
    let mut diff_sum_seed = 0.0;

    let mut e_half_prev = 0.0;
    let mut e_half_init_done = false;
    let mut e_full_prev = 0.0;
    let mut e_full_init_done = false;
    let alpha_half = if half > 0 {
        2.0 / (half as f64 + 1.0)
    } else {
        0.0
    };
    let alpha_full = if hull_length > 0 {
        2.0 / (hull_length as f64 + 1.0)
    } else {
        0.0
    };

    let is_wma = hull_ma_type == "WMA";

    for i in first..n {
        let x = data[i];

        if !e0_init_done {
            if i >= i0_e {
                let start = i + 1 - ema_length;
                let sum = dma_sum_chronological_f64(&data[start..=i]);
                e0_prev = sum / ema_length as f64;
                e0_init_done = true;
            }
        } else {
            e0_prev = x.mul_add(alpha_e, (1.0 - alpha_e) * e0_prev);
        }

        let mut diff_now = f64::NAN;

        if is_wma {
            if half > 0 {
                if !half_ready {
                    if i >= i0_half {
                        let start = i + 1 - half;
                        let (sum, wsum_local) = dma_seed_wma_chronological_f64(&data[start..=i]);
                        a_half = sum;
                        s_half = wsum_local;
                        half_ready = true;
                    }
                } else {
                    let a_prev = a_half;
                    a_half = a_prev + x - data[i - half];
                    s_half = s_half + (half as f64) * x - a_prev;
                }
            }

            if hull_length > 0 {
                if !full_ready {
                    if i >= i0_full {
                        let start = i + 1 - hull_length;
                        let (sum, wsum_local) = dma_seed_wma_chronological_f64(&data[start..=i]);
                        a_full = sum;
                        s_full = wsum_local;
                        full_ready = true;
                    }
                } else {
                    let a_prev = a_full;
                    a_full = a_prev + x - data[i - hull_length];
                    s_full = s_full + (hull_length as f64) * x - a_prev;
                }
            }

            if half_ready && full_ready {
                let w_half = s_half / wsum(half).max(1.0);
                let w_full = s_full / wsum(hull_length).max(1.0);
                diff_now = 2.0 * w_half - w_full;
            }
        } else {
            if half > 0 {
                if !e_half_init_done {
                    if i >= i0_half {
                        let start = i + 1 - half;
                        let sum = dma_sum_chronological_f64(&data[start..=i]);
                        e_half_prev = sum / half as f64;
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = x.mul_add(alpha_half, (1.0 - alpha_half) * e_half_prev);
                }
            }

            if hull_length > 0 {
                if !e_full_init_done {
                    if i >= i0_full {
                        let start = i + 1 - hull_length;
                        let sum = dma_sum_chronological_f64(&data[start..=i]);
                        e_full_prev = sum / hull_length as f64;
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = x.mul_add(alpha_full, (1.0 - alpha_full) * e_full_prev);
                }
            }

            if e_half_init_done && e_full_init_done {
                diff_now = 2.0 * e_half_prev - e_full_prev;
            }
        }

        if diff_now.is_finite() && sqrt_len > 0 {
            if diff_filled < sqrt_len {
                diff_ring.push(diff_now);
                diff_sum_seed += diff_now;
                diff_filled += 1;

                if diff_filled == sqrt_len {
                    if is_wma {
                        (a_diff, s_diff) = dma_seed_wma_chronological_f64(&diff_ring);
                        diff_wma_init_done = true;
                        let wsum_d = (sqrt_len * (sqrt_len + 1)) as f64 / 2.0;
                        hull_val = s_diff / wsum_d.max(1.0);
                    } else {
                        diff_ema = diff_sum_seed / sqrt_len as f64;
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                let old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos = (diff_pos + 1) % sqrt_len;

                if is_wma {
                    let a_prev = a_diff;
                    a_diff = a_prev + diff_now - old;
                    s_diff = s_diff + (sqrt_len as f64) * diff_now - a_prev;
                    let wsum_d = (sqrt_len * (sqrt_len + 1)) as f64 / 2.0;
                    hull_val = s_diff / wsum_d.max(1.0);
                } else {
                    diff_ema = diff_now.mul_add(alpha_sqrt, (1.0 - alpha_sqrt) * diff_ema);
                    hull_val = diff_ema;
                }
            }
        }

        let mut ec_now = f64::NAN;
        if e0_init_done {
            if !ec_init_done {
                ec_prev = e0_prev;
                ec_init_done = true;
                ec_now = ec_prev;
            } else {
                let dx = x - ec_prev;
                let t = alpha_e * dx;
                let base = e0_prev.mul_add(alpha_e, (1.0 - alpha_e) * ec_prev);
                let r = x - base;
                let g_sel = dma_quantized_best_gain_f64(r, t, ema_gain_limit);

                ec_now = (e0_prev + g_sel * dx).mul_add(alpha_e, (1.0 - alpha_e) * ec_prev);
                ec_prev = ec_now;
            }
        }

        if hull_val.is_finite() && ec_now.is_finite() {
            out[i] = 0.5 * (hull_val + ec_now);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dma_avx512(
    data: &[f64],
    hull_length: usize,
    ema_length: usize,
    ema_gain_limit: usize,
    hull_ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let alpha_e = 2.0 / (ema_length as f64 + 1.0);
    let i0_e = first + ema_length.saturating_sub(1);
    let mut e0_prev = 0.0;
    let mut e0_init_done = false;
    let mut ec_prev = 0.0;
    let mut ec_init_done = false;

    let half = hull_length / 2;
    let sqrt_len = (hull_length as f64).sqrt().round() as usize;

    let mut hull_val = f64::NAN;

    let wsum = |p: usize| -> f64 { (p * (p + 1)) as f64 / 2.0 };
    let i0_half = first + half.saturating_sub(1);
    let i0_full = first + hull_length.saturating_sub(1);

    let mut a_half = 0.0;
    let mut s_half = 0.0;
    let mut half_ready = false;

    let mut a_full = 0.0;
    let mut s_full = 0.0;
    let mut full_ready = false;

    let mut diff_ring: Vec<f64> = Vec::with_capacity(sqrt_len.max(1));
    let mut diff_pos: usize = 0;
    let mut diff_filled = 0usize;

    let mut a_diff = 0.0;
    let mut s_diff = 0.0;
    let mut diff_wma_init_done = false;

    let alpha_sqrt = if sqrt_len > 0 {
        2.0 / (sqrt_len as f64 + 1.0)
    } else {
        0.0
    };
    let mut diff_ema = 0.0;
    let mut diff_ema_init_done = false;
    let mut diff_sum_seed = 0.0;

    let mut e_half_prev = 0.0;
    let mut e_half_init_done = false;
    let mut e_full_prev = 0.0;
    let mut e_full_init_done = false;
    let alpha_half = if half > 0 {
        2.0 / (half as f64 + 1.0)
    } else {
        0.0
    };
    let alpha_full = if hull_length > 0 {
        2.0 / (hull_length as f64 + 1.0)
    } else {
        0.0
    };

    let is_wma = hull_ma_type == "WMA";

    for i in first..n {
        let x = data[i];

        if !e0_init_done {
            if i >= i0_e {
                let start = i + 1 - ema_length;
                let sum = dma_sum_chronological_f64(&data[start..=i]);
                e0_prev = sum / ema_length as f64;
                e0_init_done = true;
            }
        } else {
            e0_prev = x.mul_add(alpha_e, (1.0 - alpha_e) * e0_prev);
        }

        let mut diff_now = f64::NAN;

        if is_wma {
            if half > 0 {
                if !half_ready {
                    if i >= i0_half {
                        let start = i + 1 - half;
                        let (sum, wsum_local) = dma_seed_wma_chronological_f64(&data[start..=i]);
                        a_half = sum;
                        s_half = wsum_local;
                        half_ready = true;
                    }
                } else {
                    let a_prev = a_half;
                    a_half = a_prev + x - data[i - half];
                    s_half = s_half + (half as f64) * x - a_prev;
                }
            }

            if hull_length > 0 {
                if !full_ready {
                    if i >= i0_full {
                        let start = i + 1 - hull_length;
                        let (sum, wsum_local) = dma_seed_wma_chronological_f64(&data[start..=i]);
                        a_full = sum;
                        s_full = wsum_local;
                        full_ready = true;
                    }
                } else {
                    let a_prev = a_full;
                    a_full = a_prev + x - data[i - hull_length];
                    s_full = s_full + (hull_length as f64) * x - a_prev;
                }
            }

            if half_ready && full_ready {
                let w_half = s_half / wsum(half).max(1.0);
                let w_full = s_full / wsum(hull_length).max(1.0);
                diff_now = 2.0 * w_half - w_full;
            }
        } else {
            if half > 0 {
                if !e_half_init_done {
                    if i >= i0_half {
                        let start = i + 1 - half;
                        let sum = dma_sum_chronological_f64(&data[start..=i]);
                        e_half_prev = sum / half as f64;
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = x.mul_add(alpha_half, (1.0 - alpha_half) * e_half_prev);
                }
            }

            if hull_length > 0 {
                if !e_full_init_done {
                    if i >= i0_full {
                        let start = i + 1 - hull_length;
                        let sum = dma_sum_chronological_f64(&data[start..=i]);
                        e_full_prev = sum / hull_length as f64;
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = x.mul_add(alpha_full, (1.0 - alpha_full) * e_full_prev);
                }
            }

            if e_half_init_done && e_full_init_done {
                diff_now = 2.0 * e_half_prev - e_full_prev;
            }
        }

        if diff_now.is_finite() && sqrt_len > 0 {
            if diff_filled < sqrt_len {
                diff_ring.push(diff_now);
                diff_sum_seed += diff_now;
                diff_filled += 1;

                if diff_filled == sqrt_len {
                    if is_wma {
                        (a_diff, s_diff) = dma_seed_wma_chronological_f64(&diff_ring);
                        diff_wma_init_done = true;
                        let wsum_d = (sqrt_len * (sqrt_len + 1)) as f64 / 2.0;
                        hull_val = s_diff / wsum_d.max(1.0);
                    } else {
                        diff_ema = diff_sum_seed / sqrt_len as f64;
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                let old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos = (diff_pos + 1) % sqrt_len;

                if is_wma {
                    let a_prev = a_diff;
                    a_diff = a_prev + diff_now - old;
                    s_diff = s_diff + (sqrt_len as f64) * diff_now - a_prev;
                    let wsum_d = (sqrt_len * (sqrt_len + 1)) as f64 / 2.0;
                    hull_val = s_diff / wsum_d.max(1.0);
                } else {
                    diff_ema = diff_now.mul_add(alpha_sqrt, (1.0 - alpha_sqrt) * diff_ema);
                    hull_val = diff_ema;
                }
            }
        }

        let mut ec_now = f64::NAN;
        if e0_init_done {
            if !ec_init_done {
                ec_prev = e0_prev;
                ec_init_done = true;
                ec_now = ec_prev;
            } else {
                let dx = x - ec_prev;
                let t = alpha_e * dx;
                let base = e0_prev.mul_add(alpha_e, (1.0 - alpha_e) * ec_prev);
                let r = x - base;
                let g_sel = dma_quantized_best_gain_f64(r, t, ema_gain_limit);

                ec_now = (e0_prev + g_sel * dx).mul_add(alpha_e, (1.0 - alpha_e) * ec_prev);
                ec_prev = ec_now;
            }
        }

        if hull_val.is_finite() && ec_now.is_finite() {
            out[i] = 0.5 * (hull_val + ec_now);
        }
    }
}
#[derive(Debug, Clone)]
pub struct DmaStream {
    ema_length: usize,
    ema_gain_limit: usize,
    hull_length: usize,
    half: usize,
    sqrt_len: usize,
    is_wma: bool,

    cap: usize,
    ring: Vec<f64>,
    head: usize,
    filled: usize,

    i: usize,
    seen_first: bool,

    alpha_e: f64,
    e0_prev: f64,
    e0_ready: bool,

    ec_prev: f64,
    ec_ready: bool,

    sum_half: f64,
    sum_full: f64,
    s_half: f64,
    s_full: f64,
    half_ready: bool,
    full_ready: bool,

    alpha_half: f64,
    alpha_full: f64,
    e_half_prev: f64,
    e_full_prev: f64,
    e_half_ready: bool,
    e_full_ready: bool,

    a_diff: f64,
    s_diff: f64,

    alpha_sqrt: f64,
    diff_ema: f64,

    diff_ring: Vec<f64>,
    diff_head: usize,
    diff_filled: usize,
}

impl DmaStream {
    pub fn try_new(params: DmaParams) -> Result<Self, DmaError> {
        let hull_length = params.hull_length.unwrap_or(7);
        let ema_length = params.ema_length.unwrap_or(20);
        let ema_gain_limit = params.ema_gain_limit.unwrap_or(50);
        let hull_ma_type = params.hull_ma_type.unwrap_or_else(|| "WMA".to_string());
        if hull_length == 0 || ema_length == 0 {
            return Err(DmaError::InvalidPeriod {
                period: hull_length.max(ema_length),
                data_len: 0,
            });
        }
        if hull_ma_type != "WMA" && hull_ma_type != "EMA" {
            return Err(DmaError::InvalidHullMAType {
                value: hull_ma_type,
            });
        }

        let half = hull_length / 2;
        let sqrt_len = (hull_length as f64).sqrt().round() as usize;
        let cap = hull_length.max(ema_length).max(1);

        Ok(Self {
            ema_length,
            ema_gain_limit,
            hull_length,
            half,
            sqrt_len,
            is_wma: hull_ma_type == "WMA",

            cap,
            ring: vec![f64::NAN; cap],
            head: 0,
            filled: 0,
            i: 0,
            seen_first: false,

            alpha_e: 2.0 / (ema_length as f64 + 1.0),
            e0_prev: 0.0,
            e0_ready: false,

            ec_prev: 0.0,
            ec_ready: false,

            sum_half: 0.0,
            sum_full: 0.0,
            s_half: 0.0,
            s_full: 0.0,
            half_ready: false,
            full_ready: false,

            alpha_half: if half > 0 {
                2.0 / (half as f64 + 1.0)
            } else {
                0.0
            },
            alpha_full: 2.0 / (hull_length as f64 + 1.0),
            e_half_prev: 0.0,
            e_full_prev: 0.0,
            e_half_ready: false,
            e_full_ready: false,

            a_diff: 0.0,
            s_diff: 0.0,
            alpha_sqrt: if sqrt_len > 0 {
                2.0 / (sqrt_len as f64 + 1.0)
            } else {
                0.0
            },
            diff_ema: 0.0,
            diff_ring: vec![f64::NAN; sqrt_len.max(1)],
            diff_head: 0,
            diff_filled: 0,
        })
    }

    #[inline]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        if !self.seen_first {
            self.i += 1;
            if x.is_nan() {
                return None;
            }
            self.seen_first = true;
        }

        let old_head = self.head;
        let overwritten = self.ring[old_head];
        self.ring[old_head] = x;
        self.head = (old_head + 1) % self.cap;
        if self.filled < self.cap {
            self.filled += 1;
        }

        #[inline(always)]
        fn kback(ring: &[f64], head: usize, cap: usize, k: usize) -> f64 {
            let idx = (head + cap - k % cap) % cap;
            ring[idx]
        }

        #[inline(always)]
        fn outgoing(ring: &[f64], head: usize, cap: usize, window: usize, overwritten: f64) -> f64 {
            if window == cap {
                overwritten
            } else {
                kback(ring, head, cap, window + 1)
            }
        }

        #[inline]
        fn ring_sum_chronological(ring: &[f64], head: usize, cap: usize, window: usize) -> f64 {
            let mut sum = 0.0;
            for j in 0..window {
                sum += kback(ring, head, cap, window - j);
            }
            sum
        }

        #[inline]
        fn ring_wma_seed_chronological(
            ring: &[f64],
            head: usize,
            cap: usize,
            window: usize,
        ) -> (f64, f64) {
            let mut sum = 0.0;
            let mut weighted_sum = 0.0;
            for j in 0..window {
                let value = kback(ring, head, cap, window - j);
                sum += value;
                let weighted_value = (j + 1) as f64 * value;
                weighted_sum += weighted_value;
            }
            (sum, weighted_sum)
        }

        if !self.e0_ready {
            if self.filled >= self.ema_length {
                self.e0_prev =
                    ring_sum_chronological(&self.ring, self.head, self.cap, self.ema_length)
                        / self.ema_length as f64;
                self.e0_ready = true;
            }
        } else {
            self.e0_prev = x.mul_add(self.alpha_e, (1.0 - self.alpha_e) * self.e0_prev);
        }

        let mut diff_now = f64::NAN;

        if self.is_wma {
            if self.half > 0 {
                if !self.half_ready {
                    if self.filled >= self.half {
                        (self.sum_half, self.s_half) =
                            ring_wma_seed_chronological(&self.ring, self.head, self.cap, self.half);
                        self.half_ready = true;
                    }
                } else {
                    let old = outgoing(&self.ring, self.head, self.cap, self.half, overwritten);
                    let a_prev = self.sum_half;
                    self.sum_half = a_prev + x - old;
                    self.s_half = self.s_half + (self.half as f64) * x - a_prev;
                }
            }

            if !self.full_ready {
                if self.filled >= self.hull_length {
                    (self.sum_full, self.s_full) = ring_wma_seed_chronological(
                        &self.ring,
                        self.head,
                        self.cap,
                        self.hull_length,
                    );
                    self.full_ready = true;
                }
            } else {
                let old = outgoing(
                    &self.ring,
                    self.head,
                    self.cap,
                    self.hull_length,
                    overwritten,
                );
                let a_prev = self.sum_full;
                self.sum_full = a_prev + x - old;
                self.s_full = self.s_full + (self.hull_length as f64) * x - a_prev;
            }

            if self.half_ready && self.full_ready {
                let wsum = |p: usize| (p * (p + 1)) as f64 / 2.0;
                let w_half = self.s_half / wsum(self.half).max(1.0);
                let w_full = self.s_full / wsum(self.hull_length).max(1.0);
                diff_now = 2.0 * w_half - w_full;
            }
        } else {
            if self.half > 0 {
                if !self.e_half_ready {
                    if self.filled >= self.half {
                        self.e_half_prev =
                            ring_sum_chronological(&self.ring, self.head, self.cap, self.half)
                                / self.half as f64;
                        self.e_half_ready = true;
                    }
                } else {
                    self.e_half_prev =
                        x.mul_add(self.alpha_half, (1.0 - self.alpha_half) * self.e_half_prev);
                }
            }

            if !self.e_full_ready {
                if self.filled >= self.hull_length {
                    self.e_full_prev =
                        ring_sum_chronological(&self.ring, self.head, self.cap, self.hull_length)
                            / self.hull_length as f64;
                    self.e_full_ready = true;
                }
            } else {
                self.e_full_prev =
                    x.mul_add(self.alpha_full, (1.0 - self.alpha_full) * self.e_full_prev);
            }

            if self.e_half_ready && self.e_full_ready {
                diff_now = 2.0 * self.e_half_prev - self.e_full_prev;
            }
        }

        let mut hull_val = f64::NAN;
        if diff_now.is_finite() && self.sqrt_len > 0 {
            if self.diff_filled < self.sqrt_len {
                self.diff_ring[self.diff_filled] = diff_now;
                self.diff_filled += 1;

                if self.diff_filled == self.sqrt_len {
                    if self.is_wma {
                        (self.a_diff, self.s_diff) =
                            dma_seed_wma_chronological_f64(&self.diff_ring[..self.sqrt_len]);
                        let wsum = (self.sqrt_len * (self.sqrt_len + 1)) as f64 / 2.0;
                        hull_val = self.s_diff / wsum.max(1.0);
                    } else {
                        self.diff_ema = dma_sum_chronological_f64(&self.diff_ring[..self.sqrt_len])
                            / self.sqrt_len as f64;
                        hull_val = self.diff_ema;
                    }
                }
            } else {
                let old = self.diff_ring[self.diff_head];
                self.diff_ring[self.diff_head] = diff_now;
                self.diff_head = (self.diff_head + 1) % self.sqrt_len;

                if self.is_wma {
                    let a_prev = self.a_diff;
                    self.a_diff = a_prev + diff_now - old;
                    self.s_diff = self.s_diff + (self.sqrt_len as f64) * diff_now - a_prev;
                    let wsum = (self.sqrt_len * (self.sqrt_len + 1)) as f64 / 2.0;
                    hull_val = self.s_diff / wsum.max(1.0);
                } else {
                    self.diff_ema =
                        diff_now.mul_add(self.alpha_sqrt, (1.0 - self.alpha_sqrt) * self.diff_ema);
                    hull_val = self.diff_ema;
                }
            }
        }

        let mut ec_now = f64::NAN;
        if self.e0_ready {
            if !self.ec_ready {
                self.ec_prev = self.e0_prev;
                self.ec_ready = true;
                ec_now = self.ec_prev;
            } else {
                let one_minus_alpha_e = 1.0 - self.alpha_e;
                let dx = x - self.ec_prev;
                let t = self.alpha_e * dx;
                let base = self
                    .e0_prev
                    .mul_add(self.alpha_e, one_minus_alpha_e * self.ec_prev);
                let r = x - base;
                let g_sel = dma_quantized_best_gain_f64(r, t, self.ema_gain_limit);

                let ec = (self.e0_prev + g_sel * dx)
                    .mul_add(self.alpha_e, one_minus_alpha_e * self.ec_prev);
                self.ec_prev = ec;
                ec_now = ec;
            }
        }

        self.i += 1;

        if hull_val.is_finite() && ec_now.is_finite() {
            Some(0.5 * (hull_val + ec_now))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct DmaBatchRange {
    pub hull_length: (usize, usize, usize),
    pub ema_length: (usize, usize, usize),
    pub ema_gain_limit: (usize, usize, usize),
    pub hull_ma_type: String,
}

impl Default for DmaBatchRange {
    fn default() -> Self {
        Self {
            hull_length: (7, 7, 0),
            ema_length: (20, 269, 1),
            ema_gain_limit: (50, 50, 0),
            hull_ma_type: "WMA".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DmaBatchBuilder {
    range: DmaBatchRange,
    kernel: Kernel,
}

impl DmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn hull_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.hull_length = (start, end, step);
        self
    }

    #[inline]
    pub fn hull_length_static(mut self, val: usize) -> Self {
        self.range.hull_length = (val, val, 0);
        self
    }

    #[inline]
    pub fn ema_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ema_length = (start, end, step);
        self
    }

    #[inline]
    pub fn ema_length_static(mut self, val: usize) -> Self {
        self.range.ema_length = (val, val, 0);
        self
    }

    #[inline]
    pub fn ema_gain_limit_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ema_gain_limit = (start, end, step);
        self
    }

    #[inline]
    pub fn ema_gain_limit_static(mut self, val: usize) -> Self {
        self.range.ema_gain_limit = (val, val, 0);
        self
    }

    #[inline]
    pub fn hull_ma_type(mut self, val: String) -> Self {
        self.range.hull_ma_type = val;
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<DmaBatchOutput, DmaError> {
        dma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DmaBatchOutput, DmaError> {
        DmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DmaBatchOutput, DmaError> {
        let slice = dma_source_slice(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<DmaBatchOutput, DmaError> {
        DmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct DmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DmaBatchOutput {
    pub fn row_for_params(&self, p: &DmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.hull_length.unwrap_or(7) == p.hull_length.unwrap_or(7)
                && c.ema_length.unwrap_or(20) == p.ema_length.unwrap_or(20)
                && c.ema_gain_limit.unwrap_or(50) == p.ema_gain_limit.unwrap_or(50)
                && c.hull_ma_type.as_ref().unwrap_or(&"WMA".to_string())
                    == p.hull_ma_type.as_ref().unwrap_or(&"WMA".to_string())
        })
    }

    pub fn values_for(&self, p: &DmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_dma(r: &DmaBatchRange) -> Vec<DmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start < end {
            return (start..=end).step_by(step).collect();
        }

        let mut v: Vec<usize> = (end..=start).step_by(step).collect();
        v.reverse();
        v
    }

    let hull_lengths = axis_usize(r.hull_length);
    let ema_lengths = axis_usize(r.ema_length);
    let ema_gain_limits = axis_usize(r.ema_gain_limit);

    let mut combos = Vec::new();
    for &h in &hull_lengths {
        for &e in &ema_lengths {
            for &g in &ema_gain_limits {
                combos.push(DmaParams {
                    hull_length: Some(h),
                    ema_length: Some(e),
                    ema_gain_limit: Some(g),
                    hull_ma_type: Some(r.hull_ma_type.clone()),
                });
            }
        }
    }
    combos
}

#[inline(always)]
pub fn dma_batch_slice(
    data: &[f64],
    sweep: &DmaBatchRange,
    kern: Kernel,
) -> Result<DmaBatchOutput, DmaError> {
    dma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dma_batch_par_slice(
    data: &[f64],
    sweep: &DmaBatchRange,
    kern: Kernel,
) -> Result<DmaBatchOutput, DmaError> {
    dma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn dma_batch_inner(
    data: &[f64],
    sweep: &DmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DmaBatchOutput, DmaError> {
    let combos = expand_grid_dma(sweep);
    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(DmaError::EmptyInputData);
    }
    if rows == 0 {
        return Err(DmaError::InvalidRange {
            start: sweep.hull_length.0,
            end: sweep.hull_length.1,
            step: sweep.hull_length.2,
        });
    }

    let _cap = rows.checked_mul(cols).ok_or(DmaError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DmaError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let h = c.hull_length.unwrap();
            let e = c.ema_length.unwrap();
            let sqrt_len = (h as f64).sqrt().round() as usize;
            first + h.max(e) + sqrt_len - 1
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    dma_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(DmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn dma_batch_with_kernel(
    data: &[f64],
    sweep: &DmaBatchRange,
    k: Kernel,
) -> Result<DmaBatchOutput, DmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DmaError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn dma_batch_inner_into(
    data: &[f64],
    sweep: &DmaBatchRange,
    k: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DmaParams>, DmaError> {
    let combos = expand_grid_dma(sweep);
    if combos.is_empty() {
        return Err(DmaError::InvalidRange {
            start: sweep.hull_length.0,
            end: sweep.hull_length.1,
            step: sweep.hull_length.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DmaError::AllValuesNaN)?;
    let cols = data.len();

    let actual = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let simd = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => actual,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };
        let prm = &combos[row];
        let hull_len = prm.hull_length.unwrap_or(7);
        let ema_len = prm.ema_length.unwrap_or(20);

        let sqrt_len = (hull_len as f64).sqrt().round() as usize;
        let warmup_end = first + hull_len.max(ema_len) + sqrt_len - 1;
        let warmup_end = warmup_end.min(dst.len());

        for i in 0..warmup_end {
            dst[i] = f64::NAN;
        }

        dma_compute_into(
            data,
            hull_len,
            ema_len,
            prm.ema_gain_limit.unwrap_or(50),
            prm.hull_ma_type.as_ref().unwrap_or(&"WMA".to_string()),
            first,
            simd,
            dst,
        );
    };

    let dst_mu = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        dst_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, row)| do_row(r, row));
        #[cfg(target_arch = "wasm32")]
        for (r, row) in dst_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    } else {
        for (r, row) in dst_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    fn check_dma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = DmaInput::from_candles(&candles, "close", DmaParams::default());
        let result = dma_with_kernel(&input, kernel)?;

        let expected_last_five = [
            59404.62489256,
            59326.48766951,
            59195.35128538,
            59153.22811529,
            58933.88503421,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 0.001,
                "[{}] DMA {:?} mismatch at idx {}: got {}, expected {}, diff {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i],
                diff
            );
        }
        Ok(())
    }

    fn check_dma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = DmaParams {
            hull_length: None,
            ema_length: None,
            ema_gain_limit: None,
            hull_ma_type: None,
        };
        let input = DmaInput::from_candles(&candles, "close", default_params);
        let output = dma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_dma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = DmaInput::with_default_candles(&candles);
        match input.data {
            DmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected DmaData::Candles"),
        }
        let output = dma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_dma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DmaParams {
            hull_length: Some(0),
            ema_length: None,
            ema_gain_limit: None,
            hull_ma_type: None,
        };
        let input = DmaInput::from_slice(&input_data, params);
        let res = dma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_dma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = DmaParams {
            hull_length: Some(10),
            ema_length: None,
            ema_gain_limit: None,
            hull_ma_type: None,
        };
        let input = DmaInput::from_slice(&data_small, params);
        let res = dma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_dma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = DmaParams::default();
        let input = DmaInput::from_slice(&single_point, params);
        let res = dma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_dma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = DmaParams::default();
        let input = DmaInput::from_slice(&empty, params);
        let res = dma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_dma_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = DmaParams::default();
        let input = DmaInput::from_slice(&nan_data, params);
        let res = dma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DMA should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_dma_invalid_hull_type(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0; 50];
        let params = DmaParams {
            hull_length: Some(7),
            ema_length: Some(20),
            ema_gain_limit: Some(50),
            hull_ma_type: Some("INVALID".to_string()),
        };
        let input = DmaInput::from_slice(&data, params);
        let res = dma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DMA should fail with invalid hull_ma_type",
            test_name
        );
        Ok(())
    }

    macro_rules! generate_all_dma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test] fn [<$test_fn _scalar>]() -> Result<(), Box<dyn Error>> { $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar) }
                    #[test] fn [<$test_fn _auto>  ]() -> Result<(), Box<dyn Error>> { $test_fn(stringify!([<$test_fn _auto>]),   Kernel::Auto) }
                )*
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2>  ]() -> Result<(), Box<dyn Error>> { $test_fn(stringify!([<$test_fn _avx2>]),   Kernel::Avx2) }
                    #[test] fn [<$test_fn _avx512>]() -> Result<(), Box<dyn Error>> { $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512) }
                )*
            }
        }
    }

    generate_all_dma_tests!(
        check_dma_accuracy,
        check_dma_partial_params,
        check_dma_default_candles,
        check_dma_zero_period,
        check_dma_period_exceeds_length,
        check_dma_very_small_dataset,
        check_dma_empty_input,
        check_dma_all_nan,
        check_dma_invalid_hull_type
    );

    macro_rules! generate_dma_batch_tests {
        ($($fn_name:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$fn_name _scalar_batch>]() -> Result<(), Box<dyn Error>> {
                        $fn_name(stringify!([<$fn_name _scalar_batch>]), Kernel::ScalarBatch)
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$fn_name _avx2_batch>]() -> Result<(), Box<dyn Error>> {
                        $fn_name(stringify!([<$fn_name _avx2_batch>]), Kernel::Avx2Batch)
                    }
                    #[test]
                    fn [<$fn_name _avx512_batch>]() -> Result<(), Box<dyn Error>> {
                        $fn_name(stringify!([<$fn_name _avx512_batch>]), Kernel::Avx512Batch)
                    }
                )*
            }
        };
    }

    generate_dma_batch_tests!(check_dma_batch_basic);

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

    gen_batch_tests!(check_batch_sweep);

    fn check_dma_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let first = DmaInput::from_candles(&c, "close", DmaParams::default());
        let out1 = dma_with_kernel(&first, kernel)?.values;

        let second = DmaInput::from_slice(&out1, DmaParams::default());
        let out2 = dma_with_kernel(&second, kernel)?.values;

        assert_eq!(out2.len(), out1.len());
        Ok(())
    }

    fn check_dma_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let p = DmaParams::default();
        let input = DmaInput::from_candles(&c, "close", p.clone());
        let out = dma_with_kernel(&input, kernel)?.values;

        let first = c.close.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let sqrt_len = (p.hull_length.unwrap_or(7) as f64).sqrt().round() as usize;
        let warm =
            first + p.hull_length.unwrap_or(7).max(p.ema_length.unwrap_or(20)) + sqrt_len - 1;
        for (i, &v) in out.iter().enumerate().skip(warm.min(out.len())) {
            assert!(!v.is_nan(), "[{test}] unexpected NaN at {i}");
        }
        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = DmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = DmaParams::default();
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out = DmaBatchBuilder::new()
            .kernel(kernel)
            .hull_length_range(7, 18, 1)
            .ema_length_range(10, 15, 1)
            .ema_gain_limit_range(10, 20, 5)
            .apply_candles(&c, "close")?;
        let expected = 12 * 6 * 3;
        assert_eq!(out.combos.len(), expected);
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    fn check_dma_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let p = DmaParams::default();

        let batch =
            dma_with_kernel(&DmaInput::from_candles(&c, "close", p.clone()), kernel)?.values;

        let mut s = DmaStream::try_new(p)?;
        let mut stream = Vec::with_capacity(c.close.len());
        for &x in &c.close {
            stream.push(s.update(x).unwrap_or(f64::NAN));
        }

        assert_eq!(batch.len(), stream.len());
        for (i, (&b, &t)) in batch.iter().zip(&stream).enumerate() {
            if b.is_nan() && t.is_nan() {
                continue;
            }
            assert!(
                (b - t).abs() < 1e-9,
                "[{test}] idx {i} diff {}",
                (b - t).abs()
            );
        }
        Ok(())
    }

    macro_rules! gen_added_dma_tests {
        ($($f:ident),*) => {
            paste::paste! {
                $(
                    #[test] fn [<$f _scalar>]() -> Result<(), Box<dyn Error>> {
                        $f(stringify!([<$f _scalar>]), Kernel::Scalar)
                    }
                    #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                    #[test] fn [<$f _avx2>]() -> Result<(), Box<dyn Error>> {
                        $f(stringify!([<$f _avx2>]), Kernel::Avx2)
                    }
                    #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                    #[test] fn [<$f _avx512>]() -> Result<(), Box<dyn Error>> {
                        $f(stringify!([<$f _avx512>]), Kernel::Avx512)
                    }
                )*
            }
        }
    }

    gen_added_dma_tests!(check_dma_reinput, check_dma_nan_handling);

    macro_rules! gen_batch_sweep_tests {
        ($($f:ident),*) => {
            paste::paste! {
                $(
                    #[test] fn [<$f _scalar_batch>]() -> Result<(), Box<dyn Error>> {
                        $f(stringify!([<$f _scalar_batch>]), Kernel::ScalarBatch)
                    }
                    #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                    #[test] fn [<$f _avx2_batch>]() -> Result<(), Box<dyn Error>> {
                        $f(stringify!([<$f _avx2_batch>]), Kernel::Avx2Batch)
                    }
                    #[cfg(all(feature="nightly-avx", target_arch="x86_64"))]
                    #[test] fn [<$f _avx512_batch>]() -> Result<(), Box<dyn Error>> {
                        $f(stringify!([<$f _avx512_batch>]), Kernel::Avx512Batch)
                    }
                )*
            }
        }
    }

    gen_batch_sweep_tests!(check_batch_default_row);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_dma_simd128_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = DmaParams::default();
        let input = DmaInput::from_slice(&data, p);
        let scalar = dma_with_kernel(&input, Kernel::Scalar).unwrap();
        let simd = dma_with_kernel(&input, Kernel::Scalar).unwrap();
        assert_eq!(scalar.values.len(), simd.values.len());
        for (a, b) in scalar.values.iter().zip(simd.values.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_dma_no_poison_values() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = DmaInput::from_candles(&candles, "close", DmaParams::default());
        let output = dma(&input)?;

        for &v in &output.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();

            assert_ne!(
                b, 0x11111111_11111111,
                "Found poison value 0x11111111_11111111"
            );
            assert_ne!(
                b, 0x22222222_22222222,
                "Found poison value 0x22222222_22222222"
            );
            assert_ne!(
                b, 0x33333333_33333333,
                "Found poison value 0x33333333_33333333"
            );
            assert_ne!(
                b, 0xDEADBEEF_DEADBEEF,
                "Found poison value 0xDEADBEEF_DEADBEEF"
            );
            assert_ne!(
                b, 0xFEEEFEEE_FEEEFEEE,
                "Found poison value 0xFEEEFEEE_FEEEFEEE"
            );
        }
        Ok(())
    }

    fn check_dma_batch_basic(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];

        let sweep = DmaBatchRange {
            hull_length: (3, 5, 1),
            ema_length: (5, 5, 0),
            ema_gain_limit: (10, 10, 0),
            hull_ma_type: "WMA".to_string(),
        };
        let output = dma_batch_with_kernel(&data, &sweep, kernel)?;

        assert_eq!(
            output.rows, 3,
            "[{}] Expected 3 rows for hull_length range 3-5",
            test_name
        );
        assert_eq!(output.cols, data.len());
        assert_eq!(output.values.len(), output.rows * output.cols);
        assert_eq!(output.combos.len(), output.rows);

        Ok(())
    }

    #[test]
    fn test_dma_stream_incremental() -> Result<(), Box<dyn Error>> {
        let params = DmaParams {
            hull_length: Some(3),
            ema_length: Some(3),
            ema_gain_limit: Some(10),
            hull_ma_type: Some("WMA".to_string()),
        };

        let mut stream = DmaStream::try_new(params.clone())?;
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let mut results = Vec::new();
        for &val in &data {
            if let Some(result) = stream.update(val) {
                results.push(result);
            }
        }

        assert!(
            !results.is_empty(),
            "Stream should produce results after warmup"
        );

        Ok(())
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_dma_batch_no_poison_values() -> Result<(), Box<dyn std::error::Error>> {
        use crate::utilities::data_loader::read_candles_from_vortex;
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out = DmaBatchBuilder::new()
            .hull_length_range(3, 8, 1)
            .ema_length_range(5, 10, 1)
            .ema_gain_limit_static(10)
            .apply_slice(&c.close)?;
        for &v in &out.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(b, 0x11111111_11111111);
            assert_ne!(b, 0x22222222_22222222);
            assert_ne!(b, 0x33333333_33333333);
        }
        Ok(())
    }

    #[test]
    fn test_dma_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = Vec::with_capacity(160);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..157 {
            let x = (i as f64 * 0.15).sin() * 5.0 + (i as f64) * 0.01;
            data.push(x);
        }

        let input = DmaInput::from_slice(&data, DmaParams::default());

        let baseline = dma(&input)?;

        let mut out = vec![0.0; data.len()];
        {
            dma_into(&input, &mut out)?;
        }

        assert_eq!(baseline.values.len(), out.len());

        for (a, b) in baseline.values.iter().copied().zip(out.iter().copied()) {
            let both_nan = a.is_nan() && b.is_nan();
            assert!(both_nan || a == b, "mismatch: got {b:?}, expected {a:?}");
        }
        Ok(())
    }

    fn reviewed_routeable_close_v3(rows: usize) -> Vec<f64> {
        let mut close = Vec::with_capacity(rows);
        for row in 0..rows {
            let drift = row as f64 * 0.000_000_7;
            let wave = match row % 11 {
                0 => 0.000_041,
                1 => -0.000_027,
                2 => 0.000_013,
                3 => -0.000_036,
                4 => 0.000_022,
                5 => -0.000_009,
                6 => 0.000_033,
                7 => -0.000_019,
                8 => 0.000_006,
                9 => -0.000_031,
                _ => 0.000_017,
            };
            close.push(1.075 + drift + wave);
        }
        let final_row = rows - 1;
        close[final_row] = f64::from_bits(close[final_row].to_bits() ^ 1);
        close
    }

    #[test]
    fn dma_f64_reviewed_route_uses_one_chronological_schedule() -> Result<(), Box<dyn Error>> {
        let close = reviewed_routeable_close_v3(4_096);
        let sweep = DmaBatchRange {
            hull_length: (14, 14, 0),
            ema_length: (20, 20, 0),
            ema_gain_limit: (50, 50, 0),
            hull_ma_type: "WMA".to_string(),
        };
        let scalar = dma_batch_with_kernel(&close, &sweep, Kernel::ScalarBatch)?;
        let auto = dma_batch_with_kernel(&close, &sweep, Kernel::Auto)?;

        // Independent exact-rational evaluation of the binary64 fixture and
        // Dickson/Hull equations rounds row 21 to 0x3ff13340e42dc558.
        // The frozen chronological RN schedule yields ...55f (6.704 ULP from
        // that real-valued oracle); the old AVX horizontal reduction yielded
        // ...560 (7.704 ULP away) and was the RTX fixture's first mismatch.
        const CHRONOLOGICAL_ROW_21_BITS: u64 = 0x3ff1_3340_e42d_c55f;
        assert_eq!(scalar.values[21].to_bits(), CHRONOLOGICAL_ROW_21_BITS);
        assert_eq!(auto.values[21].to_bits(), CHRONOLOGICAL_ROW_21_BITS);
        assert_eq!(
            scalar
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            auto.values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn dma_f64_uses_creator_symmetric_gain_domain_and_stream_is_exact() -> Result<(), Box<dyn Error>>
    {
        let data = (0..80)
            .map(|row| {
                if row < 30 {
                    row as f64
                } else {
                    (60 - row) as f64
                }
            })
            .collect::<Vec<_>>();
        let params = DmaParams {
            hull_length: Some(7),
            ema_length: Some(20),
            ema_gain_limit: Some(50),
            hull_ma_type: Some("WMA".to_string()),
        };
        let scalar = dma_with_kernel(&DmaInput::from_slice(&data, params.clone()), Kernel::Scalar)?;

        // Nathan Dickson's original loop is -gain_limit..=+gain_limit.
        // On this exact-integer reversal, row 33 selects -5.0. Both an
        // exhaustive high-precision oracle and the frozen binary64 schedule
        // round the final DMA value to this bit pattern. The old one-sided
        // 0..=limit implementation selected 0.0 and returned ...26e8.
        const CREATOR_SYMMETRIC_ROW_33_BITS: u64 = 0x403b_8ddc_7c4e_0dc3;
        assert_eq!(scalar.values[33].to_bits(), CREATOR_SYMMETRIC_ROW_33_BITS);

        let mut stream = DmaStream::try_new(params.clone())?;
        let streamed = data
            .iter()
            .map(|&value| stream.update(value).unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        assert_eq!(streamed[33].to_bits(), CREATOR_SYMMETRIC_ROW_33_BITS);
        assert_eq!(
            scalar
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            streamed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        // A second, continuous piecewise-linear fixture selects the interior
        // creator candidate -3.3 at row 21 (not merely the -5.0 clamp). The
        // exact-rational formula over these binary64 inputs rounds to
        // 0x4000cfe9b0ef58cd; the canonical binary64 schedule is two ULP above
        // it. The old one-sided route selected 0.0 and returned ...6376.
        let interior_negative = (0..80)
            .map(|row| {
                if row < 20 {
                    row as f64 * 0.15
                } else {
                    19.0 * 0.15 - (row - 19) as f64 * 0.5
                }
            })
            .collect::<Vec<_>>();
        let interior_scalar = dma_with_kernel(
            &DmaInput::from_slice(&interior_negative, params.clone()),
            Kernel::Scalar,
        )?;
        const INTERIOR_NEGATIVE_ROW_21_BITS: u64 = 0x4000_cfe9_b0ef_58cf;
        assert_eq!(
            interior_scalar.values[21].to_bits(),
            INTERIOR_NEGATIVE_ROW_21_BITS
        );

        // Preserve the creator's `value1 / 10` binary64 construction too;
        // `-33 * 0.1` is the adjacent, less accurate bit pattern ...6667.
        assert_eq!(
            dma_quantized_best_gain_f64(-3.3, 1.0, 50).to_bits(),
            0xc00a_6666_6666_6666
        );

        let mut interior_stream = DmaStream::try_new(params)?;
        let interior_streamed = interior_negative
            .iter()
            .map(|&value| interior_stream.update(value).unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        assert_eq!(
            interior_scalar
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            interior_streamed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }
}
