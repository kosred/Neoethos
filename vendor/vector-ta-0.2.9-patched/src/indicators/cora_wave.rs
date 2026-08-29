use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

/// The published RedK compound-ratio CoRa Wave semantics implemented here.
///
/// The original open-source publication defines weights that increase by one
/// constant compound ratio from the oldest sample to the newest sample. The
/// exact row returned by [`cora_wave_exact_weight_row`] is the sole coefficient
/// authority for scalar, batch, streaming, and CUDA execution.
pub const CORA_WAVE_REDK_COMPOUND_RATIO_SEMANTICS_V1: &str =
    "https://www.tradingview.com/script/NgLjvBWA-RedK-Compound-Ratio-Moving-Average-CoRa-Wave/";
pub const CORA_WAVE_REDK_OPEN_SOURCE_V3: &str = "https://pine-facade.tradingview.com/pine-facade/get/PUB%3BpOgLeMa347Zkml8worsHeXu8bvuxiHrh/last";

impl<'a> AsRef<[f64]> for CoraWaveInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CoraWaveData::Slice(slice) => slice,
            CoraWaveData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CoraWaveData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CoraWaveOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CoraWaveParams {
    pub period: Option<usize>,
    pub r_multi: Option<f64>,
    pub smooth: Option<bool>,
}

impl Default for CoraWaveParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            r_multi: Some(2.0),
            smooth: Some(true),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoraWaveInput<'a> {
    pub data: CoraWaveData<'a>,
    pub params: CoraWaveParams,
}

impl<'a> CoraWaveInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CoraWaveParams) -> Self {
        Self {
            data: CoraWaveData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CoraWaveParams) -> Self {
        Self {
            data: CoraWaveData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CoraWaveParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }

    #[inline]
    pub fn get_r_multi(&self) -> f64 {
        self.params.r_multi.unwrap_or(2.0)
    }

    #[inline]
    pub fn get_smooth(&self) -> bool {
        self.params.smooth.unwrap_or(true)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CoraWaveBuilder {
    period: Option<usize>,
    r_multi: Option<f64>,
    smooth: Option<bool>,
    kernel: Kernel,
}

impl Default for CoraWaveBuilder {
    fn default() -> Self {
        Self {
            period: None,
            r_multi: None,
            smooth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CoraWaveBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, val: usize) -> Self {
        self.period = Some(val);
        self
    }

    #[inline(always)]
    pub fn r_multi(mut self, val: f64) -> Self {
        self.r_multi = Some(val);
        self
    }

    #[inline(always)]
    pub fn smooth(mut self, val: bool) -> Self {
        self.smooth = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CoraWaveOutput, CoraWaveError> {
        let p = CoraWaveParams {
            period: self.period,
            r_multi: self.r_multi,
            smooth: self.smooth,
        };
        let i = CoraWaveInput::from_candles(c, "close", p);
        cora_wave_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CoraWaveOutput, CoraWaveError> {
        let p = CoraWaveParams {
            period: self.period,
            r_multi: self.r_multi,
            smooth: self.smooth,
        };
        let i = CoraWaveInput::from_slice(d, p);
        cora_wave_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CoraWaveStream, CoraWaveError> {
        let p = CoraWaveParams {
            period: self.period,
            r_multi: self.r_multi,
            smooth: self.smooth,
        };
        CoraWaveStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CoraWaveError {
    #[error("cora_wave: Input data slice is empty.")]
    EmptyInputData,

    #[error("cora_wave: All values are NaN.")]
    AllValuesNaN,

    #[error("cora_wave: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("cora_wave: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("cora_wave: Invalid r_multi: {value}")]
    InvalidRMulti { value: f64 },

    #[error("cora_wave: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("cora_wave: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: f64, end: f64, step: f64 },

    #[error("cora_wave: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("cora_wave: invalid input: {0}")]
    InvalidInput(String),
}

/// Builds one exact oldest-to-newest RedK compound-ratio weight row.
///
/// The repeated multiplication is intentional. Replacing it with per-element
/// `powi` calls changes coefficient bits and causes scalar, batch, streaming,
/// and CUDA paths to implement different filters.
pub(crate) fn cora_wave_exact_weight_row(
    period: usize,
    r_multi: f64,
) -> Result<(Vec<f64>, f64), CoraWaveError> {
    let _semantics = CORA_WAVE_REDK_COMPOUND_RATIO_SEMANTICS_V1;
    if period == 0 {
        return Err(CoraWaveError::InvalidPeriod {
            period,
            data_len: 0,
        });
    }
    if r_multi < 0.0 || !r_multi.is_finite() {
        return Err(CoraWaveError::InvalidRMulti { value: r_multi });
    }
    if period == 1 {
        return Ok((vec![1.0], 1.0));
    }

    let start_wt = 0.01;
    let end_wt = period as f64;
    let r = (end_wt / start_wt).powf(1.0 / (period as f64 - 1.0)) - 1.0;
    let base = 1.0 + r * r_multi;
    if !base.is_finite() || base <= 0.0 {
        return Err(CoraWaveError::InvalidInput(format!(
            "non-finite compound-ratio base for period={period}, r_multi={r_multi}"
        )));
    }

    let mut weights = Vec::with_capacity(period);
    let mut sum = 0.0;
    let mut w = start_wt * base;
    for _ in 0..period {
        weights.push(w);
        sum += w;
        w *= base;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(CoraWaveError::InvalidInput(format!(
            "non-finite compound-ratio weight sum for period={period}, r_multi={r_multi}"
        )));
    }
    Ok((weights, 1.0 / sum))
}

/// Advances the mathematically equivalent rolling weighted sum with an
/// explicit cross-device contraction order.
#[inline(always)]
fn cora_wave_roll_forward_exact_v1(
    previous: f64,
    inv_r: f64,
    a_old: f64,
    x_old: f64,
    w_last: f64,
    x_new: f64,
) -> f64 {
    let without_old = (-a_old).mul_add(x_old, previous * inv_r);
    w_last.mul_add(x_new, without_old)
}

#[inline]
pub fn cora_wave(input: &CoraWaveInput) -> Result<CoraWaveOutput, CoraWaveError> {
    cora_wave_with_kernel(input, Kernel::Auto)
}

pub fn cora_wave_with_kernel(
    input: &CoraWaveInput,
    kernel: Kernel,
) -> Result<CoraWaveOutput, CoraWaveError> {
    let (data, weights, inv_wsum, smooth_period, first, chosen) = cora_wave_prepare(input, kernel)?;
    let period = weights.len();
    let warm = first + period - 1 + smooth_period.saturating_sub(1);

    let mut out = alloc_with_nan_prefix(data.len(), warm);
    cora_wave_compute_into(
        data,
        &weights,
        inv_wsum,
        smooth_period,
        first,
        chosen,
        &mut out,
    );
    Ok(CoraWaveOutput { values: out })
}

#[inline]
pub fn cora_wave_into(input: &CoraWaveInput, out: &mut [f64]) -> Result<(), CoraWaveError> {
    let (data, weights, inv_wsum, smooth_period, first, chosen) =
        cora_wave_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(CoraWaveError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let warm = first + weights.len() - 1 + smooth_period.saturating_sub(1);
    let warm = warm.min(out.len());
    if warm > 0 {
        let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
        for v in &mut out[..warm] {
            *v = qnan;
        }
    }

    cora_wave_compute_into(data, &weights, inv_wsum, smooth_period, first, chosen, out);
    Ok(())
}

#[inline]
pub fn cora_wave_into_slice(
    dst: &mut [f64],
    input: &CoraWaveInput,
    kern: Kernel,
) -> Result<(), CoraWaveError> {
    let (data, weights, inv_wsum, smooth_period, first, chosen) = cora_wave_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(CoraWaveError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    cora_wave_compute_into(data, &weights, inv_wsum, smooth_period, first, chosen, dst);

    let warm = first + weights.len() - 1 + smooth_period.saturating_sub(1);
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
fn cora_wave_prepare<'a>(
    input: &'a CoraWaveInput,
    kernel: Kernel,
) -> Result<(&'a [f64], Vec<f64>, f64, usize, usize, Kernel), CoraWaveError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CoraWaveError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoraWaveError::AllValuesNaN)?;

    let period = input.get_period();
    let r_multi = input.get_r_multi();
    let smooth = input.get_smooth();

    if period == 0 || period > len {
        return Err(CoraWaveError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CoraWaveError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if r_multi < 0.0 || !r_multi.is_finite() {
        return Err(CoraWaveError::InvalidRMulti { value: r_multi });
    }

    let (weights, inv_sum) = cora_wave_exact_weight_row(period, r_multi)?;

    let smooth_period = if smooth {
        ((period as f64).sqrt().round() as usize).max(1)
    } else {
        1
    };
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, weights, inv_sum, smooth_period, first, chosen))
}

#[inline(always)]
fn cora_wave_compute_into(
    data: &[f64],
    weights: &[f64],
    inv_wsum: f64,
    smooth_period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                cora_wave_scalar_with_weights(data, weights, inv_wsum, smooth_period, first, out);
                return;
            }
        }
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cora_wave_scalar_with_weights(data, weights, inv_wsum, smooth_period, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                cora_wave_scalar_with_weights(data, weights, inv_wsum, smooth_period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn cora_wave_scalar_with_weights(
    data: &[f64],
    weights: &[f64],
    inv_wsum: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    let n = data.len();
    let p = weights.len();
    if p == 0 || n == 0 {
        return;
    }

    if smooth_period == 1 {
        if p == 1 {
            let start = first_val;
            if start < n {
                for i in start..n {
                    unsafe {
                        let v = *data.get_unchecked(i);
                        *out.get_unchecked_mut(i) = v * inv_wsum;
                    }
                }
            }
            return;
        }

        let w0 = unsafe { *weights.get_unchecked(0) };
        let w1 = unsafe { *weights.get_unchecked(1) };
        let inv_R = w0 / w1;
        let a_old = w0 * inv_R;
        let w_last = unsafe { *weights.get_unchecked(p - 1) };

        let warm0 = first_val + p - 1;
        if warm0 >= n {
            return;
        }
        let start0 = warm0 + 1 - p;

        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;
        let mut j = 0usize;
        let end4 = p & !3usize;

        unsafe {
            let xptr = data.as_ptr().add(start0);
            let wptr = weights.as_ptr();
            while j < end4 {
                let x0 = *xptr.add(j);
                let x1 = *xptr.add(j + 1);
                let x2 = *xptr.add(j + 2);
                let x3 = *xptr.add(j + 3);

                let y0 = *wptr.add(j);
                let y1 = *wptr.add(j + 1);
                let y2 = *wptr.add(j + 2);
                let y3 = *wptr.add(j + 3);

                acc0 = x0.mul_add(y0, acc0);
                acc1 = x1.mul_add(y1, acc1);
                acc2 = x2.mul_add(y2, acc2);
                acc3 = x3.mul_add(y3, acc3);

                j += 4;
            }
            let mut S = (acc0 + acc1) + (acc2 + acc3);
            while j < p {
                let x = *xptr.add(j);
                let y = *wptr.add(j);
                S = x.mul_add(y, S);
                j += 1;
            }

            *out.get_unchecked_mut(warm0) = S * inv_wsum;

            let mut i = warm0;
            while i + 1 < n {
                let x_old = *data.get_unchecked(i + 1 - p);
                let x_new = *data.get_unchecked(i + 1);
                S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
                *out.get_unchecked_mut(i + 1) = S * inv_wsum;
                i += 1;
            }
        }
        return;
    }

    let m = smooth_period;
    let wma_sum = (m as f64) * ((m as f64) + 1.0) * 0.5;

    if p == 1 {
        let warm0 = first_val;
        if warm0 >= n {
            return;
        }

        let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
        if n < 100_000 {
            let mut head = 0usize;
            let warm_total = warm0 + m - 1;
            unsafe {
                for i in warm0..n {
                    ring_mu
                        .get_unchecked_mut(head)
                        .write(*data.get_unchecked(i));
                    head = (head + 1) % m;

                    if i >= warm_total {
                        let mut acc = 0.0;
                        for k in 0..m {
                            let idx = (head + k) % m;
                            let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                            acc += v * ((k + 1) as f64);
                        }
                        *out.get_unchecked_mut(i) = acc / wma_sum;
                    }
                }
            }
            return;
        }

        let mut fill = 0usize;
        let warm_total = warm0 + m - 1;
        unsafe {
            let mut i = warm0;
            while i <= warm_total && i < n {
                ring_mu
                    .get_unchecked_mut(fill)
                    .write(*data.get_unchecked(i));
                fill += 1;
                i += 1;
            }
            if warm_total >= n {
                return;
            }

            let mut ssum = 0.0;
            let mut wsum = 0.0;
            for k in 0..m {
                let v = *ring_mu.get_unchecked(k).assume_init_ref();
                ssum += v;
                wsum += v * ((k + 1) as f64);
            }
            let mut head = 0usize;
            let mut t = warm_total;
            *out.get_unchecked_mut(t) = wsum / wma_sum;

            while t + 1 < n {
                let y_old = *ring_mu.get_unchecked(head).assume_init_ref();
                let y_new = *data.get_unchecked(t + 1);

                wsum = wsum - ssum + (m as f64) * y_new;

                ring_mu.get_unchecked_mut(head).write(y_new);
                ssum = ssum + y_new - y_old;
                head = (head + 1) % m;

                *out.get_unchecked_mut(t + 1) = wsum / wma_sum;
                t += 1;
            }
        }
        return;
    }

    let w0 = unsafe { *weights.get_unchecked(0) };
    let w1 = unsafe { *weights.get_unchecked(1) };
    let inv_R = w0 / w1;
    let a_old = w0 * inv_R;
    let w_last = unsafe { *weights.get_unchecked(p - 1) };

    let warm0 = first_val + p - 1;
    if warm0 >= n {
        return;
    }
    let start0 = warm0 + 1 - p;

    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut j = 0usize;
    let end4 = p & !3usize;

    unsafe {
        let xptr = data.as_ptr().add(start0);
        let wptr = weights.as_ptr();
        while j < end4 {
            let x0 = *xptr.add(j);
            let x1 = *xptr.add(j + 1);
            let x2 = *xptr.add(j + 2);
            let x3 = *xptr.add(j + 3);

            let y0 = *wptr.add(j);
            let y1 = *wptr.add(j + 1);
            let y2 = *wptr.add(j + 2);
            let y3 = *wptr.add(j + 3);

            acc0 = x0.mul_add(y0, acc0);
            acc1 = x1.mul_add(y1, acc1);
            acc2 = x2.mul_add(y2, acc2);
            acc3 = x3.mul_add(y3, acc3);

            j += 4;
        }
        let mut S = (acc0 + acc1) + (acc2 + acc3);
        while j < p {
            let x = *xptr.add(j);
            let y = *wptr.add(j);
            S = x.mul_add(y, S);
            j += 1;
        }

        let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
        let mut fill = 0usize;

        let mut y = S * inv_wsum;
        ring_mu.get_unchecked_mut(fill).write(y);
        fill += 1;

        let warm_total = warm0 + m - 1;
        let mut i = warm0;
        while i + 1 <= warm_total && i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
            y = S * inv_wsum;
            ring_mu.get_unchecked_mut(fill).write(y);
            fill += 1;
            i += 1;
        }
        if warm_total >= n {
            return;
        }

        if n < 100_000 {
            let mut head = 0usize;

            {
                let mut acc = 0.0;
                for k in 0..m {
                    let idx = (head + k) % m;
                    let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                    acc += v * ((k + 1) as f64);
                }
                *out.get_unchecked_mut(warm_total) = acc / wma_sum;
            }

            while i + 1 < n {
                let x_old = *data.get_unchecked(i + 1 - p);
                let x_new = *data.get_unchecked(i + 1);
                S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
                let y_new = S * inv_wsum;

                ring_mu.get_unchecked_mut(head).write(y_new);
                head = (head + 1) % m;

                let mut acc = 0.0;
                for k in 0..m {
                    let idx = (head + k) % m;
                    let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                    acc += v * ((k + 1) as f64);
                }
                *out.get_unchecked_mut(i + 1) = acc / wma_sum;
                i += 1;
            }
            return;
        }

        let mut head = 0usize;
        let mut ssum = 0.0;
        let mut wsum = 0.0;
        for k in 0..m {
            let v = *ring_mu.get_unchecked(k).assume_init_ref();
            ssum += v;
            wsum += v * ((k + 1) as f64);
        }
        *out.get_unchecked_mut(warm_total) = wsum / wma_sum;

        while i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
            let y_new = S * inv_wsum;

            wsum = wsum - ssum + (m as f64) * y_new;

            let y_old = *ring_mu.get_unchecked(head).assume_init_ref();
            ring_mu.get_unchecked_mut(head).write(y_new);
            ssum = ssum + y_new - y_old;
            head = (head + 1) % m;

            *out.get_unchecked_mut(i + 1) = wsum / wma_sum;
            i += 1;
        }
    }
}

#[inline]
pub fn cora_wave_scalar(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    let Ok((weights, inv_wsum)) = cora_wave_exact_weight_row(period, r_multi) else {
        out.fill(f64::NAN);
        return;
    };

    cora_wave_scalar_with_weights(data, &weights, inv_wsum, smooth_period, first_val, out);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn cora_wave_simd128(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;

    cora_wave_scalar(data, period, r_multi, smooth_period, first_val, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn cora_wave_avx2(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    cora_wave_scalar(data, period, r_multi, smooth_period, first_val, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn cora_wave_avx512(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    cora_wave_scalar(data, period, r_multi, smooth_period, first_val, out);
}

#[derive(Debug, Clone)]
pub struct CoraWaveStream {
    period: usize,
    r_multi: f64,
    smooth: bool,
    smooth_period: usize,

    weights: Vec<f64>,
    inv_R: f64,
    a_old: f64,
    w_last: f64,
    inv_wsum: f64,

    ring_x: Vec<f64>,
    head_x: usize,
    idx: usize,
    have_S: bool,
    S: f64,

    m: usize,
    wma_sum: f64,
    ring_y: Vec<f64>,
    head_y: usize,
    y_count: usize,

    Ssum_y: f64,
    Wsum_y: f64,

    fast_smooth: bool,
}

impl CoraWaveStream {
    #[inline]
    pub fn try_new(params: CoraWaveParams) -> Result<Self, CoraWaveError> {
        let period = params.period.unwrap_or(20);
        let r_multi = params.r_multi.unwrap_or(2.0);
        let smooth = params.smooth.unwrap_or(true);

        if period == 0 {
            return Err(CoraWaveError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if r_multi < 0.0 || !r_multi.is_finite() {
            return Err(CoraWaveError::InvalidRMulti { value: r_multi });
        }

        let m = if smooth {
            ((period as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };

        let p = period;
        let (weights, inv_wsum) = cora_wave_exact_weight_row(p, r_multi)?;
        let (inv_R, a_old, w_last) = if p == 1 {
            (1.0, 1.0, 1.0)
        } else {
            let inv_r = weights[0] / weights[1];
            (inv_r, weights[0] * inv_r, weights[p - 1])
        };

        let wma_sum = (m as f64) * ((m as f64) + 1.0) * 0.5;

        const FAST_WMA_O1_DEFAULT: bool = false;

        Ok(Self {
            period: p,
            r_multi,
            smooth,
            smooth_period: m,
            weights,
            inv_R,
            a_old,
            w_last,
            inv_wsum,
            ring_x: vec![0.0; p],
            head_x: 0,
            idx: 0,
            have_S: false,
            S: 0.0,
            m,
            wma_sum,
            ring_y: vec![0.0; m.max(1)],
            head_y: 0,
            y_count: 0,
            Ssum_y: 0.0,
            Wsum_y: 0.0,
            fast_smooth: FAST_WMA_O1_DEFAULT,
        })
    }

    #[inline]
    pub fn update(&mut self, x_new: f64) -> Option<f64> {
        let pos = self.head_x;
        let x_old = self.ring_x[pos];
        self.ring_x[pos] = x_new;
        self.head_x = (pos + 1) % self.period;
        self.idx += 1;

        if !self.have_S {
            if self.idx < self.period {
                return None;
            }

            let mut S = 0.0;
            let mut i = 0usize;
            while i < self.period {
                let xi = self.ring_x[(self.head_x + i) % self.period];
                S = xi.mul_add(self.weights[i], S);
                i += 1;
            }
            self.S = S;
            self.have_S = true;

            let y = self.S * self.inv_wsum;
            if self.m == 1 {
                return Some(y);
            }

            self.ring_y[self.y_count] = y;
            self.y_count += 1;

            self.head_y = self.y_count % self.m;
            if self.fast_smooth {
                self.Ssum_y += y;
                self.Wsum_y += (self.y_count as f64) * y;
            }
            return None;
        }

        self.S = cora_wave_roll_forward_exact_v1(
            self.S,
            self.inv_R,
            self.a_old,
            x_old,
            self.w_last,
            x_new,
        );
        let y = self.S * self.inv_wsum;

        if self.m == 1 {
            return Some(y);
        }

        if !self.fast_smooth {
            if self.y_count < self.m {
                self.ring_y[self.head_y] = y;
                self.head_y = (self.head_y + 1) % self.m;
                self.y_count += 1;
                if self.y_count < self.m {
                    return None;
                }

                let mut acc = 0.0;
                let mut k = 0usize;
                while k < self.m {
                    let idx = (self.head_y + k) % self.m;
                    let v = self.ring_y[idx];
                    acc = v.mul_add((k + 1) as f64, acc);
                    k += 1;
                }
                return Some(acc / self.wma_sum);
            } else {
                self.ring_y[self.head_y] = y;
                self.head_y = (self.head_y + 1) % self.m;
                let mut acc = 0.0;
                let mut k = 0usize;
                while k < self.m {
                    let idx = (self.head_y + k) % self.m;
                    let v = self.ring_y[idx];
                    acc = v.mul_add((k + 1) as f64, acc);
                    k += 1;
                }
                return Some(acc / self.wma_sum);
            }
        } else {
            if self.y_count < self.m {
                self.ring_y[self.y_count] = y;
                self.y_count += 1;
                self.Ssum_y += y;
                self.Wsum_y += (self.y_count as f64) * y;
                if self.y_count < self.m {
                    return None;
                }

                self.head_y = 0;
                return Some(self.Wsum_y / self.wma_sum);
            }

            let y_old = self.ring_y[self.head_y];

            self.Wsum_y = self.Wsum_y - self.Ssum_y + (self.m as f64) * y;
            self.ring_y[self.head_y] = y;
            self.Ssum_y = self.Ssum_y + y - y_old;
            self.head_y = (self.head_y + 1) % self.m;

            Some(self.Wsum_y / self.wma_sum)
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoraWaveBatchRange {
    pub period: (usize, usize, usize),
    pub r_multi: (f64, f64, f64),
    pub smooth: bool,
}

impl Default for CoraWaveBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
            r_multi: (2.0, 2.249, 0.001),
            smooth: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoraWaveBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CoraWaveParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CoraWaveBatchOutput {
    pub fn row_for_params(&self, p: &CoraWaveParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.r_multi.unwrap_or(2.0) - p.r_multi.unwrap_or(2.0)).abs() < 1e-12
                && c.smooth.unwrap_or(true) == p.smooth.unwrap_or(true)
        })
    }

    pub fn values_for(&self, p: &CoraWaveParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_usize((s, e, t): (usize, usize, usize)) -> Result<Vec<usize>, CoraWaveError> {
    if t == 0 || s == e {
        return Ok(vec![s]);
    }
    let mut v = Vec::new();
    if s < e {
        v = (s..=e).step_by(t).collect();
    } else if s > e {
        let mut x = s;
        loop {
            v.push(x);
            if x <= e {
                break;
            }
            if x < t {
                break;
            }
            let next = x - t;
            if next < e {
                break;
            }
            x = next;
        }

        if *v.last().unwrap_or(&s) != e && s >= e {}
    }
    if v.is_empty() {
        return Err(CoraWaveError::InvalidRange {
            start: s as f64,
            end: e as f64,
            step: t as f64,
        });
    }
    Ok(v)
}
#[inline(always)]
fn axis_f64((s, e, t): (f64, f64, f64)) -> Result<Vec<f64>, CoraWaveError> {
    if t.abs() < 1e-12 || (s - e).abs() < 1e-12 {
        return Ok(vec![s]);
    }
    let step = t.abs();
    let mut v = Vec::new();
    if s <= e {
        let mut x = s;
        while x <= e + 1e-12 {
            v.push(x);
            x += step;
        }
    } else {
        let mut x = s;
        while x >= e - 1e-12 {
            v.push(x);
            x -= step;
        }
    }
    if v.is_empty() {
        return Err(CoraWaveError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }
    Ok(v)
}
#[inline(always)]
fn expand_grid_cw(r: &CoraWaveBatchRange) -> Result<Vec<CoraWaveParams>, CoraWaveError> {
    let periods = axis_usize(r.period)?;
    let mults = axis_f64(r.r_multi)?;
    if periods.is_empty() || mults.is_empty() {
        return Err(CoraWaveError::InvalidRange {
            start: r.period.0 as f64,
            end: r.period.1 as f64,
            step: r.period.2 as f64,
        });
    }
    let cap = periods
        .len()
        .checked_mul(mults.len())
        .ok_or_else(|| CoraWaveError::InvalidInput("periods*mults overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &m in &mults {
            out.push(CoraWaveParams {
                period: Some(p),
                r_multi: Some(m),
                smooth: Some(r.smooth),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn cora_wave_batch_slice(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    cora_wave_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cora_wave_batch_par_slice(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    cora_wave_batch_inner(data, sweep, kern, true)
}

pub fn cora_wave_batch_with_kernel(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    k: Kernel,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CoraWaveError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cora_wave_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn cora_wave_batch_inner(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    let combos = expand_grid_cw(sweep)?;
    if combos.is_empty() {
        return Err(CoraWaveError::InvalidRange {
            start: sweep.period.0 as f64,
            end: sweep.period.1 as f64,
            step: sweep.period.2 as f64,
        });
    }

    let cols = data.len();
    if cols == 0 {
        return Err(CoraWaveError::AllValuesNaN);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoraWaveError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(CoraWaveError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let rows = combos.len();
    let _total = rows
        .checked_mul(cols)
        .ok_or_else(|| CoraWaveError::InvalidInput("rows*cols overflow".into()))?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warms: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let sp = if c.smooth.unwrap_or(true) {
                ((p as f64).sqrt().round() as usize).max(1)
            } else {
                1
            };
            first + p - 1 + sp.saturating_sub(1)
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    let flat_len = rows
        .checked_mul(max_p)
        .ok_or_else(|| CoraWaveError::InvalidInput("rows*max_period overflow".into()))?;
    let mut flat_w = vec![0.0f64; flat_len];
    let mut inv_sums = vec![0.0f64; rows];

    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        let r_multi = prm.r_multi.unwrap();
        let (weights, inv_wsum) = cora_wave_exact_weight_row(p, r_multi)?;
        let start = row * max_p;
        flat_w[start..start + p].copy_from_slice(&weights);
        inv_sums[row] = inv_wsum;
    }

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_uninit: &mut [MaybeUninit<f64>] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let sp = if combos[row].smooth.unwrap_or(true) {
            ((p as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };
        let w_ptr = flat_w[row * max_p..].as_ptr();
        let inv = inv_sums[row];

        let dst = unsafe { core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols) };

        match actual {
            Kernel::Scalar
            | Kernel::ScalarBatch
            | Kernel::Avx2
            | Kernel::Avx2Batch
            | Kernel::Avx512
            | Kernel::Avx512Batch => unsafe {
                cora_wave_row_scalar_with_weights(data, first, p, w_ptr, inv, sp, dst)
            },
            _ => unreachable!(),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        if parallel {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(CoraWaveBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn cora_wave_batch_inner_into(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CoraWaveParams>, CoraWaveError> {
    let combos = expand_grid_cw(sweep)?;
    if combos.is_empty() {
        return Err(CoraWaveError::InvalidRange {
            start: sweep.period.0 as f64,
            end: sweep.period.1 as f64,
            step: sweep.period.2 as f64,
        });
    }

    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(CoraWaveError::AllValuesNaN);
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| CoraWaveError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(CoraWaveError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoraWaveError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(CoraWaveError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let warms: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let sp = if c.smooth.unwrap_or(true) {
                ((p as f64).sqrt().round() as usize).max(1)
            } else {
                1
            };
            first + p - 1 + sp.saturating_sub(1)
        })
        .collect();

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warms);

    let flat_len = rows
        .checked_mul(max_p)
        .ok_or_else(|| CoraWaveError::InvalidInput("rows*max_period overflow".into()))?;
    let mut flat_w = vec![0.0f64; flat_len];
    let mut inv_sums = vec![0.0f64; rows];
    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        let r_multi = prm.r_multi.unwrap();
        let (weights, inv_wsum) = cora_wave_exact_weight_row(p, r_multi)?;
        let start = row * max_p;
        flat_w[start..start + p].copy_from_slice(&weights);
        inv_sums[row] = inv_wsum;
    }

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let sp = if combos[row].smooth.unwrap_or(true) {
            ((p as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };
        let w_ptr = flat_w[row * max_p..].as_ptr();
        let inv = inv_sums[row];

        let dst: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols) };
        match actual {
            Kernel::Scalar
            | Kernel::ScalarBatch
            | Kernel::Avx2
            | Kernel::Avx2Batch
            | Kernel::Avx512
            | Kernel::Avx512Batch => unsafe {
                cora_wave_row_scalar_with_weights(data, first, p, w_ptr, inv, sp, dst)
            },
            _ => unreachable!(),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn cora_wave_row_scalar_with_weights(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    inv_wsum: f64,
    smooth_period: usize,
    out: &mut [f64],
) {
    let n = data.len();
    let p = period;
    if p == 0 || n == 0 {
        return;
    }

    if smooth_period == 1 {
        if p == 1 {
            let warm0 = first;
            let mut i = warm0;
            while i < n {
                *out.get_unchecked_mut(i) = *data.get_unchecked(i) * inv_wsum;
                i += 1;
            }
            return;
        }

        let w0 = *w_ptr.add(0);
        let w1 = *w_ptr.add(1);
        let inv_R = w0 / w1;
        let a_old = w0 * inv_R;
        let w_last = *w_ptr.add(p - 1);

        let warm0 = first + p - 1;
        if warm0 >= n {
            return;
        }
        let start0 = warm0 + 1 - p;

        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;
        let mut j = 0usize;
        let end4 = p & !3usize;
        let xptr = data.as_ptr().add(start0);

        while j < end4 {
            let x0 = *xptr.add(j);
            let x1 = *xptr.add(j + 1);
            let x2 = *xptr.add(j + 2);
            let x3 = *xptr.add(j + 3);

            let y0 = *w_ptr.add(j);
            let y1 = *w_ptr.add(j + 1);
            let y2 = *w_ptr.add(j + 2);
            let y3 = *w_ptr.add(j + 3);

            acc0 = x0.mul_add(y0, acc0);
            acc1 = x1.mul_add(y1, acc1);
            acc2 = x2.mul_add(y2, acc2);
            acc3 = x3.mul_add(y3, acc3);

            j += 4;
        }
        let mut S = (acc0 + acc1) + (acc2 + acc3);
        while j < p {
            let x = *xptr.add(j);
            let y = *w_ptr.add(j);
            S = x.mul_add(y, S);
            j += 1;
        }

        *out.get_unchecked_mut(warm0) = S * inv_wsum;

        let mut i = warm0;
        while i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
            *out.get_unchecked_mut(i + 1) = S * inv_wsum;
            i += 1;
        }
        return;
    }

    let m = smooth_period;
    let wma_sum = (m as f64) * ((m as f64) + 1.0) * 0.5;

    if p == 1 {
        let warm0 = first;
        if warm0 >= n {
            return;
        }

        let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
        let mut fill = 0usize;

        let warm_total = warm0 + m - 1;
        let mut i = warm0;
        while i <= warm_total && i < n {
            ring_mu
                .get_unchecked_mut(fill)
                .write(*data.get_unchecked(i));
            fill += 1;
            i += 1;
        }
        if warm_total >= n {
            return;
        }

        if n < 100_000 {
            let mut head = 0usize;
            let mut t = warm_total;
            {
                let mut acc = 0.0;
                for k in 0..m {
                    let v = *ring_mu.get_unchecked(k).assume_init_ref();
                    acc += v * ((k + 1) as f64);
                }
                *out.get_unchecked_mut(t) = acc / wma_sum;
            }
            while t + 1 < n {
                ring_mu
                    .get_unchecked_mut(head)
                    .write(*data.get_unchecked(t + 1));
                head = (head + 1) % m;
                let mut acc = 0.0;
                for k in 0..m {
                    let idx = (head + k) % m;
                    let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                    acc += v * ((k + 1) as f64);
                }
                *out.get_unchecked_mut(t + 1) = acc / wma_sum;
                t += 1;
            }
            return;
        }

        let mut Ssum = 0.0;
        let mut Wsum = 0.0;
        for k in 0..m {
            let v = *ring_mu.get_unchecked(k).assume_init_ref();
            Ssum += v;
            Wsum += v * ((k + 1) as f64);
        }
        let mut head = 0usize;
        let mut t = warm_total;
        *out.get_unchecked_mut(t) = Wsum / wma_sum;

        while t + 1 < n {
            let y_old = *ring_mu.get_unchecked(head).assume_init_ref();
            let y_new = *data.get_unchecked(t + 1);

            Wsum = Wsum - Ssum + (m as f64) * y_new;

            ring_mu.get_unchecked_mut(head).write(y_new);
            Ssum = Ssum + y_new - y_old;
            head = (head + 1) % m;

            *out.get_unchecked_mut(t + 1) = Wsum / wma_sum;
            t += 1;
        }
        return;
    }

    let w0 = *w_ptr.add(0);
    let w1 = *w_ptr.add(1);
    let inv_R = w0 / w1;
    let a_old = w0 * inv_R;
    let w_last = *w_ptr.add(p - 1);

    let warm0 = first + p - 1;
    if warm0 >= n {
        return;
    }
    let start0 = warm0 + 1 - p;

    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut j = 0usize;
    let end4 = p & !3usize;
    let xptr = data.as_ptr().add(start0);
    while j < end4 {
        let x0 = *xptr.add(j);
        let x1 = *xptr.add(j + 1);
        let x2 = *xptr.add(j + 2);
        let x3 = *xptr.add(j + 3);

        let y0 = *w_ptr.add(j);
        let y1 = *w_ptr.add(j + 1);
        let y2 = *w_ptr.add(j + 2);
        let y3 = *w_ptr.add(j + 3);

        acc0 = x0.mul_add(y0, acc0);
        acc1 = x1.mul_add(y1, acc1);
        acc2 = x2.mul_add(y2, acc2);
        acc3 = x3.mul_add(y3, acc3);

        j += 4;
    }
    let mut S = (acc0 + acc1) + (acc2 + acc3);
    while j < p {
        let x = *xptr.add(j);
        let y = *w_ptr.add(j);
        S = x.mul_add(y, S);
        j += 1;
    }

    let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
    let mut fill = 0usize;

    let mut y = S * inv_wsum;
    ring_mu.get_unchecked_mut(fill).write(y);
    fill += 1;

    let warm_total = warm0 + m - 1;
    let mut i = warm0;
    while i + 1 <= warm_total && i + 1 < n {
        let x_old = *data.get_unchecked(i + 1 - p);
        let x_new = *data.get_unchecked(i + 1);
        S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
        y = S * inv_wsum;
        ring_mu.get_unchecked_mut(fill).write(y);
        fill += 1;
        i += 1;
    }
    if warm_total >= n {
        return;
    }

    if n < 100_000 {
        let mut head = 0usize;
        {
            let mut acc = 0.0;
            for k in 0..m {
                let v = *ring_mu.get_unchecked(k).assume_init_ref();
                acc += v * ((k + 1) as f64);
            }
            *out.get_unchecked_mut(warm_total) = acc / wma_sum;
        }
        while i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
            let y_new = S * inv_wsum;

            ring_mu.get_unchecked_mut(head).write(y_new);
            head = (head + 1) % m;
            let mut acc = 0.0;
            for k in 0..m {
                let idx = (head + k) % m;
                let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                acc += v * ((k + 1) as f64);
            }
            *out.get_unchecked_mut(i + 1) = acc / wma_sum;
            i += 1;
        }
        return;
    }

    let mut Ssum = 0.0;
    let mut Wsum = 0.0;
    for k in 0..m {
        let v = *ring_mu.get_unchecked(k).assume_init_ref();
        Ssum += v;
        Wsum += v * ((k + 1) as f64);
    }
    let mut head = 0usize;
    *out.get_unchecked_mut(warm_total) = Wsum / wma_sum;

    while i + 1 < n {
        let x_old = *data.get_unchecked(i + 1 - p);
        let x_new = *data.get_unchecked(i + 1);
        S = cora_wave_roll_forward_exact_v1(S, inv_R, a_old, x_old, w_last, x_new);
        let y_new = S * inv_wsum;

        Wsum = Wsum - Ssum + (m as f64) * y_new;

        let y_old = *ring_mu.get_unchecked(head).assume_init_ref();
        ring_mu.get_unchecked_mut(head).write(y_new);
        Ssum = Ssum + y_new - y_old;
        head = (head + 1) % m;

        *out.get_unchecked_mut(i + 1) = Wsum / wma_sum;
        i += 1;
    }
}

#[derive(Clone, Debug, Default)]
pub struct CoraWaveBatchBuilder {
    range: CoraWaveBatchRange,
    kernel: Kernel,
}

impl CoraWaveBatchBuilder {
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
    pub fn period_static(mut self, val: usize) -> Self {
        self.range.period = (val, val, 0);
        self
    }

    #[inline]
    pub fn r_multi_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.r_multi = (start, end, step);
        self
    }

    #[inline]
    pub fn r_multi_static(mut self, val: f64) -> Self {
        self.range.r_multi = (val, val, 0.0);
        self
    }

    #[inline]
    pub fn smooth(mut self, val: bool) -> Self {
        self.range.smooth = val;
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        cora_wave_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        CoraWaveBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        CoraWaveBatchBuilder::new().kernel(k).apply_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    #[test]
    fn exact_weight_row_matches_the_published_compound_ratio_shape() {
        assert_eq!(
            CORA_WAVE_REDK_COMPOUND_RATIO_SEMANTICS_V1,
            "https://www.tradingview.com/script/NgLjvBWA-RedK-Compound-Ratio-Moving-Average-CoRa-Wave/"
        );
        assert_eq!(
            CORA_WAVE_REDK_OPEN_SOURCE_V3,
            "https://pine-facade.tradingview.com/pine-facade/get/PUB%3BpOgLeMa347Zkml8worsHeXu8bvuxiHrh/last"
        );
        let period = 20;
        let r_multi = 2.0;
        let (weights, inv_norm) = cora_wave_exact_weight_row(period, r_multi)
            .expect("default published coefficient row must be valid");
        let start_wt = 0.01_f64;
        let end_wt = period as f64;
        let ratio_step =
            1.0 + ((end_wt / start_wt).powf(1.0 / (period as f64 - 1.0)) - 1.0) * r_multi;

        assert_eq!(weights.len(), period);
        assert!(weights.windows(2).all(|pair| pair[1] > pair[0]));
        for (index, observed) in weights.iter().copied().enumerate() {
            let published = start_wt * ratio_step.powi((index + 1) as i32);
            let relative = (observed - published).abs() / published.abs().max(1.0);
            assert!(
                relative <= 16.0 * f64::EPSILON,
                "published coefficient {index} drifted: observed={observed:#018x}, expected={published:#018x}",
                observed = observed.to_bits(),
                published = published.to_bits()
            );
        }
        let normalized_sum = weights.iter().copied().sum::<f64>() * inv_norm;
        assert!((normalized_sum - 1.0).abs() <= f64::EPSILON);

        let (singleton, singleton_inv) =
            cora_wave_exact_weight_row(1, r_multi).expect("period one is an identity filter");
        assert_eq!(singleton, vec![1.0]);
        assert_eq!(singleton_inv, 1.0);
    }

    #[test]
    fn scalar_and_batch_share_direct_short_input_smoothing_bits() {
        let data = (0..64)
            .map(|row| {
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
                1.075 + drift + wave
            })
            .collect::<Vec<_>>();
        let scalar = cora_wave_with_kernel(
            &CoraWaveInput::from_slice(
                &data,
                CoraWaveParams {
                    period: Some(14),
                    r_multi: Some(2.0),
                    smooth: Some(true),
                },
            ),
            Kernel::Scalar,
        )
        .expect("scalar CoRa Wave fixture must be valid");
        let batch = cora_wave_batch_with_kernel(
            &data,
            &CoraWaveBatchRange {
                period: (14, 14, 0),
                r_multi: (2.0, 2.0, 0.0),
                smooth: true,
            },
            Kernel::ScalarBatch,
        )
        .expect("batch CoRa Wave fixture must be valid");

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for (row, (scalar_value, batch_value)) in
            scalar.values.iter().zip(&batch.values).enumerate()
        {
            assert!(
                (scalar_value.is_nan() && batch_value.is_nan())
                    || scalar_value.to_bits() == batch_value.to_bits(),
                "CoRa Wave scalar/batch smoothing drifted at row {row}: scalar=0x{:016x}, batch=0x{:016x}",
                scalar_value.to_bits(),
                batch_value.to_bits()
            );
        }
    }

    #[test]
    fn test_cora_wave_into_matches_api() {
        let mut data = vec![f64::NAN; 5];
        for i in 0..256 {
            let x = (i as f64 * 0.03141592653589793).sin() * 10.0 + 50.0;
            data.push(x);
        }

        let input = CoraWaveInput::from_slice(&data, CoraWaveParams::default());

        let baseline = cora_wave(&input).expect("baseline cora_wave() failed");
        let mut out = vec![0.0; data.len()];
        cora_wave_into(&input, &mut out).expect("cora_wave_into() failed");

        assert_eq!(baseline.values.len(), out.len());
        for (i, (&a, &b)) in baseline.values.iter().zip(&out).enumerate() {
            let ok = if a.is_nan() && b.is_nan() {
                true
            } else if a == b {
                true
            } else {
                (a - b).abs() <= 1e-12
            };
            assert!(ok, "mismatch at index {}: {} vs {}", i, a, b);
        }
    }

    fn check_cora_wave_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CoraWaveInput::from_candles(&candles, "close", CoraWaveParams::default());
        let result = cora_wave_with_kernel(&input, kernel)?;

        let expected_last_five = [
            59248.63632114,
            59251.74238978,
            59203.36944998,
            59171.14999178,
            59053.74201623,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 0.01,
                "[{}] CoRa Wave {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_cora_wave_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = CoraWaveParams {
            period: None,
            r_multi: None,
            smooth: None,
        };
        let input = CoraWaveInput::from_candles(&candles, "close", default_params);
        let output = cora_wave_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cora_wave_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CoraWaveInput::with_default_candles(&candles);
        match input.data {
            CoraWaveData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CoraWaveData::Candles"),
        }
        let output = cora_wave_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cora_wave_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CoraWaveParams {
            period: Some(0),
            r_multi: None,
            smooth: None,
        };
        let input = CoraWaveInput::from_slice(&input_data, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CoraWaveParams {
            period: Some(10),
            r_multi: None,
            smooth: None,
        };
        let input = CoraWaveInput::from_slice(&data_small, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&single_point, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&empty, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&nan_data, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_candles(&c, "close", params.clone());
        let batch = cora_wave_with_kernel(&input, kernel)?.values;

        let mut stream = CoraWaveStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(c.close.len());
        for &px in &c.close {
            match stream.update(px) {
                Some(v) => streamed.push(v),
                None => streamed.push(f64::NAN),
            }
        }

        assert_eq!(batch.len(), streamed.len());
        for (i, (&b, &s)) in batch.iter().zip(&streamed).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let d = (b - s).abs();
            assert!(
                d < 1e-9,
                "[{}] streaming mismatch at {}: {} vs {}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }

    fn check_cora_wave_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = cora_wave_with_kernel(&CoraWaveInput::with_default_candles(&c), kernel)?.values;
        if out.len() > 240 {
            for (i, &v) in out[240..].iter().enumerate() {
                assert!(!v.is_nan(), "[{}] unexpected NaN at {}", test_name, 240 + i);
            }
        }
        Ok(())
    }

    fn check_cora_wave_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let first = cora_wave_with_kernel(
            &CoraWaveInput::from_candles(&c, "close", CoraWaveParams::default()),
            kernel,
        )?;

        let second_in = CoraWaveInput::from_slice(&first.values, CoraWaveParams::default());
        let second = cora_wave_with_kernel(&second_in, kernel)?;
        let second_ref = cora_wave_with_kernel(&second_in, Kernel::Scalar)?;

        assert_eq!(second.values.len(), first.values.len());
        for (i, (&a, &b)) in second.values.iter().zip(&second_ref.values).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-9,
                "[{}] reinput mismatch at {}: {} vs {}",
                test_name,
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = CoraWaveBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CoraWaveParams::default();
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            59248.63632114,
            59251.74238978,
            59203.36944998,
            59171.14999178,
            59053.74201623,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 0.01,
                "[{test}] default-row mismatch at idx {i}: {v}"
            );
        }
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = CoraWaveBatchBuilder::new()
            .kernel(kernel)
            .period_range(20, 60, 1)
            .r_multi_range(1.0, 3.0, 0.25)
            .apply_candles(&c, "close")?;

        let expected = 41 * 9;
        assert_eq!(out.combos.len(), expected);
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = CoraWaveBatchBuilder::new()
            .kernel(kernel)
            .period_range(20, 24, 2)
            .r_multi_range(1.5, 2.5, 0.5)
            .apply_candles(&c, "close")?;

        for (idx, &v) in out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let bits = v.to_bits();
            if bits == 0x11111111_11111111
                || bits == 0x22222222_22222222
                || bits == 0x33333333_33333333
            {
                let row = idx / out.cols;
                let col = idx % out.cols;
                let combo = &out.combos[row];
                panic!(
                    "[{}] poison value at row {} col {} (idx {}) params: period={}, r_multi={}, smooth={}",
                    test,
                    row,
                    col,
                    idx,
                    combo.period.unwrap_or(20),
                    combo.r_multi.unwrap_or(2.0),
                    combo.smooth.unwrap_or(true),
                );
            }
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_cora_wave_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let out = cora_wave_with_kernel(&CoraWaveInput::with_default_candles(&c), kernel)?.values;
        for (i, &v) in out.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                !(b == 0x11111111_11111111 || b == 0x22222222_22222222 || b == 0x33333333_33333333),
                "[{test}] poison at idx {i}"
            );
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_cora_wave_no_poison(_: &str, _: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cora_wave_tests {
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
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        };
    }

    generate_all_cora_wave_tests!(
        check_cora_wave_accuracy,
        check_cora_wave_partial_params,
        check_cora_wave_default_candles,
        check_cora_wave_zero_period,
        check_cora_wave_period_exceeds_length,
        check_cora_wave_very_small_dataset,
        check_cora_wave_empty_input,
        check_cora_wave_all_nan,
        check_cora_wave_nan_handling,
        check_cora_wave_streaming,
        check_cora_wave_reinput,
        check_cora_wave_no_poison
    );

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_cora_wave_simd128_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&data, params);
        let scalar = cora_wave_with_kernel(&input, Kernel::Scalar).unwrap();
        let simd = cora_wave_with_kernel(&input, Kernel::Scalar).unwrap();
        assert_eq!(scalar.values.len(), simd.values.len());
        for (i, (a, b)) in scalar.values.iter().zip(simd.values.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "SIMD128 mismatch at {i}: {a} vs {b}");
        }
    }

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_cora_wave_no_panic(data: Vec<f64>, period in 1usize..100) {
            let params = CoraWaveParams {
                period: Some(period),
                r_multi: Some(2.0),
                smooth: Some(true),
            };
            let input = CoraWaveInput::from_slice(&data, params);
            let _ = cora_wave(&input);
        }

        #[test]
        fn test_cora_wave_length_preservation(size in 10usize..100) {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let params = CoraWaveParams::default();
            let input = CoraWaveInput::from_slice(&data, params);

            if let Ok(output) = cora_wave(&input) {
                prop_assert_eq!(output.values.len(), size);
            }
        }
    }
}
