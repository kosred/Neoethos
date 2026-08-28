use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone, Default)]
pub struct AdParams {}

#[derive(Debug, Clone)]
pub struct AdInput<'a> {
    pub data: AdData<'a>,
    pub params: AdParams,
}

impl<'a> AdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AdParams) -> Self {
        Self {
            data: AdData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: AdParams,
    ) -> Self {
        Self {
            data: AdData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AdParams::default())
    }
}

#[derive(Debug, Clone)]
pub struct AdOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AdBuilder {
    kernel: Kernel,
}

impl AdBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<AdOutput, AdError> {
        let input = AdInput::from_candles(c, AdParams::default());
        ad_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AdOutput, AdError> {
        let input = AdInput::from_slices(high, low, close, volume, AdParams::default());
        ad_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AdStream, AdError> {
        AdStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum AdError {
    #[error("ad: candle field error: {0}")]
    CandleFieldError(String),
    #[error(
        "ad: Data length mismatch: high={high_len}, low={low_len}, close={close_len}, volume={volume_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("ad: invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ad: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ad: not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ad: empty input data")]
    EmptyInputData,
    #[error("ad: all values are NaN")]
    AllValuesNaN,
    #[error("ad: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: isize,
        end: isize,
        step: isize,
    },
    #[error("ad: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("ad: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn ad(input: &AdInput) -> Result<AdOutput, AdError> {
    ad_with_kernel(input, Kernel::Auto)
}

pub fn ad_with_kernel(input: &AdInput, kernel: Kernel) -> Result<AdOutput, AdError> {
    let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        AdData::Candles { candles } => {
            (&candles.high, &candles.low, &candles.close, &candles.volume)
        }
        AdData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let size = high.len();
    if size == 0 {
        return Err(AdError::EmptyInputData);
    }

    let mut chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kernel, Kernel::Auto) && matches!(chosen, Kernel::Avx512 | Kernel::Avx512Batch) {
        chosen = if size >= 262_144 {
            Kernel::Avx2
        } else {
            Kernel::Avx512
        };
    }

    let mut out = alloc_with_nan_prefix(size, 0);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ad_scalar(high, low, close, volume, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ad_avx2(high, low, close, volume, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ad_avx512(high, low, close, volume, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(AdOutput { values: out })
}

#[inline]
pub fn ad_into(input: &AdInput, out: &mut [f64]) -> Result<(), AdError> {
    ad_into_slice(out, input, Kernel::Auto)
}

pub fn ad_into_slice(dst: &mut [f64], input: &AdInput, kern: Kernel) -> Result<(), AdError> {
    let (high, low, close, volume) = match &input.data {
        AdData::Candles { candles, .. } => (
            &candles.high[..],
            &candles.low[..],
            &candles.close[..],
            &candles.volume[..],
        ),
        AdData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() {
        return Err(AdError::EmptyInputData);
    }

    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    if dst.len() != high.len() {
        return Err(AdError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }

    match kern {
        Kernel::Auto => {
            let mut k = detect_best_kernel();
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            if matches!(k, Kernel::Avx512) {
                k = if high.len() >= 262_144 {
                    Kernel::Avx2
                } else {
                    Kernel::Avx512
                };
            }
            match k {
                Kernel::Scalar => ad_scalar(high, low, close, volume, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => ad_avx2(high, low, close, volume, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => ad_avx512(high, low, close, volume, dst),
                _ => ad_scalar(high, low, close, volume, dst),
            }
        }
        Kernel::Scalar => ad_scalar(high, low, close, volume, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => ad_avx2(high, low, close, volume, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => ad_avx512(high, low, close, volume, dst),
        _ => ad_scalar(high, low, close, volume, dst),
    }

    Ok(())
}

#[inline]
pub fn ad_scalar(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    // Official TA-Lib authority: range must be strictly positive; zero or
    // inverted bars contribute nothing and carry the accumulator unchanged.
    // https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_AD.c
    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(high.len(), close.len());
    debug_assert_eq!(high.len(), volume.len());
    debug_assert_eq!(high.len(), out.len());

    let mut sum = 0.0f64;
    for ((((&h, &l), &c), &v), o) in high
        .iter()
        .zip(low)
        .zip(close)
        .zip(volume)
        .zip(out.iter_mut())
    {
        let hl = h - l;
        if hl > 0.0 {
            let num = (c - l) - (h - c);
            sum += (num / hl) * v;
        }
        *o = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx2(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    unsafe { ad_avx2_inner(high, low, close, volume, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ad_avx2_inner(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    use core::arch::x86_64::*;

    let n = high.len();
    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let v = volume.as_ptr();
    let o = out.as_mut_ptr();

    let mut base = 0.0f64;
    let mut i = 0usize;

    while i + 4 <= n {
        let hv = _mm256_loadu_pd(h.add(i));
        let lv = _mm256_loadu_pd(l.add(i));
        let cv = _mm256_loadu_pd(c.add(i));
        let vv = _mm256_loadu_pd(v.add(i));

        let hl = _mm256_sub_pd(hv, lv);
        let num = _mm256_sub_pd(_mm256_sub_pd(cv, lv), _mm256_sub_pd(hv, cv));
        let mfm = _mm256_div_pd(num, hl);
        let mfv_unmasked = _mm256_mul_pd(mfm, vv);

        let z = _mm256_set1_pd(0.0);
        let mask = _mm256_cmp_pd(hl, z, _CMP_GT_OQ);
        let mfv = _mm256_and_pd(mfv_unmasked, mask);

        let mut tmp: [f64; 4] = core::mem::zeroed();
        _mm256_storeu_pd(tmp.as_mut_ptr(), mfv);
        *o.add(i + 0) = {
            base += tmp[0];
            base
        };
        *o.add(i + 1) = {
            base += tmp[1];
            base
        };
        *o.add(i + 2) = {
            base += tmp[2];
            base
        };
        *o.add(i + 3) = {
            base += tmp[3];
            base
        };

        i += 4;
    }

    while i < n {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let cl = *c.add(i);
        let vo = *v.add(i);
        let hl = hi - lo;
        if hl > 0.0 {
            let num = (cl - lo) - (hi - cl);
            base += (num / hl) * vo;
        }
        *o.add(i) = base;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    unsafe { ad_avx512_inner(high, low, close, volume, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ad_avx512_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let n = high.len();
    let h = high.as_ptr();
    let l = low.as_ptr();
    let c = close.as_ptr();
    let v = volume.as_ptr();
    let o = out.as_mut_ptr();

    let mut base = 0.0f64;
    let mut i = 0usize;

    while i + 8 <= n {
        let hv = _mm512_loadu_pd(h.add(i));
        let lv = _mm512_loadu_pd(l.add(i));
        let cv = _mm512_loadu_pd(c.add(i));
        let vv = _mm512_loadu_pd(v.add(i));

        let hl = _mm512_sub_pd(hv, lv);
        let num = _mm512_sub_pd(_mm512_sub_pd(cv, lv), _mm512_sub_pd(hv, cv));
        let mfm = _mm512_div_pd(num, hl);
        let mfv_unmasked = _mm512_mul_pd(mfm, vv);

        let mask = _mm512_cmp_pd_mask(hl, _mm512_set1_pd(0.0), _CMP_GT_OQ);
        let mfv = _mm512_maskz_mov_pd(mask, mfv_unmasked);

        let mut tmp = core::mem::MaybeUninit::<[f64; 8]>::uninit();
        _mm512_storeu_pd(tmp.as_mut_ptr() as *mut f64, mfv);
        let vals = tmp.assume_init();

        *o.add(i + 0) = {
            base += vals[0];
            base
        };
        *o.add(i + 1) = {
            base += vals[1];
            base
        };
        *o.add(i + 2) = {
            base += vals[2];
            base
        };
        *o.add(i + 3) = {
            base += vals[3];
            base
        };
        *o.add(i + 4) = {
            base += vals[4];
            base
        };
        *o.add(i + 5) = {
            base += vals[5];
            base
        };
        *o.add(i + 6) = {
            base += vals[6];
            base
        };
        *o.add(i + 7) = {
            base += vals[7];
            base
        };

        i += 8;
    }

    while i < n {
        let hi = *h.add(i);
        let lo = *l.add(i);
        let cl = *c.add(i);
        let vo = *v.add(i);
        let hl = hi - lo;
        if hl > 0.0 {
            let num = (cl - lo) - (hi - cl);
            base += (num / hl) * vo;
        }
        *o.add(i) = base;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_short(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_avx512(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_long(high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64]) {
    ad_avx512(high, low, close, volume, out)
}

#[inline]
pub fn ad_batch_with_kernel(data: &AdBatchInput, k: Kernel) -> Result<AdBatchOutput, AdError> {
    let mut kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdError::InvalidKernelForBatch(other)),
    };
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(k, Kernel::Auto) && matches!(kernel, Kernel::Avx512Batch) {
        kernel = Kernel::Avx2Batch;
    }

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ad_batch_par_slice(data, simd)
}

#[derive(Clone, Debug)]
pub struct AdBatchInput<'a> {
    pub highs: &'a [&'a [f64]],
    pub lows: &'a [&'a [f64]],
    pub closes: &'a [&'a [f64]],
    pub volumes: &'a [&'a [f64]],
}

#[derive(Clone, Debug)]
pub struct AdBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
pub fn ad_batch_slice(data: &AdBatchInput, kern: Kernel) -> Result<AdBatchOutput, AdError> {
    ad_batch_inner(data, kern, false)
}

#[inline(always)]
pub fn ad_batch_par_slice(data: &AdBatchInput, kern: Kernel) -> Result<AdBatchOutput, AdError> {
    ad_batch_inner(data, kern, true)
}

fn ad_batch_inner(
    data: &AdBatchInput,
    kern: Kernel,
    parallel: bool,
) -> Result<AdBatchOutput, AdError> {
    let rows = data.highs.len();
    let cols = if rows > 0 { data.highs[0].len() } else { 0 };
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| AdError::InvalidInput("rows*cols overflow".into()))?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let values = unsafe {
        let ptr = buf_mu.as_mut_ptr() as *mut f64;
        let slice = std::slice::from_raw_parts_mut(ptr, len);

        ad_batch_inner_into(data, kern, parallel, slice)?;

        Vec::from_raw_parts(ptr, len, len)
    };
    std::mem::forget(buf_mu);

    Ok(AdBatchOutput { values, rows, cols })
}

fn ad_batch_inner_into(
    data: &AdBatchInput,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), AdError> {
    let rows = data.highs.len();
    let cols = if rows > 0 { data.highs[0].len() } else { 0 };

    if data.lows.len() != rows || data.closes.len() != rows || data.volumes.len() != rows {
        return Err(AdError::DataLengthMismatch {
            high_len: data.highs.len(),
            low_len: data.lows.len(),
            close_len: data.closes.len(),
            volume_len: data.volumes.len(),
        });
    }

    for row in 0..rows {
        let h_len = data.highs[row].len();
        let l_len = data.lows[row].len();
        let c_len = data.closes[row].len();
        let v_len = data.volumes[row].len();

        if h_len != cols || l_len != cols || c_len != cols || v_len != cols {
            return Err(AdError::DataLengthMismatch {
                high_len: h_len,
                low_len: l_len,
                close_len: c_len,
                volume_len: v_len,
            });
        }
    }

    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| AdError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(AdError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let mut actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kern, Kernel::Auto) && matches!(actual, Kernel::Avx512Batch) {
        actual = Kernel::Avx2Batch;
    }

    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        match actual {
            Kernel::Scalar | Kernel::ScalarBatch => ad_row_scalar(
                data.highs[row],
                data.lows[row],
                data.closes[row],
                data.volumes[row],
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ad_row_avx2(
                data.highs[row],
                data.lows[row],
                data.closes[row],
                data.volumes[row],
                dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ad_row_avx512(
                data.highs[row],
                data.lows[row],
                data.closes[row],
                data.volumes[row],
                dst,
            ),
            _ => ad_row_scalar(
                data.highs[row],
                data.lows[row],
                data.closes[row],
                data.volumes[row],
                dst,
            ),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, s) in out.chunks_mut(cols).enumerate() {
                do_row(r, s);
            }
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(())
}

#[inline(always)]
pub unsafe fn ad_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_avx2(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_avx512(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_avx512(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out: &mut [f64],
) {
    ad_avx512(high, low, close, volume, out)
}

#[derive(Debug, Clone)]
pub struct AdStream {
    sum: f64,
}

impl AdStream {
    #[inline(always)]
    pub fn try_new() -> Result<Self, AdError> {
        Ok(Self { sum: 0.0 })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        if volume == 0.0 {
            return self.sum;
        }

        let hl = high - low;
        if hl > 0.0 {
            let num = (close - low) - (high - close);

            self.sum += (num / hl) * volume;
        }
        self.sum
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdBatchBuilder {
    pub kernel: Kernel,
}

impl AdBatchBuilder {
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(
        self,
        highs: &[&[f64]],
        lows: &[&[f64]],
        closes: &[&[f64]],
        volumes: &[&[f64]],
    ) -> Result<AdBatchOutput, AdError> {
        let batch = AdBatchInput {
            highs,
            lows,
            closes,
            volumes,
        };
        ad_batch_with_kernel(&batch, self.kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::{Candles, read_candles_from_vortex};
    use crate::utilities::enums::Kernel;

    #[test]
    fn talib_nonpositive_range_carries_scalar_and_stream() {
        let high = [2.0, 1.0, 2.0];
        let low = [1.0, 2.0, 2.0];
        let close = [1.75, 1.75, 2.0];
        let volume = [8.0, 8.0, 8.0];
        let mut out = [f64::NAN; 3];
        ad_scalar(&high, &low, &close, &volume, &mut out);
        assert_eq!(out.map(f64::to_bits), [4.0f64.to_bits(); 3]);

        let mut stream = AdStream::try_new().expect("valid AD stream");
        assert_eq!(
            stream.update(2.0, 1.0, 1.75, 8.0).to_bits(),
            4.0f64.to_bits()
        );
        assert_eq!(
            stream.update(1.0, 2.0, 1.75, 8.0).to_bits(),
            4.0f64.to_bits()
        );
        assert_eq!(
            stream.update(2.0, 2.0, 2.0, 8.0).to_bits(),
            4.0f64.to_bits()
        );
    }

    fn check_ad_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = AdParams::default();
        let input = AdInput::from_candles(&candles, default_params);
        let output = ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ad_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let ad_result = ad_with_kernel(&input, kernel)?;
        assert_eq!(ad_result.values.len(), candles.close.len());
        let expected_last_five = [1645918.16, 1645876.11, 1645824.27, 1645828.87, 1645728.78];
        let start = ad_result.values.len() - 5;
        let actual = &ad_result.values[start..];
        for (i, &val) in actual.iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-1,
                "[{}] AD mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ad_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_input = AdInput::with_default_candles(&candles);
        let first_result = ad_with_kernel(&first_input, kernel)?;
        let second_input = AdInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            &first_result.values,
            AdParams::default(),
        );
        let second_result = ad_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 50..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_ad_input_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        match input.data {
            AdData::Candles { .. } => {}
            _ => panic!("Expected AdData::Candles variant"),
        }
        Ok(())
    }

    fn check_ad_accuracy_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let ad_result = ad_with_kernel(&input, kernel)?;
        assert_eq!(ad_result.values.len(), candles.close.len());
        if ad_result.values.len() > 50 {
            for i in 50..ad_result.values.len() {
                assert!(
                    !ad_result.values[i].is_nan(),
                    "[{}] Expected no NaN after index 50, but found NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_ad_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let batch = ad_with_kernel(&input, kernel)?.values;
        let mut stream = AdStream::try_new()?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            let val = stream.update(
                candles.high[i],
                candles.low[i],
                candles.close[i],
                candles.volume[i],
            );
            stream_values.push(val);
        }
        assert_eq!(batch.len(), stream_values.len());
        for (b, s) in batch.iter().zip(stream_values.iter()) {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (b - s).abs() < 1e-9,
                "[{}] AD streaming mismatch",
                test_name
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ad_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = AdInput::with_default_candles(&candles);
        let output = ad_with_kernel(&input, kernel)?;

        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

        let slice_input = AdInput::from_slices(
            &candles.high,
            &candles.low,
            &candles.close,
            &candles.volume,
            AdParams::default(),
        );
        let slice_output = ad_with_kernel(&slice_input, kernel)?;

        for (i, &val) in slice_output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (slice test)",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (slice test)",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (slice test)",
                    test_name, val, bits, i
                );
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ad_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_ad_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ad_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (10usize..400).prop_flat_map(|len| {
            prop::collection::vec(
                (
                    1.0f64..1000.0f64,
                    0.0f64..500.0f64,
                    0.0f64..1.0f64,
                    0.0f64..1e6f64,
                )
                    .prop_filter("finite values", |(l, hd, cr, v)| {
                        l.is_finite()
                            && hd.is_finite()
                            && cr.is_finite()
                            && v.is_finite()
                            && *v >= 0.0
                    })
                    .prop_map(|(low, high_delta, close_ratio, volume)| {
                        let high = low + high_delta;
                        let close = if high_delta == 0.0 {
                            low
                        } else {
                            low + high_delta * close_ratio
                        };
                        (high, low, close, volume)
                    }),
                len,
            )
            .prop_map(|data| {
                let (highs, lows, closes, volumes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
                    data.into_iter().map(|(h, l, c, v)| (h, l, c, v)).unzip4();
                (highs, lows, closes, volumes)
            })
        });

        trait Unzip4<A, B, C, D> {
            fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>);
        }

        impl<I, A, B, C, D> Unzip4<A, B, C, D> for I
        where
            I: Iterator<Item = (A, B, C, D)>,
        {
            fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>) {
                let (mut a, mut b, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for (av, bv, cv, dv) in self {
                    a.push(av);
                    b.push(bv);
                    c.push(cv);
                    d.push(dv);
                }
                (a, b, c, d)
            }
        }

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(highs, lows, closes, volumes)| {
                let input =
                    AdInput::from_slices(&highs, &lows, &closes, &volumes, AdParams::default());

                let AdOutput { values: out } = ad_with_kernel(&input, kernel).unwrap();

                let AdOutput { values: ref_out } = ad_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), highs.len(), "Output length mismatch");

                for (i, &val) in out.iter().enumerate() {
                    prop_assert!(
                        !val.is_nan(),
                        "Unexpected NaN at index {}: AD should not have NaN values",
                        i
                    );
                }

                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y_bits,
                            r_bits,
                            "Special value mismatch at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Value mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                for i in 1..volumes.len() {
                    if volumes[i] == 0.0 {
                        prop_assert!(
                            (out[i] - out[i - 1]).abs() < 1e-10,
                            "AD should not change when volume is 0 at index {}",
                            i
                        );
                    }
                }

                for i in 0..highs.len() {
                    if (highs[i] - lows[i]).abs() < 1e-10 {
                        if i == 0 {
                            prop_assert!(
                                out[i].abs() < 1e-10,
                                "When high=low, first AD value should be 0, got {}",
                                out[i]
                            );
                        } else {
                            prop_assert!(
                                (out[i] - out[i - 1]).abs() < 1e-10,
                                "When high=low at index {}, AD should remain unchanged",
                                i
                            );
                        }
                    }
                }

                let mut expected_ad = 0.0;
                for i in 0..highs.len() {
                    let hl = highs[i] - lows[i];
                    if hl > 0.0 {
                        let mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl;
                        let mfv = mfm * volumes[i];
                        expected_ad += mfv;
                    }
                    prop_assert!(
                        (out[i] - expected_ad).abs() < 1e-9,
                        "Cumulative property violation at index {}: expected {}, got {}",
                        i,
                        expected_ad,
                        out[i]
                    );
                }

                if !highs.is_empty() {
                    let hl = highs[0] - lows[0];
                    let expected_first = if hl > 0.0 {
                        ((closes[0] - lows[0]) - (highs[0] - closes[0])) / hl * volumes[0]
                    } else {
                        0.0
                    };
                    prop_assert!(
                        (out[0] - expected_first).abs() < 1e-10,
                        "First value mismatch: expected {}, got {}",
                        expected_first,
                        out[0]
                    );
                }

                for i in 0..highs.len() {
                    prop_assert!(
                        lows[i] <= closes[i] + 1e-10 && closes[i] <= highs[i] + 1e-10,
                        "Price constraint violation at index {}: low={}, close={}, high={}",
                        i,
                        lows[i],
                        closes[i],
                        highs[i]
                    );
                }

                let all_equal = highs
                    .iter()
                    .zip(lows.iter())
                    .zip(closes.iter())
                    .all(|((&h, &l), &c)| (h - l).abs() < 1e-10 && (l - c).abs() < 1e-10);

                if all_equal {
                    for (i, &val) in out.iter().enumerate() {
                        prop_assert!(
                            val.abs() < 1e-10,
                            "When all prices are equal, AD should be 0 at index {}, got {}",
                            i,
                            val
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_ad_tests!(
        check_ad_partial_params,
        check_ad_accuracy,
        check_ad_input_with_default_candles,
        check_ad_with_slice_data_reinput,
        check_ad_accuracy_nan_check,
        check_ad_streaming,
        check_ad_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ad_tests!(check_ad_property);

    fn check_batch_single_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let highs: Vec<&[f64]> = vec![&candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume];

        let single = ad_with_kernel(
            &AdInput::from_candles(&candles, AdParams::default()),
            kernel,
        )?
        .values;

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.values.len(), candles.close.len());

        for (i, (a, b)) in single.iter().zip(&batch.values).enumerate() {
            assert!(
                (a - b).abs() < 1e-8,
                "[{}] AD batch single row mismatch at {}: {} vs {}",
                test,
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_batch_multi_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let highs: Vec<&[f64]> = vec![&candles.high, &candles.high, &candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low, &candles.low, &candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close, &candles.close, &candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume, &candles.volume, &candles.volume];

        let single = ad_with_kernel(
            &AdInput::from_candles(&candles, AdParams::default()),
            kernel,
        )?
        .values;

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.values.len(), 3 * candles.close.len());

        for row in 0..3 {
            let row_slice = &batch.values[row * batch.cols..(row + 1) * batch.cols];
            for (i, (a, b)) in single.iter().zip(row_slice.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-8,
                    "[{}] AD batch multi row mismatch row {} idx {}: {} vs {}",
                    test,
                    row,
                    i,
                    a,
                    b
                );
            }
        }
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
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let mut highs: Vec<&[f64]> = vec![];
        let mut lows: Vec<&[f64]> = vec![];
        let mut closes: Vec<&[f64]> = vec![];
        let mut volumes: Vec<&[f64]> = vec![];

        highs.push(&c.high);
        lows.push(&c.low);
        closes.push(&c.close);
        volumes.push(&c.volume);

        let high_rev: Vec<f64> = c.high.iter().rev().copied().collect();
        let low_rev: Vec<f64> = c.low.iter().rev().copied().collect();
        let close_rev: Vec<f64> = c.close.iter().rev().copied().collect();
        let volume_rev: Vec<f64> = c.volume.iter().rev().copied().collect();

        highs.push(&high_rev);
        lows.push(&low_rev);
        closes.push(&close_rev);
        volumes.push(&volume_rev);

        if c.high.len() > 100 {
            highs.push(&c.high[50..]);
            lows.push(&c.low[50..]);
            closes.push(&c.close[50..]);
            volumes.push(&c.volume[50..]);
        }

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        for (idx, &val) in batch.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / batch.cols;
            let col = idx % batch.cols;

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
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

    gen_batch_tests!(check_batch_single_row);
    gen_batch_tests!(check_batch_multi_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_ad_into_matches_api() {
        let n = 256usize;
        let mut ts = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let i_f = i as f64;
            ts.push(i as i64);
            let o = 100.0 + (i % 13) as f64 * 0.75;
            let l = o - 2.0;
            let h = o + 2.0 + ((i % 3) as f64) * 0.1;
            let c = l + ((i % 5) as f64) * 0.5;
            let v = 1000.0 + 10.0 * i_f;
            open.push(o);
            low.push(l);
            high.push(h);
            close.push(c);
            volume.push(v);
        }

        let candles = Candles::new(
            ts,
            open,
            high.clone(),
            low.clone(),
            close.clone(),
            volume.clone(),
        );
        let input = AdInput::with_default_candles(&candles);

        let baseline = ad(&input).expect("ad() should succeed").values;

        let mut out = vec![0.0; baseline.len()];
        ad_into(&input, &mut out).expect("ad_into() should succeed");

        assert_eq!(out.len(), baseline.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for (i, (a, b)) in out
            .iter()
            .copied()
            .zip(baseline.iter().copied())
            .enumerate()
        {
            assert!(
                eq_or_both_nan(a, b),
                "ad_into parity failed at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }
}
