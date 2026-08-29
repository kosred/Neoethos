use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum NviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct NviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct NviParams;

#[inline(always)]
fn nvi_step(
    previous_nvi: f64,
    close: f64,
    previous_close: f64,
    volume: f64,
    previous_volume: f64,
) -> f64 {
    if volume < previous_volume && previous_close != 0.0 {
        // Keep TA-Lib's compound-assignment operation order.  The candidate is
        // committed only when it remains representable.
        let mut candidate = previous_nvi;
        candidate += (close - previous_close) / previous_close * candidate;
        if candidate.is_finite() {
            return candidate;
        }
    }
    previous_nvi
}

#[derive(Debug, Clone)]
pub struct NviInput<'a> {
    pub data: NviData<'a>,
    pub params: NviParams,
}

impl<'a> NviInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, close_source: &'a str, params: NviParams) -> Self {
        Self {
            data: NviData::Candles {
                candles,
                close_source,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: NviParams) -> Self {
        Self {
            data: NviData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", NviParams)
    }
}

#[derive(Debug, Error)]
pub enum NviError {
    #[error("nvi: Empty data provided.")]
    EmptyInputData,
    #[error("nvi: Empty data provided.")]
    EmptyData,
    #[error("nvi: All values are NaN in both close and volume.")]
    AllValuesNaN,
    #[error("nvi: All close values are NaN.")]
    AllCloseValuesNaN,
    #[error("nvi: All volume values are NaN.")]
    AllVolumeValuesNaN,
    #[error("nvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("nvi: Close and volume length mismatch: close={close_len}, volume={volume_len}")]
    MismatchedLength { close_len: usize, volume_len: usize },
    #[error(
        "nvi: Destination length mismatch: dst={dst_len}, close={close_len}, volume={volume_len}"
    )]
    DestinationLengthMismatch {
        dst_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("nvi: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("nvi: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("nvi: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("nvi: invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NviBuilder {
    kernel: Kernel,
}
impl NviBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<NviOutput, NviError> {
        let i = NviInput::with_default_candles(c);
        nvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<NviOutput, NviError> {
        let i = NviInput::from_slices(close, volume, NviParams);
        nvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NviStream, NviError> {
        NviStream::try_new()
    }
}

#[derive(Debug, Clone)]
pub struct NviStream {
    prev_close: f64,
    prev_volume: f64,
    nvi_val: f64,
    started: bool,
}

impl NviStream {
    #[inline]
    pub fn try_new() -> Result<Self, NviError> {
        Ok(Self {
            prev_close: 0.0,
            prev_volume: 0.0,
            nvi_val: 1000.0,
            started: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        if !self.started {
            if close.is_nan() || volume.is_nan() {
                return None;
            }
            self.prev_close = close;
            self.prev_volume = volume;
            self.started = true;
            return Some(self.nvi_val);
        }

        let nvi = nvi_step(
            self.nvi_val,
            close,
            self.prev_close,
            volume,
            self.prev_volume,
        );

        self.nvi_val = nvi;
        self.prev_close = close;
        self.prev_volume = volume;

        Some(nvi)
    }
}

#[derive(Clone, Debug)]
pub struct NviBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline]
pub fn nvi(input: &NviInput) -> Result<NviOutput, NviError> {
    nvi_with_kernel(input, Kernel::Auto)
}
pub fn nvi_with_kernel(input: &NviInput, kernel: Kernel) -> Result<NviOutput, NviError> {
    let (close, volume): (&[f64], &[f64]) = match &input.data {
        NviData::Candles {
            candles,
            close_source,
        } => {
            let close = source_type(candles, close_source);
            (close, candles.volume.as_slice())
        }
        NviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(NviError::MismatchedLength {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume)
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| {
            if close.iter().all(|&c| c.is_nan()) {
                NviError::AllCloseValuesNaN
            } else {
                NviError::AllVolumeValuesNaN
            }
        })?;
    let mut out = alloc_with_nan_prefix(close.len(), first);
    let _chosen = match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => Kernel::Scalar,
    };
    nvi_scalar(close, volume, first, &mut out);
    Ok(NviOutput { values: out })
}

#[inline]
pub fn nvi_into(input: &NviInput, out: &mut [f64]) -> Result<(), NviError> {
    let (close, volume): (&[f64], &[f64]) = match &input.data {
        NviData::Candles {
            candles,
            close_source,
        } => {
            let close = source_type(candles, close_source);
            (close, candles.volume.as_slice())
        }
        NviData::Slices { close, volume } => (*close, *volume),
    };

    nvi_into_slice(out, close, volume, Kernel::Auto)
}

#[inline]
pub fn nvi_into_slice(
    dst: &mut [f64],
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), NviError> {
    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(NviError::MismatchedLength {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if dst.len() != close.len() {
        return Err(NviError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }

    let first = close
        .iter()
        .zip(volume)
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| {
            if close.iter().all(|&c| c.is_nan()) {
                NviError::AllCloseValuesNaN
            } else {
                NviError::AllVolumeValuesNaN
            }
        })?;

    let _chosen = match kern {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => Kernel::Scalar,
    };
    nvi_scalar(close, volume, first, dst);

    for v in &mut dst[..first] {
        *v = f64::NAN;
    }

    Ok(())
}

pub fn nvi_scalar(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    debug_assert!(
        close.len() == volume.len() && volume.len() == out.len(),
        "Input slices must all have the same length."
    );

    let len = close.len();
    if len == 0 || first_valid >= len {
        return;
    }

    let mut nvi_val = 1000.0;

    unsafe {
        let close_ptr = close.as_ptr();
        let vol_ptr = volume.as_ptr();
        let out_ptr = out.as_mut_ptr();

        *out_ptr.add(first_valid) = nvi_val;

        let mut i = first_valid + 1;
        if i >= len {
            return;
        }

        let mut prev_close = *close_ptr.add(i - 1);
        let mut prev_volume = *vol_ptr.add(i - 1);

        while i < len {
            let c = *close_ptr.add(i);
            let v = *vol_ptr.add(i);

            nvi_val = nvi_step(nvi_val, c, prev_close, v, prev_volume);

            *out_ptr.add(i) = nvi_val;

            prev_close = c;
            prev_volume = v;
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx2(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    let len = close.len();
    if len == 0 || first_valid >= len {
        return;
    }

    let close_ptr = close.as_ptr();
    let vol_ptr = volume.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut nvi_val = 1000.0;
    *out_ptr.add(first_valid) = nvi_val;

    let mut i = first_valid + 1;
    if i >= len {
        return;
    }

    while i + 3 < len {
        let curr_c = _mm256_loadu_pd(close_ptr.add(i) as *const f64);
        let prev_c = _mm256_loadu_pd(close_ptr.add(i - 1) as *const f64);

        let curr_v = _mm256_loadu_pd(vol_ptr.add(i) as *const f64);
        let prev_v = _mm256_loadu_pd(vol_ptr.add(i - 1) as *const f64);

        let mut closes = [0.0; 4];
        let mut previous_closes = [0.0; 4];
        let mut volumes = [0.0; 4];
        let mut previous_volumes = [0.0; 4];
        _mm256_storeu_pd(closes.as_mut_ptr(), curr_c);
        _mm256_storeu_pd(previous_closes.as_mut_ptr(), prev_c);
        _mm256_storeu_pd(volumes.as_mut_ptr(), curr_v);
        _mm256_storeu_pd(previous_volumes.as_mut_ptr(), prev_v);

        for offset in 0..4 {
            nvi_val = nvi_step(
                nvi_val,
                closes[offset],
                previous_closes[offset],
                volumes[offset],
                previous_volumes[offset],
            );
            *out_ptr.add(i + offset) = nvi_val;
        }

        i += 4;
    }

    while i < len {
        let c = *close_ptr.add(i);
        let v = *vol_ptr.add(i);

        nvi_val = nvi_step(nvi_val, c, *close_ptr.add(i - 1), v, *vol_ptr.add(i - 1));
        *out_ptr.add(i) = nvi_val;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    let len = close.len();
    if len == 0 || first_valid >= len {
        return;
    }

    let close_ptr = close.as_ptr();
    let vol_ptr = volume.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut nvi_val = 1000.0;
    *out_ptr.add(first_valid) = nvi_val;

    let mut i = first_valid + 1;
    if i >= len {
        return;
    }

    while i + 7 < len {
        let curr_c = _mm512_loadu_pd(close_ptr.add(i) as *const f64);
        let prev_c = _mm512_loadu_pd(close_ptr.add(i - 1) as *const f64);

        let curr_v = _mm512_loadu_pd(vol_ptr.add(i) as *const f64);
        let prev_v = _mm512_loadu_pd(vol_ptr.add(i - 1) as *const f64);

        let mut closes = [0.0; 8];
        let mut previous_closes = [0.0; 8];
        let mut volumes = [0.0; 8];
        let mut previous_volumes = [0.0; 8];
        _mm512_storeu_pd(closes.as_mut_ptr(), curr_c);
        _mm512_storeu_pd(previous_closes.as_mut_ptr(), prev_c);
        _mm512_storeu_pd(volumes.as_mut_ptr(), curr_v);
        _mm512_storeu_pd(previous_volumes.as_mut_ptr(), prev_v);

        for offset in 0..8 {
            nvi_val = nvi_step(
                nvi_val,
                closes[offset],
                previous_closes[offset],
                volumes[offset],
                previous_volumes[offset],
            );
            *out_ptr.add(i + offset) = nvi_val;
        }

        i += 8;
    }

    while i < len {
        let c = *close_ptr.add(i);
        let v = *vol_ptr.add(i);

        nvi_val = nvi_step(nvi_val, c, *close_ptr.add(i - 1), v, *vol_ptr.add(i - 1));
        *out_ptr.add(i) = nvi_val;
        i += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512_short(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_avx512(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512_long(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_avx512(close, volume, first, out)
}

#[inline(always)]
pub fn nvi_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    k: Kernel,
) -> Result<NviBatchOutput, NviError> {
    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(NviError::MismatchedLength {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let cols = close.len();
    let first = close
        .iter()
        .zip(volume)
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| {
            if close.iter().all(|&c| c.is_nan()) {
                NviError::AllCloseValuesNaN
            } else {
                NviError::AllVolumeValuesNaN
            }
        })?;
    let mut buf_mu = make_uninit_matrix(1, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first]);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let chosen = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(NviError::InvalidKernelForBatch(other)),
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => nvi_row_scalar(close, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => nvi_row_scalar(close, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => nvi_row_scalar(close, volume, first, out),
            _ => unreachable!(),
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(NviBatchOutput {
        values,
        rows: 1,
        cols,
    })
}

#[inline(always)]
unsafe fn nvi_row_scalar(close: &[f64], volume: &[f64], first: usize, row_out_flat: &mut [f64]) {
    let len = close.len();
    let out = &mut row_out_flat[..len];
    let mut nvi_val = 1000.0;
    out[first] = nvi_val;

    let mut prev_close = close[first];
    let mut prev_volume = volume[first];

    for i in (first + 1)..len {
        nvi_val = nvi_step(nvi_val, close[i], prev_close, volume[i], prev_volume);
        out[i] = nvi_val;
        prev_close = close[i];
        prev_volume = volume[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_nvi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = NviInput::with_default_candles(&candles);
        let output = nvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_nvi_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = NviInput::with_default_candles(&candles);
        let result = nvi_with_kernel(&input, kernel)?;
        let expected_last_five = [
            154243.6925373456,
            153973.11239019397,
            153973.11239019397,
            154275.63921207888,
            154275.63921207888,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-5,
                "[{}] NVI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_nvi_empty_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data: [f64; 0] = [];
        let volume_data: [f64; 0] = [];
        let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
        let res = nvi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NVI should fail with empty data",
            test_name
        );
        Ok(())
    }

    fn check_nvi_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 120.0];
        let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
        let res = nvi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NVI should fail with not enough valid data",
            test_name
        );
        Ok(())
    }

    fn check_nvi_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close = candles.select_candle_field("close")?;
        let volume = candles.select_candle_field("volume")?;
        let input = NviInput::from_slices(close, volume, NviParams);
        let batch_output = nvi_with_kernel(&input, kernel)?.values;
        let mut stream = NviStream::try_new()?;

        let first_valid = close
            .iter()
            .zip(volume.iter())
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .unwrap_or(0);

        let mut stream_values = alloc_with_nan_prefix(close.len(), first_valid);

        for (i, (&c, &v)) in close.iter().zip(volume.iter()).enumerate() {
            if let Some(nvi_val) = stream.update(c, v) {
                stream_values[i] = nvi_val;
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] NVI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_nvi_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_scenarios = vec![
            ("default_candles", NviInput::with_default_candles(&candles)),
            (
                "close_source",
                NviInput::from_candles(&candles, "close", NviParams),
            ),
            (
                "high_source",
                NviInput::from_candles(&candles, "high", NviParams),
            ),
            (
                "low_source",
                NviInput::from_candles(&candles, "low", NviParams),
            ),
            (
                "open_source",
                NviInput::from_candles(&candles, "open", NviParams),
            ),
        ];

        for (scenario_idx, (scenario_name, input)) in test_scenarios.iter().enumerate() {
            let output = nvi_with_kernel(input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with scenario: {} (scenario set {})",
                        test_name, val, bits, i, scenario_name, scenario_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with scenario: {} (scenario set {})",
                        test_name, val, bits, i, scenario_name, scenario_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with scenario: {} (scenario set {})",
                        test_name, val, bits, i, scenario_name, scenario_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_nvi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(test)]
    fn check_nvi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (50usize..=500)
            .prop_flat_map(|len| {
                (
                    prop::collection::vec(
                        prop::strategy::Union::new(vec![
                            (0.001f64..0.1f64).boxed(),
                            (10f64..10000f64).boxed(),
                            (1e6f64..1e8f64).boxed(),
                        ])
                        .prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    prop::collection::vec(
                        prop::strategy::Union::new(vec![
                            (100f64..1000f64).boxed(),
                            (1000f64..1e6f64).boxed(),
                            (1e6f64..1e9f64).boxed(),
                        ])
                        .prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    0usize..=7,
                )
            })
            .prop_map(|(mut prices, mut volumes, scenario)| {
                match scenario {
                    0 => {}
                    1 => {
                        let const_vol = volumes[0];
                        volumes.iter_mut().for_each(|v| *v = const_vol);
                    }
                    2 => {
                        volumes.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    }
                    3 => {
                        volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    }
                    4 => {
                        for i in 0..volumes.len() {
                            volumes[i] = if i % 2 == 0 { 1000.0 } else { 500.0 };
                        }
                    }
                    5 => {
                        let const_price = prices[0];
                        prices.iter_mut().for_each(|p| *p = const_price);
                    }
                    6 => {
                        let start = prices[0];
                        let trend = 0.01f64;
                        for i in 0..prices.len() {
                            prices[i] = start * (1.0 + trend).powi(i as i32);
                        }
                    }
                    7 => {
                        let base = prices[0];
                        for i in 0..prices.len() {
                            prices[i] = base * (1.0 + 0.1 * ((i as f64 * 0.5).sin()));
                        }

                        for i in 0..volumes.len() {
                            volumes[i] *= (1.0 - (i as f64 / volumes.len() as f64) * 0.5);
                        }
                    }
                    _ => unreachable!(),
                }
                (prices, volumes, scenario)
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(close_data, volume_data, scenario)| {
                let input = NviInput::from_slices(&close_data, &volume_data, NviParams);

                let NviOutput { values: out } = nvi_with_kernel(&input, kernel)?;

                let NviOutput { values: ref_out } = nvi_with_kernel(&input, Kernel::Scalar)?;

                let first_valid = close_data
                    .iter()
                    .zip(volume_data.iter())
                    .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
                    .unwrap_or(close_data.len());

                if first_valid >= close_data.len() {
                    return Ok(());
                }

                prop_assert!(
                    (out[first_valid] - 1000.0).abs() < 1e-9,
                    "NVI should start at 1000.0, got {} at index {} (scenario {})",
                    out[first_valid],
                    first_valid,
                    scenario
                );

                let mut prev_nvi = 1000.0;
                let mut prev_close = close_data[first_valid];
                let mut prev_volume = volume_data[first_valid];

                for i in (first_valid + 1)..close_data.len() {
                    let curr_close = close_data[i];
                    let curr_volume = volume_data[i];
                    let curr_nvi = out[i];

                    if curr_volume < prev_volume {
                        let expected_pct = (curr_close - prev_close) / prev_close;
                        let expected_nvi = prev_nvi + prev_nvi * expected_pct;

                        prop_assert!(
							(curr_nvi - expected_nvi).abs() < 1e-9 ||
							(curr_nvi - expected_nvi).abs() / expected_nvi.abs() < 1e-9,
							"NVI calculation error at index {} (scenario {}): expected {}, got {}, \
							prev_nvi={}, pct_change={}, volume {} -> {}",
							i, scenario, expected_nvi, curr_nvi, prev_nvi, expected_pct,
							prev_volume, curr_volume
						);
                    } else {
                        prop_assert!(
							(curr_nvi - prev_nvi).abs() < 1e-9,
							"NVI should not change when volume doesn't decrease at index {} (scenario {}): \
							prev_nvi={}, curr_nvi={}, volume {} -> {}",
							i, scenario, prev_nvi, curr_nvi, prev_volume, curr_volume
						);
                    }

                    prev_nvi = curr_nvi;
                    prev_close = curr_close;
                    prev_volume = curr_volume;
                }

                for i in first_valid..close_data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "Kernel finite/NaN mismatch at index {} (scenario {}): {} vs {}",
                            i,
                            scenario,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at index {} (scenario {}): {} vs {} (ULP={})",
                            i,
                            scenario,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                match scenario {
                    1 => {
                        for i in (first_valid + 1)..out.len() {
                            prop_assert!(
								(out[i] - 1000.0).abs() < 1e-9,
								"NVI should stay at 1000.0 with constant volume, got {} at index {}",
								out[i], i
							);
                        }
                    }
                    3 => {
                        for i in (first_valid + 1)..out.len() {
                            prop_assert!(
								(out[i] - 1000.0).abs() < 1e-9,
								"NVI should stay at 1000.0 with always increasing volume, got {} at index {}",
								out[i], i
							);
                        }
                    }
                    5 => {
                        if first_valid + 1 < out.len() {
                            let mut expected_nvi = out[first_valid];
                            for i in (first_valid + 1)..out.len() {
                                prop_assert!(
									(out[i] - expected_nvi).abs() < 1e-9,
									"NVI should stay constant at {} with constant prices, got {} at index {}",
									expected_nvi, out[i], i
								);
                            }
                        }
                    }
                    _ => {}
                }

                let mut stream = NviStream::try_new()?;
                for i in 0..close_data.len() {
                    if let Some(stream_val) = stream.update(close_data[i], volume_data[i]) {
                        let batch_val = out[i];
                        if !batch_val.is_nan() {
                            prop_assert!(
                                (stream_val - batch_val).abs() < 1e-9,
                                "Streaming mismatch at index {} (scenario {}): stream={}, batch={}",
                                i,
                                scenario,
                                stream_val,
                                batch_val
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_nvi_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }

    generate_all_nvi_tests!(
        check_nvi_partial_params,
        check_nvi_accuracy,
        check_nvi_empty_data,
        check_nvi_not_enough_valid_data,
        check_nvi_streaming,
        check_nvi_no_poison
    );

    #[cfg(test)]
    generate_all_nvi_tests!(check_nvi_property);

    #[test]
    fn talib_nvi_carries_across_zero_previous_close() {
        let close = [0.0, 5.0, 10.0];
        let volume = [100.0, 50.0, 25.0];
        let input = NviInput::from_slices(&close, &volume, NviParams);

        let values = nvi_with_kernel(&input, Kernel::Scalar)
            .expect("TA-Lib-authoritative NVI fixture must evaluate")
            .values;

        assert_eq!(values[0].to_bits(), 1000.0f64.to_bits());
        assert_eq!(values[1].to_bits(), 1000.0f64.to_bits());
        assert_eq!(values[2].to_bits(), 2000.0f64.to_bits());
    }

    #[test]
    fn talib_nvi_zero_lookback_emits_single_seed_bar() {
        let close = [42.0];
        let volume = [7.0];
        let input = NviInput::from_slices(&close, &volume, NviParams);

        let values = nvi_with_kernel(&input, Kernel::Scalar)
            .expect("TA-Lib NVI has zero lookback")
            .values;
        assert_eq!(values, vec![1000.0]);

        let batch = nvi_batch_with_kernel(&close, &volume, Kernel::ScalarBatch)
            .expect("batch route must preserve the zero-lookback seed");
        assert_eq!(batch.values, values);
    }

    #[test]
    fn talib_nvi_keeps_last_representable_value_on_nonfinite_update() {
        let close = [1.0, f64::MAX, f64::MAX];
        let volume = [3.0, 2.0, 1.0];
        let input = NviInput::from_slices(&close, &volume, NviParams);

        let values = nvi_with_kernel(&input, Kernel::Scalar)
            .expect("overflow fixture must remain representable")
            .values;
        assert_eq!(values, vec![1000.0, 1000.0, 1000.0]);

        let mut stream = NviStream::try_new().expect("stream construction");
        let streamed: Vec<f64> = close
            .iter()
            .zip(&volume)
            .map(|(&c, &v)| stream.update(c, v).expect("valid stream bar"))
            .collect();
        assert_eq!(streamed, values);

        let batch = nvi_batch_with_kernel(&close, &volume, Kernel::ScalarBatch)
            .expect("batch overflow fixture must remain representable");
        assert_eq!(batch.values, values);
    }

    #[test]
    fn test_nvi_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let len = 256usize;
        let mut close = vec![f64::NAN; len];
        let mut volume = vec![f64::NAN; len];

        for i in 5..len {
            let t = (i - 5) as f64;

            close[i] = 100.0 + 0.05 * t + (0.01 * t).sin();

            volume[i] = 2000.0 + ((i as i64 % 7) as f64 - 3.0) * 40.0;
        }

        let input = NviInput::from_slices(&close, &volume, NviParams);

        let baseline = nvi(&input)?.values;

        let mut out = vec![0.0; len];
        {
            nvi_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());
        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            let equal = (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12;
            assert!(
                equal,
                "nvi_into parity mismatch at index {}: {} vs {}",
                i, a, b
            );
        }
        Ok(())
    }
}
