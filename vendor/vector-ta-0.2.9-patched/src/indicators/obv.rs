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
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ObvData<'a> {
    Candles { candles: &'a Candles },
    Slices { close: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct ObvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ObvParams;

#[derive(Debug, Clone)]
pub struct ObvInput<'a> {
    pub data: ObvData<'a>,
    pub params: ObvParams,
}

impl<'a> ObvInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ObvParams) -> Self {
        Self {
            data: ObvData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: ObvParams) -> Self {
        Self {
            data: ObvData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, ObvParams::default())
    }
    #[inline(always)]
    fn as_refs(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            ObvData::Candles { candles } => (candles.close.as_slice(), candles.volume.as_slice()),
            ObvData::Slices { close, volume } => (*close, *volume),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ObvBuilder {
    kernel: Kernel,
}

impl Default for ObvBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl ObvBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<ObvOutput, ObvError> {
        let i = ObvInput::from_candles(candles, ObvParams::default());
        obv_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<ObvOutput, ObvError> {
        let i = ObvInput::from_slices(close, volume, ObvParams::default());
        obv_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> ObvStream {
        ObvStream::new()
    }
}

#[derive(Debug, Error)]
pub enum ObvError {
    #[error("obv: Input data slice is empty.")]
    EmptyInputData,
    #[error("obv: Data length mismatch: close_len = {close_len}, volume_len = {volume_len}")]
    DataLengthMismatch { close_len: usize, volume_len: usize },
    #[error("obv: All values are NaN.")]
    AllValuesNaN,
    #[error("obv: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("obv: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("obv: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("obv: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("obv: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

impl From<Box<dyn std::error::Error>> for ObvError {
    fn from(_: Box<dyn std::error::Error>) -> Self {
        ObvError::EmptyInputData
    }
}

#[inline(always)]
pub fn obv(input: &ObvInput) -> Result<ObvOutput, ObvError> {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        obv_with_kernel(input, Kernel::Avx2)
    }
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        obv_with_kernel(input, Kernel::Scalar)
    }
}

pub fn obv_with_kernel(input: &ObvInput, kernel: Kernel) -> Result<ObvOutput, ObvError> {
    let (close, volume) = input.as_refs();

    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    let mut out = alloc_with_nan_prefix(close.len(), first);

    let chosen = resolve_obv_kernel(kernel);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => obv_scalar(close, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => obv_avx2(close, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => obv_avx512(close, volume, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(ObvOutput { values: out })
}

#[inline(always)]
fn resolve_obv_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    }
}

#[inline]
pub fn obv_into(input: &ObvInput, out: &mut [f64]) -> Result<(), ObvError> {
    let (close, volume) = input.as_refs();
    obv_into_slice(out, close, volume, Kernel::Auto)
}

#[inline(always)]
fn obv_step(prev_obv: f64, close: f64, prev_close: f64, volume: f64) -> f64 {
    if close > prev_close {
        prev_obv + volume
    } else if close < prev_close {
        prev_obv - volume
    } else {
        prev_obv
    }
}

#[inline(always)]
pub fn obv_scalar(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    let mut prev_obv = volume[first_valid];
    let mut prev_close = unsafe { *close.get_unchecked(first_valid) };
    unsafe {
        *out.get_unchecked_mut(first_valid) = prev_obv;
    }

    let mut i = first_valid + 1;
    while i < close.len() {
        unsafe {
            let c = *close.get_unchecked(i);
            let v = *volume.get_unchecked(i);
            prev_obv = obv_step(prev_obv, c, prev_close, v);
            *out.get_unchecked_mut(i) = prev_obv;
            prev_close = c;
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx2(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    // OBV is a recurrence. SIMD re-association changes the accumulator, so
    // the named AVX route deliberately shares the canonical sequential lane.
    obv_scalar(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    obv_avx2(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512_short(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    obv_avx2(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512_long(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    obv_avx2(close, volume, first_valid, out)
}

#[inline(always)]
pub unsafe fn obv_row_scalar(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx2(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx2(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx512(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx512_short(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx512_long(close, volume, first, out)
}

#[derive(Clone, Debug)]
pub struct ObvStream {
    prev_close: f64,
    prev_obv: f64,
    initialized: bool,
}

impl ObvStream {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            prev_close: f64::NAN,
            prev_obv: 0.0,
            initialized: false,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        if !self.initialized {
            if !close.is_nan() && !volume.is_nan() {
                self.prev_close = close;
                self.prev_obv = volume;
                self.initialized = true;
                return Some(volume);
            } else {
                return None;
            }
        }

        self.prev_obv = obv_step(self.prev_obv, close, self.prev_close, volume);

        self.prev_close = close;

        Some(self.prev_obv)
    }

    #[inline(always)]
    pub fn last(&self) -> Option<f64> {
        if self.initialized {
            Some(self.prev_obv)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.prev_obv = 0.0;
        self.initialized = false;
    }
}

#[derive(Clone, Debug)]
pub struct ObvBatchRange {
    pub reserved: usize,
}

impl Default for ObvBatchRange {
    fn default() -> Self {
        Self { reserved: 1 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ObvBatchBuilder {
    kernel: Kernel,
}

impl ObvBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<ObvBatchOutput, ObvError> {
        obv_batch_with_kernel(close, volume, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<ObvBatchOutput, ObvError> {
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(close, volume)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ObvBatchOutput, ObvError> {
        ObvBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

pub struct ObvBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

pub fn obv_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<ObvBatchOutput, ObvError> {
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(ObvError::InvalidKernelForBatch(other)),
    };
    obv_batch_par_slice(close, volume, chosen)
}

#[inline(always)]
pub fn obv_batch_slice(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<ObvBatchOutput, ObvError> {
    obv_batch_inner(close, volume, kern, false)
}

#[inline(always)]
pub fn obv_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<ObvBatchOutput, ObvError> {
    obv_batch_inner(close, volume, kern, true)
}

#[inline(always)]
fn obv_batch_inner(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<ObvBatchOutput, ObvError> {
    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    let rows = 1usize;
    let cols = close.len();

    let _ = rows
        .checked_mul(cols)
        .ok_or_else(|| ObvError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first]);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, rows * cols) };

    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => obv_row_scalar(
                close,
                volume,
                first,
                0,
                0,
                core::ptr::null(),
                0.0,
                out_slice,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => obv_row_avx2(
                close,
                volume,
                first,
                0,
                0,
                core::ptr::null(),
                0.0,
                out_slice,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => obv_row_avx512(
                close,
                volume,
                first,
                0,
                0,
                core::ptr::null(),
                0.0,
                out_slice,
            ),
            _ => unreachable!(),
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            rows * cols,
            guard.capacity(),
        )
    };
    Ok(ObvBatchOutput { values, rows, cols })
}

#[inline(always)]
fn obv_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), ObvError> {
    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if out.len() != close.len() {
        return Err(ObvError::OutputLengthMismatch {
            expected: close.len(),
            got: out.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    for v in &mut out[..first] {
        *v = f64::NAN;
    }

    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => {
                obv_row_scalar(close, volume, first, 0, 0, core::ptr::null(), 0.0, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => {
                obv_row_avx2(close, volume, first, 0, 0, core::ptr::null(), 0.0, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => {
                obv_row_avx512(close, volume, first, 0, 0, core::ptr::null(), 0.0, out)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline(always)]
fn expand_grid(_r: &ObvBatchRange) -> Vec<ObvParams> {
    vec![ObvParams::default()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    #[test]
    fn obv_auto_uses_the_authoritative_runtime_detector() {
        assert_eq!(
            resolve_obv_kernel(Kernel::Auto),
            crate::utilities::helpers::detect_best_kernel()
        );
        for explicit in [Kernel::Scalar, Kernel::Avx2, Kernel::Avx512] {
            assert_eq!(resolve_obv_kernel(explicit), explicit);
        }
    }

    #[test]
    fn talib_obv_lookback_zero_seeds_the_first_output_with_volume() {
        let close = [10.0];
        let volume = [5.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams);

        let output = obv_with_kernel(&input, Kernel::Scalar).expect("TA-Lib OBV seed");

        assert_eq!(output.values, [5.0]);
    }

    #[test]
    fn talib_obv_stream_uses_sequential_add_subtract_and_flat_carry() {
        let close = [10.0, 11.0, 10.0, 10.0];
        let volume = [5.0, 7.0, 3.0, 9.0];
        let expected = [5.0, 12.0, 9.0, 9.0];
        let mut stream = ObvStream::new();

        let actual: Vec<f64> = close
            .into_iter()
            .zip(volume)
            .map(|(close, volume)| stream.update(close, volume).expect("valid OBV bar"))
            .collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn talib_obv_scalar_and_batch_are_bit_identical() {
        let close = [f64::NAN, 10.0, 11.0, 10.0, 10.0];
        let volume = [f64::NAN, 5.0, 7.0, 3.0, 9.0];
        let expected = [f64::NAN, 5.0, 12.0, 9.0, 9.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams);

        let scalar = obv_with_kernel(&input, Kernel::Scalar).expect("scalar OBV");
        let batch = obv_batch_slice(&close, &volume, Kernel::ScalarBatch).expect("batch OBV");

        assert_eq!(scalar.values[0].is_nan(), expected[0].is_nan());
        assert_eq!(batch.values[0].is_nan(), expected[0].is_nan());
        for index in 1..expected.len() {
            assert_eq!(scalar.values[index].to_bits(), expected[index].to_bits());
            assert_eq!(batch.values[index].to_bits(), expected[index].to_bits());
            assert_eq!(
                batch.values[index].to_bits(),
                scalar.values[index].to_bits()
            );
        }
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn talib_obv_named_avx_routes_preserve_the_sequential_recurrence() {
        let close = [10.0, 11.0, 10.0, 10.0];
        let volume = [5.0, 7.0, 3.0, f64::INFINITY];
        let expected = [5.0, 12.0, 9.0, 9.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams);

        for kernel in [Kernel::Avx2, Kernel::Avx512] {
            let output = obv_with_kernel(&input, kernel).expect("named AVX OBV");
            assert_eq!(output.values, expected);
        }
    }

    #[test]
    fn test_obv_into_matches_api() {
        let n = 256usize;
        let mut close = vec![f64::NAN; n];
        let mut volume = vec![f64::NAN; n];

        for i in 0..n {
            if i >= 5 {
                let base = 100.0 + ((i as i32 % 11) - 5) as f64;
                let wiggle = ((i as f64) * 0.03).sin();
                close[i] = base + wiggle;
            }
            if i >= 7 {
                let v = ((i * 37) % 1000) as f64;
                volume[i] = if i % 10 == 0 { 0.0 } else { v + 0.5 };
            }
        }

        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let baseline = obv(&input).expect("baseline obv").values;

        let mut out = vec![0.0; n];
        obv_into(&input, &mut out).expect("obv_into");

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at {}: baseline={} into={}",
                i,
                baseline[i],
                out[i]
            );
        }
    }
    fn check_obv_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for empty data");
        Ok(())
    }
    fn check_obv_data_length_mismatch(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [1.0, 2.0, 3.0];
        let volume = [100.0, 200.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for mismatched data length");
        Ok(())
    }
    fn check_obv_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all NaN data");
        Ok(())
    }
    fn check_obv_csv_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close = source_type(&candles, "close");
        let volume = source_type(&candles, "volume");
        let input = ObvInput::from_candles(&candles, ObvParams::default());
        let obv_result = obv_with_kernel(&input, kernel)?;
        assert_eq!(obv_result.values.len(), close.len());
        let last_five_expected = [
            -327518.33864076994,
            -327624.59700968995,
            -327746.66483339,
            -327658.07136720995,
            -328074.9213672099,
        ];
        let start_idx = obv_result.values.len() - 5;
        let result_tail = &obv_result.values[start_idx..];
        for (i, &val) in result_tail.iter().enumerate() {
            let exp_val = last_five_expected[i];
            let diff = (val - exp_val).abs();
            assert!(
                diff < 1e-6,
                "OBV mismatch at tail index {}: expected {}, got {}",
                i,
                exp_val,
                val
            );
        }
        Ok(())
    }

    macro_rules! generate_all_obv_tests {
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
    fn check_obv_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let close = source_type(&candles, "close");
        let volume = source_type(&candles, "volume");

        let test_params = vec![ObvParams::default()];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = ObvInput::from_candles(&candles, params.clone());
            let output = obv_with_kernel(&input, kernel)?;

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
    fn check_obv_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec!["Testing OBV batch with default configuration"];

        for (cfg_idx, _config_name) in test_configs.iter().enumerate() {
            let output = ObvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with OBV (no params)",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with OBV (no params)",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with OBV (no params)",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_obv_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = prop::collection::vec(
            (
                (-1e6f64..1e6f64).prop_filter("finite close", |x| x.is_finite()),
                (0f64..1e6f64)
                    .prop_filter("finite positive volume", |x| x.is_finite() && *x >= 0.0),
            ),
            10..400,
        );

        proptest::test_runner::TestRunner::default().run(&strat, |price_volume_pairs| {
            let (close, volume): (Vec<f64>, Vec<f64>) = price_volume_pairs.into_iter().unzip();

            let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
            let ObvOutput { values: out } = obv_with_kernel(&input, kernel)?;
            let ObvOutput { values: ref_out } = obv_with_kernel(&input, Kernel::Scalar)?;

            let first_valid = close
                .iter()
                .zip(volume.iter())
                .position(|(c, v)| !c.is_nan() && !v.is_nan());

            prop_assert_eq!(
                first_valid,
                Some(0),
                "Expected first valid index to be 0 for finite input data"
            );

            if let Some(first_idx) = first_valid {
                for i in 0..first_idx {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN at index {} (before first_valid), got {}",
                        i,
                        out[i]
                    );
                }

                for i in first_idx..out.len() {
                    prop_assert!(
                        !out[i].is_nan(),
                        "Expected valid value at index {} (after first_valid), got NaN",
                        i
                    );
                }

                prop_assert_eq!(
                    out[first_idx],
                    volume[first_idx],
                    "First valid OBV at index {} should equal first volume, got {}",
                    first_idx,
                    out[first_idx]
                );

                for i in (first_idx + 1)..close.len() {
                    if !out[i].is_nan() && i > 0 && !out[i - 1].is_nan() {
                        let obv_diff = out[i] - out[i - 1];
                        let price_diff = close[i] - close[i - 1];

                        if price_diff > 0.0 {
                            prop_assert!(
                                (obv_diff - volume[i]).abs() < 1e-9,
                                "At index {}: OBV diff {} should equal volume {} (price increased)",
                                i,
                                obv_diff,
                                volume[i]
                            );
                        } else if price_diff < 0.0 {
                            prop_assert!(
									(obv_diff + volume[i]).abs() < 1e-9,
									"At index {}: OBV diff {} should equal -volume {} (price decreased)",
									i, obv_diff, -volume[i]
								);
                        } else {
                            prop_assert!(
									obv_diff.abs() < 1e-9,
									"At index {}: OBV should not change when price is unchanged, diff = {}",
									i, obv_diff
								);
                        }
                    }
                }

                for i in 0..out.len() {
                    if out[i].is_nan() && ref_out[i].is_nan() {
                        continue;
                    }
                    prop_assert!(
                        (out[i] - ref_out[i]).abs() < 1e-9,
                        "Kernel mismatch at index {}: {} (kernel) vs {} (scalar)",
                        i,
                        out[i],
                        ref_out[i]
                    );
                }

                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111
                                && bits != 0x22222222_22222222
                                && bits != 0x33333333_33333333,
                            "Found poison value at index {}: {} (0x{:016X})",
                            i,
                            val,
                            bits
                        );
                    }
                }

                if close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) {
                    for i in first_idx..out.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                (out[i] - volume[first_idx]).abs() < 1e-9,
                                "OBV should remain at first volume for constant price, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                if close.windows(2).all(|w| w[1] > w[0]) {
                    let mut expected_obv = volume[first_idx];
                    for i in first_idx..out.len() {
                        if i > first_idx {
                            expected_obv += volume[i];
                        }
                        if !out[i].is_nan() {
                            prop_assert!(
									(out[i] - expected_obv).abs() < 1e-9,
									"For monotonic increasing price at index {}: expected OBV {}, got {}",
									i, expected_obv, out[i]
								);
                        }
                    }
                }

                if close.windows(2).all(|w| w[1] < w[0]) {
                    let mut expected_obv = volume[first_idx];
                    for i in first_idx..out.len() {
                        if i > first_idx {
                            expected_obv -= volume[i];
                        }
                        if !out[i].is_nan() {
                            prop_assert!(
									(out[i] - expected_obv).abs() < 1e-9,
									"For monotonic decreasing price at index {}: expected OBV {}, got {}",
									i, expected_obv, out[i]
								);
                        }
                    }
                }

                for i in (first_idx + 1)..close.len() {
                    if volume[i] == 0.0 && i > 0 && !out[i].is_nan() && !out[i - 1].is_nan() {
                        prop_assert!(
								(out[i] - out[i - 1]).abs() < 1e-9,
								"OBV should not change when volume is 0 at index {}, but changed from {} to {}",
								i, out[i - 1], out[i]
							);
                    }
                }

                let max_possible_obv = 1e6 * (out.len() as f64);
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val.abs() <= max_possible_obv,
                            "OBV at index {} exceeds reasonable bounds: {} > {}",
                            i,
                            val.abs(),
                            max_possible_obv
                        );
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_obv_tests!(
        check_obv_empty_data,
        check_obv_data_length_mismatch,
        check_obv_all_nan,
        check_obv_csv_accuracy,
        check_obv_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_obv_tests!(check_obv_property);
}

pub fn obv_into_slice(
    dst: &mut [f64],
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), ObvError> {
    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if dst.len() != close.len() {
        return Err(ObvError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }

    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    let chosen = resolve_obv_kernel(kern);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => obv_scalar(close, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => obv_avx2(close, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => obv_avx512(close, volume, first, dst),
            _ => unreachable!(),
        }
    }

    for v in &mut dst[..first] {
        *v = f64::NAN;
    }
    Ok(())
}
