use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[inline(always)]
fn deviation_auto_kernel() -> Kernel {
    detect_best_kernel()
}

impl<'a> AsRef<[f64]> for DeviationInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DeviationData::Slice(slice) => slice,
            DeviationData::Candles { candles, source } => deviation_source_type(candles, source),
        }
    }
}

#[inline(always)]
fn deviation_source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "close" => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "hlcc4" | "hlcc" => &candles.hlcc4,
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum DeviationData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DeviationInput<'a> {
    pub data: DeviationData<'a>,
    pub params: DeviationParams,
}

impl<'a> DeviationInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DeviationParams) -> Self {
        Self {
            data: DeviationData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: DeviationParams) -> Self {
        Self {
            data: DeviationData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_defaults(data: &'a [f64]) -> Self {
        Self {
            data: DeviationData::Slice(data),
            params: DeviationParams::default(),
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DeviationParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }

    #[inline]
    pub fn get_devtype(&self) -> usize {
        self.params.devtype.unwrap_or(0)
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        self.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct DeviationOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DeviationParams {
    pub period: Option<usize>,
    pub devtype: Option<usize>,
}
impl Default for DeviationParams {
    fn default() -> Self {
        Self {
            period: Some(9),
            devtype: Some(0),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DeviationBuilder {
    period: Option<usize>,
    devtype: Option<usize>,
    kernel: Kernel,
}
impl Default for DeviationBuilder {
    fn default() -> Self {
        Self {
            period: None,
            devtype: None,
            kernel: Kernel::Auto,
        }
    }
}
impl DeviationBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn devtype(mut self, d: usize) -> Self {
        self.devtype = Some(d);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles, s: &str) -> Result<DeviationOutput, DeviationError> {
        let p = DeviationParams {
            period: self.period,
            devtype: self.devtype,
        };
        let i = DeviationInput::from_candles(c, s, p);
        deviation_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DeviationOutput, DeviationError> {
        let p = DeviationParams {
            period: self.period,
            devtype: self.devtype,
        };
        let i = DeviationInput::from_slice(d, p);
        deviation_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DeviationStream, DeviationError> {
        let p = DeviationParams {
            period: self.period,
            devtype: self.devtype,
        };
        DeviationStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DeviationError {
    #[error("deviation: empty input data")]
    EmptyInputData,
    #[error("deviation: All values are NaN.")]
    AllValuesNaN,
    #[error("deviation: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("deviation: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("deviation: output length mismatch: expected={expected} got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("deviation: Invalid devtype (must be 0, 1, or 2). devtype={devtype}")]
    InvalidDevType { devtype: usize },
    #[error("deviation: invalid range expansion start={start} end={end} step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("deviation: non-batch kernel passed to batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("deviation: Calculation error: {0}")]
    CalculationError(String),
}

#[inline(always)]
pub fn deviation(input: &DeviationInput) -> Result<DeviationOutput, DeviationError> {
    deviation_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn deviation_into(input: &DeviationInput, out: &mut [f64]) -> Result<(), DeviationError> {
    deviation_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
fn deviation_prepare<'a>(
    input: &'a DeviationInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), DeviationError> {
    let data = input.as_slice();
    let len = data.len();
    if len == 0 {
        return Err(DeviationError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DeviationError::AllValuesNaN)?;
    let period = input.get_period();
    let devtype = input.get_devtype();

    if period == 0 || period > len {
        return Err(DeviationError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(DeviationError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if !(0..=3).contains(&devtype) {
        return Err(DeviationError::InvalidDevType { devtype });
    }

    let chosen = match kernel {
        Kernel::Auto => deviation_auto_kernel(),
        k => k,
    };
    Ok((data, period, devtype, first, chosen))
}

const DEVIATION_POPULATION_F64_AUTHORITY_V2: &str =
    "deviation_population_f64_global_pow2_anchored_neumaier_two_pass_fma_sqrt_rn_v2";
const DEVIATION_CANONICAL_QNAN_BITS_V2: u64 = 0x7ff8_0000_0000_0000;

#[derive(Clone, Copy, Debug, Default)]
struct NeumaierSumF64V2 {
    sum: f64,
    correction: f64,
}

impl NeumaierSumF64V2 {
    #[inline(always)]
    fn add(&mut self, value: f64) {
        let updated = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - updated) + value;
        } else {
            self.correction += (value - updated) + self.sum;
        }
        self.sum = updated;
    }

    #[inline(always)]
    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

#[inline(always)]
fn deviation_canonical_qnan_v2() -> f64 {
    f64::from_bits(DEVIATION_CANONICAL_QNAN_BITS_V2)
}

#[inline(always)]
fn floor_power_of_two_input_scale_v2(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value > 0.0);

    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    if exponent != 0 {
        return f64::from_bits(exponent << 52);
    }

    let fraction = bits & ((1_u64 << 52) - 1);
    debug_assert_ne!(fraction, 0);
    let highest_fraction_bit = 63 - fraction.leading_zeros();
    f64::from_bits(1_u64 << highest_fraction_bit)
}

#[inline]
fn stable_population_deviation_window_v2<F>(len: usize, mut value_at: F) -> f64
where
    F: FnMut(usize) -> f64,
{
    let _semantic_authority = DEVIATION_POPULATION_F64_AUTHORITY_V2;
    if len == 0 {
        return deviation_canonical_qnan_v2();
    }

    let mut max_abs_input = 0.0f64;
    for index in 0..len {
        let value = value_at(index);
        if !value.is_finite() {
            return deviation_canonical_qnan_v2();
        }
        max_abs_input = max_abs_input.max(value.abs());
    }
    if max_abs_input == 0.0 {
        return 0.0;
    }

    let scale = floor_power_of_two_input_scale_v2(max_abs_input);
    let anchor = value_at(0) / scale;
    let mut shifted_sum = NeumaierSumF64V2::default();
    for index in 0..len {
        let value = value_at(index);
        let normalized_value = value / scale;
        let delta = normalized_value - anchor;
        shifted_sum.add(delta);
    }

    let n = len as f64;
    let mean_delta = shifted_sum.value() / n;
    if !mean_delta.is_finite() {
        return deviation_canonical_qnan_v2();
    }

    let mut normalized_squares = NeumaierSumF64V2::default();
    for index in 0..len {
        let value = value_at(index);
        let normalized_value = value / scale;
        let centered = (normalized_value - anchor) - mean_delta;
        normalized_squares.add(centered.mul_add(centered, 0.0));
    }

    let square_sum = normalized_squares.value();
    if !square_sum.is_finite() || square_sum < 0.0 {
        return deviation_canonical_qnan_v2();
    }
    let normalized_variance = square_sum / n;
    if !normalized_variance.is_finite() || normalized_variance < 0.0 {
        return deviation_canonical_qnan_v2();
    }

    let deviation = scale * normalized_variance.sqrt();
    if deviation.is_finite() {
        if deviation == 0.0 { 0.0 } else { deviation }
    } else {
        deviation_canonical_qnan_v2()
    }
}

#[inline]
fn stable_population_deviation_rolling_into_v2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    if period == 0 {
        return Err(DeviationError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    if data.len().saturating_sub(first) < period {
        return Err(DeviationError::NotEnoughValidData {
            needed: period,
            valid: data.len().saturating_sub(first),
        });
    }

    let warm = first + period - 1;
    for end in warm..data.len() {
        let start = end + 1 - period;
        out[end] = stable_population_deviation_window_v2(period, |offset| unsafe {
            *data.get_unchecked(start + offset)
        });
    }
    Ok(())
}

#[inline(always)]
fn deviation_compute_into(
    data: &[f64],
    period: usize,
    devtype: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    match kernel {
        Kernel::Scalar | Kernel::ScalarBatch => match devtype {
            0 => standard_deviation_rolling_into(data, period, first, out),
            1 => mean_absolute_deviation_rolling_into(data, period, first, out),
            2 => median_absolute_deviation_rolling_into(data, period, first, out),
            3 => mode_deviation_rolling_into(data, period, first, out),
            _ => unreachable!(),
        },
        Kernel::Avx2 | Kernel::Avx2Batch => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if devtype == 0 || devtype == 3 {
                    deviation_avx2(data, period, first, devtype, out);
                    return Ok(());
                }
            }
            match devtype {
                0 => standard_deviation_rolling_into(data, period, first, out),
                1 => mean_absolute_deviation_rolling_into(data, period, first, out),
                2 => median_absolute_deviation_rolling_into(data, period, first, out),
                3 => mode_deviation_rolling_into(data, period, first, out),
                _ => unreachable!(),
            }
        }
        Kernel::Avx512 | Kernel::Avx512Batch => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if devtype == 0 || devtype == 3 {
                    deviation_avx512(data, period, first, devtype, out);
                    return Ok(());
                }
            }
            match devtype {
                0 => standard_deviation_rolling_into(data, period, first, out),
                1 => mean_absolute_deviation_rolling_into(data, period, first, out),
                2 => median_absolute_deviation_rolling_into(data, period, first, out),
                3 => mode_deviation_rolling_into(data, period, first, out),
                _ => unreachable!(),
            }
        }
        Kernel::Auto => match devtype {
            0 => standard_deviation_rolling_into(data, period, first, out),
            1 => mean_absolute_deviation_rolling_into(data, period, first, out),
            2 => median_absolute_deviation_rolling_into(data, period, first, out),
            3 => mode_deviation_rolling_into(data, period, first, out),
            _ => unreachable!(),
        },
    }
}

pub fn deviation_with_kernel(
    input: &DeviationInput,
    kernel: Kernel,
) -> Result<DeviationOutput, DeviationError> {
    let (data, period, devtype, first, chosen) = deviation_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);
    deviation_compute_into(data, period, devtype, first, chosen, &mut out)?;
    Ok(DeviationOutput { values: out })
}

pub fn deviation_into_slice(
    dst: &mut [f64],
    input: &DeviationInput,
    kernel: Kernel,
) -> Result<(), DeviationError> {
    let (data, period, devtype, first, chosen) = deviation_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(DeviationError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    deviation_compute_into(data, period, devtype, first, chosen, dst)?;

    let warm = first + period - 1;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
pub fn deviation_scalar(
    data: &[f64],
    period: usize,
    devtype: usize,
) -> Result<Vec<f64>, DeviationError> {
    match devtype {
        0 => standard_deviation_rolling(data, period)
            .map_err(|e| DeviationError::CalculationError(e.to_string())),
        1 => mean_absolute_deviation_rolling(data, period)
            .map_err(|e| DeviationError::CalculationError(e.to_string())),
        2 => median_absolute_deviation_rolling(data, period)
            .map_err(|e| DeviationError::CalculationError(e.to_string())),
        3 => mode_deviation_rolling(data, period)
            .map_err(|e| DeviationError::CalculationError(e.to_string())),
        _ => Err(DeviationError::InvalidDevType { devtype }),
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx2(data: &[f64], period: usize, first: usize, devtype: usize, out: &mut [f64]) {
    let result = if devtype == 0 || devtype == 3 {
        standard_deviation_rolling_into(data, period, first, out)
    } else {
        legacy_raw_standard_deviation_rolling_into_v1(data, period, first, out)
    };
    let _ = result;
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    devtype: usize,
    out: &mut [f64],
) {
    let result = if devtype == 0 || devtype == 3 {
        standard_deviation_rolling_into(data, period, first, out)
    } else {
        legacy_raw_standard_deviation_rolling_into_v1(data, period, first, out)
    };
    let _ = result;
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    devtype: usize,
    out: &mut [f64],
) {
    deviation_avx512(data, period, first, devtype, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn deviation_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    devtype: usize,
    out: &mut [f64],
) {
    deviation_avx512(data, period, first, devtype, out);
}

#[derive(Clone, Debug)]
pub struct DeviationBatchRange {
    pub period: (usize, usize, usize),
    pub devtype: (usize, usize, usize),
}
impl Default for DeviationBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 258, 1),
            devtype: (0, 0, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeviationBatchBuilder {
    range: DeviationBatchRange,
    kernel: Kernel,
}
impl DeviationBatchBuilder {
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
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.devtype = (start, end, step);
        self
    }
    #[inline]
    pub fn devtype_static(mut self, d: usize) -> Self {
        self.range.devtype = (d, d, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<DeviationBatchOutput, DeviationError> {
        deviation_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<DeviationBatchOutput, DeviationError> {
        DeviationBatchBuilder::new().kernel(k).apply_slice(data)
    }
}

pub fn deviation_batch_with_kernel(
    data: &[f64],
    sweep: &DeviationBatchRange,
    k: Kernel,
) -> Result<DeviationBatchOutput, DeviationError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(DeviationError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    deviation_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DeviationBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DeviationParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DeviationBatchOutput {
    pub fn row_for_params(&self, p: &DeviationParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(9) == p.period.unwrap_or(9)
                && c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
        })
    }
    pub fn values_for(&self, p: &DeviationParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DeviationBatchRange) -> Vec<DeviationParams> {
    fn expand_axis(start: usize, end: usize, step: usize) -> Option<Vec<usize>> {
        if step == 0 {
            return Some(vec![start]);
        }
        let mut v = Vec::new();
        if start <= end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(n) => {
                        if n == cur {
                            break;
                        }
                        cur = n;
                    }
                    None => break,
                }
            }
        } else {
            let mut cur = start;
            loop {
                v.push(cur);
                if cur <= end {
                    break;
                }
                match cur.checked_sub(step) {
                    Some(n) => cur = n,
                    None => break,
                }
                if cur < end {
                    break;
                }
            }
        }
        if v.is_empty() { None } else { Some(v) }
    }
    let periods = match expand_axis(r.period.0, r.period.1, r.period.2) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let devtypes = match expand_axis(r.devtype.0, r.devtype.1, r.devtype.2) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let cap = match periods.len().checked_mul(devtypes.len()) {
        Some(c) => c,
        None => 0,
    };
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &d in &devtypes {
            out.push(DeviationParams {
                period: Some(p),
                devtype: Some(d),
            });
        }
    }
    out
}

#[inline(always)]
fn deviation_batch_inner_into(
    data: &[f64],
    sweep: &DeviationBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DeviationParams>, DeviationError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DeviationError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DeviationError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(DeviationError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or(DeviationError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    if out.len() != expected {
        return Err(DeviationError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let out_mu = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| {
            let warmup = first + c.period.unwrap() - 1;
            warmup.min(cols)
        })
        .collect();
    init_matrix_prefixes(out_mu, cols, &warms);

    let do_row = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| {
        let period = combos[row].period.unwrap();
        let devtype = combos[row].devtype.unwrap();
        let dst = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };

        let _ = deviation_compute_into(data, period, devtype, first, kern, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, chunk)| do_row(r, chunk));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, chunk) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, chunk);
            }
        }
    } else {
        for (r, chunk) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, chunk);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn deviation_batch_slice(
    data: &[f64],
    sweep: &DeviationBatchRange,
    kern: Kernel,
) -> Result<DeviationBatchOutput, DeviationError> {
    deviation_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn deviation_batch_par_slice(
    data: &[f64],
    sweep: &DeviationBatchRange,
    kern: Kernel,
) -> Result<DeviationBatchOutput, DeviationError> {
    deviation_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn deviation_batch_inner(
    data: &[f64],
    sweep: &DeviationBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DeviationBatchOutput, DeviationError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DeviationError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let _expected = rows.checked_mul(cols).ok_or(DeviationError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_f64 =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let _ = deviation_batch_inner_into(data, sweep, kern, parallel, out_f64)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(DeviationBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline]
fn standard_deviation_rolling_into(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    if data[first..].iter().all(|value| value.is_finite()) {
        standard_deviation_rolling_finite_into(data, period, first, out)
    } else {
        stable_population_deviation_rolling_into_v2(data, period, first, out)
    }
}

#[inline]
fn standard_deviation_rolling_finite_into(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    stable_population_deviation_rolling_into_v2(data, period, first, out)
}

#[inline]
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
fn legacy_raw_standard_deviation_rolling_into_v1(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    if period == 1 {
        for i in first..data.len() {
            out[i] = 0.0;
        }
        return Ok(());
    }
    if period < 1 {
        return Err(DeviationError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    if data.len() - first < period {
        return Err(DeviationError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }
    if data[first..].iter().all(|x| x.is_finite()) {
        return legacy_raw_standard_deviation_rolling_finite_into_v1(data, period, first, out);
    }

    let n = period as f64;
    let warm = first + period - 1;

    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut bad = 0usize;

    let mut j = first;
    let end0 = first + period;
    while j < end0 {
        let v = unsafe { *data.get_unchecked(j) };
        if !v.is_finite() {
            bad += 1;
        } else {
            sum += v;
            sumsq = v.mul_add(v, sumsq);
        }
        j += 1;
    }

    if bad > 0 || !sum.is_finite() || !sumsq.is_finite() {
        out[warm] = f64::NAN;
    } else {
        let mean = sum / n;
        let mut var = (sumsq / n) - mean * mean;

        let raw_scale = (sumsq / n).abs();
        if var.abs() / (raw_scale.max(1e-30)) < 1e-10 {
            let start = warm + 1 - period;
            let mut v2 = 0.0;
            let mut k = start;
            while k <= warm {
                let x = unsafe { *data.get_unchecked(k) };
                let d = x - mean;
                v2 = d.mul_add(d, v2);
                k += 1;
            }
            var = v2 / n;
        }
        if var < 0.0 {
            var = 0.0;
        }
        out[warm] = var.sqrt();
    }

    let mut i = warm + 1;
    while i < data.len() {
        let v_in = unsafe { *data.get_unchecked(i) };
        let v_out = unsafe { *data.get_unchecked(i - period) };

        if !v_in.is_finite() {
            bad += 1;
        } else {
            sum += v_in;
            sumsq = v_in.mul_add(v_in, sumsq);
        }
        if !v_out.is_finite() {
            bad = bad.saturating_sub(1);
        } else {
            sum -= v_out;
            sumsq -= v_out * v_out;
        }

        if bad > 0 || !sum.is_finite() || !sumsq.is_finite() {
            if bad == 0 {
                let start = i + 1 - period;
                let mut s = 0.0;
                let mut s2 = 0.0;
                let mut k = start;
                while k <= i {
                    let v = unsafe { *data.get_unchecked(k) };
                    s += v;
                    s2 = v.mul_add(v, s2);
                    k += 1;
                }
                if s.is_finite() && s2.is_finite() {
                    let mean = s / n;
                    let mut var = (s2 / n) - mean * mean;

                    let raw_scale = (s2 / n).abs();
                    if var.abs() / (raw_scale.max(1e-30)) < 1e-10 {
                        let mut v2 = 0.0;
                        let mut k = start;
                        while k <= i {
                            let x = unsafe { *data.get_unchecked(k) };
                            let d = x - mean;
                            v2 = d.mul_add(d, v2);
                            k += 1;
                        }
                        var = v2 / n;
                    }
                    if var < 0.0 {
                        var = 0.0;
                    }
                    out[i] = var.sqrt();
                } else {
                    out[i] = f64::NAN;
                }
            } else {
                out[i] = f64::NAN;
            }
        } else {
            let mean = sum / n;
            let mut var = (sumsq / n) - mean * mean;
            let raw_scale = (sumsq / n).abs();
            if var.abs() / (raw_scale.max(1e-30)) < 1e-10 {
                let start = i + 1 - period;
                let mut v2 = 0.0;
                let mut k = start;
                while k <= i {
                    let x = unsafe { *data.get_unchecked(k) };
                    let d = x - mean;
                    v2 = d.mul_add(d, v2);
                    k += 1;
                }
                var = v2 / n;
            }
            if var < 0.0 {
                var = 0.0;
            }
            out[i] = var.sqrt();
        }
        i += 1;
    }
    Ok(())
}

#[inline]
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
fn legacy_raw_standard_deviation_rolling_finite_into_v1(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    let n = period as f64;
    let warm = first + period - 1;

    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let end0 = first + period;
    let mut j = first;
    while j < end0 {
        let v = unsafe { *data.get_unchecked(j) };
        sum += v;
        sumsq = v.mul_add(v, sumsq);
        j += 1;
    }

    if !sum.is_finite() || !sumsq.is_finite() {
        out[warm] = f64::NAN;
    } else {
        let mut mean = sum / n;
        let mut var = (sumsq / n) - mean * mean;
        let raw_scale = (sumsq / n).abs();
        if var.abs() / raw_scale.max(1e-30) < 1e-10 {
            let mut v2 = 0.0;
            let mut k = first;
            while k <= warm {
                let x = unsafe { *data.get_unchecked(k) };
                let d = x - mean;
                v2 = d.mul_add(d, v2);
                k += 1;
            }
            var = v2 / n;
        }
        if var < 0.0 {
            var = 0.0;
        }
        out[warm] = var.sqrt();
    }

    let mut i = warm + 1;
    while i < data.len() {
        let v_in = unsafe { *data.get_unchecked(i) };
        let v_out = unsafe { *data.get_unchecked(i - period) };
        sum += v_in;
        sumsq = v_in.mul_add(v_in, sumsq);
        sum -= v_out;
        sumsq -= v_out * v_out;

        let start = i + 1 - period;
        if !sum.is_finite() || !sumsq.is_finite() {
            sum = 0.0;
            sumsq = 0.0;
            let mut k = start;
            while k <= i {
                let x = unsafe { *data.get_unchecked(k) };
                sum += x;
                sumsq = x.mul_add(x, sumsq);
                k += 1;
            }
            if !sum.is_finite() || !sumsq.is_finite() {
                out[i] = f64::NAN;
                i += 1;
                continue;
            }
        }

        let mean = sum / n;
        let mut var = (sumsq / n) - mean * mean;
        let raw_scale = (sumsq / n).abs();
        if var.abs() / raw_scale.max(1e-30) < 1e-10 {
            let mut v2 = 0.0;
            let mut k = start;
            while k <= i {
                let x = unsafe { *data.get_unchecked(k) };
                let d = x - mean;
                v2 = d.mul_add(d, v2);
                k += 1;
            }
            var = v2 / n;
        }
        if var < 0.0 {
            var = 0.0;
        }
        out[i] = var.sqrt();
        i += 1;
    }
    Ok(())
}

#[inline]
fn standard_deviation_rolling(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if period == 0 {
        return Err("Period must be >= 1 for standard deviation.".into());
    }
    let first_valid_idx = match data.iter().position(|&value| !value.is_nan()) {
        Some(index) => index,
        None => return Err("All values are NaN.".into()),
    };
    if data.len() - first_valid_idx < period {
        return Err(format!(
            "Not enough valid data: need {}, but only {} valid from index {}.",
            period,
            data.len() - first_valid_idx,
            first_valid_idx
        )
        .into());
    }

    let mut result = alloc_with_nan_prefix(data.len(), first_valid_idx + period - 1);
    stable_population_deviation_rolling_into_v2(data, period, first_valid_idx, &mut result)?;
    Ok(result)
}

#[inline]
fn mean_absolute_deviation_rolling_into(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    if data.len() - first < period {
        return Err(DeviationError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    let n = period as f64;
    let warm = first + period - 1;

    let mut sum = 0.0f64;
    let mut bad = 0usize;

    let mut j = first;
    let end0 = first + period;
    while j < end0 {
        let v = unsafe { *data.get_unchecked(j) };
        if !v.is_finite() {
            bad += 1;
        } else {
            sum += v;
        }
        j += 1;
    }

    if bad > 0 {
        out[warm] = f64::NAN;
    } else {
        let start = warm + 1 - period;
        let a = unsafe { *data.get_unchecked(start) };
        let mut res = 0.0f64;
        let mut k = start + 1;
        while k <= warm {
            res += unsafe { *data.get_unchecked(k) } - a;
            k += 1;
        }
        let mean = a + res / n;

        let mut abs_sum = 0.0f64;
        let mut k2 = start;
        let stop = k2 + (period & !3);
        while k2 < stop {
            let a0 = unsafe { *data.get_unchecked(k2) };
            let a1 = unsafe { *data.get_unchecked(k2 + 1) };
            let a2 = unsafe { *data.get_unchecked(k2 + 2) };
            let a3 = unsafe { *data.get_unchecked(k2 + 3) };
            abs_sum += (a0 - mean).abs();
            abs_sum += (a1 - mean).abs();
            abs_sum += (a2 - mean).abs();
            abs_sum += (a3 - mean).abs();
            k2 += 4;
        }
        while k2 <= warm {
            let a = unsafe { *data.get_unchecked(k2) };
            abs_sum += (a - mean).abs();
            k2 += 1;
        }
        out[warm] = abs_sum / n;
    }

    let mut i = warm + 1;
    while i < data.len() {
        let v_in = unsafe { *data.get_unchecked(i) };
        let v_out = unsafe { *data.get_unchecked(i - period) };

        if !v_in.is_finite() {
            bad += 1;
        } else {
            sum += v_in;
        }
        if !v_out.is_finite() {
            bad = bad.saturating_sub(1);
        } else {
            sum -= v_out;
        }

        if bad > 0 {
            out[i] = f64::NAN;
        } else {
            let start = i + 1 - period;
            let mean = if sum.is_finite() {
                sum / n
            } else {
                let a0 = unsafe { *data.get_unchecked(start) };
                let mut res = 0.0f64;
                let mut k = start + 1;
                while k <= i {
                    res += unsafe { *data.get_unchecked(k) } - a0;
                    k += 1;
                }
                a0 + res / n
            };
            let mut k = start;
            let mut abs_sum = 0.0f64;
            let stop = k + (period & !3);
            while k < stop {
                let a0 = unsafe { *data.get_unchecked(k) };
                let a1 = unsafe { *data.get_unchecked(k + 1) };
                let a2 = unsafe { *data.get_unchecked(k + 2) };
                let a3 = unsafe { *data.get_unchecked(k + 3) };
                abs_sum += (a0 - mean).abs();
                abs_sum += (a1 - mean).abs();
                abs_sum += (a2 - mean).abs();
                abs_sum += (a3 - mean).abs();
                k += 4;
            }
            while k <= i {
                let a = unsafe { *data.get_unchecked(k) };
                abs_sum += (a - mean).abs();
                k += 1;
            }
            out[i] = abs_sum / n;
        }
        i += 1;
    }
    Ok(())
}

#[inline]
fn mean_absolute_deviation_rolling(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values are NaN.".into()),
    };
    if data.len() - first_valid_idx < period {
        return Err(format!(
            "Not enough valid data: need {}, but only {} valid from index {}.",
            period,
            data.len() - first_valid_idx,
            first_valid_idx
        )
        .into());
    }
    let mut result = alloc_with_nan_prefix(data.len(), first_valid_idx + period - 1);
    let start_window_end = first_valid_idx + period - 1;
    for i in start_window_end..data.len() {
        let window_start = i + 1 - period;
        if window_start < first_valid_idx {
            continue;
        }
        let window = &data[window_start..=i];

        if window.iter().any(|&x| x.is_nan()) {
            result[i] = f64::NAN;
        } else {
            let mean = window.iter().sum::<f64>() / (period as f64);
            let abs_sum = window.iter().fold(0.0, |acc, &x| acc + (x - mean).abs());
            result[i] = abs_sum / (period as f64);
        }
    }
    Ok(result)
}

#[inline]
fn median_absolute_deviation_rolling_into(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    if data.len() - first < period {
        return Err(DeviationError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    const STACK_SIZE: usize = 256;
    let mut stack: [f64; STACK_SIZE] = [0.0; STACK_SIZE];
    let mut heap: Vec<f64> = if period > STACK_SIZE {
        vec![0.0; period]
    } else {
        Vec::new()
    };

    let warm = first + period - 1;
    let mut bad = 0usize;

    #[inline(always)]
    fn median_in_place(buf: &mut [f64]) -> f64 {
        let len = buf.len();
        let mid = len >> 1;
        if (len & 1) == 1 {
            let (_, m, _) = buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            *m
        } else {
            let (left, m, _right) =
                buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let hi = *m;
            let mut lo = left[0];
            for &v in &left[1..] {
                if v > lo {
                    lo = v;
                }
            }
            0.5 * (lo + hi)
        }
    }

    {
        let start = warm + 1 - period;
        let mut j = start;
        while j <= warm {
            if !unsafe { *data.get_unchecked(j) }.is_finite() {
                bad += 1;
            }
            j += 1;
        }
        if bad > 0 {
            out[warm] = f64::NAN;
        } else {
            let buf = if period <= STACK_SIZE {
                let tmp = &mut stack[..period];
                let mut k = 0;
                while k < period {
                    unsafe { *tmp.get_unchecked_mut(k) = *data.get_unchecked(start + k) };
                    k += 1;
                }
                tmp
            } else {
                let tmp = &mut heap[..period];
                tmp.copy_from_slice(&data[start..=warm]);
                tmp
            };

            let med = median_in_place(buf);
            let mut k = 0;
            while k < period {
                unsafe {
                    let x = *buf.get_unchecked(k);
                    *buf.get_unchecked_mut(k) = (x - med).abs();
                }
                k += 1;
            }
            out[warm] = median_in_place(buf);
        }
    }

    let mut i = warm + 1;
    while i < data.len() {
        let v_in = unsafe { *data.get_unchecked(i) };
        let v_out = unsafe { *data.get_unchecked(i - period) };
        if !v_in.is_finite() {
            bad += 1;
        }
        if !v_out.is_finite() {
            bad = bad.saturating_sub(1);
        }

        if bad > 0 {
            out[i] = f64::NAN;
        } else {
            let start = i + 1 - period;
            let buf = if period <= STACK_SIZE {
                let tmp = &mut stack[..period];
                let mut k = 0;
                while k < period {
                    unsafe { *tmp.get_unchecked_mut(k) = *data.get_unchecked(start + k) };
                    k += 1;
                }
                tmp
            } else {
                let tmp = &mut heap[..period];
                tmp.copy_from_slice(&data[start..=i]);
                tmp
            };

            let med = median_in_place(buf);
            let mut k = 0;
            while k < period {
                unsafe {
                    let x = *buf.get_unchecked(k);
                    *buf.get_unchecked_mut(k) = (x - med).abs();
                }
                k += 1;
            }
            out[i] = median_in_place(buf);
        }
        i += 1;
    }

    Ok(())
}

#[inline]
fn median_absolute_deviation_rolling(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values are NaN.".into()),
    };
    if data.len() - first_valid_idx < period {
        return Err(format!(
            "Not enough valid data: need {}, but only {} valid from index {}.",
            period,
            data.len() - first_valid_idx,
            first_valid_idx
        )
        .into());
    }
    let mut result = alloc_with_nan_prefix(data.len(), first_valid_idx + period - 1);
    let start_window_end = first_valid_idx + period - 1;

    const STACK_SIZE: usize = 256;
    let mut stack_buffer: [f64; STACK_SIZE] = [0.0; STACK_SIZE];
    let mut heap_buffer: Vec<f64> = if period > STACK_SIZE {
        vec![0.0; period]
    } else {
        Vec::new()
    };

    for i in start_window_end..data.len() {
        let window_start = i + 1 - period;
        if window_start < first_valid_idx {
            continue;
        }
        let window = &data[window_start..=i];

        if window.iter().any(|&x| x.is_nan()) {
            result[i] = f64::NAN;
        } else {
            let median = find_median(window);

            let abs_devs = if period <= STACK_SIZE {
                &mut stack_buffer[..period]
            } else {
                &mut heap_buffer[..period]
            };

            for (j, &x) in window.iter().enumerate() {
                abs_devs[j] = (x - median).abs();
            }

            result[i] = find_median(abs_devs);
        }
    }
    Ok(result)
}

#[inline]
fn mode_deviation_rolling_into(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), DeviationError> {
    standard_deviation_rolling_into(data, period, first, out)
}

#[inline]
fn mode_deviation_rolling(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    standard_deviation_rolling(data, period)
}

#[inline]
fn find_median(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NAN;
    }

    const STACK_SIZE: usize = 256;

    if slice.len() <= STACK_SIZE {
        let mut buf: [f64; STACK_SIZE] = [0.0; STACK_SIZE];
        let n = slice.len();
        buf[..n].copy_from_slice(slice);
        let b = &mut buf[..n];

        let mid = n >> 1;
        if (n & 1) == 1 {
            let (_, m, _) = b.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            *m
        } else {
            let (left, m, _right) = b.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let hi = *m;
            let mut lo = left[0];
            for &v in &left[1..] {
                if v > lo {
                    lo = v;
                }
            }
            0.5 * (lo + hi)
        }
    } else {
        let mut v = slice.to_vec();
        let n = v.len();
        let mid = n >> 1;
        if (n & 1) == 1 {
            let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            *m
        } else {
            let (left, m, _right) = v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            let hi = *m;
            let mut lo = left[0];
            for &x in &left[1..] {
                if x > lo {
                    lo = x;
                }
            }
            0.5 * (lo + hi)
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviationStream {
    period: usize,
    devtype: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,

    sum: f64,
    count: usize,

    inv_n: f64,

    tree: OstTreap,
}

impl DeviationStream {
    pub fn try_new(params: DeviationParams) -> Result<Self, DeviationError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(DeviationError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let devtype = params.devtype.unwrap_or(0);
        if !(0..=3).contains(&devtype) {
            return Err(DeviationError::InvalidDevType { devtype });
        }
        Ok(Self {
            period,
            devtype,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            sum: 0.0,
            count: 0,
            inv_n: 1.0 / (period as f64),
            tree: OstTreap::new(),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.update_impl(value)
    }

    #[inline(always)]
    fn std_dev_ring_o1(&self) -> f64 {
        if !self.filled {
            return deviation_canonical_qnan_v2();
        }

        stable_population_deviation_window_v2(self.period, |offset| {
            let physical = self.head + offset;
            let index = if physical >= self.period {
                physical - self.period
            } else {
                physical
            };
            self.buffer[index]
        })
    }

    #[inline(always)]
    fn mean_abs_dev_ring(&self) -> f64 {
        if self.buffer.iter().any(|&x| !x.is_finite()) {
            return f64::NAN;
        }

        let n = self.period as f64;
        let sum: f64 = self.buffer.iter().sum();
        let mean = sum / n;
        if !mean.is_finite() {
            return f64::NAN;
        }
        let abs_sum = self
            .buffer
            .iter()
            .fold(0.0, |acc, &x| acc + (x - mean).abs());
        abs_sum / n
    }

    #[inline(always)]
    fn median_abs_dev_ring(&self) -> f64 {
        if self.buffer.iter().any(|&x| x.is_nan()) {
            return f64::NAN;
        }

        let median = find_median(&self.buffer);
        let mut abs_devs: Vec<f64> = self.buffer.iter().map(|&x| (x - median).abs()).collect();
        find_median(&abs_devs)
    }
}

impl DeviationStream {
    #[inline(always)]
    fn push_pop(&mut self, value: f64) {
        let out = self.buffer[self.head];
        if out.is_finite() {
            self.sum -= out;
            self.count -= 1;
            self.tree.erase(out);
        }
        self.buffer[self.head] = value;
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
            self.tree.insert(value);
        }
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
            if !self.filled {
                self.filled = true;
            }
        }
    }

    #[inline(always)]
    fn mad_log_n(&self) -> f64 {
        if !self.filled || self.count < self.period {
            return f64::NAN;
        }
        let n = self.period as i64;
        let m = self.sum * self.inv_n;
        let k_le = self.tree.count_leq(m) as i64;
        let s_le = self.tree.sum_leq(m);
        let s_all = self.tree.sum_all();
        let abs_sum = m * ((2 * k_le - n) as f64) + (s_all - 2.0 * s_le);
        abs_sum * self.inv_n
    }

    #[inline(always)]
    fn medad_log_n(&self) -> f64 {
        if !self.filled || self.count < self.period {
            return f64::NAN;
        }
        let n = self.period;
        let med = if (n & 1) == 1 {
            self.tree.kth((n / 2 + 1) as u32)
        } else {
            let a = self.tree.kth((n / 2) as u32);
            let b = self.tree.kth((n / 2 + 1) as u32);
            0.5 * (a + b)
        };
        if (n & 1) == 1 {
            self.kth_abs_distance(med, (n / 2 + 1) as u32)
        } else {
            let d1 = self.kth_abs_distance(med, (n / 2) as u32);
            let d2 = self.kth_abs_distance(med, (n / 2 + 1) as u32);
            0.5 * (d1 + d2)
        }
    }

    #[inline(always)]
    fn kth_abs_distance(&self, m: f64, k: u32) -> f64 {
        let n = self.period as u32;
        let n_l = self.tree.count_lt(m) as u32;
        let n_r = n - n_l;
        let left_at = |idx: u32| -> f64 {
            let rank = n_l - idx;
            let x = self.tree.kth(rank);
            m - x
        };
        let right_at = |idx: u32| -> f64 {
            let rank = n_l + 1 + idx;
            let x = self.tree.kth(rank);
            x - m
        };
        let mut lo = k.saturating_sub(n_r);
        let mut hi = k.min(n_l);
        while lo <= hi {
            let i = (lo + hi) >> 1;
            let j = k - i;
            let a_left = if i == 0 {
                f64::NEG_INFINITY
            } else {
                left_at(i - 1)
            };
            let a_right = if i == n_l { f64::INFINITY } else { left_at(i) };
            let b_left = if j == 0 {
                f64::NEG_INFINITY
            } else {
                right_at(j - 1)
            };
            let b_right = if j == n_r { f64::INFINITY } else { right_at(j) };
            if a_left <= b_right && b_left <= a_right {
                return a_left.max(b_left);
            } else if a_left > b_right {
                hi = i - 1;
            } else {
                lo = i + 1;
            }
        }
        0.0
    }

    #[inline(always)]
    fn update_impl(&mut self, value: f64) -> Option<f64> {
        self.push_pop(value);
        if !self.filled {
            return None;
        }
        Some(match self.devtype {
            0 => self.std_dev_ring_o1(),
            1 => self.mad_log_n(),
            2 => self.medad_log_n(),
            3 => self.std_dev_ring_o1(),
            _ => f64::NAN,
        })
    }
}

#[inline(always)]
fn norm(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

#[derive(Debug, Clone, Default)]
struct OstTreap {
    root: Link,
    rng: u64,
}
type Link = Option<Box<Node>>;
#[derive(Debug, Clone)]
struct Node {
    key: f64,
    pri: u64,
    cnt: u32,
    size: u32,
    sum: f64,
    l: Link,
    r: Link,
}

impl OstTreap {
    #[inline(always)]
    fn new() -> Self {
        Self {
            root: None,
            rng: 0x9E3779B97F4A7C15,
        }
    }

    #[inline(always)]
    fn insert(&mut self, x: f64) {
        debug_assert!(x.is_finite());
        self.root = Self::ins(self.root.take(), norm(x), self.next());
    }
    #[inline(always)]
    fn erase(&mut self, x: f64) {
        debug_assert!(x.is_finite());
        self.root = Self::del(self.root.take(), norm(x));
    }
    #[inline(always)]
    fn count_lt(&self, x: f64) -> usize {
        Self::cnt_lt(&self.root, x) as usize
    }
    #[inline(always)]
    fn count_leq(&self, x: f64) -> usize {
        Self::cnt_leq(&self.root, x) as usize
    }
    #[inline(always)]
    fn sum_leq(&self, x: f64) -> f64 {
        Self::sum_leq_impl(&self.root, x)
    }
    #[inline(always)]
    fn sum_all(&self) -> f64 {
        Self::sum(&self.root)
    }
    #[inline(always)]
    fn kth(&self, k: u32) -> f64 {
        debug_assert!(k >= 1 && k <= Self::sz(&self.root));
        Self::kth_impl(&self.root, k)
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }
    #[inline(always)]
    fn sz(n: &Link) -> u32 {
        n.as_ref().map(|p| p.size).unwrap_or(0)
    }
    #[inline(always)]
    fn sum(n: &Link) -> f64 {
        n.as_ref().map(|p| p.sum).unwrap_or(0.0)
    }
    #[inline(always)]
    fn pull(n: &mut Box<Node>) {
        n.size = n.cnt + Self::sz(&n.l) + Self::sz(&n.r);
        n.sum = n.cnt as f64 * n.key + Self::sum(&n.l) + Self::sum(&n.r);
    }

    #[inline(always)]
    fn rot_right(mut y: Box<Node>) -> Box<Node> {
        let mut x = y.l.take().expect("rotate right");
        y.l = x.r.take();
        Self::pull(&mut y);
        x.r = Some(y);
        Self::pull(&mut x);
        x
    }
    #[inline(always)]
    fn rot_left(mut x: Box<Node>) -> Box<Node> {
        let mut y = x.r.take().expect("rotate left");
        x.r = y.l.take();
        Self::pull(&mut x);
        y.l = Some(x);
        Self::pull(&mut y);
        y
    }

    fn ins(t: Link, key: f64, pri: u64) -> Link {
        match t {
            None => Some(Box::new(Node {
                key,
                pri,
                cnt: 1,
                size: 1,
                sum: key,
                l: None,
                r: None,
            })),
            Some(mut n) => match key.total_cmp(&n.key) {
                core::cmp::Ordering::Equal => {
                    n.cnt += 1;
                    n.size += 1;
                    n.sum += key;
                    Some(n)
                }
                core::cmp::Ordering::Less => {
                    n.l = Self::ins(n.l.take(), key, pri);
                    if n.l.as_ref().unwrap().pri > n.pri {
                        n = Self::rot_right(n);
                    } else {
                        Self::pull(&mut n);
                    }
                    Some(n)
                }
                core::cmp::Ordering::Greater => {
                    n.r = Self::ins(n.r.take(), key, pri);
                    if n.r.as_ref().unwrap().pri > n.pri {
                        n = Self::rot_left(n);
                    } else {
                        Self::pull(&mut n);
                    }
                    Some(n)
                }
            },
        }
    }

    fn del(t: Link, key: f64) -> Link {
        match t {
            None => None,
            Some(mut n) => match key.total_cmp(&n.key) {
                core::cmp::Ordering::Equal => {
                    if n.cnt > 1 {
                        n.cnt -= 1;
                        n.size -= 1;
                        n.sum -= key;
                        Some(n)
                    } else {
                        match (n.l.take(), n.r.take()) {
                            (None, None) => None,
                            (Some(l), None) => Some(l),
                            (None, Some(r)) => Some(r),
                            (Some(l), Some(r)) => {
                                if l.pri > r.pri {
                                    let mut new = Self::rot_right(Box::new(Node {
                                        l: Some(l),
                                        r: Some(r),
                                        ..*n
                                    }));
                                    new.r = Self::del(new.r.take(), key);
                                    Self::pull(&mut new);
                                    Some(new)
                                } else {
                                    let mut new = Self::rot_left(Box::new(Node {
                                        l: Some(l),
                                        r: Some(r),
                                        ..*n
                                    }));
                                    new.l = Self::del(new.l.take(), key);
                                    Self::pull(&mut new);
                                    Some(new)
                                }
                            }
                        }
                    }
                }
                core::cmp::Ordering::Less => {
                    n.l = Self::del(n.l.take(), key);
                    Self::pull(&mut n);
                    Some(n)
                }
                core::cmp::Ordering::Greater => {
                    n.r = Self::del(n.r.take(), key);
                    Self::pull(&mut n);
                    Some(n)
                }
            },
        }
    }

    fn kth_impl(t: &Link, mut k: u32) -> f64 {
        let n = t.as_ref().unwrap();
        let ls = Self::sz(&n.l);
        if k <= ls {
            return Self::kth_impl(&n.l, k);
        }
        k -= ls;
        if k <= n.cnt {
            return n.key;
        }
        k -= n.cnt;
        Self::kth_impl(&n.r, k)
    }

    fn cnt_leq(t: &Link, x: f64) -> u32 {
        match t {
            None => 0,
            Some(n) => match x.total_cmp(&n.key) {
                core::cmp::Ordering::Less => Self::cnt_leq(&n.l, x),
                core::cmp::Ordering::Equal => Self::sz(&n.l) + n.cnt,
                core::cmp::Ordering::Greater => Self::sz(&n.l) + n.cnt + Self::cnt_leq(&n.r, x),
            },
        }
    }

    fn cnt_lt(t: &Link, x: f64) -> u32 {
        match t {
            None => 0,
            Some(n) => match x.total_cmp(&n.key) {
                core::cmp::Ordering::Less => Self::cnt_lt(&n.l, x),
                core::cmp::Ordering::Equal => Self::sz(&n.l),
                core::cmp::Ordering::Greater => Self::sz(&n.l) + n.cnt + Self::cnt_lt(&n.r, x),
            },
        }
    }

    fn sum_leq_impl(t: &Link, x: f64) -> f64 {
        match t {
            None => 0.0,
            Some(n) => match x.total_cmp(&n.key) {
                core::cmp::Ordering::Less => Self::sum_leq_impl(&n.l, x),
                core::cmp::Ordering::Equal => Self::sum(&n.l) + n.cnt as f64 * n.key,
                core::cmp::Ordering::Greater => {
                    Self::sum(&n.l) + n.cnt as f64 * n.key + Self::sum_leq_impl(&n.r, x)
                }
            },
        }
    }
}

#[inline(always)]
pub unsafe fn deviation_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    devtype: usize,
    out: &mut [f64],
) {
    match devtype {
        0 => {
            let _ = standard_deviation_rolling_into(data, period, first, out);
        }
        1 => {
            let _ = mean_absolute_deviation_rolling_into(data, period, first, out);
        }
        2 => {
            let _ = median_absolute_deviation_rolling_into(data, period, first, out);
        }
        3 => {
            let _ = mode_deviation_rolling_into(data, period, first, out);
        }
        _ => {}
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    devtype: usize,
    out: &mut [f64],
) {
    deviation_row_scalar(data, first, period, stride, devtype, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    devtype: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        deviation_row_avx512_short(data, first, period, stride, devtype, out);
    } else {
        deviation_row_avx512_long(data, first, period, stride, devtype, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    devtype: usize,
    out: &mut [f64],
) {
    deviation_row_scalar(data, first, period, stride, devtype, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn deviation_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    devtype: usize,
    out: &mut [f64],
) {
    deviation_row_scalar(data, first, period, stride, devtype, out);
}

#[inline(always)]
pub fn deviation_expand_grid(r: &DeviationBatchRange) -> Vec<DeviationParams> {
    expand_grid(r)
}

pub use DeviationError as DevError;
pub use DeviationInput as DevInput;
pub use DeviationParams as DevParams;

use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};
impl Index<usize> for DeviationOutput {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.values[idx]
    }
}
impl IndexMut<usize> for DeviationOutput {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.values[idx]
    }
}
impl<'a> IntoIterator for &'a DeviationOutput {
    type Item = &'a f64;
    type IntoIter = Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}
impl<'a> IntoIterator for &'a mut DeviationOutput {
    type Item = &'a mut f64;
    type IntoIter = IterMut<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}
impl DeviationOutput {
    pub fn last(&self) -> Option<&f64> {
        self.values.last()
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    const DEVIATION_RTX_PERIOD_9_INPUT_BITS_V1: [u64; 9] = [
        0x3ff1_335e_310d_bf05,
        0x3ff1_3317_9f58_f63b,
        0x3ff1_3342_4cab_aa3d,
        0x3ff1_330f_a73c_f04b,
        0x3ff1_334d_3466_391a,
        0x3ff1_332d_6ece_13f5,
        0x3ff1_335a_34ff_bc0d,
        0x3ff1_3324_6a42_93f9,
        0x3ff1_333f_5d0d_2150,
    ];
    const DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1: u64 = 0x3efa_abdb_f868_38c1;

    fn deviation_rtx_period_9_input_v1() -> [f64; 9] {
        DEVIATION_RTX_PERIOD_9_INPUT_BITS_V1.map(f64::from_bits)
    }

    fn period_9_output_bits_v1(data: &[f64], kernel: Kernel) -> u64 {
        let output = deviation_with_kernel(
            &DeviationInput::from_slice(
                data,
                DeviationParams {
                    period: Some(9),
                    devtype: Some(0),
                },
            ),
            kernel,
        )
        .expect("period-9 deviation fixture must be admitted");
        output.values[8].to_bits()
    }

    #[test]
    fn deviation_stable_f64_authority_v1_exact_fixture() {
        let data = deviation_rtx_period_9_input_v1();

        assert_eq!(
            period_9_output_bits_v1(&data, Kernel::Scalar),
            DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1,
            "scalar must use the corrected central-moment authority"
        );
        assert_eq!(
            period_9_output_bits_v1(&data, Kernel::Auto),
            DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1,
            "Auto must not select an architecture-dependent reduction tree"
        );

        let mut stream = DeviationStream::try_new(DeviationParams {
            period: Some(9),
            devtype: Some(0),
        })
        .expect("period-9 deviation stream must be admitted");
        let mut last = None;
        for value in data {
            last = stream.update(value);
        }
        assert_eq!(
            last.expect("ninth finite value must publish").to_bits(),
            DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1,
            "stream and batch must share the same f64 authority"
        );
    }

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    #[test]
    fn deviation_stable_f64_authority_v1_is_architecture_independent() {
        let data = deviation_rtx_period_9_input_v1();

        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            assert_eq!(
                period_9_output_bits_v1(&data, Kernel::Avx2),
                DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1
            );
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
        {
            assert_eq!(
                period_9_output_bits_v1(&data, Kernel::Avx512),
                DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1
            );
        }
    }

    #[test]
    fn deviation_stable_f64_authority_v1_handles_large_offsets() {
        let data: Vec<f64> = (0..9)
            .map(|index| 1.0e12 + (index as f64 - 8.0) * 2.441_406_25e-4)
            .collect();

        assert_eq!(
            period_9_output_bits_v1(&data, Kernel::Scalar),
            0x3f44_a7e9_cb8a_3491,
            "large offsets must not destroy a small finite variance"
        );
    }

    #[test]
    fn deviation_stable_f64_authority_v1_tracks_variance_collapse() {
        let ulp_at_one = f64::EPSILON;
        let mut data: Vec<f64> = [
            1000.0, -1000.0, 999.0, -999.0, 500.0, -500.0, 1.0, -1.0, 1.0,
        ]
        .into_iter()
        .map(|offset: f64| offset.mul_add(ulp_at_one, 1.0))
        .collect();
        data.extend(std::iter::repeat_n(1.0, 8));

        let output = deviation_with_kernel(
            &DeviationInput::from_slice(
                &data,
                DeviationParams {
                    period: Some(9),
                    devtype: Some(0),
                },
            ),
            Kernel::Scalar,
        )
        .expect("variance-collapse fixture must be admitted");

        assert_eq!(
            output
                .values
                .last()
                .copied()
                .expect("output is nonempty")
                .to_bits(),
            0x3c94_1cfe_93ff_5199,
            "one remaining one-ULP outlier must not inherit rolling-state drift"
        );
    }

    #[test]
    fn deviation_period_one_rejects_nonfinite_singletons_v1() {
        let data = [1.0, f64::INFINITY, f64::NAN, 2.0];
        let output = deviation_with_kernel(
            &DeviationInput::from_slice(
                &data,
                DeviationParams {
                    period: Some(1),
                    devtype: Some(0),
                },
            ),
            Kernel::Scalar,
        )
        .expect("period-one deviation must be admitted");

        assert_eq!(output.values[0].to_bits(), 0.0f64.to_bits());
        assert!(output.values[1].is_nan());
        assert!(output.values[2].is_nan());
        assert_eq!(output.values[3].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn deviation_stable_f64_authority_v2_routes_alias_batch_stream_and_row() {
        let data = deviation_rtx_period_9_input_v1();
        let sweep = DeviationBatchRange {
            period: (9, 9, 0),
            devtype: (0, 3, 3),
        };
        let batch = deviation_batch_slice(&data, &sweep, Kernel::Scalar)
            .expect("V2 deviation batch routes must be admitted");

        assert_eq!(batch.rows, 2);
        for row in 0..batch.rows {
            assert_eq!(
                batch.values[row * batch.cols + 8].to_bits(),
                DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1,
                "devtypes 0 and 3 must share the V2 batch authority"
            );
        }

        let mut stream = DeviationStream::try_new(DeviationParams {
            period: Some(9),
            devtype: Some(3),
        })
        .expect("devtype-3 stream must be admitted");
        let mut streamed = None;
        for value in data {
            streamed = stream.update(value);
        }
        assert_eq!(
            streamed
                .expect("full devtype-3 ring must publish")
                .to_bits(),
            DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1
        );

        for devtype in [0, 3] {
            let mut row = vec![f64::NAN; data.len()];
            unsafe {
                deviation_row_scalar(&data, 0, 9, 1, devtype, &mut row);
            }
            assert_eq!(
                row[8].to_bits(),
                DEVIATION_RTX_PERIOD_9_STABLE_BITS_V1,
                "row devtype {devtype} must share the V2 authority"
            );
        }
    }

    #[test]
    fn deviation_stable_f64_authority_v2_handles_extremes_subnormals_and_hole_recovery() {
        let extremes = [f64::MAX, -f64::MAX];
        let extreme_output = deviation_with_kernel(
            &DeviationInput::from_slice(
                &extremes,
                DeviationParams {
                    period: Some(2),
                    devtype: Some(0),
                },
            ),
            Kernel::Scalar,
        )
        .expect("opposite finite extremes must be admitted");
        assert_eq!(extreme_output.values[1].to_bits(), f64::MAX.to_bits());

        let min_subnormal = f64::from_bits(1);
        let subnormals = [0.0, min_subnormal * 2.0];
        let subnormal_output = deviation_with_kernel(
            &DeviationInput::from_slice(
                &subnormals,
                DeviationParams {
                    period: Some(2),
                    devtype: Some(0),
                },
            ),
            Kernel::Scalar,
        )
        .expect("subnormal spread must be admitted");
        assert_eq!(
            subnormal_output.values[1].to_bits(),
            min_subnormal.to_bits()
        );

        let with_hole = [1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        let recovered = deviation_with_kernel(
            &DeviationInput::from_slice(
                &with_hole,
                DeviationParams {
                    period: Some(2),
                    devtype: Some(0),
                },
            ),
            Kernel::Scalar,
        )
        .expect("interior non-finite windows must be admitted");
        assert!(recovered.values[2].is_nan());
        assert!(recovered.values[3].is_nan());
        assert_eq!(recovered.values[4].to_bits(), 0.5f64.to_bits());
    }

    #[test]
    fn deviation_stable_f64_authority_v2_period_one_is_consistent_across_routes() {
        let data = [1.0, f64::INFINITY, f64::NAN, -2.0];
        let expected = [0.0, f64::NAN, f64::NAN, 0.0];

        for devtype in [0, 3] {
            let legacy = deviation_scalar(&data, 1, devtype)
                .expect("legacy period-one deviation must be admitted");
            let mut stream = DeviationStream::try_new(DeviationParams {
                period: Some(1),
                devtype: Some(devtype),
            })
            .expect("period-one stream must be admitted");
            let streamed: Vec<f64> = data
                .iter()
                .map(|&value| stream.update(value).expect("period-one must publish"))
                .collect();
            let mut row = vec![f64::NAN; data.len()];
            unsafe {
                deviation_row_scalar(&data, 0, 1, 1, devtype, &mut row);
            }

            for (index, expected_value) in expected.iter().enumerate() {
                if expected_value.is_nan() {
                    assert!(legacy[index].is_nan());
                    assert!(streamed[index].is_nan());
                    assert!(row[index].is_nan());
                } else {
                    assert_eq!(legacy[index].to_bits(), expected_value.to_bits());
                    assert_eq!(streamed[index].to_bits(), expected_value.to_bits());
                    assert_eq!(row[index].to_bits(), expected_value.to_bits());
                }
            }
        }

        let sweep = DeviationBatchRange {
            period: (1, 1, 0),
            devtype: (0, 3, 3),
        };
        let batch = deviation_batch_slice(&data, &sweep, Kernel::Scalar)
            .expect("period-one batch routes must be admitted");
        for row in 0..batch.rows {
            for (index, expected_value) in expected.iter().enumerate() {
                let actual = batch.values[row * batch.cols + index];
                if expected_value.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual.to_bits(), expected_value.to_bits());
                }
            }
        }
    }

    #[test]
    fn test_deviation_into_matches_api_v2() -> Result<(), Box<dyn std::error::Error>> {
        let len = 256;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let x = (i as f64 * 0.1).sin() * 10.0 + (i % 7) as f64;
            data.push(x);
        }

        if len >= 2 {
            data[0] = f64::NAN;
            data[1] = f64::NAN;
        }

        let input = DeviationInput::from_slice(&data, DeviationParams::default());

        let baseline = deviation(&input)?.values;

        let mut into_out = vec![0.0; len];
        {
            deviation_into(&input, &mut into_out)?;
        }

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        assert_eq!(baseline.len(), into_out.len());
        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline[i], into_out[i]),
                "mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                into_out[i]
            );
        }

        Ok(())
    }

    fn check_deviation_partial_params(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let input = DeviationInput::with_defaults(&data);
        let output = deviation_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), data.len());
        Ok(())
    }
    fn check_deviation_accuracy(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DeviationParams {
            period: Some(3),
            devtype: Some(0),
        };
        let input = DeviationInput::from_slice(&data, params);
        let result = deviation_with_kernel(&input, kernel)?;
        let expected = 0.816496580927726;
        for &val in &result.values[2..] {
            assert!(
                (val - expected).abs() < 1e-12,
                "[{test}] deviation mismatch: got {}, expected {}",
                val,
                expected
            );
        }
        Ok(())
    }
    fn check_deviation_default_params(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let input = DeviationInput::with_defaults(&data);
        let output = deviation_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), data.len());
        Ok(())
    }
    fn check_deviation_zero_period(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0];
        let params = DeviationParams {
            period: Some(0),
            devtype: Some(0),
        };
        let input = DeviationInput::from_slice(&data, params);
        let res = deviation_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{test}] deviation should error with zero period"
        );
        Ok(())
    }
    fn check_deviation_period_exceeds_length(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0];
        let params = DeviationParams {
            period: Some(10),
            devtype: Some(0),
        };
        let input = DeviationInput::from_slice(&data, params);
        let res = deviation_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{test}] deviation should error if period > data.len()"
        );
        Ok(())
    }
    fn check_deviation_very_small_dataset(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let single = [42.0];
        let params = DeviationParams {
            period: Some(9),
            devtype: Some(0),
        };
        let input = DeviationInput::from_slice(&single, params);
        let res = deviation_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{test}] deviation should error if not enough data"
        );
        Ok(())
    }
    fn check_deviation_nan_handling(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DeviationParams {
            period: Some(3),
            devtype: Some(0),
        };
        let input = DeviationInput::from_slice(&data, params);
        let res = deviation_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), data.len());
        for (i, &v) in res.values.iter().enumerate().skip(4) {
            assert!(!v.is_nan(), "[{test}] Unexpected NaN at out-index {}", i);
        }
        Ok(())
    }
    fn check_deviation_streaming(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let period = 3;
        let devtype = 0;
        let input = DeviationInput::from_slice(
            &data,
            DeviationParams {
                period: Some(period),
                devtype: Some(devtype),
            },
        );
        let batch_output = deviation_with_kernel(&input, kernel)?.values;
        let mut stream = DeviationStream::try_new(DeviationParams {
            period: Some(period),
            devtype: Some(devtype),
        })?;
        let mut stream_values = Vec::with_capacity(data.len());
        for &val in &data {
            match stream.update(val) {
                Some(out_val) => stream_values.push(out_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (b, s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (b - s).abs() < 1e-9,
                "[{test}] streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                i,
                b,
                s,
                (b - s).abs()
            );
        }
        Ok(())
    }
    fn check_deviation_mean_absolute(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DeviationParams {
            period: Some(3),
            devtype: Some(1),
        };
        let input = DeviationInput::from_slice(&data, params);
        let result = deviation_with_kernel(&input, kernel)?;
        let expected = 2.0 / 3.0;
        for &val in &result.values[2..] {
            assert!(
                (val - expected).abs() < 1e-12,
                "[{test}] mean abs deviation mismatch: got {}, expected {}",
                val,
                expected
            );
        }
        Ok(())
    }
    fn check_deviation_median_absolute(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DeviationParams {
            period: Some(3),
            devtype: Some(2),
        };
        let input = DeviationInput::from_slice(&data, params);
        let result = deviation_with_kernel(&input, kernel)?;
        let expected = 1.0;
        for &val in &result.values[2..] {
            assert!(
                (val - expected).abs() < 1e-12,
                "[{test}] median abs deviation mismatch: got {}, expected {}",
                val,
                expected
            );
        }
        Ok(())
    }
    fn check_deviation_invalid_devtype(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0];
        let params = DeviationParams {
            period: Some(2),
            devtype: Some(999),
        };
        let input = DeviationInput::from_slice(&data, params);
        let result = deviation_with_kernel(&input, kernel);
        assert!(
            matches!(result, Err(DeviationError::InvalidDevType { .. })),
            "[{test}] invalid devtype should error"
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_deviation_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let data = candles.select_candle_field("close")?;

        let test_params = vec![
            DeviationParams::default(),
            DeviationParams {
                period: Some(2),
                devtype: Some(0),
            },
            DeviationParams {
                period: Some(5),
                devtype: Some(0),
            },
            DeviationParams {
                period: Some(5),
                devtype: Some(1),
            },
            DeviationParams {
                period: Some(5),
                devtype: Some(2),
            },
            DeviationParams {
                period: Some(20),
                devtype: Some(0),
            },
            DeviationParams {
                period: Some(20),
                devtype: Some(1),
            },
            DeviationParams {
                period: Some(20),
                devtype: Some(2),
            },
            DeviationParams {
                period: Some(50),
                devtype: Some(0),
            },
            DeviationParams {
                period: Some(50),
                devtype: Some(1),
            },
            DeviationParams {
                period: Some(100),
                devtype: Some(0),
            },
            DeviationParams {
                period: Some(100),
                devtype: Some(2),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DeviationInput::from_slice(&data, params.clone());
            let output = deviation_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, devtype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        params.devtype.unwrap_or(0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_deviation_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_deviation_tests {
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
    generate_all_deviation_tests!(
        check_deviation_partial_params,
        check_deviation_accuracy,
        check_deviation_default_params,
        check_deviation_zero_period,
        check_deviation_period_exceeds_length,
        check_deviation_very_small_dataset,
        check_deviation_nan_handling,
        check_deviation_streaming,
        check_deviation_mean_absolute,
        check_deviation_median_absolute,
        check_deviation_invalid_devtype,
        check_deviation_no_poison
    );

    #[test]
    fn test_deviation_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let mut data = Vec::with_capacity(256);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..251usize {
            let x = (i as f64 * 0.113).sin() * 3.7 + (i as f64 * 0.017).cos() * 1.3 + 42.0;
            data.push(x);
        }

        let input = DeviationInput::with_defaults(&data);

        let baseline = deviation(&input)?.values;

        let mut out = vec![0.0f64; data.len()];
        {
            deviation_into(&input, &mut out)?;
        }

        assert_eq!(baseline.len(), out.len());
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }
        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "deviation_into parity mismatch at index {}: vec={} into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_deviation_tests!(check_deviation_property);

    #[cfg(test)]
    mod deviation_property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn deviation_property_test(
                data in prop::collection::vec(prop::num::f64::ANY, 10..=1000),
                period in 2usize..=50,
                devtype in 0usize..=2
            ) {

                if data.iter().all(|x| x.is_nan()) {
                    return Ok(());
                }


                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                if data.len() - first_valid < period {
                    return Ok(());
                }

                let params = DeviationParams {
                    period: Some(period),
                    devtype: Some(devtype),
                };
                let input = DeviationInput::from_slice(&data, params);


                let result = deviation(&input);


                if let Ok(output) = result {
                    prop_assert_eq!(output.values.len(), data.len());


                    for i in 0..(first_valid + period - 1).min(data.len()) {
                        prop_assert!(output.values[i].is_nan());
                    }


                    if devtype <= 2 {
                        for i in (first_valid + period - 1)..data.len() {

                            let window_start = if i >= period - 1 { i + 1 - period } else { 0 };
                            let window = &data[window_start..=i];
                            let window_has_nan = window.iter().any(|x| x.is_nan());


                            let would_overflow = match devtype {
                                0 => {

                                    let sum: f64 = window.iter().sum();
                                    let sumsq: f64 = window.iter().map(|&x| x * x).sum();
                                    !sum.is_finite() || !sumsq.is_finite()
                                },
                                1 => {

                                    window.iter().any(|&x| !x.is_finite())
                                },
                                2 => {

                                    window.iter().any(|&x| !x.is_finite())
                                },
                                _ => false,
                            };


                            if window_has_nan || would_overflow {
                                prop_assert!(output.values[i].is_nan());
                            } else {
                                prop_assert!(!output.values[i].is_nan());
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "proptest")]
    fn check_deviation_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                    period + 10..400,
                ),
                Just(period),
                0usize..=2,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, devtype)| {
                let params = DeviationParams {
                    period: Some(period),
                    devtype: Some(devtype),
                };
                let input = DeviationInput::from_slice(&data, params);

                let DeviationOutput { values: out } =
                    deviation_with_kernel(&input, kernel).unwrap();
                let DeviationOutput { values: ref_out } =
                    deviation_with_kernel(&input, Kernel::Scalar).unwrap();

                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_period = first_valid + period - 1;

                for i in 0..warmup_period.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                prop_assert_eq!(out.len(), data.len());

                for i in warmup_period..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    prop_assert!(
                        y.is_nan() || y >= -1e-12,
                        "Deviation at index {} is negative: {}",
                        i,
                        y
                    );

                    let window = &data[i + 1 - period..=i];
                    let all_same = window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-14);
                    if all_same && window.iter().all(|x| x.is_finite()) {
                        prop_assert!(
								y.abs() < 1e-2 || y.is_nan(),
								"Deviation should be ~0 (or NaN due to precision bug) for constant window at index {}: {}",
								i,
								y
							);
                    }

                    if devtype == 0 && y.is_finite() && y > 1e-10 {
                        let variance = y * y;

                        let window_mean = window.iter().sum::<f64>() / (period as f64);
                        let computed_var = window
                            .iter()
                            .map(|&x| (x - window_mean).powi(2))
                            .sum::<f64>()
                            / (period as f64);

                        let var_diff = (variance - computed_var).abs();
                        let relative_error = var_diff / computed_var.max(1e-10);
                        let var_tol = computed_var.abs() * 1e-6 + 1e-8;
                        prop_assert!(
							var_diff <= var_tol,
							"Variance relationship failed at index {}: stddev²={} vs computed_var={} (rel_err={})",
							i,
							variance,
							computed_var,
							relative_error
						);
                    }

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
                    let abs_diff = (y - r).abs();

                    let tol = (r.abs() * 1e-7_f64).max(1e-6_f64);
                    prop_assert!(
                        abs_diff <= tol || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );

                    let window_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let window_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let window_range = window_max - window_min;

                    prop_assert!(
                        y <= window_range + 1e-9,
                        "Deviation {} exceeds window range {} at index {}",
                        y,
                        window_range,
                        i
                    );

                    match devtype {
                        0 => {
                            if y.is_finite() && y > 0.0 {
                                let window_mean = window.iter().sum::<f64>() / (period as f64);
                                let theoretical_var = window
                                    .iter()
                                    .map(|&x| (x - window_mean).powi(2))
                                    .sum::<f64>()
                                    / (period as f64);
                                let theoretical_std = theoretical_var.sqrt();

                                let tolerance = theoretical_std * 1e-4 + 1e-10;
                                prop_assert!(
									y <= theoretical_std + tolerance,
									"StdDev {} exceeds theoretical value {} by more than tolerance at index {}",
									y,
									theoretical_std,
									i
								);

                                let window_min =
                                    window.iter().cloned().fold(f64::INFINITY, f64::min);
                                let window_max =
                                    window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let max_possible_std = (window_max - window_min) / 2.0;
                                let max_bound = max_possible_std * 1.001 + 1e-8;

                                prop_assert!(
                                    y <= max_bound,
                                    "StdDev {} exceeds maximum possible {} (bound={}) at index {}",
                                    y,
                                    max_possible_std,
                                    max_bound,
                                    i
                                );
                            }
                        }
                        1 => {
                            let std_dev_params = DeviationParams {
                                period: Some(period),
                                devtype: Some(0),
                            };
                            let std_input = DeviationInput::from_slice(&data, std_dev_params);
                            if let Ok(std_output) = deviation_with_kernel(&std_input, kernel) {
                                let std_val = std_output.values[i];
                                if std_val.is_finite() && y.is_finite() {
                                    let tolerance = std_val * 1e-7 + 1e-9;
                                    prop_assert!(
                                        y <= std_val + tolerance,
                                        "MAD {} exceeds StdDev {} at index {}",
                                        y,
                                        std_val,
                                        i
                                    );
                                }
                            }
                        }
                        2 => {
                            if y.is_finite() && y > 0.0 {
                                prop_assert!(
                                    y <= window_range + 1e-12,
                                    "MedianAbsDev {} exceeds window range {} at index {}",
                                    y,
                                    window_range,
                                    i
                                );

                                let std_dev_params = DeviationParams {
                                    period: Some(period),
                                    devtype: Some(0),
                                };
                                let std_input = DeviationInput::from_slice(&data, std_dev_params);
                                if let Ok(std_output) = deviation_with_kernel(&std_input, kernel) {
                                    let std_val = std_output.values[i];
                                    if std_val.is_finite() && std_val > 0.0 {
                                        prop_assert!(
                                            y <= std_val * 1.5 + 1e-9,
                                            "MedAD {} exceeds 1.5x StdDev {} at index {}",
                                            y,
                                            std_val,
                                            i
                                        );
                                    }
                                }

                                let mut sorted_window: Vec<f64> = window.iter().cloned().collect();
                                sorted_window.sort_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                let median = if period % 2 == 0 {
                                    (sorted_window[period / 2 - 1] + sorted_window[period / 2])
                                        / 2.0
                                } else {
                                    sorted_window[period / 2]
                                };
                                let identical_count = window
                                    .iter()
                                    .filter(|&&x| (x - median).abs() < 1e-14)
                                    .count();
                                if identical_count > period / 2 {
                                    prop_assert!(
										y < 1e-9,
										"MedAD should be ~0 when >50% values are identical at index {}: {}",
										i,
										y
									);
                                }
                            }
                        }
                        _ => {}
                    }

                    if i >= warmup_period + period && y.is_finite() {
                        let old_idx = i - period - 1;
                        if old_idx < data.len() {
                            let current_window = &data[i + 1 - period..=i];
                            let window_variance = match devtype {
                                0 => {
                                    let mean = current_window.iter().sum::<f64>() / (period as f64);
                                    let var = current_window
                                        .iter()
                                        .map(|&x| (x - mean).powi(2))
                                        .sum::<f64>()
                                        / (period as f64);
                                    var.sqrt()
                                }
                                1 => {
                                    let mean = current_window.iter().sum::<f64>() / (period as f64);
                                    current_window
                                        .iter()
                                        .map(|&x| (x - mean).abs())
                                        .sum::<f64>()
                                        / (period as f64)
                                }
                                2 => y,
                                _ => y,
                            };

                            if devtype != 2 {
                                let diff = (y - window_variance).abs();
                                let tolerance = window_variance * 1e-6 + 1e-8;
                                prop_assert!(
									diff <= tolerance,
									"Rolling window deviation mismatch at index {}: computed {} vs expected {} (diff={})",
									i,
									y,
									window_variance,
									diff
								);
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let output = DeviationBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&data)?;
        let def = DeviationParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), data.len());
        let single = DeviationInput::from_slice(&data, def.clone());
        let single_out = deviation_with_kernel(&single, kernel)?.values;
        for (i, (r, s)) in row.iter().zip(single_out.iter()).enumerate() {
            if r.is_nan() && s.is_nan() {
                continue;
            }
            assert!(
                (r - s).abs() < 1e-12,
                "[{test}] default-row batch mismatch at idx {}: {} vs {}",
                i,
                r,
                s
            );
        }
        Ok(())
    }
    fn check_batch_varying_params(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let batch_output = DeviationBatchBuilder::new()
            .period_range(2, 3, 1)
            .devtype_range(0, 2, 1)
            .kernel(kernel)
            .apply_slice(&data)?;
        assert!(batch_output.rows >= 3, "[{test}] Not enough batch rows");
        for params in &batch_output.combos {
            let single = DeviationInput::from_slice(&data, params.clone());
            let single_out = deviation_with_kernel(&single, kernel)?.values;
            let row = batch_output.values_for(params).unwrap();
            for (i, (r, s)) in row.iter().zip(single_out.iter()).enumerate() {
                if r.is_nan() && s.is_nan() {
                    continue;
                }
                assert!(
                    (r - s).abs() < 1e-12,
                    "[{test}] batch grid row mismatch at idx {}: {} vs {}",
                    i,
                    r,
                    s
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let data = c.select_candle_field("close")?;

        let test_configs = vec![
            (2, 10, 2, 0, 2, 1),
            (5, 25, 5, 0, 0, 0),
            (5, 25, 5, 1, 1, 0),
            (5, 25, 5, 2, 2, 0),
            (30, 60, 15, 0, 2, 1),
            (2, 5, 1, 0, 2, 1),
            (50, 100, 25, 0, 0, 0),
            (10, 10, 0, 0, 2, 1),
        ];

        for (cfg_idx, &(p_start, p_end, p_step, d_start, d_end, d_step)) in
            test_configs.iter().enumerate()
        {
            let output = DeviationBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .devtype_range(d_start, d_end, d_step)
                .apply_slice(&data)?;

            for (idx, &val) in output.values.iter().enumerate() {
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
						 at row {} col {} (flat index {}) with params: period={}, devtype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9),
                        combo.devtype.unwrap_or(0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, devtype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9),
                        combo.devtype.unwrap_or(0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, devtype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(9),
                        combo.devtype.unwrap_or(0)
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

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_varying_params);
    gen_batch_tests!(check_batch_no_poison);
}
