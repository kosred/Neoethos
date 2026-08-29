use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::detect_best_batch_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use thiserror::Error;

#[inline(always)]
fn cci_cycle_candle_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    if source.eq_ignore_ascii_case("close") {
        &candles.close
    } else {
        source_type(candles, source)
    }
}

impl<'a> AsRef<[f64]> for CciCycleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CciCycleData::Slice(slice) => slice,
            CciCycleData::Candles { candles, source } => cci_cycle_candle_source(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CciCycleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CciCycleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CciCycleParams {
    pub length: Option<usize>,
    pub factor: Option<f64>,
}

impl Default for CciCycleParams {
    fn default() -> Self {
        Self {
            length: Some(10),
            factor: Some(0.5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CciCycleInput<'a> {
    pub data: CciCycleData<'a>,
    pub params: CciCycleParams,
}

impl<'a> CciCycleInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CciCycleParams) -> Self {
        Self {
            data: CciCycleData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CciCycleParams) -> Self {
        Self {
            data: CciCycleData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CciCycleParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(10)
    }

    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CciCycleBuilder {
    length: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for CciCycleBuilder {
    fn default() -> Self {
        Self {
            length: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CciCycleBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, val: usize) -> Self {
        self.length = Some(val);
        self
    }

    #[inline(always)]
    pub fn factor(mut self, val: f64) -> Self {
        self.factor = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CciCycleOutput, CciCycleError> {
        let p = CciCycleParams {
            length: self.length,
            factor: self.factor,
        };
        let i = CciCycleInput::from_candles(c, "close", p);
        cci_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CciCycleOutput, CciCycleError> {
        let p = CciCycleParams {
            length: self.length,
            factor: self.factor,
        };
        let i = CciCycleInput::from_slice(d, p);
        cci_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CciCycleStream, CciCycleError> {
        let p = CciCycleParams {
            length: self.length,
            factor: self.factor,
        };
        CciCycleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CciCycleError {
    #[error("cci_cycle: Input data slice is empty.")]
    EmptyInputData,

    #[error("cci_cycle: All values are NaN.")]
    AllValuesNaN,

    #[error("cci_cycle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("cci_cycle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("cci_cycle: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("cci_cycle: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },

    #[error("cci_cycle: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("cci_cycle: Invalid factor: {factor}")]
    InvalidFactor { factor: f64 },

    #[error("cci_cycle: invalid input: {0}")]
    InvalidInput(String),

    #[error("cci_cycle: CCI calculation failed: {0}")]
    CciError(String),

    #[error("cci_cycle: EMA calculation failed: {0}")]
    EmaError(String),

    #[error("cci_cycle: SMMA calculation failed: {0}")]
    SmmaError(String),
}

/// Classic semantic-v9 is the creator-aligned, local-current-resolution
/// VectorTA implementation with finite-segment resets.
pub const CCI_CYCLE_CLASSIC_SEMANTIC_VERSION: u32 = 9;
pub const CCI_CYCLE_CLASSIC_SEMANTIC_IDENTITY: &str =
    "cci-cycle-classic-v9-local-current-resolution-finite-segment-reset-v1";
/// Primary creator source used only as a mathematical audit oracle. VectorTA
/// remains the sole runtime implementation.
pub const CCI_CYCLE_CREATOR_AUDIT_ORACLE_URL: &str = "https://pine-facade.tradingview.com/pine-facade/get/PUB%3B4YdqejUxlibWlfiGTKzYYBMwmnDSUxc3/1?no_4xx=true";
pub const CCI_CYCLE_CREATOR_AUDIT_ORACLE_SHA256: &str =
    "d00a0186f28989a34eb1da24eb9fae9a8906736afe413e2492ded9dc4b2a9c9f";

#[derive(Debug, Clone)]
struct SmaSeededAverage {
    period: usize,
    alpha: f64,
    seed_sum: f64,
    seed_count: usize,
    state: Option<f64>,
}

impl SmaSeededAverage {
    #[inline]
    fn new(period: usize, alpha: f64) -> Self {
        Self {
            period,
            alpha,
            seed_sum: 0.0,
            seed_count: 0,
            state: None,
        }
    }

    #[inline]
    fn update(&mut self, value: Option<f64>) -> Option<f64> {
        if let Some(value) = value.filter(|value| value.is_finite()) {
            if let Some(state) = self.state.as_mut() {
                *state += self.alpha * (value - *state);
            } else {
                self.seed_sum += value;
                self.seed_count += 1;
                if self.seed_count == self.period {
                    self.state = Some(self.seed_sum / self.period as f64);
                }
            }
        }
        self.state
    }
}

#[derive(Debug, Clone)]
struct CciCycleClassicV9State {
    length: usize,
    factor: f64,
    close_window: Vec<f64>,
    ema_short: SmaSeededAverage,
    ema_long: SmaSeededAverage,
    rma: SmaSeededAverage,
    ccis_window: Vec<f64>,
    pf_window: Vec<f64>,
    previous_f1: f64,
    previous_pf: f64,
    previous_f2: f64,
    previous_pff: f64,
}

impl CciCycleClassicV9State {
    #[inline]
    fn new(length: usize, factor: f64) -> Self {
        let half = length / 2;
        let rma_length = ((length as f64).sqrt().round() as usize).max(1);
        Self {
            length,
            factor,
            close_window: Vec::with_capacity(length),
            ema_short: SmaSeededAverage::new(half, 2.0 / (half as f64 + 1.0)),
            ema_long: SmaSeededAverage::new(length, 2.0 / (length as f64 + 1.0)),
            rma: SmaSeededAverage::new(rma_length, 1.0 / rma_length as f64),
            ccis_window: Vec::with_capacity(length),
            pf_window: Vec::with_capacity(length),
            previous_f1: 0.0,
            previous_pf: 0.0,
            previous_f2: 0.0,
            previous_pff: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::new(self.length, self.factor);
    }

    #[inline]
    fn push_window(window: &mut Vec<f64>, value: f64, length: usize) {
        if window.len() == length {
            window.remove(0);
        }
        window.push(value);
    }

    #[inline]
    fn finite_range(window: &[f64]) -> Option<(f64, f64)> {
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for &value in window {
            if value.is_finite() {
                low = low.min(value);
                high = high.max(value);
            }
        }
        if low.is_finite() && high.is_finite() {
            Some((low, high))
        } else {
            None
        }
    }

    #[inline]
    fn cci(&self, close: f64) -> Option<f64> {
        if self.close_window.len() != self.length {
            return None;
        }
        let mean = self.close_window.iter().sum::<f64>() / self.length as f64;
        let deviation = self
            .close_window
            .iter()
            .map(|value| (value - mean).abs())
            .sum::<f64>()
            / self.length as f64;
        if deviation > 0.0 && deviation.is_finite() {
            let value = (close - mean) / (0.015 * deviation);
            value.is_finite().then_some(value)
        } else {
            None
        }
    }

    #[inline]
    fn update(&mut self, close: f64) -> Option<f64> {
        if !close.is_finite() {
            self.reset();
            return None;
        }

        Self::push_window(&mut self.close_window, close, self.length);
        let cci = self.cci(close);
        let ema_short = self.ema_short.update(cci);
        let ema_long = self.ema_long.update(cci);
        let de = ema_short
            .zip(ema_long)
            .map(|(short, long)| short + short - long)
            .filter(|value| value.is_finite());
        let ccis = self.rma.update(de);

        Self::push_window(&mut self.ccis_window, ccis.unwrap_or(f64::NAN), self.length);
        let f1 = match (ccis, Self::finite_range(&self.ccis_window)) {
            (Some(value), Some((low, high))) if high > low => (value - low) / (high - low) * 100.0,
            _ => self.previous_f1,
        };
        let pf = self.previous_pf + self.factor * (f1 - self.previous_pf);
        self.previous_f1 = f1;
        self.previous_pf = pf;

        Self::push_window(&mut self.pf_window, pf, self.length);
        let f2 = match Self::finite_range(&self.pf_window) {
            Some((low, high)) if high > low => (pf - low) / (high - low) * 100.0,
            _ => self.previous_f2,
        };
        let pff = self.previous_pff + self.factor * (f2 - self.previous_pff);
        self.previous_f2 = f2;
        self.previous_pff = pff;
        Some(pff)
    }
}

#[inline]
fn cci_cycle_classic_v9_into(data: &[f64], length: usize, factor: f64, out: &mut [f64]) {
    let mut state = CciCycleClassicV9State::new(length, factor);
    for (&value, output) in data.iter().zip(out) {
        *output = state.update(value).unwrap_or(f64::NAN);
    }
}

#[inline]
pub fn cci_cycle(input: &CciCycleInput) -> Result<CciCycleOutput, CciCycleError> {
    cci_cycle_with_kernel(input, Kernel::Auto)
}

pub fn cci_cycle_with_kernel(
    input: &CciCycleInput,
    kernel: Kernel,
) -> Result<CciCycleOutput, CciCycleError> {
    let (data, length, factor) = cci_cycle_prepare(input, kernel)?;
    let mut out = vec![f64::NAN; data.len()];
    cci_cycle_classic_v9_into(data, length, factor, &mut out);
    Ok(CciCycleOutput { values: out })
}

#[inline]
pub fn cci_cycle_into_slice(
    dst: &mut [f64],
    input: &CciCycleInput,
    kern: Kernel,
) -> Result<(), CciCycleError> {
    let (data, length, factor) = cci_cycle_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(CciCycleError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    cci_cycle_classic_v9_into(data, length, factor, dst);
    Ok(())
}

#[inline]
pub fn cci_cycle_into(input: &CciCycleInput, out: &mut [f64]) -> Result<(), CciCycleError> {
    cci_cycle_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
fn cci_cycle_prepare<'a>(
    input: &'a CciCycleInput,
    _kernel: Kernel,
) -> Result<(&'a [f64], usize, f64), CciCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(CciCycleError::EmptyInputData);
    }

    if !data.iter().any(|value| value.is_finite()) {
        return Err(CciCycleError::AllValuesNaN);
    }

    let length = input.get_length();
    let factor = input.get_factor();

    if length < 2 || length > len {
        return Err(CciCycleError::InvalidPeriod {
            period: length,
            data_len: len,
        });
    }

    if !factor.is_finite() {
        return Err(CciCycleError::InvalidFactor { factor });
    }

    Ok((data, length, factor))
}

#[derive(Debug, Clone)]
pub struct CciCycleStream {
    state: CciCycleClassicV9State,
}

impl CciCycleStream {
    #[inline]
    pub fn try_new(params: CciCycleParams) -> Result<Self, CciCycleError> {
        let length = params.length.unwrap_or(10);
        let factor = params.factor.unwrap_or(0.5);
        if length < 2 {
            return Err(CciCycleError::InvalidPeriod {
                period: length,
                data_len: 0,
            });
        }
        if !factor.is_finite() {
            return Err(CciCycleError::InvalidFactor { factor });
        }
        Ok(Self {
            state: CciCycleClassicV9State::new(length, factor),
        })
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.state.update(value)
    }
}

#[derive(Clone, Debug)]
pub struct CciCycleBatchRange {
    pub length: (usize, usize, usize),
    pub factor: (f64, f64, f64),
}

impl Default for CciCycleBatchRange {
    fn default() -> Self {
        Self {
            length: (10, 259, 1),
            factor: (0.5, 0.5, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CciCycleBatchBuilder {
    range: CciCycleBatchRange,
    kernel: Kernel,
}

impl CciCycleBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, val: usize) -> Self {
        self.range.length = (val, val, 0);
        self
    }

    #[inline]
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }

    #[inline]
    pub fn factor_static(mut self, val: f64) -> Self {
        self.range.factor = (val, val, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<CciCycleBatchOutput, CciCycleError> {
        cci_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<CciCycleBatchOutput, CciCycleError> {
        let data = source_type(c, src);
        cci_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<CciCycleBatchOutput, CciCycleError> {
        CciCycleBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(
        c: &Candles,
        k: Kernel,
    ) -> Result<CciCycleBatchOutput, CciCycleError> {
        CciCycleBatchBuilder::new()
            .kernel(k)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct CciCycleBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CciCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CciCycleBatchOutput {
    pub fn row_for_params(&self, p: &CciCycleParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(10) == p.length.unwrap_or(10)
                && (c.factor.unwrap_or(0.5) - p.factor.unwrap_or(0.5)).abs() < 1e-12
        })
    }

    pub fn values_for(&self, p: &CciCycleParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CciCycleBatchRange) -> Result<Vec<CciCycleParams>, CciCycleError> {
    fn axis_usize((s, e, st): (usize, usize, usize)) -> Result<Vec<usize>, CciCycleError> {
        if st == 0 || s == e {
            return Ok(vec![s]);
        }
        let mut vals = Vec::new();
        if s < e {
            let mut v = s;
            while v <= e {
                vals.push(v);
                v = match v.checked_add(st) {
                    Some(n) => n,
                    None => break,
                };
            }
        } else {
            let mut v = s;
            while v >= e {
                vals.push(v);
                if v < st {
                    break;
                }
                v -= st;
                if v == 0 && e > 0 {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(CciCycleError::InvalidRange {
                start: s.to_string(),
                end: e.to_string(),
                step: st.to_string(),
            });
        }
        Ok(vals)
    }
    fn axis_f64((s, e, st): (f64, f64, f64)) -> Result<Vec<f64>, CciCycleError> {
        if !st.is_finite() {
            return Err(CciCycleError::InvalidRange {
                start: s.to_string(),
                end: e.to_string(),
                step: st.to_string(),
            });
        }
        if st.abs() < 1e-12 || (s - e).abs() < 1e-12 {
            return Ok(vec![s]);
        }
        let mut vals = Vec::new();
        let step = st.abs();
        let eps = 1e-12;
        if s <= e {
            let mut x = s;
            while x <= e + eps {
                vals.push(x);
                x += step;
            }
        } else {
            let mut x = s;
            while x >= e - eps {
                vals.push(x);
                x -= step;
            }
        }
        if vals.is_empty() {
            return Err(CciCycleError::InvalidRange {
                start: s.to_string(),
                end: e.to_string(),
                step: st.to_string(),
            });
        }
        Ok(vals)
    }
    let lens = axis_usize(r.length)?;
    let facts = axis_f64(r.factor)?;
    let cap = lens
        .len()
        .checked_mul(facts.len())
        .ok_or_else(|| CciCycleError::InvalidInput("rows*cols overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &l in &lens {
        for &f in &facts {
            out.push(CciCycleParams {
                length: Some(l),
                factor: Some(f),
            });
        }
    }
    Ok(out)
}

pub fn cci_cycle_batch_with_kernel(
    data: &[f64],
    sweep: &CciCycleBatchRange,
    k: Kernel,
) -> Result<CciCycleBatchOutput, CciCycleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(CciCycleError::InvalidKernelForBatch(other)),
    };

    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(CciCycleError::AllValuesNaN);
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| CciCycleError::InvalidInput("rows*cols overflow".into()))?;

    let mut values = vec![f64::NAN; total];

    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), CciCycleError> {
        let prm = combos[row].clone();
        let inp = CciCycleInput::from_slice(data, prm);

        let rk = match kernel {
            Kernel::ScalarBatch => Kernel::Scalar,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::Avx512Batch => Kernel::Avx512,
            _ => Kernel::Scalar,
        };
        cci_cycle_into_slice(dst, &inp, rk)
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        values
            .par_chunks_mut(cols)
            .enumerate()
            .try_for_each(|(r, s)| do_row(r, s))?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (r, slice) in values.chunks_mut(cols).enumerate() {
            do_row(r, slice)?;
        }
    }

    Ok(CciCycleBatchOutput {
        values,
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
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    macro_rules! generate_all_cci_cycle_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar)
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2)
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx512>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512)
                    }
                )*
            }
        };
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch)
                }
            }
        };
    }

    fn check_cci_cycle_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CciCycleInput::from_candles(&candles, "close", CciCycleParams::default());
        let result = cci_cycle_with_kernel(&input, kernel)?;

        let expected_last_five = [
            9.25177192,
            20.49219826,
            35.42917181,
            55.57843075,
            77.78921538,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] CCI_CYCLE {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_cci_cycle_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut data: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.037).sin() * 2.0 + (i as f64) * 0.01)
            .collect();
        data[0] = f64::NAN;
        data[1] = f64::NAN;
        data[2] = f64::NAN;

        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&data, params);

        let baseline = cci_cycle(&input)?.values;

        let mut out = vec![0.0; data.len()];
        cci_cycle_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..out.len() {
            let a = baseline[i];
            let b = out[i];
            assert!(
                eq_or_both_nan(a, b) || (a - b).abs() <= 1e-12,
                "cci_cycle_into parity mismatch at {}: got {}, expected {}",
                i,
                b,
                a
            );
        }
        Ok(())
    }

    fn check_cci_cycle_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let out = cci_cycle_with_kernel(&CciCycleInput::with_default_candles(&c), kernel)?.values;
        for (i, &v) in out.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x11111111_11111111,
                "[{}] alloc_with_nan_prefix poison at {}",
                test_name, i
            );
            assert_ne!(
                b, 0x22222222_22222222,
                "[{}] init_matrix_prefixes poison at {}",
                test_name, i
            );
            assert_ne!(
                b, 0x33333333_33333333,
                "[{}] make_uninit_matrix poison at {}",
                test_name, i
            );
        }
        Ok(())
    }

    fn check_cci_cycle_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = CciCycleParams {
            length: None,
            factor: None,
        };
        let input = CciCycleInput::from_candles(&candles, "close", default_params);
        let output = cci_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cci_cycle_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CciCycleInput::with_default_candles(&candles);
        match input.data {
            CciCycleData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CciCycleData::Candles"),
        }
        let output = cci_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cci_cycle_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CciCycleParams {
            length: Some(0),
            factor: None,
        };
        let input = CciCycleInput::from_slice(&input_data, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cci_cycle_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CciCycleParams {
            length: Some(10),
            factor: None,
        };
        let input = CciCycleInput::from_slice(&data_small, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cci_cycle_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&single_point, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cci_cycle_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&empty, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_cci_cycle_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = CciCycleParams::default();
        let input = CciCycleInput::from_slice(&nan_data, params);
        let res = cci_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI_CYCLE should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_cci_cycle_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CciCycleInput::from_candles(&candles, "close", CciCycleParams::default());
        let output1 = cci_cycle_with_kernel(&input, kernel)?;

        let input2 = CciCycleInput::from_slice(&output1.values, CciCycleParams::default());
        let output2 = cci_cycle_with_kernel(&input2, kernel)?;

        assert_eq!(output2.values.len(), output1.values.len());

        let non_nan_count = output2.values.iter().filter(|&&v| !v.is_nan()).count();
        assert!(
            non_nan_count > 0,
            "[{}] Reinput produced all NaN values",
            test_name
        );

        Ok(())
    }

    fn check_cci_cycle_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data_with_nans = vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            f64::NAN,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            35.0,
            36.0,
            37.0,
            38.0,
            39.0,
            40.0,
        ];

        let params = CciCycleParams {
            length: Some(5),
            factor: Some(0.5),
        };
        let input = CciCycleInput::from_slice(&data_with_nans, params.clone());
        let result = cci_cycle_with_kernel(&input, kernel);

        assert!(
            result.is_ok(),
            "[{}] Should handle data with some NaN values",
            test_name
        );

        if let Ok(output) = result {
            assert_eq!(output.values.len(), data_with_nans.len());

            let valid_count = output.values.iter().filter(|&&v| !v.is_nan()).count();
            assert!(
                valid_count > 0,
                "[{}] Should produce some valid values",
                test_name
            );
        }

        let mostly_nans = vec![f64::NAN; 20];
        let input2 = CciCycleInput::from_slice(&mostly_nans, params);
        let result2 = cci_cycle_with_kernel(&input2, kernel);
        assert!(
            result2.is_err(),
            "[{}] Should fail with all NaN values",
            test_name
        );

        Ok(())
    }

    fn check_cci_cycle_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        let params = CciCycleParams {
            length: Some(10),
            factor: Some(0.5),
        };

        let stream_result = CciCycleStream::try_new(params.clone());
        assert!(
            stream_result.is_ok(),
            "[{}] Stream creation should succeed",
            test_name
        );

        let mut stream = stream_result?;

        let test_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];

        for &value in &test_data {
            let _ = stream.update(value);
        }

        Ok(())
    }

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file)?;

        let output = CciCycleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles, "close")?;

        let default_params = CciCycleParams::default();
        let row = output.values_for(&default_params);

        assert!(
            row.is_some(),
            "[{}] Default parameters not found in batch output",
            test_name
        );

        if let Some(values) = row {
            assert_eq!(values.len(), candles.close.len());

            let non_nan_count = values.iter().filter(|&&v| !v.is_nan()).count();
            assert!(
                non_nan_count > 0,
                "[{}] Default row has no valid values",
                test_name
            );
        }

        assert_eq!(output.cols, candles.close.len());

        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = vec![1.0; 100];

        let output = CciCycleBatchBuilder::new()
            .kernel(kernel)
            .length_range(10, 20, 5)
            .factor_range(0.3, 0.7, 0.2)
            .apply_slice(&data)?;

        assert_eq!(
            output.combos.len(),
            9,
            "[{}] Unexpected number of parameter combinations",
            test_name
        );
        assert_eq!(output.rows, 9);
        assert_eq!(output.cols, 100);
        assert_eq!(output.values.len(), 900);

        Ok(())
    }

    generate_all_cci_cycle_tests!(
        check_cci_cycle_accuracy,
        check_cci_cycle_no_poison,
        check_cci_cycle_partial_params,
        check_cci_cycle_default_candles,
        check_cci_cycle_zero_period,
        check_cci_cycle_period_exceeds_length,
        check_cci_cycle_very_small_dataset,
        check_cci_cycle_empty_input,
        check_cci_cycle_all_nan,
        check_cci_cycle_reinput,
        check_cci_cycle_nan_handling,
        check_cci_cycle_streaming
    );

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_cci_cycle_no_panic(data: Vec<f64>, length in 1usize..100) {
            let params = CciCycleParams {
                length: Some(length),
                factor: Some(0.5),
            };
            let input = CciCycleInput::from_slice(&data, params);
            let _ = cci_cycle(&input);
        }

        #[test]
        fn test_cci_cycle_length_preservation(size in 10usize..100) {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let params = CciCycleParams::default();
            let input = CciCycleInput::from_slice(&data, params);

            if let Ok(output) = cci_cycle(&input) {
                prop_assert_eq!(output.values.len(), size);
            }
        }
    }
}
