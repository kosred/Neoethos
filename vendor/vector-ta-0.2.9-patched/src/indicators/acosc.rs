use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use thiserror::Error;

const ACOSC_MEDIAN_FAST_PERIOD: usize = 5;
const ACOSC_MEDIAN_SLOW_PERIOD: usize = 34;
const ACOSC_AO_SIGNAL_PERIOD: usize = 5;
const ACOSC_FIRST_VALUE_BARS: usize = ACOSC_MEDIAN_SLOW_PERIOD + ACOSC_AO_SIGNAL_PERIOD - 1;
const ACOSC_QNAN: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

#[derive(Debug, Clone)]
pub enum AcoscData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone, Default)]
pub struct AcoscParams {}

#[derive(Debug, Clone)]
pub struct AcoscInput<'a> {
    pub data: AcoscData<'a>,
    pub params: AcoscParams,
}
impl<'a> AcoscInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AcoscParams) -> Self {
        Self {
            data: AcoscData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: AcoscParams) -> Self {
        Self {
            data: AcoscData::Slices { high, low },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AcoscData::Candles { candles },
            params: AcoscParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcoscOutput {
    pub osc: Vec<f64>,
    pub change: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcoscOutputField {
    Osc,
    Change,
}

#[derive(Debug, Error)]
pub enum AcoscError {
    #[error("acosc: Failed to get high/low fields from candles: {msg}")]
    CandleFieldError { msg: String },
    #[error(
        "acosc: Mismatch in high/low candle data lengths: high_len={high_len}, low_len={low_len}"
    )]
    LengthMismatch { high_len: usize, low_len: usize },
    #[error("acosc: Empty input data")]
    EmptyInputData,
    #[error("acosc: Not enough data: all values are NaN")]
    AllValuesNaN,
    #[error("acosc: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("acosc: Not enough data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("acosc: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("acosc: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: i64, end: i64, step: i64 },
    #[error("acosc: Invalid kernel for batch operation. Expected batch kernel, got: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("acosc: Not enough data points: required={required}, actual={actual}")]
    NotEnoughData { required: usize, actual: usize },
    #[error("acosc: Invalid kernel for batch operation. Expected batch kernel, got: {kernel:?}")]
    InvalidBatchKernel { kernel: Kernel },
}

#[inline]
pub fn acosc(input: &AcoscInput) -> Result<AcoscOutput, AcoscError> {
    acosc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn acosc_prepare<'a>(
    input: &'a AcoscInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], Kernel), AcoscError> {
    let (high, low) = match &input.data {
        AcoscData::Candles { candles } => {
            let h = candles.high.as_slice();
            let l = candles.low.as_slice();
            (h, l)
        }
        AcoscData::Slices { high, low } => (*high, *low),
    };

    if high.len() != low.len() {
        return Err(AcoscError::LengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let len = high.len();
    if len == 0 {
        return Err(AcoscError::EmptyInputData);
    }
    let mut current_run = 0usize;
    let mut longest_run = 0usize;
    for (&high_value, &low_value) in high.iter().zip(low) {
        if high_value.is_finite() && low_value.is_finite() {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    if longest_run == 0 {
        return Err(AcoscError::AllValuesNaN);
    }
    if longest_run < ACOSC_FIRST_VALUE_BARS {
        return Err(AcoscError::NotEnoughValidData {
            needed: ACOSC_FIRST_VALUE_BARS,
            valid: longest_run,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    Ok((high, low, chosen))
}
pub fn acosc_with_kernel(input: &AcoscInput, kernel: Kernel) -> Result<AcoscOutput, AcoscError> {
    let (high, low, chosen) = acosc_prepare(input, kernel)?;

    let len = low.len();
    let mut osc = vec![ACOSC_QNAN; len];
    let mut change = vec![ACOSC_QNAN; len];
    acosc_compute_into(high, low, chosen, &mut osc, &mut change);

    Ok(AcoscOutput { osc, change })
}

#[inline(always)]
fn acosc_compute_into(
    high: &[f64],
    low: &[f64],
    kernel: Kernel,
    osc_out: &mut [f64],
    change_out: &mut [f64],
) {
    match kernel {
        Kernel::Scalar | Kernel::ScalarBatch => acosc_scalar(high, low, osc_out, change_out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => acosc_avx2(high, low, osc_out, change_out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => acosc_avx512(high, low, osc_out, change_out),
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            acosc_scalar(high, low, osc_out, change_out)
        }
        Kernel::Auto => unreachable!("Kernel::Auto should be resolved before calling compute_into"),
    }
}

#[derive(Debug, Clone)]
struct AcoscState {
    median_fast: [f64; ACOSC_MEDIAN_FAST_PERIOD],
    median_slow: [f64; ACOSC_MEDIAN_SLOW_PERIOD],
    ao_signal: [f64; ACOSC_AO_SIGNAL_PERIOD],
    median_fast_sum: f64,
    median_slow_sum: f64,
    ao_signal_sum: f64,
    median_fast_index: usize,
    median_slow_index: usize,
    ao_signal_index: usize,
    median_count: usize,
    ao_count: usize,
    previous_ac: Option<f64>,
}

impl Default for AcoscState {
    fn default() -> Self {
        Self {
            median_fast: [0.0; ACOSC_MEDIAN_FAST_PERIOD],
            median_slow: [0.0; ACOSC_MEDIAN_SLOW_PERIOD],
            ao_signal: [0.0; ACOSC_AO_SIGNAL_PERIOD],
            median_fast_sum: 0.0,
            median_slow_sum: 0.0,
            ao_signal_sum: 0.0,
            median_fast_index: 0,
            median_slow_index: 0,
            ao_signal_index: 0,
            median_count: 0,
            ao_count: 0,
            previous_ac: None,
        }
    }
}

impl AcoscState {
    #[inline(always)]
    fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64) -> (f64, f64) {
        if !high.is_finite() || !low.is_finite() {
            self.reset();
            return (ACOSC_QNAN, ACOSC_QNAN);
        }

        let median = (high + low) * 0.5;
        if !median.is_finite() {
            self.reset();
            return (ACOSC_QNAN, ACOSC_QNAN);
        }

        if self.median_count < ACOSC_MEDIAN_FAST_PERIOD {
            self.median_fast[self.median_count] = median;
            self.median_fast_sum += median;
        } else {
            self.median_fast_sum += median - self.median_fast[self.median_fast_index];
            self.median_fast[self.median_fast_index] = median;
            self.median_fast_index = (self.median_fast_index + 1) % ACOSC_MEDIAN_FAST_PERIOD;
        }

        if self.median_count < ACOSC_MEDIAN_SLOW_PERIOD {
            self.median_slow[self.median_count] = median;
            self.median_slow_sum += median;
            self.median_count += 1;
        } else {
            self.median_slow_sum += median - self.median_slow[self.median_slow_index];
            self.median_slow[self.median_slow_index] = median;
            self.median_slow_index = (self.median_slow_index + 1) % ACOSC_MEDIAN_SLOW_PERIOD;
        }

        if self.median_count < ACOSC_MEDIAN_SLOW_PERIOD {
            return (ACOSC_QNAN, ACOSC_QNAN);
        }

        let ao = self.median_fast_sum / ACOSC_MEDIAN_FAST_PERIOD as f64
            - self.median_slow_sum / ACOSC_MEDIAN_SLOW_PERIOD as f64;
        if !ao.is_finite() {
            self.reset();
            return (ACOSC_QNAN, ACOSC_QNAN);
        }

        if self.ao_count < ACOSC_AO_SIGNAL_PERIOD {
            self.ao_signal[self.ao_count] = ao;
            self.ao_signal_sum += ao;
            self.ao_count += 1;
            if self.ao_count < ACOSC_AO_SIGNAL_PERIOD {
                return (ACOSC_QNAN, ACOSC_QNAN);
            }
        } else {
            self.ao_signal_sum += ao - self.ao_signal[self.ao_signal_index];
            self.ao_signal[self.ao_signal_index] = ao;
            self.ao_signal_index = (self.ao_signal_index + 1) % ACOSC_AO_SIGNAL_PERIOD;
        }

        let ac = ao - self.ao_signal_sum / ACOSC_AO_SIGNAL_PERIOD as f64;
        let change = self
            .previous_ac
            .map_or(ACOSC_QNAN, |previous| ac - previous);
        self.previous_ac = Some(ac);
        (ac, change)
    }
}

#[inline(always)]
pub fn acosc_scalar(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(osc.len(), high.len());
    debug_assert_eq!(change.len(), high.len());
    osc.fill(ACOSC_QNAN);
    change.fill(ACOSC_QNAN);

    let mut state = AcoscState::default();
    for (i, (&high_value, &low_value)) in high.iter().zip(low).enumerate() {
        let (ac, delta) = state.update(high_value, low_value);
        osc[i] = ac;
        change[i] = delta;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn acosc_avx512(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn acosc_avx2(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}
#[inline]
pub fn acosc_avx512_short(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}
#[inline]
pub fn acosc_avx512_long(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}

#[derive(Debug, Clone)]
pub struct AcoscStream {
    state: AcoscState,
}
impl AcoscStream {
    pub fn try_new(_params: AcoscParams) -> Result<Self, AcoscError> {
        Ok(Self {
            state: AcoscState::default(),
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        let output = self.state.update(high, low);
        output.0.is_finite().then_some(output)
    }
}

#[derive(Clone, Debug)]
pub struct AcoscBatchRange {}

impl Default for AcoscBatchRange {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug, Default)]
pub struct AcoscBatchBuilder {
    kernel: Kernel,
}
impl AcoscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<AcoscBatchOutput, AcoscError> {
        acosc_batch_with_kernel(high, low, self.kernel)
    }
    pub fn with_default_slice(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<AcoscBatchOutput, AcoscError> {
        AcoscBatchBuilder::new().kernel(k).apply_slice(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AcoscBatchOutput, AcoscError> {
        let high = c
            .select_candle_field("high")
            .map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
        let low = c
            .select_candle_field("low")
            .map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
        self.apply_slice(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AcoscBatchOutput, AcoscError> {
        AcoscBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}
#[derive(Clone, Debug)]
pub struct AcoscBatchOutput {
    pub osc: Vec<f64>,
    pub change: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
pub fn acosc_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    k: Kernel,
) -> Result<AcoscBatchOutput, AcoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AcoscError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    acosc_batch_par_slice(high, low, simd)
}
#[inline(always)]
pub fn acosc_batch_slice(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<AcoscBatchOutput, AcoscError> {
    acosc_batch_inner(high, low, kern, false)
}
#[inline(always)]
pub fn acosc_batch_par_slice(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
) -> Result<AcoscBatchOutput, AcoscError> {
    acosc_batch_inner(high, low, kern, true)
}
#[inline(always)]
fn acosc_batch_inner(
    high: &[f64],
    low: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<AcoscBatchOutput, AcoscError> {
    let cols = high.len();
    let rows: usize = 1;

    let _total = rows.checked_mul(cols).ok_or(AcoscError::InvalidRange {
        start: 0,
        end: cols as i64,
        step: 0,
    })?;

    let simd = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let input = AcoscInput::from_slices(high, low, AcoscParams::default());
    let AcoscOutput { osc, change } = acosc_with_kernel(&input, simd)?;

    Ok(AcoscBatchOutput {
        osc,
        change,
        rows,
        cols,
    })
}
#[inline(always)]
pub fn expand_grid(_r: &AcoscBatchRange) -> Vec<AcoscParams> {
    vec![AcoscParams::default()]
}

#[inline(always)]
pub unsafe fn acosc_row_scalar(
    high: &[f64],
    low: &[f64],
    out_osc: &mut [f64],
    out_change: &mut [f64],
) {
    acosc_scalar(high, low, out_osc, out_change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn acosc_row_avx2(
    high: &[f64],
    low: &[f64],
    out_osc: &mut [f64],
    out_change: &mut [f64],
) {
    acosc_avx2(high, low, out_osc, out_change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn acosc_row_avx512(
    high: &[f64],
    low: &[f64],
    out_osc: &mut [f64],
    out_change: &mut [f64],
) {
    acosc_avx512(high, low, out_osc, out_change)
}
#[inline(always)]
pub fn acosc_row_avx512_short(
    high: &[f64],
    low: &[f64],
    out_osc: &mut [f64],
    out_change: &mut [f64],
) {
    acosc_scalar(high, low, out_osc, out_change)
}
#[inline(always)]
pub fn acosc_row_avx512_long(
    high: &[f64],
    low: &[f64],
    out_osc: &mut [f64],
    out_change: &mut [f64],
) {
    acosc_scalar(high, low, out_osc, out_change)
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AcoscBuilder {
    kernel: Kernel,
}
impl AcoscBuilder {
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
    pub fn apply_candles(self, candles: &Candles) -> Result<AcoscOutput, AcoscError> {
        let input = AcoscInput::with_default_candles(candles);
        acosc_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AcoscOutput, AcoscError> {
        let input = AcoscInput::from_slices(high, low, AcoscParams::default());
        acosc_with_kernel(&input, self.kernel)
    }
}

pub fn acosc_into_slice(
    osc_dst: &mut [f64],
    change_dst: &mut [f64],
    input: &AcoscInput,
    kern: Kernel,
) -> Result<(), AcoscError> {
    let (high, low, kernel) = acosc_prepare(input, kern)?;

    if osc_dst.len() != high.len() {
        return Err(AcoscError::OutputLengthMismatch {
            expected: high.len(),
            got: osc_dst.len(),
        });
    }
    if change_dst.len() != high.len() {
        return Err(AcoscError::OutputLengthMismatch {
            expected: high.len(),
            got: change_dst.len(),
        });
    }

    acosc_compute_into(high, low, kernel, osc_dst, change_dst);
    Ok(())
}

pub fn acosc_output_into_slice(
    dst: &mut [f64],
    input: &AcoscInput,
    kern: Kernel,
    field: AcoscOutputField,
) -> Result<(), AcoscError> {
    let (high, low, kernel) = acosc_prepare(input, kern)?;

    if dst.len() != high.len() {
        return Err(AcoscError::OutputLengthMismatch {
            expected: high.len(),
            got: dst.len(),
        });
    }

    let mut other = vec![ACOSC_QNAN; dst.len()];
    match field {
        AcoscOutputField::Osc => acosc_compute_into(high, low, kernel, dst, &mut other),
        AcoscOutputField::Change => acosc_compute_into(high, low, kernel, &mut other, dst),
    }
    Ok(())
}

#[inline]
pub fn acosc_into(
    input: &AcoscInput,
    osc_out: &mut [f64],
    change_out: &mut [f64],
) -> Result<(), AcoscError> {
    acosc_into_slice(osc_out, change_out, input, Kernel::Auto)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use std::error::Error;

    fn direct_definition_acosc(high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(high.len(), low.len());
        let mut ao = vec![f64::NAN; high.len()];
        let mut osc = vec![f64::NAN; high.len()];
        let mut change = vec![f64::NAN; high.len()];

        let mut segment_start = 0usize;
        while segment_start < high.len() {
            while segment_start < high.len()
                && (!high[segment_start].is_finite() || !low[segment_start].is_finite())
            {
                segment_start += 1;
            }
            if segment_start == high.len() {
                break;
            }

            let mut segment_end = segment_start;
            while segment_end < high.len()
                && high[segment_end].is_finite()
                && low[segment_end].is_finite()
            {
                segment_end += 1;
            }

            for i in (segment_start + 33)..segment_end {
                let sma5 = (i - 4..=i).map(|j| (high[j] + low[j]) * 0.5).sum::<f64>() / 5.0;
                let sma34 = (i - 33..=i).map(|j| (high[j] + low[j]) * 0.5).sum::<f64>() / 34.0;
                ao[i] = sma5 - sma34;
            }

            for i in (segment_start + 37)..segment_end {
                let ao_sma5 = ao[i - 4..=i].iter().sum::<f64>() / 5.0;
                osc[i] = ao[i] - ao_sma5;
                if i > segment_start + 37 {
                    change[i] = osc[i] - osc[i - 1];
                }
            }

            segment_start = segment_end.saturating_add(1);
        }

        (osc, change)
    }

    fn nonlinear_fixture(len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            let median = 100.0
                + x * 0.17
                + (x * 0.31).sin() * 2.75
                + ((i * i * 7 + 3 * i) % 19) as f64 * 0.013;
            let half_range = 0.25 + ((i * 11) % 7) as f64 * 0.031;
            high.push(median + half_range);
            low.push(median - half_range);
        }
        (high, low)
    }

    fn fixture_candles(len: usize) -> Candles {
        let (high, low) = nonlinear_fixture(len);
        let close: Vec<f64> = high
            .iter()
            .zip(&low)
            .map(|(&h, &l)| (h + l) * 0.5)
            .collect();
        let open = close.clone();
        let timestamp = (0..len).map(|i| i as i64 * 60_000).collect();
        let volume = (0..len).map(|i| 1000.0 + i as f64).collect();
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn assert_same_validity_and_close(label: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                got.is_finite(),
                want.is_finite(),
                "{label} validity mismatch at {i}: got={got:?}, expected={want:?}"
            );
            if want.is_finite() {
                let tolerance = 2.0e-12_f64.max(want.abs() * 2.0e-13);
                assert!(
                    (got - want).abs() <= tolerance,
                    "{label} value mismatch at {i}: got={got:.17e}, expected={want:.17e}, delta={:.3e}",
                    (got - want).abs()
                );
            }
        }
    }

    #[test]
    fn official_definition_linear_series_has_exact_warmup_and_change_validity() {
        let median: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let high: Vec<f64> = median.iter().map(|value| value + 1.0).collect();
        let low: Vec<f64> = median.iter().map(|value| value - 1.0).collect();
        let input = AcoscInput::from_slices(&high, &low, AcoscParams::default());

        let output = acosc_with_kernel(&input, Kernel::Scalar).unwrap();

        assert!(output.osc[..37].iter().all(|value| value.is_nan()));
        assert_eq!(output.osc[37].to_bits(), 0.0f64.to_bits());
        assert!(output.change[..=37].iter().all(|value| value.is_nan()));
        assert_eq!(output.change[38].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn scalar_matches_independent_direct_formula() {
        let (high, low) = nonlinear_fixture(121);
        let expected = direct_definition_acosc(&high, &low);
        let input = AcoscInput::from_slices(&high, &low, AcoscParams::default());

        let actual = acosc_with_kernel(&input, Kernel::Scalar).unwrap();

        assert_same_validity_and_close("osc", &actual.osc, &expected.0);
        assert_same_validity_and_close("change", &actual.change, &expected.1);
    }

    #[test]
    fn non_finite_gap_resets_the_entire_indicator_state() {
        let (mut high, mut low) = nonlinear_fixture(90);
        high[40] = f64::NAN;
        low[40] = f64::NAN;
        let expected = direct_definition_acosc(&high, &low);
        let input = AcoscInput::from_slices(&high, &low, AcoscParams::default());

        let actual = acosc_with_kernel(&input, Kernel::Scalar).unwrap();

        assert_same_validity_and_close("gap osc", &actual.osc, &expected.0);
        assert_same_validity_and_close("gap change", &actual.change, &expected.1);
        assert!(actual.osc[40..78].iter().all(|value| value.is_nan()));
        assert!(actual.osc[78].is_finite());
        assert!(actual.change[78].is_nan());
        assert!(actual.change[79].is_finite());
    }

    #[test]
    fn stream_matches_batch_across_a_gap() {
        let (mut high, mut low) = nonlinear_fixture(96);
        high[43] = f64::INFINITY;
        low[43] = f64::NEG_INFINITY;
        let input = AcoscInput::from_slices(&high, &low, AcoscParams::default());
        let batch = acosc_with_kernel(&input, Kernel::Scalar).unwrap();
        let mut stream = AcoscStream::try_new(AcoscParams::default()).unwrap();

        let streamed: Vec<(f64, f64)> = high
            .iter()
            .zip(&low)
            .map(|(&h, &l)| stream.update(h, l).unwrap_or((f64::NAN, f64::NAN)))
            .collect();
        let streamed_osc: Vec<f64> = streamed.iter().map(|value| value.0).collect();
        let streamed_change: Vec<f64> = streamed.iter().map(|value| value.1).collect();

        assert_same_validity_and_close("stream osc", &streamed_osc, &batch.osc);
        assert_same_validity_and_close("stream change", &streamed_change, &batch.change);
    }

    #[test]
    fn exactly_38_contiguous_bars_are_sufficient_for_first_ac_value() {
        let (high38, low38) = nonlinear_fixture(38);
        let input38 = AcoscInput::from_slices(&high38, &low38, AcoscParams::default());
        let output = acosc_with_kernel(&input38, Kernel::Scalar).unwrap();
        assert!(output.osc[37].is_finite());
        assert!(output.change[37].is_nan());

        let input37 = AcoscInput::from_slices(&high38[..37], &low38[..37], AcoscParams::default());
        assert!(matches!(
            acosc_with_kernel(&input37, Kernel::Scalar),
            Err(AcoscError::NotEnoughValidData {
                needed: 38,
                valid: 37
            })
        ));
    }

    fn check_acosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let default_params = AcoscParams::default();
        let input = AcoscInput::from_candles(&candles, default_params);
        let output = acosc_with_kernel(&input, kernel)?;
        assert_eq!(output.osc.len(), candles.close.len());
        assert_eq!(output.change.len(), candles.close.len());
        Ok(())
    }

    fn check_acosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let input = AcoscInput::with_default_candles(&candles);
        let result = acosc_with_kernel(&input, kernel)?;
        assert_eq!(result.osc.len(), candles.close.len());
        assert_eq!(result.change.len(), candles.close.len());
        let expected = direct_definition_acosc(&candles.high, &candles.low);
        assert_same_validity_and_close(&format!("{test_name} osc"), &result.osc, &expected.0);
        assert_same_validity_and_close(&format!("{test_name} change"), &result.change, &expected.1);
        Ok(())
    }

    fn check_acosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let input = AcoscInput::with_default_candles(&candles);
        match input.data {
            AcoscData::Candles { .. } => {}
            _ => panic!("Expected AcoscData::Candles variant"),
        }
        let output = acosc_with_kernel(&input, kernel)?;
        assert_eq!(output.osc.len(), candles.close.len());
        Ok(())
    }

    fn check_acosc_too_short(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 101.0];
        let low = [99.0, 98.0];
        let params = AcoscParams::default();
        let input = AcoscInput::from_slices(&high, &low, params);
        let result = acosc_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Should fail with not enough data",
            test_name
        );
        Ok(())
    }

    fn check_acosc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let input = AcoscInput::with_default_candles(&candles);
        let first_result = acosc_with_kernel(&input, kernel)?;
        assert_eq!(first_result.osc.len(), candles.close.len());
        assert_eq!(first_result.change.len(), candles.close.len());
        let input2 = AcoscInput::from_slices(&candles.high, &candles.low, AcoscParams::default());
        let second_result = acosc_with_kernel(&input2, kernel)?;
        assert_eq!(second_result.osc.len(), candles.close.len());
        for (a, b) in second_result.osc.iter().zip(first_result.osc.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-8,
                "Reinput values mismatch: {} vs {}",
                a,
                b
            );
        }
        Ok(())
    }

    fn check_acosc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let input = AcoscInput::with_default_candles(&candles);
        let result = acosc_with_kernel(&input, kernel)?;
        if result.osc.len() > 240 {
            for i in 240..result.osc.len() {
                assert!(!result.osc[i].is_nan(), "Found NaN in osc at {}", i);
                assert!(!result.change[i].is_nan(), "Found NaN in change at {}", i);
            }
        }
        Ok(())
    }

    fn check_acosc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let input = AcoscInput::with_default_candles(&candles);
        let batch = acosc_with_kernel(&input, kernel)?;
        let mut stream = AcoscStream::try_new(AcoscParams::default())?;
        let mut osc_stream = Vec::with_capacity(candles.close.len());
        let mut change_stream = Vec::with_capacity(candles.close.len());
        for (&h, &l) in candles.high.iter().zip(candles.low.iter()) {
            match stream.update(h, l) {
                Some((o, c)) => {
                    osc_stream.push(o);
                    change_stream.push(c);
                }
                None => {
                    osc_stream.push(f64::NAN);
                    change_stream.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch.osc.len(), osc_stream.len());
        assert_eq!(batch.change.len(), change_stream.len());
        for (i, (&a, &b)) in batch.osc.iter().zip(osc_stream.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-9,
                "Streaming osc mismatch at idx {}: {} vs {}",
                i,
                a,
                b
            );
        }
        for (i, (&a, &b)) in batch.change.iter().zip(change_stream.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-9,
                "Streaming change mismatch at idx {}: {} vs {}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    macro_rules! generate_all_acosc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test]
                  fn [<$test_fn _scalar_f64>]() {
                      $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar).unwrap();
                  })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test]
                  fn [<$test_fn _avx2_f64>]() {
                      $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2).unwrap();
                  }
                  #[test]
                  fn [<$test_fn _avx512_f64>]() {
                      $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512).unwrap();
                  })*
            }
        }
    }
    generate_all_acosc_tests!(
        check_acosc_partial_params,
        check_acosc_accuracy,
        check_acosc_default_candles,
        check_acosc_too_short,
        check_acosc_reinput,
        check_acosc_nan_handling,
        check_acosc_streaming,
        check_acosc_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_acosc_tests!(check_acosc_property);

    #[cfg(feature = "proptest")]
    fn check_acosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (40usize..=400).prop_flat_map(|len| {
            prop::collection::vec(
                (1.0f64..10000.0f64)
                    .prop_flat_map(|base_price| {
                        (0.0f64..0.1f64).prop_map(move |spread_pct| {
                            let half_spread = base_price * spread_pct * 0.5;
                            let high = base_price + half_spread;
                            let low = base_price - half_spread;
                            (high, low)
                        })
                    })
                    .prop_filter("prices must be finite", |(h, l)| {
                        h.is_finite() && l.is_finite()
                    }),
                len,
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |price_pairs| {
            let (high_vec, low_vec): (Vec<f64>, Vec<f64>) = price_pairs.into_iter().unzip();
            let params = AcoscParams::default();
            let input = AcoscInput::from_slices(&high_vec, &low_vec, params);

            let result = acosc_with_kernel(&input, kernel).unwrap();
            let scalar_result = acosc_with_kernel(&input, Kernel::Scalar).unwrap();

            for i in 0..result.osc.len() {
                let y = result.osc[i];
                let r = scalar_result.osc[i];

                if !y.is_finite() || !r.is_finite() {
                    prop_assert_eq!(
                        y.to_bits(),
                        r.to_bits(),
                        "NaN/finite mismatch in osc at idx {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                    continue;
                }

                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                    "Kernel mismatch in osc at idx {}: {} vs {} (ULP={})",
                    i,
                    y,
                    r,
                    ulp_diff
                );
            }

            for i in 0..result.change.len() {
                let y = result.change[i];
                let r = scalar_result.change[i];

                if !y.is_finite() || !r.is_finite() {
                    prop_assert_eq!(
                        y.to_bits(),
                        r.to_bits(),
                        "NaN/finite mismatch in change at idx {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                    continue;
                }

                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                    "Kernel mismatch in change at idx {}: {} vs {} (ULP={})",
                    i,
                    y,
                    r,
                    ulp_diff
                );
            }

            for i in 0..37.min(result.osc.len()) {
                prop_assert!(
                    result.osc[i].is_nan(),
                    "Expected NaN in osc warmup at idx {}, got {}",
                    i,
                    result.osc[i]
                );
                prop_assert!(
                    result.change[i].is_nan(),
                    "Expected NaN in change warmup at idx {}, got {}",
                    i,
                    result.change[i]
                );
            }

            if result.osc.len() > 37 {
                prop_assert!(
                    result.osc[37].is_finite(),
                    "Expected finite value at idx 37 in osc, got {}",
                    result.osc[37]
                );
                prop_assert!(
                    result.change[37].is_nan(),
                    "Expected undefined first change at idx 37, got {}",
                    result.change[37]
                );
            }

            if result.change.len() > 38 {
                prop_assert!(
                    result.change[38].is_finite(),
                    "Expected first finite change at idx 38, got {}",
                    result.change[38]
                );
            }

            for i in 38..result.osc.len() {
                if result.osc[i].is_finite() && result.osc[i - 1].is_finite() {
                    let expected_change = result.osc[i] - result.osc[i - 1];
                    let actual_change = result.change[i];

                    prop_assert!(
                        (expected_change - actual_change).abs() <= 1e-9,
                        "Change formula mismatch at idx {}: expected {} ({}−{}), got {}",
                        i,
                        expected_change,
                        result.osc[i],
                        result.osc[i - 1],
                        actual_change
                    );
                }
            }

            if high_vec.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                && low_vec.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
            {
                for i in 37..result.osc.len() {
                    prop_assert!(
                        result.osc[i].abs() <= 1e-6,
                        "Expected near-zero osc with constant prices at idx {}, got {}",
                        i,
                        result.osc[i]
                    );
                }
            }

            Ok(())
        })?;

        let (high_data, low_data) = nonlinear_fixture(200);
        let test_len = high_data.len();

        {
            let params = AcoscParams::default();
            let input = AcoscInput::from_slices(&high_data, &low_data, params.clone());
            let batch_result = acosc_with_kernel(&input, kernel)?;

            let mut stream = AcoscStream::try_new(params)?;
            let mut stream_osc = Vec::with_capacity(test_len);
            let mut stream_change = Vec::with_capacity(test_len);

            for i in 0..test_len {
                match stream.update(high_data[i], low_data[i]) {
                    Some((osc, change)) => {
                        stream_osc.push(osc);
                        stream_change.push(change);
                    }
                    None => {
                        stream_osc.push(f64::NAN);
                        stream_change.push(f64::NAN);
                    }
                }
            }

            for i in 0..test_len {
                let batch_o = batch_result.osc[i];
                let stream_o = stream_osc[i];

                if batch_o.is_nan() && stream_o.is_nan() {
                    continue;
                }

                assert!(
                    (batch_o - stream_o).abs() <= 1e-9,
                    "[{}] Streaming vs batch mismatch in osc at idx {}: {} vs {}",
                    test_name,
                    i,
                    batch_o,
                    stream_o
                );

                let batch_c = batch_result.change[i];
                let stream_c = stream_change[i];

                if batch_c.is_nan() && stream_c.is_nan() {
                    continue;
                }

                assert!(
                    (batch_c - stream_c).abs() <= 1e-9,
                    "[{}] Streaming vs batch mismatch in change at idx {}: {} vs {}",
                    test_name,
                    i,
                    batch_c,
                    stream_c
                );
            }
        }

        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let c = fixture_candles(512);
        let output = AcoscBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        assert_eq!(output.osc.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_acosc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let candles = fixture_candles(512);
        let input = AcoscInput::with_default_candles(&candles);
        let output = acosc_with_kernel(&input, kernel)?;

        for (i, &val) in output.osc.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in osc",
                    test_name, val, bits, i
                );
            }
        }

        for (i, &val) in output.change.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in change",
                    test_name, val, bits, i
                );
            }
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let c = fixture_candles(512);
        let output = AcoscBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        for (idx, &val) in output.osc.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in osc",
                    test, val, bits, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in osc",
                    test, val, bits, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in osc",
                    test, val, bits, idx
                );
            }
        }

        for (idx, &val) in output.change.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in change",
                    test, val, bits, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in change",
                    test, val, bits, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in change",
                    test, val, bits, idx
                );
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_acosc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch).unwrap();
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch).unwrap();
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch).unwrap();
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto).unwrap();
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_batch_kernel_error() {
        let high = vec![100.0; 50];
        let low = vec![99.0; 50];

        let result = acosc_batch_with_kernel(&high, &low, Kernel::Scalar);
        assert!(result.is_err());

        match result.unwrap_err() {
            AcoscError::InvalidKernelForBatch(kernel) => {
                assert_eq!(kernel, Kernel::Scalar);
            }
            _ => panic!("Expected InvalidKernelForBatch error"),
        }

        let result = acosc_batch_with_kernel(&high, &low, Kernel::Avx2);
        assert!(matches!(
            result,
            Err(AcoscError::InvalidKernelForBatch(Kernel::Avx2))
        ));
    }

    #[test]
    fn test_acosc_into_matches_api() -> Result<(), Box<dyn Error>> {
        let candles = fixture_candles(512);
        let n = candles.high.len().min(512).max(64);
        let high = &candles.high[..n];
        let low = &candles.low[..n];

        let params = AcoscParams::default();
        let input = AcoscInput::from_slices(high, low, params);

        let base = acosc(&input)?;

        let mut out_osc = vec![0.0; n];
        let mut out_change = vec![0.0; n];

        acosc_into(&input, &mut out_osc, &mut out_change)?;

        assert_eq!(base.osc.len(), out_osc.len());
        assert_eq!(base.change.len(), out_change.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(base.osc[i], out_osc[i]),
                "osc mismatch at {}: base={} out={}",
                i,
                base.osc[i],
                out_osc[i]
            );
            assert!(
                eq_or_both_nan(base.change[i], out_change[i]),
                "change mismatch at {}: base={} out={}",
                i,
                base.change[i],
                out_change[i]
            );
        }

        Ok(())
    }
}
