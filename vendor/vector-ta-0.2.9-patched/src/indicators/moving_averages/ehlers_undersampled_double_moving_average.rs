use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::sync::OnceLock;
use thiserror::Error;

const DEFAULT_FAST_LENGTH: usize = 6;
const DEFAULT_SLOW_LENGTH: usize = 12;
const DEFAULT_SAMPLE_LENGTH: usize = 5;
const MAX_LENGTH: usize = 4096;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
static EUDMA_AUTO_KERNEL: OnceLock<Kernel> = OnceLock::new();

impl<'a> AsRef<[f64]> for EhlersUndersampledDoubleMovingAverageInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersUndersampledDoubleMovingAverageData::Slice(slice) => slice,
            EhlersUndersampledDoubleMovingAverageData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersUndersampledDoubleMovingAverageData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersUndersampledDoubleMovingAverageOutput {
    pub fast: Vec<f64>,
    pub slow: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EhlersUndersampledDoubleMovingAverageParams {
    pub fast_length: Option<usize>,
    pub slow_length: Option<usize>,
    pub sample_length: Option<usize>,
}

impl Default for EhlersUndersampledDoubleMovingAverageParams {
    fn default() -> Self {
        Self {
            fast_length: Some(DEFAULT_FAST_LENGTH),
            slow_length: Some(DEFAULT_SLOW_LENGTH),
            sample_length: Some(DEFAULT_SAMPLE_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersUndersampledDoubleMovingAverageInput<'a> {
    pub data: EhlersUndersampledDoubleMovingAverageData<'a>,
    pub params: EhlersUndersampledDoubleMovingAverageParams,
}

impl<'a> EhlersUndersampledDoubleMovingAverageInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersUndersampledDoubleMovingAverageParams,
    ) -> Self {
        Self {
            data: EhlersUndersampledDoubleMovingAverageData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        slice: &'a [f64],
        params: EhlersUndersampledDoubleMovingAverageParams,
    ) -> Self {
        Self {
            data: EhlersUndersampledDoubleMovingAverageData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "hlcc4",
            EhlersUndersampledDoubleMovingAverageParams::default(),
        )
    }

    #[inline]
    pub fn get_fast_length(&self) -> usize {
        self.params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH)
    }

    #[inline]
    pub fn get_slow_length(&self) -> usize {
        self.params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH)
    }

    #[inline]
    pub fn get_sample_length(&self) -> usize {
        self.params.sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersUndersampledDoubleMovingAverageBuilder {
    fast_length: Option<usize>,
    slow_length: Option<usize>,
    sample_length: Option<usize>,
    kernel: Kernel,
}

impl Default for EhlersUndersampledDoubleMovingAverageBuilder {
    fn default() -> Self {
        Self {
            fast_length: None,
            slow_length: None,
            sample_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersUndersampledDoubleMovingAverageBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_length(mut self, fast_length: usize) -> Self {
        self.fast_length = Some(fast_length);
        self
    }

    #[inline(always)]
    pub fn slow_length(mut self, slow_length: usize) -> Self {
        self.slow_length = Some(slow_length);
        self
    }

    #[inline(always)]
    pub fn sample_length(mut self, sample_length: usize) -> Self {
        self.sample_length = Some(sample_length);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        EhlersUndersampledDoubleMovingAverageOutput,
        EhlersUndersampledDoubleMovingAverageError,
    > {
        let input = EhlersUndersampledDoubleMovingAverageInput::from_candles(
            candles,
            "hlcc4",
            EhlersUndersampledDoubleMovingAverageParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
                sample_length: self.sample_length,
            },
        );
        ehlers_undersampled_double_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<
        EhlersUndersampledDoubleMovingAverageOutput,
        EhlersUndersampledDoubleMovingAverageError,
    > {
        let input = EhlersUndersampledDoubleMovingAverageInput::from_slice(
            data,
            EhlersUndersampledDoubleMovingAverageParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
                sample_length: self.sample_length,
            },
        );
        ehlers_undersampled_double_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<
        EhlersUndersampledDoubleMovingAverageStream,
        EhlersUndersampledDoubleMovingAverageError,
    > {
        EhlersUndersampledDoubleMovingAverageStream::try_new(
            EhlersUndersampledDoubleMovingAverageParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
                sample_length: self.sample_length,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum EhlersUndersampledDoubleMovingAverageError {
    #[error("ehlers_undersampled_double_moving_average: input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_undersampled_double_moving_average: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "ehlers_undersampled_double_moving_average: invalid fast_length: {fast_length}. Expected 1..={MAX_LENGTH}."
    )]
    InvalidFastLength { fast_length: usize },
    #[error(
        "ehlers_undersampled_double_moving_average: invalid slow_length: {slow_length}. Expected 1..={MAX_LENGTH}."
    )]
    InvalidSlowLength { slow_length: usize },
    #[error(
        "ehlers_undersampled_double_moving_average: invalid sample_length: {sample_length}. Expected 1..={MAX_LENGTH}."
    )]
    InvalidSampleLength { sample_length: usize },
    #[error(
        "ehlers_undersampled_double_moving_average: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "ehlers_undersampled_double_moving_average: invalid fast_length range: start={start}, end={end}, step={step}"
    )]
    InvalidFastLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error(
        "ehlers_undersampled_double_moving_average: invalid slow_length range: start={start}, end={end}, step={step}"
    )]
    InvalidSlowLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error(
        "ehlers_undersampled_double_moving_average: invalid sample_length range: start={start}, end={end}, step={step}"
    )]
    InvalidSampleLengthRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("ehlers_undersampled_double_moving_average: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug)]
struct PreparedEudma<'a> {
    data: &'a [f64],
    first_valid: usize,
    fast_length: usize,
    slow_length: usize,
    sample_length: usize,
}

#[derive(Clone, Debug)]
struct HannFilterState {
    weights: Vec<f64>,
    norm: f64,
    ring: Vec<f64>,
    head: usize,
    count: usize,
}

impl HannFilterState {
    #[inline]
    fn new(length: usize) -> Self {
        let mut weights = Vec::with_capacity(length);
        let mut norm = 0.0;
        for i in 1..=length {
            let weight = 1.0 - (2.0 * core::f64::consts::PI * i as f64 / (length + 1) as f64).cos();
            weights.push(weight);
            norm += weight;
        }
        Self {
            weights,
            norm,
            ring: vec![0.0; length],
            head: 0,
            count: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.ring.fill(0.0);
        self.head = 0;
        self.count = 0;
    }

    #[inline(always)]
    fn update(&mut self, sample: f64) -> f64 {
        let len = self.weights.len();
        let full = self.count == len;
        self.ring[self.head] = sample;
        self.head += 1;
        if self.head == len {
            self.head = 0;
        }
        if !full {
            self.count += 1;
        }

        let mut acc = 0.0;
        let mut idx = if self.head == 0 {
            len - 1
        } else {
            self.head - 1
        };
        if full {
            for offset in 0..len {
                let current = self.ring[idx];
                let value = if current.is_finite() { current } else { 0.0 };
                acc += self.weights[offset] * value;
                idx = if idx == 0 { len - 1 } else { idx - 1 };
            }
        } else {
            for offset in 0..len {
                let value = if offset < self.count {
                    let current = self.ring[idx];
                    if current.is_finite() { current } else { 0.0 }
                } else {
                    0.0
                };
                acc += self.weights[offset] * value;
                idx = if idx == 0 { len - 1 } else { idx - 1 };
            }
        }

        if self.norm == 0.0 {
            0.0
        } else {
            acc / self.norm
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EhlersUndersampledDoubleMovingAverageExactHannPayload {
    pub fast_weights: Vec<f64>,
    pub slow_weights: Vec<f64>,
    pub fast_norm: f64,
    pub slow_norm: f64,
}

/// Build the immutable Hann coefficients from the same scalar CPU constructor
/// used by [`EudmaCore`]. The production CUDA route uploads only this bounded
/// parameter payload; price inputs and result matrices remain device-resident.
pub(crate) fn ehlers_undersampled_double_moving_average_exact_hann_payload(
    fast_length: usize,
    slow_length: usize,
) -> Result<
    EhlersUndersampledDoubleMovingAverageExactHannPayload,
    EhlersUndersampledDoubleMovingAverageError,
> {
    validate_params(fast_length, slow_length, 1)?;
    let fast = HannFilterState::new(fast_length);
    let slow = HannFilterState::new(slow_length);
    Ok(EhlersUndersampledDoubleMovingAverageExactHannPayload {
        fast_weights: fast.weights,
        slow_weights: slow.weights,
        fast_norm: fast.norm,
        slow_norm: slow.norm,
    })
}

#[derive(Clone, Debug)]
struct EudmaCore {
    fast_filter: HannFilterState,
    slow_filter: HannFilterState,
    sample_length: usize,
    sample_countdown: usize,
    last_sample: f64,
}

impl EudmaCore {
    #[inline]
    fn new(fast_length: usize, slow_length: usize, sample_length: usize) -> Self {
        Self {
            fast_filter: HannFilterState::new(fast_length),
            slow_filter: HannFilterState::new(slow_length),
            sample_length,
            sample_countdown: 0,
            last_sample: f64::NAN,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.fast_filter.reset();
        self.slow_filter.reset();
        self.sample_countdown = 0;
        self.last_sample = f64::NAN;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> (f64, f64) {
        let sampled = if self.sample_countdown == 0 {
            self.sample_countdown = self.sample_length - 1;
            value
        } else if self.last_sample.is_finite() {
            self.sample_countdown -= 1;
            self.last_sample
        } else {
            self.sample_countdown -= 1;
            0.0
        };
        self.last_sample = sampled;
        (
            self.fast_filter.update(sampled),
            self.slow_filter.update(sampled),
        )
    }
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_eudma_kernel(),
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    }
}

#[inline(always)]
fn detect_eudma_kernel() -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        return *EUDMA_AUTO_KERNEL.get_or_init(|| {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                Kernel::Avx2
            } else {
                Kernel::Scalar
            }
        });
    }

    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        Kernel::Scalar
    }
}

#[inline(always)]
fn validate_params(
    fast_length: usize,
    slow_length: usize,
    sample_length: usize,
) -> Result<(), EhlersUndersampledDoubleMovingAverageError> {
    if fast_length == 0 || fast_length > MAX_LENGTH {
        return Err(EhlersUndersampledDoubleMovingAverageError::InvalidFastLength { fast_length });
    }
    if slow_length == 0 || slow_length > MAX_LENGTH {
        return Err(EhlersUndersampledDoubleMovingAverageError::InvalidSlowLength { slow_length });
    }
    if sample_length == 0 || sample_length > MAX_LENGTH {
        return Err(
            EhlersUndersampledDoubleMovingAverageError::InvalidSampleLength { sample_length },
        );
    }
    Ok(())
}

#[inline(always)]
fn eudma_prepare<'a>(
    input: &'a EhlersUndersampledDoubleMovingAverageInput<'a>,
    kernel: Kernel,
) -> Result<(PreparedEudma<'a>, Kernel), EhlersUndersampledDoubleMovingAverageError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(EhlersUndersampledDoubleMovingAverageError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(EhlersUndersampledDoubleMovingAverageError::AllValuesNaN)?;
    let fast_length = input.get_fast_length();
    let slow_length = input.get_slow_length();
    let sample_length = input.get_sample_length();
    validate_params(fast_length, slow_length, sample_length)?;

    Ok((
        PreparedEudma {
            data,
            first_valid,
            fast_length,
            slow_length,
            sample_length,
        },
        normalize_single_kernel(kernel),
    ))
}

#[inline(always)]
fn compute_eudma_into(prepared: PreparedEudma<'_>, fast_out: &mut [f64], slow_out: &mut [f64]) {
    let mut core = EudmaCore::new(
        prepared.fast_length,
        prepared.slow_length,
        prepared.sample_length,
    );
    let first = prepared.first_valid.min(prepared.data.len());
    for &value in &prepared.data[..first] {
        let _ = core.update(value);
    }
    for idx in first..prepared.data.len() {
        let (fast, slow) = core.update(prepared.data[idx]);
        fast_out[idx] = fast;
        slow_out[idx] = slow;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn compute_eudma_with_kernel(
    prepared: PreparedEudma<'_>,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    kernel: Kernel,
) {
    unsafe {
        match kernel {
            Kernel::Avx2 => compute_eudma_avx2(prepared, fast_out, slow_out),
            Kernel::Avx512 => compute_eudma_avx512(prepared, fast_out, slow_out),
            _ => compute_eudma_into(prepared, fast_out, slow_out),
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_eudma_avx2(
    prepared: PreparedEudma<'_>,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    compute_eudma_into(prepared, fast_out, slow_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn compute_eudma_avx512(
    prepared: PreparedEudma<'_>,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) {
    compute_eudma_into(prepared, fast_out, slow_out)
}

#[inline]
pub fn ehlers_undersampled_double_moving_average(
    input: &EhlersUndersampledDoubleMovingAverageInput,
) -> Result<EhlersUndersampledDoubleMovingAverageOutput, EhlersUndersampledDoubleMovingAverageError>
{
    ehlers_undersampled_double_moving_average_with_kernel(input, Kernel::Auto)
}

pub fn ehlers_undersampled_double_moving_average_with_kernel(
    input: &EhlersUndersampledDoubleMovingAverageInput,
    kernel: Kernel,
) -> Result<EhlersUndersampledDoubleMovingAverageOutput, EhlersUndersampledDoubleMovingAverageError>
{
    let (prepared, kernel) = eudma_prepare(input, kernel)?;
    let mut fast = alloc_with_nan_prefix(prepared.data.len(), prepared.first_valid);
    let mut slow = alloc_with_nan_prefix(prepared.data.len(), prepared.first_valid);
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    compute_eudma_with_kernel(prepared, &mut fast, &mut slow, kernel);
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        let _ = kernel;
        compute_eudma_into(prepared, &mut fast, &mut slow);
    }
    Ok(EhlersUndersampledDoubleMovingAverageOutput { fast, slow })
}

#[inline]
pub fn ehlers_undersampled_double_moving_average_into_slices(
    fast_out: &mut [f64],
    slow_out: &mut [f64],
    input: &EhlersUndersampledDoubleMovingAverageInput,
    kernel: Kernel,
) -> Result<(), EhlersUndersampledDoubleMovingAverageError> {
    let (prepared, kernel) = eudma_prepare(input, kernel)?;
    if fast_out.len() != prepared.data.len() {
        return Err(
            EhlersUndersampledDoubleMovingAverageError::OutputLengthMismatch {
                expected: prepared.data.len(),
                got: fast_out.len(),
            },
        );
    }
    if slow_out.len() != prepared.data.len() {
        return Err(
            EhlersUndersampledDoubleMovingAverageError::OutputLengthMismatch {
                expected: prepared.data.len(),
                got: slow_out.len(),
            },
        );
    }

    let warm = prepared.first_valid.min(fast_out.len());
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for value in &mut fast_out[..warm] {
        *value = qnan;
    }
    for value in &mut slow_out[..warm] {
        *value = qnan;
    }
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    compute_eudma_with_kernel(prepared, fast_out, slow_out, kernel);
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        let _ = kernel;
        compute_eudma_into(prepared, fast_out, slow_out);
    }
    Ok(())
}

#[inline]
pub fn ehlers_undersampled_double_moving_average_into(
    input: &EhlersUndersampledDoubleMovingAverageInput,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) -> Result<(), EhlersUndersampledDoubleMovingAverageError> {
    ehlers_undersampled_double_moving_average_into_slices(fast_out, slow_out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct EhlersUndersampledDoubleMovingAverageStream {
    core: EudmaCore,
    seen_finite: bool,
}

impl EhlersUndersampledDoubleMovingAverageStream {
    pub fn try_new(
        params: EhlersUndersampledDoubleMovingAverageParams,
    ) -> Result<Self, EhlersUndersampledDoubleMovingAverageError> {
        let fast_length = params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
        let slow_length = params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
        let sample_length = params.sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH);
        validate_params(fast_length, slow_length, sample_length)?;
        Ok(Self {
            core: EudmaCore::new(fast_length, slow_length, sample_length),
            seen_finite: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let outputs = self.core.update(value);
        if !self.seen_finite {
            if value.is_finite() {
                self.seen_finite = true;
                return Some(outputs);
            }
            return None;
        }
        Some(outputs)
    }

    #[inline]
    pub fn reset(&mut self) {
        self.core.reset();
        self.seen_finite = false;
    }
}

#[derive(Clone, Debug)]
pub struct EhlersUndersampledDoubleMovingAverageBatchRange {
    pub fast_length: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
    pub sample_length: (usize, usize, usize),
}

impl Default for EhlersUndersampledDoubleMovingAverageBatchRange {
    fn default() -> Self {
        Self {
            fast_length: (DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
            slow_length: (DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
            sample_length: (DEFAULT_SAMPLE_LENGTH, DEFAULT_SAMPLE_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersUndersampledDoubleMovingAverageBatchBuilder {
    range: EhlersUndersampledDoubleMovingAverageBatchRange,
    kernel: Kernel,
}

impl Default for EhlersUndersampledDoubleMovingAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersUndersampledDoubleMovingAverageBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersUndersampledDoubleMovingAverageBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn fast_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_length = (start, end, step);
        self
    }

    pub fn slow_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_length = (start, end, step);
        self
    }

    pub fn sample_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sample_length = (start, end, step);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<
        EhlersUndersampledDoubleMovingAverageBatchOutput,
        EhlersUndersampledDoubleMovingAverageError,
    > {
        ehlers_undersampled_double_moving_average_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<
        EhlersUndersampledDoubleMovingAverageBatchOutput,
        EhlersUndersampledDoubleMovingAverageError,
    > {
        self.apply_slice(source_type(candles, "hlcc4"))
    }

    pub fn apply_candles_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<
        EhlersUndersampledDoubleMovingAverageBatchOutput,
        EhlersUndersampledDoubleMovingAverageError,
    > {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct EhlersUndersampledDoubleMovingAverageBatchOutput {
    pub fast_values: Vec<f64>,
    pub slow_values: Vec<f64>,
    pub combos: Vec<EhlersUndersampledDoubleMovingAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersUndersampledDoubleMovingAverageBatchOutput {
    pub fn row_for_params(
        &self,
        params: &EhlersUndersampledDoubleMovingAverageParams,
    ) -> Option<usize> {
        let fast_length = params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
        let slow_length = params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
        let sample_length = params.sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH);
        self.combos.iter().position(|combo| {
            combo.fast_length.unwrap_or(DEFAULT_FAST_LENGTH) == fast_length
                && combo.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH) == slow_length
                && combo.sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH) == sample_length
        })
    }

    pub fn fast_values_for(
        &self,
        params: &EhlersUndersampledDoubleMovingAverageParams,
    ) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.fast_values[start..start + self.cols]
        })
    }

    pub fn slow_values_for(
        &self,
        params: &EhlersUndersampledDoubleMovingAverageParams,
    ) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.slow_values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_axis(
    axis: (usize, usize, usize),
    which: &'static str,
) -> Result<Vec<usize>, EhlersUndersampledDoubleMovingAverageError> {
    let (start, end, step) = axis;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut values = Vec::new();
    if start < end {
        let mut current = start;
        while current <= end {
            values.push(current);
            match current.checked_add(step) {
                Some(next) if next > current => current = next,
                _ => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            values.push(current);
            if current < end.saturating_add(step) {
                break;
            }
            current = current.saturating_sub(step);
        }
    }

    if values.is_empty() {
        return Err(match which {
            "fast_length" => EhlersUndersampledDoubleMovingAverageError::InvalidFastLengthRange {
                start,
                end,
                step,
            },
            "slow_length" => EhlersUndersampledDoubleMovingAverageError::InvalidSlowLengthRange {
                start,
                end,
                step,
            },
            _ => EhlersUndersampledDoubleMovingAverageError::InvalidSampleLengthRange {
                start,
                end,
                step,
            },
        });
    }

    Ok(values)
}

#[inline(always)]
pub fn expand_grid_ehlers_undersampled_double_moving_average(
    range: &EhlersUndersampledDoubleMovingAverageBatchRange,
) -> Result<
    Vec<EhlersUndersampledDoubleMovingAverageParams>,
    EhlersUndersampledDoubleMovingAverageError,
> {
    let fast_lengths = expand_axis(range.fast_length, "fast_length")?;
    let slow_lengths = expand_axis(range.slow_length, "slow_length")?;
    let sample_lengths = expand_axis(range.sample_length, "sample_length")?;

    let mut combos =
        Vec::with_capacity(fast_lengths.len() * slow_lengths.len() * sample_lengths.len());
    for &fast_length in &fast_lengths {
        for &slow_length in &slow_lengths {
            for &sample_length in &sample_lengths {
                validate_params(fast_length, slow_length, sample_length)?;
                combos.push(EhlersUndersampledDoubleMovingAverageParams {
                    fast_length: Some(fast_length),
                    slow_length: Some(slow_length),
                    sample_length: Some(sample_length),
                });
            }
        }
    }

    Ok(combos)
}

pub fn ehlers_undersampled_double_moving_average_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersUndersampledDoubleMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<
    EhlersUndersampledDoubleMovingAverageBatchOutput,
    EhlersUndersampledDoubleMovingAverageError,
> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(EhlersUndersampledDoubleMovingAverageError::InvalidKernelForBatch(other));
        }
    };
    ehlers_undersampled_double_moving_average_batch_inner(
        data,
        sweep,
        batch_kernel,
        !matches!(batch_kernel, Kernel::ScalarBatch),
    )
}

#[inline(always)]
pub fn ehlers_undersampled_double_moving_average_batch_slice(
    data: &[f64],
    sweep: &EhlersUndersampledDoubleMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<
    EhlersUndersampledDoubleMovingAverageBatchOutput,
    EhlersUndersampledDoubleMovingAverageError,
> {
    ehlers_undersampled_double_moving_average_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn ehlers_undersampled_double_moving_average_batch_par_slice(
    data: &[f64],
    sweep: &EhlersUndersampledDoubleMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<
    EhlersUndersampledDoubleMovingAverageBatchOutput,
    EhlersUndersampledDoubleMovingAverageError,
> {
    ehlers_undersampled_double_moving_average_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn ehlers_undersampled_double_moving_average_batch_inner(
    data: &[f64],
    sweep: &EhlersUndersampledDoubleMovingAverageBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<
    EhlersUndersampledDoubleMovingAverageBatchOutput,
    EhlersUndersampledDoubleMovingAverageError,
> {
    let combos = expand_grid_ehlers_undersampled_double_moving_average(sweep)?;
    if data.is_empty() {
        return Err(EhlersUndersampledDoubleMovingAverageError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(EhlersUndersampledDoubleMovingAverageError::AllValuesNaN)?;

    let rows = combos.len();
    let cols = data.len();
    let mut fast_buf = make_uninit_matrix(rows, cols);
    let mut slow_buf = make_uninit_matrix(rows, cols);
    let warm_prefixes = vec![first_valid; rows];
    init_matrix_prefixes(&mut fast_buf, cols, &warm_prefixes);
    init_matrix_prefixes(&mut slow_buf, cols, &warm_prefixes);

    let mut fast_guard = ManuallyDrop::new(fast_buf);
    let mut slow_guard = ManuallyDrop::new(slow_buf);
    let fast_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(fast_guard.as_mut_ptr(), fast_guard.len()) };
    let slow_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(slow_guard.as_mut_ptr(), slow_guard.len()) };

    let _ = normalize_single_kernel(kernel);

    let do_row = |row: usize,
                  fast_row_mu: &mut [MaybeUninit<f64>],
                  slow_row_mu: &mut [MaybeUninit<f64>]| {
        let fast_row = unsafe {
            std::slice::from_raw_parts_mut(fast_row_mu.as_mut_ptr() as *mut f64, fast_row_mu.len())
        };
        let slow_row = unsafe {
            std::slice::from_raw_parts_mut(slow_row_mu.as_mut_ptr() as *mut f64, slow_row_mu.len())
        };
        compute_eudma_into(
            PreparedEudma {
                data,
                first_valid,
                fast_length: combos[row].fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
                slow_length: combos[row].slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
                sample_length: combos[row].sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH),
            },
            fast_row,
            slow_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        fast_mu
            .par_chunks_mut(cols)
            .zip(slow_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (fast_row_mu, slow_row_mu))| do_row(row, fast_row_mu, slow_row_mu));
        #[cfg(target_arch = "wasm32")]
        for (row, (fast_row_mu, slow_row_mu)) in fast_mu
            .chunks_mut(cols)
            .zip(slow_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, fast_row_mu, slow_row_mu);
        }
    } else {
        for (row, (fast_row_mu, slow_row_mu)) in fast_mu
            .chunks_mut(cols)
            .zip(slow_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, fast_row_mu, slow_row_mu);
        }
    }

    let fast_values = unsafe {
        Vec::from_raw_parts(
            fast_guard.as_mut_ptr() as *mut f64,
            fast_guard.len(),
            fast_guard.capacity(),
        )
    };
    let slow_values = unsafe {
        Vec::from_raw_parts(
            slow_guard.as_mut_ptr() as *mut f64,
            slow_guard.len(),
            slow_guard.capacity(),
        )
    };

    Ok(EhlersUndersampledDoubleMovingAverageBatchOutput {
        fast_values,
        slow_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn ehlers_undersampled_double_moving_average_batch_inner_into(
    data: &[f64],
    sweep: &EhlersUndersampledDoubleMovingAverageBatchRange,
    kernel: Kernel,
    parallel: bool,
    fast_out: &mut [f64],
    slow_out: &mut [f64],
) -> Result<
    Vec<EhlersUndersampledDoubleMovingAverageParams>,
    EhlersUndersampledDoubleMovingAverageError,
> {
    let combos = expand_grid_ehlers_undersampled_double_moving_average(sweep)?;
    if data.is_empty() {
        return Err(EhlersUndersampledDoubleMovingAverageError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(EhlersUndersampledDoubleMovingAverageError::AllValuesNaN)?;

    let rows = combos.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or(
        EhlersUndersampledDoubleMovingAverageError::OutputLengthMismatch {
            expected: usize::MAX,
            got: fast_out.len(),
        },
    )?;
    if fast_out.len() != expected {
        return Err(
            EhlersUndersampledDoubleMovingAverageError::OutputLengthMismatch {
                expected,
                got: fast_out.len(),
            },
        );
    }
    if slow_out.len() != expected {
        return Err(
            EhlersUndersampledDoubleMovingAverageError::OutputLengthMismatch {
                expected,
                got: slow_out.len(),
            },
        );
    }

    let _ = normalize_single_kernel(kernel);
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        fast_out
            .par_chunks_mut(cols)
            .zip(slow_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (fast_row, slow_row))| {
                for value in &mut fast_row[..first_valid.min(cols)] {
                    *value = f64::NAN;
                }
                for value in &mut slow_row[..first_valid.min(cols)] {
                    *value = f64::NAN;
                }
                compute_eudma_into(
                    PreparedEudma {
                        data,
                        first_valid,
                        fast_length: combos[row].fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
                        slow_length: combos[row].slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
                        sample_length: combos[row].sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH),
                    },
                    fast_row,
                    slow_row,
                );
            });
        #[cfg(target_arch = "wasm32")]
        for (row, (fast_row, slow_row)) in fast_out
            .chunks_mut(cols)
            .zip(slow_out.chunks_mut(cols))
            .enumerate()
        {
            for value in &mut fast_row[..first_valid.min(cols)] {
                *value = f64::NAN;
            }
            for value in &mut slow_row[..first_valid.min(cols)] {
                *value = f64::NAN;
            }
            compute_eudma_into(
                PreparedEudma {
                    data,
                    first_valid,
                    fast_length: combos[row].fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
                    slow_length: combos[row].slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
                    sample_length: combos[row].sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH),
                },
                fast_row,
                slow_row,
            );
        }
    } else {
        for (row, (fast_row, slow_row)) in fast_out
            .chunks_mut(cols)
            .zip(slow_out.chunks_mut(cols))
            .enumerate()
        {
            for value in &mut fast_row[..first_valid.min(cols)] {
                *value = f64::NAN;
            }
            for value in &mut slow_row[..first_valid.min(cols)] {
                *value = f64::NAN;
            }
            compute_eudma_into(
                PreparedEudma {
                    data,
                    first_valid,
                    fast_length: combos[row].fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
                    slow_length: combos[row].slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
                    sample_length: combos[row].sample_length.unwrap_or(DEFAULT_SAMPLE_LENGTH),
                },
                fast_row,
                slow_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::moving_averages::ma::MaData;
    use crate::indicators::moving_averages::ma_batch::{
        MaBatchParamKV, MaBatchParamValue, ma_batch_with_kernel_and_typed_params,
    };
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn naive_series(
        data: &[f64],
        fast_length: usize,
        slow_length: usize,
        sample_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut fast = vec![f64::NAN; data.len()];
        let mut slow = vec![f64::NAN; data.len()];
        let Some(first_valid) = data.iter().position(|value| !value.is_nan()) else {
            return (fast, slow);
        };

        let fast_weights: Vec<f64> = (1..=fast_length)
            .map(|i| {
                1.0 - (2.0 * core::f64::consts::PI * i as f64 / (fast_length + 1) as f64).cos()
            })
            .collect();
        let slow_weights: Vec<f64> = (1..=slow_length)
            .map(|i| {
                1.0 - (2.0 * core::f64::consts::PI * i as f64 / (slow_length + 1) as f64).cos()
            })
            .collect();
        let fast_norm: f64 = fast_weights.iter().sum();
        let slow_norm: f64 = slow_weights.iter().sum();

        let mut sampled = vec![f64::NAN; data.len()];
        for idx in 0..data.len() {
            sampled[idx] = if idx % sample_length == 0 {
                data[idx]
            } else if idx > 0 && sampled[idx - 1].is_finite() {
                sampled[idx - 1]
            } else {
                0.0
            };
        }

        for idx in first_valid..data.len() {
            let mut fast_acc = 0.0;
            for offset in 0..fast_length {
                if idx >= offset {
                    let value = sampled[idx - offset];
                    fast_acc += fast_weights[offset] * if value.is_finite() { value } else { 0.0 };
                }
            }
            fast[idx] = fast_acc / fast_norm;

            let mut slow_acc = 0.0;
            for offset in 0..slow_length {
                if idx >= offset {
                    let value = sampled[idx - offset];
                    slow_acc += slow_weights[offset] * if value.is_finite() { value } else { 0.0 };
                }
            }
            slow[idx] = slow_acc / slow_norm;
        }

        (fast, slow)
    }

    fn sample_data() -> Vec<f64> {
        (0..256)
            .map(|idx| {
                let x = idx as f64;
                100.0 + (x * 0.07).sin() * 4.0 + (x * 0.033).cos() * 2.25 + x * 0.03
            })
            .collect()
    }

    #[test]
    fn eudma_matches_naive_reference() -> Result<(), Box<dyn std::error::Error>> {
        let data = vec![f64::NAN, 10.0, 11.0, 12.0, 13.0, 12.5, 14.0, 15.0];
        let input = EhlersUndersampledDoubleMovingAverageInput::from_slice(
            &data,
            EhlersUndersampledDoubleMovingAverageParams {
                fast_length: Some(3),
                slow_length: Some(4),
                sample_length: Some(2),
            },
        );
        let out = ehlers_undersampled_double_moving_average(&input)?;
        let (expected_fast, expected_slow) = naive_series(&data, 3, 4, 2);
        for ((actual_fast, expected_fast), (actual_slow, expected_slow)) in out
            .fast
            .iter()
            .zip(expected_fast.iter())
            .zip(out.slow.iter().zip(expected_slow.iter()))
        {
            assert!(
                (actual_fast.is_nan() && expected_fast.is_nan())
                    || (actual_fast - expected_fast).abs() <= 1e-12
            );
            assert!(
                (actual_slow.is_nan() && expected_slow.is_nan())
                    || (actual_slow - expected_slow).abs() <= 1e-12
            );
        }
        Ok(())
    }

    #[test]
    fn eudma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let data = sample_data();
        let input = EhlersUndersampledDoubleMovingAverageInput::from_slice(
            &data,
            EhlersUndersampledDoubleMovingAverageParams::default(),
        );
        let baseline = ehlers_undersampled_double_moving_average(&input)?;
        let mut fast = vec![0.0; data.len()];
        let mut slow = vec![0.0; data.len()];
        ehlers_undersampled_double_moving_average_into(&input, &mut fast, &mut slow)?;
        for ((actual_fast, expected_fast), (actual_slow, expected_slow)) in fast
            .iter()
            .zip(baseline.fast.iter())
            .zip(slow.iter().zip(baseline.slow.iter()))
        {
            assert!(
                (actual_fast.is_nan() && expected_fast.is_nan())
                    || (actual_fast - expected_fast).abs() <= 1e-12
            );
            assert!(
                (actual_slow.is_nan() && expected_slow.is_nan())
                    || (actual_slow - expected_slow).abs() <= 1e-12
            );
        }
        Ok(())
    }

    #[test]
    fn eudma_stream_matches_batch() -> Result<(), Box<dyn std::error::Error>> {
        let data = sample_data();
        let batch = ehlers_undersampled_double_moving_average(
            &EhlersUndersampledDoubleMovingAverageInput::from_slice(
                &data,
                EhlersUndersampledDoubleMovingAverageParams::default(),
            ),
        )?;
        let mut stream = EhlersUndersampledDoubleMovingAverageStream::try_new(
            EhlersUndersampledDoubleMovingAverageParams::default(),
        )?;
        let mut fast = vec![f64::NAN; data.len()];
        let mut slow = vec![f64::NAN; data.len()];
        for (idx, value) in data.iter().copied().enumerate() {
            if let Some((fast_value, slow_value)) = stream.update(value) {
                fast[idx] = fast_value;
                slow[idx] = slow_value;
            }
        }
        for ((actual_fast, expected_fast), (actual_slow, expected_slow)) in fast
            .iter()
            .zip(batch.fast.iter())
            .zip(slow.iter().zip(batch.slow.iter()))
        {
            assert!(
                (actual_fast.is_nan() && expected_fast.is_nan())
                    || (actual_fast - expected_fast).abs() <= 1e-12
            );
            assert!(
                (actual_slow.is_nan() && expected_slow.is_nan())
                    || (actual_slow - expected_slow).abs() <= 1e-12
            );
        }
        stream.reset();
        assert!(stream.update(f64::NAN).is_none());
        Ok(())
    }

    #[test]
    fn eudma_batch_matches_single() -> Result<(), Box<dyn std::error::Error>> {
        let data = sample_data();
        let batch = ehlers_undersampled_double_moving_average_batch_with_kernel(
            &data,
            &EhlersUndersampledDoubleMovingAverageBatchRange {
                fast_length: (4, 6, 2),
                slow_length: (8, 10, 2),
                sample_length: (3, 5, 2),
            },
            Kernel::ScalarBatch,
        )?;

        for (row, combo) in batch.combos.iter().enumerate() {
            let single = ehlers_undersampled_double_moving_average(
                &EhlersUndersampledDoubleMovingAverageInput::from_slice(&data, combo.clone()),
            )?;
            let start = row * batch.cols;
            let fast_row = &batch.fast_values[start..start + batch.cols];
            let slow_row = &batch.slow_values[start..start + batch.cols];
            for ((actual_fast, expected_fast), (actual_slow, expected_slow)) in fast_row
                .iter()
                .zip(single.fast.iter())
                .zip(slow_row.iter().zip(single.slow.iter()))
            {
                assert!(
                    (actual_fast.is_nan() && expected_fast.is_nan())
                        || (actual_fast - expected_fast).abs() <= 1e-12
                );
                assert!(
                    (actual_slow.is_nan() && expected_slow.is_nan())
                        || (actual_slow - expected_slow).abs() <= 1e-12
                );
            }
        }
        Ok(())
    }

    #[test]
    fn eudma_ma_batch_typed_output_selection_matches_direct()
    -> Result<(), Box<dyn std::error::Error>> {
        let data = sample_data();
        let params = [
            MaBatchParamKV {
                key: "fast_length",
                value: MaBatchParamValue::Int(6),
            },
            MaBatchParamKV {
                key: "slow_length",
                value: MaBatchParamValue::Int(12),
            },
            MaBatchParamKV {
                key: "sample_length",
                value: MaBatchParamValue::Int(5),
            },
            MaBatchParamKV {
                key: "output",
                value: MaBatchParamValue::EnumString("slow"),
            },
        ];
        let batch = ma_batch_with_kernel_and_typed_params(
            "ehlers_undersampled_double_moving_average",
            MaData::Slice(&data),
            (14, 14, 0),
            Kernel::Auto,
            &params,
        )?;
        let direct = ehlers_undersampled_double_moving_average(
            &EhlersUndersampledDoubleMovingAverageInput::from_slice(
                &data,
                EhlersUndersampledDoubleMovingAverageParams::default(),
            ),
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for (actual, expected) in batch.values.iter().zip(direct.slow.iter()) {
            assert!((actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn eudma_fixture_has_values() -> Result<(), Box<dyn std::error::Error>> {
        let candles = read_candles_from_vortex("src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex")?;
        let out = ehlers_undersampled_double_moving_average(
            &EhlersUndersampledDoubleMovingAverageInput::with_default_candles(&candles),
        )?;
        assert_eq!(out.fast.len(), candles.close.len());
        assert_eq!(out.slow.len(), candles.close.len());
        assert!(out.fast.iter().skip(16).any(|value| value.is_finite()));
        assert!(out.slow.iter().skip(16).any(|value| value.is_finite()));
        Ok(())
    }
}
