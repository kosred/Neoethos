use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::detect_best_batch_kernel;
use chrono::{DateTime, Timelike, Utc};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::VecDeque;
use thiserror::Error;

#[path = "half_causal_estimator_stable_math.rs"]
mod stable_math;
use stable_math::{NeumaierSum, StablePopulationMoments};

const DEFAULT_DATA_PERIOD: usize = 5;
const DEFAULT_FILTER_LENGTH: usize = 20;
const DEFAULT_KERNEL_WIDTH: f64 = 20.0;
const DEFAULT_MAXIMUM_CONFIDENCE_ADJUST: f64 = 100.0;
const DEFAULT_ENABLE_EXPECTED_VALUE: bool = false;
const DEFAULT_EXTRA_SMOOTHING: usize = 0;
const DEFAULT_SOURCE: &str = "volume";
const DAY_MS: i64 = 86_400_000;
pub const HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1: usize = 64 * 1024 * 1024;
pub const HALF_CAUSAL_ESTIMATOR_F64_SEMANTICS_V2: &str = "half-causal-estimator-f64-v2-neoethos-canonical-pine6-script24-utc-day-slot-session-proxy-cached-future-windows-stable-f64-registry-ratio-dl;public-retained-budget-64mib/v1";
pub const HALF_CAUSAL_ESTIMATOR_CREATOR_AUDIT_ORACLE_URL: &str = "https://pine-facade.tradingview.com/pine-facade/get/PUB%3B28b6b0520c9b45c597b96d7644327a89/last";
pub const HALF_CAUSAL_ESTIMATOR_CREATOR_SOURCE_SHA256: &str =
    "4B7FD8AEC6B333A4ECE967D7CFA6D957357CE436CB098E96EB1EB8A1480A8080";
pub const HALF_CAUSAL_ESTIMATOR_CREATOR_RECEIPT_PATH: &str =
    "audit_receipts/half_causal_estimator/script24_receipt.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfCausalEstimatorKernelType {
    Gaussian,
    Epanechnikov,
    Triangular,
    Sinc,
}

impl Default for HalfCausalEstimatorKernelType {
    fn default() -> Self {
        Self::Epanechnikov
    }
}

impl HalfCausalEstimatorKernelType {
    #[inline]
    fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "gaussian" => Some(Self::Gaussian),
            "epanechnikov" => Some(Self::Epanechnikov),
            "triangular" => Some(Self::Triangular),
            "sinc" | "blackman_windowed_sinc" | "blackman-windowed-sinc" => Some(Self::Sinc),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfCausalEstimatorConfidenceAdjust {
    Symmetric,
    Linear,
    None,
}

impl Default for HalfCausalEstimatorConfidenceAdjust {
    fn default() -> Self {
        Self::Symmetric
    }
}

impl HalfCausalEstimatorConfidenceAdjust {
    #[inline]
    fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "symmetric" => Some(Self::Symmetric),
            "linear" => Some(Self::Linear),
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HalfCausalEstimatorParams {
    pub slots_per_day: Option<usize>,
    pub data_period: Option<usize>,
    pub filter_length: Option<usize>,
    pub kernel_width: Option<f64>,
    pub kernel_type: Option<HalfCausalEstimatorKernelType>,
    pub confidence_adjust: Option<HalfCausalEstimatorConfidenceAdjust>,
    pub maximum_confidence_adjust: Option<f64>,
    pub enable_expected_value: Option<bool>,
    pub extra_smoothing: Option<usize>,
}

impl Default for HalfCausalEstimatorParams {
    fn default() -> Self {
        Self {
            slots_per_day: None,
            data_period: Some(DEFAULT_DATA_PERIOD),
            filter_length: Some(DEFAULT_FILTER_LENGTH),
            kernel_width: Some(DEFAULT_KERNEL_WIDTH),
            kernel_type: Some(HalfCausalEstimatorKernelType::Epanechnikov),
            confidence_adjust: Some(HalfCausalEstimatorConfidenceAdjust::Symmetric),
            maximum_confidence_adjust: Some(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST),
            enable_expected_value: Some(DEFAULT_ENABLE_EXPECTED_VALUE),
            extra_smoothing: Some(DEFAULT_EXTRA_SMOOTHING),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HalfCausalEstimatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorInput<'a> {
    pub data: HalfCausalEstimatorData<'a>,
    pub params: HalfCausalEstimatorParams,
}

impl<'a> HalfCausalEstimatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: HalfCausalEstimatorParams,
    ) -> Self {
        Self {
            data: HalfCausalEstimatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: HalfCausalEstimatorParams) -> Self {
        Self {
            data: HalfCausalEstimatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            HalfCausalEstimatorParams::default(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorOutput {
    pub estimate: Vec<f64>,
    pub expected_value: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBatchOutput {
    pub estimate_values: Vec<f64>,
    pub expected_value_values: Vec<f64>,
    pub combos: Vec<HalfCausalEstimatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HalfCausalEstimatorBatchOutput {
    #[inline]
    pub fn estimate_for(&self, row: usize) -> Option<&[f64]> {
        row.checked_mul(self.cols)
            .and_then(|start| self.estimate_values.get(start..start + self.cols))
    }

    #[inline]
    pub fn expected_value_for(&self, row: usize) -> Option<&[f64]> {
        row.checked_mul(self.cols)
            .and_then(|start| self.expected_value_values.get(start..start + self.cols))
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBatchRange {
    pub slots_per_day: Option<usize>,
    pub data_period: (usize, usize, usize),
    pub filter_length: (usize, usize, usize),
    pub kernel_width: (f64, f64, f64),
    pub maximum_confidence_adjust: (f64, f64, f64),
    pub extra_smoothing: (usize, usize, usize),
    pub kernel_type: HalfCausalEstimatorKernelType,
    pub confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    pub enable_expected_value: bool,
}

impl Default for HalfCausalEstimatorBatchRange {
    fn default() -> Self {
        Self {
            slots_per_day: None,
            data_period: (DEFAULT_DATA_PERIOD, DEFAULT_DATA_PERIOD, 0),
            filter_length: (DEFAULT_FILTER_LENGTH, DEFAULT_FILTER_LENGTH, 0),
            kernel_width: (DEFAULT_KERNEL_WIDTH, DEFAULT_KERNEL_WIDTH, 0.0),
            maximum_confidence_adjust: (
                DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
                DEFAULT_MAXIMUM_CONFIDENCE_ADJUST,
                0.0,
            ),
            extra_smoothing: (DEFAULT_EXTRA_SMOOTHING, DEFAULT_EXTRA_SMOOTHING, 0),
            kernel_type: HalfCausalEstimatorKernelType::Epanechnikov,
            confidence_adjust: HalfCausalEstimatorConfidenceAdjust::Symmetric,
            enable_expected_value: DEFAULT_ENABLE_EXPECTED_VALUE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBuilder {
    slots_per_day: Option<usize>,
    data_period: Option<usize>,
    filter_length: Option<usize>,
    kernel_width: Option<f64>,
    kernel_type: Option<HalfCausalEstimatorKernelType>,
    confidence_adjust: Option<HalfCausalEstimatorConfidenceAdjust>,
    maximum_confidence_adjust: Option<f64>,
    enable_expected_value: Option<bool>,
    extra_smoothing: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for HalfCausalEstimatorBuilder {
    fn default() -> Self {
        Self {
            slots_per_day: None,
            data_period: None,
            filter_length: None,
            kernel_width: None,
            kernel_type: None,
            confidence_adjust: None,
            maximum_confidence_adjust: None,
            enable_expected_value: None,
            extra_smoothing: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HalfCausalEstimatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn slots_per_day(mut self, slots_per_day: usize) -> Self {
        self.slots_per_day = Some(slots_per_day);
        self
    }

    #[inline]
    pub fn data_period(mut self, data_period: usize) -> Self {
        self.data_period = Some(data_period);
        self
    }

    #[inline]
    pub fn filter_length(mut self, filter_length: usize) -> Self {
        self.filter_length = Some(filter_length);
        self
    }

    #[inline]
    pub fn kernel_width(mut self, kernel_width: f64) -> Self {
        self.kernel_width = Some(kernel_width);
        self
    }

    #[inline]
    pub fn kernel_type(mut self, kernel_type: HalfCausalEstimatorKernelType) -> Self {
        self.kernel_type = Some(kernel_type);
        self
    }

    #[inline]
    pub fn confidence_adjust(
        mut self,
        confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    ) -> Self {
        self.confidence_adjust = Some(confidence_adjust);
        self
    }

    #[inline]
    pub fn maximum_confidence_adjust(mut self, maximum_confidence_adjust: f64) -> Self {
        self.maximum_confidence_adjust = Some(maximum_confidence_adjust);
        self
    }

    #[inline]
    pub fn enable_expected_value(mut self, enable_expected_value: bool) -> Self {
        self.enable_expected_value = Some(enable_expected_value);
        self
    }

    #[inline]
    pub fn extra_smoothing(mut self, extra_smoothing: usize) -> Self {
        self.extra_smoothing = Some(extra_smoothing);
        self
    }

    #[inline]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    fn build_params(&self) -> HalfCausalEstimatorParams {
        HalfCausalEstimatorParams {
            slots_per_day: self.slots_per_day,
            data_period: self.data_period,
            filter_length: self.filter_length,
            kernel_width: self.kernel_width,
            kernel_type: self.kernel_type,
            confidence_adjust: self.confidence_adjust,
            maximum_confidence_adjust: self.maximum_confidence_adjust,
            enable_expected_value: self.enable_expected_value,
            extra_smoothing: self.extra_smoothing,
        }
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
        let input = HalfCausalEstimatorInput::from_slice(data, self.build_params());
        half_causal_estimator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
        let source = self.source.as_deref().unwrap_or(DEFAULT_SOURCE);
        let input = HalfCausalEstimatorInput::from_candles(candles, source, self.build_params());
        half_causal_estimator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<HalfCausalEstimatorStream, HalfCausalEstimatorError> {
        HalfCausalEstimatorStream::try_new(self.build_params())
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorBatchBuilder {
    range: HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    source: Option<String>,
}

impl Default for HalfCausalEstimatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: HalfCausalEstimatorBatchRange::default(),
            kernel: Kernel::Auto,
            source: None,
        }
    }
}

impl HalfCausalEstimatorBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    #[inline]
    pub fn slots_per_day(mut self, slots_per_day: usize) -> Self {
        self.range.slots_per_day = Some(slots_per_day);
        self
    }

    #[inline]
    pub fn data_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.data_period = (start, end, step);
        self
    }

    #[inline]
    pub fn filter_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.filter_length = (start, end, step);
        self
    }

    #[inline]
    pub fn kernel_width_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.kernel_width = (start, end, step);
        self
    }

    #[inline]
    pub fn maximum_confidence_adjust_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.maximum_confidence_adjust = (start, end, step);
        self
    }

    #[inline]
    pub fn extra_smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.extra_smoothing = (start, end, step);
        self
    }

    #[inline]
    pub fn kernel_type(mut self, kernel_type: HalfCausalEstimatorKernelType) -> Self {
        self.range.kernel_type = kernel_type;
        self
    }

    #[inline]
    pub fn confidence_adjust(
        mut self,
        confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    ) -> Self {
        self.range.confidence_adjust = confidence_adjust;
        self
    }

    #[inline]
    pub fn enable_expected_value(mut self, enable_expected_value: bool) -> Self {
        self.range.enable_expected_value = enable_expected_value;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
        half_causal_estimator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
        let source = self.source.as_deref().unwrap_or(DEFAULT_SOURCE);
        let params = HalfCausalEstimatorParams {
            slots_per_day: self.range.slots_per_day,
            data_period: Some(self.range.data_period.0),
            filter_length: Some(self.range.filter_length.0),
            kernel_width: Some(self.range.kernel_width.0),
            kernel_type: Some(self.range.kernel_type),
            confidence_adjust: Some(self.range.confidence_adjust),
            maximum_confidence_adjust: Some(self.range.maximum_confidence_adjust.0),
            enable_expected_value: Some(self.range.enable_expected_value),
            extra_smoothing: Some(self.range.extra_smoothing.0),
        };
        let prepared = prepare_source_and_slots(&HalfCausalEstimatorInput::from_candles(
            candles, source, params,
        ))?;
        let mut sweep = self.range.clone();
        if sweep.slots_per_day.is_none() {
            sweep.slots_per_day = Some(prepared.slots_per_day);
        }
        half_causal_estimator_batch_prepared(&prepared.values, &prepared.slots, &sweep, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum HalfCausalEstimatorError {
    #[error("half_causal_estimator: Input data slice is empty.")]
    EmptyInputData,
    #[error("half_causal_estimator: All values are NaN.")]
    AllValuesNaN,
    #[error("half_causal_estimator: Missing slots_per_day for slice input.")]
    MissingSlotsPerDay,
    #[error("half_causal_estimator: Invalid slots_per_day: {slots_per_day}")]
    InvalidSlotsPerDay { slots_per_day: usize },
    #[error("half_causal_estimator: Invalid data_period: {data_period}")]
    InvalidDataPeriod { data_period: usize },
    #[error("half_causal_estimator: Invalid filter_length: {filter_length}")]
    InvalidFilterLength { filter_length: usize },
    #[error("half_causal_estimator: Invalid kernel_width: {kernel_width}")]
    InvalidKernelWidth { kernel_width: f64 },
    #[error(
        "half_causal_estimator: Invalid maximum_confidence_adjust: {maximum_confidence_adjust}"
    )]
    InvalidMaximumConfidenceAdjust { maximum_confidence_adjust: f64 },
    #[error("half_causal_estimator: Invalid source: {source_name}")]
    InvalidSource { source_name: String },
    #[error("half_causal_estimator: Unable to infer minute timeframe from timestamps.")]
    UnableToInferMinuteTimeframe,
    #[error("half_causal_estimator: Invalid timestamp: {timestamp}")]
    InvalidTimestamp { timestamp: i64 },
    #[error(
        "half_causal_estimator: Candle field length mismatch for {field}: expected={expected}, got={got}"
    )]
    CandleFieldLengthMismatch {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("half_causal_estimator: Invalid extra_smoothing: {extra_smoothing}")]
    InvalidExtraSmoothing { extra_smoothing: usize },
    #[error(
        "half_causal_estimator: Prepared slot {slot} at row {index} is outside slots_per_day={slots_per_day}"
    )]
    InvalidPreparedSlot {
        index: usize,
        slot: usize,
        slots_per_day: usize,
    },
    #[error(
        "half_causal_estimator: Output length mismatch: expected = {expected}, estimate = {estimate_got}, expected_value = {expected_value_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        estimate_got: usize,
        expected_value_got: usize,
    },
    #[error("half_causal_estimator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("half_causal_estimator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("half_causal_estimator: Arithmetic overflow while resolving {context}.")]
    ArithmeticOverflow { context: &'static str },
    #[error(
        "half_causal_estimator: Allocation failed for {context}: requested {elements} elements."
    )]
    AllocationFailed {
        context: &'static str,
        elements: usize,
    },
    #[error(
        "half_causal_estimator: Public CPU retained-memory budget exceeded for {context}: requested={requested_bytes} bytes, budget={budget_bytes} bytes."
    )]
    PublicRetainedMemoryBudgetExceeded {
        context: &'static str,
        requested_bytes: usize,
        budget_bytes: usize,
    },
    #[error("half_causal_estimator: Sweep cardinality overflow.")]
    SweepCardinalityOverflow,
}

#[inline]
fn try_vec_with_capacity<T>(
    elements: usize,
    context: &'static str,
) -> Result<Vec<T>, HalfCausalEstimatorError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| HalfCausalEstimatorError::AllocationFailed { context, elements })?;
    Ok(values)
}

#[inline]
fn try_filled_vec<T: Clone>(
    elements: usize,
    value: T,
    context: &'static str,
) -> Result<Vec<T>, HalfCausalEstimatorError> {
    let mut values = try_vec_with_capacity(elements, context)?;
    values.resize(elements, value);
    Ok(values)
}

#[inline]
fn try_alloc_f64(
    elements: usize,
    context: &'static str,
) -> Result<Vec<f64>, HalfCausalEstimatorError> {
    try_filled_vec(elements, f64::NAN, context)
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    slots_per_day: usize,
    data_period: usize,
    filter_length: usize,
    real_filter_length: usize,
    window_size: usize,
    kernel_width: f64,
    kernel_type: HalfCausalEstimatorKernelType,
    confidence_adjust: HalfCausalEstimatorConfidenceAdjust,
    maximum_confidence_adjust_factor: f64,
    enable_expected_value: bool,
    wma_length: usize,
}

#[derive(Debug, Clone)]
struct PreparedInput<'a> {
    values: Cow<'a, [f64]>,
    slots: PreparedSlots,
    slots_per_day: usize,
}

#[derive(Debug, Clone)]
enum PreparedSlots {
    Sequential {
        slots_per_day: usize,
    },
    Explicit {
        slots: Vec<usize>,
        session_starts: Vec<bool>,
    },
}

#[derive(Debug, Clone)]
struct TimeOfDayBucket {
    values: Vec<f64>,
    next: usize,
    count: usize,
    moments: StablePopulationMoments,
    bounded: bool,
}

impl TimeOfDayBucket {
    #[inline]
    fn try_new(capacity: usize) -> Result<Self, HalfCausalEstimatorError> {
        let bounded = capacity > 0;
        Ok(Self {
            values: if bounded {
                try_filled_vec(capacity, 0.0, "time-of-day bucket")?
            } else {
                Vec::new()
            },
            next: 0,
            count: 0,
            moments: StablePopulationMoments::default(),
            bounded,
        })
    }

    #[inline]
    fn recompute_bounded_moments(&mut self) {
        let mut moments = StablePopulationMoments::default();
        let start = if self.count == self.values.len() {
            self.next
        } else {
            0
        };
        for offset in 0..self.count {
            let index = (start + offset) % self.values.len();
            moments.add(self.values[index]);
        }
        self.moments = moments;
    }

    #[inline]
    fn add(&mut self, value: f64) {
        if self.bounded {
            if self.values.is_empty() {
                return;
            }
            if self.count < self.values.len() {
                self.values[self.next] = value;
                self.count += 1;
            } else {
                self.values[self.next] = value;
            }
            self.next += 1;
            if self.next == self.values.len() {
                self.next = 0;
            }
            self.recompute_bounded_moments();
        } else {
            self.count += 1;
            self.moments.add(value);
        }
    }

    #[inline]
    fn has_values(&self) -> bool {
        self.moments.count() > 0
    }

    #[inline]
    fn mean(&self) -> Option<f64> {
        self.moments.mean()
    }
}

#[derive(Debug, Clone)]
struct TimeOfDayStore {
    buckets: Vec<TimeOfDayBucket>,
}

impl TimeOfDayStore {
    #[inline]
    fn try_new(slots_per_day: usize, data_period: usize) -> Result<Self, HalfCausalEstimatorError> {
        slots_per_day.checked_mul(data_period).ok_or(
            HalfCausalEstimatorError::ArithmeticOverflow {
                context: "time-of-day store elements",
            },
        )?;
        let mut buckets = try_vec_with_capacity(slots_per_day, "time-of-day buckets")?;
        for _ in 0..slots_per_day {
            buckets.push(TimeOfDayBucket::try_new(data_period)?);
        }
        Ok(Self { buckets })
    }

    #[inline]
    fn add(&mut self, slot: usize, value: f64) {
        self.buckets[slot].add(value);
    }

    #[inline]
    fn has_values(&self, slot: usize) -> bool {
        self.buckets[slot].has_values()
    }

    #[inline]
    fn mean(&self, slot: usize) -> Option<f64> {
        self.buckets[slot].mean()
    }

    #[inline]
    fn icv(&self, slot: usize, maximum_confidence_adjust_factor: f64) -> f64 {
        self.buckets[slot]
            .moments
            .creator_inverse_cv(maximum_confidence_adjust_factor)
    }
}

#[derive(Debug, Clone)]
struct FixedFrontBuffer {
    values: Vec<f64>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl FixedFrontBuffer {
    #[inline]
    fn try_new(capacity: usize, context: &'static str) -> Result<Self, HalfCausalEstimatorError> {
        Ok(Self {
            values: try_filled_vec(capacity, 0.0, context)?,
            capacity,
            head: 0,
            len: 0,
        })
    }

    #[inline]
    fn push(&mut self, value: f64) {
        if self.capacity == 0 {
            return;
        }
        if self.len == 0 {
            self.values[0] = value;
            self.len = 1;
            self.head = 0;
            return;
        }
        self.head = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        self.values[self.head] = value;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.len).map(move |offset| {
            let mut index = self.head + offset;
            if index >= self.capacity {
                index -= self.capacity;
            }
            self.values[index]
        })
    }
}

#[derive(Debug, Clone)]
struct FillWmaState {
    length: usize,
    first: Option<f64>,
    values: VecDeque<f64>,
    denominator: f64,
}

impl FillWmaState {
    #[inline]
    fn try_new(length: usize) -> Result<Self, HalfCausalEstimatorError> {
        let mut values = VecDeque::new();
        values.try_reserve_exact(length).map_err(|_| {
            HalfCausalEstimatorError::AllocationFailed {
                context: "fill WMA history",
                elements: length,
            }
        })?;
        Ok(Self {
            length,
            first: None,
            values,
            denominator: (length * (length + 1) / 2) as f64,
        })
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        // Pine series history advances on every bar, including an `na`
        // estimate. Compressing holes would make `source[i]` refer to a
        // different bar after the first missing estimate.
        self.values.push_front(value);
        if self.values.len() > self.length {
            let _ = self.values.pop_back();
        }
        if !value.is_finite() {
            return None;
        }
        let first = *self.first.get_or_insert(value);
        if self.length == 1 {
            return Some(value);
        }

        let mut sum = 0.0;
        for i in 0..self.length {
            let sample = self
                .values
                .get(i)
                .copied()
                .filter(|sample| sample.is_finite())
                .unwrap_or(first);
            sum += sample * (self.length - i) as f64;
        }
        Some(sum / self.denominator)
    }
}

#[derive(Debug, Clone)]
struct FutureWindowCache {
    values: VecDeque<f64>,
    weights: VecDeque<f64>,
    window_key: Option<usize>,
    length: usize,
}

impl FutureWindowCache {
    #[inline]
    fn try_new(length: usize) -> Result<Self, HalfCausalEstimatorError> {
        let mut values = VecDeque::new();
        values.try_reserve_exact(length).map_err(|_| {
            HalfCausalEstimatorError::AllocationFailed {
                context: "future value cache",
                elements: length,
            }
        })?;
        let mut weights = VecDeque::new();
        weights.try_reserve_exact(length).map_err(|_| {
            HalfCausalEstimatorError::AllocationFailed {
                context: "future confidence cache",
                elements: length,
            }
        })?;
        Ok(Self {
            values,
            weights,
            window_key: None,
            length,
        })
    }

    #[inline]
    fn next_valid(
        store: &TimeOfDayStore,
        start_key: usize,
        maximum_confidence_adjust_factor: f64,
    ) -> Option<(usize, f64, f64)> {
        let slots_per_day = store.buckets.len();
        for offset in 1..=slots_per_day {
            let key = (start_key + offset) % slots_per_day;
            if store.has_values(key) {
                return Some((
                    key,
                    store.mean(key)?,
                    store.icv(key, maximum_confidence_adjust_factor).max(0.0),
                ));
            }
        }
        None
    }

    #[inline]
    fn initialize(
        &mut self,
        store: &TimeOfDayStore,
        current_key: usize,
        maximum_confidence_adjust_factor: f64,
    ) -> Option<()> {
        self.values.clear();
        self.weights.clear();
        self.window_key = None;
        let mut key = current_key;
        while self.values.len() < self.length {
            let (next_key, value, weight) =
                Self::next_valid(store, key, maximum_confidence_adjust_factor)?;
            key = next_key;
            // Pine `unshift`s every newly discovered point. The final array is
            // therefore farthest-first, immediately followed by causal data.
            self.values.push_front(value);
            self.weights.push_front(weight);
        }
        self.window_key = Some(key);
        Some(())
    }

    #[inline]
    fn maintain(
        &mut self,
        store: &TimeOfDayStore,
        maximum_confidence_adjust_factor: f64,
    ) -> Option<()> {
        if self.values.len() != self.length || self.weights.len() != self.length {
            return None;
        }
        let key = self.window_key?;
        let (next_key, value, weight) =
            Self::next_valid(store, key, maximum_confidence_adjust_factor)?;
        let _ = self.values.pop_back();
        let _ = self.weights.pop_back();
        self.values.push_front(value);
        self.weights.push_front(weight);
        self.window_key = Some(next_key);
        Some(())
    }
}

#[derive(Debug, Clone)]
struct ExpectedWindowCache {
    values: VecDeque<f64>,
    window_key: Option<usize>,
    window_size: usize,
}

impl ExpectedWindowCache {
    #[inline]
    fn try_new(window_size: usize) -> Result<Self, HalfCausalEstimatorError> {
        let mut values = VecDeque::new();
        values.try_reserve_exact(window_size).map_err(|_| {
            HalfCausalEstimatorError::AllocationFailed {
                context: "expected-value cache",
                elements: window_size,
            }
        })?;
        Ok(Self {
            values,
            window_key: None,
            window_size,
        })
    }

    #[inline]
    fn initialize(
        &mut self,
        store: &TimeOfDayStore,
        current_key: usize,
        causal_buffer: &FixedFrontBuffer,
    ) -> Option<()> {
        self.values.clear();
        self.values.extend(causal_buffer.iter());
        self.window_key = None;
        let mut key = current_key;
        while self.values.len() < self.window_size {
            let (next_key, value, _) = FutureWindowCache::next_valid(store, key, 0.0)?;
            key = next_key;
            self.values.push_front(value);
        }
        self.window_key = Some(key);
        Some(())
    }

    #[inline]
    fn maintain(&mut self, store: &TimeOfDayStore) -> Option<()> {
        if self.values.len() != self.window_size {
            return None;
        }
        let key = self.window_key?;
        let (next_key, value, _) = FutureWindowCache::next_valid(store, key, 0.0)?;
        let _ = self.values.pop_back();
        self.values.push_front(value);
        self.window_key = Some(next_key);
        Some(())
    }
}

#[derive(Debug, Clone)]
struct HalfCausalEstimatorContext {
    params: ResolvedParams,
    store: TimeOfDayStore,
    source_buffer: FixedFrontBuffer,
    average_buffer: FixedFrontBuffer,
    wma: FillWmaState,
    kernel: Vec<f64>,
    future: FutureWindowCache,
    expected: ExpectedWindowCache,
    ready: bool,
    index: usize,
}

#[inline]
fn public_retained_budget_error(requested_bytes: usize) -> HalfCausalEstimatorError {
    HalfCausalEstimatorError::PublicRetainedMemoryBudgetExceeded {
        context: "public CPU retained context v1",
        requested_bytes,
        budget_bytes: HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1,
    }
}

#[inline]
fn retained_checked_add(left: usize, right: usize) -> Result<usize, HalfCausalEstimatorError> {
    left.checked_add(right)
        .ok_or_else(|| public_retained_budget_error(usize::MAX))
}

#[inline]
fn retained_checked_mul(left: usize, right: usize) -> Result<usize, HalfCausalEstimatorError> {
    left.checked_mul(right)
        .ok_or_else(|| public_retained_budget_error(usize::MAX))
}

/// Conservative logical bytes retained by one public CPU context. The budget
/// counts every requested heap element plus the bucket/context structures
/// before any of their backing `Vec`/`VecDeque` allocations are attempted.
#[inline]
fn public_cpu_retained_bytes_v1(params: ResolvedParams) -> Result<usize, HalfCausalEstimatorError> {
    let bucket_value_elements = retained_checked_mul(params.slots_per_day, params.data_period)?;
    let future_elements = retained_checked_mul(params.real_filter_length - 1, 2)?;
    let expected_elements = if params.enable_expected_value {
        params.window_size
    } else {
        0
    };
    let mut f64_elements = bucket_value_elements;
    for elements in [
        params.real_filter_length,
        params.real_filter_length,
        params.wma_length,
        params.window_size,
        future_elements,
        expected_elements,
    ] {
        f64_elements = retained_checked_add(f64_elements, elements)?;
    }
    let f64_bytes = retained_checked_mul(f64_elements, std::mem::size_of::<f64>())?;
    let bucket_bytes =
        retained_checked_mul(params.slots_per_day, std::mem::size_of::<TimeOfDayBucket>())?;
    retained_checked_add(
        retained_checked_add(
            std::mem::size_of::<HalfCausalEstimatorContext>(),
            bucket_bytes,
        )?,
        f64_bytes,
    )
}

#[inline]
fn validate_public_cpu_retained_budget_v1(
    params: ResolvedParams,
) -> Result<(), HalfCausalEstimatorError> {
    let requested_bytes = public_cpu_retained_bytes_v1(params)?;
    if requested_bytes > HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1 {
        return Err(public_retained_budget_error(requested_bytes));
    }
    Ok(())
}

impl HalfCausalEstimatorContext {
    #[inline]
    fn try_new(params: ResolvedParams) -> Result<Self, HalfCausalEstimatorError> {
        validate_public_cpu_retained_budget_v1(params)?;
        Ok(Self {
            store: TimeOfDayStore::try_new(params.slots_per_day, params.data_period)?,
            source_buffer: FixedFrontBuffer::try_new(
                params.real_filter_length,
                "causal source buffer",
            )?,
            average_buffer: FixedFrontBuffer::try_new(
                params.real_filter_length,
                "causal average buffer",
            )?,
            wma: FillWmaState::try_new(params.wma_length)?,
            kernel: build_kernel(params)?,
            future: FutureWindowCache::try_new(params.real_filter_length - 1)?,
            expected: ExpectedWindowCache::try_new(if params.enable_expected_value {
                params.window_size
            } else {
                0
            })?,
            ready: false,
            index: 0,
            params,
        })
    }

    #[inline]
    fn update(
        &mut self,
        value: f64,
        slot: usize,
        session_start: bool,
    ) -> (Option<f64>, Option<f64>) {
        if !self.ready && self.index > self.params.window_size && session_start {
            self.ready = true;
        }

        self.source_buffer.push(value);
        self.average_buffer
            .push(self.store.mean(slot).unwrap_or(f64::NAN));

        let future_ready = if self.ready {
            if session_start {
                self.future.initialize(
                    &self.store,
                    slot,
                    self.params.maximum_confidence_adjust_factor,
                )
            } else {
                self.future
                    .maintain(&self.store, self.params.maximum_confidence_adjust_factor)
            }
        } else {
            None
        };

        let expected_ready = if self.params.enable_expected_value && self.ready {
            if session_start {
                self.expected
                    .initialize(&self.store, slot, &self.average_buffer)
            } else {
                self.expected.maintain(&self.store)
            }
        } else {
            None
        };

        let estimate_raw = if future_ready.is_some() && self.source_buffer.is_full() {
            self.compute_estimate_window()
        } else {
            None
        };
        // The WMA is a Pine series function and therefore advances even when
        // this bar's raw estimate is missing.
        let estimate = self.wma.update(estimate_raw.unwrap_or(f64::NAN));
        let expected_value = if expected_ready.is_some() {
            self.compute_expected_window()
        } else {
            None
        };

        if value.is_finite() {
            self.store.add(slot, value);
        }

        self.index += 1;
        (estimate, expected_value)
    }

    #[inline]
    fn compute_estimate_window(&self) -> Option<f64> {
        let future_len = self.params.real_filter_length.saturating_sub(1);
        let uses_confidence = !matches!(
            self.params.confidence_adjust,
            HalfCausalEstimatorConfidenceAdjust::None
        );
        let causal_values = &self.source_buffer;
        if causal_values.len != self.params.real_filter_length {
            return None;
        }
        if self.future.values.len() + causal_values.len != self.params.window_size {
            return None;
        }
        if self.future.weights.len() != future_len {
            return None;
        }

        let mut sum = NeumaierSum::default();
        let mut kernel_index = 0usize;
        let linear_fill = if uses_confidence
            && matches!(
                self.params.confidence_adjust,
                HalfCausalEstimatorConfidenceAdjust::Linear
            ) {
            let weight_sum: f64 = self.future.weights.iter().copied().sum();
            if self.params.real_filter_length > 1 {
                2.0 - weight_sum / future_len as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        for i in 0..future_len {
            let value = self.future.values[i];
            if !value.is_finite() {
                return None;
            }
            let confidence = if uses_confidence {
                self.future.weights[i]
            } else {
                1.0
            };
            sum.add_weighted(value, confidence, self.kernel[kernel_index]);
            kernel_index += 1;
        }

        for (i, value) in causal_values.iter().enumerate() {
            if !value.is_finite() {
                return None;
            }
            let confidence = match self.params.confidence_adjust {
                HalfCausalEstimatorConfidenceAdjust::None => 1.0,
                HalfCausalEstimatorConfidenceAdjust::Symmetric => {
                    if i == 0 {
                        1.0
                    } else {
                        2.0 - self.future.weights[future_len - i]
                    }
                }
                HalfCausalEstimatorConfidenceAdjust::Linear => linear_fill,
            };
            sum.add_weighted(value, confidence, self.kernel[kernel_index]);
            kernel_index += 1;
        }

        Some(sum.total())
    }

    #[inline]
    fn compute_expected_window(&self) -> Option<f64> {
        if self.expected.values.len() != self.params.window_size {
            return None;
        }
        let mut sum = NeumaierSum::default();
        for (value, coefficient) in self.expected.values.iter().zip(&self.kernel) {
            if !value.is_finite() {
                return None;
            }
            sum.add_weighted(*value, 1.0, *coefficient);
        }
        Some(sum.total())
    }
}

#[derive(Debug, Clone)]
pub struct HalfCausalEstimatorStream {
    ctx: HalfCausalEstimatorContext,
    next_slot: usize,
}

impl HalfCausalEstimatorStream {
    #[inline]
    pub fn try_new(params: HalfCausalEstimatorParams) -> Result<Self, HalfCausalEstimatorError> {
        let slots_per_day = params
            .slots_per_day
            .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
        let resolved = resolve_params(&params, slots_per_day)?;
        Ok(Self {
            ctx: HalfCausalEstimatorContext::try_new(resolved)?,
            next_slot: 0,
        })
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.ctx.params.slots_per_day + self.ctx.params.window_size
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> (Option<f64>, Option<f64>) {
        let session_start = self.next_slot == 0;
        let out = self.ctx.update(value, self.next_slot, session_start);
        self.next_slot += 1;
        if self.next_slot == self.ctx.params.slots_per_day {
            self.next_slot = 0;
        }
        out
    }
}

#[inline(always)]
fn resolve_params(
    params: &HalfCausalEstimatorParams,
    slots_per_day: usize,
) -> Result<ResolvedParams, HalfCausalEstimatorError> {
    if slots_per_day < 2 || slots_per_day > 1440 || 1440 % slots_per_day != 0 {
        return Err(HalfCausalEstimatorError::InvalidSlotsPerDay { slots_per_day });
    }

    let data_period = params.data_period.unwrap_or(DEFAULT_DATA_PERIOD);

    let filter_length = params.filter_length.unwrap_or(DEFAULT_FILTER_LENGTH);
    if filter_length < 2 {
        return Err(HalfCausalEstimatorError::InvalidFilterLength { filter_length });
    }

    let kernel_width = params.kernel_width.unwrap_or(DEFAULT_KERNEL_WIDTH);
    if !kernel_width.is_finite() || kernel_width <= 0.0 {
        return Err(HalfCausalEstimatorError::InvalidKernelWidth { kernel_width });
    }

    let maximum_confidence_adjust = params
        .maximum_confidence_adjust
        .unwrap_or(DEFAULT_MAXIMUM_CONFIDENCE_ADJUST);
    if !maximum_confidence_adjust.is_finite() || !(0.0..=100.0).contains(&maximum_confidence_adjust)
    {
        return Err(HalfCausalEstimatorError::InvalidMaximumConfidenceAdjust {
            maximum_confidence_adjust,
        });
    }

    let kernel_type = params.kernel_type.unwrap_or_default();
    let confidence_adjust = params.confidence_adjust.unwrap_or_default();
    let extra_smoothing = params.extra_smoothing.unwrap_or(DEFAULT_EXTRA_SMOOTHING);
    if extra_smoothing > 2 {
        return Err(HalfCausalEstimatorError::InvalidExtraSmoothing { extra_smoothing });
    }
    let wma_length = extra_smoothing
        .checked_add(1)
        .ok_or(HalfCausalEstimatorError::InvalidExtraSmoothing { extra_smoothing })?;
    let real_filter_length = if matches!(kernel_type, HalfCausalEstimatorKernelType::Sinc) {
        filter_length
            .checked_mul(2)
            .ok_or(HalfCausalEstimatorError::ArithmeticOverflow {
                context: "Sinc real_filter_length",
            })?
    } else {
        filter_length
    };
    let window_size = real_filter_length
        .checked_mul(2)
        .and_then(|twice| twice.checked_sub(1))
        .ok_or(HalfCausalEstimatorError::ArithmeticOverflow {
            context: "half-causal window_size",
        })?;

    Ok(ResolvedParams {
        slots_per_day,
        data_period,
        filter_length,
        real_filter_length,
        window_size,
        kernel_width,
        kernel_type,
        confidence_adjust,
        maximum_confidence_adjust_factor: maximum_confidence_adjust * 0.01,
        enable_expected_value: params
            .enable_expected_value
            .unwrap_or(DEFAULT_ENABLE_EXPECTED_VALUE),
        wma_length,
    })
}

#[inline(always)]
fn gaussian_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    (-ratio * ratio * 0.25).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[inline(always)]
fn epanechnikov_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    if ratio.abs() <= 1.0 {
        0.75 * (1.0 - ratio * ratio)
    } else {
        0.0
    }
}

#[inline(always)]
fn triangular_kernel(centered_index: f64, bandwidth: f64) -> f64 {
    let ratio = centered_index / bandwidth;
    if ratio.abs() <= 1.0 {
        1.0 - ratio.abs()
    } else {
        0.0
    }
}

#[inline(always)]
fn blackman(index: f64, length: f64) -> f64 {
    0.42 - 0.5 * ((2.0 * std::f64::consts::PI * index) / (length - 1.0)).cos()
        + 0.08 * ((4.0 * std::f64::consts::PI * index) / (length - 1.0)).cos()
}

#[inline(always)]
fn sinc(centered_index: f64, width: f64) -> f64 {
    let fc = 0.5 / width;
    if centered_index.abs() <= f64::EPSILON {
        1.0
    } else {
        let x = std::f64::consts::PI * fc * centered_index;
        x.sin() / x
    }
}

#[inline(always)]
fn build_kernel(params: ResolvedParams) -> Result<Vec<f64>, HalfCausalEstimatorError> {
    let mut kernel = try_vec_with_capacity(params.window_size, "kernel coefficients")?;
    let center = (params.window_size - 1) as f64 * 0.5;
    let length = params.window_size as f64;
    let mut normalization = 0.0;

    for i in 0..params.window_size {
        let index = i as f64;
        let centered = index - center;
        let weight = match params.kernel_type {
            HalfCausalEstimatorKernelType::Gaussian => {
                gaussian_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Epanechnikov => {
                epanechnikov_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Triangular => {
                triangular_kernel(centered, params.kernel_width)
            }
            HalfCausalEstimatorKernelType::Sinc => {
                sinc(centered, params.kernel_width) * blackman(index, length)
            }
        };
        normalization += weight;
        kernel.push(weight);
    }

    if normalization != 0.0 {
        for weight in &mut kernel {
            *weight /= normalization;
        }
    }

    Ok(kernel)
}

#[inline]
fn infer_slots_per_day(timestamps: &[i64]) -> Result<usize, HalfCausalEstimatorError> {
    for &timestamp in timestamps {
        let _ = validate_timestamp(timestamp)?;
    }
    let mut min_positive = i64::MAX;
    for pair in timestamps.windows(2) {
        let delta = pair[1].checked_sub(pair[0]).unwrap_or(i64::MAX);
        if delta > 0 && delta < DAY_MS && delta < min_positive {
            min_positive = delta;
        }
    }

    if min_positive == i64::MAX || min_positive % 60_000 != 0 {
        return Err(HalfCausalEstimatorError::UnableToInferMinuteTimeframe);
    }

    let minutes = (min_positive / 60_000) as usize;
    if minutes == 0 || 1440 % minutes != 0 {
        return Err(HalfCausalEstimatorError::UnableToInferMinuteTimeframe);
    }
    Ok(1440 / minutes)
}

#[inline]
fn validate_timestamp(timestamp: i64) -> Result<DateTime<Utc>, HalfCausalEstimatorError> {
    DateTime::<Utc>::from_timestamp_millis(timestamp)
        .ok_or(HalfCausalEstimatorError::InvalidTimestamp { timestamp })
}

#[inline]
fn slot_from_timestamp(
    timestamp: i64,
    slots_per_day: usize,
) -> Result<usize, HalfCausalEstimatorError> {
    let dt = validate_timestamp(timestamp)?;
    let minutes = dt.hour() as usize * 60 + dt.minute() as usize;
    let minutes_per_slot = 1440 / slots_per_day;
    Ok(minutes / minutes_per_slot)
}

#[inline]
fn validate_candle_field_length(
    field: &'static str,
    expected: usize,
    got: usize,
) -> Result<(), HalfCausalEstimatorError> {
    if got != expected {
        return Err(HalfCausalEstimatorError::CandleFieldLengthMismatch {
            field,
            expected,
            got,
        });
    }
    Ok(())
}

#[inline]
fn validate_candle_source_lengths(
    candles: &Candles,
    source: &str,
) -> Result<usize, HalfCausalEstimatorError> {
    let expected = candles.close.len();
    if expected == 0 {
        return Err(HalfCausalEstimatorError::EmptyInputData);
    }
    validate_candle_field_length("timestamp", expected, candles.timestamp.len())?;
    match source.to_ascii_lowercase().as_str() {
        "volume" => validate_candle_field_length("volume", expected, candles.volume.len())?,
        "tr" => {
            validate_candle_field_length("high", expected, candles.high.len())?;
            validate_candle_field_length("low", expected, candles.low.len())?;
        }
        "change" | "test" => {}
        _ => {
            return Err(HalfCausalEstimatorError::InvalidSource {
                source_name: source.to_string(),
            });
        }
    }
    Ok(expected)
}

#[inline]
fn source_from_candles<'a>(
    candles: &'a Candles,
    source: &str,
    slots_per_day: usize,
) -> Result<Cow<'a, [f64]>, HalfCausalEstimatorError> {
    match source.to_ascii_lowercase().as_str() {
        "volume" => Ok(Cow::Borrowed(&candles.volume)),
        "tr" => {
            let mut out = try_vec_with_capacity(candles.close.len(), "true-range source")?;
            for (&high, &low) in candles.high.iter().zip(candles.low.iter()) {
                out.push(if low.is_finite() && low != 0.0 {
                    (high - low) / low * 100.0
                } else {
                    f64::NAN
                });
            }
            Ok(Cow::Owned(out))
        }
        "change" => {
            let mut out = try_vec_with_capacity(candles.close.len(), "change source")?;
            let mut previous: Option<f64> = None;
            for &close in &candles.close {
                let prior = previous.filter(|value| value.is_finite()).unwrap_or(close);
                let denom = close.min(prior);
                if denom.is_finite() && denom != 0.0 {
                    out.push((close - prior).abs() / denom * 100.0);
                } else {
                    out.push(f64::NAN);
                }
                previous = Some(close);
            }
            Ok(Cow::Owned(out))
        }
        "test" => {
            let mut out = try_vec_with_capacity(candles.timestamp.len(), "test source")?;
            for &timestamp in &candles.timestamp {
                let slot = slot_from_timestamp(timestamp, slots_per_day)?;
                let cycle = slots_per_day as f64;
                let value = ((std::f64::consts::PI / cycle) * slot as f64).sin();
                out.push((value * value).max(0.0) * 100.0);
            }
            Ok(Cow::Owned(out))
        }
        _ => Err(HalfCausalEstimatorError::InvalidSource {
            source_name: source.to_string(),
        }),
    }
}

#[inline]
fn prepare_source_and_slots<'a>(
    input: &HalfCausalEstimatorInput<'a>,
) -> Result<PreparedInput<'a>, HalfCausalEstimatorError> {
    match &input.data {
        HalfCausalEstimatorData::Slice(values) => {
            if values.is_empty() {
                return Err(HalfCausalEstimatorError::EmptyInputData);
            }
            let slots_per_day = input
                .params
                .slots_per_day
                .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
            Ok(PreparedInput {
                values: Cow::Borrowed(values),
                slots: PreparedSlots::Sequential { slots_per_day },
                slots_per_day,
            })
        }
        HalfCausalEstimatorData::Candles { candles, source } => {
            let candle_len = validate_candle_source_lengths(candles, source)?;
            for &timestamp in &candles.timestamp {
                let _ = validate_timestamp(timestamp)?;
            }
            let slots_per_day = match input.params.slots_per_day {
                Some(slots) => slots,
                None => infer_slots_per_day(&candles.timestamp)?,
            };
            if slots_per_day < 2 || slots_per_day > 1440 || 1440 % slots_per_day != 0 {
                return Err(HalfCausalEstimatorError::InvalidSlotsPerDay { slots_per_day });
            }
            let mut slots = try_vec_with_capacity(candle_len, "prepared candle slots")?;
            let mut session_starts =
                try_vec_with_capacity(candle_len, "prepared candle session starts")?;
            let mut previous_utc_day = None;
            for &timestamp in &candles.timestamp {
                slots.push(slot_from_timestamp(timestamp, slots_per_day)?);
                let utc_day = timestamp.div_euclid(DAY_MS);
                session_starts.push(
                    previous_utc_day
                        .map(|previous| previous != utc_day)
                        .unwrap_or(true),
                );
                previous_utc_day = Some(utc_day);
            }
            let values = source_from_candles(candles, source, slots_per_day)?;
            Ok(PreparedInput {
                values,
                slots: PreparedSlots::Explicit {
                    slots,
                    session_starts,
                },
                slots_per_day,
            })
        }
    }
}

#[inline]
fn first_finite(values: &[f64]) -> usize {
    values
        .iter()
        .position(|value| value.is_finite())
        .unwrap_or(values.len())
}

#[inline]
fn resolve_and_prepare<'a>(
    input: &HalfCausalEstimatorInput<'a>,
) -> Result<(PreparedInput<'a>, ResolvedParams), HalfCausalEstimatorError> {
    let prepared = prepare_source_and_slots(input)?;
    let first = first_finite(&prepared.values);
    if first >= prepared.values.len() {
        return Err(HalfCausalEstimatorError::AllValuesNaN);
    }
    let resolved = effective_data_period_for_frame(
        resolve_params(&input.params, prepared.slots_per_day)?,
        &prepared.values,
        &prepared.slots,
    )?;
    Ok((prepared, resolved))
}

#[inline]
fn effective_data_period_for_frame(
    mut params: ResolvedParams,
    values: &[f64],
    slots: &PreparedSlots,
) -> Result<ResolvedParams, HalfCausalEstimatorError> {
    // Pine's data_period=0 means all available history and is represented by
    // online Welford moments without a retained sample buffer. For a bounded
    // data_period on a finite frame, no slot can observe more finite samples
    // than are present in that frame, so retaining a larger ring is needless.
    if params.data_period > 0 {
        let mut finite_per_slot =
            try_filled_vec(params.slots_per_day, 0usize, "finite slot counts")?;
        match slots {
            PreparedSlots::Sequential { slots_per_day } => {
                for (index, value) in values.iter().enumerate() {
                    if value.is_finite() {
                        let slot = index % *slots_per_day;
                        finite_per_slot[slot] += 1;
                    }
                }
            }
            PreparedSlots::Explicit { slots, .. } => {
                for (&slot, value) in slots.iter().zip(values) {
                    if value.is_finite() {
                        finite_per_slot[slot] += 1;
                    }
                }
            }
        }
        let frame_max = finite_per_slot.into_iter().max().unwrap_or(0);
        params.data_period = params.data_period.min(frame_max);
    }
    Ok(params)
}

#[inline]
fn resolve_grid_for_frame(
    combos: &[HalfCausalEstimatorParams],
    slots_per_day: usize,
    values: &[f64],
    slots: &PreparedSlots,
) -> Result<Vec<ResolvedParams>, HalfCausalEstimatorError> {
    let mut resolved = try_vec_with_capacity(combos.len(), "resolved sweep parameters")?;
    for params in combos {
        resolved.push(effective_data_period_for_frame(
            resolve_params(params, slots_per_day)?,
            values,
            slots,
        )?);
    }
    Ok(resolved)
}

#[inline]
fn compute_row(
    values: &[f64],
    slots: &PreparedSlots,
    params: ResolvedParams,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) -> Result<(), HalfCausalEstimatorError> {
    if !frame_can_become_ready(values.len(), slots, params.window_size) {
        estimate_out.fill(f64::NAN);
        expected_value_out.fill(f64::NAN);
        return Ok(());
    }
    let mut ctx = HalfCausalEstimatorContext::try_new(params)?;
    match slots {
        PreparedSlots::Sequential { slots_per_day } => {
            let mut slot = 0usize;
            for i in 0..values.len() {
                let session_start = slot == 0;
                let (estimate, expected_value) = ctx.update(values[i], slot, session_start);
                estimate_out[i] = estimate.unwrap_or(f64::NAN);
                expected_value_out[i] = expected_value.unwrap_or(f64::NAN);
                slot += 1;
                if slot == *slots_per_day {
                    slot = 0;
                }
            }
        }
        PreparedSlots::Explicit {
            slots,
            session_starts,
        } => {
            for i in 0..values.len() {
                let (estimate, expected_value) = ctx.update(values[i], slots[i], session_starts[i]);
                estimate_out[i] = estimate.unwrap_or(f64::NAN);
                expected_value_out[i] = expected_value.unwrap_or(f64::NAN);
            }
        }
    }
    Ok(())
}

#[inline]
fn frame_can_become_ready(len: usize, slots: &PreparedSlots, window_size: usize) -> bool {
    let Some(last_index) = len.checked_sub(1) else {
        return false;
    };
    match slots {
        PreparedSlots::Sequential { slots_per_day } => {
            let last_session_start = (last_index / *slots_per_day) * *slots_per_day;
            last_session_start > window_size
        }
        PreparedSlots::Explicit { session_starts, .. } => session_starts
            .iter()
            .take(len)
            .enumerate()
            .any(|(index, session_start)| index > window_size && *session_start),
    }
}

#[inline]
fn validate_frame_public_cpu_retained_budget_v1(
    params: ResolvedParams,
    len: usize,
    slots: &PreparedSlots,
) -> Result<(), HalfCausalEstimatorError> {
    if frame_can_become_ready(len, slots, params.window_size) {
        validate_public_cpu_retained_budget_v1(params)?;
    }
    Ok(())
}

#[inline]
pub fn half_causal_estimator(
    input: &HalfCausalEstimatorInput<'_>,
) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
    half_causal_estimator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn half_causal_estimator_with_kernel(
    input: &HalfCausalEstimatorInput<'_>,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorOutput, HalfCausalEstimatorError> {
    let _ = kernel;
    let (prepared, params) = resolve_and_prepare(input)?;
    validate_frame_public_cpu_retained_budget_v1(params, prepared.values.len(), &prepared.slots)?;
    let mut estimate = try_alloc_f64(prepared.values.len(), "estimate output")?;
    let mut expected_value = try_alloc_f64(prepared.values.len(), "expected-value output")?;
    compute_row(
        &prepared.values,
        &prepared.slots,
        params,
        &mut estimate,
        &mut expected_value,
    )?;
    Ok(HalfCausalEstimatorOutput {
        estimate,
        expected_value,
    })
}

#[inline]
pub fn half_causal_estimator_into_slices(
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
    input: &HalfCausalEstimatorInput<'_>,
    kernel: Kernel,
) -> Result<(), HalfCausalEstimatorError> {
    let _ = kernel;
    let (prepared, params) = resolve_and_prepare(input)?;
    let expected = prepared.values.len();
    if estimate_out.len() != expected || expected_value_out.len() != expected {
        return Err(HalfCausalEstimatorError::OutputLengthMismatch {
            expected,
            estimate_got: estimate_out.len(),
            expected_value_got: expected_value_out.len(),
        });
    }
    compute_row(
        &prepared.values,
        &prepared.slots,
        params,
        estimate_out,
        expected_value_out,
    )?;
    Ok(())
}

#[inline]
pub fn half_causal_estimator_into(
    input: &HalfCausalEstimatorInput<'_>,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) -> Result<(), HalfCausalEstimatorError> {
    half_causal_estimator_into_slices(estimate_out, expected_value_out, input, Kernel::Auto)
}

#[inline(always)]
fn axis_len_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<usize, HalfCausalEstimatorError> {
    if step == 0 || start == end {
        Ok(1)
    } else {
        (start.abs_diff(end) / step)
            .checked_add(1)
            .ok_or(HalfCausalEstimatorError::SweepCardinalityOverflow)
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, HalfCausalEstimatorError> {
    if step == 0 || start == end {
        let mut out = try_vec_with_capacity(1, "usize sweep axis")?;
        out.push(start);
        return Ok(out);
    }
    let count = axis_len_usize((start, end, step))?;
    let mut out = try_vec_with_capacity(count, "usize sweep axis")?;
    let mut value = start;
    if start < end {
        for _ in 0..count {
            out.push(value);
            value = value.checked_add(step).unwrap_or(value);
        }
    } else {
        for _ in 0..count {
            out.push(value);
            value = value.checked_sub(step).unwrap_or(value);
        }
    }
    if out.is_empty() {
        return Err(HalfCausalEstimatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn axis_len_f64((start, end, step): (f64, f64, f64)) -> Result<usize, HalfCausalEstimatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(HalfCausalEstimatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        return Ok(1);
    }
    let step = step.abs();
    let span = (end - start).abs();
    let intervals = (span / step + 1e-12).floor();
    if !intervals.is_finite() || intervals > (usize::MAX - 1) as f64 {
        return Err(HalfCausalEstimatorError::SweepCardinalityOverflow);
    }
    (intervals as usize)
        .checked_add(1)
        .ok_or(HalfCausalEstimatorError::SweepCardinalityOverflow)
}

#[inline(always)]
fn expand_axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, HalfCausalEstimatorError> {
    let count = axis_len_f64((start, end, step))?;
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        let mut out = try_vec_with_capacity(1, "f64 sweep axis")?;
        out.push(start);
        return Ok(out);
    }
    let step = step.abs();
    let mut out = try_vec_with_capacity(count, "f64 sweep axis")?;
    let mut value = start;
    if start < end {
        for _ in 0..count {
            out.push(value);
            value += step;
        }
    } else {
        for _ in 0..count {
            out.push(value);
            value -= step;
        }
    }
    if out.is_empty() {
        return Err(HalfCausalEstimatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn checked_sweep_cardinality(
    sweep: &HalfCausalEstimatorBatchRange,
) -> Result<usize, HalfCausalEstimatorError> {
    let axis_lengths = [
        axis_len_usize(sweep.data_period)?,
        axis_len_usize(sweep.filter_length)?,
        axis_len_f64(sweep.kernel_width)?,
        axis_len_f64(sweep.maximum_confidence_adjust)?,
        axis_len_usize(sweep.extra_smoothing)?,
    ];
    axis_lengths.into_iter().try_fold(1usize, |total, length| {
        total
            .checked_mul(length)
            .ok_or(HalfCausalEstimatorError::SweepCardinalityOverflow)
    })
}

#[inline]
fn expand_grid_half_causal_estimator(
    sweep: &HalfCausalEstimatorBatchRange,
) -> Result<Vec<HalfCausalEstimatorParams>, HalfCausalEstimatorError> {
    let cardinality = checked_sweep_cardinality(sweep)?;
    let data_periods = expand_axis_usize(sweep.data_period)?;
    let filter_lengths = expand_axis_usize(sweep.filter_length)?;
    let kernel_widths = expand_axis_f64(sweep.kernel_width)?;
    let maximum_confidence_adjusts = expand_axis_f64(sweep.maximum_confidence_adjust)?;
    let extra_smoothings = expand_axis_usize(sweep.extra_smoothing)?;

    let mut combos = try_vec_with_capacity(cardinality, "sweep combinations")?;
    for data_period in data_periods {
        for filter_length in filter_lengths.iter().copied() {
            for kernel_width in kernel_widths.iter().copied() {
                for maximum_confidence_adjust in maximum_confidence_adjusts.iter().copied() {
                    for extra_smoothing in extra_smoothings.iter().copied() {
                        let params = HalfCausalEstimatorParams {
                            slots_per_day: sweep.slots_per_day,
                            data_period: Some(data_period),
                            filter_length: Some(filter_length),
                            kernel_width: Some(kernel_width),
                            kernel_type: Some(sweep.kernel_type),
                            confidence_adjust: Some(sweep.confidence_adjust),
                            maximum_confidence_adjust: Some(maximum_confidence_adjust),
                            enable_expected_value: Some(sweep.enable_expected_value),
                            extra_smoothing: Some(extra_smoothing),
                        };
                        let slots_per_day = params
                            .slots_per_day
                            .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
                        let _ = resolve_params(&params, slots_per_day)?;
                        combos.push(params);
                    }
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn half_causal_estimator_batch_with_kernel(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    let slots_per_day = sweep
        .slots_per_day
        .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
    let slots = PreparedSlots::Sequential { slots_per_day };
    half_causal_estimator_batch_prepared(data, &slots, sweep, kernel)
}

#[inline]
fn half_causal_estimator_batch_prepared(
    data: &[f64],
    slots: &PreparedSlots,
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(HalfCausalEstimatorError::InvalidKernelForBatch(other)),
    };
    half_causal_estimator_batch_prepared_inner(
        data,
        slots,
        sweep,
        batch_kernel.to_non_batch(),
        true,
    )
}

#[inline]
pub fn half_causal_estimator_batch_slice(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    half_causal_estimator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn half_causal_estimator_batch_par_slice(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    half_causal_estimator_batch_inner(data, sweep, kernel, true)
}

#[inline]
pub fn half_causal_estimator_batch_inner(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    let slots_per_day = sweep
        .slots_per_day
        .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
    let slots = PreparedSlots::Sequential { slots_per_day };
    half_causal_estimator_batch_prepared_inner(data, &slots, sweep, kernel, parallel)
}

#[inline]
fn validate_prepared_slots(
    slots: &PreparedSlots,
    len: usize,
    slots_per_day: usize,
) -> Result<(), HalfCausalEstimatorError> {
    if let PreparedSlots::Explicit {
        slots,
        session_starts,
    } = slots
    {
        validate_candle_field_length("prepared_slots", len, slots.len())?;
        validate_candle_field_length("session_starts", len, session_starts.len())?;
        if let Some((index, &slot)) = slots
            .iter()
            .enumerate()
            .find(|(_, slot)| **slot >= slots_per_day)
        {
            return Err(HalfCausalEstimatorError::InvalidPreparedSlot {
                index,
                slot,
                slots_per_day,
            });
        }
    }
    Ok(())
}

#[inline]
fn half_causal_estimator_batch_prepared_inner(
    data: &[f64],
    slots: &PreparedSlots,
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<HalfCausalEstimatorBatchOutput, HalfCausalEstimatorError> {
    if data.is_empty() {
        return Err(HalfCausalEstimatorError::EmptyInputData);
    }
    if first_finite(data) >= data.len() {
        return Err(HalfCausalEstimatorError::AllValuesNaN);
    }
    let slots_per_day = sweep
        .slots_per_day
        .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
    validate_prepared_slots(slots, data.len(), slots_per_day)?;
    let combos = expand_grid_half_causal_estimator(sweep)?;
    let resolved = resolve_grid_for_frame(&combos, slots_per_day, data, slots)?;
    for params in &resolved {
        validate_frame_public_cpu_retained_budget_v1(*params, data.len(), slots)?;
    }
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HalfCausalEstimatorError::ArithmeticOverflow {
            context: "batch output elements",
        })?;
    let mut estimate_values = try_alloc_f64(total, "batch estimate output")?;
    let mut expected_value_values = try_alloc_f64(total, "batch expected-value output")?;

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        estimate_values
            .par_chunks_mut(cols)
            .zip(expected_value_values.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, (estimate_row, expected_row))| {
                let _ = kernel;
                compute_row(data, slots, resolved[row], estimate_row, expected_row)
            })?;

        #[cfg(target_arch = "wasm32")]
        for (row, (estimate_row, expected_row)) in estimate_values
            .chunks_mut(cols)
            .zip(expected_value_values.chunks_mut(cols))
            .enumerate()
        {
            let _ = kernel;
            compute_row(data, slots, resolved[row], estimate_row, expected_row)?;
        }
    } else {
        for (row, (estimate_row, expected_row)) in estimate_values
            .chunks_mut(cols)
            .zip(expected_value_values.chunks_mut(cols))
            .enumerate()
        {
            let _ = kernel;
            compute_row(data, slots, resolved[row], estimate_row, expected_row)?;
        }
    }

    Ok(HalfCausalEstimatorBatchOutput {
        estimate_values,
        expected_value_values,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn half_causal_estimator_batch_inner_into(
    data: &[f64],
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) -> Result<Vec<HalfCausalEstimatorParams>, HalfCausalEstimatorError> {
    let slots_per_day = sweep
        .slots_per_day
        .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
    let slots = PreparedSlots::Sequential { slots_per_day };
    half_causal_estimator_batch_prepared_into(
        data,
        &slots,
        sweep,
        kernel,
        parallel,
        estimate_out,
        expected_value_out,
    )
}

#[inline]
fn half_causal_estimator_batch_prepared_into(
    data: &[f64],
    slots: &PreparedSlots,
    sweep: &HalfCausalEstimatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    estimate_out: &mut [f64],
    expected_value_out: &mut [f64],
) -> Result<Vec<HalfCausalEstimatorParams>, HalfCausalEstimatorError> {
    if data.is_empty() {
        return Err(HalfCausalEstimatorError::EmptyInputData);
    }
    if first_finite(data) >= data.len() {
        return Err(HalfCausalEstimatorError::AllValuesNaN);
    }
    let slots_per_day = sweep
        .slots_per_day
        .ok_or(HalfCausalEstimatorError::MissingSlotsPerDay)?;
    validate_prepared_slots(slots, data.len(), slots_per_day)?;
    let combos = expand_grid_half_causal_estimator(sweep)?;
    let resolved = resolve_grid_for_frame(&combos, slots_per_day, data, slots)?;
    for params in &resolved {
        validate_frame_public_cpu_retained_budget_v1(*params, data.len(), slots)?;
    }
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HalfCausalEstimatorError::OutputLengthMismatch {
            expected: usize::MAX,
            estimate_got: estimate_out.len(),
            expected_value_got: expected_value_out.len(),
        })?;
    if estimate_out.len() != total || expected_value_out.len() != total {
        return Err(HalfCausalEstimatorError::OutputLengthMismatch {
            expected: total,
            estimate_got: estimate_out.len(),
            expected_value_got: expected_value_out.len(),
        });
    }

    estimate_out.fill(f64::NAN);
    expected_value_out.fill(f64::NAN);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        estimate_out
            .par_chunks_mut(cols)
            .zip(expected_value_out.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, (estimate_row, expected_row))| {
                let _ = kernel;
                compute_row(data, slots, resolved[row], estimate_row, expected_row)
            })?;

        #[cfg(target_arch = "wasm32")]
        for (row, (estimate_row, expected_row)) in estimate_out
            .chunks_mut(cols)
            .zip(expected_value_out.chunks_mut(cols))
            .enumerate()
        {
            let _ = kernel;
            compute_row(data, slots, resolved[row], estimate_row, expected_row)?;
        }
    } else {
        for (row, (estimate_row, expected_row)) in estimate_out
            .chunks_mut(cols)
            .zip(expected_value_out.chunks_mut(cols))
            .enumerate()
        {
            let _ = kernel;
            compute_row(data, slots, resolved[row], estimate_row, expected_row)?;
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::Candles;

    fn sample_source(length: usize, slots_per_day: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(length);
        for i in 0..length {
            let slot = (i % slots_per_day) as f64;
            let day = (i / slots_per_day) as f64;
            out.push(
                1000.0
                    + day * 5.0
                    + (slot * 0.11).sin() * 30.0
                    + (slot * 0.03).cos() * 12.0
                    + (slot / slots_per_day as f64) * 25.0,
            );
        }
        out
    }

    fn sample_candles(days: usize, minutes_per_bar: usize) -> Candles {
        let slots_per_day = 1440 / minutes_per_bar;
        let len = days * slots_per_day;
        let mut timestamp = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        let start_ms = 1_700_000_000_000i64;
        let interval_ms = minutes_per_bar as i64 * 60_000;
        for i in 0..len {
            let x = i as f64;
            timestamp.push(start_ms + i as i64 * interval_ms);
            let c = 100.0 + (x * 0.05).sin() * 2.0 + x * 0.001;
            open.push(c - 0.2);
            high.push(c + 0.6);
            low.push(c - 0.7);
            close.push(c);
            volume.push(1000.0 + (x * 0.07).cos() * 80.0 + x * 0.5);
        }
        Candles::new(timestamp, open, high, low, close, volume)
    }

    #[test]
    fn bounded_bucket_forgets_evicted_rounding_history() {
        let mut bucket = TimeOfDayBucket::try_new(2).unwrap();
        bucket.add(1.0e16);
        bucket.add(1.0);
        bucket.add(1.0);

        assert_eq!(bucket.mean(), Some(1.0));
        assert_eq!(bucket.moments.population_stdev(), Some(0.0));
        assert_eq!(bucket.moments.creator_inverse_cv(1.0), 1.0);
    }

    #[test]
    fn pine_cached_future_window_wrap_l20_slots12_is_frozen() {
        let data = (0..144)
            .map(|row| 100.0 + (row % 17) as f64 * 0.75 + row as f64 * 0.01)
            .collect::<Vec<_>>();
        let params = resolve_params(
            &HalfCausalEstimatorParams {
                slots_per_day: Some(12),
                ..HalfCausalEstimatorParams::default()
            },
            12,
        )
        .unwrap();
        let slots = PreparedSlots::Sequential { slots_per_day: 12 };
        let mut estimate = vec![f64::NAN; data.len()];
        let mut expected = vec![f64::NAN; data.len()];
        compute_row(&data, &slots, params, &mut estimate, &mut expected).unwrap();

        assert_eq!(estimate[49].to_bits(), 0x405a_a4b5_cc6d_006d);
    }

    #[test]
    fn pine_cached_future_window_sparse_slots48_keeps_prior_key() {
        let data = (0..384)
            .map(|row| 700.0 + (row % 31) as f64 * 0.125 + row as f64 * 0.002)
            .collect::<Vec<_>>();
        let sparse_day = [0, 1, 5, 9, 13, 17, 22, 26, 31, 36, 41, 45];
        let slots = (0..data.len())
            .map(|row| sparse_day[row % sparse_day.len()])
            .collect::<Vec<_>>();
        let session_starts = (0..data.len())
            .map(|row| row % sparse_day.len() == 0)
            .collect::<Vec<_>>();
        let prepared_slots = PreparedSlots::Explicit {
            slots,
            session_starts,
        };
        let params = resolve_params(
            &HalfCausalEstimatorParams {
                slots_per_day: Some(48),
                ..HalfCausalEstimatorParams::default()
            },
            48,
        )
        .unwrap();
        let mut estimate = vec![f64::NAN; data.len()];
        let mut expected = vec![f64::NAN; data.len()];
        compute_row(&data, &prepared_slots, params, &mut estimate, &mut expected).unwrap();

        assert_eq!(estimate[49].to_bits(), 0x4085_ec62_d649_f32e);
    }

    #[test]
    fn wma_holes_advance_series_history_and_use_first_finite_fill() {
        let mut wma = FillWmaState::try_new(3).unwrap();
        assert_eq!(wma.update(f64::NAN), None);
        assert_eq!(wma.update(10.0), Some(10.0));
        assert_eq!(wma.update(f64::NAN), None);
        assert_eq!(wma.update(20.0), Some(15.0));
    }

    #[test]
    fn change_source_uses_current_when_previous_close_is_nonfinite() {
        let candles = Candles::new(
            vec![0, 60_000, 120_000],
            vec![10.0, 11.0, 12.0],
            vec![10.0, 11.0, 12.0],
            vec![10.0, 11.0, 12.0],
            vec![10.0, f64::NAN, 12.0],
            vec![1.0, 1.0, 1.0],
        );
        let source = source_from_candles(&candles, "change", 1440).unwrap();

        assert_eq!(source[0], 0.0);
        assert!(source[1].is_nan());
        assert_eq!(source[2], 0.0);
    }

    #[test]
    fn creator_proxy_validation_closes_before_state_allocation() {
        let invalid_slots = HalfCausalEstimatorParams {
            slots_per_day: Some(7),
            ..HalfCausalEstimatorParams::default()
        };
        assert!(matches!(
            resolve_params(&invalid_slots, 7),
            Err(HalfCausalEstimatorError::InvalidSlotsPerDay { slots_per_day: 7 })
        ));

        let unbounded_period = HalfCausalEstimatorParams {
            slots_per_day: Some(12),
            data_period: Some(0),
            ..HalfCausalEstimatorParams::default()
        };
        assert_eq!(
            resolve_params(&unbounded_period, 12).unwrap().data_period,
            0
        );

        let invalid_smoothing = HalfCausalEstimatorParams {
            slots_per_day: Some(12),
            extra_smoothing: Some(usize::MAX),
            ..HalfCausalEstimatorParams::default()
        };
        assert!(matches!(
            resolve_params(&invalid_smoothing, 12),
            Err(HalfCausalEstimatorError::InvalidExtraSmoothing {
                extra_smoothing: usize::MAX
            })
        ));

        let mut candles = sample_candles(2, 60);
        let _ = candles.volume.pop();
        let input = HalfCausalEstimatorInput::from_candles(
            &candles,
            "volume",
            HalfCausalEstimatorParams::default(),
        );
        assert!(matches!(
            resolve_and_prepare(&input),
            Err(HalfCausalEstimatorError::CandleFieldLengthMismatch {
                field: "volume",
                ..
            })
        ));
    }

    #[test]
    fn authoritative_volume_fixture_freezes_row_849() {
        let data = (0..4096)
            .map(|row| 900.0 + (row % 97) as f64 * 3.25 + row as f64 * 0.001)
            .collect::<Vec<_>>();
        let input = HalfCausalEstimatorInput::from_slice(
            &data,
            HalfCausalEstimatorParams {
                slots_per_day: Some(288),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let out = half_causal_estimator(&input).unwrap();

        assert_eq!(out.estimate[849].to_bits(), 0x4091_ca58_b879_8573);
    }

    #[test]
    fn half_causal_estimator_output_contract() {
        let slots_per_day = 60;
        let data = sample_source(slots_per_day * 4, slots_per_day);
        let input = HalfCausalEstimatorInput::from_slice(
            &data,
            HalfCausalEstimatorParams {
                slots_per_day: Some(slots_per_day),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let out = half_causal_estimator(&input).unwrap();
        assert_eq!(out.estimate.len(), data.len());
        assert_eq!(out.expected_value.len(), data.len());
        assert!(out.estimate.iter().any(|value| value.is_finite()));
        assert!(out.expected_value.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn half_causal_estimator_stream_matches_batch() {
        let slots_per_day = 48;
        let data = sample_source(slots_per_day * 5, slots_per_day);
        let params = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            enable_expected_value: Some(true),
            extra_smoothing: Some(2),
            ..HalfCausalEstimatorParams::default()
        };
        let input = HalfCausalEstimatorInput::from_slice(&data, params.clone());
        let out = half_causal_estimator(&input).unwrap();
        let mut stream = HalfCausalEstimatorStream::try_new(params).unwrap();
        let mut est = Vec::with_capacity(data.len());
        let mut exp = Vec::with_capacity(data.len());
        for value in data {
            let (estimate, expected_value) = stream.update(value);
            est.push(estimate.unwrap_or(f64::NAN));
            exp.push(expected_value.unwrap_or(f64::NAN));
        }
        for i in 0..est.len() {
            if out.estimate[i].is_nan() {
                assert!(est[i].is_nan());
            } else {
                assert_eq!(est[i].to_bits(), out.estimate[i].to_bits());
            }
            if out.expected_value[i].is_nan() {
                assert!(exp[i].is_nan());
            } else {
                assert_eq!(exp[i].to_bits(), out.expected_value[i].to_bits());
            }
        }
    }

    #[test]
    fn unbounded_data_period_direct_batch_stream_match_through_holes() {
        let slots_per_day = 12;
        let mut data = sample_source(slots_per_day * 14, slots_per_day);
        for index in [7, 28, 47, 74, 99, 121, 146] {
            data[index] = f64::NAN;
        }
        let params = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            data_period: Some(0),
            ..HalfCausalEstimatorParams::default()
        };
        let direct =
            half_causal_estimator(&HalfCausalEstimatorInput::from_slice(&data, params.clone()))
                .unwrap();
        let batch = half_causal_estimator_batch_with_kernel(
            &data,
            &HalfCausalEstimatorBatchRange {
                slots_per_day: Some(slots_per_day),
                data_period: (0, 0, 0),
                ..HalfCausalEstimatorBatchRange::default()
            },
            Kernel::Auto,
        )
        .unwrap();
        let mut stream = HalfCausalEstimatorStream::try_new(params).unwrap();
        let streamed = data
            .iter()
            .map(|value| stream.update(*value).0.unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        let batch_row = batch.estimate_for(0).unwrap();
        for index in 0..data.len() {
            if direct.estimate[index].is_nan() {
                assert!(batch_row[index].is_nan());
                assert!(streamed[index].is_nan());
            } else {
                assert_eq!(batch_row[index].to_bits(), direct.estimate[index].to_bits());
                assert_eq!(streamed[index].to_bits(), direct.estimate[index].to_bits());
            }
        }
        assert!(
            direct
                .estimate
                .iter()
                .skip(slots_per_day * 6)
                .any(|value| value.is_finite())
        );
    }

    #[test]
    fn finite_frame_effective_data_period_avoids_a_false_public_d_cap() {
        let slots_per_day = 12;
        let data = sample_source(slots_per_day * 8, slots_per_day);
        let output = half_causal_estimator(&HalfCausalEstimatorInput::from_slice(
            &data,
            HalfCausalEstimatorParams {
                slots_per_day: Some(slots_per_day),
                data_period: Some(usize::MAX),
                ..HalfCausalEstimatorParams::default()
            },
        ))
        .unwrap();
        assert!(output.estimate.iter().any(|value| value.is_finite()));
    }

    #[test]
    fn registry_anchor_21_is_not_base_20_at_the_creator_readiness_boundary() {
        let slots_per_day = 40;
        let data = sample_source(3 * slots_per_day, slots_per_day);
        let run = |filter_length| {
            half_causal_estimator(&HalfCausalEstimatorInput::from_slice(
                &data,
                HalfCausalEstimatorParams {
                    slots_per_day: Some(slots_per_day),
                    data_period: Some(5),
                    filter_length: Some(filter_length),
                    ..HalfCausalEstimatorParams::default()
                },
            ))
            .unwrap()
        };
        let length_20 = run(20);
        let length_21 = run(21);

        assert!(length_20.estimate[40].is_finite());
        assert!(length_21.estimate[40].is_nan());
        assert!(
            length_21.estimate[40..80]
                .iter()
                .all(|value| value.is_nan())
        );
        assert!(length_21.estimate[80].is_finite());
    }

    #[test]
    fn huge_public_contexts_fail_typed_or_short_frame_skips_allocation() {
        let slots_per_day = 2;
        let data = sample_source(120, slots_per_day);
        let huge_data_period = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            data_period: Some(1_000_000_000),
            ..HalfCausalEstimatorParams::default()
        };
        let direct = half_causal_estimator(&HalfCausalEstimatorInput::from_slice(
            &data,
            huge_data_period.clone(),
        ))
        .unwrap();
        assert!(direct.estimate.iter().any(|value| value.is_finite()));
        assert!(matches!(
            HalfCausalEstimatorStream::try_new(huge_data_period),
            Err(
                HalfCausalEstimatorError::PublicRetainedMemoryBudgetExceeded {
                    budget_bytes: HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1,
                    ..
                }
            )
        ));

        let huge_nonoverflow_filter = HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1;
        let huge_filter_params = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            filter_length: Some(huge_nonoverflow_filter),
            ..HalfCausalEstimatorParams::default()
        };
        let short_data = &data[..8];
        let direct = half_causal_estimator(&HalfCausalEstimatorInput::from_slice(
            short_data,
            huge_filter_params.clone(),
        ))
        .unwrap();
        assert!(direct.estimate.iter().all(|value| value.is_nan()));
        let batch = half_causal_estimator_batch_with_kernel(
            short_data,
            &HalfCausalEstimatorBatchRange {
                slots_per_day: Some(slots_per_day),
                filter_length: (huge_nonoverflow_filter, huge_nonoverflow_filter, 0),
                ..HalfCausalEstimatorBatchRange::default()
            },
            Kernel::Auto,
        )
        .unwrap();
        assert!(batch.estimate_values.iter().all(|value| value.is_nan()));
        let mut estimate_into = vec![123.0; short_data.len()];
        let mut expected_into = vec![456.0; short_data.len()];
        half_causal_estimator_into_slices(
            &mut estimate_into,
            &mut expected_into,
            &HalfCausalEstimatorInput::from_slice(short_data, huge_filter_params.clone()),
            Kernel::Scalar,
        )
        .unwrap();
        assert!(estimate_into.iter().all(|value| value.is_nan()));
        assert!(expected_into.iter().all(|value| value.is_nan()));
        assert!(matches!(
            HalfCausalEstimatorStream::try_new(huge_filter_params),
            Err(
                HalfCausalEstimatorError::PublicRetainedMemoryBudgetExceeded {
                    budget_bytes: HALF_CAUSAL_ESTIMATOR_PUBLIC_CPU_RETAINED_BUDGET_BYTES_V1,
                    ..
                }
            )
        ));
    }

    #[test]
    fn hostile_window_and_sweep_shapes_fail_typed_before_allocation() {
        let sinc_overflow = HalfCausalEstimatorParams {
            slots_per_day: Some(12),
            filter_length: Some(usize::MAX),
            kernel_type: Some(HalfCausalEstimatorKernelType::Sinc),
            ..HalfCausalEstimatorParams::default()
        };
        assert!(matches!(
            resolve_params(&sinc_overflow, 12),
            Err(HalfCausalEstimatorError::ArithmeticOverflow {
                context: "Sinc real_filter_length"
            })
        ));
        let window_overflow = HalfCausalEstimatorParams {
            slots_per_day: Some(12),
            filter_length: Some(usize::MAX),
            ..HalfCausalEstimatorParams::default()
        };
        assert!(matches!(
            resolve_params(&window_overflow, 12),
            Err(HalfCausalEstimatorError::ArithmeticOverflow {
                context: "half-causal window_size"
            })
        ));
        let huge_sweep = HalfCausalEstimatorBatchRange {
            slots_per_day: Some(12),
            data_period: (0, usize::MAX, 1),
            ..HalfCausalEstimatorBatchRange::default()
        };
        assert!(matches!(
            checked_sweep_cardinality(&huge_sweep),
            Err(HalfCausalEstimatorError::SweepCardinalityOverflow)
        ));
        assert!(matches!(
            axis_len_f64((0.0, 18_446_744_073_709_551_616.0, 1.0)),
            Err(HalfCausalEstimatorError::SweepCardinalityOverflow)
        ));
    }

    #[test]
    fn half_causal_estimator_batch_single_matches_direct() {
        let slots_per_day = 60;
        let data = sample_source(slots_per_day * 4, slots_per_day);
        let params = HalfCausalEstimatorParams {
            slots_per_day: Some(slots_per_day),
            enable_expected_value: Some(true),
            ..HalfCausalEstimatorParams::default()
        };
        let input = HalfCausalEstimatorInput::from_slice(&data, params.clone());
        let direct = half_causal_estimator(&input).unwrap();
        let batch = half_causal_estimator_batch_with_kernel(
            &data,
            &HalfCausalEstimatorBatchRange {
                slots_per_day: Some(slots_per_day),
                enable_expected_value: true,
                ..HalfCausalEstimatorBatchRange::default()
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for (&lhs, &rhs) in batch
            .estimate_for(0)
            .unwrap()
            .iter()
            .zip(direct.estimate.iter())
        {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert_eq!(lhs.to_bits(), rhs.to_bits());
            }
        }
        for (&lhs, &rhs) in batch
            .expected_value_for(0)
            .unwrap()
            .iter()
            .zip(direct.expected_value.iter())
        {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert_eq!(lhs.to_bits(), rhs.to_bits());
            }
        }
    }

    #[test]
    fn half_causal_estimator_candle_test_source_runs() {
        let candles = sample_candles(4, 30);
        let input = HalfCausalEstimatorInput::from_candles(
            &candles,
            "test",
            HalfCausalEstimatorParams {
                enable_expected_value: Some(true),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let out = half_causal_estimator(&input).unwrap();
        assert_eq!(out.estimate.len(), candles.close.len());
        assert!(out.estimate.iter().any(|value| value.is_finite()));
        assert!(out.expected_value.iter().any(|value| value.is_finite()));
    }

    #[test]
    fn half_causal_estimator_rejects_invalid_params() {
        let data = sample_source(128, 32);
        let input = HalfCausalEstimatorInput::from_slice(
            &data,
            HalfCausalEstimatorParams {
                slots_per_day: Some(1),
                ..HalfCausalEstimatorParams::default()
            },
        );
        let err = half_causal_estimator(&input).unwrap_err();
        assert!(matches!(
            err,
            HalfCausalEstimatorError::InvalidSlotsPerDay { .. }
        ));
    }
}
