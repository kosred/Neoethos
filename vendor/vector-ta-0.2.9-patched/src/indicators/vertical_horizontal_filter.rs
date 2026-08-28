use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 28;
const DEFAULT_SOURCE: &str = "close";

impl<'a> AsRef<[f64]> for VerticalHorizontalFilterInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VerticalHorizontalFilterData::Slice(slice) => slice,
            VerticalHorizontalFilterData::Candles { candles, source } => match *source {
                DEFAULT_SOURCE => &candles.close,
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum VerticalHorizontalFilterData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VerticalHorizontalFilterOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VerticalHorizontalFilterParams {
    pub length: Option<usize>,
}

impl Default for VerticalHorizontalFilterParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerticalHorizontalFilterInput<'a> {
    pub data: VerticalHorizontalFilterData<'a>,
    pub params: VerticalHorizontalFilterParams,
}

impl<'a> VerticalHorizontalFilterInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: VerticalHorizontalFilterParams,
    ) -> Self {
        Self {
            data: VerticalHorizontalFilterData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: VerticalHorizontalFilterParams) -> Self {
        Self {
            data: VerticalHorizontalFilterData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            VerticalHorizontalFilterParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VerticalHorizontalFilterBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for VerticalHorizontalFilterBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VerticalHorizontalFilterBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
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
        source: &str,
    ) -> Result<VerticalHorizontalFilterOutput, VerticalHorizontalFilterError> {
        let input = VerticalHorizontalFilterInput::from_candles(
            candles,
            source,
            VerticalHorizontalFilterParams {
                length: self.length,
            },
        );
        vertical_horizontal_filter_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<VerticalHorizontalFilterOutput, VerticalHorizontalFilterError> {
        let input = VerticalHorizontalFilterInput::from_slice(
            data,
            VerticalHorizontalFilterParams {
                length: self.length,
            },
        );
        vertical_horizontal_filter_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<VerticalHorizontalFilterStream, VerticalHorizontalFilterError> {
        VerticalHorizontalFilterStream::try_new(VerticalHorizontalFilterParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum VerticalHorizontalFilterError {
    #[error("vertical_horizontal_filter: Input data slice is empty.")]
    EmptyInputData,
    #[error("vertical_horizontal_filter: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "vertical_horizontal_filter: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "vertical_horizontal_filter: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "vertical_horizontal_filter: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vertical_horizontal_filter: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("vertical_horizontal_filter: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct VerticalHorizontalFilterStream {
    length: usize,
    started: bool,
    seen: usize,
    prev: f64,
    dq_max: VecDeque<(usize, f64)>,
    dq_min: VecDeque<(usize, f64)>,
    src_valid: Vec<u8>,
    src_valid_idx: usize,
    src_valid_cnt: usize,
    src_total_cnt: usize,
    change_valid: Vec<u8>,
    change_vals: Vec<f64>,
    change_idx: usize,
    change_valid_cnt: usize,
    change_total_cnt: usize,
    change_sum: f64,
}

impl VerticalHorizontalFilterStream {
    pub fn try_new(
        params: VerticalHorizontalFilterParams,
    ) -> Result<VerticalHorizontalFilterStream, VerticalHorizontalFilterError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length == 0 {
            return Err(VerticalHorizontalFilterError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self {
            length,
            started: false,
            seen: 0,
            prev: f64::NAN,
            dq_max: VecDeque::with_capacity(length + 1),
            dq_min: VecDeque::with_capacity(length + 1),
            src_valid: vec![0u8; length],
            src_valid_idx: 0,
            src_valid_cnt: 0,
            src_total_cnt: 0,
            change_valid: vec![0u8; length],
            change_vals: vec![0.0; length],
            change_idx: 0,
            change_valid_cnt: 0,
            change_total_cnt: 0,
            change_sum: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let is_valid = valid_value(value);
        if !self.started {
            if !is_valid {
                return None;
            }
            self.started = true;
        }

        let i = self.seen;

        if self.src_total_cnt >= self.length {
            let old = self.src_valid[self.src_valid_idx];
            if old != 0 {
                self.src_valid_cnt = self.src_valid_cnt.saturating_sub(1);
            }
        } else {
            self.src_total_cnt += 1;
        }

        if is_valid {
            while let Some(&(_, back)) = self.dq_max.back() {
                if back <= value {
                    self.dq_max.pop_back();
                } else {
                    break;
                }
            }
            self.dq_max.push_back((i, value));

            while let Some(&(_, back)) = self.dq_min.back() {
                if back >= value {
                    self.dq_min.pop_back();
                } else {
                    break;
                }
            }
            self.dq_min.push_back((i, value));

            self.src_valid[self.src_valid_idx] = 1;
            self.src_valid_cnt += 1;
        } else {
            self.src_valid[self.src_valid_idx] = 0;
        }

        self.src_valid_idx += 1;
        if self.src_valid_idx == self.length {
            self.src_valid_idx = 0;
        }

        let start = i.saturating_add(1).saturating_sub(self.length);
        while let Some(&(idx, _)) = self.dq_max.front() {
            if idx < start {
                self.dq_max.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(idx, _)) = self.dq_min.front() {
            if idx < start {
                self.dq_min.pop_front();
            } else {
                break;
            }
        }

        if self.seen > 0 {
            if self.change_total_cnt >= self.length {
                if self.change_valid[self.change_idx] != 0 {
                    self.change_valid_cnt = self.change_valid_cnt.saturating_sub(1);
                    self.change_sum -= self.change_vals[self.change_idx];
                }
            } else {
                self.change_total_cnt += 1;
            }

            if is_valid && self.prev.is_finite() {
                let diff = (value - self.prev).abs();
                self.change_valid[self.change_idx] = 1;
                self.change_vals[self.change_idx] = diff;
                self.change_valid_cnt += 1;
                self.change_sum += diff;
            } else {
                self.change_valid[self.change_idx] = 0;
                self.change_vals[self.change_idx] = 0.0;
            }

            self.change_idx += 1;
            if self.change_idx == self.length {
                self.change_idx = 0;
            }
        }

        self.prev = value;
        self.seen = i + 1;

        if self.src_total_cnt < self.length || self.change_total_cnt < self.length {
            return None;
        }
        if self.src_valid_cnt != self.length || self.change_valid_cnt != self.length {
            return Some(f64::NAN);
        }

        let highest = self
            .dq_max
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::NEG_INFINITY);
        let lowest = self
            .dq_min
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::INFINITY);
        Some(vhf_value(highest, lowest, self.change_sum))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.length
    }
}

#[inline]
pub fn vertical_horizontal_filter(
    input: &VerticalHorizontalFilterInput,
) -> Result<VerticalHorizontalFilterOutput, VerticalHorizontalFilterError> {
    vertical_horizontal_filter_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_value(value: f64) -> bool {
    value.is_finite()
}

#[inline(always)]
fn valid_change(prev: f64, current: f64) -> bool {
    prev.is_finite() && current.is_finite()
}

#[inline(always)]
fn vhf_value(highest: f64, lowest: f64, denom: f64) -> f64 {
    if highest.is_finite() && lowest.is_finite() && denom.is_finite() && denom > 0.0 {
        (highest - lowest).abs() / denom
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if valid_value(data[i]) {
            break;
        }
        i += 1;
    }
    i.min(data.len())
}

#[inline(always)]
fn scan_validity(data: &[f64]) -> (usize, usize, usize, bool) {
    let len = data.len();
    let mut first_source = len;
    let mut first_change = len;
    let mut valid_changes = 0usize;
    let mut all_valid = true;

    for i in 0..len {
        let is_valid = valid_value(data[i]);
        if is_valid {
            if first_source == len {
                first_source = i;
            }
        } else {
            all_valid = false;
        }

        if i > 0 && valid_change(data[i - 1], data[i]) {
            if first_change == len {
                first_change = i;
            }
            valid_changes += 1;
        }
    }

    (first_source, first_change, valid_changes, all_valid)
}

#[inline(always)]
fn first_valid_change(data: &[f64]) -> usize {
    if data.len() < 2 {
        return data.len();
    }
    let mut i = 1usize;
    while i < data.len() {
        if valid_change(data[i - 1], data[i]) {
            break;
        }
        i += 1;
    }
    i.min(data.len())
}

#[inline(always)]
fn count_valid_changes(data: &[f64]) -> usize {
    if data.len() < 2 {
        return 0;
    }
    let mut count = 0usize;
    for i in 1..data.len() {
        if valid_change(data[i - 1], data[i]) {
            count += 1;
        }
    }
    count
}

#[inline(always)]
fn build_prefixes(data: &[f64]) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let len = data.len();
    let mut prefix_source = vec![0u32; len + 1];
    let mut prefix_change_valid = vec![0u32; len + 1];
    let mut prefix_change_sum = vec![0.0; len + 1];

    for i in 0..len {
        prefix_source[i + 1] = prefix_source[i] + if valid_value(data[i]) { 1 } else { 0 };
        prefix_change_valid[i + 1] = prefix_change_valid[i];
        prefix_change_sum[i + 1] = prefix_change_sum[i];
        if i > 0 && valid_change(data[i - 1], data[i]) {
            prefix_change_valid[i + 1] += 1;
            prefix_change_sum[i + 1] += (data[i] - data[i - 1]).abs();
        }
    }

    (prefix_source, prefix_change_valid, prefix_change_sum)
}

#[inline(always)]
fn build_change_sum_prefix_all_finite(data: &[f64]) -> Vec<f64> {
    let len = data.len();
    let mut prefix_sum = vec![0.0; len + 1];
    for i in 0..len {
        prefix_sum[i + 1] = prefix_sum[i];
        if i > 0 {
            prefix_sum[i + 1] += (data[i] - data[i - 1]).abs();
        }
    }
    prefix_sum
}

#[inline(always)]
fn vhf_row_from_slice(
    data: &[f64],
    prefix_source: &[u32],
    prefix_change_valid: &[u32],
    prefix_change_sum: &[f64],
    length: usize,
    first_source: usize,
    first_change: usize,
    out: &mut [f64],
) {
    out.fill(f64::NAN);

    let warmup = first_change.saturating_add(length.saturating_sub(1));
    let length_u32 = length as u32;
    let mut dq_max = VecDeque::<usize>::with_capacity(length + 1);
    let mut dq_min = VecDeque::<usize>::with_capacity(length + 1);

    for i in first_source..data.len() {
        let value = data[i];
        if valid_value(value) {
            while let Some(&j) = dq_max.back() {
                if data[j] <= value {
                    dq_max.pop_back();
                } else {
                    break;
                }
            }
            dq_max.push_back(i);

            while let Some(&j) = dq_min.back() {
                if data[j] >= value {
                    dq_min.pop_back();
                } else {
                    break;
                }
            }
            dq_min.push_back(i);
        }

        if i < warmup || i + 1 < length {
            continue;
        }

        let start = i + 1 - length;
        while let Some(&j) = dq_max.front() {
            if j < start {
                dq_max.pop_front();
            } else {
                break;
            }
        }
        while let Some(&j) = dq_min.front() {
            if j < start {
                dq_min.pop_front();
            } else {
                break;
            }
        }

        let source_count = prefix_source[i + 1] - prefix_source[start];
        if source_count != length_u32 {
            continue;
        }

        let change_count = prefix_change_valid[i + 1] - prefix_change_valid[start];
        if change_count != length_u32 {
            continue;
        }

        let denom = prefix_change_sum[i + 1] - prefix_change_sum[start];
        let highest = dq_max
            .front()
            .map(|&j| data[j])
            .unwrap_or(f64::NEG_INFINITY);
        let lowest = dq_min.front().map(|&j| data[j]).unwrap_or(f64::INFINITY);
        out[i] = vhf_value(highest, lowest, denom);
    }
}

#[inline(always)]
fn vhf_row_all_finite(data: &[f64], prefix_change_sum: &[f64], length: usize, out: &mut [f64]) {
    out.fill(f64::NAN);

    let mut dq_max = VecDeque::<usize>::with_capacity(length + 1);
    let mut dq_min = VecDeque::<usize>::with_capacity(length + 1);

    for i in 0..data.len() {
        let value = data[i];

        while let Some(&j) = dq_max.back() {
            if data[j] <= value {
                dq_max.pop_back();
            } else {
                break;
            }
        }
        dq_max.push_back(i);

        while let Some(&j) = dq_min.back() {
            if data[j] >= value {
                dq_min.pop_back();
            } else {
                break;
            }
        }
        dq_min.push_back(i);

        if i < length {
            continue;
        }

        let start = i + 1 - length;
        while let Some(&j) = dq_max.front() {
            if j < start {
                dq_max.pop_front();
            } else {
                break;
            }
        }
        while let Some(&j) = dq_min.front() {
            if j < start {
                dq_min.pop_front();
            } else {
                break;
            }
        }

        let highest = dq_max
            .front()
            .map(|&j| data[j])
            .unwrap_or(f64::NEG_INFINITY);
        let lowest = dq_min.front().map(|&j| data[j]).unwrap_or(f64::INFINITY);
        let denom = prefix_change_sum[i + 1] - prefix_change_sum[start];
        out[i] = vhf_value(highest, lowest, denom);
    }
}

#[inline(always)]
fn vertical_horizontal_filter_prepare<'a>(
    input: &'a VerticalHorizontalFilterInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, bool, Kernel), VerticalHorizontalFilterError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(VerticalHorizontalFilterError::EmptyInputData);
    }

    let (first_source, first_change, valid, all_valid) = scan_validity(data);
    if first_source >= len {
        return Err(VerticalHorizontalFilterError::AllValuesNaN);
    }

    let length = input.get_length();
    if length == 0 || length > len {
        return Err(VerticalHorizontalFilterError::InvalidLength {
            length,
            data_len: len,
        });
    }

    if valid < length {
        return Err(VerticalHorizontalFilterError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };

    Ok((data, length, first_source, first_change, all_valid, chosen))
}

#[inline]
pub fn vertical_horizontal_filter_with_kernel(
    input: &VerticalHorizontalFilterInput,
    kernel: Kernel,
) -> Result<VerticalHorizontalFilterOutput, VerticalHorizontalFilterError> {
    let (data, length, first_source, first_change, all_valid, _chosen) =
        vertical_horizontal_filter_prepare(input, kernel)?;
    let mut values = alloc_uninit_f64(data.len());
    if all_valid {
        let prefix_change_sum = build_change_sum_prefix_all_finite(data);
        vhf_row_all_finite(data, &prefix_change_sum, length, &mut values);
    } else {
        let (prefix_source, prefix_change_valid, prefix_change_sum) = build_prefixes(data);
        vhf_row_from_slice(
            data,
            &prefix_source,
            &prefix_change_valid,
            &prefix_change_sum,
            length,
            first_source,
            first_change,
            &mut values,
        );
    }
    Ok(VerticalHorizontalFilterOutput { values })
}

#[inline]
pub fn vertical_horizontal_filter_into_slice(
    dst: &mut [f64],
    input: &VerticalHorizontalFilterInput,
    kernel: Kernel,
) -> Result<(), VerticalHorizontalFilterError> {
    let (data, length, first_source, first_change, all_valid, _chosen) =
        vertical_horizontal_filter_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(VerticalHorizontalFilterError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    if all_valid {
        let prefix_change_sum = build_change_sum_prefix_all_finite(data);
        vhf_row_all_finite(data, &prefix_change_sum, length, dst);
    } else {
        let (prefix_source, prefix_change_valid, prefix_change_sum) = build_prefixes(data);
        vhf_row_from_slice(
            data,
            &prefix_source,
            &prefix_change_valid,
            &prefix_change_sum,
            length,
            first_source,
            first_change,
            dst,
        );
    }
    Ok(())
}

#[inline]
pub fn vertical_horizontal_filter_into(
    input: &VerticalHorizontalFilterInput,
    out: &mut [f64],
) -> Result<(), VerticalHorizontalFilterError> {
    vertical_horizontal_filter_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct VerticalHorizontalFilterBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for VerticalHorizontalFilterBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, 252, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VerticalHorizontalFilterBatchBuilder {
    range: VerticalHorizontalFilterBatchRange,
    kernel: Kernel,
}

impl VerticalHorizontalFilterBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
        vertical_horizontal_filter_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
        self.apply_slice(source_type(candles, source))
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
        VerticalHorizontalFilterBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles, "close")
    }
}

#[derive(Clone, Debug)]
pub struct VerticalHorizontalFilterBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VerticalHorizontalFilterParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VerticalHorizontalFilterBatchOutput {
    pub fn row_for_params(&self, params: &VerticalHorizontalFilterParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == params.length.unwrap_or(DEFAULT_LENGTH)
        })
    }

    pub fn values_for(&self, params: &VerticalHorizontalFilterParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

fn expand_grid_vertical_horizontal_filter(
    range: &VerticalHorizontalFilterBatchRange,
) -> Result<Vec<VerticalHorizontalFilterParams>, VerticalHorizontalFilterError> {
    let (start, end, step) = range.length;
    let lengths = if start == end {
        vec![start]
    } else {
        if step == 0 {
            return Err(VerticalHorizontalFilterError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut out = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                out.push(x);
                let next = x.saturating_add(step);
                if next == x {
                    break;
                }
                x = next;
            }
        } else {
            let mut x = start;
            while x >= end {
                out.push(x);
                let next = x.saturating_sub(step);
                if next == x {
                    break;
                }
                x = next;
            }
        }
        if out.is_empty() {
            return Err(VerticalHorizontalFilterError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        out
    };

    if let Some(&bad) = lengths.iter().find(|&&length| length == 0) {
        return Err(VerticalHorizontalFilterError::InvalidLength {
            length: bad,
            data_len: 0,
        });
    }

    Ok(lengths
        .into_iter()
        .map(|length| VerticalHorizontalFilterParams {
            length: Some(length),
        })
        .collect())
}

#[inline]
pub fn vertical_horizontal_filter_batch_with_kernel(
    data: &[f64],
    sweep: &VerticalHorizontalFilterBatchRange,
    kernel: Kernel,
) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VerticalHorizontalFilterError::InvalidKernelForBatch(other)),
    };
    vertical_horizontal_filter_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn vertical_horizontal_filter_batch_slice(
    data: &[f64],
    sweep: &VerticalHorizontalFilterBatchRange,
    kernel: Kernel,
) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
    vertical_horizontal_filter_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn vertical_horizontal_filter_batch_par_slice(
    data: &[f64],
    sweep: &VerticalHorizontalFilterBatchRange,
    kernel: Kernel,
) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
    vertical_horizontal_filter_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn vertical_horizontal_filter_batch_inner(
    data: &[f64],
    sweep: &VerticalHorizontalFilterBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterError> {
    let combos = expand_grid_vertical_horizontal_filter(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(VerticalHorizontalFilterError::EmptyInputData);
    }

    let first_source = first_valid_value(data);
    if first_source >= cols {
        return Err(VerticalHorizontalFilterError::AllValuesNaN);
    }
    let valid = count_valid_changes(data);
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(0);
    if valid < max_length {
        return Err(VerticalHorizontalFilterError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }
    let first_change = first_valid_change(data);
    let (prefix_source, prefix_change_valid, prefix_change_sum) = build_prefixes(data);

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first_change.saturating_add(combo.length.unwrap_or(DEFAULT_LENGTH).saturating_sub(1))
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                vhf_row_from_slice(
                    data,
                    &prefix_source,
                    &prefix_change_valid,
                    &prefix_change_sum,
                    combo.length.unwrap_or(DEFAULT_LENGTH),
                    first_source,
                    first_change,
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            vhf_row_from_slice(
                data,
                &prefix_source,
                &prefix_change_valid,
                &prefix_change_sum,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                first_source,
                first_change,
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            vhf_row_from_slice(
                data,
                &prefix_source,
                &prefix_change_valid,
                &prefix_change_sum,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                first_source,
                first_change,
                out_row,
            );
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VerticalHorizontalFilterBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn vertical_horizontal_filter_batch_inner_into(
    data: &[f64],
    sweep: &VerticalHorizontalFilterBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VerticalHorizontalFilterParams>, VerticalHorizontalFilterError> {
    let combos = expand_grid_vertical_horizontal_filter(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(VerticalHorizontalFilterError::EmptyInputData);
    }
    let total = rows.checked_mul(cols).ok_or_else(|| {
        VerticalHorizontalFilterError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        }
    })?;
    if out.len() != total {
        return Err(VerticalHorizontalFilterError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let first_source = first_valid_value(data);
    if first_source >= cols {
        return Err(VerticalHorizontalFilterError::AllValuesNaN);
    }
    let valid = count_valid_changes(data);
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(0);
    if valid < max_length {
        return Err(VerticalHorizontalFilterError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let first_change = first_valid_change(data);
    let (prefix_source, prefix_change_valid, prefix_change_sum) = build_prefixes(data);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                vhf_row_from_slice(
                    data,
                    &prefix_source,
                    &prefix_change_valid,
                    &prefix_change_sum,
                    combo.length.unwrap_or(DEFAULT_LENGTH),
                    first_source,
                    first_change,
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            vhf_row_from_slice(
                data,
                &prefix_source,
                &prefix_change_valid,
                &prefix_change_sum,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                first_source,
                first_change,
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            vhf_row_from_slice(
                data,
                &prefix_source,
                &prefix_change_valid,
                &prefix_change_sum,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                first_source,
                first_change,
                out_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use std::error::Error;

    fn load_close() -> Result<Vec<f64>, Box<dyn Error>> {
        let candles = read_candles_from_vortex("src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex")?;
        Ok(candles.close.clone())
    }

    #[test]
    fn vertical_horizontal_filter_output_contract() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = VerticalHorizontalFilterInput::from_slice(
            &close,
            VerticalHorizontalFilterParams { length: Some(28) },
        );
        let out = vertical_horizontal_filter_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values.len(), close.len());
        let first_valid = out.values.iter().position(|v| !v.is_nan()).unwrap();
        assert!(first_valid >= 28);
        assert!(out.values[first_valid..].iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn vertical_horizontal_filter_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = VerticalHorizontalFilterInput::from_slice(
            &close,
            VerticalHorizontalFilterParams { length: Some(21) },
        );
        let auto = vertical_horizontal_filter_with_kernel(&input, Kernel::Auto)?;
        let scalar = vertical_horizontal_filter_with_kernel(&input, Kernel::Scalar)?;
        for (a, b) in auto.values.iter().zip(scalar.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn vertical_horizontal_filter_into_matches_api() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = VerticalHorizontalFilterInput::from_slice(
            &close,
            VerticalHorizontalFilterParams { length: Some(18) },
        );
        let api = vertical_horizontal_filter_with_kernel(&input, Kernel::Auto)?;
        let mut out = vec![0.0; close.len()];
        vertical_horizontal_filter_into(&input, &mut out)?;
        for (a, b) in api.values.iter().zip(out.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() <= 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn vertical_horizontal_filter_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let params = VerticalHorizontalFilterParams { length: Some(16) };
        let input = VerticalHorizontalFilterInput::from_slice(&close, params.clone());
        let batch = vertical_horizontal_filter_with_kernel(&input, Kernel::Scalar)?;
        let mut stream = VerticalHorizontalFilterStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(close.len());
        for &value in &close {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in streamed.iter().zip(batch.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn vertical_horizontal_filter_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let single = vertical_horizontal_filter_with_kernel(
            &VerticalHorizontalFilterInput::from_slice(
                &close,
                VerticalHorizontalFilterParams { length: Some(28) },
            ),
            Kernel::Scalar,
        )?;
        let batch = vertical_horizontal_filter_batch_with_kernel(
            &close,
            &VerticalHorizontalFilterBatchRange {
                length: (28, 28, 0),
            },
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn vertical_horizontal_filter_invalid_window_recovers() -> Result<(), Box<dyn Error>> {
        let mut close = load_close()?;
        close.truncate(96);
        close[30] = f64::NAN;
        let out = vertical_horizontal_filter_with_kernel(
            &VerticalHorizontalFilterInput::from_slice(
                &close,
                VerticalHorizontalFilterParams { length: Some(10) },
            ),
            Kernel::Scalar,
        )?;
        assert!(out.values[30].is_nan());
        assert!(out.values[40].is_nan());
        assert!(out.values[41].is_finite());
        Ok(())
    }

    #[test]
    fn vertical_horizontal_filter_rejects_invalid_length() {
        let data = [1.0, 2.0, 3.0];
        let err = vertical_horizontal_filter_with_kernel(
            &VerticalHorizontalFilterInput::from_slice(
                &data,
                VerticalHorizontalFilterParams { length: Some(0) },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            VerticalHorizontalFilterError::InvalidLength { .. }
        ));
    }
}
