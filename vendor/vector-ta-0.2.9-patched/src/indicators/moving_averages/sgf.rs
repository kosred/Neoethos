use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use std::sync::OnceLock;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SgfInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SgfData::Slice(slice) => slice,
            SgfData::Candles { candles, source } => match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
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
pub enum SgfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SgfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SgfParams {
    pub period: Option<usize>,
    pub poly_order: Option<usize>,
}

impl Default for SgfParams {
    fn default() -> Self {
        Self {
            period: Some(21),
            poly_order: Some(2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SgfInput<'a> {
    pub data: SgfData<'a>,
    pub params: SgfParams,
}

impl<'a> SgfInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SgfParams) -> Self {
        Self {
            data: SgfData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: SgfParams) -> Self {
        Self {
            data: SgfData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", SgfParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(21)
    }

    #[inline]
    pub fn get_poly_order(&self) -> usize {
        self.params.poly_order.unwrap_or(2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SgfBuilder {
    period: Option<usize>,
    poly_order: Option<usize>,
    kernel: Kernel,
}

impl Default for SgfBuilder {
    fn default() -> Self {
        Self {
            period: None,
            poly_order: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SgfBuilder {
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
    pub fn poly_order(mut self, n: usize) -> Self {
        self.poly_order = Some(n);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<SgfOutput, SgfError> {
        let input = SgfInput::from_candles(
            candles,
            "close",
            SgfParams {
                period: self.period,
                poly_order: self.poly_order,
            },
        );
        sgf_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<SgfOutput, SgfError> {
        let input = SgfInput::from_slice(
            data,
            SgfParams {
                period: self.period,
                poly_order: self.poly_order,
            },
        );
        sgf_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SgfStream, SgfError> {
        SgfStream::try_new(SgfParams {
            period: self.period,
            poly_order: self.poly_order,
        })
    }
}

#[derive(Debug, Error)]
pub enum SgfError {
    #[error("sgf: input data slice is empty.")]
    EmptyInputData,
    #[error("sgf: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "sgf: invalid period: period = {period}, effective_period = {effective_period}, data length = {data_len}"
    )]
    InvalidPeriod {
        period: usize,
        effective_period: usize,
        data_len: usize,
    },
    #[error(
        "sgf: invalid polynomial order: poly_order = {poly_order}, effective_period = {effective_period}"
    )]
    InvalidPolyOrder {
        poly_order: usize,
        effective_period: usize,
    },
    #[error("sgf: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("sgf: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("sgf: invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("sgf: invalid poly-order range expansion: start={start}, end={end}, step={step}")]
    InvalidPolyOrderRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("sgf: invalid kernel passed to batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
pub(crate) fn effective_period(period: usize) -> usize {
    if period <= 1 {
        period
    } else if (period & 1) == 0 {
        period - 1
    } else {
        period
    }
}

#[inline]
pub(crate) fn validate_period_and_order(
    period: usize,
    poly_order: usize,
    len: usize,
) -> Result<usize, SgfError> {
    let effective = effective_period(period);
    if period < 3 || effective < 3 || effective > len {
        return Err(SgfError::InvalidPeriod {
            period,
            effective_period: effective,
            data_len: len,
        });
    }
    if poly_order >= effective {
        return Err(SgfError::InvalidPolyOrder {
            poly_order,
            effective_period: effective,
        });
    }
    Ok(effective)
}

fn solve_linear_system(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>, SgfError> {
    for pivot in 0..n {
        let mut best_row = pivot;
        let mut best_abs = a[pivot * n + pivot].abs();
        for row in (pivot + 1)..n {
            let cand = a[row * n + pivot].abs();
            if cand > best_abs {
                best_abs = cand;
                best_row = row;
            }
        }

        if best_abs <= 1e-15 {
            return Err(SgfError::InvalidPolyOrder {
                poly_order: n - 1,
                effective_period: 0,
            });
        }

        if best_row != pivot {
            for col in pivot..n {
                a.swap(pivot * n + col, best_row * n + col);
            }
            b.swap(pivot, best_row);
        }

        let pivot_val = a[pivot * n + pivot];
        for col in pivot..n {
            a[pivot * n + col] /= pivot_val;
        }
        b[pivot] /= pivot_val;

        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row * n + pivot];
            if factor == 0.0 {
                continue;
            }
            for col in pivot..n {
                a[row * n + col] -= factor * a[pivot * n + col];
            }
            b[row] -= factor * b[pivot];
        }
    }

    Ok(b)
}

pub(crate) fn build_endpoint_sgf_weights(
    period: usize,
    poly_order: usize,
) -> Result<AVec<f64>, SgfError> {
    let effective = validate_period_and_order(period, poly_order, period)?;
    let order = poly_order + 1;
    let mut gram = vec![0.0f64; order * order];

    for i in 0..effective {
        let x = (i as f64) - ((effective - 1) as f64);
        let mut powers = vec![1.0f64; order];
        for k in 1..order {
            powers[k] = powers[k - 1] * x;
        }
        for row in 0..order {
            for col in 0..order {
                gram[row * order + col] += powers[row] * powers[col];
            }
        }
    }

    let mut rhs = vec![0.0f64; order];
    rhs[0] = 1.0;
    let coeffs = solve_linear_system(gram, rhs, order)?;

    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, effective);
    let mut sum = 0.0f64;
    for i in 0..effective {
        let x = (i as f64) - ((effective - 1) as f64);
        let mut power = 1.0f64;
        let mut weight = 0.0f64;
        for &coef in &coeffs {
            weight += coef * power;
            power *= x;
        }
        weights.push(weight);
        sum += weight;
    }

    if sum != 0.0 {
        for weight in weights.iter_mut() {
            *weight /= sum;
        }
    }

    Ok(weights)
}

static SGF_DEFAULT_WEIGHTS_21_2: OnceLock<AVec<f64>> = OnceLock::new();

#[derive(Clone)]
enum SgfWeights {
    Static(&'static [f64]),
    Owned(AVec<f64>),
}

impl SgfWeights {
    #[inline(always)]
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Static(weights) => weights,
            Self::Owned(weights) => weights,
        }
    }
}

#[inline]
fn sgf_weights(
    requested_period: usize,
    effective_period: usize,
    poly_order: usize,
) -> Result<SgfWeights, SgfError> {
    if effective_period == 21 && poly_order == 2 {
        let weights = SGF_DEFAULT_WEIGHTS_21_2
            .get_or_init(|| build_endpoint_sgf_weights(21, 2).expect("valid default SGF weights"));
        Ok(SgfWeights::Static(weights.as_slice()))
    } else {
        Ok(SgfWeights::Owned(build_endpoint_sgf_weights(
            requested_period,
            poly_order,
        )?))
    }
}

#[derive(Clone)]
struct SgfPrepared<'a> {
    data: &'a [f64],
    weights: SgfWeights,
    period: usize,
    poly_order: usize,
    first: usize,
    kernel: Kernel,
}

#[inline]
pub fn sgf(input: &SgfInput) -> Result<SgfOutput, SgfError> {
    sgf_with_kernel(input, Kernel::Auto)
}

#[inline]
fn sgf_prepare<'a>(input: &'a SgfInput, kernel: Kernel) -> Result<SgfPrepared<'a>, SgfError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(SgfError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SgfError::AllValuesNaN)?;
    let requested_period = input.get_period();
    let poly_order = input.get_poly_order();
    let period = validate_period_and_order(requested_period, poly_order, len)?;

    if len - first < period {
        return Err(SgfError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let weights = sgf_weights(requested_period, period, poly_order)?;
    let kernel = match kernel {
        Kernel::Auto => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                detect_best_kernel()
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                Kernel::Scalar
            }
        }
        other => other,
    };

    Ok(SgfPrepared {
        data,
        weights,
        period,
        poly_order,
        first,
        kernel,
    })
}

#[inline(always)]
fn sgf_dot(window: &[f64], weights: &[f64]) -> f64 {
    let mut acc0 = 0.0f64;
    let mut acc1 = 0.0f64;
    let mut acc2 = 0.0f64;
    let mut acc3 = 0.0f64;
    let mut idx = 0usize;
    let len = weights.len();

    while idx + 3 < len {
        acc0 += window[idx] * weights[idx];
        acc1 += window[idx + 1] * weights[idx + 1];
        acc2 += window[idx + 2] * weights[idx + 2];
        acc3 += window[idx + 3] * weights[idx + 3];
        idx += 4;
    }
    while idx < len {
        acc0 += window[idx] * weights[idx];
        idx += 1;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

#[inline(always)]
fn sgf_dot_21(data: &[f64], from: usize, weights: &[f64]) -> f64 {
    unsafe {
        let d = data.as_ptr().add(from);
        let w = weights.as_ptr();

        let mut acc0 = *d.add(0) * *w.add(0);
        acc0 += *d.add(4) * *w.add(4);
        acc0 += *d.add(8) * *w.add(8);
        acc0 += *d.add(12) * *w.add(12);
        acc0 += *d.add(16) * *w.add(16);
        acc0 += *d.add(20) * *w.add(20);

        let mut acc1 = *d.add(1) * *w.add(1);
        acc1 += *d.add(5) * *w.add(5);
        acc1 += *d.add(9) * *w.add(9);
        acc1 += *d.add(13) * *w.add(13);
        acc1 += *d.add(17) * *w.add(17);

        let mut acc2 = *d.add(2) * *w.add(2);
        acc2 += *d.add(6) * *w.add(6);
        acc2 += *d.add(10) * *w.add(10);
        acc2 += *d.add(14) * *w.add(14);
        acc2 += *d.add(18) * *w.add(18);

        let mut acc3 = *d.add(3) * *w.add(3);
        acc3 += *d.add(7) * *w.add(7);
        acc3 += *d.add(11) * *w.add(11);
        acc3 += *d.add(15) * *w.add(15);
        acc3 += *d.add(19) * *w.add(19);

        (acc0 + acc1) + (acc2 + acc3)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sgf_compute_21_avx2(data: &[f64], weights: &[f64], first: usize, out: &mut [f64]) {
    let start = first + 20;
    let w0 = _mm256_loadu_pd(weights.as_ptr());
    let w4 = _mm256_loadu_pd(weights.as_ptr().add(4));
    let w8 = _mm256_loadu_pd(weights.as_ptr().add(8));
    let w12 = _mm256_loadu_pd(weights.as_ptr().add(12));
    let w16 = _mm256_loadu_pd(weights.as_ptr().add(16));
    let w20 = *weights.get_unchecked(20);
    let mut lanes = [0.0f64; 4];

    for idx in start..data.len() {
        let from = idx - 20;
        let d0 = _mm256_loadu_pd(data.as_ptr().add(from));
        let d4 = _mm256_loadu_pd(data.as_ptr().add(from + 4));
        let d8 = _mm256_loadu_pd(data.as_ptr().add(from + 8));
        let d12 = _mm256_loadu_pd(data.as_ptr().add(from + 12));
        let d16 = _mm256_loadu_pd(data.as_ptr().add(from + 16));

        let mut acc = _mm256_mul_pd(d0, w0);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(d4, w4));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(d8, w8));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(d12, w12));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(d16, w16));

        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
        lanes[0] += *data.get_unchecked(from + 20) * w20;
        *out.get_unchecked_mut(idx) = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
}

#[inline(always)]
fn sgf_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    let start = first + period - 1;
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if period == 21
        && matches!(
            kernel,
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch
        )
        && std::arch::is_x86_feature_detected!("avx2")
    {
        unsafe {
            sgf_compute_21_avx2(data, weights, first, out);
        }
        return;
    }

    if period == 21 {
        for idx in start..data.len() {
            let from = idx + 1 - 21;
            unsafe {
                *out.get_unchecked_mut(idx) = sgf_dot_21(data, from, weights);
            }
        }
        return;
    }

    for idx in start..data.len() {
        let from = idx + 1 - period;
        out[idx] = sgf_dot(&data[from..(idx + 1)], weights);
    }
}

pub fn sgf_with_kernel(input: &SgfInput, kernel: Kernel) -> Result<SgfOutput, SgfError> {
    let prepared = sgf_prepare(input, kernel)?;
    let warm = prepared.first + prepared.period - 1;
    let mut out = alloc_with_nan_prefix(prepared.data.len(), warm);
    sgf_compute_into(
        prepared.data,
        prepared.weights.as_slice(),
        prepared.period,
        prepared.first,
        prepared.kernel,
        &mut out,
    );
    Ok(SgfOutput { values: out })
}

#[inline]
pub fn sgf_into_slice(dst: &mut [f64], input: &SgfInput, kernel: Kernel) -> Result<(), SgfError> {
    let prepared = sgf_prepare(input, kernel)?;
    if dst.len() != prepared.data.len() {
        return Err(SgfError::OutputLengthMismatch {
            expected: prepared.data.len(),
            got: dst.len(),
        });
    }

    let warm = prepared.first + prepared.period - 1;
    for value in &mut dst[..warm] {
        *value = f64::from_bits(0x7ff8_0000_0000_0000);
    }
    sgf_compute_into(
        prepared.data,
        prepared.weights.as_slice(),
        prepared.period,
        prepared.first,
        prepared.kernel,
        dst,
    );
    Ok(())
}

#[inline]
pub fn sgf_into(input: &SgfInput, out: &mut [f64]) -> Result<(), SgfError> {
    sgf_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct SgfStream {
    period: usize,
    weights: AVec<f64>,
    ring: AVec<f64>,
    next: usize,
    count: usize,
}

impl SgfStream {
    pub fn try_new(params: SgfParams) -> Result<Self, SgfError> {
        let requested_period = params.period.unwrap_or(21);
        let poly_order = params.poly_order.unwrap_or(2);
        let period = validate_period_and_order(requested_period, poly_order, requested_period)?;
        let weights = build_endpoint_sgf_weights(requested_period, poly_order)?;
        let mut ring = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
        ring.resize(period, 0.0);
        Ok(Self {
            period,
            weights,
            ring,
            next: 0,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.ring[self.next] = value;
        self.next += 1;
        if self.next == self.period {
            self.next = 0;
        }
        if self.count < self.period {
            self.count += 1;
        }
        if self.count < self.period {
            return None;
        }

        let mut acc = 0.0f64;
        for idx in 0..self.period {
            let ring_idx = (self.next + idx) % self.period;
            acc += self.ring[ring_idx] * self.weights[idx];
        }
        Some(acc)
    }
}

#[derive(Clone, Debug)]
pub struct SgfBatchRange {
    pub period: (usize, usize, usize),
    pub poly_order: (usize, usize, usize),
}

impl Default for SgfBatchRange {
    fn default() -> Self {
        Self {
            period: (21, 81, 2),
            poly_order: (2, 2, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SgfBatchBuilder {
    range: SgfBatchRange,
    kernel: Kernel,
}

impl SgfBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    pub fn period_static(mut self, period: usize) -> Self {
        self.range.period = (period, period, 0);
        self
    }

    pub fn poly_order_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.poly_order = (start, end, step);
        self
    }

    pub fn poly_order_static(mut self, poly_order: usize) -> Self {
        self.range.poly_order = (poly_order, poly_order, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<SgfBatchOutput, SgfError> {
        sgf_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<SgfBatchOutput, SgfError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct SgfBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SgfParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SgfBatchOutput {
    pub fn row_for_params(&self, params: &SgfParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.period == params.period && combo.poly_order == params.poly_order
        })
    }

    pub fn values_for(&self, params: &SgfParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_axis(range: (usize, usize, usize), is_poly_order: bool) -> Result<Vec<usize>, SgfError> {
    let (start, end, step) = range;
    let values = if step == 0 || start == end {
        vec![start]
    } else if start < end {
        (start..=end).step_by(step.max(1)).collect()
    } else {
        let mut out = Vec::new();
        let mut cur = start;
        loop {
            out.push(cur);
            if cur <= end {
                break;
            }
            match cur.checked_sub(step.max(1)) {
                Some(next) => {
                    cur = next;
                    if cur < end {
                        break;
                    }
                }
                None => break,
            }
        }
        out
    };

    if values.is_empty() {
        if is_poly_order {
            return Err(SgfError::InvalidPolyOrderRange { start, end, step });
        }
        return Err(SgfError::InvalidRange { start, end, step });
    }
    Ok(values)
}

#[inline(always)]
pub fn expand_grid(range: &SgfBatchRange) -> Result<Vec<SgfParams>, SgfError> {
    let periods = expand_axis(range.period, false)?;
    let poly_orders = expand_axis(range.poly_order, true)?;
    let mut out = Vec::with_capacity(periods.len() * poly_orders.len());
    for &period in &periods {
        for &poly_order in &poly_orders {
            out.push(SgfParams {
                period: Some(period),
                poly_order: Some(poly_order),
            });
        }
    }
    Ok(out)
}

pub fn sgf_batch_with_kernel(
    data: &[f64],
    sweep: &SgfBatchRange,
    kernel: Kernel,
) -> Result<SgfBatchOutput, SgfError> {
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(SgfError::InvalidKernelForBatch(other)),
    };
    let single_kernel = match kernel {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    };
    sgf_batch_inner(data, sweep, single_kernel, true)
}

#[inline(always)]
pub fn sgf_batch_slice(
    data: &[f64],
    sweep: &SgfBatchRange,
    kernel: Kernel,
) -> Result<SgfBatchOutput, SgfError> {
    sgf_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn sgf_batch_par_slice(
    data: &[f64],
    sweep: &SgfBatchRange,
    kernel: Kernel,
) -> Result<SgfBatchOutput, SgfError> {
    sgf_batch_inner(data, sweep, kernel, true)
}

pub fn sgf_batch_into_slice(
    dst: &mut [f64],
    data: &[f64],
    sweep: &SgfBatchRange,
    kernel: Kernel,
) -> Result<Vec<SgfParams>, SgfError> {
    sgf_batch_inner_into(data, sweep, kernel, true, dst)
}

fn sgf_batch_inner(
    data: &[f64],
    sweep: &SgfBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<SgfBatchOutput, SgfError> {
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(SgfError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SgfError::AllValuesNaN)?;

    let rows = combos.len();
    let cols = data.len();
    let max_period = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(21))
        .map(effective_period)
        .max()
        .unwrap_or(0);

    if max_period == 0 || max_period > cols {
        return Err(SgfError::InvalidPeriod {
            period: max_period,
            effective_period: max_period,
            data_len: cols,
        });
    }
    if cols - first < max_period {
        return Err(SgfError::NotEnoughValidData {
            needed: max_period,
            valid: cols - first,
        });
    }

    let mut weights_flat = AVec::<f64>::with_capacity(
        CACHELINE_ALIGN,
        rows.checked_mul(max_period).ok_or(SgfError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?,
    );
    weights_flat.resize(rows * max_period, 0.0);

    let mut periods = Vec::with_capacity(rows);
    let mut warm = Vec::with_capacity(rows);
    for (row, combo) in combos.iter().enumerate() {
        let requested_period = combo.period.unwrap_or(21);
        let poly_order = combo.poly_order.unwrap_or(2);
        let period = validate_period_and_order(requested_period, poly_order, cols)?;
        if cols - first < period {
            return Err(SgfError::NotEnoughValidData {
                needed: period,
                valid: cols - first,
            });
        }
        let weights = build_endpoint_sgf_weights(requested_period, poly_order)?;
        let row_offset = row * max_period;
        weights_flat[row_offset..row_offset + period].copy_from_slice(&weights);
        periods.push(period);
        warm.push(first + period - 1);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);
    let row_fn = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = periods[row];
        let row_offset = row * max_period;
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        sgf_compute_into(
            data,
            &weights_flat[row_offset..row_offset + period],
            period,
            first,
            kernel,
            out_row,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        buf_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| row_fn(row, slice));
    } else {
        for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
            row_fn(row, slice);
        }
    }

    #[cfg(target_arch = "wasm32")]
    for (row, slice) in buf_mu.chunks_mut(cols).enumerate() {
        row_fn(row, slice);
    }

    use core::mem::ManuallyDrop;
    let mut guard = ManuallyDrop::new(buf_mu);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SgfBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn sgf_batch_inner_into(
    data: &[f64],
    sweep: &SgfBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SgfParams>, SgfError> {
    let result = sgf_batch_inner(data, sweep, kernel, parallel)?;
    let expected = result.values.len();
    if out.len() != expected {
        return Err(SgfError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    out.copy_from_slice(&result.values);
    Ok(result.combos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polynomial_series(len: usize, coeffs: &[f64], warm_prefix: usize) -> Vec<f64> {
        let mut data = vec![f64::NAN; len];
        for (idx, slot) in data.iter_mut().enumerate().skip(warm_prefix) {
            let x = idx as f64;
            let mut pow = 1.0;
            let mut y = 0.0;
            for &coef in coeffs {
                y += coef * pow;
                pow *= x;
            }
            *slot = y;
        }
        data
    }

    #[test]
    fn sgf_reproduces_quadratic_endpoint() {
        let data = polynomial_series(64, &[3.0, -0.25, 0.75], 3);
        let out = SgfBuilder::new()
            .period(9)
            .poly_order(2)
            .apply_slice(&data)
            .unwrap();
        for idx in 11..data.len() {
            assert!((out.values[idx] - data[idx]).abs() < 1e-10);
        }
    }

    #[test]
    fn sgf_reproduces_quartic_endpoint() {
        let data = polynomial_series(96, &[1.0, -2.0, 0.5, 0.125, -0.01], 4);
        let out = SgfBuilder::new()
            .period(11)
            .poly_order(4)
            .apply_slice(&data)
            .unwrap();
        for idx in 14..data.len() {
            assert!((out.values[idx] - data[idx]).abs() < 1e-7);
        }
    }

    #[test]
    fn sgf_stream_matches_batch() {
        let data = polynomial_series(80, &[2.0, 0.1, -0.03], 2);
        let batch = SgfBuilder::new()
            .period(9)
            .poly_order(2)
            .apply_slice(&data)
            .unwrap();
        let mut stream = SgfBuilder::new()
            .period(9)
            .poly_order(2)
            .into_stream()
            .unwrap();
        let mut streamed = vec![f64::NAN; data.len()];
        for (idx, &value) in data.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            if let Some(out) = stream.update(value) {
                streamed[idx] = out;
            }
        }
        assert_eq!(batch.values.len(), streamed.len());
        for idx in 0..streamed.len() {
            assert!(
                (batch.values[idx].is_nan() && streamed[idx].is_nan())
                    || (batch.values[idx] - streamed[idx]).abs() < 1e-10
            );
        }
    }

    #[test]
    fn sgf_batch_rows_match_single() {
        let data = polynomial_series(72, &[0.5, -0.2, 0.03], 1);
        let sweep = SgfBatchRange {
            period: (7, 11, 2),
            poly_order: (2, 2, 0),
        };
        let batch = sgf_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch).unwrap();
        for period in [7usize, 9, 11] {
            let params = SgfParams {
                period: Some(period),
                poly_order: Some(2),
            };
            let row = batch.values_for(&params).unwrap();
            let single = sgf(&SgfInput::from_slice(&data, params.clone())).unwrap();
            for idx in 0..data.len() {
                assert!(
                    (row[idx].is_nan() && single.values[idx].is_nan())
                        || (row[idx] - single.values[idx]).abs() < 1e-10
                );
            }
        }
    }

    #[test]
    fn sgf_rejects_invalid_poly_order() {
        let data = polynomial_series(32, &[1.0, 2.0], 0);
        let err = SgfBuilder::new()
            .period(5)
            .poly_order(5)
            .apply_slice(&data)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid polynomial order"));
    }
}
