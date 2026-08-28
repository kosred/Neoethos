use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SwmaData::Slice(slice) => slice,
            SwmaData::Candles { candles, source } => match *source {
                "open" => candles.open.as_slice(),
                "high" => candles.high.as_slice(),
                "low" => candles.low.as_slice(),
                "close" => candles.close.as_slice(),
                "volume" => candles.volume.as_slice(),
                "hl2" => candles.hl2.as_slice(),
                "hlc3" => candles.hlc3.as_slice(),
                "ohlc4" => candles.ohlc4.as_slice(),
                "hlcc4" | "hlcc" => candles.hlcc4.as_slice(),
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum SwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SwmaParams {
    pub period: Option<usize>,
}

impl Default for SwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct SwmaInput<'a> {
    pub data: SwmaData<'a>,
    pub params: SwmaParams,
}

impl<'a> SwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SwmaParams) -> Self {
        Self {
            data: SwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SwmaParams) -> Self {
        Self {
            data: SwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SwmaBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SwmaOutput, SwmaError> {
        let p = SwmaParams {
            period: self.period,
        };
        let i = SwmaInput::from_candles(c, "close", p);
        swma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SwmaOutput, SwmaError> {
        let p = SwmaParams {
            period: self.period,
        };
        let i = SwmaInput::from_slice(d, p);
        swma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SwmaStream, SwmaError> {
        let p = SwmaParams {
            period: self.period,
        };
        SwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SwmaError {
    #[error("swma: Input data slice is empty.")]
    EmptyInputData,
    #[error("swma: All values are NaN.")]
    AllValuesNaN,

    #[error(
        "swma: Invalid period: period = {period}, data length = {data_len}. Period must be between 1 and data length."
    )]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("swma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("swma: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("swma: Invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("swma: Invalid kernel passed to batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn swma(input: &SwmaInput) -> Result<SwmaOutput, SwmaError> {
    swma_with_kernel(input, Kernel::Auto)
}

#[inline]
fn swma_prepare<'a>(
    input: &'a SwmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], AVec<f64>, usize, usize, Kernel), SwmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(SwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SwmaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(SwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let weights = build_symmetric_triangle_avec(period);
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((data, weights, period, first, chosen))
}

#[inline(always)]
fn swma_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => swma_scalar(data, weights, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => swma_avx2(data, weights, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => swma_avx512(data, weights, period, first, out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                swma_scalar(data, weights, period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

pub fn swma_with_kernel(input: &SwmaInput, kernel: Kernel) -> Result<SwmaOutput, SwmaError> {
    let (data, weights, period, first, chosen) = swma_prepare(input, kernel)?;

    let len = data.len();
    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);

    swma_compute_into(data, &weights, period, first, chosen, &mut out);

    Ok(SwmaOutput { values: out })
}

#[inline]
pub fn swma_into_slice(dst: &mut [f64], input: &SwmaInput, kern: Kernel) -> Result<(), SwmaError> {
    let (data, weights, period, first, chosen) = swma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(SwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    swma_compute_into(data, &weights, period, first, chosen, dst);

    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    Ok(())
}

#[inline]
pub fn swma_into(input: &SwmaInput, out: &mut [f64]) -> Result<(), SwmaError> {
    let (data, weights, period, first, chosen) = swma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(SwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let warm = (first + period - 1).min(out.len());
    for v in &mut out[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    swma_compute_into(data, &weights, period, first, chosen, out);
    Ok(())
}

#[inline(always)]
fn build_symmetric_triangle_vec(n: usize) -> Vec<f64> {
    let mut w = Vec::with_capacity(n);
    if n == 1 {
        w.push(1.0);
    } else if n == 2 {
        w.extend_from_slice(&[0.5, 0.5]);
    } else if n % 2 == 0 {
        let half = n / 2;
        for i in 1..=half {
            w.push(i as f64);
        }
        for i in (1..=half).rev() {
            w.push(i as f64);
        }
        let sum: f64 = triangle_weight_sum(n);
        for x in &mut w {
            *x /= sum;
        }
    } else {
        let half_plus = (n + 1) / 2;
        for i in 1..=half_plus {
            w.push(i as f64);
        }
        for i in (1..half_plus).rev() {
            w.push(i as f64);
        }
        let sum: f64 = triangle_weight_sum(n);
        for x in &mut w {
            *x /= sum;
        }
    }
    w
}

#[inline(always)]
fn triangle_weight_sum(n: usize) -> f64 {
    if (n & 1) == 0 {
        let m = (n >> 1) as f64;
        m * (m + 1.0)
    } else {
        let m = ((n + 1) >> 1) as f64;
        m * m
    }
}

#[inline(always)]
fn build_symmetric_triangle_avec(n: usize) -> AVec<f64> {
    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, n);

    if n == 1 {
        weights.push(1.0);
    } else if n == 2 {
        weights.push(0.5);
        weights.push(0.5);
    } else if n % 2 == 0 {
        let half = n / 2;

        for i in 1..=half {
            weights.push(i as f64);
        }

        for i in (1..=half).rev() {
            weights.push(i as f64);
        }
    } else {
        let half_plus = (n + 1) / 2;

        for i in 1..=half_plus {
            weights.push(i as f64);
        }

        for i in (1..half_plus).rev() {
            weights.push(i as f64);
        }
    }

    let sum: f64 = if n <= 2 { 1.0 } else { triangle_weight_sum(n) };
    for w in weights.iter_mut() {
        *w /= sum;
    }

    weights
}

#[inline]
pub fn swma_scalar(
    data: &[f64],
    _weights: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    debug_assert!(out.len() >= data.len());
    debug_assert!(period >= 1);

    let len = data.len();
    if len == 0 {
        return;
    }

    let (a, b) = if (period & 1) != 0 {
        let m = (period + 1) >> 1;
        (m, m)
    } else {
        let m = period >> 1;
        (m, m + 1)
    };

    if period == 1 {
        unsafe {
            for i in first_val..len {
                *out.get_unchecked_mut(i) = *data.get_unchecked(i);
            }
        }
        return;
    }

    if period == 2 {
        unsafe {
            for i in (first_val + 1)..len {
                *out.get_unchecked_mut(i) =
                    (*data.get_unchecked(i - 1) + *data.get_unchecked(i)) * 0.5;
            }
        }
        return;
    }

    let inv_ab = 1.0 / ((a as f64) * (b as f64));
    let start_full_a = first_val + a - 1;
    let start_full_ab = first_val + period - 1;

    let mut ring = AVec::<f64>::with_capacity(CACHELINE_ALIGN, b);
    ring.resize(b, 0.0);
    let mut rb_idx = 0usize;

    let mut s1_sum = 0.0_f64;
    let mut s2_sum = 0.0_f64;

    unsafe {
        for i in first_val..len {
            s1_sum += *data.get_unchecked(i);

            if i >= start_full_a {
                let old = *ring.get_unchecked(rb_idx);
                s2_sum = s2_sum + (s1_sum - old);
                *ring.get_unchecked_mut(rb_idx) = s1_sum;

                rb_idx += 1;
                if rb_idx == b {
                    rb_idx = 0;
                }

                if i >= start_full_ab {
                    *out.get_unchecked_mut(i) = s2_sum * inv_ab;
                }

                s1_sum -= *data.get_unchecked(i + 1 - a);
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn swma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { swma_avx512_short(data, weights, period, first_valid, out) }
    } else {
        unsafe { swma_avx512_long(data, weights, period, first_valid, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn swma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    swma_scalar(data, weights, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn swma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    swma_scalar(data, weights, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn swma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    swma_scalar(data, weights, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct SwmaStream {
    period: usize,

    a: usize,
    b: usize,
    inv_ab: f64,

    ring_a: aligned_vec::AVec<f64>,
    idx_a: usize,
    cnt_a: usize,
    s1_sum: f64,

    ring_b: aligned_vec::AVec<f64>,
    idx_b: usize,
    cnt_b: usize,
    s2_sum: f64,
}

impl SwmaStream {
    pub fn try_new(params: SwmaParams) -> Result<Self, SwmaError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(SwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let (a, b) = if (period & 1) != 0 {
            let m = (period + 1) >> 1;
            (m, m)
        } else {
            let m = period >> 1;
            (m, m + 1)
        };

        let mut ring_a = aligned_vec::AVec::<f64>::with_capacity(aligned_vec::CACHELINE_ALIGN, a);
        ring_a.resize(a, 0.0);

        let mut ring_b = aligned_vec::AVec::<f64>::with_capacity(aligned_vec::CACHELINE_ALIGN, b);
        ring_b.resize(b, 0.0);

        Ok(Self {
            period,
            a,
            b,
            inv_ab: 1.0 / ((a as f64) * (b as f64)),
            ring_a,
            idx_a: 0,
            cnt_a: 0,
            s1_sum: 0.0,
            ring_b,
            idx_b: 0,
            cnt_b: 0,
            s2_sum: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        let ia = self.idx_a;

        let old_a = self.ring_a[ia];

        if self.cnt_a == self.a {
            self.s1_sum -= old_a;
        } else {
            self.cnt_a += 1;
        }
        self.ring_a[ia] = x;
        self.s1_sum += x;

        self.idx_a = ia + 1;
        if self.idx_a == self.a {
            self.idx_a = 0;
        }

        if self.cnt_a == self.a {
            let ib = self.idx_b;
            let old_s1 = self.ring_b[ib];

            if self.cnt_b == self.b {
                self.s2_sum -= old_s1;
            } else {
                self.cnt_b += 1;
            }
            self.ring_b[ib] = self.s1_sum;
            self.s2_sum += self.s1_sum;

            self.idx_b = ib + 1;
            if self.idx_b == self.b {
                self.idx_b = 0;
            }

            if self.cnt_b == self.b {
                return Some(self.s2_sum * self.inv_ab);
            }
        }

        None
    }
}

#[derive(Clone, Debug)]
pub struct SwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SwmaBatchBuilder {
    range: SwmaBatchRange,
    kernel: Kernel,
}

impl SwmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<SwmaBatchOutput, SwmaError> {
        swma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SwmaBatchOutput, SwmaError> {
        SwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SwmaBatchOutput, SwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<SwmaBatchOutput, SwmaError> {
        SwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn swma_batch_with_kernel(
    data: &[f64],
    sweep: &SwmaBatchRange,
    k: Kernel,
) -> Result<SwmaBatchOutput, SwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SwmaError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    swma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SwmaBatchOutput {
    pub fn row_for_params(&self, p: &SwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }

    pub fn values_for(&self, p: &SwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SwmaBatchRange) -> Vec<SwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start < end {
            return (start..=end).step_by(step.max(1)).collect();
        }

        let mut v = Vec::new();
        let mut cur = start;
        loop {
            v.push(cur);
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
        v
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn swma_batch_slice(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
) -> Result<SwmaBatchOutput, SwmaError> {
    swma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn swma_batch_par_slice(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
) -> Result<SwmaBatchOutput, SwmaError> {
    swma_batch_inner(data, sweep, kern, true)
}

pub fn swma_batch_into_slice(
    dst: &mut [f64],
    data: &[f64],
    sweep: &SwmaBatchRange,
    k: Kernel,
) -> Result<Vec<SwmaParams>, SwmaError> {
    swma_batch_inner_into(data, sweep, k, true, dst)
}

#[inline(always)]
fn swma_batch_inner(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SwmaBatchOutput, SwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(SwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }

    let len = data.len();
    if len == 0 {
        return Err(SwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

    if max_p == 0 || max_p > len {
        return Err(SwmaError::InvalidPeriod {
            period: max_p,
            data_len: len,
        });
    }
    if len - first < max_p {
        return Err(SwmaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let cap = rows.checked_mul(max_p).ok_or_else(|| {
        let (s, e, t) = sweep.period;
        SwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        }
    })?;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let w_start = row * max_p;

        if period == 1 {
            flat_w[w_start] = 1.0;
        } else if period == 2 {
            flat_w[w_start] = 0.5;
            flat_w[w_start + 1] = 0.5;
        } else if period % 2 == 0 {
            let half = period / 2;

            for i in 1..=half {
                flat_w[w_start + i - 1] = i as f64;
            }

            for i in (1..=half).rev() {
                flat_w[w_start + period - i] = i as f64;
            }

            let sum: f64 = flat_w[w_start..w_start + period].iter().sum();
            for i in 0..period {
                flat_w[w_start + i] /= sum;
            }
        } else {
            let half_plus = (period + 1) / 2;

            for i in 1..=half_plus {
                flat_w[w_start + i - 1] = i as f64;
            }

            for i in (1..half_plus).rev() {
                flat_w[w_start + period - i] = i as f64;
            }

            let sum: f64 = flat_w[w_start..w_start + period].iter().sum();
            for i in 0..period {
                flat_w[w_start + i] /= sum;
            }
        }
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    let _ = rows.checked_mul(cols).ok_or_else(|| {
        let (s, e, t) = sweep.period;
        SwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        }
    })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual_kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,

        other => other,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match simd {
            Kernel::Scalar => swma_row_scalar(data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => swma_row_avx2(data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => swma_row_avx512(data, first, period, w_ptr, out_row),
            _ => swma_row_scalar(data, first, period, w_ptr, out_row),
        }
    };

    {
        use std::mem::MaybeUninit;
        let rows_mut: &mut [MaybeUninit<f64>] = &mut buf_mu;
        #[cfg(not(target_arch = "wasm32"))]
        if parallel {
            use rayon::prelude::*;
            rows_mut
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in rows_mut.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in rows_mut.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
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

    Ok(SwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn swma_batch_inner_into(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SwmaParams>, SwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(SwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        });
    }

    let len = data.len();
    if len == 0 {
        return Err(SwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

    if max_p == 0 || max_p > len {
        return Err(SwmaError::InvalidPeriod {
            period: max_p,
            data_len: len,
        });
    }
    if len - first < max_p {
        return Err(SwmaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let cap = rows.checked_mul(max_p).ok_or_else(|| {
        let (s, e, t) = sweep.period;
        SwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        }
    })?;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let w_start = row * max_p;

        if period == 1 {
            flat_w[w_start] = 1.0;
        } else if period == 2 {
            flat_w[w_start] = 0.5;
            flat_w[w_start + 1] = 0.5;
        } else if period % 2 == 0 {
            let half = period / 2;

            for i in 1..=half {
                flat_w[w_start + i - 1] = i as f64;
            }

            for i in (1..=half).rev() {
                flat_w[w_start + period - i] = i as f64;
            }

            let sum: f64 = flat_w[w_start..w_start + period].iter().sum();
            for i in 0..period {
                flat_w[w_start + i] /= sum;
            }
        } else {
            let half_plus = (period + 1) / 2;

            for i in 1..=half_plus {
                flat_w[w_start + i - 1] = i as f64;
            }

            for i in (1..half_plus).rev() {
                flat_w[w_start + period - i] = i as f64;
            }

            let sum: f64 = flat_w[w_start..w_start + period].iter().sum();
            for i in 0..period {
                flat_w[w_start + i] /= sum;
            }
        }
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    let expected_len = rows.checked_mul(cols).ok_or_else(|| {
        let (s, e, t) = sweep.period;
        SwmaError::InvalidRange {
            start: s,
            end: e,
            step: t,
        }
    })?;
    if out.len() != expected_len {
        return Err(SwmaError::OutputLengthMismatch {
            expected: expected_len,
            got: out.len(),
        });
    }
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_uninit, cols, &warm);

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual_kern {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match simd {
            Kernel::Scalar => swma_row_scalar(data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => swma_row_avx2(data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => swma_row_avx512(data, first, period, w_ptr, out_row),
            _ => swma_row_scalar(data, first, period, w_ptr, out_row),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn swma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _w_ptr: *const f64,
    out: &mut [f64],
) {
    let len = data.len();
    if len == 0 {
        return;
    }

    let (a, b) = if (period & 1) != 0 {
        let m = (period + 1) >> 1;
        (m, m)
    } else {
        let m = period >> 1;
        (m, m + 1)
    };

    if period == 1 {
        for i in first..len {
            *out.get_unchecked_mut(i) = *data.get_unchecked(i);
        }
        return;
    }
    if period == 2 {
        for i in (first + 1)..len {
            *out.get_unchecked_mut(i) = (*data.get_unchecked(i - 1) + *data.get_unchecked(i)) * 0.5;
        }
        return;
    }

    let inv_ab = 1.0 / ((a as f64) * (b as f64));
    let start_full_a = first + a - 1;
    let start_full_ab = first + period - 1;

    let mut ring = AVec::<f64>::with_capacity(CACHELINE_ALIGN, b);
    ring.resize(b, 0.0);
    let mut rb_idx = 0usize;

    let mut s1_sum = 0.0_f64;
    let mut s2_sum = 0.0_f64;

    for i in first..len {
        s1_sum += *data.get_unchecked(i);

        if i >= start_full_a {
            let old = *ring.get_unchecked(rb_idx);
            s2_sum = s2_sum + (s1_sum - old);
            *ring.get_unchecked_mut(rb_idx) = s1_sum;
            rb_idx += 1;
            if rb_idx == b {
                rb_idx = 0;
            }

            if i >= start_full_ab {
                *out.get_unchecked_mut(i) = s2_sum * inv_ab;
            }

            s1_sum -= *data.get_unchecked(i + 1 - a);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn swma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    swma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn swma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    if period <= 32 {
        swma_row_avx512_short(data, first, period, w_ptr, out);
    } else {
        swma_row_avx512_long(data, first, period, w_ptr, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn swma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    swma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn swma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    swma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_swma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = SwmaParams { period: None };
        let input = SwmaInput::from_candles(&candles, "close", default_params);
        let output = swma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_swma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = SwmaInput::from_candles(&candles, "close", SwmaParams::default());
        let result = swma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59288.22222222222,
            59301.99999999999,
            59247.33333333333,
            59179.88888888889,
            59080.99999999999,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] SWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_swma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = SwmaInput::with_default_candles(&candles);
        match input.data {
            SwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SwmaData::Candles"),
        }
        let output = swma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_swma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SwmaParams { period: Some(0) };
        let input = SwmaInput::from_slice(&input_data, params);
        let res = swma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_swma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SwmaParams { period: Some(10) };
        let input = SwmaInput::from_slice(&data_small, params);
        let res = swma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_swma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SwmaParams { period: Some(5) };
        let input = SwmaInput::from_slice(&single_point, params);
        let res = swma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_swma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let first_params = SwmaParams { period: Some(5) };
        let first_input = SwmaInput::from_candles(&candles, "close", first_params);
        let first_result = swma_with_kernel(&first_input, kernel)?;
        let second_params = SwmaParams { period: Some(3) };
        let second_input = SwmaInput::from_slice(&first_result.values, second_params);
        let second_result = swma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_swma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = SwmaParams { period: Some(5) };
        let input = SwmaInput::from_candles(&candles, "close", params);
        let res = swma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_swma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let period = 5;
        let input = SwmaInput::from_candles(
            &candles,
            "close",
            SwmaParams {
                period: Some(period),
            },
        );
        let batch_output = swma_with_kernel(&input, kernel)?.values;
        let mut stream = SwmaStream::try_new(SwmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(swma_val) => stream_values.push(swma_val),
                None => stream_values.push(f64::NAN),
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
                "[{}] SWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_swma_tests {
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
    fn check_swma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_periods = vec![1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100];

        for period in test_periods {
            let params = SwmaParams {
                period: Some(period),
            };
            let input = SwmaInput::from_candles(&candles, "close", params);

            if period > candles.close.len() {
                continue;
            }

            let output = swma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_swma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_swma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period.max(2)..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = SwmaParams {
                    period: Some(period),
                };
                let input = SwmaInput::from_slice(&data, params);

                let SwmaOutput { values: out } = swma_with_kernel(&input, kernel).unwrap();
                let SwmaOutput { values: ref_out } =
                    swma_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                if period > 1 {
                    for i in 0..(period - 1) {
                        prop_assert!(
                            out[i].is_nan(),
                            "Expected NaN during warmup at index {}, got {}",
                            i,
                            out[i]
                        );
                    }
                }

                let weights = build_symmetric_triangle_avec(period);

                let weight_sum: f64 = weights.iter().sum();
                prop_assert!(
                    (weight_sum - 1.0).abs() < 1e-10,
                    "Weights don't sum to 1.0, got {}",
                    weight_sum
                );

                for i in 0..period / 2 {
                    let left = weights[i];
                    let right = weights[period - 1 - i];
                    prop_assert!(
                        (left - right).abs() < 1e-10,
                        "Weights not symmetric at positions {} and {}: {} vs {}",
                        i,
                        period - 1 - i,
                        left,
                        right
                    );
                }

                for i in (period - 1)..data.len() {
                    let window = &data[i + 1 - period..=i];
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];
                    let r = ref_out[i];

                    prop_assert!(
                        y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                        "idx {}: {} ∉ [{}, {}]",
                        i,
                        y,
                        lo,
                        hi
                    );

                    if period == 1 {
                        prop_assert!(
                            (y - data[i]).abs() <= f64::EPSILON,
                            "Period=1 should return input value at idx {}: {} vs {}",
                            i,
                            y,
                            data[i]
                        );
                    }

                    if period == 2 && i >= 1 {
                        let expected = (data[i - 1] + data[i]) / 2.0;
                        prop_assert!(
                            (y - expected).abs() < 1e-9,
                            "Period=2 should return average at idx {}: {} vs {}",
                            i,
                            y,
                            expected
                        );
                    }

                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                        prop_assert!(
                            (y - data[0]).abs() < 1e-9,
                            "Constant data should produce constant output at idx {}: {} vs {}",
                            i,
                            y,
                            data[0]
                        );
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    let max_ulp = if matches!(kernel, Kernel::Avx512) {
                        20
                    } else {
                        10
                    };

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= max_ulp,
                        "mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_swma_tests!(
        check_swma_partial_params,
        check_swma_accuracy,
        check_swma_default_candles,
        check_swma_zero_period,
        check_swma_period_exceeds_length,
        check_swma_very_small_dataset,
        check_swma_reinput,
        check_swma_nan_handling,
        check_swma_streaming,
        check_swma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_swma_tests!(check_swma_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = SwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = SwmaParams::default();
        let period = def.period.unwrap_or(5);
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59288.22222222222,
            59301.99999999999,
            59247.33333333333,
            59179.88888888889,
            59080.99999999999,
        ];
        let tail = &row[row.len() - 5..];
        for (i, &v) in tail.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{test}] default-row mismatch at idx {i}: {v} vs {}",
                expected[i]
            );
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

        let batch_configs = vec![
            (1, 10, 1),
            (3, 9, 3),
            (5, 25, 5),
            (10, 50, 10),
            (2, 2, 1),
            (1, 30, 2),
        ];

        for (start, end, step) in batch_configs {
            if end > c.close.len() {
                continue;
            }

            let output = SwmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let period = if row < output.combos.len() {
                    output.combos[row].period.unwrap_or(0)
                } else {
                    0
                };

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }
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

    #[test]
    fn test_swma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = SwmaInput::with_default_candles(&candles);
        let baseline = swma(&input)?.values;

        let mut out = vec![0.0f64; baseline.len()];

        {
            swma_into(&input, &mut out)?;
        }

        assert_eq!(out.len(), baseline.len());

        for (i, (&a, &b)) in out.iter().zip(baseline.iter()).enumerate() {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(
                equal,
                "into parity mismatch at idx {}: got {}, expected {}",
                i, a, b
            );
        }

        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
