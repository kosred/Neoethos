use crate::utilities::aligned_vector::AlignedVec;
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
use std::mem::MaybeUninit;
use thiserror::Error;

const JESSE_CWMA_PRIMARY_SOURCE_V1: &str = "https://raw.githubusercontent.com/jesse-ai/jesse/2f24de176d62e10d38f435e74590bad451815d6d/jesse/indicators/cwma.py";

#[inline(always)]
fn cube(x: f64) -> f64 {
    x * x * x
}

/// Creator-order CWMA value: newest to oldest, one rounded multiply followed
/// by one rounded add per term, then one final division by the weight sum.
///
/// Primary source: [`JESSE_CWMA_PRIMARY_SOURCE_V1`].
#[inline(always)]
fn cwma_creator_exact_value_v1(data: &[f64], row: usize, weights: &[f64], norm: f64) -> f64 {
    let _ = JESSE_CWMA_PRIMARY_SOURCE_V1;
    let mut sum = 0.0;
    for offset in 0..weights.len() {
        let term = data[row - offset] * weights[offset];
        sum += term;
    }
    sum / norm
}

#[inline(always)]
fn cwma_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "close" => &candles.close,
        "volume" => &candles.volume,
        _ => source_type(candles, source),
    }
}

impl<'a> AsRef<[f64]> for CwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CwmaData::Slice(slice) => slice,
            CwmaData::Candles { candles, source } => cwma_source(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CwmaParams {
    pub period: Option<usize>,
}

impl Default for CwmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CwmaInput<'a> {
    pub data: CwmaData<'a>,
    pub params: CwmaParams,
}

impl<'a> CwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CwmaParams) -> Self {
        Self {
            data: CwmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CwmaParams) -> Self {
        Self {
            data: CwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CwmaOutput, CwmaError> {
        let p = CwmaParams {
            period: self.period,
        };
        let i = CwmaInput::from_candles(c, "close", p);
        cwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CwmaOutput, CwmaError> {
        let p = CwmaParams {
            period: self.period,
        };
        let i = CwmaInput::from_slice(d, p);
        cwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CwmaStream, CwmaError> {
        let p = CwmaParams {
            period: self.period,
        };
        CwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CwmaError {
    #[error("cwma: Input data slice is empty.")]
    EmptyInputData,
    #[error("cwma: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "cwma: Invalid period specified for CWMA calculation: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },
    #[error(
        "cwma: Not enough valid data points to compute CWMA: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cwma: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("cwma: invalid sweep range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("cwma: invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("cwma: size overflow while computing {ctx}")]
    SizeOverflow { ctx: &'static str },
}

#[inline]
pub fn cwma(input: &CwmaInput) -> Result<CwmaOutput, CwmaError> {
    cwma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn cwma_prepare<'a>(
    input: &'a CwmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], Vec<f64>, usize, usize, f64, usize, Kernel), CwmaError> {
    let data: &[f64] = match &input.data {
        CwmaData::Candles { candles, source } => cwma_source(candles, source),
        CwmaData::Slice(sl) => sl,
    };
    let len = data.len();
    if len == 0 {
        return Err(CwmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CwmaError::AllValuesNaN)?;

    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    if period == 1 {
        return Err(CwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(CwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut weights = Vec::with_capacity(period - 1);
    let mut norm = 0.0;
    for i in 0..period - 1 {
        let w = cube((period - i) as f64);
        weights.push(w);
        norm += w;
    }
    let warm = first + period - 1;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, weights, period, first, norm, warm, chosen))
}

#[inline(always)]
fn cwma_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    norm: f64,
    chosen: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cwma_row_scalar(data, first, period, period - 1, weights.as_ptr(), norm, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cwma_avx2(data, weights, period, first, norm, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cwma_avx512(data, weights, period, first, norm, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                cwma_scalar(data, weights, period, first, norm, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn cwma_into_slice(dst: &mut [f64], input: &CwmaInput, kern: Kernel) -> Result<(), CwmaError> {
    let (data, weights, period, first, norm, warm, chosen) = cwma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(CwmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    cwma_compute_into(data, &weights, period, first, norm, chosen, dst);

    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }

    Ok(())
}

pub fn cwma_with_kernel(input: &CwmaInput, kernel: Kernel) -> Result<CwmaOutput, CwmaError> {
    let (data, weights, period, first, norm, warm, chosen) = cwma_prepare(input, kernel)?;
    let len = data.len();
    let mut out = alloc_with_nan_prefix(len, warm);
    cwma_compute_into(data, &weights, period, first, norm, chosen, &mut out);
    Ok(CwmaOutput { values: out })
}

#[inline]
pub fn cwma_into(input: &CwmaInput, out: &mut [f64]) -> Result<(), CwmaError> {
    cwma_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub unsafe fn cwma_scalar(
    data: &[f64],
    weights: &[f64],
    _period: usize,
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    let first_out = first_valid + weights.len();
    for row in first_out..data.len().min(out.len()) {
        *out.get_unchecked_mut(row) = cwma_creator_exact_value_v1(data, row, weights, norm);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cwma_avx2_creator_exact_v1(
    data: &[f64],
    weights: &[f64],
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    const LANES: usize = 4;
    let first_out = first_valid + weights.len();
    let end = data.len().min(out.len());
    let norm_lanes = _mm256_set1_pd(norm);
    let mut row = first_out;

    while row + LANES <= end {
        let mut sums = _mm256_setzero_pd();
        for offset in 0..weights.len() {
            let values = _mm256_loadu_pd(data.as_ptr().add(row - offset));
            let weight = _mm256_set1_pd(*weights.get_unchecked(offset));
            let terms = _mm256_mul_pd(values, weight);
            sums = _mm256_add_pd(sums, terms);
        }
        let values = _mm256_div_pd(sums, norm_lanes);
        _mm256_storeu_pd(out.as_mut_ptr().add(row), values);
        row += LANES;
    }

    while row < end {
        *out.get_unchecked_mut(row) = cwma_creator_exact_value_v1(data, row, weights, norm);
        row += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn cwma_avx2(
    data: &[f64],
    weights: &[f64],
    _period: usize,
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    cwma_avx2_creator_exact_v1(data, weights, first_valid, norm, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn cwma_avx512_creator_exact_v1(
    data: &[f64],
    weights: &[f64],
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    const LANES: usize = 8;
    let first_out = first_valid + weights.len();
    let end = data.len().min(out.len());
    let norm_lanes = _mm512_set1_pd(norm);
    let mut row = first_out;

    while row + LANES <= end {
        let mut sums = _mm512_setzero_pd();
        for offset in 0..weights.len() {
            let values = _mm512_loadu_pd(data.as_ptr().add(row - offset));
            let weight = _mm512_set1_pd(*weights.get_unchecked(offset));
            let terms = _mm512_mul_pd(values, weight);
            sums = _mm512_add_pd(sums, terms);
        }
        let values = _mm512_div_pd(sums, norm_lanes);
        _mm512_storeu_pd(out.as_mut_ptr().add(row), values);
        row += LANES;
    }

    while row < end {
        *out.get_unchecked_mut(row) = cwma_creator_exact_value_v1(data, row, weights, norm);
        row += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cwma_avx512(
    data: &[f64],
    weights: &[f64],
    _period: usize,
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    unsafe { cwma_avx512_creator_exact_v1(data, weights, first_valid, norm, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn cwma_avx512_short(
    data: &[f64],
    weights: &[f64],
    _period: usize,
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    cwma_avx512_creator_exact_v1(data, weights, first_valid, norm, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn cwma_avx512_long(
    data: &[f64],
    weights: &[f64],
    _period: usize,
    first_valid: usize,
    norm: f64,
    out: &mut [f64],
) {
    cwma_avx512_creator_exact_v1(data, weights, first_valid, norm, out);
}

#[derive(Debug, Clone)]
pub struct CwmaStream {
    period: usize,
    norm: f64,

    n: usize,

    ring: Vec<f64>,
    head: usize,
    filled: usize,
    nan_count: usize,

    total_count: usize,
    found_first: bool,
    first_idx: usize,

    m0: f64,
    m1: f64,
    m2: f64,
    m3: f64,

    s: f64,

    a: f64,
    w1: f64,
    wn: f64,
    alpha0: f64,
    alpha1: f64,
    alpha2: f64,

    n_f: f64,
    n_sq: f64,
    n_p1: f64,
    n_p1_sq: f64,

    moments_ready: bool,
}

impl CwmaStream {
    pub fn try_new(params: CwmaParams) -> Result<Self, CwmaError> {
        let period = params.period.unwrap_or(14);
        if period <= 1 {
            return Err(CwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let n = period - 1;

        let mut norm = 0.0;
        for j in 2..=period {
            let jf = j as f64;
            norm += jf * jf * jf;
        }
        let n_f = n as f64;
        let n_p1 = (n + 1) as f64;
        let n_p1_sq = n_p1 * n_p1;
        let a = (n + 2) as f64;

        let w1 = n_p1 * n_p1 * n_p1;
        let wn = 8.0;

        let alpha0 = -3.0 * a * a + 3.0 * a - 1.0;
        let alpha1 = 6.0 * a - 3.0;
        let alpha2 = -3.0;

        Ok(Self {
            period,
            norm,
            n,
            ring: vec![f64::NAN; n.max(1)],
            head: 0,
            filled: 0,
            nan_count: 0,
            total_count: 0,
            found_first: false,
            first_idx: 0,

            m0: 0.0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            s: 0.0,

            a,
            w1,
            wn,
            alpha0,
            alpha1,
            alpha2,
            n_f,
            n_sq: n_f * n_f,
            n_p1,
            n_p1_sq,
            moments_ready: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let idx = self.total_count;
        self.total_count = idx + 1;

        if !self.found_first {
            if value.is_nan() {
                return None;
            } else {
                self.found_first = true;
                self.first_idx = idx;
            }
        }

        let mut old = f64::NAN;
        if self.filled >= self.n {
            old = self.ring[self.head];
        }

        let new_nan = value.is_nan() as usize;
        let old_nan = (self.filled >= self.n && old.is_nan()) as usize;

        if self.n > 0 {
            self.ring[self.head] = value;
            self.head = (self.head + 1) % self.n;
        }

        if self.filled <= self.n {
            self.filled += 1;
            self.nan_count += new_nan;

            if self.filled == self.n + 1 {
                self.nan_count -= old_nan;
            }

            if self.filled <= self.n {
                return None;
            }

            if self.nan_count > 0 {
                self.moments_ready = false;
                return Some(f64::NAN);
            }

            self.rebuild_moments_and_sum();
            self.moments_ready = true;
            return Some(self.sum_weighted() / self.norm);
        }

        self.nan_count = self.nan_count + new_nan - old_nan;

        if self.nan_count > 0 {
            self.moments_ready = false;
            return Some(f64::NAN);
        }

        if !self.moments_ready {
            self.rebuild_moments_and_sum();
            self.moments_ready = true;
            return Some(self.sum_weighted() / self.norm);
        }

        let m0_prev = self.m0;
        let m1_prev = self.m1;
        let m2_prev = self.m2;
        let m3_prev = self.m3;

        let newv = value;
        let oldv = old;

        self.m0 = m0_prev + newv - oldv;
        self.m1 = (-self.n_p1).mul_add(oldv, m1_prev + m0_prev + newv);
        let tmp2 = m1_prev.mul_add(2.0, m2_prev + m0_prev + newv);
        self.m2 = (-self.n_p1_sq).mul_add(oldv, tmp2);
        let np13 = self.n_p1 * self.n_p1 * self.n_p1;
        let tmp3 = m2_prev.mul_add(3.0, m3_prev + m0_prev + newv);
        let tmp3 = m1_prev.mul_add(3.0, tmp3);
        self.m3 = (-np13).mul_add(oldv, tmp3);

        let mut ds = newv.mul_add(self.w1, 0.0);
        ds = oldv.mul_add(-self.wn, ds);
        let t1 = self.alpha0.mul_add(m0_prev - oldv, ds);
        let u1 = (-self.n_f).mul_add(oldv, m1_prev);
        let t2 = self.alpha1.mul_add(u1, t1);
        let u2 = (-self.n_sq).mul_add(oldv, m2_prev);
        let delta_s = self.alpha2.mul_add(u2, t2);
        self.s += delta_s;

        Some(self.sum_weighted() / self.norm)
    }

    #[inline(always)]
    fn rebuild_moments_and_sum(&mut self) {
        debug_assert!(self.nan_count == 0, "rebuild called with NaNs present");
        let mut m0 = 0.0;
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut s = 0.0;

        let a = self.a;
        for r in 1..=self.n {
            let idx = (self.head + self.n - r) % self.n;
            let v = self.ring[idx];
            let rf = r as f64;

            m0 += v;
            m1 += rf * v;
            m2 += (rf * rf) * v;
            m3 += (rf * rf * rf) * v;

            let w = {
                let t = a - rf;
                t * t * t
            };
            let term = v * w;
            s += term;
        }

        self.m0 = m0;
        self.m1 = m1;
        self.m2 = m2;
        self.m3 = m3;
        self.s = s;
    }

    #[inline(always)]
    fn sum_weighted(&self) -> f64 {
        let mut s = 0.0;
        let a = self.a;
        if self.n == 0 {
            return 0.0;
        }
        for r in 1..=self.n {
            let idx = (self.head + self.n - r) % self.n;
            let v = self.ring[idx];
            let rf = r as f64;
            let t = a - rf;
            let w = t * t * t;
            let term = v * w;
            s += term;
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct CwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CwmaBatchBuilder {
    range: CwmaBatchRange,
    kernel: Kernel,
}

impl CwmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<CwmaBatchOutput, CwmaError> {
        cwma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CwmaBatchOutput, CwmaError> {
        CwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CwmaBatchOutput, CwmaError> {
        let slice = cwma_source(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<CwmaBatchOutput, CwmaError> {
        CwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn cwma_batch_with_kernel(
    data: &[f64],
    sweep: &CwmaBatchRange,
    k: Kernel,
) -> Result<CwmaBatchOutput, CwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(CwmaError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CwmaBatchOutput {
    pub fn row_for_params(&self, p: &CwmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &CwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CwmaBatchRange) -> Vec<CwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        (lo..=hi).step_by(step).collect()
    }

    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn cwma_batch_slice(
    data: &[f64],
    sweep: &CwmaBatchRange,
    kern: Kernel,
) -> Result<CwmaBatchOutput, CwmaError> {
    cwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cwma_batch_par_slice(
    data: &[f64],
    sweep: &CwmaBatchRange,
    kern: Kernel,
) -> Result<CwmaBatchOutput, CwmaError> {
    cwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cwma_batch_inner(
    data: &[f64],
    sweep: &CwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CwmaBatchOutput, CwmaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();

    let _total = rows
        .checked_mul(cols)
        .ok_or(CwmaError::SizeOverflow { ctx: "rows*cols" })?;

    if cols == 0 {
        return Err(CwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CwmaError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

    if (cols - first) < max_p {
        return Err(CwmaError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    cwma_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(CwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn cwma_batch_inner_into(
    data: &[f64],
    sweep: &CwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CwmaParams>, CwmaError> {
    let combos = expand_grid(sweep);

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CwmaError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(CwmaError::SizeOverflow { ctx: "rows*cols" })?;
    if out.len() != expected {
        return Err(CwmaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let mut norms = vec![0.0; rows];

    let cap = rows
        .checked_mul(max_p)
        .ok_or(CwmaError::SizeOverflow { ctx: "rows*max_p" })?;
    let mut aligned = AlignedVec::with_capacity(cap);
    let flat_w = aligned.as_mut_slice();

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let mut norm = 0.0;
        for i in 0..period - 1 {
            let w = cube((period - i) as f64);
            flat_w[row * max_p + i] = w;
            norm += w;
        }
        norms[row] = norm;
    }

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let norm = *norms.get_unchecked(row);

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cwma_row_avx512(data, first, period, max_p, w_ptr, norm, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cwma_row_avx2(data, first, period, max_p, w_ptr, norm, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                cwma_row_scalar(data, first, period, max_p, w_ptr, norm, out_row)
            }
            _ => cwma_row_scalar(data, first, period, max_p, w_ptr, norm, out_row),
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
unsafe fn cwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    norm: f64,
    out: &mut [f64],
) {
    let wlen = period - 1;
    let start_idx = first + wlen;
    for i in start_idx..data.len().min(out.len()) {
        let weights = core::slice::from_raw_parts(w_ptr, wlen);
        out[i] = cwma_creator_exact_value_v1(data, i, weights, norm);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn cwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    norm: f64,
    out: &mut [f64],
) {
    let weights = core::slice::from_raw_parts(w_ptr, period - 1);
    cwma_avx512_creator_exact_v1(data, weights, first, norm, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    norm: f64,
    out: &mut [f64],
) {
    let weights = core::slice::from_raw_parts(w_ptr, period - 1);
    cwma_avx2_creator_exact_v1(data, weights, first, norm, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn reviewed_routeable_subset_close_v3(rows: usize) -> Vec<f64> {
        (0..rows)
            .map(|row| {
                let drift = row as f64 * 0.000_000_7;
                let wave = match row % 11 {
                    0 => 0.000_041,
                    1 => -0.000_027,
                    2 => 0.000_013,
                    3 => -0.000_036,
                    4 => 0.000_022,
                    5 => -0.000_009,
                    6 => 0.000_033,
                    7 => -0.000_019,
                    8 => 0.000_006,
                    9 => -0.000_031,
                    _ => 0.000_017,
                };
                1.075 + drift + wave
            })
            .collect()
    }

    fn assert_creator_bits(values: &[f64], data: &[f64], period: usize) {
        let mut weights = Vec::with_capacity(period - 1);
        let mut norm = 0.0;
        for offset in 0..period - 1 {
            let base = (period - offset) as f64;
            let weight = base * base * base;
            weights.push(weight);
            norm += weight;
        }
        for row in period - 1..data.len() {
            let expected = cwma_creator_exact_value_v1(data, row, &weights, norm);
            assert_eq!(
                values[row].to_bits(),
                expected.to_bits(),
                "creator-order mismatch at row {row}"
            );
        }
    }

    #[test]
    fn jesse_creator_order_is_exact_across_cpu_lanes() {
        const PERIOD: usize = 14;
        let data = reviewed_routeable_subset_close_v3(96);
        let input = CwmaInput::from_slice(
            &data,
            CwmaParams {
                period: Some(PERIOD),
            },
        );

        let scalar = cwma_with_kernel(&input, Kernel::Scalar).unwrap();
        assert_eq!(scalar.values[18].to_bits(), 0x3ff1_333f_5fc7_4bcd);
        assert_creator_bits(&scalar.values, &data, PERIOD);

        let auto = cwma_with_kernel(&input, Kernel::Auto).unwrap();
        assert_creator_bits(&auto.values, &data, PERIOD);

        let batch = cwma_batch_with_kernel(
            &data,
            &CwmaBatchRange {
                period: (PERIOD, PERIOD, 0),
            },
            Kernel::ScalarBatch,
        )
        .unwrap();
        assert_creator_bits(&batch.values[..data.len()], &data, PERIOD);

        let mut stream = CwmaStream::try_new(CwmaParams {
            period: Some(PERIOD),
        })
        .unwrap();
        let streamed = data
            .iter()
            .copied()
            .map(|value| stream.update(value).unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        assert_creator_bits(&streamed, &data, PERIOD);

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                let avx2 = cwma_with_kernel(&input, Kernel::Avx2).unwrap();
                assert_creator_bits(&avx2.values, &data, PERIOD);
            }
            if is_x86_feature_detected!("avx512f") {
                let avx512 = cwma_with_kernel(&input, Kernel::Avx512).unwrap();
                assert_creator_bits(&avx512.values, &data, PERIOD);
            }
        }
    }

    #[test]
    fn test_cwma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CwmaInput::with_default_candles(&candles);

        let baseline = cwma(&input)?.values;

        let mut out = vec![0.0; candles.close.len()];
        cwma_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for (i, (a, b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(*a, *b),
                "mismatch at index {}: baseline={}, into={}",
                i,
                a,
                b
            );
        }

        Ok(())
    }

    fn check_cwma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let default_params = CwmaParams { period: None };
        let input_def = CwmaInput::from_candles(&candles, "close", default_params);
        let output_def = cwma_with_kernel(&input_def, kernel)?;
        assert_eq!(output_def.values.len(), candles.close.len());

        let params_14 = CwmaParams { period: Some(14) };
        let input_14 = CwmaInput::from_candles(&candles, "hl2", params_14);
        let output_14 = cwma_with_kernel(&input_14, kernel)?;
        assert_eq!(output_14.values.len(), candles.close.len());

        let params_custom = CwmaParams { period: Some(20) };
        let input_custom = CwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = cwma_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cwma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CwmaInput::with_default_candles(&candles);
        let result = cwma_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());

        let expected_last_five = [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-9,
                "[{}] CWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_cwma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CwmaInput::with_default_candles(&candles);
        match input.data {
            CwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CwmaData::Candles"),
        }
        let output = cwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cwma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CwmaParams { period: Some(0) };
        let input = CwmaInput::from_slice(&input_data, params);
        let res = cwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }

    fn check_cwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CwmaParams { period: Some(10) };
        let input = CwmaInput::from_slice(&data_small, params);
        let res = cwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CwmaParams { period: Some(9) };
        let input = CwmaInput::from_slice(&single_point, params);
        let res = cwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cwma_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = CwmaInput::from_slice(&empty, CwmaParams::default());
        let res = cwma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(CwmaError::EmptyInputData)),
            "[{}] Should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_cwma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let first_params = CwmaParams { period: Some(80) };
        let first_input = CwmaInput::from_candles(&candles, "close", first_params);
        let first_result = cwma_with_kernel(&first_input, kernel)?;

        let second_params = CwmaParams { period: Some(60) };
        let second_input = CwmaInput::from_slice(&first_result.values, second_params);
        let second_result = cwma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());

        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "[{}] Found unexpected NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_cwma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = CwmaInput::from_candles(&candles, "close", CwmaParams { period: Some(9) });
        let res = cwma_with_kernel(&input, kernel)?;
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

    fn check_cwma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let period = 9;
        let input = CwmaInput::from_candles(
            &candles,
            "close",
            CwmaParams {
                period: Some(period),
            },
        );
        let batch_output = cwma_with_kernel(&input, kernel)?.values;

        let mut stream = CwmaStream::try_new(CwmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] CWMA streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_cwma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            CwmaParams::default(),
            CwmaParams { period: Some(2) },
            CwmaParams { period: Some(3) },
            CwmaParams { period: Some(5) },
            CwmaParams { period: Some(7) },
            CwmaParams { period: Some(10) },
            CwmaParams { period: Some(14) },
            CwmaParams { period: Some(20) },
            CwmaParams { period: Some(30) },
            CwmaParams { period: Some(50) },
            CwmaParams { period: Some(100) },
            CwmaParams { period: Some(200) },
            CwmaParams { period: Some(250) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = CwmaInput::from_candles(&candles, "close", params.clone());
            let output = cwma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_cwma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_cwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=32).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                (-1e3f64..1e3f64).prop_filter("finite a", |a| a.is_finite() && *a != 0.0),
                -1e3f64..1e3f64,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, a, b)| {
                let params = CwmaParams {
                    period: Some(period),
                };
                let input = CwmaInput::from_slice(&data, params.clone());

                let fast = cwma_with_kernel(&input, kernel);
                let slow = cwma_with_kernel(&input, Kernel::Scalar);

                match (fast, slow) {
                    (Err(e1), Err(e2))
                        if std::mem::discriminant(&e1) == std::mem::discriminant(&e2) =>
                    {
                        return Ok(());
                    }

                    (Err(e1), Err(e2)) => {
                        prop_assert!(false, "different errors: fast={:?} slow={:?}", e1, e2)
                    }

                    (Err(e1), Ok(_)) => {
                        prop_assert!(false, "fast errored {e1:?} but scalar succeeded")
                    }
                    (Ok(_), Err(e2)) => {
                        prop_assert!(false, "scalar errored {e2:?} but fast succeeded")
                    }

                    (Ok(fast), Ok(reference)) => {
                        let CwmaOutput { values: out } = fast;
                        let CwmaOutput { values: rref } = reference;

                        let mut stream = CwmaStream::try_new(params.clone()).unwrap();
                        let mut s_out = Vec::with_capacity(data.len());
                        for &v in &data {
                            s_out.push(stream.update(v).unwrap_or(f64::NAN));
                        }

                        let transformed: Vec<f64> = data.iter().map(|x| a * x + b).collect();
                        let t_out = cwma(&CwmaInput::from_slice(&transformed, params))?.values;

                        for i in (period - 1)..data.len() {
                            let w = &data[i + 1 - period..=i];
                            let (lo, hi) = w
                                .iter()
                                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                                    (l.min(v), h.max(v))
                                });
                            let y = out[i];
                            let yr = rref[i];
                            let ys = s_out[i];
                            let yt = t_out[i];

                            prop_assert!(
                                y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                                "idx {i}: {y} ∉ [{lo}, {hi}]"
                            );

                            if period == 1 && y.is_finite() {
                                prop_assert!((y - data[i]).abs() <= f64::EPSILON);
                            }

                            if w.iter().all(|v| *v == w[0]) {
                                prop_assert!((y - w[0]).abs() <= 1e-9);
                            }

                            if data[..=i].windows(2).all(|p| p[0] <= p[1])
                                && y.is_finite()
                                && out[i - 1].is_finite()
                            {
                                prop_assert!(y >= out[i - 1] - 1e-12);
                            }

                            {
                                let expected = a * y + b;
                                let diff = (yt - expected).abs();
                                let tol_abs = 1e-9_f64;
                                let tol_rel = expected.abs() * 1e-9;
                                let ulp = yt.to_bits().abs_diff(expected.to_bits());

                                prop_assert!(
                                    diff <= tol_abs.max(tol_rel) || ulp <= 8,
                                    "idx {i}: affine mismatch diff={diff:e}  ULP={ulp}"
                                );
                            }

                            let ulp = y.to_bits().abs_diff(yr.to_bits());
                            prop_assert!(
                                (y - yr).abs() <= 1e-9 || ulp <= 4,
                                "idx {i}: fast={y} ref={yr} ULP={ulp}"
                            );

                            prop_assert!(
                                (y - ys).abs() <= 1e-9 || (y.is_nan() && ys.is_nan()),
                                "idx {i}: stream mismatch"
                            );
                        }

                        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(data.len());
                        let warm = first + period - 1;
                        prop_assert!(out[..warm].iter().all(|v| v.is_nan()));
                    }
                }

                Ok(())
            })
            .unwrap();

        assert!(cwma(&CwmaInput::from_slice(&[], CwmaParams::default())).is_err());
        assert!(
            cwma(&CwmaInput::from_slice(
                &[f64::NAN; 12],
                CwmaParams::default()
            ))
            .is_err()
        );
        assert!(
            cwma(&CwmaInput::from_slice(
                &[1.0; 5],
                CwmaParams { period: Some(8) }
            ))
            .is_err()
        );
        assert!(
            cwma(&CwmaInput::from_slice(
                &[1.0; 5],
                CwmaParams { period: Some(0) }
            ))
            .is_err()
        );

        Ok(())
    }

    macro_rules! generate_all_cwma_tests {
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

    generate_all_cwma_tests!(
        check_cwma_partial_params,
        check_cwma_accuracy,
        check_cwma_default_candles,
        check_cwma_zero_period,
        check_cwma_period_exceeds_length,
        check_cwma_very_small_dataset,
        check_cwma_empty_input,
        check_cwma_reinput,
        check_cwma_nan_handling,
        check_cwma_streaming,
        check_cwma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_cwma_tests!(check_cwma_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let output = CwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = CwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59224.641237300435,
            59213.64831277214,
            59171.21190130624,
            59167.01279027576,
            59039.413552249636,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 5, 1),
            (5, 25, 5),
            (10, 50, 10),
            (2, 4, 1),
            (50, 150, 25),
            (9, 21, 2),
            (9, 21, 4),
            (100, 300, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = CwmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

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
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
