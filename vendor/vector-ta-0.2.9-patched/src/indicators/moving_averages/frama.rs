use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for FramaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            FramaData::Candles { candles } => &candles.close,
            FramaData::Slices { close, .. } => close,
        }
    }
}

/// Stable identity handed to the Classic semantic-v9 source-closure owner.
pub const FRAMA_F64_SEMANTIC_VERSION: u32 = 3;
pub const FRAMA_F64_SEMANTIC_IDENTITY: &str =
    "frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2";
pub const FRAMA_MAX_WINDOW: usize = 1024;

const FRAMA_CANONICAL_QNAN_BITS: u64 = 0x7ff8_0000_0000_0000;

#[inline(always)]
fn frama_canonical_nan_f64_v3() -> f64 {
    f64::from_bits(FRAMA_CANONICAL_QNAN_BITS)
}

#[inline(always)]
fn frama_is_finite_triplet_v3(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite()
}

/// Stable f64 FRAMA affine recurrence, shared by every host execution route.
///
/// The subtraction is rounded first and the multiply-add is fused once.  The
/// strict CUDA mirror uses `__dsub_rn` followed by `__fma_rn` in the same
/// order, so this schedule does not depend on host contraction choices.
#[inline(always)]
fn frama_stable_update_f64_v2(close: f64, previous: f64, alpha: f64) -> f64 {
    alpha.mul_add(close - previous, previous)
}

#[derive(Debug, Clone)]
pub enum FramaData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct FramaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FramaParams {
    pub window: Option<usize>,
    pub sc: Option<usize>,
    pub fc: Option<usize>,
}

impl Default for FramaParams {
    fn default() -> Self {
        Self {
            window: Some(10),
            sc: Some(300),
            fc: Some(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FramaInput<'a> {
    pub data: FramaData<'a>,
    pub params: FramaParams,
}

impl<'a> FramaInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FramaParams) -> Self {
        Self {
            data: FramaData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FramaParams,
    ) -> Self {
        Self {
            data: FramaData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FramaParams::default())
    }
    #[inline]
    pub fn get_window(&self) -> usize {
        self.params.window.unwrap_or(10)
    }
    #[inline]
    pub fn get_sc(&self) -> usize {
        self.params.sc.unwrap_or(300)
    }
    #[inline]
    pub fn get_fc(&self) -> usize {
        self.params.fc.unwrap_or(1)
    }

    #[inline]
    pub fn slices(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            FramaData::Candles { candles } => (&candles.high, &candles.low, &candles.close),
            FramaData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FramaBuilder {
    window: Option<usize>,
    sc: Option<usize>,
    fc: Option<usize>,
    kernel: Kernel,
}

impl Default for FramaBuilder {
    fn default() -> Self {
        Self {
            window: None,
            sc: None,
            fc: None,
            kernel: Kernel::Auto,
        }
    }
}
impl FramaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn window(mut self, n: usize) -> Self {
        self.window = Some(n);
        self
    }
    #[inline(always)]
    pub fn sc(mut self, x: usize) -> Self {
        self.sc = Some(x);
        self
    }
    #[inline(always)]
    pub fn fc(mut self, x: usize) -> Self {
        self.fc = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<FramaOutput, FramaError> {
        let p = FramaParams {
            window: self.window,
            sc: self.sc,
            fc: self.fc,
        };
        let i = FramaInput::from_candles(c, p);
        frama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FramaOutput, FramaError> {
        let p = FramaParams {
            window: self.window,
            sc: self.sc,
            fc: self.fc,
        };
        let i = FramaInput::from_slices(high, low, close, p);
        frama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<FramaStream, FramaError> {
        let p = FramaParams {
            window: self.window,
            sc: self.sc,
            fc: self.fc,
        };
        FramaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum FramaError {
    #[error("frama: Input data slice is empty.")]
    EmptyInputData,

    #[error("frama: Mismatched slice lengths: high={high}, low={low}, close={close}")]
    MismatchedInputLength {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("frama: All values are NaN.")]
    AllValuesNaN,
    #[error("frama: Invalid window: window = {window}, data length = {data_len}")]
    InvalidWindow { window: usize, data_len: usize },
    #[error("frama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("frama: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("frama: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("frama: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("frama: Invalid smoothing constants: sc={sc}, fc={fc}")]
    InvalidSmoothing { sc: usize, fc: usize },

    #[error("frama: arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
}

#[inline(always)]
fn frama_evenized_window_v3(window: usize, data_len: usize) -> Result<usize, FramaError> {
    if window == 0 || window > FRAMA_MAX_WINDOW {
        return Err(FramaError::InvalidWindow { window, data_len });
    }
    let evenized = window
        .checked_add(window & 1)
        .ok_or(FramaError::InvalidWindow { window, data_len })?;
    if evenized > FRAMA_MAX_WINDOW {
        return Err(FramaError::InvalidWindow { window, data_len });
    }
    Ok(evenized)
}

#[inline(always)]
fn frama_validate_combo_windows_v3(
    combos: &[FramaParams],
    data_len: usize,
) -> Result<usize, FramaError> {
    let mut max_evenized = 0usize;
    for combo in combos {
        let window = combo.window.unwrap_or(10);
        max_evenized = max_evenized.max(frama_evenized_window_v3(window, data_len)?);
    }
    Ok(max_evenized)
}

#[inline(always)]
fn frama_validate_window_axis_v3(
    (start, end, step): (usize, usize, usize),
    data_len: usize,
) -> Result<(), FramaError> {
    if step == 0 || start == end {
        frama_evenized_window_v3(start, data_len)?;
        return Ok(());
    }

    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut window = lo;
    loop {
        frama_evenized_window_v3(window, data_len)?;
        match window.checked_add(step) {
            Some(next) if next <= hi => window = next,
            _ => break,
        }
    }
    Ok(())
}

#[inline]
pub fn frama(input: &FramaInput) -> Result<FramaOutput, FramaError> {
    frama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn frama_prepare<'a>(
    input: &'a FramaInput,
    kernel: Kernel,
) -> Result<
    (
        (&'a [f64], &'a [f64], &'a [f64]),
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        Kernel,
    ),
    FramaError,
> {
    let (high, low, close) = input.slices();
    let len = high.len();
    if len == 0 {
        return Err(FramaError::EmptyInputData);
    }
    if low.len() != len || close.len() != len {
        return Err(FramaError::MismatchedInputLength {
            high: len,
            low: low.len(),
            close: close.len(),
        });
    }
    let window = input.get_window();
    let sc = input.get_sc();
    let fc = input.get_fc();
    if sc == 0 || fc == 0 {
        return Err(FramaError::InvalidSmoothing { sc, fc });
    }
    let win = frama_evenized_window_v3(window, len)?;
    if window > len {
        return Err(FramaError::InvalidWindow {
            window,
            data_len: len,
        });
    }
    let first = (0..len)
        .find(|&i| frama_is_finite_triplet_v3(high[i], low[i], close[i]))
        .ok_or(FramaError::AllValuesNaN)?;

    if (len - first) < win {
        return Err(FramaError::NotEnoughValidData {
            needed: win,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let warm = first + win - 1;

    Ok(((high, low, close), window, sc, fc, first, len, warm, chosen))
}

#[inline(always)]
fn frama_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
    _warm: usize,
    chosen: Kernel,
    out: &mut [f64],
) -> Result<(), FramaError> {
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {}
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {}
        _ => unreachable!("`Auto` must be resolved above"),
    }
    frama_f64_segmented_deque_v3(high, low, close, window, sc, fc, first, len, out)
}

pub fn frama_with_kernel(input: &FramaInput, kernel: Kernel) -> Result<FramaOutput, FramaError> {
    let ((high, low, close), window, sc, fc, first, len, warm, chosen) =
        frama_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(len, warm);
    frama_compute_into(
        high, low, close, window, sc, fc, first, len, warm, chosen, &mut out,
    )?;
    Ok(FramaOutput { values: out })
}

#[inline]
pub fn frama_into(input: &FramaInput, out: &mut [f64]) -> Result<(), FramaError> {
    frama_into_slice(out, input, Kernel::Auto)
}

#[derive(Copy, Clone)]
struct MonoDeque<const CAP: usize> {
    buf: [usize; CAP],
    head: usize,
    tail: usize,
}
impl<const CAP: usize> MonoDeque<CAP> {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            buf: [0; CAP],
            head: 0,
            tail: 0,
        }
    }
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
    }

    #[inline(always)]
    unsafe fn front(&self) -> usize {
        *self.buf.get_unchecked(self.head)
    }

    #[inline(always)]
    fn expire(&mut self, idx_out: usize) {
        if !self.is_empty() && unsafe { self.front() } == idx_out {
            self.head = (self.head + 1) % CAP;
        }
    }

    #[inline(always)]
    unsafe fn push_max(&mut self, idx: usize, data: &[f64]) {
        while !self.is_empty() {
            let last = self.buf[(self.tail + CAP - 1) % CAP];
            if *data.get_unchecked(last) >= *data.get_unchecked(idx) {
                break;
            }
            self.tail = (self.tail + CAP - 1) % CAP;
        }
        self.buf[self.tail] = idx;
        self.tail = (self.tail + 1) % CAP;
    }

    #[inline(always)]
    unsafe fn push_min(&mut self, idx: usize, data: &[f64]) {
        while !self.is_empty() {
            let last = self.buf[(self.tail + CAP - 1) % CAP];
            if *data.get_unchecked(last) <= *data.get_unchecked(idx) {
                break;
            }
            self.tail = (self.tail + CAP - 1) % CAP;
        }
        self.buf[self.tail] = idx;
        self.tail = (self.tail + 1) % CAP;
    }
}

#[inline(always)]
fn frama_f64_segmented_deque_v3(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    sc: usize,
    fc: usize,
    _first: usize,
    len: usize,
    out: &mut [f64],
) -> Result<(), FramaError> {
    let window = frama_evenized_window_v3(window, len)?;
    let half = window / 2;

    out[..len].fill(frama_canonical_nan_f64_v3());

    let mut d_left_max: MonoDeque<FRAMA_MAX_WINDOW> = MonoDeque::new();
    let mut d_left_min: MonoDeque<FRAMA_MAX_WINDOW> = MonoDeque::new();
    let mut d_right_max: MonoDeque<FRAMA_MAX_WINDOW> = MonoDeque::new();
    let mut d_right_min: MonoDeque<FRAMA_MAX_WINDOW> = MonoDeque::new();

    let w_ln = (2.0 / (sc as f64 + 1.0)).ln();
    let sc_lim = 2.0 / (sc as f64 + 1.0);
    let mut d_prev = 1.0;
    let mut finite_run = 0usize;
    let mut seed_sum = 0.0;
    let mut previous = frama_canonical_nan_f64_v3();

    for i in 0..len {
        if !frama_is_finite_triplet_v3(high[i], low[i], close[i]) {
            d_left_max.clear();
            d_left_min.clear();
            d_right_max.clear();
            d_right_min.clear();
            finite_run = 0;
            seed_sum = 0.0;
            d_prev = 1.0;
            previous = frama_canonical_nan_f64_v3();
            out[i] = previous;
            continue;
        }

        if finite_run < window {
            seed_sum += close[i];
            unsafe {
                if finite_run < half {
                    d_left_max.push_max(i, high);
                    d_left_min.push_min(i, low);
                } else {
                    d_right_max.push_max(i, high);
                    d_right_min.push_min(i, low);
                }
            }
            finite_run += 1;
            if finite_run == window {
                previous = seed_sum / window as f64;
                out[i] = previous;
            }
            continue;
        }

        let max1 = high[unsafe { d_right_max.front() }];
        let min1 = low[unsafe { d_right_min.front() }];
        let max2 = high[unsafe { d_left_max.front() }];
        let min2 = low[unsafe { d_left_min.front() }];
        let max3 = max1.max(max2);
        let min3 = min1.min(min2);

        let n1 = (max1 - min1) / (half as f64);
        let n2 = (max2 - min2) / (half as f64);
        let n3 = (max3 - min3) / (window as f64);

        let d_cur = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2
        } else {
            d_prev
        };
        d_prev = d_cur;

        let mut alpha0 = (w_ln * (d_cur - 1.0)).exp();
        if alpha0 < 0.1 {
            alpha0 = 0.1;
        }
        if alpha0 > 1.0 {
            alpha0 = 1.0;
        }
        let old_n = (2.0 - alpha0) / alpha0;
        let new_n = (sc - fc) as f64 * ((old_n - 1.0) / (sc as f64 - 1.0)) + fc as f64;
        let mut alpha = 2.0 / (new_n + 1.0);
        if alpha < sc_lim {
            alpha = sc_lim;
        }
        if alpha > 1.0 {
            alpha = 1.0;
        }

        previous = frama_stable_update_f64_v2(close[i], previous, alpha);
        out[i] = previous;

        let idx_out = i - window;
        let crossing = i - half;
        d_left_max.expire(idx_out);
        d_left_min.expire(idx_out);
        d_right_max.expire(crossing);
        d_right_min.expire(crossing);

        unsafe {
            d_left_max.push_max(crossing, high);
            d_left_min.push_min(crossing, low);
            d_right_max.push_max(i, high);
            d_right_min.push_min(i, low);
        }
    }

    Ok(())
}

#[inline(always)]
pub fn frama_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    sc: usize,
    fc: usize,
    first: usize,
    len: usize,
) -> Result<FramaOutput, FramaError> {
    let win = frama_evenized_window_v3(window, len)?;
    if window > len {
        return Err(FramaError::InvalidWindow {
            window,
            data_len: len,
        });
    }
    let warm = first + win - 1;

    let mut out = alloc_with_nan_prefix(len, warm);
    frama_compute_into(
        high,
        low,
        close,
        window,
        sc,
        fc,
        first,
        len,
        warm,
        Kernel::Scalar,
        &mut out,
    )?;
    Ok(FramaOutput { values: out })
}

#[derive(Clone, Debug)]
pub struct FramaBatchRange {
    pub window: (usize, usize, usize),
    pub sc: (usize, usize, usize),
    pub fc: (usize, usize, usize),
}
impl Default for FramaBatchRange {
    fn default() -> Self {
        Self {
            window: (10, 259, 1),
            sc: (300, 300, 0),
            fc: (1, 1, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct FramaBatchBuilder {
    range: FramaBatchRange,
    kernel: Kernel,
}
impl FramaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn window_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.window = (start, end, step);
        self
    }
    #[inline]
    pub fn sc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sc = (start, end, step);
        self
    }
    #[inline]
    pub fn fc_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fc = (start, end, step);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FramaBatchOutput, FramaError> {
        frama_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_slice(self, slice: &[f64]) -> Result<FramaBatchOutput, FramaError> {
        self.apply_slices(slice, slice, slice)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k: Kernel,
    ) -> Result<FramaBatchOutput, FramaError> {
        FramaBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<FramaBatchOutput, FramaError> {
        let h = c.select_candle_field("high").unwrap();
        let l = c.select_candle_field("low").unwrap();
        let o = c.select_candle_field("close").unwrap();
        self.apply_slices(h, l, o)
    }
    pub fn with_default_candles(c: &Candles) -> Result<FramaBatchOutput, FramaError> {
        FramaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}
#[derive(Clone, Debug)]
pub struct FramaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<FramaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl FramaBatchOutput {
    pub fn row_for_params(&self, p: &FramaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.window.unwrap_or(10) == p.window.unwrap_or(10)
                && c.sc.unwrap_or(300) == p.sc.unwrap_or(300)
                && c.fc.unwrap_or(1) == p.fc.unwrap_or(1)
        })
    }
    pub fn values_for(&self, p: &FramaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}
#[inline(always)]
fn expand_grid(r: &FramaBatchRange) -> Vec<FramaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }

        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let mut v = Vec::new();
        let mut x = lo;
        loop {
            v.push(x);
            match x.checked_add(step) {
                Some(nx) if nx <= hi => x = nx,
                _ => break,
            }
        }
        if start > end {
            v.reverse();
        }
        v
    }
    let windows = axis_usize(r.window);
    let scs = axis_usize(r.sc);
    let fcs = axis_usize(r.fc);

    let cap = windows
        .len()
        .checked_mul(scs.len())
        .and_then(|x| x.checked_mul(fcs.len()))
        .unwrap_or(0);
    let mut out = Vec::with_capacity(cap);
    for &w in &windows {
        for &s in &scs {
            for &f in &fcs {
                out.push(FramaParams {
                    window: Some(w),
                    sc: Some(s),
                    fc: Some(f),
                });
            }
        }
    }
    out
}

pub fn frama_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    k: Kernel,
) -> Result<FramaBatchOutput, FramaError> {
    let kernel = match k {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx2Batch,
            other => other,
        },
        other if other.is_batch() => other,
        other => return Err(FramaError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    frama_batch_inner(high, low, close, sweep, simd, true)
}

#[inline(always)]
pub fn frama_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
) -> Result<FramaBatchOutput, FramaError> {
    frama_batch_inner(high, low, close, sweep, kern, false)
}
#[inline(always)]
pub fn frama_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
) -> Result<FramaBatchOutput, FramaError> {
    frama_batch_inner(high, low, close, sweep, kern, true)
}

#[inline]
fn frama_batch_admission_v3(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
) -> Result<(Vec<FramaParams>, usize, usize), FramaError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FramaError::EmptyInputData);
    }
    if low.len() != high.len() || close.len() != high.len() {
        return Err(FramaError::MismatchedInputLength {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }

    let len = high.len();
    frama_validate_window_axis_v3(sweep.window, len)?;
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(FramaError::InvalidRange {
            start: sweep.window.0,
            end: sweep.window.1,
            step: sweep.window.2,
        });
    }
    let max_even_window = frama_validate_combo_windows_v3(&combos, len)?;
    let first = (0..len)
        .find(|&index| frama_is_finite_triplet_v3(high[index], low[index], close[index]))
        .ok_or(FramaError::AllValuesNaN)?;
    let valid_tail = len - first;
    if valid_tail < max_even_window {
        return Err(FramaError::NotEnoughValidData {
            needed: max_even_window,
            valid: valid_tail,
        });
    }
    Ok((combos, first, max_even_window))
}

#[inline(always)]
fn frama_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<FramaBatchOutput, FramaError> {
    let (combos, first, _) = frama_batch_admission_v3(high, low, close, sweep)?;

    let rows = combos.len();
    let cols = close.len();

    let _ = rows
        .checked_mul(cols)
        .ok_or(FramaError::ArithmeticOverflow {
            context: "rows*cols",
        })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|p| {
            let mut win = p.window.unwrap();

            if win & 1 == 1 {
                win += 1;
            }
            first + win - 1
        })
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let combos_ret = {
        let out: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(buf_mu.as_mut_ptr() as *mut f64, buf_mu.len())
        };
        frama_batch_inner_into(high, low, close, sweep, kern, parallel, out)?
    };
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(FramaBatchOutput {
        values,
        combos: combos_ret,
        rows,
        cols,
    })
}

#[inline(always)]
fn frama_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FramaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<FramaParams>, FramaError> {
    let (combos, first, _) = frama_batch_admission_v3(high, low, close, sweep)?;

    let rows = combos.len();
    let cols = high.len();

    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        let p = &combos[row];
        let window = p.window.unwrap();
        let sc = p.sc.unwrap();
        let fc = p.fc.unwrap();

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => frama_row_avx512(high, low, close, first, window, dst, sc, fc),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => frama_row_avx2(high, low, close, first, window, dst, sc, fc),
            _ => frama_row_scalar(high, low, close, first, window, dst, sc, fc),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[derive(Debug, Clone)]
pub struct FramaStream {
    window: usize,
    sc: usize,
    fc: usize,
    n: usize,
    w: f64,
    buffer: Vec<(f64, f64, f64)>,
    head: usize,
    filled: bool,
    last_val: f64,
    d_prev: f64,
    alpha_prev: f64,

    half: usize,
    idx: usize,

    dq_r_max: DqMax,
    dq_r_min: DqMin,
    dq_l_max: DqMax,
    dq_l_min: DqMin,
    dq_w_max: DqMax,
    dq_w_min: DqMin,

    pm_right: f64,
    pn_right: f64,
    pm_left: f64,
    pn_left: f64,
    pm_full: f64,
    pn_full: f64,

    sc_floor: f64,
}
impl FramaStream {
    pub fn try_new(params: FramaParams) -> Result<Self, FramaError> {
        let window = params.window.unwrap_or(10);
        let sc = params.sc.unwrap_or(300);
        let fc = params.fc.unwrap_or(1);
        let n = frama_evenized_window_v3(window, 0)?;
        Ok(Self {
            window,
            sc,
            fc,
            n,
            w: (2.0 / (sc as f64 + 1.0)).ln(),
            buffer: vec![(f64::NAN, f64::NAN, f64::NAN); n],
            head: 0,
            filled: false,
            last_val: f64::NAN,
            d_prev: 1.0,
            alpha_prev: 2.0 / (sc as f64 + 1.0),

            half: n / 2,
            idx: 0,
            dq_r_max: DqMax::default(),
            dq_r_min: DqMin::default(),
            dq_l_max: DqMax::default(),
            dq_l_min: DqMin::default(),
            dq_w_max: DqMax::default(),
            dq_w_min: DqMin::default(),
            pm_right: f64::NAN,
            pn_right: f64::NAN,
            pm_left: f64::NAN,
            pn_left: f64::NAN,
            pm_full: f64::NAN,
            pn_full: f64::NAN,
            sc_floor: 2.0 / (sc as f64 + 1.0),
        })
    }

    #[inline(always)]
    fn reset_finite_segment_v3(&mut self) {
        let nan = frama_canonical_nan_f64_v3();
        self.buffer.fill((nan, nan, nan));
        self.head = 0;
        self.filled = false;
        self.last_val = nan;
        self.d_prev = 1.0;
        self.alpha_prev = self.sc_floor;
        self.idx = 0;
        self.dq_r_max.clear();
        self.dq_r_min.clear();
        self.dq_l_max.clear();
        self.dq_l_min.clear();
        self.dq_w_max.clear();
        self.dq_w_min.clear();
        self.pm_right = nan;
        self.pn_right = nan;
        self.pm_left = nan;
        self.pn_left = nan;
        self.pm_full = nan;
        self.pn_full = nan;
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if !frama_is_finite_triplet_v3(high, low, close) {
            self.reset_finite_segment_v3();
            return None;
        }

        if !self.filled {
            self.buffer[self.head] = (high, low, close);
            self.head += 1;

            if self.head == self.n {
                self.head = 0;
                self.filled = true;

                let sum: f64 = self.buffer.iter().map(|&(_, _, c)| c).sum();
                self.last_val = sum / self.n as f64;

                self.dq_r_max.clear();
                self.dq_r_min.clear();
                self.dq_l_max.clear();
                self.dq_l_min.clear();
                self.dq_w_max.clear();
                self.dq_w_min.clear();

                for j in 0..self.n {
                    let (h, l, _) = self.buffer[j];
                    self.dq_w_max.push(j, h);
                    self.dq_w_min.push(j, l);
                    if j < self.half {
                        self.dq_l_max.push(j, h);
                        self.dq_l_min.push(j, l);
                    } else {
                        self.dq_r_max.push(j, h);
                        self.dq_r_min.push(j, l);
                    }
                }

                self.pm_right = self.dq_r_max.front_val().unwrap_or(f64::NAN);
                self.pn_right = self.dq_r_min.front_val().unwrap_or(f64::NAN);
                self.pm_left = self.dq_l_max.front_val().unwrap_or(f64::NAN);
                self.pn_left = self.dq_l_min.front_val().unwrap_or(f64::NAN);
                self.pm_full = self.dq_w_max.front_val().unwrap_or(f64::NAN);
                self.pn_full = self.dq_w_min.front_val().unwrap_or(f64::NAN);

                self.idx = self.n;

                return Some(self.last_val);
            }

            return None;
        }

        let i = self.idx;

        let right_lb = i.saturating_sub(self.half);
        let left_lb = i.saturating_sub(self.n);
        self.dq_r_max.expire_lt(right_lb);
        self.dq_r_min.expire_lt(right_lb);
        self.dq_l_max.expire_lt(left_lb);
        self.dq_l_min.expire_lt(left_lb);
        self.dq_w_max.expire_lt(left_lb);
        self.dq_w_min.expire_lt(left_lb);

        let (max_r, min_r) = {
            let mr = self.dq_r_max.front_val().unwrap_or(self.pm_right);
            let nr = self.dq_r_min.front_val().unwrap_or(self.pn_right);
            (mr, nr)
        };
        let (max_l, min_l) = {
            let ml = self.dq_l_max.front_val().unwrap_or(self.pm_left);
            let nl = self.dq_l_min.front_val().unwrap_or(self.pn_left);
            (ml, nl)
        };
        let (max_w, min_w) = {
            let mw = self.dq_w_max.front_val().unwrap_or(self.pm_full);
            let nw = self.dq_w_min.front_val().unwrap_or(self.pn_full);
            (mw, nw)
        };

        self.pm_right = max_r;
        self.pn_right = min_r;
        self.pm_left = max_l;
        self.pn_left = min_l;
        self.pm_full = max_w;
        self.pn_full = min_w;

        let half_f = self.half as f64;
        let win_f = self.n as f64;

        let n1 = (max_r - min_r) / half_f;
        let n2 = (max_l - min_l) / half_f;
        let n3 = (max_w - min_w) / win_f;

        let d = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
            ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2
        } else {
            self.d_prev
        };
        self.d_prev = d;

        let mut a0 = (self.w * (d - 1.0)).exp();
        if a0 < 0.1 {
            a0 = 0.1;
        }
        if a0 > 1.0 {
            a0 = 1.0;
        }

        let old_n = (2.0 - a0) / a0;
        let new_n =
            (self.sc - self.fc) as f64 * ((old_n - 1.0) / (self.sc as f64 - 1.0)) + self.fc as f64;

        let mut alpha = 2.0 / (new_n + 1.0);
        if alpha < self.sc_floor {
            alpha = self.sc_floor;
        }
        if alpha > 1.0 {
            alpha = 1.0;
        }
        self.alpha_prev = alpha;

        let output = frama_stable_update_f64_v2(close, self.last_val, alpha);

        self.dq_r_max.push(i, high);
        self.dq_r_min.push(i, low);
        self.dq_w_max.push(i, high);
        self.dq_w_min.push(i, low);

        if i >= self.half {
            let j = i - self.half;
            let (h_l, l_l, _) = self.buffer[j % self.n];
            self.dq_l_max.push(j, h_l);
            self.dq_l_min.push(j, l_l);
        }

        self.buffer[self.head] = (high, low, close);
        self.head = (self.head + 1) % self.n;

        self.idx += 1;
        self.last_val = output;
        Some(output)
    }
}

#[derive(Default, Debug, Clone)]
struct DqMax {
    q: VecDeque<(usize, f64)>,
}
#[derive(Default, Debug, Clone)]
struct DqMin {
    q: VecDeque<(usize, f64)>,
}

impl DqMax {
    #[inline(always)]
    fn clear(&mut self) {
        self.q.clear();
    }
    #[inline(always)]
    fn expire_lt(&mut self, bound: usize) {
        while let Some(&(i, _)) = self.q.front() {
            if i < bound {
                self.q.pop_front();
            } else {
                break;
            }
        }
    }
    #[inline(always)]
    fn push(&mut self, idx: usize, val: f64) {
        while let Some(&(_, v)) = self.q.back() {
            if v >= val {
                break;
            }
            self.q.pop_back();
        }
        self.q.push_back((idx, val));
    }
    #[inline(always)]
    fn front_val(&self) -> Option<f64> {
        self.q.front().map(|&(_, v)| v)
    }
}
impl DqMin {
    #[inline(always)]
    fn clear(&mut self) {
        self.q.clear();
    }
    #[inline(always)]
    fn expire_lt(&mut self, bound: usize) {
        while let Some(&(i, _)) = self.q.front() {
            if i < bound {
                self.q.pop_front();
            } else {
                break;
            }
        }
    }
    #[inline(always)]
    fn push(&mut self, idx: usize, val: f64) {
        while let Some(&(_, v)) = self.q.back() {
            if v <= val {
                break;
            }
            self.q.pop_back();
        }
        self.q.push_back((idx, val));
    }
    #[inline(always)]
    fn front_val(&self) -> Option<f64> {
        self.q.front().map(|&(_, v)| v)
    }
}

#[inline(always)]
pub unsafe fn frama_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    window: usize,
    out: &mut [f64],
    sc: usize,
    fc: usize,
) {
    frama_f64_segmented_deque_v3(high, low, close, window, sc, fc, first, high.len(), out).unwrap();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn frama_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    window: usize,
    out: &mut [f64],
    sc: usize,
    fc: usize,
) {
    frama_f64_segmented_deque_v3(high, low, close, window, sc, fc, first, high.len(), out).unwrap();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn frama_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    window: usize,
    out: &mut [f64],
    sc: usize,
    fc: usize,
) {
    frama_f64_segmented_deque_v3(high, low, close, window, sc, fc, first, high.len(), out).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAMA_RUST_SOURCE: &str = include_str!("frama.rs");
    const FRAMA_CUDA_SOURCE: &str =
        include_str!("../../../kernels/cuda/moving_averages/frama_kernel.cu");

    #[test]
    fn frama_stable_f64_recurrence_v2_source_contract() {
        let production = FRAMA_RUST_SOURCE
            .split("#[cfg(test)]")
            .next()
            .expect("frama.rs must retain a production section");

        assert!(
            production.contains("frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2")
        );
        assert!(production.contains("fn frama_f64_segmented_deque_v3("));
        assert!(production.contains("fn frama_is_finite_triplet_v3("));
        assert!(production.contains("fn frama_canonical_nan_f64_v3()"));
        assert_eq!(
            production.matches("frama_f64_segmented_deque_v3(").count(),
            5,
            "one definition plus direct and the scalar/AVX batch row labels"
        );
        assert!(production.contains(
            "fn frama_stable_update_f64_v2(close: f64, previous: f64, alpha: f64) -> f64"
        ));
        assert!(production.contains("alpha.mul_add(close - previous, previous)"));
        assert_eq!(
            production.matches("frama_stable_update_f64_v2(").count(),
            3,
            "one definition plus the common static authority and stream calls"
        );
        assert!(!production.contains("fn frama_small_scan"));
        assert!(!production.contains("fn frama_avx2_small"));
        assert!(!production.contains("fn frama_avx512_small"));
        assert!(!production.contains("close.mul_add(alpha, (1.0 - alpha)"));
        assert!(!production.contains("alpha * close[i] + (1.0 - alpha)"));

        assert!(
            FRAMA_CUDA_SOURCE
                .contains("__device__ __forceinline__ double neo_frama_stable_update_f64_v2(")
        );
        assert!(
            FRAMA_CUDA_SOURCE
                .contains("return __fma_rn(alpha, __dsub_rn(close, previous), previous);")
        );
        assert_eq!(
            FRAMA_CUDA_SOURCE
                .matches("neo_frama_stable_update_f64_v2(")
                .count(),
            2,
            "one device helper definition and one strict-f64 call"
        );
        assert!(!FRAMA_CUDA_SOURCE.contains("fma(close[i], alpha, (1.0 - alpha) * o[i - 1])"));
        assert!(FRAMA_CUDA_SOURCE.contains("row_out[i] = fmaf(alpha, (close_i - prev), prev);"));
        assert!(
            FRAMA_CUDA_SOURCE
                .contains("frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2")
        );
        assert!(
            FRAMA_CUDA_SOURCE
                .contains("if (!isfinite(high[i]) || !isfinite(low[i]) || !isfinite(close[i]))")
        );
        assert!(FRAMA_CUDA_SOURCE.contains("finite_run = 0;"));
        assert!(FRAMA_CUDA_SOURCE.contains("d_prev = 1.0;"));
    }

    #[test]
    fn frama_stable_f64_recurrence_v2_has_the_reviewed_rounding_gate() {
        let alpha = f64::from_bits(0x3fd0_530d_08f1_7f5c);
        let close = f64::from_bits(0x3ff0_caee_225a_2949);
        let previous = f64::from_bits(0x3ff0_b81c_8de4_fa9d);

        let actual = frama_stable_update_f64_v2(close, previous, alpha);
        assert_eq!(actual.to_bits(), 0x3ff0_bce9_5ea4_019e);

        let superseded = close.mul_add(alpha, (1.0 - alpha) * previous);
        assert_eq!(superseded.to_bits(), 0x3ff0_bce9_5ea4_019d);
        assert_ne!(actual.to_bits(), superseded.to_bits());
    }

    #[test]
    fn frama_f64_hole_contract_exact_witness_resets_every_route() {
        const LEN: usize = 36;
        const WINDOW: usize = 34;
        const QNAN_BITS: u64 = 0x7ff8_0000_0000_0000;

        let mut high = vec![1.0; LEN];
        let mut low = vec![0.0; LEN];
        let mut close = vec![0.5; LEN];
        high[34] = f64::from_bits(QNAN_BITS);
        low[34] = f64::from_bits(QNAN_BITS);
        close[34] = f64::from_bits(QNAN_BITS);

        let params = FramaParams {
            window: Some(WINDOW),
            sc: Some(300),
            fc: Some(1),
        };
        let input = FramaInput::from_slices(&high, &low, &close, params.clone());
        let direct = frama_with_kernel(&input, Kernel::Scalar)
            .expect("the exact hole witness is a valid FRAMA input");
        let batch = frama_batch_with_kernel(
            &high,
            &low,
            &close,
            &FramaBatchRange {
                window: (WINDOW, WINDOW, 0),
                sc: (300, 300, 0),
                fc: (1, 1, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("the exact hole witness is a valid FRAMA batch input");
        let mut stream = FramaStream::try_new(params).expect("stream parameters are valid");
        let streamed = (0..LEN)
            .map(|row| stream.update(high[row], low[row], close[row]))
            .collect::<Vec<_>>();

        for row in 0..33 {
            assert_eq!(direct.values[row].to_bits(), QNAN_BITS, "direct row {row}");
            assert_eq!(batch.values[row].to_bits(), QNAN_BITS, "batch row {row}");
            assert_eq!(streamed[row], None, "stream row {row}");
        }
        assert_eq!(direct.values[33].to_bits(), 0.5f64.to_bits());
        assert_eq!(batch.values[33].to_bits(), 0.5f64.to_bits());
        assert_eq!(streamed[33].map(f64::to_bits), Some(0.5f64.to_bits()));
        for row in 34..LEN {
            assert_eq!(direct.values[row].to_bits(), QNAN_BITS, "direct row {row}");
            assert_eq!(batch.values[row].to_bits(), QNAN_BITS, "batch row {row}");
            assert_eq!(streamed[row], None, "stream row {row}");
        }
    }
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    use crate::utilities::enums::Kernel;
    use paste::paste;
    use proptest::prelude::*;

    fn frama_direct_halves_reference(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        window: usize,
        sc: usize,
        fc: usize,
    ) -> Vec<f64> {
        let mut win = window;
        if win & 1 == 1 {
            win += 1;
        }
        let half = win / 2;
        let mut out = vec![f64::NAN; close.len()];
        let mut seed = 0.0;
        for value in &close[..win] {
            seed += *value;
        }
        out[win - 1] = seed / win as f64;

        let w_ln = (2.0 / (sc as f64 + 1.0)).ln();
        let sc_floor = 2.0 / (sc as f64 + 1.0);
        let mut d_prev = 1.0;
        for i in win..close.len() {
            let start = i - win;
            let mid = start + half;
            let mut max2 = f64::MIN;
            let mut min2 = f64::MAX;
            for j in start..mid {
                max2 = max2.max(high[j]);
                min2 = min2.min(low[j]);
            }
            let mut max1 = f64::MIN;
            let mut min1 = f64::MAX;
            for j in mid..i {
                max1 = max1.max(high[j]);
                min1 = min1.min(low[j]);
            }
            let max3 = max1.max(max2);
            let min3 = min1.min(min2);
            let n1 = (max1 - min1) / half as f64;
            let n2 = (max2 - min2) / half as f64;
            let n3 = (max3 - min3) / win as f64;
            let d_cur = if n1 > 0.0 && n2 > 0.0 && n3 > 0.0 {
                ((n1 + n2).ln() - n3.ln()) / std::f64::consts::LN_2
            } else {
                d_prev
            };
            d_prev = d_cur;

            let mut alpha0 = (w_ln * (d_cur - 1.0)).exp();
            if alpha0 < 0.1 {
                alpha0 = 0.1;
            }
            if alpha0 > 1.0 {
                alpha0 = 1.0;
            }
            let old_n = (2.0 - alpha0) / alpha0;
            let new_n = (sc - fc) as f64 * ((old_n - 1.0) / (sc as f64 - 1.0)) + fc as f64;
            let mut alpha = 2.0 / (new_n + 1.0);
            if alpha < sc_floor {
                alpha = sc_floor;
            }
            if alpha > 1.0 {
                alpha = 1.0;
            }
            out[i] = frama_stable_update_f64_v2(close[i], out[i - 1], alpha);
        }
        out
    }

    fn frama_finite_segments_reference(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        window: usize,
        sc: usize,
        fc: usize,
    ) -> Vec<f64> {
        let mut out = vec![frama_canonical_nan_f64_v3(); close.len()];
        let mut cursor = 0usize;
        while cursor < close.len() {
            if !frama_is_finite_triplet_v3(high[cursor], low[cursor], close[cursor]) {
                cursor += 1;
                continue;
            }
            let segment_start = cursor;
            while cursor < close.len()
                && frama_is_finite_triplet_v3(high[cursor], low[cursor], close[cursor])
            {
                cursor += 1;
            }
            let segment_end = cursor;
            let even_window = if window & 1 == 1 { window + 1 } else { window };
            if segment_end - segment_start >= even_window {
                let segment = frama_direct_halves_reference(
                    &high[segment_start..segment_end],
                    &low[segment_start..segment_end],
                    &close[segment_start..segment_end],
                    window,
                    sc,
                    fc,
                );
                out[segment_start..segment_end].copy_from_slice(&segment);
            }
        }
        out
    }

    #[test]
    fn frama_f64_finite_segments_match_fresh_direct_runs_across_routes_and_windows() {
        const LEN: usize = 1_000;
        const QNAN: f64 = f64::from_bits(0x7ff8_0000_0000_0000);
        let mut high = Vec::with_capacity(LEN);
        let mut low = Vec::with_capacity(LEN);
        let mut close = Vec::with_capacity(LEN);
        for row in 0..LEN {
            let center = 1.075 + row as f64 * 0.000_000_7 + ((row % 17) as f64 - 8.0) * 0.000_003;
            high.push(center + 0.000_08 + (row % 7) as f64 * 0.000_001);
            low.push(center - 0.000_07 - (row % 5) as f64 * 0.000_001);
            close.push(center + ((row % 11) as f64 - 5.0) * 0.000_002);
        }
        high[250] = QNAN;
        close[600] = f64::INFINITY;
        low[850] = f64::NEG_INFINITY;

        for window in [7, 21, 34, 50, 100, 200] {
            let expected = frama_finite_segments_reference(&high, &low, &close, window, 300, 1);
            let params = FramaParams {
                window: Some(window),
                sc: Some(300),
                fc: Some(1),
            };
            let direct = frama_with_kernel(
                &FramaInput::from_slices(&high, &low, &close, params.clone()),
                Kernel::Scalar,
            )
            .expect("finite-segment fixture is a valid direct input");
            let batch = frama_batch_with_kernel(
                &high,
                &low,
                &close,
                &FramaBatchRange {
                    window: (window, window, 0),
                    sc: (300, 300, 0),
                    fc: (1, 1, 0),
                },
                Kernel::ScalarBatch,
            )
            .expect("finite-segment fixture is a valid batch input");
            let mut stream = FramaStream::try_new(params).expect("stream parameters are valid");
            let mut streamed = vec![frama_canonical_nan_f64_v3(); LEN];
            for row in 0..LEN {
                if let Some(value) = stream.update(high[row], low[row], close[row]) {
                    streamed[row] = value;
                }
            }

            for row in 0..LEN {
                let expected_bits = expected[row].to_bits();
                assert_eq!(
                    direct.values[row].to_bits(),
                    expected_bits,
                    "direct/deque mismatch at window {window}, row {row}"
                );
                assert_eq!(
                    batch.values[row].to_bits(),
                    expected_bits,
                    "batch mismatch at window {window}, row {row}"
                );
                assert_eq!(
                    streamed[row].to_bits(),
                    expected_bits,
                    "stream mismatch at window {window}, row {row}"
                );
            }
        }
    }

    #[test]
    fn frama_large_window_deque_matches_direct_halves() {
        const LEN: usize = 4096;
        const WINDOW: usize = 200;
        let waves = [
            0.000_041, -0.000_027, 0.000_013, -0.000_036, 0.000_022, -0.000_009, 0.000_033,
            -0.000_019, 0.000_006, -0.000_031, 0.000_017,
        ];
        let mut high = Vec::with_capacity(LEN);
        let mut low = Vec::with_capacity(LEN);
        let mut close = Vec::with_capacity(LEN);
        for row in 0..LEN {
            let drift = row as f64 * 0.000_000_7;
            let open = 1.075 + drift;
            let row_close = open + waves[row % waves.len()];
            high.push(open.max(row_close) + 0.000_08 + (row % 7) as f64 * 0.000_001);
            low.push(open.min(row_close) - 0.000_07 - (row % 5) as f64 * 0.000_001);
            close.push(row_close);
        }
        let final_row = LEN - 1;
        close[final_row] = f64::from_bits(close[final_row].to_bits() ^ 1);
        high[final_row] = high[final_row].max(close[final_row] + 0.000_001);
        low[final_row] = low[final_row].min(close[final_row] - 0.000_001);

        let expected = frama_direct_halves_reference(&high, &low, &close, WINDOW, 300, 1);
        let input = FramaInput::from_slices(
            &high,
            &low,
            &close,
            FramaParams {
                window: Some(WINDOW),
                sc: Some(300),
                fc: Some(1),
            },
        );
        let actual = frama_with_kernel(&input, Kernel::Scalar)
            .expect("large-window scalar route should accept the deterministic fixture");
        let batch = frama_batch_with_kernel(
            &high,
            &low,
            &close,
            &FramaBatchRange {
                window: (WINDOW, WINDOW, 0),
                sc: (300, 300, 0),
                fc: (1, 1, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("large-window batch route should accept the deterministic fixture");
        assert_eq!(batch.rows, 1);
        let batch_values = &batch.values[..LEN];

        for row in (WINDOW - 1)..LEN {
            assert_eq!(
                actual.values[row].to_bits(),
                expected[row].to_bits(),
                "large-window deque drifted from the direct halves at row {row}: actual={:#018x}, expected={:#018x}",
                actual.values[row].to_bits(),
                expected[row].to_bits(),
            );
            assert_eq!(
                batch_values[row].to_bits(),
                expected[row].to_bits(),
                "large-window batch deque drifted from the direct halves at row {row}",
            );
        }
    }

    #[test]
    fn frama_large_window_deque_supports_public_maximum_and_odd_rounding() {
        const LEN: usize = 1300;
        let close = (0..LEN)
            .map(|row| 10_000.0 - row as f64 * 0.25)
            .collect::<Vec<_>>();
        let high = close
            .iter()
            .enumerate()
            .map(|(row, value)| value + 1.0 + (row % 3) as f64 * 0.001)
            .collect::<Vec<_>>();
        let low = close
            .iter()
            .enumerate()
            .map(|(row, value)| value - 1.0 - (row % 5) as f64 * 0.001)
            .collect::<Vec<_>>();

        for requested_window in [1023, 1024] {
            let expected =
                frama_direct_halves_reference(&high, &low, &close, requested_window, 300, 1);
            let actual = frama_with_kernel(
                &FramaInput::from_slices(
                    &high,
                    &low,
                    &close,
                    FramaParams {
                        window: Some(requested_window),
                        sc: Some(300),
                        fc: Some(1),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("public maximum/evenized large window should remain supported");
            for row in 1023..LEN {
                assert_eq!(
                    actual.values[row].to_bits(),
                    expected[row].to_bits(),
                    "large-window deque drifted at requested window {requested_window}, row {row}",
                );
            }
        }
    }

    #[test]
    fn frama_evenized_window_cap_rejects_before_safe_route_allocation_or_dispatch() {
        const LEN: usize = 1026;
        assert_eq!(FRAMA_MAX_WINDOW, 1024);

        let high = vec![2.0; LEN];
        let low = vec![0.0; LEN];
        let close = vec![1.0; LEN];

        for window in [1025, usize::MAX] {
            let params = FramaParams {
                window: Some(window),
                sc: Some(300),
                fc: Some(1),
            };
            assert!(matches!(
                frama_with_kernel(
                    &FramaInput::from_slices(&high, &low, &close, params.clone()),
                    Kernel::Scalar,
                ),
                Err(FramaError::InvalidWindow { window: rejected, .. }) if rejected == window
            ));
            assert!(matches!(
                frama_scalar(&high, &low, &close, window, 300, 1, 0, LEN),
                Err(FramaError::InvalidWindow { window: rejected, .. }) if rejected == window
            ));

            let sweep = FramaBatchRange {
                window: (window, window, 0),
                sc: (300, 300, 0),
                fc: (1, 1, 0),
            };
            assert!(matches!(
                frama_batch_slice(&high, &low, &close, &sweep, Kernel::Scalar),
                Err(FramaError::InvalidWindow { window: rejected, .. }) if rejected == window
            ));

            let mut into = vec![0.25; LEN];
            assert!(matches!(
                frama_batch_inner_into(
                    &high,
                    &low,
                    &close,
                    &sweep,
                    Kernel::Scalar,
                    false,
                    &mut into,
                ),
                Err(FramaError::InvalidWindow { window: rejected, .. }) if rejected == window
            ));
            assert!(
                into.iter()
                    .all(|value| value.to_bits() == 0.25f64.to_bits())
            );

            assert!(matches!(
                FramaStream::try_new(params),
                Err(FramaError::InvalidWindow { window: rejected, .. }) if rejected == window
            ));
        }
    }

    #[test]
    fn frama_batch_admission_rejects_before_allocation_or_indexing() {
        const LEN: usize = 64;
        let sweep = FramaBatchRange {
            window: (4, 4, 0),
            sc: (300, 300, 0),
            fc: (1, 1, 0),
        };

        let all_nonfinite = vec![f64::NAN; LEN];
        for _ in 0..64 {
            assert!(matches!(
                frama_batch_slice(
                    &all_nonfinite,
                    &all_nonfinite,
                    &all_nonfinite,
                    &sweep,
                    Kernel::Scalar,
                ),
                Err(FramaError::AllValuesNaN)
            ));
        }

        let mut late_finite = vec![f64::NAN; LEN];
        late_finite[LEN - 1] = 1.0;
        assert!(matches!(
            frama_batch_slice(
                &late_finite,
                &late_finite,
                &late_finite,
                &sweep,
                Kernel::Scalar,
            ),
            Err(FramaError::NotEnoughValidData {
                needed: 4,
                valid: 1,
            })
        ));

        assert!(matches!(
            frama_batch_slice(
                &vec![2.0; LEN],
                &vec![0.0; LEN - 1],
                &vec![1.0; LEN],
                &sweep,
                Kernel::Scalar,
            ),
            Err(FramaError::MismatchedInputLength {
                high: LEN,
                low,
                close: LEN,
            }) if low == LEN - 1
        ));
    }

    fn check_frama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let default_params = FramaParams {
            window: None,
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_candles(&candles, default_params);
        let output = frama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_frama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = FramaInput::from_candles(&candles, FramaParams::default());
        let result = frama_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59337.23056930512,
            59321.607512374605,
            59286.677929994796,
            59268.00202402624,
            59160.03888720062,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] FRAMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_frama_zero_window(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = FramaParams {
            window: Some(0),
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_slices(&input_data, &input_data, &input_data, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FRAMA should fail with zero window",
            test_name
        );
        Ok(())
    }
    fn check_frama_window_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = FramaParams {
            window: Some(10),
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_slices(&data_small, &data_small, &data_small, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FRAMA should fail with window exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_frama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = FramaParams {
            window: Some(9),
            sc: None,
            fc: None,
        };
        let input = FramaInput::from_slices(&single_point, &single_point, &single_point, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] FRAMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_frama_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = FramaParams::default();
        let input = FramaInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_frama_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = FramaParams::default();
        let input = FramaInput::from_slices(&empty, &empty, &empty, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(matches!(res, Err(FramaError::EmptyInputData)));
        Ok(())
    }

    fn check_frama_mismatched_len(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let h = [1.0, 2.0, 3.0];
        let l = [1.0, 2.0];
        let c = [1.0, 2.0, 3.0];
        let params = FramaParams::default();
        let input = FramaInput::from_slices(&h, &l, &c, params);
        let res = frama_with_kernel(&input, kernel);
        assert!(matches!(res, Err(FramaError::MismatchedInputLength { .. })));
        Ok(())
    }

    fn check_frama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let params = FramaParams::default();
        let first_input = FramaInput::from_candles(&candles, params.clone());
        let first_res = frama_with_kernel(&first_input, kernel)?;

        let second_input = FramaInput::from_slices(
            &first_res.values,
            &first_res.values,
            &first_res.values,
            params,
        );
        let second_res = frama_with_kernel(&second_input, kernel)?;
        assert_eq!(first_res.values.len(), second_res.values.len());
        Ok(())
    }

    fn check_frama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = FramaInput::from_candles(&candles, FramaParams::default());
        let res = frama_with_kernel(&input, kernel)?;
        if res.values.len() > 240 {
            for (i, &v) in res.values[240..].iter().enumerate() {
                assert!(
                    !v.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_frama_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let high = candles.select_candle_field("high").unwrap();
        let low = candles.select_candle_field("low").unwrap();
        let close = candles.select_candle_field("close").unwrap();

        let data_len = high.len();
        let strat = (
            4usize..=64,
            50usize..500,
            1usize..50,
            0usize..data_len.saturating_sub(200),
            100usize..=200,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(window, sc, fc, start_idx, slice_len)| {
                let end_idx = (start_idx + slice_len).min(data_len);
                let actual_len = end_idx - start_idx;

                if actual_len < window * 2 {
                    return Ok(());
                }

                let high_slice = &high[start_idx..end_idx];
                let low_slice = &low[start_idx..end_idx];
                let close_slice = &close[start_idx..end_idx];

                let params = FramaParams {
                    window: Some(window),
                    sc: Some(sc),
                    fc: Some(fc),
                };

                let input = FramaInput::from_slices(high_slice, low_slice, close_slice, params);
                let result = frama_with_kernel(&input, kernel);

                prop_assert!(result.is_ok(), "FRAMA failed: {:?}", result.err());
                let FramaOutput { values: out } = result.unwrap();

                let FramaOutput { values: ref_out } =
                    frama_with_kernel(&input, Kernel::Scalar).unwrap();

                let actual_window = if window & 1 == 1 { window + 1 } else { window };

                let first_output_idx = actual_window - 1;

                for i in 0..first_output_idx.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                for i in first_output_idx..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    let all_high_max = high_slice
                        .iter()
                        .filter(|x| x.is_finite())
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let all_low_min = low_slice
                        .iter()
                        .filter(|x| x.is_finite())
                        .cloned()
                        .fold(f64::INFINITY, f64::min);

                    if all_high_max.is_finite() && all_low_min.is_finite() {
                        let tolerance = (all_high_max - all_low_min) * 0.01;
                        prop_assert!(
                            y.is_nan()
                                || (y >= all_low_min - tolerance && y <= all_high_max + tolerance),
                            "idx {}: {} not in overall range [{}, {}] with tolerance {}",
                            i,
                            y,
                            all_low_min,
                            all_high_max,
                            tolerance
                        );
                    }

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "NaN mismatch at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }

                    if fc >= sc && i > first_output_idx {
                        let change = (y - out[i - 1]).abs();
                        let price_change = (close_slice[i] - close_slice[i - 1]).abs();
                        prop_assert!(
							change <= price_change + 1e-6,
							"Unexpected large change at idx {} with fc >= sc: {} vs price change {}",
							i,
							change,
							price_change
						);
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }
    fn check_frama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let high = candles.select_candle_field("high").unwrap();
        let low = candles.select_candle_field("low").unwrap();
        let close = candles.select_candle_field("close").unwrap();
        let period = 10;
        let sc = 300;
        let fc = 1;
        let input = FramaInput::from_slices(
            high,
            low,
            close,
            FramaParams {
                window: Some(period),
                sc: Some(sc),
                fc: Some(fc),
            },
        );
        let batch_output = frama_with_kernel(&input, kernel)?.values;
        let mut stream = FramaStream::try_new(FramaParams {
            window: Some(period),
            sc: Some(sc),
            fc: Some(fc),
        })?;
        let mut stream_values = Vec::with_capacity(close.len());
        for ((&h, &l), &c) in high.iter().zip(low.iter()).zip(close.iter()) {
            match stream.update(h, l, c) {
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
                diff < 1e-7,
                "[{}] FRAMA streaming mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }
    fn check_frama_default_candles(test: &str, k: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(k, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let input = FramaInput::with_default_candles(&c);
        match input.data {
            FramaData::Candles { .. } => {}
            _ => panic!("Expected FramaData::Candles"),
        }
        let out = frama_with_kernel(&input, k)?;
        assert_eq!(out.values.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_frama_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_cases = vec![
            FramaParams::default(),
            FramaParams {
                window: Some(4),
                sc: Some(300),
                fc: Some(1),
            },
            FramaParams {
                window: Some(8),
                sc: Some(150),
                fc: Some(1),
            },
            FramaParams {
                window: Some(10),
                sc: Some(200),
                fc: Some(2),
            },
            FramaParams {
                window: Some(12),
                sc: Some(400),
                fc: Some(1),
            },
            FramaParams {
                window: Some(20),
                sc: Some(300),
                fc: Some(1),
            },
            FramaParams {
                window: Some(30),
                sc: Some(500),
                fc: Some(3),
            },
            FramaParams {
                window: Some(16),
                sc: Some(100),
                fc: Some(1),
            },
            FramaParams {
                window: Some(14),
                sc: Some(600),
                fc: Some(4),
            },
        ];

        for params in test_cases {
            let input = FramaInput::from_candles(&candles, params.clone());
            let output = frama_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params window={:?}, sc={:?}, fc={:?}",
                        test_name, val, bits, i, params.window, params.sc, params.fc
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params window={:?}, sc={:?}, fc={:?}",
                        test_name, val, bits, i, params.window, params.sc, params.fc
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params window={:?}, sc={:?}, fc={:?}",
                        test_name, val, bits, i, params.window, params.sc, params.fc
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_frama_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_frama_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                $(
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                )*
                $(
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }
    generate_all_frama_tests!(
        check_frama_partial_params,
        check_frama_accuracy,
        check_frama_zero_window,
        check_frama_window_exceeds_length,
        check_frama_very_small_dataset,
        check_frama_all_nan,
        check_frama_empty_input,
        check_frama_mismatched_len,
        check_frama_reinput,
        check_frama_nan_handling,
        check_frama_property,
        check_frama_streaming,
        check_frama_default_candles,
        check_frama_no_poison
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = FramaBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = FramaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59337.23056930512,
            59321.607512374605,
            59286.677929994796,
            59268.00202402624,
            59160.03888720062,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            ((4, 8, 2), (100, 300, 100), (1, 2, 1)),
            ((10, 20, 5), (200, 400, 100), (1, 3, 1)),
            ((20, 30, 5), (300, 600, 150), (1, 4, 1)),
            ((6, 12, 2), (150, 450, 50), (1, 2, 1)),
            ((8, 16, 2), (100, 500, 100), (1, 5, 1)),
        ];

        for (window_range, sc_range, fc_range) in test_configs {
            let output = FramaBatchBuilder::new()
                .kernel(kernel)
                .window_range(window_range.0, window_range.1, window_range.2)
                .sc_range(sc_range.0, sc_range.1, sc_range.2)
                .fc_range(fc_range.0, fc_range.1, fc_range.2)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let params = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: window={:?}, sc={:?}, fc={:?})",
                        test, val, bits, row, col, params.window, params.sc, params.fc
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: window={:?}, sc={:?}, fc={:?})",
                        test, val, bits, row, col, params.window, params.sc, params.fc
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: window={:?}, sc={:?}, fc={:?})",
                        test, val, bits, row, col, params.window, params.sc, params.fc
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
            paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_frama_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = FramaInput::with_default_candles(&candles);
        let baseline = frama(&input)?.values;

        let mut out = vec![0.0; candles.close.len()];

        {
            frama_into(&input, &mut out)?;
        }

        assert_eq!(out.len(), baseline.len());
        for i in 0..out.len() {
            let a = out[i];
            let b = baseline[i];
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN mismatch at index {}", i);
            } else {
                assert!(a == b, "Value mismatch at index {}: {} != {}", i, a, b);
            }
        }
        Ok(())
    }
}

#[inline]
pub fn frama_into_slice(
    dst: &mut [f64],
    input: &FramaInput,
    kern: Kernel,
) -> Result<(), FramaError> {
    let ((high, low, close), window, sc, fc, first, len, _warm_from_prepare, chosen) =
        frama_prepare(input, kern)?;

    if dst.len() != len {
        return Err(FramaError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let mut win = window;
    if win & 1 == 1 {
        win += 1;
    }
    let warm = first + win - 1;

    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }

    frama_compute_into(
        high, low, close, window, sc, fc, first, len, warm, chosen, dst,
    )?;

    Ok(())
}
