use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_kernel};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EhlersPmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersPmaOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EhlersPmaParams;

impl Default for EhlersPmaParams {
    #[inline]
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct EhlersPmaBatchRange {
    pub combos: usize,
}

impl Default for EhlersPmaBatchRange {
    #[inline]
    fn default() -> Self {
        Self { combos: 250 }
    }
}

#[inline]
pub fn expand_grid(range: &EhlersPmaBatchRange) -> Vec<EhlersPmaParams> {
    let count = range.combos;
    if count == 0 {
        return Vec::new();
    }
    core::iter::repeat(EhlersPmaParams::default())
        .take(count)
        .collect()
}

#[derive(Debug, Clone)]
pub struct EhlersPmaInput<'a> {
    pub data: EhlersPmaData<'a>,
    pub params: EhlersPmaParams,
}

impl<'a> EhlersPmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersPmaParams) -> Self {
        Self {
            data: EhlersPmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EhlersPmaParams) -> Self {
        Self {
            data: EhlersPmaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersPmaParams::default())
    }
}

impl<'a> AsRef<[f64]> for EhlersPmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersPmaData::Slice(s) => s,
            EhlersPmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersPmaBuilder {
    kernel: Kernel,
}

impl Default for EhlersPmaBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersPmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EhlersPmaOutput, EhlersPmaError> {
        let i = EhlersPmaInput::from_candles(c, "close", EhlersPmaParams::default());
        ehlers_pma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersPmaOutput, EhlersPmaError> {
        let i = EhlersPmaInput::from_slice(d, EhlersPmaParams::default());
        ehlers_pma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersPmaStream, EhlersPmaError> {
        EhlersPmaStream::try_new(EhlersPmaParams::default())
    }
}

#[derive(Debug, Error)]
pub enum EhlersPmaError {
    #[error("ehlers_pma: Input data slice is empty.")]
    EmptyInputData,

    #[error("ehlers_pma: All values are NaN.")]
    AllValuesNaN,

    #[error("ehlers_pma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("ehlers_pma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("ehlers_pma: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("ehlers_pma: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error("ehlers_pma: invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("ehlers_pma: size overflow computing rows*cols: rows = {rows}, cols = {cols}")]
    SizeOverflow { rows: usize, cols: usize },
}

#[inline]
pub fn ehlers_pma(input: &EhlersPmaInput) -> Result<EhlersPmaOutput, EhlersPmaError> {
    ehlers_pma_with_kernel(input, Kernel::Auto)
}

pub fn ehlers_pma_with_kernel(
    input: &EhlersPmaInput,
    kernel: Kernel,
) -> Result<EhlersPmaOutput, EhlersPmaError> {
    let data = input.as_ref();
    let len = data.len();

    if len == 0 {
        return Err(EhlersPmaError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersPmaError::AllValuesNaN)?;

    const MIN_REQUIRED: usize = 14;
    if len - first_valid < MIN_REQUIRED {
        return Err(EhlersPmaError::NotEnoughValidData {
            needed: MIN_REQUIRED,
            valid: len - first_valid,
        });
    }

    let warm_wma1 = first_valid + 7;
    let warm_wma2 = first_valid + 13;
    let warm_predict = warm_wma2;
    let warm_trigger = warm_wma2 + 3;

    let mut predict = alloc_with_nan_prefix(len, warm_predict);
    let mut trigger = alloc_with_nan_prefix(len, warm_trigger);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match chosen {
            Kernel::Avx512 | Kernel::Avx512Batch => ehlers_pma_avx512(
                data,
                &mut predict,
                &mut trigger,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            ),
            Kernel::Avx2 | Kernel::Avx2Batch => ehlers_pma_avx2(
                data,
                &mut predict,
                &mut trigger,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            ),
            Kernel::Scalar | Kernel::ScalarBatch => ehlers_pma_scalar_direct(
                data,
                &mut predict,
                &mut trigger,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            ),
            Kernel::Auto => unreachable!(),
        }

        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            let _ = chosen;
            ehlers_pma_scalar_direct(
                data,
                &mut predict,
                &mut trigger,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            );
        }
    }

    Ok(EhlersPmaOutput { predict, trigger })
}

#[inline]
fn ehlers_pma_scalar_direct(
    data: &[f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_trigger: usize,
) {
    debug_assert_eq!(predict.len(), data.len());
    debug_assert_eq!(trigger.len(), data.len());

    let len = data.len();
    if warm_wma1 >= len {
        return;
    }

    let inv28 = 1.0 / 28.0;
    let inv10 = 1.0 / 10.0;

    let mut w_ring = [0.0f64; 7];
    let mut w_head = 0usize;

    for i in warm_wma1..len {
        let w1 = (7.0 * data[i - 1]
            + 6.0 * data[i - 2]
            + 5.0 * data[i - 3]
            + 4.0 * data[i - 4]
            + 3.0 * data[i - 5]
            + 2.0 * data[i - 6]
            + 1.0 * data[i - 7])
            * inv28;

        w_ring[w_head] = w1;
        w_head += 1;
        if w_head == 7 {
            w_head = 0;
        }

        if i < warm_wma2 {
            continue;
        }

        let k0 = if w_head == 0 { 6 } else { w_head - 1 };
        let k1 = if k0 == 0 { 6 } else { k0 - 1 };
        let k2 = if k1 == 0 { 6 } else { k1 - 1 };
        let k3 = if k2 == 0 { 6 } else { k2 - 1 };
        let k4 = if k3 == 0 { 6 } else { k3 - 1 };
        let k5 = if k4 == 0 { 6 } else { k4 - 1 };
        let k6 = if k5 == 0 { 6 } else { k5 - 1 };

        let w2 = (7.0 * w_ring[k0]
            + 6.0 * w_ring[k1]
            + 5.0 * w_ring[k2]
            + 4.0 * w_ring[k3]
            + 3.0 * w_ring[k4]
            + 2.0 * w_ring[k5]
            + 1.0 * w_ring[k6])
            * inv28;

        let p = 2.0 * w1 - w2;
        predict[i] = p;

        if i >= warm_trigger {
            trigger[i] =
                (4.0 * p + 3.0 * predict[i - 1] + 2.0 * predict[i - 2] + 1.0 * predict[i - 3])
                    * inv10;
        }
    }
}

#[inline]
pub fn ehlers_pma_scalar(
    data: &[f64],
    wma1: &mut [f64],
    wma2: &mut [f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_predict: usize,
    warm_trigger: usize,
) {
    let len = data.len();
    if warm_wma1 >= len {
        return;
    }

    let inv28 = 1.0 / 28.0;
    let inv10 = 1.0 / 10.0;

    for i in warm_wma1..len {
        wma1[i] = (7.0 * data[i - 1]
            + 6.0 * data[i - 2]
            + 5.0 * data[i - 3]
            + 4.0 * data[i - 4]
            + 3.0 * data[i - 5]
            + 2.0 * data[i - 6]
            + 1.0 * data[i - 7])
            * inv28;
    }

    for i in warm_wma2..len {
        wma2[i] = (7.0 * wma1[i]
            + 6.0 * wma1[i - 1]
            + 5.0 * wma1[i - 2]
            + 4.0 * wma1[i - 3]
            + 3.0 * wma1[i - 4]
            + 2.0 * wma1[i - 5]
            + 1.0 * wma1[i - 6])
            * inv28;
    }

    for i in warm_predict..len {
        predict[i] = 2.0 * wma1[i] - wma2[i];
    }

    for i in warm_trigger..len {
        trigger[i] =
            (4.0 * predict[i] + 3.0 * predict[i - 1] + 2.0 * predict[i - 2] + 1.0 * predict[i - 3])
                * inv10;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn ehlers_pma_avx2(
    data: &[f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_trigger: usize,
) {
    ehlers_pma_scalar_direct(data, predict, trigger, warm_wma1, warm_wma2, warm_trigger)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ehlers_pma_avx512(
    data: &[f64],
    predict: &mut [f64],
    trigger: &mut [f64],
    warm_wma1: usize,
    warm_wma2: usize,
    warm_trigger: usize,
) {
    ehlers_pma_scalar_direct(data, predict, trigger, warm_wma1, warm_wma2, warm_trigger)
}

#[inline]
pub fn ehlers_pma_into_flat_with_kernel(
    out: &mut [f64],
    input: &EhlersPmaInput,
    kernel: Kernel,
) -> Result<(usize, usize), EhlersPmaError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EhlersPmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersPmaError::AllValuesNaN)?;
    const MIN_REQUIRED: usize = 14;
    if len - first < MIN_REQUIRED {
        return Err(EhlersPmaError::NotEnoughValidData {
            needed: MIN_REQUIRED,
            valid: len - first,
        });
    }

    let rows = 2usize;
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or(EhlersPmaError::SizeOverflow { rows, cols })?;
    if out.len() != expected {
        return Err(EhlersPmaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let (predict_flat, trigger_flat) = out.split_at_mut(cols);

    let warm_wma1 = first + 7;
    let warm_wma2 = first + 13;
    let warm_predict = warm_wma2;
    let warm_trigger = warm_wma2 + 3;

    for v in &mut predict_flat[..warm_predict.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut trigger_flat[..warm_trigger.min(len)] {
        *v = f64::NAN;
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match chosen {
            Kernel::Avx512 | Kernel::Avx512Batch => ehlers_pma_avx512(
                data,
                predict_flat,
                trigger_flat,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            ),
            Kernel::Avx2 | Kernel::Avx2Batch => ehlers_pma_avx2(
                data,
                predict_flat,
                trigger_flat,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            ),
            Kernel::Scalar | Kernel::ScalarBatch => ehlers_pma_scalar_direct(
                data,
                predict_flat,
                trigger_flat,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            ),
            Kernel::Auto => unreachable!(),
        }

        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            let _ = chosen;
            ehlers_pma_scalar_direct(
                data,
                predict_flat,
                trigger_flat,
                warm_wma1,
                warm_wma2,
                warm_trigger,
            );
        }
    }

    Ok((rows, cols))
}

#[inline]
pub fn ehlers_pma_into_flat(
    out: &mut [f64],
    input: &EhlersPmaInput,
) -> Result<(usize, usize), EhlersPmaError> {
    ehlers_pma_into_flat_with_kernel(out, input, Kernel::Auto)
}

#[inline]
pub fn ehlers_pma_into_slices_with_kernel(
    predict: &mut [f64],
    trigger: &mut [f64],
    input: &EhlersPmaInput,
    kernel: Kernel,
) -> Result<(), EhlersPmaError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EhlersPmaError::EmptyInputData);
    }
    if predict.len() != len || trigger.len() != len {
        return Err(EhlersPmaError::OutputLengthMismatch {
            expected: len,
            got: predict.len().min(trigger.len()),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersPmaError::AllValuesNaN)?;
    const MIN_REQUIRED: usize = 14;
    if len - first < MIN_REQUIRED {
        return Err(EhlersPmaError::NotEnoughValidData {
            needed: MIN_REQUIRED,
            valid: len - first,
        });
    }

    let warm_wma1 = first + 7;
    let warm_wma2 = first + 13;
    let warm_predict = warm_wma2;
    let warm_trigger = warm_wma2 + 3;

    for v in &mut predict[..warm_predict.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut trigger[..warm_trigger.min(len)] {
        *v = f64::NAN;
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        match chosen {
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ehlers_pma_avx512(data, predict, trigger, warm_wma1, warm_wma2, warm_trigger)
            }
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ehlers_pma_avx2(data, predict, trigger, warm_wma1, warm_wma2, warm_trigger)
            }
            Kernel::Scalar | Kernel::ScalarBatch => {
                ehlers_pma_scalar_direct(data, predict, trigger, warm_wma1, warm_wma2, warm_trigger)
            }
            Kernel::Auto => unreachable!(),
        }

        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            let _ = chosen;
            ehlers_pma_scalar_direct(data, predict, trigger, warm_wma1, warm_wma2, warm_trigger);
        }
    }
    Ok(())
}

#[inline]
pub fn ehlers_pma_into_slices(
    predict: &mut [f64],
    trigger: &mut [f64],
    input: &EhlersPmaInput,
) -> Result<(), EhlersPmaError> {
    ehlers_pma_into_slices_with_kernel(predict, trigger, input, Kernel::Auto)
}

#[inline]
pub fn ehlers_pma_into(
    input: &EhlersPmaInput,
    predict: &mut [f64],
    trigger: &mut [f64],
) -> Result<(), EhlersPmaError> {
    ehlers_pma_into_slices_with_kernel(predict, trigger, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct EhlersPmaStream {
    prev: Option<f64>,

    x_ring: [f64; 7],
    w_ring: [f64; 7],
    p_ring: [f64; 4],

    x_head: usize,
    w_head: usize,
    p_head: usize,

    filled_x: usize,
    filled_w: usize,
    filled_p: usize,

    A: f64,
    S: f64,

    A1: f64,
    S1: f64,

    A2: f64,
    T: f64,
}

impl EhlersPmaStream {
    #[inline]
    pub fn try_new(_params: EhlersPmaParams) -> Result<Self, EhlersPmaError> {
        Ok(Self {
            prev: None,
            x_ring: [0.0; 7],
            w_ring: [0.0; 7],
            p_ring: [0.0; 4],
            x_head: 0,
            w_head: 0,
            p_head: 0,
            filled_x: 0,
            filled_w: 0,
            filled_p: 0,
            A: 0.0,
            S: 0.0,
            A1: 0.0,
            S1: 0.0,
            A2: 0.0,
            T: 0.0,
        })
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if value.is_nan() {
            return None;
        }

        let src_lag = match self.prev {
            None => {
                self.prev = Some(value);
                return None;
            }
            Some(p) => {
                self.prev = Some(value);
                p
            }
        };

        const INV_28: f64 = 1.0 / 28.0;
        const INV_10: f64 = 1.0 / 10.0;

        if self.filled_x < 7 {
            self.x_ring[self.x_head] = src_lag;
            self.x_head += 1;
            if self.x_head == 7 {
                self.x_head = 0;
            }
            self.filled_x += 1;

            if self.filled_x < 7 {
                return None;
            }

            let x = &self.x_ring;
            self.A = ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + x[6]);

            self.S = 7.0 * x[6]
                + 6.0 * x[5]
                + 5.0 * x[4]
                + 4.0 * x[3]
                + 3.0 * x[2]
                + 2.0 * x[1]
                + 1.0 * x[0];

            let w1 = self.S * INV_28;

            let old_A1 = self.A1;
            let w_old = self.w_ring[self.w_head];
            self.S1 = self.S1 + 7.0 * w1 - old_A1;
            self.A1 = self.A1 + w1 - w_old;

            self.w_ring[self.w_head] = w1;
            self.w_head += 1;
            if self.w_head == 7 {
                self.w_head = 0;
            }
            self.filled_w = 1;

            return None;
        }

        let x_old = self.x_ring[self.x_head];
        let old_A = self.A;
        self.A = self.A + src_lag - x_old;
        self.S = self.S + 7.0 * src_lag - old_A;
        self.x_ring[self.x_head] = src_lag;
        self.x_head += 1;
        if self.x_head == 7 {
            self.x_head = 0;
        }

        let i0 = if self.x_head == 0 { 6 } else { self.x_head - 1 };
        let i1 = if i0 == 0 { 6 } else { i0 - 1 };
        let i2 = if i1 == 0 { 6 } else { i1 - 1 };
        let i3 = if i2 == 0 { 6 } else { i2 - 1 };
        let i4 = if i3 == 0 { 6 } else { i3 - 1 };
        let i5 = if i4 == 0 { 6 } else { i4 - 1 };
        let i6 = if i5 == 0 { 6 } else { i5 - 1 };
        let w1_num = 7.0 * self.x_ring[i0]
            + 6.0 * self.x_ring[i1]
            + 5.0 * self.x_ring[i2]
            + 4.0 * self.x_ring[i3]
            + 3.0 * self.x_ring[i4]
            + 2.0 * self.x_ring[i5]
            + 1.0 * self.x_ring[i6];
        let w1 = w1_num * INV_28;

        let old_A1 = self.A1;
        let w_old = self.w_ring[self.w_head];
        self.S1 = self.S1 + 7.0 * w1 - old_A1;
        self.A1 = self.A1 + w1 - w_old;
        self.w_ring[self.w_head] = w1;
        self.w_head += 1;
        if self.w_head == 7 {
            self.w_head = 0;
        }
        if self.filled_w < 7 {
            self.filled_w += 1;
        }
        if self.filled_w < 7 {
            return None;
        }

        let k0 = if self.w_head == 0 { 6 } else { self.w_head - 1 };
        let k1 = if k0 == 0 { 6 } else { k0 - 1 };
        let k2 = if k1 == 0 { 6 } else { k1 - 1 };
        let k3 = if k2 == 0 { 6 } else { k2 - 1 };
        let k4 = if k3 == 0 { 6 } else { k3 - 1 };
        let k5 = if k4 == 0 { 6 } else { k4 - 1 };
        let k6 = if k5 == 0 { 6 } else { k5 - 1 };
        let w2_num = 7.0 * self.w_ring[k0]
            + 6.0 * self.w_ring[k1]
            + 5.0 * self.w_ring[k2]
            + 4.0 * self.w_ring[k3]
            + 3.0 * self.w_ring[k4]
            + 2.0 * self.w_ring[k5]
            + 1.0 * self.w_ring[k6];
        let w2 = w2_num * INV_28;
        let predict = 2.0 * w1 - w2;

        let old_A2 = self.A2;
        let p_old = self.p_ring[self.p_head];
        self.T = self.T + 4.0 * predict - old_A2;
        self.A2 = self.A2 + predict - p_old;
        self.p_ring[self.p_head] = predict;
        self.p_head += 1;
        if self.p_head == 4 {
            self.p_head = 0;
        }

        if self.filled_p < 4 {
            self.filled_p += 1;
        }
        if self.filled_p < 4 {
            return Some((predict, f64::NAN));
        }

        let j0 = if self.p_head == 0 { 3 } else { self.p_head - 1 };
        let j1 = if j0 == 0 { 3 } else { j0 - 1 };
        let j2 = if j1 == 0 { 3 } else { j1 - 1 };
        let j3 = if j2 == 0 { 3 } else { j2 - 1 };
        let trigger = (4.0 * self.p_ring[j0]
            + 3.0 * self.p_ring[j1]
            + 2.0 * self.p_ring[j2]
            + 1.0 * self.p_ring[j3])
            * INV_10;
        Some((predict, trigger))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.prev = None;
        self.x_ring = [0.0; 7];
        self.w_ring = [0.0; 7];
        self.p_ring = [0.0; 4];
        self.x_head = 0;
        self.w_head = 0;
        self.p_head = 0;
        self.filled_x = 0;
        self.filled_w = 0;
        self.filled_p = 0;
        self.A = 0.0;
        self.S = 0.0;
        self.A1 = 0.0;
        self.S1 = 0.0;
        self.A2 = 0.0;
        self.T = 0.0;
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct RollingLwma<const N: usize> {
    buf: [f64; N],
    head: usize,
    filled: usize,
    total: f64,
    num: f64,
}

#[allow(dead_code)]
impl<const N: usize> RollingLwma<N> {
    #[inline]
    fn new() -> Self {
        Self {
            buf: [f64::NAN; N],
            head: 0,
            filled: 0,
            total: 0.0,
            num: 0.0,
        }
    }
    #[inline]
    fn reset(&mut self) {
        self.buf.fill(f64::NAN);
        self.head = 0;
        self.filled = 0;
        self.total = 0.0;
        self.num = 0.0;
    }

    #[inline]
    fn push(&mut self, x: f64) -> Option<f64> {
        if self.filled < N {
            self.buf[self.head] = x;
            self.head = (self.head + 1) % N;
            self.filled += 1;
            if self.filled == N {
                let mut w = 1.0;
                let mut acc = 0.0;
                let mut sum = 0.0;
                let mut i = self.head;
                for _ in 0..N {
                    let v = self.buf[i];
                    acc += w * v;
                    sum += v;
                    w += 1.0;
                    i += 1;
                    if i == N {
                        i = 0;
                    }
                }
                self.num = acc;
                self.total = sum;
                let den = (N * (N + 1) / 2) as f64;
                return Some(self.num / den);
            }
            return None;
        }

        let x_old = self.buf[self.head];
        let total_old = self.total;
        self.buf[self.head] = x;
        self.head = (self.head + 1) % N;

        self.num = self.num + (N as f64) * x - total_old;
        self.total = total_old + x - x_old;

        let den = (N * (N + 1) / 2) as f64;
        Some(self.num / den)
    }
}

#[inline]
fn usize_range_len(range: (usize, usize, usize)) -> Result<usize, EhlersPmaError> {
    let (start, end, step) = range;
    if step == 0 {
        return Ok(1);
    }
    let lo = start.min(end);
    let hi = start.max(end);
    let span = hi.saturating_sub(lo);
    let n = span / step + 1;
    if n == 0 {
        return Err(EhlersPmaError::InvalidRange { start, end, step });
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_ehlers_pma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let input = EhlersPmaInput::from_candles(&candles, "close", EhlersPmaParams::default());

        let out = ehlers_pma_with_kernel(&input, kernel)?;

        let expected_predict_last_five = [
            59161.97066327,
            59240.51785714,
            59260.29846939,
            59225.19005102,
            59192.78443878,
        ];
        let expected_trigger_last_five = [
            59020.56403061,
            59141.96938776,
            59214.56709184,
            59232.46619898,
            59220.78227041,
        ];

        let start = out.predict.len().saturating_sub(5);
        for (i, &val) in out.predict[start..].iter().enumerate() {
            let diff = (val - expected_predict_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Ehlers PMA predict {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_predict_last_five[i]
            );
        }

        for (i, &val) in out.trigger[start..].iter().enumerate() {
            let diff = (val - expected_trigger_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Ehlers PMA trigger {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_trigger_last_five[i]
            );
        }

        Ok(())
    }

    fn check_ehlers_pma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = EhlersPmaInput::with_default_candles(&candles);
        match input.data {
            EhlersPmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersPmaData::Candles"),
        }
        let output = ehlers_pma_with_kernel(&input, kernel)?;
        assert_eq!(output.predict.len(), candles.close.len());
        assert_eq!(output.trigger.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_pma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EhlersPmaInput::from_slice(&empty, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersPmaError::EmptyInputData)),
            "[{}] Ehlers PMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 20];
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersPmaError::AllValuesNaN)),
            "[{}] Ehlers PMA should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_insufficient_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersPmaError::NotEnoughValidData { .. })),
            "[{}] Ehlers PMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let input = EhlersPmaInput::from_slice(&single_point, EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Ehlers PMA should fail with single data point",
            test_name
        );

        let two_points = [42.0, 43.0];
        let input2 = EhlersPmaInput::from_slice(&two_points, EhlersPmaParams::default());
        let res2 = ehlers_pma_with_kernel(&input2, kernel);
        assert!(
            res2.is_err(),
            "[{}] Ehlers PMA should fail with only two data points",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_pma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = EhlersPmaInput::from_candles(&candles, "hl2", EhlersPmaParams::default());
        let res = ehlers_pma_with_kernel(&input, kernel)?;

        assert_eq!(res.predict.len(), candles.close.len());
        assert_eq!(res.trigger.len(), candles.close.len());

        if res.predict.len() > 20 {
            for (i, &val) in res.predict[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN in predict at index {}",
                    test_name,
                    20 + i
                );
            }
        }
        if res.trigger.len() > 20 {
            for (i, &val) in res.trigger[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN in trigger at index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }

    fn check_ehlers_pma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let hl2: Vec<f64> = candles
            .high
            .iter()
            .zip(candles.low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        let input = EhlersPmaInput::from_slice(&hl2, EhlersPmaParams::default());
        let batch_output = ehlers_pma_with_kernel(&input, kernel)?;

        let mut stream = EhlersPmaStream::try_new(EhlersPmaParams::default())?;
        let mut stream_predict = Vec::with_capacity(hl2.len());
        let mut stream_trigger = Vec::with_capacity(hl2.len());

        for &value in &hl2 {
            match stream.update(value) {
                Some((p, t)) => {
                    stream_predict.push(p);
                    stream_trigger.push(t);
                }
                None => {
                    stream_predict.push(f64::NAN);
                    stream_trigger.push(f64::NAN);
                }
            }
        }

        assert_eq!(batch_output.predict.len(), stream_predict.len());
        assert_eq!(batch_output.trigger.len(), stream_trigger.len());

        for (i, ((&bp, &bt), (&sp, &st))) in batch_output
            .predict
            .iter()
            .zip(batch_output.trigger.iter())
            .zip(stream_predict.iter().zip(stream_trigger.iter()))
            .enumerate()
        {
            if bp.is_nan() && sp.is_nan() {
                continue;
            }
            if bt.is_nan() && st.is_nan() {
                continue;
            }

            let predict_diff = (bp - sp).abs();
            let trigger_diff = (bt - st).abs();

            assert!(
                predict_diff < 1e-9,
                "[{}] Predict streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                bp,
                sp,
                predict_diff
            );
            assert!(
                trigger_diff < 1e-9,
                "[{}] Trigger streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                bt,
                st,
                trigger_diff
            );
        }
        Ok(())
    }

    fn check_ehlers_pma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let hl2: Vec<f64> = candles
            .high
            .iter()
            .zip(candles.low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        let first_input = EhlersPmaInput::from_slice(&hl2, EhlersPmaParams::default());
        let first_result = ehlers_pma_with_kernel(&first_input, kernel)?;

        let second_input =
            EhlersPmaInput::from_slice(&first_result.predict, EhlersPmaParams::default());
        let second_result = ehlers_pma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.predict.len(), first_result.predict.len());
        assert_eq!(second_result.trigger.len(), first_result.trigger.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ehlers_pma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let input = EhlersPmaInput::from_candles(&candles, "hl2", EhlersPmaParams::default());
        let output = ehlers_pma_with_kernel(&input, kernel)?;

        for (arr_name, arr) in [
            ("predict", &output.predict[..]),
            ("trigger", &output.trigger[..]),
        ] {
            for (i, &val) in arr.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }
                let bits = val.to_bits();

                if bits == 0x1111_1111_1111_1111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in {} at index {}",
                        test_name, val, bits, arr_name, i
                    );
                }
                if bits == 0x2222_2222_2222_2222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in {} at index {}",
                        test_name, val, bits, arr_name, i
                    );
                }
                if bits == 0x3333_3333_3333_3333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in {} at index {}",
                        test_name, val, bits, arr_name, i
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ehlers_pma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_ehlers_pma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = prop::collection::vec(
            (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
            14..400,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |data| {
                let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());

                let result = ehlers_pma_with_kernel(&input, kernel).unwrap();
                let ref_result = ehlers_pma_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(result.predict.len(), data.len());
                prop_assert_eq!(result.trigger.len(), data.len());

                for i in 0..data.len() {
                    let p = result.predict[i];
                    let t = result.trigger[i];
                    let ref_p = ref_result.predict[i];
                    let ref_t = ref_result.trigger[i];

                    if !p.is_finite() || !ref_p.is_finite() {
                        prop_assert_eq!(
                            p.to_bits(),
                            ref_p.to_bits(),
                            "Predict finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            p,
                            ref_p
                        );
                        continue;
                    }
                    if !t.is_finite() || !ref_t.is_finite() {
                        prop_assert_eq!(
                            t.to_bits(),
                            ref_t.to_bits(),
                            "Trigger finite/NaN mismatch at idx {}: {} vs {}",
                            i,
                            t,
                            ref_t
                        );
                        continue;
                    }

                    let p_ulp_diff = p.to_bits().abs_diff(ref_p.to_bits());
                    let t_ulp_diff = t.to_bits().abs_diff(ref_t.to_bits());

                    prop_assert!(
                        (p - ref_p).abs() <= 1e-9 || p_ulp_diff <= 4,
                        "Predict mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        p,
                        ref_p,
                        p_ulp_diff
                    );
                    prop_assert!(
                        (t - ref_t).abs() <= 1e-9 || t_ulp_diff <= 4,
                        "Trigger mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        t,
                        ref_t,
                        t_ulp_diff
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_ehlers_pma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512);
                    }
                )*
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    fn check_ehlers_pma_invalid_output_len(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let data = vec![1.0; 20];
        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let mut p = vec![0.0; 19];
        let mut t = vec![0.0; 20];
        let res = ehlers_pma_into_slices(&mut p, &mut t, &input);
        assert!(
            matches!(res, Err(EhlersPmaError::OutputLengthMismatch { .. })),
            "[{}] expected OutputLengthMismatch",
            test_name
        );
        Ok(())
    }

    generate_all_ehlers_pma_tests!(
        check_ehlers_pma_accuracy,
        check_ehlers_pma_default_candles,
        check_ehlers_pma_empty_input,
        check_ehlers_pma_all_nan,
        check_ehlers_pma_insufficient_data,
        check_ehlers_pma_very_small_dataset,
        check_ehlers_pma_nan_handling,
        check_ehlers_pma_streaming,
        check_ehlers_pma_reinput,
        check_ehlers_pma_no_poison,
        check_ehlers_pma_invalid_output_len
    );

    #[cfg(feature = "proptest")]
    generate_all_ehlers_pma_tests!(check_ehlers_pma_property);

    #[test]
    fn test_ehlers_pma_basic() {
        let data = vec![
            59161.97066327,
            59240.51785714,
            59260.29846939,
            59225.19005102,
            59192.78443878,
            59200.0,
            59180.0,
            59220.0,
            59250.0,
            59230.0,
            59210.0,
            59240.0,
            59260.0,
            59280.0,
            59270.0,
            59250.0,
            59300.0,
        ];

        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let result = ehlers_pma(&input).unwrap();

        assert_eq!(result.predict.len(), data.len());
        assert_eq!(result.trigger.len(), data.len());

        for i in 0..13 {
            assert!(result.predict[i].is_nan());
        }

        for i in 0..16 {
            assert!(result.trigger[i].is_nan());
        }

        assert!(!result.predict[13].is_nan());
        assert!(!result.trigger[16].is_nan());
    }

    #[test]
    fn test_ehlers_pma_into_flat() {
        let data = vec![
            59161.97066327,
            59240.51785714,
            59260.29846939,
            59225.19005102,
            59192.78443878,
            59200.0,
            59180.0,
            59220.0,
            59250.0,
            59230.0,
            59210.0,
            59240.0,
            59260.0,
            59280.0,
            59270.0,
            59250.0,
            59300.0,
        ];

        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());
        let mut output = vec![0.0; data.len() * 2];
        let (rows, cols) = ehlers_pma_into_flat(&mut output, &input).unwrap();

        assert_eq!(rows, 2);
        assert_eq!(cols, data.len());

        let (predict_flat, trigger_flat) = output.split_at(data.len());

        for i in 0..13 {
            assert!(predict_flat[i].is_nan());
        }

        for i in 0..16 {
            assert!(trigger_flat[i].is_nan());
        }
    }

    #[test]
    fn check_ehlers_pma_into_slices_noalloc() -> Result<(), Box<dyn std::error::Error>> {
        use crate::utilities::data_loader::read_candles_from_vortex;
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let input = EhlersPmaInput::with_default_candles(&c);

        let batch = ehlers_pma(&input).unwrap();

        let mut p = vec![0.0; c.close.len()];
        let mut t = vec![0.0; c.close.len()];
        ehlers_pma_into_slices(&mut p, &mut t, &input).unwrap();

        assert_eq!(p.len(), batch.predict.len());
        assert_eq!(t.len(), batch.trigger.len());
        for i in 0..p.len() {
            let (a, b) = (p[i], batch.predict[i]);
            if a.is_nan() || b.is_nan() {
                assert_eq!(a.to_bits(), b.to_bits());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
            let (a2, b2) = (t[i], batch.trigger[i]);
            if a2.is_nan() || b2.is_nan() {
                assert_eq!(a2.to_bits(), b2.to_bits());
            } else {
                assert!((a2 - b2).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn test_ehlers_pma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let len = 256usize;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let v = 1000.0 + (i as f64) * 0.25 + ((i % 7) as f64 - 3.0) * 0.75;
            data.push(v);
        }

        let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams::default());

        let baseline = ehlers_pma(&input)?;

        let mut predict = vec![0.0; len];
        let mut trigger = vec![0.0; len];
        ehlers_pma_into(&input, &mut predict, &mut trigger)?;

        assert_eq!(predict.len(), baseline.predict.len());
        assert_eq!(trigger.len(), baseline.trigger.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(predict[i], baseline.predict[i]),
                "predict mismatch at {}: got {}, expected {}",
                i,
                predict[i],
                baseline.predict[i]
            );
            assert!(
                eq_or_both_nan(trigger[i], baseline.trigger[i]),
                "trigger mismatch at {}: got {}, expected {}",
                i,
                trigger[i],
                baseline.trigger[i]
            );
        }

        Ok(())
    }
}
