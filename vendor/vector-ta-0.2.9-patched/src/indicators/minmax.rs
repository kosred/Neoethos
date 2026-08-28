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
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MinmaxData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MinmaxOutput {
    pub is_min: Vec<f64>,
    pub is_max: Vec<f64>,
    pub last_min: Vec<f64>,
    pub last_max: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MinmaxParams {
    pub order: Option<usize>,
}

impl Default for MinmaxParams {
    fn default() -> Self {
        Self { order: Some(3) }
    }
}

#[derive(Debug, Clone)]
pub struct MinmaxInput<'a> {
    pub data: MinmaxData<'a>,
    pub params: MinmaxParams,
}

impl<'a> MinmaxInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        params: MinmaxParams,
    ) -> Self {
        Self {
            data: MinmaxData::Candles {
                candles,
                high_src,
                low_src,
            },
            params,
        }
    }
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MinmaxParams) -> Self {
        Self {
            data: MinmaxData::Slices { high, low },
            params,
        }
    }
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "high", "low", MinmaxParams::default())
    }
    pub fn get_order(&self) -> usize {
        self.params.order.unwrap_or(3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MinmaxBuilder {
    order: Option<usize>,
    kernel: Kernel,
}

impl Default for MinmaxBuilder {
    fn default() -> Self {
        Self {
            order: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MinmaxBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn order(mut self, n: usize) -> Self {
        self.order = Some(n);
        self
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply(self, candles: &Candles) -> Result<MinmaxOutput, MinmaxError> {
        let params = MinmaxParams { order: self.order };
        let input = MinmaxInput::from_candles(candles, "high", "low", params);
        minmax_with_kernel(&input, self.kernel)
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MinmaxOutput, MinmaxError> {
        let params = MinmaxParams { order: self.order };
        let input = MinmaxInput::from_slices(high, low, params);
        minmax_with_kernel(&input, self.kernel)
    }
    pub fn into_stream(self) -> Result<MinmaxStream, MinmaxError> {
        let params = MinmaxParams { order: self.order };
        MinmaxStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum MinmaxError {
    #[error("minmax: Empty data provided.")]
    EmptyInputData,
    #[error("minmax: Invalid order: order = {order}, data length = {data_len}")]
    InvalidOrder { order: usize, data_len: usize },
    #[error("minmax: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("minmax: All values are NaN.")]
    AllValuesNaN,
    #[error("minmax: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("minmax: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("minmax: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn minmax(input: &MinmaxInput) -> Result<MinmaxOutput, MinmaxError> {
    minmax_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn minmax_into_slice(
    is_min_dst: &mut [f64],
    is_max_dst: &mut [f64],
    last_min_dst: &mut [f64],
    last_max_dst: &mut [f64],
    input: &MinmaxInput,
    kern: Kernel,
) -> Result<(), MinmaxError> {
    let (high, low) = match &input.data {
        MinmaxData::Candles {
            candles,
            high_src,
            low_src,
        } => (
            minmax_source(candles, high_src),
            minmax_source(candles, low_src),
        ),
        MinmaxData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }

    let len = high.len();
    if is_min_dst.len() != len
        || is_max_dst.len() != len
        || last_min_dst.len() != len
        || last_max_dst.len() != len
    {
        return Err(MinmaxError::OutputLengthMismatch {
            expected: len,
            got: is_min_dst.len(),
        });
    }

    let order = input.get_order();
    if order == 0 || order > len {
        return Err(MinmaxError::InvalidOrder {
            order,
            data_len: len,
        });
    }

    let first_valid_idx = first_valid_pair(high, low).ok_or(MinmaxError::AllValuesNaN)?;

    if (len - first_valid_idx) < order {
        return Err(MinmaxError::NotEnoughValidData {
            needed: order,
            valid: len - first_valid_idx,
        });
    }

    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for i in 0..first_valid_idx {
        is_min_dst[i] = qnan;
        is_max_dst[i] = qnan;
        last_min_dst[i] = qnan;
        last_max_dst[i] = qnan;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => minmax_scalar(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => minmax_avx2(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => minmax_avx512(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
            _ => minmax_scalar(
                high,
                low,
                order,
                first_valid_idx,
                is_min_dst,
                is_max_dst,
                last_min_dst,
                last_max_dst,
            ),
        }
    }

    Ok(())
}

#[inline]
pub fn minmax_into(
    input: &MinmaxInput,
    out_is_min: &mut [f64],
    out_is_max: &mut [f64],
    out_last_min: &mut [f64],
    out_last_max: &mut [f64],
) -> Result<(), MinmaxError> {
    minmax_into_slice(
        out_is_min,
        out_is_max,
        out_last_min,
        out_last_max,
        input,
        Kernel::Auto,
    )
}

pub fn minmax_with_kernel(
    input: &MinmaxInput,
    kernel: Kernel,
) -> Result<MinmaxOutput, MinmaxError> {
    let (high, low) = match &input.data {
        MinmaxData::Candles {
            candles,
            high_src,
            low_src,
        } => (
            minmax_source(candles, high_src),
            minmax_source(candles, low_src),
        ),
        MinmaxData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }
    let len = high.len();
    let order = input.get_order();
    if order == 0 || order > len {
        return Err(MinmaxError::InvalidOrder {
            order,
            data_len: len,
        });
    }
    let first_valid_idx = first_valid_pair(high, low).ok_or(MinmaxError::AllValuesNaN)?;

    if (len - first_valid_idx) < order {
        return Err(MinmaxError::NotEnoughValidData {
            needed: order,
            valid: len - first_valid_idx,
        });
    }

    let mut is_min = alloc_with_nan_prefix(len, first_valid_idx);
    let mut is_max = alloc_with_nan_prefix(len, first_valid_idx);
    let mut last_min = alloc_with_nan_prefix(len, first_valid_idx);
    let mut last_max = alloc_with_nan_prefix(len, first_valid_idx);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => minmax_scalar(
                high,
                low,
                order,
                first_valid_idx,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => minmax_avx2(
                high,
                low,
                order,
                first_valid_idx,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => minmax_avx512(
                high,
                low,
                order,
                first_valid_idx,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            ),
            _ => unreachable!(),
        }
    }
    Ok(MinmaxOutput {
        is_min,
        is_max,
        last_min,
        last_max,
    })
}

#[inline]
pub fn minmax_scalar(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    #[inline(always)]
    fn fmin(a: f64, b: f64) -> f64 {
        if a < b { a } else { b }
    }
    #[inline(always)]
    fn fmax(a: f64, b: f64) -> f64 {
        if a > b { a } else { b }
    }

    let len = high.len();

    for i in 0..first_valid_idx {
        is_min[i] = f64::NAN;
        is_max[i] = f64::NAN;
        last_min[i] = f64::NAN;
        last_max[i] = f64::NAN;
    }

    if order == 3 && len >= 50_000 {
        minmax_scalar_order_3(
            high,
            low,
            first_valid_idx,
            is_min,
            is_max,
            last_min,
            last_max,
        );
        return;
    }

    const SMALL_ORDER_THRESHOLD: usize = 8;
    if order <= SMALL_ORDER_THRESHOLD {
        let mut last_min_val = f64::NAN;
        let mut last_max_val = f64::NAN;
        for i in first_valid_idx..len {
            let mut min_here = f64::NAN;
            let mut max_here = f64::NAN;

            if i >= order && i + order < len {
                unsafe {
                    let ch = *high.get_unchecked(i);
                    let cl = *low.get_unchecked(i);
                    if ch.is_finite() & cl.is_finite() {
                        let mut less_than_neighbors = true;
                        let mut greater_than_neighbors = true;

                        let mut o = 1usize;
                        while o <= order {
                            let lh = *high.get_unchecked(i - o);
                            let rh = *high.get_unchecked(i + o);
                            let ll = *low.get_unchecked(i - o);
                            let rl = *low.get_unchecked(i + o);

                            if less_than_neighbors {
                                if !(ll.is_finite() & rl.is_finite()) || !(cl < ll && cl < rl) {
                                    less_than_neighbors = false;
                                }
                            }
                            if greater_than_neighbors {
                                if !(lh.is_finite() & rh.is_finite()) || !(ch > lh && ch > rh) {
                                    greater_than_neighbors = false;
                                }
                            }

                            if !less_than_neighbors && !greater_than_neighbors {
                                break;
                            }
                            o += 1;
                        }

                        if less_than_neighbors {
                            min_here = cl;
                        }
                        if greater_than_neighbors {
                            max_here = ch;
                        }
                    }
                }
            }

            is_min[i] = min_here;
            is_max[i] = max_here;

            if min_here.is_finite() {
                last_min_val = min_here;
            }
            if max_here.is_finite() {
                last_max_val = max_here;
            }

            last_min[i] = last_min_val;
            last_max[i] = last_max_val;
        }
        return;
    }

    let n = len;
    if first_valid_idx >= n {
        return;
    }

    let mut left_min_low = vec![0.0f64; n];
    let mut right_min_low = vec![0.0f64; n];
    let mut left_max_high = vec![0.0f64; n];
    let mut right_max_high = vec![0.0f64; n];

    let mut left_all_low = vec![0u8; n];
    let mut right_all_low = vec![0u8; n];
    let mut left_all_high = vec![0u8; n];
    let mut right_all_high = vec![0u8; n];

    for i in 0..n {
        unsafe {
            let l = *low.get_unchecked(i);
            let h = *high.get_unchecked(i);
            let lf = l.is_finite() as u8;
            let hf = h.is_finite() as u8;
            if i % order == 0 {
                *left_min_low.get_unchecked_mut(i) = l;
                *left_max_high.get_unchecked_mut(i) = h;
                *left_all_low.get_unchecked_mut(i) = lf;
                *left_all_high.get_unchecked_mut(i) = hf;
            } else {
                let p = i - 1;
                *left_min_low.get_unchecked_mut(i) = fmin(*left_min_low.get_unchecked(p), l);
                *left_max_high.get_unchecked_mut(i) = fmax(*left_max_high.get_unchecked(p), h);
                *left_all_low.get_unchecked_mut(i) = *left_all_low.get_unchecked(p) & lf;
                *left_all_high.get_unchecked_mut(i) = *left_all_high.get_unchecked(p) & hf;
            }
        }
    }

    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        unsafe {
            let l = *low.get_unchecked(i);
            let h = *high.get_unchecked(i);
            let lf = l.is_finite() as u8;
            let hf = h.is_finite() as u8;
            if ((i + 1) % order) == 0 || i == n - 1 {
                *right_min_low.get_unchecked_mut(i) = l;
                *right_max_high.get_unchecked_mut(i) = h;
                *right_all_low.get_unchecked_mut(i) = lf;
                *right_all_high.get_unchecked_mut(i) = hf;
            } else {
                let n1 = i + 1;
                *right_min_low.get_unchecked_mut(i) = fmin(*right_min_low.get_unchecked(n1), l);
                *right_max_high.get_unchecked_mut(i) = fmax(*right_max_high.get_unchecked(n1), h);
                *right_all_low.get_unchecked_mut(i) = *right_all_low.get_unchecked(n1) & lf;
                *right_all_high.get_unchecked_mut(i) = *right_all_high.get_unchecked(n1) & hf;
            }
        }
    }

    let mut last_min_val = f64::NAN;
    let mut last_max_val = f64::NAN;
    for i in first_valid_idx..n {
        unsafe {
            let ch = *high.get_unchecked(i);
            let cl = *low.get_unchecked(i);
            let mut min_here = f64::NAN;
            let mut max_here = f64::NAN;

            if i >= order && i + order < n && ch.is_finite() && cl.is_finite() {
                let s_l = i - order;
                let e_l = i - 1;
                let s_r = i + 1;
                let e_r = i + order;

                let left_low_ok =
                    (*right_all_low.get_unchecked(s_l) & *left_all_low.get_unchecked(e_l)) == 1;
                let right_low_ok =
                    (*right_all_low.get_unchecked(s_r) & *left_all_low.get_unchecked(e_r)) == 1;
                let left_high_ok =
                    (*right_all_high.get_unchecked(s_l) & *left_all_high.get_unchecked(e_l)) == 1;
                let right_high_ok =
                    (*right_all_high.get_unchecked(s_r) & *left_all_high.get_unchecked(e_r)) == 1;

                if left_low_ok & right_low_ok {
                    let lmin = fmin(
                        *right_min_low.get_unchecked(s_l),
                        *left_min_low.get_unchecked(e_l),
                    );
                    let rmin = fmin(
                        *right_min_low.get_unchecked(s_r),
                        *left_min_low.get_unchecked(e_r),
                    );
                    if cl < lmin && cl < rmin {
                        min_here = cl;
                    }
                }

                if left_high_ok & right_high_ok {
                    let lmax = fmax(
                        *right_max_high.get_unchecked(s_l),
                        *left_max_high.get_unchecked(e_l),
                    );
                    let rmax = fmax(
                        *right_max_high.get_unchecked(s_r),
                        *left_max_high.get_unchecked(e_r),
                    );
                    if ch > lmax && ch > rmax {
                        max_here = ch;
                    }
                }
            }

            *is_min.get_unchecked_mut(i) = min_here;
            *is_max.get_unchecked_mut(i) = max_here;

            if min_here.is_finite() {
                last_min_val = min_here;
            }
            if max_here.is_finite() {
                last_max_val = max_here;
            }
            *last_min.get_unchecked_mut(i) = last_min_val;
            *last_max.get_unchecked_mut(i) = last_max_val;
        }
    }
}

#[inline(always)]
fn minmax_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
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
    }
}

#[inline(always)]
fn first_valid_pair(high: &[f64], low: &[f64]) -> Option<usize> {
    let len = high.len();
    let mut i = 0usize;
    unsafe {
        let high_ptr = high.as_ptr();
        let low_ptr = low.as_ptr();
        while i < len {
            if !(*high_ptr.add(i)).is_nan() && !(*low_ptr.add(i)).is_nan() {
                return Some(i);
            }
            i += 1;
        }
    }
    None
}

#[inline(always)]
fn minmax_scalar_order_3(
    high: &[f64],
    low: &[f64],
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    let len = high.len();
    let mut last_min_val = f64::NAN;
    let mut last_max_val = f64::NAN;

    unsafe {
        let high_ptr = high.as_ptr();
        let low_ptr = low.as_ptr();
        let is_min_ptr = is_min.as_mut_ptr();
        let is_max_ptr = is_max.as_mut_ptr();
        let last_min_ptr = last_min.as_mut_ptr();
        let last_max_ptr = last_max.as_mut_ptr();

        for i in first_valid_idx..len {
            let mut min_here = f64::NAN;
            let mut max_here = f64::NAN;

            if i >= 3 && i + 3 < len {
                let ch = *high_ptr.add(i);
                let cl = *low_ptr.add(i);
                if ch.is_finite() & cl.is_finite() {
                    let ll1 = *low_ptr.add(i - 1);
                    let ll2 = *low_ptr.add(i - 2);
                    let ll3 = *low_ptr.add(i - 3);
                    let rl1 = *low_ptr.add(i + 1);
                    let rl2 = *low_ptr.add(i + 2);
                    let rl3 = *low_ptr.add(i + 3);

                    let lh1 = *high_ptr.add(i - 1);
                    let lh2 = *high_ptr.add(i - 2);
                    let lh3 = *high_ptr.add(i - 3);
                    let rh1 = *high_ptr.add(i + 1);
                    let rh2 = *high_ptr.add(i + 2);
                    let rh3 = *high_ptr.add(i + 3);

                    if ll1.is_finite()
                        & ll2.is_finite()
                        & ll3.is_finite()
                        & rl1.is_finite()
                        & rl2.is_finite()
                        & rl3.is_finite()
                        & (cl < ll1)
                        & (cl < ll2)
                        & (cl < ll3)
                        & (cl < rl1)
                        & (cl < rl2)
                        & (cl < rl3)
                    {
                        min_here = cl;
                    }

                    if lh1.is_finite()
                        & lh2.is_finite()
                        & lh3.is_finite()
                        & rh1.is_finite()
                        & rh2.is_finite()
                        & rh3.is_finite()
                        & (ch > lh1)
                        & (ch > lh2)
                        & (ch > lh3)
                        & (ch > rh1)
                        & (ch > rh2)
                        & (ch > rh3)
                    {
                        max_here = ch;
                    }
                }
            }

            *is_min_ptr.add(i) = min_here;
            *is_max_ptr.add(i) = max_here;

            if min_here.is_finite() {
                last_min_val = min_here;
            }
            if max_here.is_finite() {
                last_max_val = max_here;
            }

            *last_min_ptr.add(i) = last_min_val;
            *last_max_ptr.add(i) = last_max_val;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx2(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid_idx,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    if order <= 16 {
        minmax_avx512_short(
            high,
            low,
            order,
            first_valid_idx,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    } else {
        minmax_avx512_long(
            high,
            low,
            order,
            first_valid_idx,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512_short(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid_idx,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512_long(
    high: &[f64],
    low: &[f64],
    order: usize,
    first_valid_idx: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid_idx,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct MinmaxStream {
    order: usize,
    len: usize,
    idx: usize,
    seen: usize,
    filled: bool,

    kplus1: usize,
    ring_pos: usize,
    ring_high: Vec<f64>,
    ring_low: Vec<f64>,

    rq_min_low: VecDeque<(usize, f64)>,
    rq_max_high: VecDeque<(usize, f64)>,

    right_flags_pos: usize,
    right_low_flags: Vec<u8>,
    right_high_flags: Vec<u8>,
    right_low_count: usize,
    right_high_count: usize,

    hist_rmin_low: Vec<f64>,
    hist_rmax_high: Vec<f64>,
    hist_right_low_count: Vec<usize>,
    hist_right_high_count: Vec<usize>,

    last_min: f64,
    last_max: f64,
}

impl MinmaxStream {
    pub fn try_new(params: MinmaxParams) -> Result<Self, MinmaxError> {
        let order = params.order.unwrap_or(3);
        if order == 0 {
            return Err(MinmaxError::InvalidOrder { order, data_len: 0 });
        }
        let k = order;
        let kplus1 = k + 1;
        Ok(Self {
            order: k,
            len: k * 2 + 1,
            idx: 0,
            seen: 0,
            filled: false,

            kplus1,
            ring_pos: 0,
            ring_high: vec![f64::NAN; kplus1],
            ring_low: vec![f64::NAN; kplus1],

            rq_min_low: VecDeque::with_capacity(k),
            rq_max_high: VecDeque::with_capacity(k),

            right_flags_pos: 0,
            right_low_flags: vec![0; k],
            right_high_flags: vec![0; k],
            right_low_count: 0,
            right_high_count: 0,

            hist_rmin_low: vec![f64::NAN; kplus1],
            hist_rmax_high: vec![f64::NAN; kplus1],
            hist_right_low_count: vec![0; kplus1],
            hist_right_high_count: vec![0; kplus1],

            last_min: f64::NAN,
            last_max: f64::NAN,
        })
    }

    #[inline(always)]
    fn evict_old(&mut self) {
        let cutoff = self.idx.saturating_sub(self.order);
        while let Some(&(j, _)) = self.rq_min_low.front() {
            if j <= cutoff {
                self.rq_min_low.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(j, _)) = self.rq_max_high.front() {
            if j <= cutoff {
                self.rq_max_high.pop_front();
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn push_right_low(&mut self, idx: usize, val: f64) {
        if val.is_finite() {
            while let Some(&(_, v)) = self.rq_min_low.back() {
                if v >= val {
                    self.rq_min_low.pop_back();
                } else {
                    break;
                }
            }
            self.rq_min_low.push_back((idx, val));
        }
    }

    #[inline(always)]
    fn push_right_high(&mut self, idx: usize, val: f64) {
        if val.is_finite() {
            while let Some(&(_, v)) = self.rq_max_high.back() {
                if v <= val {
                    self.rq_max_high.pop_back();
                } else {
                    break;
                }
            }
            self.rq_max_high.push_back((idx, val));
        }
    }

    #[inline(always)]
    fn update_right_counts(&mut self, high: f64, low: f64) {
        let pos = self.right_flags_pos;
        let old_low = self.right_low_flags[pos] as isize;
        let old_high = self.right_high_flags[pos] as isize;
        let new_low = low.is_finite() as u8;
        let new_high = high.is_finite() as u8;
        self.right_low_flags[pos] = new_low;
        self.right_high_flags[pos] = new_high;
        self.right_low_count =
            (self.right_low_count as isize + (new_low as isize - old_low)) as usize;
        self.right_high_count =
            (self.right_high_count as isize + (new_high as isize - old_high)) as usize;

        if self.right_flags_pos + 1 == self.order {
            self.right_flags_pos = 0;
        } else {
            self.right_flags_pos += 1;
        }
    }

    pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, f64, f64) {
        let k = self.order;
        let kp = self.kplus1;
        let pos = self.ring_pos;

        let left_min_low = self.hist_rmin_low[pos];
        let left_max_high = self.hist_rmax_high[pos];
        let left_low_count = self.hist_right_low_count[pos];
        let left_high_count = self.hist_right_high_count[pos];

        self.ring_high[pos] = high;
        self.ring_low[pos] = low;

        self.evict_old();
        self.push_right_low(self.idx, low);
        self.push_right_high(self.idx, high);
        self.update_right_counts(high, low);

        let right_min_low = self.rq_min_low.front().map(|&(_, v)| v).unwrap_or(f64::NAN);
        let right_max_high = self
            .rq_max_high
            .front()
            .map(|&(_, v)| v)
            .unwrap_or(f64::NAN);

        self.hist_rmin_low[pos] = right_min_low;
        self.hist_rmax_high[pos] = right_max_high;
        self.hist_right_low_count[pos] = self.right_low_count;
        self.hist_right_high_count[pos] = self.right_high_count;

        self.idx = self.idx.wrapping_add(1);
        self.seen = self.seen.saturating_add(1);
        if self.ring_pos + 1 == kp {
            self.ring_pos = 0;
        } else {
            self.ring_pos += 1;
        }
        if !self.filled && self.seen >= self.len {
            self.filled = true;
        }

        if !self.filled {
            return (None, None, self.last_min, self.last_max);
        }

        let cpos = if pos + 1 == kp { 0 } else { pos + 1 };
        let ch = self.ring_high[cpos];
        let cl = self.ring_low[cpos];

        let mut out_min: Option<f64> = None;
        let mut out_max: Option<f64> = None;

        if ch.is_finite() & cl.is_finite() {
            if left_low_count == k
                && self.right_low_count == k
                && cl < left_min_low
                && cl < right_min_low
            {
                out_min = Some(cl);
                self.last_min = cl;
            }
            if left_high_count == k
                && self.right_high_count == k
                && ch > left_max_high
                && ch > right_max_high
            {
                out_max = Some(ch);
                self.last_max = ch;
            }
        }

        (out_min, out_max, self.last_min, self.last_max)
    }
}

#[derive(Clone, Debug)]
pub struct MinmaxBatchRange {
    pub order: (usize, usize, usize),
}

impl Default for MinmaxBatchRange {
    fn default() -> Self {
        Self { order: (3, 252, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MinmaxBatchBuilder {
    range: MinmaxBatchRange,
    kernel: Kernel,
}

impl MinmaxBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn order_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.order = (start, end, step);
        self
    }
    pub fn order_static(mut self, o: usize) -> Self {
        self.range.order = (o, o, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MinmaxBatchOutput, MinmaxError> {
        minmax_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<MinmaxBatchOutput, MinmaxError> {
        MinmaxBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<MinmaxBatchOutput, MinmaxError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MinmaxBatchOutput, MinmaxError> {
        MinmaxBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub fn minmax_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    k: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MinmaxError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    minmax_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MinmaxBatchOutput {
    pub is_min: Vec<f64>,
    pub is_max: Vec<f64>,
    pub last_min: Vec<f64>,
    pub last_max: Vec<f64>,
    pub combos: Vec<MinmaxParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MinmaxBatchOutput {
    pub fn row_for_params(&self, p: &MinmaxParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.order.unwrap_or(3) == p.order.unwrap_or(3))
    }
    pub fn is_min_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.is_min[start..start + self.cols]
        })
    }
    pub fn is_max_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.is_max[start..start + self.cols]
        })
    }
    pub fn last_min_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.last_min[start..start + self.cols]
        })
    }
    pub fn last_max_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.last_max[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MinmaxBatchRange) -> Result<Vec<MinmaxParams>, MinmaxError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, MinmaxError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let st = step.max(1);
            let mut v = start;
            while v <= end {
                out.push(v);
                match v.checked_add(st) {
                    Some(next) => {
                        if next == v {
                            break;
                        }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            let st = step.max(1) as isize;
            let mut v = start as isize;
            let end_i = end as isize;
            while v >= end_i {
                out.push(v as usize);
                v -= st;
            }
        }
        if out.is_empty() {
            return Err(MinmaxError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }
    let orders = axis_usize(r.order)?;
    let mut out = Vec::with_capacity(orders.len());
    for &o in &orders {
        out.push(MinmaxParams { order: Some(o) });
    }
    Ok(out)
}

#[inline(always)]
pub fn minmax_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    minmax_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn minmax_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    minmax_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn minmax_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MinmaxBatchOutput, MinmaxError> {
    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }

    let combos = expand_grid(sweep)?;

    let len = high.len();
    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
        .ok_or(MinmaxError::AllValuesNaN)?;
    let max_o = combos.iter().map(|c| c.order.unwrap()).max().unwrap();
    if len - first < max_o {
        return Err(MinmaxError::NotEnoughValidData {
            needed: max_o,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| MinmaxError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols overflow".to_string(),
        })?;

    let mut min_mu = make_uninit_matrix(rows, cols);
    let mut max_mu = make_uninit_matrix(rows, cols);
    let mut lmin_mu = make_uninit_matrix(rows, cols);
    let mut lmax_mu = make_uninit_matrix(rows, cols);

    let warm = vec![first; rows];
    init_matrix_prefixes(&mut min_mu, cols, &warm);
    init_matrix_prefixes(&mut max_mu, cols, &warm);
    init_matrix_prefixes(&mut lmin_mu, cols, &warm);
    init_matrix_prefixes(&mut lmax_mu, cols, &warm);

    let mut min_guard = core::mem::ManuallyDrop::new(min_mu);
    let mut max_guard = core::mem::ManuallyDrop::new(max_mu);
    let mut lmin_guard = core::mem::ManuallyDrop::new(lmin_mu);
    let mut lmax_guard = core::mem::ManuallyDrop::new(lmax_mu);

    let is_min: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(min_guard.as_mut_ptr() as *mut f64, min_guard.len())
    };
    let is_max: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(max_guard.as_mut_ptr() as *mut f64, max_guard.len())
    };
    let last_min: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lmin_guard.as_mut_ptr() as *mut f64, lmin_guard.len())
    };
    let last_max: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lmax_guard.as_mut_ptr() as *mut f64, lmax_guard.len())
    };

    let do_row = |row: usize,
                  out_min: &mut [f64],
                  out_max: &mut [f64],
                  out_lmin: &mut [f64],
                  out_lmax: &mut [f64]| unsafe {
        let o = combos[row].order.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                minmax_row_avx2(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                minmax_row_avx512(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            _ => minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            is_min
                .par_chunks_mut(cols)
                .zip(is_max.par_chunks_mut(cols))
                .zip(
                    last_min
                        .par_chunks_mut(cols)
                        .zip(last_max.par_chunks_mut(cols)),
                )
                .enumerate()
                .for_each(|(row, ((m, x), (lm, lx)))| do_row(row, m, x, lm, lx));
        }
        #[cfg(target_arch = "wasm32")]
        for (row, ((m, x), (lm, lx))) in is_min
            .chunks_mut(cols)
            .zip(is_max.chunks_mut(cols))
            .zip(last_min.chunks_mut(cols).zip(last_max.chunks_mut(cols)))
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    } else {
        for (row, ((m, x), (lm, lx))) in is_min
            .chunks_mut(cols)
            .zip(is_max.chunks_mut(cols))
            .zip(last_min.chunks_mut(cols).zip(last_max.chunks_mut(cols)))
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    }

    let is_min = unsafe {
        Vec::from_raw_parts(
            min_guard.as_mut_ptr() as *mut f64,
            min_guard.len(),
            min_guard.capacity(),
        )
    };
    let is_max = unsafe {
        Vec::from_raw_parts(
            max_guard.as_mut_ptr() as *mut f64,
            max_guard.len(),
            max_guard.capacity(),
        )
    };
    let last_min = unsafe {
        Vec::from_raw_parts(
            lmin_guard.as_mut_ptr() as *mut f64,
            lmin_guard.len(),
            lmin_guard.capacity(),
        )
    };
    let last_max = unsafe {
        Vec::from_raw_parts(
            lmax_guard.as_mut_ptr() as *mut f64,
            lmax_guard.len(),
            lmax_guard.capacity(),
        )
    };

    Ok(MinmaxBatchOutput {
        is_min,
        is_max,
        last_min,
        last_max,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn minmax_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &MinmaxBatchRange,
    kern: Kernel,
    parallel: bool,
    is_min_out: &mut [f64],
    is_max_out: &mut [f64],
    last_min_out: &mut [f64],
    last_max_out: &mut [f64],
) -> Result<Vec<MinmaxParams>, MinmaxError> {
    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(MinmaxError::InvalidOrder {
            order: 0,
            data_len: high.len().max(low.len()),
        });
    }

    let combos = expand_grid(sweep)?;

    let len = high.len();
    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| MinmaxError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols overflow".to_string(),
        })?;

    if is_min_out.len() != total
        || is_max_out.len() != total
        || last_min_out.len() != total
        || last_max_out.len() != total
    {
        return Err(MinmaxError::OutputLengthMismatch {
            expected: total,
            got: is_min_out.len(),
        });
    }

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
        .ok_or(MinmaxError::AllValuesNaN)?;
    let max_o = combos.iter().map(|c| c.order.unwrap()).max().unwrap();
    if len - first < max_o {
        return Err(MinmaxError::NotEnoughValidData {
            needed: max_o,
            valid: len - first,
        });
    }

    let warm = vec![first; rows];
    let (min_mu, max_mu, lmin_mu, lmax_mu) = unsafe {
        (
            core::slice::from_raw_parts_mut(
                is_min_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                total,
            ),
            core::slice::from_raw_parts_mut(
                is_max_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                total,
            ),
            core::slice::from_raw_parts_mut(
                last_min_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                total,
            ),
            core::slice::from_raw_parts_mut(
                last_max_out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
                total,
            ),
        )
    };
    init_matrix_prefixes(min_mu, cols, &warm);
    init_matrix_prefixes(max_mu, cols, &warm);
    init_matrix_prefixes(lmin_mu, cols, &warm);
    init_matrix_prefixes(lmax_mu, cols, &warm);

    let do_row = |row: usize,
                  out_min: &mut [f64],
                  out_max: &mut [f64],
                  out_lmin: &mut [f64],
                  out_lmax: &mut [f64]| unsafe {
        let o = combos[row].order.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                minmax_row_avx2(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                minmax_row_avx512(high, low, first, o, out_min, out_max, out_lmin, out_lmax)
            }
            _ => minmax_row_scalar(high, low, first, o, out_min, out_max, out_lmin, out_lmax),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            is_min_out
                .par_chunks_mut(cols)
                .zip(is_max_out.par_chunks_mut(cols))
                .zip(
                    last_min_out
                        .par_chunks_mut(cols)
                        .zip(last_max_out.par_chunks_mut(cols)),
                )
                .enumerate()
                .for_each(|(row, ((m, x), (lm, lx)))| do_row(row, m, x, lm, lx));
        }
        #[cfg(target_arch = "wasm32")]
        for (row, ((m, x), (lm, lx))) in is_min_out
            .chunks_mut(cols)
            .zip(is_max_out.chunks_mut(cols))
            .zip(
                last_min_out
                    .chunks_mut(cols)
                    .zip(last_max_out.chunks_mut(cols)),
            )
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    } else {
        for (row, ((m, x), (lm, lx))) in is_min_out
            .chunks_mut(cols)
            .zip(is_max_out.chunks_mut(cols))
            .zip(
                last_min_out
                    .chunks_mut(cols)
                    .zip(last_max_out.chunks_mut(cols)),
            )
            .enumerate()
        {
            do_row(row, m, x, lm, lx);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn minmax_row_scalar(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_scalar(
        high,
        low,
        order,
        first_valid,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx2(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_row_scalar(
        high,
        low,
        first_valid,
        order,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    if order <= 16 {
        minmax_row_avx512_short(
            high,
            low,
            first_valid,
            order,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    } else {
        minmax_row_avx512_long(
            high,
            low,
            first_valid,
            order,
            is_min,
            is_max,
            last_min,
            last_max,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_row_scalar(
        high,
        low,
        first_valid,
        order,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    order: usize,
    is_min: &mut [f64],
    is_max: &mut [f64],
    last_min: &mut [f64],
    last_max: &mut [f64],
) {
    minmax_row_scalar(
        high,
        low,
        first_valid,
        order,
        is_min,
        is_max,
        last_min,
        last_max,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_vortex;

    fn check_minmax_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = MinmaxParams { order: None };
        let input = MinmaxInput::from_candles(&candles, "high", "low", params);
        let output = minmax_with_kernel(&input, kernel)?;
        assert_eq!(output.is_min.len(), candles.close.len());
        Ok(())
    }

    fn check_minmax_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;
        let params = MinmaxParams { order: Some(3) };
        let input = MinmaxInput::from_candles(&candles, "high", "low", params);
        let output = minmax_with_kernel(&input, kernel)?;
        assert_eq!(output.is_min.len(), candles.close.len());
        let count = output.is_min.len();
        assert!(count >= 5, "Not enough data to check last 5");
        let start_index = count - 5;
        for &val in &output.is_min[start_index..] {
            assert!(val.is_nan());
        }
        for &val in &output.is_max[start_index..] {
            assert!(val.is_nan());
        }
        let expected_last_five_min = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
        let last_min_slice = &output.last_min[start_index..];
        for (i, &val) in last_min_slice.iter().enumerate() {
            let expected_val = expected_last_five_min[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "Minmax last_min mismatch at idx {}: {} vs {}",
                i,
                val,
                expected_val
            );
        }
        let expected_last_five_max = [60102.0, 60102.0, 60102.0, 60102.0, 60102.0];
        let last_max_slice = &output.last_max[start_index..];
        for (i, &val) in last_max_slice.iter().enumerate() {
            let expected_val = expected_last_five_max[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "Minmax last_max mismatch at idx {}: {} vs {}",
                i,
                val,
                expected_val
            );
        }
        Ok(())
    }

    fn check_minmax_zero_order(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [1.0, 2.0, 3.0];
        let params = MinmaxParams { order: Some(0) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with zero order",
            test_name
        );
        Ok(())
    }

    fn check_minmax_order_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [1.0, 2.0, 3.0];
        let params = MinmaxParams { order: Some(10) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with order > length",
            test_name
        );
        Ok(())
    }

    fn check_minmax_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = MinmaxParams { order: Some(1) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with all NaN data",
            test_name
        );
        Ok(())
    }

    fn check_minmax_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, 10.0];
        let low = [f64::NAN, 5.0];
        let params = MinmaxParams { order: Some(3) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let res = minmax_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Minmax should fail with not enough valid data",
            test_name
        );
        Ok(())
    }

    fn check_minmax_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [50.0, 55.0, 60.0, 55.0, 50.0, 45.0, 50.0, 55.0];
        let low = [40.0, 38.0, 35.0, 38.0, 40.0, 42.0, 41.0, 39.0];
        let params = MinmaxParams { order: Some(2) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let output = minmax_with_kernel(&input, kernel)?;
        assert_eq!(output.is_min.len(), 8);
        assert_eq!(output.is_max.len(), 8);
        assert_eq!(output.last_min.len(), 8);
        assert_eq!(output.last_max.len(), 8);
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_minmax_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles = read_candles_from_vortex(file_path)?;

        let test_params = vec![
            MinmaxParams::default(),
            MinmaxParams { order: Some(1) },
            MinmaxParams { order: Some(2) },
            MinmaxParams { order: Some(5) },
            MinmaxParams { order: Some(10) },
            MinmaxParams { order: Some(20) },
            MinmaxParams { order: Some(50) },
            MinmaxParams { order: Some(100) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MinmaxInput::from_candles(&candles, "high", "low", params.clone());
            let output = minmax_with_kernel(&input, kernel)?;

            let arrays = [
                (&output.is_min, "is_min"),
                (&output.is_max, "is_max"),
                (&output.last_min, "last_min"),
                (&output.last_max, "last_max"),
            ];

            for (array, array_name) in arrays.iter() {
                for (i, &val) in array.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in {} with params: order={} (param set {})",
                            test_name,
                            val,
                            bits,
                            i,
                            array_name,
                            params.order.unwrap_or(3),
                            param_idx
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in {} with params: order={} (param set {})",
                            test_name,
                            val,
                            bits,
                            i,
                            array_name,
                            params.order.unwrap_or(3),
                            param_idx
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in {} with params: order={} (param set {})",
                            test_name,
                            val,
                            bits,
                            i,
                            array_name,
                            params.order.unwrap_or(3),
                            param_idx
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_minmax_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_minmax_tests {
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

    generate_all_minmax_tests!(
        check_minmax_partial_params,
        check_minmax_accuracy,
        check_minmax_zero_order,
        check_minmax_order_exceeds_length,
        check_minmax_nan_handling,
        check_minmax_very_small_dataset,
        check_minmax_basic_slices,
        check_minmax_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_minmax_tests!(check_minmax_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;
        let output = MinmaxBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = MinmaxParams::default();
        let row = output.is_min_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[test]
    fn test_minmax_into_matches_api() {
        let mut high = Vec::with_capacity(256);
        let mut low = Vec::with_capacity(256);

        for _ in 0..5 {
            high.push(f64::NAN);
            low.push(f64::NAN);
        }

        for i in 0..200usize {
            let t = i as f64;
            high.push(100.0 + (t / 5.0).sin() * 10.0 + (t / 7.0).cos() * 3.0);
            low.push(90.0 - (t / 6.0).sin() * 9.0 - (t / 8.0).cos() * 2.0);
        }

        for j in 0..51usize {
            let t = j as f64;
            high.push(105.0 + (t * 0.01).sin());
            low.push(95.0 - (t * 0.01).cos());
        }

        let params = MinmaxParams::default();
        let input = MinmaxInput::from_slices(&high, &low, params);

        let baseline = minmax(&input).expect("baseline minmax() should succeed");

        let n = high.len();
        let mut is_min = vec![0.0; n];
        let mut is_max = vec![0.0; n];
        let mut last_min = vec![0.0; n];
        let mut last_max = vec![0.0; n];

        {
            minmax_into(
                &input,
                &mut is_min,
                &mut is_max,
                &mut last_min,
                &mut last_max,
            )
            .expect("minmax_into should succeed");

            assert_eq!(is_min.len(), baseline.is_min.len());
            assert_eq!(is_max.len(), baseline.is_max.len());
            assert_eq!(last_min.len(), baseline.last_min.len());
            assert_eq!(last_max.len(), baseline.last_max.len());

            fn eq_or_both_nan(a: f64, b: f64) -> bool {
                (a.is_nan() && b.is_nan()) || (a == b)
            }

            for i in 0..n {
                assert!(
                    eq_or_both_nan(is_min[i], baseline.is_min[i]),
                    "is_min mismatch at {}: {:?} vs {:?}",
                    i,
                    is_min[i],
                    baseline.is_min[i]
                );
                assert!(
                    eq_or_both_nan(is_max[i], baseline.is_max[i]),
                    "is_max mismatch at {}: {:?} vs {:?}",
                    i,
                    is_max[i],
                    baseline.is_max[i]
                );
                assert!(
                    eq_or_both_nan(last_min[i], baseline.last_min[i]),
                    "last_min mismatch at {}: {:?} vs {:?}",
                    i,
                    last_min[i],
                    baseline.last_min[i]
                );
                assert!(
                    eq_or_both_nan(last_max[i], baseline.last_max[i]),
                    "last_max mismatch at {}: {:?} vs {:?}",
                    i,
                    last_max[i],
                    baseline.last_max[i]
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let c = read_candles_from_vortex(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (5, 25, 5),
            (30, 60, 15),
            (2, 5, 1),
            (1, 1, 0),
            (10, 50, 10),
            (100, 100, 0),
        ];

        for (cfg_idx, &(order_start, order_end, order_step)) in test_configs.iter().enumerate() {
            let output = MinmaxBatchBuilder::new()
                .kernel(kernel)
                .order_range(order_start, order_end, order_step)
                .apply_candles(&c)?;

            let arrays = [
                (&output.is_min, "is_min"),
                (&output.is_max, "is_max"),
                (&output.last_min, "last_min"),
                (&output.last_max, "last_max"),
            ];

            for (array, array_name) in arrays.iter() {
                for (idx, &val) in array.iter().enumerate() {
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
							 at row {} col {} (flat index {}) in {} with params: order={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            array_name,
                            combo.order.unwrap_or(3)
                        );
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} with params: order={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            array_name,
                            combo.order.unwrap_or(3)
                        );
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} with params: order={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            array_name,
                            combo.order.unwrap_or(3)
                        );
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_minmax_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=50).prop_flat_map(|order| {
            (
                (order..400).prop_flat_map(move |len| {
                    prop::collection::vec(
                        (0.1f64..1000.0f64, 0.0f64..=0.2)
                            .prop_filter("finite", |(x, _)| x.is_finite()),
                        len,
                    )
                    .prop_map(move |pairs| {
                        let mut low = Vec::with_capacity(len);
                        let mut high = Vec::with_capacity(len);

                        for (l, spread) in pairs {
                            low.push(l);
                            high.push(l * (1.0 + spread));
                        }

                        (high, low)
                    })
                }),
                Just(order),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |((high, low), order)| {
                let params = MinmaxParams { order: Some(order) };
                let input = MinmaxInput::from_slices(&high, &low, params);

                let output = minmax_with_kernel(&input, kernel)?;
                let ref_output = minmax_with_kernel(&input, Kernel::Scalar)?;

                prop_assert_eq!(output.is_min.len(), high.len());
                prop_assert_eq!(output.is_max.len(), high.len());
                prop_assert_eq!(output.last_min.len(), high.len());
                prop_assert_eq!(output.last_max.len(), high.len());

                for i in 0..order.min(high.len()) {
                    prop_assert!(
                        output.is_min[i].is_nan(),
                        "is_min[{}] should be NaN during warmup",
                        i
                    );
                    prop_assert!(
                        output.is_max[i].is_nan(),
                        "is_max[{}] should be NaN during warmup",
                        i
                    );
                }

                for i in order..high.len().saturating_sub(order) {
                    if !output.is_min[i].is_nan() {
                        prop_assert_eq!(
                            output.is_min[i],
                            low[i],
                            "is_min[{}] should equal low[{}]",
                            i,
                            i
                        );

                        for o in 1..=order {
                            if i >= o && i + o < low.len() {
                                prop_assert!(
                                    low[i] <= low[i - o] && low[i] <= low[i + o],
                                    "Detected min at {} not <= neighbors at {} and {}",
                                    i,
                                    i - o,
                                    i + o
                                );
                            }
                        }
                    }

                    if !output.is_max[i].is_nan() {
                        prop_assert_eq!(
                            output.is_max[i],
                            high[i],
                            "is_max[{}] should equal high[{}]",
                            i,
                            i
                        );

                        for o in 1..=order {
                            if i >= o && i + o < high.len() {
                                prop_assert!(
                                    high[i] >= high[i - o] && high[i] >= high[i + o],
                                    "Detected max at {} not >= neighbors at {} and {}",
                                    i,
                                    i - o,
                                    i + o
                                );
                            }
                        }
                    }
                }

                let first_valid_idx = high
                    .iter()
                    .zip(low.iter())
                    .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
                    .unwrap_or(0);

                for i in first_valid_idx..high.len() {
                    if i > first_valid_idx {
                        if output.is_min[i].is_nan() && !output.last_min[i - 1].is_nan() {
                            prop_assert_eq!(
                                output.last_min[i],
                                output.last_min[i - 1],
                                "last_min[{}] should equal last_min[{}]",
                                i,
                                i - 1
                            );
                        }
                        if output.is_max[i].is_nan() && !output.last_max[i - 1].is_nan() {
                            prop_assert_eq!(
                                output.last_max[i],
                                output.last_max[i - 1],
                                "last_max[{}] should equal last_max[{}]",
                                i,
                                i - 1
                            );
                        }

                        if !output.is_min[i].is_nan() {
                            prop_assert_eq!(
                                output.last_min[i],
                                output.is_min[i],
                                "last_min[{}] should update to new minimum",
                                i
                            );
                        }
                        if !output.is_max[i].is_nan() {
                            prop_assert_eq!(
                                output.last_max[i],
                                output.is_max[i],
                                "last_max[{}] should update to new maximum",
                                i
                            );
                        }
                    }
                }

                for i in 0..high.len() {
                    if output.is_min[i].is_finite() && ref_output.is_min[i].is_finite() {
                        let ulp_diff = output.is_min[i]
                            .to_bits()
                            .abs_diff(ref_output.is_min[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "is_min[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.is_min[i],
                            ref_output.is_min[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.is_min[i].to_bits(),
                            ref_output.is_min[i].to_bits(),
                            "is_min[{}] NaN mismatch",
                            i
                        );
                    }

                    if output.is_max[i].is_finite() && ref_output.is_max[i].is_finite() {
                        let ulp_diff = output.is_max[i]
                            .to_bits()
                            .abs_diff(ref_output.is_max[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "is_max[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.is_max[i],
                            ref_output.is_max[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.is_max[i].to_bits(),
                            ref_output.is_max[i].to_bits(),
                            "is_max[{}] NaN mismatch",
                            i
                        );
                    }

                    if output.last_min[i].is_finite() && ref_output.last_min[i].is_finite() {
                        let ulp_diff = output.last_min[i]
                            .to_bits()
                            .abs_diff(ref_output.last_min[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "last_min[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.last_min[i],
                            ref_output.last_min[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.last_min[i].to_bits(),
                            ref_output.last_min[i].to_bits(),
                            "last_min[{}] NaN mismatch",
                            i
                        );
                    }

                    if output.last_max[i].is_finite() && ref_output.last_max[i].is_finite() {
                        let ulp_diff = output.last_max[i]
                            .to_bits()
                            .abs_diff(ref_output.last_max[i].to_bits());
                        prop_assert!(
                            ulp_diff <= 5,
                            "last_max[{}] kernel mismatch: {} vs {} (ULP={})",
                            i,
                            output.last_max[i],
                            ref_output.last_max[i],
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            output.last_max[i].to_bits(),
                            ref_output.last_max[i].to_bits(),
                            "last_max[{}] NaN mismatch",
                            i
                        );
                    }
                }

                let min_low = low.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_high = high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                for i in 0..high.len() {
                    if !output.is_min[i].is_nan() {
                        prop_assert!(
                            output.is_min[i] >= min_low && output.is_min[i] <= max_high,
                            "is_min[{}]={} outside data range [{}, {}]",
                            i,
                            output.is_min[i],
                            min_low,
                            max_high
                        );
                    }
                    if !output.is_max[i].is_nan() {
                        prop_assert!(
                            output.is_max[i] >= min_low && output.is_max[i] <= max_high,
                            "is_max[{}]={} outside data range [{}, {}]",
                            i,
                            output.is_max[i],
                            min_low,
                            max_high
                        );
                    }
                    if !output.last_min[i].is_nan() {
                        prop_assert!(
                            output.last_min[i] >= min_low && output.last_min[i] <= max_high,
                            "last_min[{}]={} outside data range [{}, {}]",
                            i,
                            output.last_min[i],
                            min_low,
                            max_high
                        );
                    }
                    if !output.last_max[i].is_nan() {
                        prop_assert!(
                            output.last_max[i] >= min_low && output.last_max[i] <= max_high,
                            "last_max[{}]={} outside data range [{}, {}]",
                            i,
                            output.last_max[i],
                            min_low,
                            max_high
                        );
                    }
                }

                if order == 1 && high.len() >= 3 {
                    for i in 1..high.len() - 1 {
                        if low[i] < low[i - 1] && low[i] < low[i + 1] {
                            prop_assert!(
                                !output.is_min[i].is_nan(),
                                "Expected minimum at {} not detected",
                                i
                            );
                        }

                        if high[i] > high[i - 1] && high[i] > high[i + 1] {
                            prop_assert!(
                                !output.is_max[i].is_nan(),
                                "Expected maximum at {} not detected",
                                i
                            );
                        }
                    }
                }

                for i in 0..high.len() {
                    prop_assert!(
                        high[i] >= low[i],
                        "Invalid data: high[{}]={} < low[{}]={}",
                        i,
                        high[i],
                        i,
                        low[i]
                    );
                }

                Ok(())
            })
            .unwrap();

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
