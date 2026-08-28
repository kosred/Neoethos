#![cfg(feature = "cuda-build-native")]

//! `smooth_theil_sen` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved the symbol `smooth_theil_sen_batch_f64` — which was a
//! one-line EMPTY kernel — threw the function away, computed all six output
//! series on the host through `Kernel::ScalarBatch`, and uploaded them with
//! `DeviceBuffer::from_slice`. The caller got six device pointers and no way to
//! know the card had been idle.
//!
//! # What it is now
//!
//! `kernels/cuda/smooth_theil_sen_kernel.cu` carries a real kernel and this
//! wrapper launches it. The CPU implementation is untouched and remains the
//! correct path for a machine with no card — it is simply not reachable from
//! here, because a `CudaSmoothTheilSen` only exists once a device context has
//! been created. A launch failure is an `Err` naming the indicator.

use crate::cuda::f64_launch::{
    DEFAULT_HEADROOM, LaunchPlanError, checked_mul, plan_slots, scratch_elems, validate_launch,
};
use crate::indicators::smooth_theil_sen::{
    SmoothTheilSenBatchRange, SmoothTheilSenDeviationType, SmoothTheilSenParams,
    SmoothTheilSenStatStyle, smooth_theil_sen_expand_grid,
};
use cust::context::Context;
use cust::device::Device;
use cust::launch;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const INDICATOR: &str = "smooth_theil_sen";
const KERNEL: &str = "smooth_theil_sen_batch_f64";
const DEFAULT_LENGTH: usize = 25;
const DEFAULT_OFFSET: usize = 0;
const DEFAULT_MULTIPLIER: f64 = 2.0;

/// The kernel's `STS_STYLE_*` codes. Declaration order in
/// `SmoothTheilSenStatStyle`, spelled out rather than cast, so adding a variant
/// upstream breaks this match instead of silently renumbering the kernel.
fn style_code(style: SmoothTheilSenStatStyle) -> i32 {
    match style {
        SmoothTheilSenStatStyle::Mean => 0,
        SmoothTheilSenStatStyle::SmoothMedian => 1,
        SmoothTheilSenStatStyle::Median => 2,
    }
}

fn deviation_code(style: SmoothTheilSenDeviationType) -> i32 {
    match style {
        SmoothTheilSenDeviationType::Mad => 0,
        SmoothTheilSenDeviationType::Rmsd => 1,
    }
}

#[derive(Debug, Error)]
pub enum CudaSmoothTheilSenError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("smooth_theil_sen: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct SmoothTheilSenDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SmoothTheilSenDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct SmoothTheilSenDeviceOutputs {
    pub value: SmoothTheilSenDeviceArrayF64,
    pub upper: SmoothTheilSenDeviceArrayF64,
    pub lower: SmoothTheilSenDeviceArrayF64,
    pub slope: SmoothTheilSenDeviceArrayF64,
    pub intercept: SmoothTheilSenDeviceArrayF64,
    pub deviation: SmoothTheilSenDeviceArrayF64,
}

impl SmoothTheilSenDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaSmoothTheilSenBatchResult {
    pub outputs: SmoothTheilSenDeviceOutputs,
    pub combos: Vec<SmoothTheilSenParams>,
}

pub struct CudaSmoothTheilSen {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSmoothTheilSen {
    pub fn new(device_id: usize) -> Result<Self, CudaSmoothTheilSenError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("smooth_theil_sen_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaSmoothTheilSenError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &SmoothTheilSenBatchRange,
    ) -> Result<CudaSmoothTheilSenBatchResult, CudaSmoothTheilSenError> {
        let cols = data.len();
        if cols == 0 {
            return Err(CudaSmoothTheilSenError::InvalidInput("empty input".into()));
        }
        // `validate_raw_data` (:1361): the first FINITE bar, shared by all rows.
        let first = data
            .iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| CudaSmoothTheilSenError::InvalidInput("all values are NaN".into()))?;

        let combos = smooth_theil_sen_expand_grid(sweep)
            .map_err(|e| CudaSmoothTheilSenError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSmoothTheilSenError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let mut lengths = Vec::with_capacity(rows);
        let mut offsets = Vec::with_capacity(rows);
        let mut multipliers = Vec::with_capacity(rows);
        let mut slope_cap = 1usize;
        let mut residual_cap = 1usize;
        let mut error_cap = 1usize;

        let include_prediction = sweep.include_prediction_in_deviation;

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            if length < 2 {
                return Err(CudaSmoothTheilSenError::InvalidInput(format!(
                    "invalid length: {length}"
                )));
            }
            let offset = combo.offset.unwrap_or(DEFAULT_OFFSET);
            let multiplier = combo.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
            if !multiplier.is_finite() || multiplier < 0.0 {
                return Err(CudaSmoothTheilSenError::InvalidInput(format!(
                    "invalid multiplier: {multiplier}"
                )));
            }
            let needed = length + offset;
            let valid = cols.saturating_sub(first);
            if valid < needed {
                return Err(CudaSmoothTheilSenError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }

            let pair_count = length
                .checked_mul(length - 1)
                .map(|value| value / 2)
                .ok_or(LaunchPlanError::SizeOverflow {
                    indicator: INDICATOR,
                    what: "pairwise slope count",
                })?;
            let error_len = if include_prediction {
                length + offset
            } else {
                length
            };
            slope_cap = slope_cap.max(pair_count);
            residual_cap = residual_cap.max(length);
            error_cap = error_cap.max(error_len);

            lengths.push(length as i32);
            offsets.push(offset as i32);
            multipliers.push(multiplier);
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let doubles_per_slot = 2 * slope_cap + 2 * residual_cap + 2 * error_cap;
        let bytes_per_slot = checked_mul(INDICATOR, "bytes/slot", doubles_per_slot, f64_size)?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 6 * f64_size)?
            .checked_add(cols * f64_size)
            .and_then(|b| {
                rows.checked_mul(2 * i32_size + f64_size)
                    .and_then(|c| b.checked_add(c))
            })
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "fixed bytes",
            })?;

        let plan = plan_slots(
            INDICATOR,
            rows,
            fixed_bytes,
            bytes_per_slot,
            DEFAULT_HEADROOM,
        )?;
        let scratch_len = scratch_elems(INDICATOR, "scratch", plan.slots, doubles_per_slot)?;

        let func = self
            .module
            .get_function(KERNEL)
            .map_err(|_| CudaSmoothTheilSenError::MissingKernelSymbol { name: KERNEL })?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_offsets = DeviceBuffer::from_slice(&offsets)?;
        let d_multipliers = DeviceBuffer::from_slice(&multipliers)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len.max(1))? };
        let d_value = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_slope = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_intercept = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_deviation = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        validate_launch(self.device_id, plan.grid, plan.block)?;
        let stream = &self.stream;
        let grid = plan.grid;
        let block = plan.block;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                first as i32,
                d_lengths.as_device_ptr(),
                d_offsets.as_device_ptr(),
                d_multipliers.as_device_ptr(),
                style_code(sweep.slope_style),
                style_code(sweep.residual_style),
                deviation_code(sweep.deviation_style),
                style_code(sweep.mad_style),
                i32::from(include_prediction),
                rows as i32,
                plan.slots as i32,
                slope_cap as i32,
                residual_cap as i32,
                error_cap as i32,
                d_scratch.as_device_ptr(),
                d_value.as_device_ptr(),
                d_upper.as_device_ptr(),
                d_lower.as_device_ptr(),
                d_slope.as_device_ptr(),
                d_intercept.as_device_ptr(),
                d_deviation.as_device_ptr()
            ))
            .map_err(|source| CudaSmoothTheilSenError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaSmoothTheilSenError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;

        let shape = |buf: DeviceBuffer<f64>| SmoothTheilSenDeviceArrayF64 { buf, rows, cols };

        Ok(CudaSmoothTheilSenBatchResult {
            outputs: SmoothTheilSenDeviceOutputs {
                value: shape(d_value),
                upper: shape(d_upper),
                lower: shape(d_lower),
                slope: shape(d_slope),
                intercept: shape(d_intercept),
                deviation: shape(d_deviation),
            },
            combos,
        })
    }
}
