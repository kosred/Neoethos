#![cfg(feature = "cuda")]

//! `ict_propulsion_block` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved `ict_propulsion_block_batch_f64` — a one-line EMPTY
//! kernel — discarded the function, computed all twelve output series on the
//! host through `Kernel::ScalarBatch`, and uploaded them.
//!
//! # What it is now
//!
//! A real kernel in `kernels/cuda/ict_propulsion_block_kernel.cu`, launched
//! from here. The CPU path is untouched and stays correct with no card; it is
//! unreachable from this file because a `CudaIctPropulsionBlock` only exists
//! once a device context has been created.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::ict_propulsion_block::{
    expand_grid_ict_propulsion_block, IctPropulsionBlockBatchRange,
    IctPropulsionBlockMitigationPrice, IctPropulsionBlockParams,
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

const INDICATOR: &str = "ict_propulsion_block";
const KERNEL: &str = "ict_propulsion_block_batch_f64";
const DEFAULT_SWING_LENGTH: usize = 3;

/// The kernel's `ICT_MITIGATION_*` codes.
fn mitigation_code(price: IctPropulsionBlockMitigationPrice) -> i32 {
    match price {
        IctPropulsionBlockMitigationPrice::Close => 0,
        IctPropulsionBlockMitigationPrice::Wick => 1,
    }
}

#[derive(Debug, Error)]
pub enum CudaIctPropulsionBlockError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("ict_propulsion_block: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct IctPropulsionBlockDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl IctPropulsionBlockDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct IctPropulsionBlockDeviceOutputs {
    pub bullish_high: IctPropulsionBlockDeviceArrayF64,
    pub bullish_low: IctPropulsionBlockDeviceArrayF64,
    pub bullish_kind: IctPropulsionBlockDeviceArrayF64,
    pub bullish_active: IctPropulsionBlockDeviceArrayF64,
    pub bullish_mitigated: IctPropulsionBlockDeviceArrayF64,
    pub bullish_new: IctPropulsionBlockDeviceArrayF64,
    pub bearish_high: IctPropulsionBlockDeviceArrayF64,
    pub bearish_low: IctPropulsionBlockDeviceArrayF64,
    pub bearish_kind: IctPropulsionBlockDeviceArrayF64,
    pub bearish_active: IctPropulsionBlockDeviceArrayF64,
    pub bearish_mitigated: IctPropulsionBlockDeviceArrayF64,
    pub bearish_new: IctPropulsionBlockDeviceArrayF64,
}

impl IctPropulsionBlockDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.bullish_high.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.bullish_high.cols
    }
}

pub struct CudaIctPropulsionBlockBatchResult {
    pub outputs: IctPropulsionBlockDeviceOutputs,
    pub combos: Vec<IctPropulsionBlockParams>,
}

pub struct CudaIctPropulsionBlock {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaIctPropulsionBlock {
    pub fn new(device_id: usize) -> Result<Self, CudaIctPropulsionBlockError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ict_propulsion_block_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaIctPropulsionBlockError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &IctPropulsionBlockBatchRange,
    ) -> Result<CudaIctPropulsionBlockBatchResult, CudaIctPropulsionBlockError> {
        let cols = close.len();
        if cols == 0 || open.is_empty() || high.is_empty() || low.is_empty() {
            return Err(CudaIctPropulsionBlockError::InvalidInput("empty input".into()));
        }
        if open.len() != cols || high.len() != cols || low.len() != cols {
            return Err(CudaIctPropulsionBlockError::InvalidInput(format!(
                "data length mismatch: open={} high={} low={} close={}",
                open.len(),
                high.len(),
                low.len(),
                cols
            )));
        }
        // first_valid_bar (:389) — at least one bar the state machine can start on.
        if !(0..cols).any(|i| {
            open[i].is_finite()
                && high[i].is_finite()
                && low[i].is_finite()
                && close[i].is_finite()
                && high[i] >= low[i]
        }) {
            return Err(CudaIctPropulsionBlockError::InvalidInput(
                "no valid bar in input".into(),
            ));
        }

        let combos = expand_grid_ict_propulsion_block(sweep)
            .map_err(|e| CudaIctPropulsionBlockError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaIctPropulsionBlockError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let mut swing_lengths = Vec::with_capacity(rows);
        let mut mitigations = Vec::with_capacity(rows);
        let mut deque_cap = 2usize;
        for combo in &combos {
            let swing_length = combo.swing_length.unwrap_or(DEFAULT_SWING_LENGTH);
            if swing_length == 0 {
                return Err(CudaIctPropulsionBlockError::InvalidInput(
                    "invalid swing_length: 0".into(),
                ));
            }
            // The monotonic deques hold at most `swing_length` entries after
            // expiry and `swing_length + 1` between the push and the sweep.
            deque_cap = deque_cap.max(swing_length + 1);
            swing_lengths.push(swing_length as i32);
            mitigations.push(mitigation_code(combo.mitigation_price.unwrap_or_default()));
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let ints_per_slot = checked_mul(INDICATOR, "deques/slot", 2, deque_cap)?;
        let bytes_per_slot = checked_mul(INDICATOR, "bytes/slot", ints_per_slot, i32_size)?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 12 * f64_size)?
            .checked_add(cols * 4 * f64_size)
            .and_then(|b| rows.checked_mul(2 * i32_size).and_then(|c| b.checked_add(c)))
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "fixed bytes",
            })?;

        let plan = plan_slots(INDICATOR, rows, fixed_bytes, bytes_per_slot, DEFAULT_HEADROOM)?;
        let scratch_ints = scratch_elems(INDICATOR, "int scratch", plan.slots, ints_per_slot)?;

        let func = self
            .module
            .get_function(KERNEL)
            .map_err(|_| CudaIctPropulsionBlockError::MissingKernelSymbol { name: KERNEL })?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_swing = DeviceBuffer::from_slice(&swing_lengths)?;
        let d_mitigation = DeviceBuffer::from_slice(&mitigations)?;
        let d_iscratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_ints.max(1))? };

        let mut outs = Vec::with_capacity(12);
        for _ in 0..12 {
            outs.push(unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? });
        }

        validate_launch(self.device_id, plan.grid, plan.block)?;
        let stream = &self.stream;
        let grid = plan.grid;
        let block = plan.block;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_swing.as_device_ptr(),
                d_mitigation.as_device_ptr(),
                rows as i32,
                plan.slots as i32,
                deque_cap as i32,
                d_iscratch.as_device_ptr(),
                outs[0].as_device_ptr(),
                outs[1].as_device_ptr(),
                outs[2].as_device_ptr(),
                outs[3].as_device_ptr(),
                outs[4].as_device_ptr(),
                outs[5].as_device_ptr(),
                outs[6].as_device_ptr(),
                outs[7].as_device_ptr(),
                outs[8].as_device_ptr(),
                outs[9].as_device_ptr(),
                outs[10].as_device_ptr(),
                outs[11].as_device_ptr()
            ))
            .map_err(|source| CudaIctPropulsionBlockError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaIctPropulsionBlockError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;

        outs.reverse();
        let mut next = move || -> Result<DeviceBuffer<f64>, CudaIctPropulsionBlockError> {
            outs.pop().ok_or_else(|| {
                CudaIctPropulsionBlockError::InvalidInput(
                    "internal: fewer output buffers than outputs".into(),
                )
            })
        };
        let shape = |buf: DeviceBuffer<f64>| IctPropulsionBlockDeviceArrayF64 { buf, rows, cols };

        let bullish_high = shape(next()?);
        let bullish_low = shape(next()?);
        let bullish_kind = shape(next()?);
        let bullish_active = shape(next()?);
        let bullish_mitigated = shape(next()?);
        let bullish_new = shape(next()?);
        let bearish_high = shape(next()?);
        let bearish_low = shape(next()?);
        let bearish_kind = shape(next()?);
        let bearish_active = shape(next()?);
        let bearish_mitigated = shape(next()?);
        let bearish_new = shape(next()?);

        Ok(CudaIctPropulsionBlockBatchResult {
            outputs: IctPropulsionBlockDeviceOutputs {
                bullish_high,
                bullish_low,
                bullish_kind,
                bullish_active,
                bullish_mitigated,
                bullish_new,
                bearish_high,
                bearish_low,
                bearish_kind,
                bearish_active,
                bearish_mitigated,
                bearish_new,
            },
            combos,
        })
    }
}
