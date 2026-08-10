#![cfg(feature = "cuda")]

//! `vdubus_divergence_wave_pattern_generator` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved `vdubus_divergence_wave_pattern_generator_batch_f64` —
//! a one-line EMPTY kernel — discarded the function, computed all twelve output
//! series on the host through `Kernel::ScalarBatch`, and uploaded them.
//!
//! # What it is now
//!
//! A real kernel in
//! `kernels/cuda/vdubus_divergence_wave_pattern_generator_kernel.cu`, launched
//! from here. The CPU path is untouched and stays correct with no card; it is
//! unreachable from this file because this type only exists once a device
//! context has been created.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::vdubus_divergence_wave_pattern_generator::{
    expand_grid_vdubus_divergence_wave_pattern_generator,
    VdubusDivergenceWavePatternGeneratorBatchRange, VdubusDivergenceWavePatternGeneratorParams,
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

const INDICATOR: &str = "vdubus_divergence_wave_pattern_generator";
const KERNEL: &str = "vdubus_divergence_wave_pattern_generator_batch_f64";

const DEFAULT_FAST_DEPTH: usize = 9;
const DEFAULT_SLOW_DEPTH: usize = 24;
const DEFAULT_FAST_LENGTH: usize = 21;
const DEFAULT_SLOW_LENGTH: usize = 34;
const DEFAULT_SIGNAL_LENGTH: usize = 5;
const DEFAULT_LOOKBACK: usize = 3;
const DEFAULT_ERR_TOL: f64 = 0.15;

#[derive(Debug, Error)]
pub enum CudaVdubusDivergenceWavePatternGeneratorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error(
        "vdubus_divergence_wave_pattern_generator: CUDA kernel `{kernel}` failed to launch: \
         {source}"
    )]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct VdubusDivergenceWavePatternGeneratorDeviceOutputs {
    pub fast_standard: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub fast_climax: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub fast_rounded: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub fast_predator: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_standard: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_climax: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_rounded: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub slow_predator: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub opposing_force: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub macd: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub signal: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
    pub hist: VdubusDivergenceWavePatternGeneratorDeviceArrayF64,
}

impl VdubusDivergenceWavePatternGeneratorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.fast_standard.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.fast_standard.cols
    }
}

pub struct CudaVdubusDivergenceWavePatternGeneratorBatchResult {
    pub outputs: VdubusDivergenceWavePatternGeneratorDeviceOutputs,
    pub combos: Vec<VdubusDivergenceWavePatternGeneratorParams>,
}

pub struct CudaVdubusDivergenceWavePatternGenerator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVdubusDivergenceWavePatternGenerator {
    pub fn new(device_id: usize) -> Result<Self, CudaVdubusDivergenceWavePatternGeneratorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("vdubus_divergence_wave_pattern_generator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVdubusDivergenceWavePatternGeneratorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    ) -> Result<
        CudaVdubusDivergenceWavePatternGeneratorBatchResult,
        CudaVdubusDivergenceWavePatternGeneratorError,
    > {
        let cols = close.len();
        if cols == 0 || high.is_empty() || low.is_empty() {
            return Err(CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != cols || low.len() != cols {
            return Err(CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(
                format!(
                    "data length mismatch: high={} low={} close={}",
                    high.len(),
                    low.len(),
                    cols
                ),
            ));
        }

        let combos = expand_grid_vdubus_divergence_wave_pattern_generator(sweep).map_err(|e| {
            CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(e.to_string())
        })?;
        if combos.is_empty() {
            return Err(CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let mut fast_depths = Vec::with_capacity(rows);
        let mut slow_depths = Vec::with_capacity(rows);
        let mut fast_lengths = Vec::with_capacity(rows);
        let mut slow_lengths = Vec::with_capacity(rows);
        let mut signal_lengths = Vec::with_capacity(rows);
        let mut lookbacks = Vec::with_capacity(rows);
        let mut err_tols = Vec::with_capacity(rows);
        let mut window_cap = 3usize;

        for combo in &combos {
            let fast_depth = combo.fast_depth.unwrap_or(DEFAULT_FAST_DEPTH);
            let slow_depth = combo.slow_depth.unwrap_or(DEFAULT_SLOW_DEPTH);
            let fast_length = combo.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
            let slow_length = combo.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
            let signal_length = combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
            let lookback = combo.lookback.unwrap_or(DEFAULT_LOOKBACK);
            let err_tol = combo.err_tol.unwrap_or(DEFAULT_ERR_TOL);

            for (name, value) in [
                ("fast_depth", fast_depth),
                ("slow_depth", slow_depth),
                ("fast_length", fast_length),
                ("slow_length", slow_length),
                ("signal_length", signal_length),
                ("lookback", lookback),
            ] {
                if value == 0 {
                    return Err(CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(
                        format!("invalid {name}: 0"),
                    ));
                }
            }
            if !err_tol.is_finite() {
                return Err(CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(
                    format!("invalid err_tol: {err_tol}"),
                ));
            }

            // Each pivot detector's window is `2 * span + 1`.
            window_cap = window_cap
                .max(2 * fast_depth + 1)
                .max(2 * slow_depth + 1)
                .max(2 * lookback + 1);

            fast_depths.push(fast_depth as i32);
            slow_depths.push(slow_depth as i32);
            fast_lengths.push(fast_length as i32);
            slow_lengths.push(slow_length as i32);
            signal_lengths.push(signal_length as i32);
            lookbacks.push(lookback as i32);
            err_tols.push(err_tol);
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let doubles_per_slot = checked_mul(INDICATOR, "pivot windows/slot", 6, window_cap)?;
        let bytes_per_slot = checked_mul(INDICATOR, "bytes/slot", doubles_per_slot, f64_size)?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 12 * f64_size)?
            .checked_add(cols * 3 * f64_size)
            .and_then(|b| {
                rows.checked_mul(6 * i32_size + f64_size)
                    .and_then(|c| b.checked_add(c))
            })
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "fixed bytes",
            })?;

        let plan = plan_slots(INDICATOR, rows, fixed_bytes, bytes_per_slot, DEFAULT_HEADROOM)?;
        let scratch_len = scratch_elems(INDICATOR, "scratch", plan.slots, doubles_per_slot)?;

        let func = self.module.get_function(KERNEL).map_err(|_| {
            CudaVdubusDivergenceWavePatternGeneratorError::MissingKernelSymbol { name: KERNEL }
        })?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_fast_depth = DeviceBuffer::from_slice(&fast_depths)?;
        let d_slow_depth = DeviceBuffer::from_slice(&slow_depths)?;
        let d_fast_length = DeviceBuffer::from_slice(&fast_lengths)?;
        let d_slow_length = DeviceBuffer::from_slice(&slow_lengths)?;
        let d_signal_length = DeviceBuffer::from_slice(&signal_lengths)?;
        let d_lookback = DeviceBuffer::from_slice(&lookbacks)?;
        let d_err_tol = DeviceBuffer::from_slice(&err_tols)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len.max(1))? };

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
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_fast_depth.as_device_ptr(),
                d_slow_depth.as_device_ptr(),
                d_fast_length.as_device_ptr(),
                d_slow_length.as_device_ptr(),
                d_signal_length.as_device_ptr(),
                d_lookback.as_device_ptr(),
                d_err_tol.as_device_ptr(),
                i32::from(sweep.show_standard),
                i32::from(sweep.show_climax),
                i32::from(sweep.show_rounded),
                i32::from(sweep.show_predator),
                i32::from(sweep.show_gartley),
                i32::from(sweep.show_bat),
                i32::from(sweep.show_butterfly),
                i32::from(sweep.show_crab),
                i32::from(sweep.show_deep),
                i32::from(sweep.show_hs),
                rows as i32,
                plan.slots as i32,
                window_cap as i32,
                d_scratch.as_device_ptr(),
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
            .map_err(
                |source| CudaVdubusDivergenceWavePatternGeneratorError::LaunchFailed {
                    kernel: KERNEL,
                    source,
                },
            )?;
        }

        self.stream.synchronize().map_err(|source| {
            CudaVdubusDivergenceWavePatternGeneratorError::LaunchFailed {
                kernel: KERNEL,
                source,
            }
        })?;

        outs.reverse();
        let mut next = move || -> Result<
            DeviceBuffer<f64>,
            CudaVdubusDivergenceWavePatternGeneratorError,
        > {
            outs.pop().ok_or_else(|| {
                CudaVdubusDivergenceWavePatternGeneratorError::InvalidInput(
                    "internal: fewer output buffers than outputs".into(),
                )
            })
        };
        let shape = |buf: DeviceBuffer<f64>| VdubusDivergenceWavePatternGeneratorDeviceArrayF64 {
            buf,
            rows,
            cols,
        };

        let fast_standard = shape(next()?);
        let fast_climax = shape(next()?);
        let fast_rounded = shape(next()?);
        let fast_predator = shape(next()?);
        let slow_standard = shape(next()?);
        let slow_climax = shape(next()?);
        let slow_rounded = shape(next()?);
        let slow_predator = shape(next()?);
        let opposing_force = shape(next()?);
        let macd = shape(next()?);
        let signal = shape(next()?);
        let hist = shape(next()?);

        Ok(CudaVdubusDivergenceWavePatternGeneratorBatchResult {
            outputs: VdubusDivergenceWavePatternGeneratorDeviceOutputs {
                fast_standard,
                fast_climax,
                fast_rounded,
                fast_predator,
                slow_standard,
                slow_climax,
                slow_rounded,
                slow_predator,
                opposing_force,
                macd,
                signal,
                hist,
            },
            combos,
        })
    }
}
