#![cfg(feature = "cuda")]

//! `rogers_satchell_volatility` on the card.
//!
//! # What this used to be
//!
//! The most complete disguise in the crate, and the one the inventory's
//! `fallback.py` missed because its filter required at least one
//! `get_function` call — this wrapper had NONE, and there was no `.cu` file for
//! the indicator at all. `compute_rs_row` and `compute_signal_row` ran the
//! whole calculation on the HOST, and all three public entry points finished
//! with `DeviceBuffer::from_slice(&host_rs)`, handing the caller device
//! pointers the card had never written. `batch_from_device` was the starkest:
//! it copied device buffers DOWN to the host, computed there, and copied the
//! answer back up.
//!
//! # What it is now
//!
//! `kernels/cuda/rogers_satchell_volatility_kernel.cu` carries five entry
//! points and this wrapper launches them. `batch_from_device` no longer touches
//! the host at all — the device buffers it is handed go straight into the
//! kernel.
//!
//! # Where the CPU path went
//!
//! `src/indicators/rogers_satchell_volatility.rs` still holds the CPU
//! implementation and it is still the correct path on a machine with no card.
//! It is not reachable from this file: a `CudaRogersSatchellVolatility` only
//! exists once `Context::new` has succeeded on a real device. A launch failure
//! is an `Err` naming the indicator, never a quiet host recomputation.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::rogers_satchell_volatility::{
    RogersSatchellVolatilityBatchRange, RogersSatchellVolatilityParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const INDICATOR: &str = "rogers_satchell_volatility";
const KERNEL_PREFIX_F32: &str = "rogers_satchell_prefix_f32in";
const KERNEL_BATCH: &str = "rogers_satchell_volatility_batch_f64";
const KERNEL_NARROW: &str = "rogers_satchell_narrow_f64_to_f32";
const KERNEL_MANY: &str = "rogers_satchell_many_series_time_major_f32in";
const DEFAULT_LOOKBACK: usize = 8;
const DEFAULT_SIGNAL_LENGTH: usize = 8;

#[derive(Debug, Error)]
pub enum CudaRogersSatchellVolatilityError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("rogers_satchell_volatility: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct DeviceArrayF32Pair {
    pub rs: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}

pub struct CudaRogersSatchellBatchResult {
    pub outputs: DeviceArrayF32Pair,
    pub combos: Vec<RogersSatchellVolatilityParams>,
}

pub struct CudaRogersSatchellManySeriesResult {
    pub rs: DeviceArrayF32,
    pub signal: DeviceArrayF32,
}

pub struct CudaRogersSatchellVolatility {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaRogersSatchellVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaRogersSatchellVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("rogers_satchell_volatility_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> {
        self._context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaRogersSatchellVolatilityError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaRogersSatchellVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let step = step.max(1);
        if start < end {
            let mut values = Vec::new();
            let mut current = start;
            while current <= end {
                values.push(current);
                match current.checked_add(step) {
                    Some(next) if next != current => current = next,
                    _ => break,
                }
            }
            if values.is_empty() {
                return Err(CudaRogersSatchellVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(values)
        } else {
            let mut values = Vec::new();
            let mut current = start;
            loop {
                values.push(current);
                if current == end {
                    break;
                }
                let next = current.saturating_sub(step);
                if next == current || next < end {
                    break;
                }
                current = next;
            }
            if values.is_empty() {
                return Err(CudaRogersSatchellVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(values)
        }
    }

    fn expand_grid(
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<Vec<RogersSatchellVolatilityParams>, CudaRogersSatchellVolatilityError> {
        let lookbacks = Self::axis_usize(sweep.lookback)?;
        let signal_lengths = Self::axis_usize(sweep.signal_length)?;
        let mut combos = Vec::with_capacity(lookbacks.len() * signal_lengths.len());
        for &lookback in &lookbacks {
            for &signal_length in &signal_lengths {
                combos.push(RogersSatchellVolatilityParams {
                    lookback: Some(lookback),
                    signal_length: Some(signal_length),
                });
            }
        }
        Ok(combos)
    }

    /// Per-row `(lookback, signal_length)` vectors from the same grid the CPU
    /// expands, in the same order, so combo `i` names output row `i`.
    fn row_params(
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<
        (Vec<RogersSatchellVolatilityParams>, Vec<i32>, Vec<i32>),
        CudaRogersSatchellVolatilityError,
    > {
        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "empty parameter grid".to_string(),
            ));
        }
        let mut lookbacks = Vec::with_capacity(combos.len());
        let mut signal_lengths = Vec::with_capacity(combos.len());
        for combo in &combos {
            let lookback = combo.lookback.unwrap_or(DEFAULT_LOOKBACK);
            let signal_length = combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
            if lookback == 0 {
                return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                    "invalid lookback: 0".to_string(),
                ));
            }
            lookbacks.push(lookback as i32);
            signal_lengths.push(signal_length as i32);
        }
        Ok((combos, lookbacks, signal_lengths))
    }

    /// Prefix pass, row pass, then narrowing — all three on the device.
    ///
    /// The prefix arrays do not depend on any swept parameter, so they are
    /// built ONCE and every row reads them. That is also what keeps the kernel
    /// comparable with the CPU bit for bit: `compute_rs_row` takes window
    /// differences of exactly these prefix sums, and an incremental rolling sum
    /// would round differently.
    fn run_batch_device(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        cols: usize,
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<CudaRogersSatchellBatchResult, CudaRogersSatchellVolatilityError> {
        let (combos, lookbacks, signal_lengths) = Self::row_params(sweep)?;
        let rows = combos.len();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let f64_size = std::mem::size_of::<f64>();
        let f32_size = std::mem::size_of::<f32>();
        let i32_size = std::mem::size_of::<i32>();
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 2 * f64_size)?
            .checked_add(checked_mul(
                INDICATOR,
                "narrowed bytes",
                output_elems,
                2 * f32_size,
            )?)
            .and_then(|b| {
                (cols + 1)
                    .checked_mul(f64_size + i32_size)
                    .and_then(|c| b.checked_add(c))
            })
            .and_then(|b| cols.checked_mul(4 * f32_size).and_then(|c| b.checked_add(c)))
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "fixed bytes",
            })?;
        // The row kernel keeps no per-row scratch — it writes straight into the
        // output matrices — so the slot count is bounded only by the row count.
        let plan = plan_slots(INDICATOR, rows, fixed_bytes, 0, DEFAULT_HEADROOM)?;

        let prefix_fn = self.module.get_function(KERNEL_PREFIX_F32).map_err(|_| {
            CudaRogersSatchellVolatilityError::MissingKernelSymbol {
                name: KERNEL_PREFIX_F32,
            }
        })?;
        let batch_fn = self.module.get_function(KERNEL_BATCH).map_err(|_| {
            CudaRogersSatchellVolatilityError::MissingKernelSymbol { name: KERNEL_BATCH }
        })?;
        let narrow_fn = self.module.get_function(KERNEL_NARROW).map_err(|_| {
            CudaRogersSatchellVolatilityError::MissingKernelSymbol {
                name: KERNEL_NARROW,
            }
        })?;

        let d_prefix_sum = unsafe { DeviceBuffer::<f64>::uninitialized(cols + 1)? };
        let d_prefix_valid = unsafe { DeviceBuffer::<i32>::uninitialized(cols + 1)? };
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let d_signal_lengths = DeviceBuffer::from_slice(&signal_lengths)?;
        let d_rs64 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_signal64 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_rs32 = unsafe { DeviceBuffer::<f32>::uninitialized(output_elems)? };
        let d_signal32 = unsafe { DeviceBuffer::<f32>::uninitialized(output_elems)? };

        let stream = &self.stream;
        let one = GridSize::x(1);
        let one_thread = BlockSize::x(1);
        validate_launch(self.device_id, one, one_thread)?;
        unsafe {
            launch!(prefix_fn<<<one, one_thread, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_prefix_sum.as_device_ptr(),
                d_prefix_valid.as_device_ptr()
            ))
            .map_err(|source| CudaRogersSatchellVolatilityError::LaunchFailed {
                kernel: KERNEL_PREFIX_F32,
                source,
            })?;
        }

        let grid = plan.grid;
        let block = plan.block;
        validate_launch(self.device_id, grid, block)?;
        unsafe {
            launch!(batch_fn<<<grid, block, 0, stream>>>(
                cols as i32,
                d_prefix_sum.as_device_ptr(),
                d_prefix_valid.as_device_ptr(),
                d_lookbacks.as_device_ptr(),
                d_signal_lengths.as_device_ptr(),
                rows as i32,
                plan.slots as i32,
                d_rs64.as_device_ptr(),
                d_signal64.as_device_ptr()
            ))
            .map_err(|source| CudaRogersSatchellVolatilityError::LaunchFailed {
                kernel: KERNEL_BATCH,
                source,
            })?;
        }

        let narrow_blocks = ((output_elems as u64) + 255) / 256;
        let narrow_grid = GridSize::x(narrow_blocks.clamp(1, 65_535) as u32);
        let narrow_block = BlockSize::x(256);
        validate_launch(self.device_id, narrow_grid, narrow_block)?;
        for (src, dst) in [(&d_rs64, &d_rs32), (&d_signal64, &d_signal32)] {
            unsafe {
                launch!(narrow_fn<<<narrow_grid, narrow_block, 0, stream>>>(
                    src.as_device_ptr(),
                    dst.as_device_ptr(),
                    output_elems as i64
                ))
                .map_err(|source| CudaRogersSatchellVolatilityError::LaunchFailed {
                    kernel: KERNEL_NARROW,
                    source,
                })?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaRogersSatchellVolatilityError::LaunchFailed {
                kernel: KERNEL_BATCH,
                source,
            })?;

        Ok(CudaRogersSatchellBatchResult {
            outputs: DeviceArrayF32Pair {
                rs: DeviceArrayF32 {
                    buf: d_rs32,
                    rows,
                    cols,
                },
                signal: DeviceArrayF32 {
                    buf: d_signal32,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }

    pub fn rogers_satchell_volatility_batch_dev(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<CudaRogersSatchellBatchResult, CudaRogersSatchellVolatilityError> {
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "OHLC slice length mismatch".to_string(),
            ));
        }
        let cols = close.len();
        if cols == 0 {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "empty input".to_string(),
            ));
        }
        // Uploading the INPUTS is a transfer, not a computation. Nothing that
        // leaves this function was calculated on the host.
        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        self.run_batch_device(&d_open, &d_high, &d_low, &d_close, cols, sweep)
    }

    pub fn rogers_satchell_volatility_batch_from_device(
        &self,
        open: &DeviceBuffer<f32>,
        high: &DeviceBuffer<f32>,
        low: &DeviceBuffer<f32>,
        close: &DeviceBuffer<f32>,
        _first_valid: usize,
        sweep: &RogersSatchellVolatilityBatchRange,
    ) -> Result<CudaRogersSatchellBatchResult, CudaRogersSatchellVolatilityError> {
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "device OHLC length mismatch".to_string(),
            ));
        }
        // The old body copied all four buffers DOWN to the host, computed
        // there, and copied the answer back up. The data now never leaves the
        // card.
        self.run_batch_device(open, high, low, close, close.len(), sweep)
    }

    pub fn rogers_satchell_volatility_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        lookback: usize,
        signal_length: usize,
    ) -> Result<CudaRogersSatchellManySeriesResult, CudaRogersSatchellVolatilityError> {
        if open_tm.len() != high_tm.len()
            || open_tm.len() != low_tm.len()
            || open_tm.len() != close_tm.len()
            || open_tm.len() != cols.saturating_mul(rows)
        {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "time-major OHLC shape mismatch".to_string(),
            ));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaRogersSatchellVolatilityError::InvalidInput(
                "empty time-major input".to_string(),
            ));
        }

        let total = open_tm.len();
        let f64_size = std::mem::size_of::<f64>();
        let f32_size = std::mem::size_of::<f32>();
        let i32_size = std::mem::size_of::<i32>();
        // One prefix pair per CONCURRENT SERIES, so the planner bounds it by
        // free VRAM rather than by how many series the caller passed.
        let bytes_per_slot =
            (rows + 1)
                .checked_mul(f64_size + i32_size)
                .ok_or(LaunchPlanError::SizeOverflow {
                    indicator: INDICATOR,
                    what: "prefix bytes/slot",
                })?;
        let fixed_bytes = checked_mul(INDICATOR, "io bytes", total, 6 * f32_size)?;
        let plan = plan_slots(INDICATOR, cols, fixed_bytes, bytes_per_slot, DEFAULT_HEADROOM)?;

        let func = self.module.get_function(KERNEL_MANY).map_err(|_| {
            CudaRogersSatchellVolatilityError::MissingKernelSymbol { name: KERNEL_MANY }
        })?;

        let d_open = DeviceBuffer::from_slice(open_tm)?;
        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_close = DeviceBuffer::from_slice(close_tm)?;
        let scratch_len = checked_mul(INDICATOR, "prefix scratch", plan.slots, rows + 1)?;
        let d_scratch_sum = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len.max(1))? };
        let d_scratch_valid = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_len.max(1))? };
        let d_rs = unsafe { DeviceBuffer::<f32>::uninitialized(total)? };
        let d_signal = unsafe { DeviceBuffer::<f32>::uninitialized(total)? };

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
                rows as i32,
                lookback as i32,
                signal_length as i32,
                plan.slots as i32,
                d_scratch_sum.as_device_ptr(),
                d_scratch_valid.as_device_ptr(),
                d_rs.as_device_ptr(),
                d_signal.as_device_ptr()
            ))
            .map_err(|source| CudaRogersSatchellVolatilityError::LaunchFailed {
                kernel: KERNEL_MANY,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaRogersSatchellVolatilityError::LaunchFailed {
                kernel: KERNEL_MANY,
                source,
            })?;

        Ok(CudaRogersSatchellManySeriesResult {
            rs: DeviceArrayF32 {
                buf: d_rs,
                rows,
                cols,
            },
            signal: DeviceArrayF32 {
                buf: d_signal,
                rows,
                cols,
            },
        })
    }
}
