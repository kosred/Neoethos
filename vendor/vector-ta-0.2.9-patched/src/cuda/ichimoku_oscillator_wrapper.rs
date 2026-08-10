#![cfg(feature = "cuda")]

//! `ichimoku_oscillator` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved the symbol `ichimoku_oscillator_batch_f64` — a one-line
//! EMPTY kernel — threw the function away, computed all thirteen output series
//! on the host through `Kernel::ScalarBatch`, and uploaded them with thirteen
//! `DeviceBuffer::from_slice` calls. Thirteen device pointers, none of which the
//! card had written.
//!
//! # What it is now
//!
//! A real kernel in `kernels/cuda/ichimoku_oscillator_kernel.cu`, launched from
//! here. The CPU implementation is unchanged and is still the right answer with
//! no card; it is not reachable from this file, because a
//! `CudaIchimokuOscillator` only exists once a device context has been created.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::ichimoku_oscillator::{
    expand_grid as expand_grid_ichimoku, IchimokuOscillatorBatchRange,
    IchimokuOscillatorNormalizeMode, IchimokuOscillatorParams,
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

const INDICATOR: &str = "ichimoku_oscillator";
const KERNEL: &str = "ichimoku_oscillator_batch_f64";

/// Must match `ICH_ARRAYS` in the kernel.
const SCRATCH_ARRAYS: usize = 14;

const DEFAULT_CONVERSION_PERIODS: usize = 9;
const DEFAULT_BASE_PERIODS: usize = 26;
const DEFAULT_LAGGING_SPAN_PERIODS: usize = 52;
const DEFAULT_DISPLACEMENT: usize = 26;
const DEFAULT_MA_LENGTH: usize = 12;
const DEFAULT_SMOOTHING_LENGTH: usize = 3;
const DEFAULT_WINDOW_SIZE: usize = 20;
const DEFAULT_TOP_BAND: f64 = 2.0;
const DEFAULT_MID_BAND: f64 = 1.5;

/// The kernel's `ICH_NORM_*` codes, spelled out rather than cast.
fn normalize_code(mode: IchimokuOscillatorNormalizeMode) -> i32 {
    match mode {
        IchimokuOscillatorNormalizeMode::All => 0,
        IchimokuOscillatorNormalizeMode::Window => 1,
        IchimokuOscillatorNormalizeMode::Disabled => 2,
    }
}

#[derive(Debug, Error)]
pub enum CudaIchimokuOscillatorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("ichimoku_oscillator: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct IchimokuOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl IchimokuOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct IchimokuOscillatorDeviceOutputs {
    pub signal: IchimokuOscillatorDeviceArrayF64,
    pub ma: IchimokuOscillatorDeviceArrayF64,
    pub conversion: IchimokuOscillatorDeviceArrayF64,
    pub base: IchimokuOscillatorDeviceArrayF64,
    pub chikou: IchimokuOscillatorDeviceArrayF64,
    pub current_kumo_a: IchimokuOscillatorDeviceArrayF64,
    pub current_kumo_b: IchimokuOscillatorDeviceArrayF64,
    pub future_kumo_a: IchimokuOscillatorDeviceArrayF64,
    pub future_kumo_b: IchimokuOscillatorDeviceArrayF64,
    pub max_level: IchimokuOscillatorDeviceArrayF64,
    pub high_level: IchimokuOscillatorDeviceArrayF64,
    pub low_level: IchimokuOscillatorDeviceArrayF64,
    pub min_level: IchimokuOscillatorDeviceArrayF64,
}

impl IchimokuOscillatorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.signal.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.signal.cols
    }
}

pub struct CudaIchimokuOscillatorBatchResult {
    pub outputs: IchimokuOscillatorDeviceOutputs,
    pub combos: Vec<IchimokuOscillatorParams>,
}

pub struct CudaIchimokuOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaIchimokuOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaIchimokuOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ichimoku_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaIchimokuOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
        sweep: &IchimokuOscillatorBatchRange,
    ) -> Result<CudaIchimokuOscillatorBatchResult, CudaIchimokuOscillatorError> {
        let cols = close.len();
        if cols == 0 || high.is_empty() || low.is_empty() || source.is_empty() {
            return Err(CudaIchimokuOscillatorError::InvalidInput("empty input".into()));
        }
        if high.len() != cols || low.len() != cols || source.len() != cols {
            return Err(CudaIchimokuOscillatorError::InvalidInput(format!(
                "inconsistent slice lengths: high={} low={} close={} source={}",
                high.len(),
                low.len(),
                cols,
                source.len()
            )));
        }

        // first_valid_hlcs (:511): the first index at which ALL FOUR are finite.
        let first = (0..cols)
            .find(|&i| {
                high[i].is_finite()
                    && low[i].is_finite()
                    && close[i].is_finite()
                    && source[i].is_finite()
            })
            .ok_or_else(|| {
                CudaIchimokuOscillatorError::InvalidInput("all values are NaN".into())
            })?;

        let combos = expand_grid_ichimoku(sweep)
            .map_err(|e| CudaIchimokuOscillatorError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaIchimokuOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let mut conversion_periods = Vec::with_capacity(rows);
        let mut base_periods = Vec::with_capacity(rows);
        let mut lagging_span_periods = Vec::with_capacity(rows);
        let mut displacements = Vec::with_capacity(rows);
        let mut ma_lengths = Vec::with_capacity(rows);
        let mut smoothing_lengths = Vec::with_capacity(rows);
        let mut window_sizes = Vec::with_capacity(rows);
        let mut top_bands = Vec::with_capacity(rows);
        let mut mid_bands = Vec::with_capacity(rows);
        let mut deque_cap = 2usize;

        let normalize = sweep.normalize;
        let valid = cols.saturating_sub(first);

        for combo in &combos {
            let conv = combo.conversion_periods.unwrap_or(DEFAULT_CONVERSION_PERIODS);
            let base_p = combo.base_periods.unwrap_or(DEFAULT_BASE_PERIODS);
            let lag = combo
                .lagging_span_periods
                .unwrap_or(DEFAULT_LAGGING_SPAN_PERIODS);
            let displacement = combo.displacement.unwrap_or(DEFAULT_DISPLACEMENT);
            let ma_length = combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
            let smoothing = combo.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
            let window = combo.window_size.unwrap_or(DEFAULT_WINDOW_SIZE);
            let top_band = combo.top_band.unwrap_or(DEFAULT_TOP_BAND);
            let mid_band = combo.mid_band.unwrap_or(DEFAULT_MID_BAND);

            // ValidatedParams::from_params (:200)
            for (name, value) in [
                ("conversion_periods", conv),
                ("base_periods", base_p),
                ("lagging_span_periods", lag),
                ("displacement", displacement),
                ("ma_length", ma_length),
                ("smoothing_length", smoothing),
            ] {
                if value == 0 {
                    return Err(CudaIchimokuOscillatorError::InvalidInput(format!(
                        "invalid period {name}: 0"
                    )));
                }
            }
            if matches!(normalize, IchimokuOscillatorNormalizeMode::Window) && window < 5 {
                return Err(CudaIchimokuOscillatorError::InvalidInput(format!(
                    "invalid window_size: {window}"
                )));
            }
            for (name, value) in [("top_band", top_band), ("mid_band", mid_band)] {
                if !value.is_finite() || value < 0.0 {
                    return Err(CudaIchimokuOscillatorError::InvalidInput(format!(
                        "invalid band {name}: {value}"
                    )));
                }
            }

            // min_required_history (:283)
            let needed = lag
                .saturating_add(displacement)
                .saturating_sub(1)
                .max(base_p)
                .max(conv)
                .max(ma_length);
            if valid < needed {
                return Err(CudaIchimokuOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }

            deque_cap = deque_cap.max(conv.max(base_p).max(lag) + 1);

            conversion_periods.push(conv as i32);
            base_periods.push(base_p as i32);
            lagging_span_periods.push(lag as i32);
            displacements.push(displacement as i32);
            ma_lengths.push(ma_length as i32);
            smoothing_lengths.push(smoothing as i32);
            window_sizes.push(window as i32);
            top_bands.push(top_band);
            mid_bands.push(mid_band);
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let doubles_per_slot = checked_mul(INDICATOR, "scratch/slot", SCRATCH_ARRAYS, cols)?;
        let ints_per_slot = checked_mul(INDICATOR, "deques/slot", 2, deque_cap)?;
        let bytes_per_slot = checked_mul(INDICATOR, "double bytes/slot", doubles_per_slot, f64_size)?
            .checked_add(checked_mul(INDICATOR, "int bytes/slot", ints_per_slot, i32_size)?)
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "bytes/slot",
            })?;
        // Thirteen output matrices plus four input series plus nine per-row
        // parameter vectors.
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 13 * f64_size)?
            .checked_add(cols * 4 * f64_size)
            .and_then(|b| {
                rows.checked_mul(7 * i32_size + 2 * f64_size)
                    .and_then(|c| b.checked_add(c))
            })
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "fixed bytes",
            })?;

        let plan = plan_slots(INDICATOR, rows, fixed_bytes, bytes_per_slot, DEFAULT_HEADROOM)?;
        let scratch_doubles =
            scratch_elems(INDICATOR, "double scratch", plan.slots, doubles_per_slot)?;
        let scratch_ints = scratch_elems(INDICATOR, "int scratch", plan.slots, ints_per_slot)?;

        let func = self
            .module
            .get_function(KERNEL)
            .map_err(|_| CudaIchimokuOscillatorError::MissingKernelSymbol { name: KERNEL })?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_source = DeviceBuffer::from_slice(source)?;
        let d_conv = DeviceBuffer::from_slice(&conversion_periods)?;
        let d_base_p = DeviceBuffer::from_slice(&base_periods)?;
        let d_lag = DeviceBuffer::from_slice(&lagging_span_periods)?;
        let d_disp = DeviceBuffer::from_slice(&displacements)?;
        let d_ma = DeviceBuffer::from_slice(&ma_lengths)?;
        let d_smooth = DeviceBuffer::from_slice(&smoothing_lengths)?;
        let d_window = DeviceBuffer::from_slice(&window_sizes)?;
        let d_top = DeviceBuffer::from_slice(&top_bands)?;
        let d_mid = DeviceBuffer::from_slice(&mid_bands)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_doubles.max(1))? };
        let d_iscratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_ints.max(1))? };

        let mut outs = Vec::with_capacity(13);
        for _ in 0..13 {
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
                d_source.as_device_ptr(),
                cols as i32,
                first as i32,
                d_conv.as_device_ptr(),
                d_base_p.as_device_ptr(),
                d_lag.as_device_ptr(),
                d_disp.as_device_ptr(),
                d_ma.as_device_ptr(),
                d_smooth.as_device_ptr(),
                d_window.as_device_ptr(),
                d_top.as_device_ptr(),
                d_mid.as_device_ptr(),
                i32::from(sweep.extra_smoothing),
                normalize_code(normalize),
                i32::from(sweep.clamp),
                rows as i32,
                plan.slots as i32,
                deque_cap as i32,
                d_scratch.as_device_ptr(),
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
                outs[11].as_device_ptr(),
                outs[12].as_device_ptr()
            ))
            .map_err(|source| CudaIchimokuOscillatorError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaIchimokuOscillatorError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;

        // Drained in the same order they were passed to the kernel. Popping
        // from the back and reversing keeps this total — no indexing, no
        // `expect`, so a length mistake is a compile error rather than a panic
        // in a trading process.
        outs.reverse();
        let mut next = move || -> Result<DeviceBuffer<f64>, CudaIchimokuOscillatorError> {
            outs.pop().ok_or_else(|| {
                CudaIchimokuOscillatorError::InvalidInput(
                    "internal: fewer output buffers than outputs".into(),
                )
            })
        };
        let shape = |buf: DeviceBuffer<f64>| IchimokuOscillatorDeviceArrayF64 { buf, rows, cols };

        let signal = shape(next()?);
        let ma = shape(next()?);
        let conversion = shape(next()?);
        let base = shape(next()?);
        let chikou = shape(next()?);
        let current_kumo_a = shape(next()?);
        let current_kumo_b = shape(next()?);
        let future_kumo_a = shape(next()?);
        let future_kumo_b = shape(next()?);
        let max_level = shape(next()?);
        let high_level = shape(next()?);
        let low_level = shape(next()?);
        let min_level = shape(next()?);

        Ok(CudaIchimokuOscillatorBatchResult {
            outputs: IchimokuOscillatorDeviceOutputs {
                signal,
                ma,
                conversion,
                base,
                chikou,
                current_kumo_a,
                current_kumo_b,
                future_kumo_a,
                future_kumo_b,
                max_level,
                high_level,
                low_level,
                min_level,
            },
            combos,
        })
    }
}
