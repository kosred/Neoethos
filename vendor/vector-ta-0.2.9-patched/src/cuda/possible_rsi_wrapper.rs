#![cfg(feature = "cuda")]

//! `possible_rsi` on the card.
//!
//! # What this used to be
//!
//! The example the f64 dispatch header names by file and line: `batch_dev`
//! resolved `possible_rsi_batch_f64` — a one-line EMPTY kernel — threw the
//! function away, computed all seven output series on the host through
//! `Kernel::ScalarBatch`, and uploaded them with seven
//! `DeviceBuffer::from_slice` calls.
//!
//! # What it is now
//!
//! A real kernel in `kernels/cuda/possible_rsi_kernel.cu`, launched from here.
//! The CPU implementation is unchanged and is still the correct path with no
//! card; it is not reachable from this file because a `CudaPossibleRsi` only
//! exists once a device context has been created. A launch failure is an `Err`
//! naming the indicator.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::possible_rsi::{
    expand_grid_possible_rsi, PossibleRsiBatchRange, PossibleRsiParams,
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

const INDICATOR: &str = "possible_rsi";
const KERNEL: &str = "possible_rsi_batch_f64";

/// Must match `PR_ARRAYS` in the kernel.
const SCRATCH_ARRAYS: usize = 6;

#[derive(Debug, Error)]
pub enum CudaPossibleRsiError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("possible_rsi: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

/// `PossibleRsiMode::from_str` (possible_rsi.rs:44), spelling included — the
/// CPU accepts the misspelling "cuttler", so this must too or a sweep that runs
/// on the CPU would be refused on the card.
fn rsi_mode_code(value: &str) -> Result<i32, CudaPossibleRsiError> {
    let v = value;
    if v.eq_ignore_ascii_case("rsx") {
        return Ok(0);
    }
    if v.eq_ignore_ascii_case("regular") {
        return Ok(1);
    }
    if v.eq_ignore_ascii_case("slow") {
        return Ok(2);
    }
    if v.eq_ignore_ascii_case("rapid") {
        return Ok(3);
    }
    if v.eq_ignore_ascii_case("harris") {
        return Ok(4);
    }
    if v.eq_ignore_ascii_case("cutler") || v.eq_ignore_ascii_case("cuttler") {
        return Ok(5);
    }
    if v.eq_ignore_ascii_case("ehlers_smoothed")
        || v.eq_ignore_ascii_case("ehlers-smoothed")
        || v.eq_ignore_ascii_case("ehlers smoothed")
    {
        return Ok(6);
    }
    Err(CudaPossibleRsiError::InvalidInput(format!(
        "invalid rsi_mode: {value}"
    )))
}

/// `PossibleRsiNormalizationMode::from_str` (possible_rsi.rs:82).
fn normalization_code(value: &str) -> Result<i32, CudaPossibleRsiError> {
    let v = value;
    if v.eq_ignore_ascii_case("gaussian_fisher")
        || v.eq_ignore_ascii_case("gaussian")
        || v.eq_ignore_ascii_case("gaussian (fisher)")
        || v.eq_ignore_ascii_case("fisher")
    {
        return Ok(0);
    }
    if v.eq_ignore_ascii_case("softmax") {
        return Ok(1);
    }
    if v.eq_ignore_ascii_case("regular_norm")
        || v.eq_ignore_ascii_case("regular norm")
        || v.eq_ignore_ascii_case("regnorm")
    {
        return Ok(2);
    }
    Err(CudaPossibleRsiError::InvalidInput(format!(
        "invalid normalization_mode: {value}"
    )))
}

/// `PossibleRsiSignalType::from_str` (possible_rsi.rs:117).
fn signal_type_code(value: &str) -> Result<i32, CudaPossibleRsiError> {
    let v = value;
    if v.eq_ignore_ascii_case("slope") {
        return Ok(0);
    }
    if v.eq_ignore_ascii_case("dynamic_middle_crossover")
        || v.eq_ignore_ascii_case("dynamic middle crossover")
    {
        return Ok(1);
    }
    if v.eq_ignore_ascii_case("levels_crossover") || v.eq_ignore_ascii_case("levels crossover") {
        return Ok(2);
    }
    if v.eq_ignore_ascii_case("zeroline_crossover")
        || v.eq_ignore_ascii_case("zeroline crossover")
        || v.eq_ignore_ascii_case("zero_line_crossover")
    {
        return Ok(3);
    }
    Err(CudaPossibleRsiError::InvalidInput(format!(
        "invalid signal_type: {value}"
    )))
}

pub struct PossibleRsiDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PossibleRsiDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct PossibleRsiDeviceOutputs {
    pub value: PossibleRsiDeviceArrayF64,
    pub buy_level: PossibleRsiDeviceArrayF64,
    pub sell_level: PossibleRsiDeviceArrayF64,
    pub middle_level: PossibleRsiDeviceArrayF64,
    pub state: PossibleRsiDeviceArrayF64,
    pub long_signal: PossibleRsiDeviceArrayF64,
    pub short_signal: PossibleRsiDeviceArrayF64,
}

impl PossibleRsiDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaPossibleRsiBatchResult {
    pub outputs: PossibleRsiDeviceOutputs,
    pub combos: Vec<PossibleRsiParams>,
}

pub struct CudaPossibleRsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPossibleRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaPossibleRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("possible_rsi_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPossibleRsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        range: &PossibleRsiBatchRange,
        base: &PossibleRsiParams,
    ) -> Result<CudaPossibleRsiBatchResult, CudaPossibleRsiError> {
        let cols = data.len();
        if cols == 0 {
            return Err(CudaPossibleRsiError::InvalidInput("empty input".into()));
        }
        if !data.iter().any(|value| value.is_finite()) {
            return Err(CudaPossibleRsiError::InvalidInput("all values are NaN".into()));
        }

        // The three string-valued parameters live on `base` and are NOT swept
        // (`expand_grid_checked`, :1740 — it clones them into every combo), so
        // they resolve once here rather than per row.
        let rsi_mode = rsi_mode_code(base.rsi_mode.as_deref().unwrap_or("regular"))?;
        let normalization_mode =
            normalization_code(base.normalization_mode.as_deref().unwrap_or("gaussian_fisher"))?;
        let signal_type =
            signal_type_code(base.signal_type.as_deref().unwrap_or("zeroline_crossover"))?;
        let run_highpass = base.run_highpass.unwrap_or(false);

        let combos = expand_grid_possible_rsi(range, base);
        if combos.is_empty() {
            return Err(CudaPossibleRsiError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let mut periods = Vec::with_capacity(rows);
        let mut norm_periods = Vec::with_capacity(rows);
        let mut normalization_lengths = Vec::with_capacity(rows);
        let mut nonlag_periods = Vec::with_capacity(rows);
        let mut dynamic_zone_periods = Vec::with_capacity(rows);
        let mut buy_probabilities = Vec::with_capacity(rows);
        let mut sell_probabilities = Vec::with_capacity(rows);
        let mut highpass_periods = Vec::with_capacity(rows);
        let mut weights_cap = 1usize;
        let mut sorted_cap = 1usize;
        let mut deque_cap = 2usize;

        for combo in &combos {
            // resolve_params (:591): every zero is an error on the CPU, so it
            // is an error here too rather than a NaN the kernel invents.
            let period = combo.period.unwrap_or(32);
            let norm_period = combo.norm_period.unwrap_or(100);
            let normalization_length = combo.normalization_length.unwrap_or(15);
            let nonlag_period = combo.nonlag_period.unwrap_or(15);
            let dynamic_zone_period = combo.dynamic_zone_period.unwrap_or(20);
            let highpass_period = combo.highpass_period.unwrap_or(15);
            for (name, value) in [
                ("period", period),
                ("norm_period", norm_period),
                ("normalization_length", normalization_length),
                ("nonlag_period", nonlag_period),
                ("dynamic_zone_period", dynamic_zone_period),
                ("highpass_period", highpass_period),
            ] {
                if value == 0 {
                    return Err(CudaPossibleRsiError::InvalidInput(format!(
                        "invalid {name}: 0"
                    )));
                }
            }
            let buy_probability = combo.buy_probability.unwrap_or(0.2);
            let sell_probability = combo.sell_probability.unwrap_or(0.2);
            for (name, value) in [
                ("buy_probability", buy_probability),
                ("sell_probability", sell_probability),
            ] {
                if !value.is_finite() || !(0.0..=0.5).contains(&value) {
                    return Err(CudaPossibleRsiError::InvalidInput(format!(
                        "invalid {name}: {value}"
                    )));
                }
            }

            // build_nonlag_weights (:1064): `len = period * 4 + (period - 1)`,
            // which is `nonlag_kernel_len` (:576) — 5p - 1.
            let wlen = nonlag_period
                .checked_mul(5)
                .and_then(|value| value.checked_sub(1))
                .ok_or(LaunchPlanError::SizeOverflow {
                    indicator: INDICATOR,
                    what: "nonlag weight length",
                })?;
            weights_cap = weights_cap.max(wlen);
            sorted_cap = sorted_cap.max(dynamic_zone_period);
            deque_cap = deque_cap.max(norm_period.max(normalization_length) + 1);

            periods.push(period as i32);
            norm_periods.push(norm_period as i32);
            normalization_lengths.push(normalization_length as i32);
            nonlag_periods.push(nonlag_period as i32);
            dynamic_zone_periods.push(dynamic_zone_period as i32);
            buy_probabilities.push(buy_probability);
            sell_probabilities.push(sell_probability);
            highpass_periods.push(highpass_period as i32);
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let doubles_per_slot = checked_mul(INDICATOR, "scratch/slot", SCRATCH_ARRAYS, cols)?
            .checked_add(weights_cap)
            .and_then(|value| value.checked_add(sorted_cap))
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "doubles/slot",
            })?;
        let ints_per_slot = checked_mul(INDICATOR, "deques/slot", 2, deque_cap)?;
        let bytes_per_slot = checked_mul(INDICATOR, "double bytes/slot", doubles_per_slot, f64_size)?
            .checked_add(checked_mul(INDICATOR, "int bytes/slot", ints_per_slot, i32_size)?)
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "bytes/slot",
            })?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 7 * f64_size)?
            .checked_add(cols * f64_size)
            .and_then(|b| {
                rows.checked_mul(6 * i32_size + 2 * f64_size)
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
            .map_err(|_| CudaPossibleRsiError::MissingKernelSymbol { name: KERNEL })?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_norm = DeviceBuffer::from_slice(&norm_periods)?;
        let d_norm_len = DeviceBuffer::from_slice(&normalization_lengths)?;
        let d_nonlag = DeviceBuffer::from_slice(&nonlag_periods)?;
        let d_dz = DeviceBuffer::from_slice(&dynamic_zone_periods)?;
        let d_buy = DeviceBuffer::from_slice(&buy_probabilities)?;
        let d_sell = DeviceBuffer::from_slice(&sell_probabilities)?;
        let d_hp = DeviceBuffer::from_slice(&highpass_periods)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_doubles.max(1))? };
        let d_iscratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_ints.max(1))? };

        let mut outs = Vec::with_capacity(7);
        for _ in 0..7 {
            outs.push(unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? });
        }

        validate_launch(self.device_id, plan.grid, plan.block)?;
        let stream = &self.stream;
        let grid = plan.grid;
        let block = plan.block;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                d_norm.as_device_ptr(),
                d_norm_len.as_device_ptr(),
                d_nonlag.as_device_ptr(),
                d_dz.as_device_ptr(),
                d_buy.as_device_ptr(),
                d_sell.as_device_ptr(),
                d_hp.as_device_ptr(),
                rsi_mode,
                normalization_mode,
                signal_type,
                i32::from(run_highpass),
                rows as i32,
                plan.slots as i32,
                weights_cap as i32,
                sorted_cap as i32,
                deque_cap as i32,
                d_scratch.as_device_ptr(),
                d_iscratch.as_device_ptr(),
                outs[0].as_device_ptr(),
                outs[1].as_device_ptr(),
                outs[2].as_device_ptr(),
                outs[3].as_device_ptr(),
                outs[4].as_device_ptr(),
                outs[5].as_device_ptr(),
                outs[6].as_device_ptr()
            ))
            .map_err(|source| CudaPossibleRsiError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaPossibleRsiError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;

        outs.reverse();
        let mut next = move || -> Result<DeviceBuffer<f64>, CudaPossibleRsiError> {
            outs.pop().ok_or_else(|| {
                CudaPossibleRsiError::InvalidInput(
                    "internal: fewer output buffers than outputs".into(),
                )
            })
        };
        let shape = |buf: DeviceBuffer<f64>| PossibleRsiDeviceArrayF64 { buf, rows, cols };

        let value = shape(next()?);
        let buy_level = shape(next()?);
        let sell_level = shape(next()?);
        let middle_level = shape(next()?);
        let state = shape(next()?);
        let long_signal = shape(next()?);
        let short_signal = shape(next()?);

        Ok(CudaPossibleRsiBatchResult {
            outputs: PossibleRsiDeviceOutputs {
                value,
                buy_level,
                sell_level,
                middle_level,
                state,
                long_signal,
                short_signal,
            },
            combos,
        })
    }
}
