#![cfg(feature = "cuda")]

//! `goertzel_cycle_composite_wave` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved the symbol `goertzel_cycle_composite_wave_batch_f64` —
//! a one-line EMPTY kernel — discarded the function, computed the indicator on
//! the host via `Kernel::ScalarBatch`, and uploaded the host answer with
//! `DeviceBuffer::from_slice`. The card did nothing and the caller could not
//! tell.
//!
//! # What it is now
//!
//! A real kernel in `kernels/cuda/goertzel_cycle_composite_wave_kernel.cu`,
//! launched from here. The CPU implementation is untouched and is still the
//! correct answer on a machine with no card; it is unreachable from this file
//! because a `CudaGoertzelCycleCompositeWave` only exists once a device context
//! has been created. A launch failure is an `Err` naming the indicator.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::goertzel_cycle_composite_wave::{
    expand_grid_goertzel_cycle_composite_wave, GoertzelCycleCompositeWaveBatchRange,
    GoertzelCycleCompositeWaveParams, GoertzelDetrendMode,
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

const INDICATOR: &str = "goertzel_cycle_composite_wave";
const KERNEL: &str = "goertzel_cycle_composite_wave_batch_f64";

const DEFAULT_MAX_PERIOD: usize = 120;
const DEFAULT_START_AT_CYCLE: usize = 1;
const DEFAULT_USE_TOP_CYCLES: usize = 2;
const DEFAULT_BAR_TO_CALCULATE: usize = 1;
const DEFAULT_DT_ZL_PER1: usize = 10;
const DEFAULT_DT_ZL_PER2: usize = 40;
const DEFAULT_DT_HP_PER1: usize = 20;
const DEFAULT_DT_HP_PER2: usize = 80;
const DEFAULT_DT_REG_ZL_SMOOTH_PER: usize = 5;
const DEFAULT_HP_SMOOTH_PER: usize = 20;
const DEFAULT_ZLMA_SMOOTH_PER: usize = 10;
const DEFAULT_BART_NO_CYCLES: usize = 5;
const DEFAULT_BART_SMOOTH_PER: usize = 2;
const DEFAULT_BART_SIG_LIMIT: usize = 50;

/// The kernel's `GZ_MODE_*` codes, spelled out rather than cast so a new
/// upstream variant breaks this match instead of silently renumbering.
fn mode_code(mode: GoertzelDetrendMode) -> i32 {
    match mode {
        GoertzelDetrendMode::None => 0,
        GoertzelDetrendMode::HodrickPrescottSmoothing => 1,
        GoertzelDetrendMode::ZeroLagSmoothing => 2,
        GoertzelDetrendMode::HodrickPrescottDetrending => 3,
        GoertzelDetrendMode::ZeroLagDetrending => 4,
        GoertzelDetrendMode::LogZeroLagRegressionDetrending => 5,
    }
}

#[derive(Debug, Error)]
pub enum CudaGoertzelCycleCompositeWaveError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("goertzel_cycle_composite_wave: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct GoertzelCycleCompositeWaveDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl GoertzelCycleCompositeWaveDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct GoertzelCycleCompositeWaveDeviceOutputs {
    pub values: GoertzelCycleCompositeWaveDeviceArrayF64,
}

impl GoertzelCycleCompositeWaveDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.values.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.values.cols
    }
}

pub struct CudaGoertzelCycleCompositeWaveBatchResult {
    pub outputs: GoertzelCycleCompositeWaveDeviceOutputs,
    pub combos: Vec<GoertzelCycleCompositeWaveParams>,
}

pub struct CudaGoertzelCycleCompositeWave {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaGoertzelCycleCompositeWave {
    pub fn new(device_id: usize) -> Result<Self, CudaGoertzelCycleCompositeWaveError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("goertzel_cycle_composite_wave_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaGoertzelCycleCompositeWaveError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &GoertzelCycleCompositeWaveBatchRange,
    ) -> Result<CudaGoertzelCycleCompositeWaveBatchResult, CudaGoertzelCycleCompositeWaveError>
    {
        let cols = data.len();
        if cols == 0 {
            return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(
                "empty input".into(),
            ));
        }

        // `validate_common` (:437) requires a RUN of finite bars at least as
        // long as the widest window, not merely one finite bar.
        let mut longest_run = 0usize;
        let mut run = 0usize;
        for value in data {
            if value.is_finite() {
                run += 1;
                if run > longest_run {
                    longest_run = run;
                }
            } else {
                run = 0;
            }
        }
        if longest_run == 0 {
            return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_goertzel_cycle_composite_wave(sweep);
        if combos.is_empty() {
            return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let base = sweep.base_params;
        let bar_to_calculate = base.bar_to_calculate.unwrap_or(DEFAULT_BAR_TO_CALCULATE);
        let bart_no_cycles = base.bart_no_cycles.unwrap_or(DEFAULT_BART_NO_CYCLES);
        let bart_smooth_per = base.bart_smooth_per.unwrap_or(DEFAULT_BART_SMOOTH_PER);
        let bart_sig_limit = base.bart_sig_limit.unwrap_or(DEFAULT_BART_SIG_LIMIT);
        let mode = base.detrend_mode.unwrap_or_default();

        // `validate_params` (:380): the two classes of period check the CPU
        // makes, reproduced so a bad sweep is refused here rather than folded
        // into a NaN by the kernel.
        let hp_periods = [
            ("dt_hp_per1", base.dt_hp_per1.unwrap_or(DEFAULT_DT_HP_PER1)),
            ("dt_hp_per2", base.dt_hp_per2.unwrap_or(DEFAULT_DT_HP_PER2)),
            (
                "hp_smooth_per",
                base.hp_smooth_per.unwrap_or(DEFAULT_HP_SMOOTH_PER),
            ),
        ];
        for (name, value) in hp_periods {
            if value < 2 {
                return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(format!(
                    "invalid {name}: {value}"
                )));
            }
        }
        let positive_periods = [
            ("dt_zl_per1", base.dt_zl_per1.unwrap_or(DEFAULT_DT_ZL_PER1)),
            ("dt_zl_per2", base.dt_zl_per2.unwrap_or(DEFAULT_DT_ZL_PER2)),
            (
                "dt_reg_zl_smooth_per",
                base.dt_reg_zl_smooth_per
                    .unwrap_or(DEFAULT_DT_REG_ZL_SMOOTH_PER),
            ),
            (
                "zlma_smooth_per",
                base.zlma_smooth_per.unwrap_or(DEFAULT_ZLMA_SMOOTH_PER),
            ),
            ("bart_no_cycles", bart_no_cycles),
            ("bart_smooth_per", bart_smooth_per),
        ];
        for (name, value) in positive_periods {
            if value == 0 {
                return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(format!(
                    "invalid {name}: 0"
                )));
            }
        }

        let mut max_periods = Vec::with_capacity(rows);
        let mut start_at_cycles = Vec::with_capacity(rows);
        let mut top_cycles = Vec::with_capacity(rows);
        let mut sample_cap = 1usize;
        let mut work_cap = 1usize;
        let mut cycle_cap = 1usize;
        let mut max_needed = 0usize;

        for combo in &combos {
            let max_period = combo.max_period.unwrap_or(DEFAULT_MAX_PERIOD);
            if max_period < 2 {
                return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(format!(
                    "invalid max_period: {max_period}"
                )));
            }
            let start_at_cycle = combo.start_at_cycle.unwrap_or(DEFAULT_START_AT_CYCLE);
            let use_top = combo.use_top_cycles.unwrap_or(DEFAULT_USE_TOP_CYCLES);
            if start_at_cycle == 0 || use_top == 0 {
                return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(
                    "start_at_cycle and use_top_cycles must be non-zero".into(),
                ));
            }

            // sample_size_for_params (:334)
            let cycle_span = (2 * max_period).max(bart_no_cycles.saturating_mul(max_period));
            let sample_size = cycle_span.saturating_add(bar_to_calculate);
            max_needed = max_needed.max(sample_size);
            sample_cap = sample_cap.max(sample_size);
            work_cap = work_cap.max(2 * max_period + 1);
            cycle_cap = cycle_cap.max(max_period + 2);

            max_periods.push(max_period as i32);
            start_at_cycles.push(start_at_cycle as i32);
            top_cycles.push(use_top as i32);
        }

        if longest_run < max_needed {
            return Err(CudaGoertzelCycleCompositeWaveError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={longest_run}"
            )));
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let doubles_per_slot = 6 * sample_cap + 4 * work_cap + 3 * cycle_cap;
        let ints_per_slot = cycle_cap;
        let bytes_per_slot = checked_mul(INDICATOR, "double scratch/slot", doubles_per_slot, f64_size)?
            .checked_add(checked_mul(INDICATOR, "int scratch/slot", ints_per_slot, i32_size)?)
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "bytes/slot",
            })?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, f64_size)?
            .checked_add(cols * f64_size)
            .and_then(|b| rows.checked_mul(3 * i32_size).and_then(|c| b.checked_add(c)))
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
            .map_err(|_| CudaGoertzelCycleCompositeWaveError::MissingKernelSymbol { name: KERNEL })?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_max_periods = DeviceBuffer::from_slice(&max_periods)?;
        let d_start_at = DeviceBuffer::from_slice(&start_at_cycles)?;
        let d_top = DeviceBuffer::from_slice(&top_cycles)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_doubles.max(1))? };
        let d_iscratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_ints.max(1))? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        validate_launch(self.device_id, plan.grid, plan.block)?;
        let stream = &self.stream;
        let grid = plan.grid;
        let block = plan.block;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_max_periods.as_device_ptr(),
                d_start_at.as_device_ptr(),
                d_top.as_device_ptr(),
                bar_to_calculate as i32,
                mode_code(mode),
                base.dt_zl_per1.unwrap_or(DEFAULT_DT_ZL_PER1) as i32,
                base.dt_zl_per2.unwrap_or(DEFAULT_DT_ZL_PER2) as i32,
                base.dt_hp_per1.unwrap_or(DEFAULT_DT_HP_PER1) as i32,
                base.dt_hp_per2.unwrap_or(DEFAULT_DT_HP_PER2) as i32,
                base.dt_reg_zl_smooth_per.unwrap_or(DEFAULT_DT_REG_ZL_SMOOTH_PER) as i32,
                base.hp_smooth_per.unwrap_or(DEFAULT_HP_SMOOTH_PER) as i32,
                base.zlma_smooth_per.unwrap_or(DEFAULT_ZLMA_SMOOTH_PER) as i32,
                i32::from(base.filter_bartels.unwrap_or(false)),
                bart_no_cycles as i32,
                bart_smooth_per as i32,
                bart_sig_limit as i32,
                i32::from(base.sort_bartels.unwrap_or(false)),
                i32::from(base.squared_amp.unwrap_or(true)),
                i32::from(base.use_cosine.unwrap_or(true)),
                i32::from(base.subtract_noise.unwrap_or(false)),
                i32::from(base.use_cycle_strength.unwrap_or(true)),
                rows as i32,
                plan.slots as i32,
                sample_cap as i32,
                work_cap as i32,
                cycle_cap as i32,
                d_scratch.as_device_ptr(),
                d_iscratch.as_device_ptr(),
                d_out.as_device_ptr()
            ))
            .map_err(|source| CudaGoertzelCycleCompositeWaveError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream.synchronize().map_err(|source| {
            CudaGoertzelCycleCompositeWaveError::LaunchFailed {
                kernel: KERNEL,
                source,
            }
        })?;

        Ok(CudaGoertzelCycleCompositeWaveBatchResult {
            outputs: GoertzelCycleCompositeWaveDeviceOutputs {
                values: GoertzelCycleCompositeWaveDeviceArrayF64 {
                    buf: d_out,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
