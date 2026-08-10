#![cfg(feature = "cuda")]

//! `kase_peak_oscillator_with_divergences` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved `kase_peak_oscillator_with_divergences_batch_f64` — a
//! one-line EMPTY kernel — discarded the function, computed all eleven output
//! series on the host through `Kernel::ScalarBatch`, and uploaded them.
//!
//! # What it is now
//!
//! A real kernel in
//! `kernels/cuda/kase_peak_oscillator_with_divergences_kernel.cu`, launched
//! from here. The CPU path is untouched and stays correct with no card; it is
//! unreachable from this file because this type only exists once a device
//! context has been created.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::kase_peak_oscillator_with_divergences::{
    expand_grid_kpo, KasePeakOscillatorWithDivergencesBatchRange,
    KasePeakOscillatorWithDivergencesParams,
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

const INDICATOR: &str = "kase_peak_oscillator_with_divergences";
const KERNEL: &str = "kase_peak_oscillator_with_divergences_batch_f64";

/// The six FIXED accumulator windows in the CPU reference (:602-607):
/// 9 + 30 + 3 + 3 + 50 + 50. Must match `KPO_RING_TOTAL` in the kernel.
const RING_TOTAL: usize = 9 + 30 + 3 + 3 + 50 + 50;

const DEFAULT_DEVIATIONS: f64 = 2.0;
const DEFAULT_SHORT_CYCLE: usize = 8;
const DEFAULT_LONG_CYCLE: usize = 65;
const DEFAULT_SENSITIVITY: f64 = 40.0;

#[derive(Debug, Error)]
pub enum CudaKasePeakOscillatorWithDivergencesError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error(
        "kase_peak_oscillator_with_divergences: CUDA kernel `{kernel}` failed to launch: {source}"
    )]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct KasePeakOscillatorWithDivergencesDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl KasePeakOscillatorWithDivergencesDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct KasePeakOscillatorWithDivergencesDeviceOutputs {
    pub oscillator: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub histogram: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub max_peak_value: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub min_peak_value: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub market_extreme: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub regular_bullish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub hidden_bullish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub regular_bearish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub hidden_bearish: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub go_long: KasePeakOscillatorWithDivergencesDeviceArrayF64,
    pub go_short: KasePeakOscillatorWithDivergencesDeviceArrayF64,
}

impl KasePeakOscillatorWithDivergencesDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaKasePeakOscillatorWithDivergencesBatchResult {
    pub outputs: KasePeakOscillatorWithDivergencesDeviceOutputs,
    pub combos: Vec<KasePeakOscillatorWithDivergencesParams>,
}

pub struct CudaKasePeakOscillatorWithDivergences {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaKasePeakOscillatorWithDivergences {
    pub fn new(device_id: usize) -> Result<Self, CudaKasePeakOscillatorWithDivergencesError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("kase_peak_oscillator_with_divergences_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaKasePeakOscillatorWithDivergencesError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &KasePeakOscillatorWithDivergencesBatchRange,
    ) -> Result<
        CudaKasePeakOscillatorWithDivergencesBatchResult,
        CudaKasePeakOscillatorWithDivergencesError,
    > {
        let cols = close.len();
        if cols == 0 || high.is_empty() || low.is_empty() {
            return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != cols || low.len() != cols {
            return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                format!(
                    "data length mismatch: high={} low={} close={}",
                    high.len(),
                    low.len(),
                    cols
                ),
            ));
        }

        let combos = expand_grid_kpo(sweep)
            .map_err(|e| CudaKasePeakOscillatorWithDivergencesError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        // The per-row lookbacks are NOT swept (`expand_grid_kpo`, :1510 copies
        // them from the sweep into every combo), so they resolve once.
        let lb_r = sweep.lb_r;
        let lb_l = sweep.lb_l;
        let range_upper = sweep.range_upper;
        let range_lower = sweep.range_lower;
        for (name, value) in [
            ("lb_r", lb_r),
            ("lb_l", lb_l),
            ("range_upper", range_upper),
            ("range_lower", range_lower),
        ] {
            if value == 0 {
                return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                    format!("invalid {name}: 0"),
                ));
            }
        }
        if range_lower > range_upper {
            return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                format!("invalid divergence range: {range_lower}..{range_upper}"),
            ));
        }

        let mut deviations = Vec::with_capacity(rows);
        let mut short_cycles = Vec::with_capacity(rows);
        let mut long_cycles = Vec::with_capacity(rows);
        let mut sensitivities = Vec::with_capacity(rows);
        let mut long_cycle_cap = 1usize;

        for combo in &combos {
            // resolve_params (:953) with `data_len = cols`.
            let deviation = combo.deviations.unwrap_or(DEFAULT_DEVIATIONS);
            if !deviation.is_finite() || deviation < 0.0 {
                return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                    format!("invalid deviations: {deviation}"),
                ));
            }
            let short_cycle = combo.short_cycle.unwrap_or(DEFAULT_SHORT_CYCLE);
            if short_cycle == 0 || short_cycle >= cols {
                return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                    format!("invalid short_cycle: {short_cycle} (data_len={cols})"),
                ));
            }
            let long_cycle = combo.long_cycle.unwrap_or(DEFAULT_LONG_CYCLE);
            if long_cycle == 0 || long_cycle > cols {
                return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                    format!("invalid long_cycle: {long_cycle} (data_len={cols})"),
                ));
            }
            if short_cycle >= long_cycle {
                return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                    format!("invalid cycle order: {short_cycle} >= {long_cycle}"),
                ));
            }
            let sensitivity = combo.sensitivity.unwrap_or(DEFAULT_SENSITIVITY);
            if !sensitivity.is_finite() {
                return Err(CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                    format!("invalid sensitivity: {sensitivity}"),
                ));
            }

            long_cycle_cap = long_cycle_cap.max(long_cycle);
            deviations.push(deviation);
            short_cycles.push(short_cycle as i32);
            long_cycles.push(long_cycle as i32);
            sensitivities.push(sensitivity);
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        // Three `cols`-long histories, the roots table, and the six rings.
        let doubles_per_slot = checked_mul(INDICATOR, "histories/slot", 3, cols)?
            .checked_add(long_cycle_cap)
            .and_then(|value| value.checked_add(RING_TOTAL))
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "doubles/slot",
            })?;
        let bytes_per_slot = checked_mul(INDICATOR, "bytes/slot", doubles_per_slot, f64_size)?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 11 * f64_size)?
            .checked_add(cols * 3 * f64_size)
            .and_then(|b| {
                rows.checked_mul(2 * i32_size + 2 * f64_size)
                    .and_then(|c| b.checked_add(c))
            })
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "fixed bytes",
            })?;

        let plan = plan_slots(INDICATOR, rows, fixed_bytes, bytes_per_slot, DEFAULT_HEADROOM)?;
        let scratch_len = scratch_elems(INDICATOR, "scratch", plan.slots, doubles_per_slot)?;

        let func = self.module.get_function(KERNEL).map_err(|_| {
            CudaKasePeakOscillatorWithDivergencesError::MissingKernelSymbol { name: KERNEL }
        })?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_dev = DeviceBuffer::from_slice(&deviations)?;
        let d_short = DeviceBuffer::from_slice(&short_cycles)?;
        let d_long = DeviceBuffer::from_slice(&long_cycles)?;
        let d_sens = DeviceBuffer::from_slice(&sensitivities)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len.max(1))? };

        let mut outs = Vec::with_capacity(11);
        for _ in 0..11 {
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
                d_dev.as_device_ptr(),
                d_short.as_device_ptr(),
                d_long.as_device_ptr(),
                d_sens.as_device_ptr(),
                i32::from(sweep.all_peaks_mode),
                lb_r as i32,
                lb_l as i32,
                range_upper as i32,
                range_lower as i32,
                i32::from(sweep.plot_bull),
                i32::from(sweep.plot_hidden_bull),
                i32::from(sweep.plot_bear),
                i32::from(sweep.plot_hidden_bear),
                rows as i32,
                plan.slots as i32,
                long_cycle_cap as i32,
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
                outs[10].as_device_ptr()
            ))
            .map_err(
                |source| CudaKasePeakOscillatorWithDivergencesError::LaunchFailed {
                    kernel: KERNEL,
                    source,
                },
            )?;
        }

        self.stream.synchronize().map_err(|source| {
            CudaKasePeakOscillatorWithDivergencesError::LaunchFailed {
                kernel: KERNEL,
                source,
            }
        })?;

        outs.reverse();
        let mut next =
            move || -> Result<DeviceBuffer<f64>, CudaKasePeakOscillatorWithDivergencesError> {
                outs.pop().ok_or_else(|| {
                    CudaKasePeakOscillatorWithDivergencesError::InvalidInput(
                        "internal: fewer output buffers than outputs".into(),
                    )
                })
            };
        let shape = |buf: DeviceBuffer<f64>| KasePeakOscillatorWithDivergencesDeviceArrayF64 {
            buf,
            rows,
            cols,
        };

        let oscillator = shape(next()?);
        let histogram = shape(next()?);
        let max_peak_value = shape(next()?);
        let min_peak_value = shape(next()?);
        let market_extreme = shape(next()?);
        let regular_bullish = shape(next()?);
        let hidden_bullish = shape(next()?);
        let regular_bearish = shape(next()?);
        let hidden_bearish = shape(next()?);
        let go_long = shape(next()?);
        let go_short = shape(next()?);

        Ok(CudaKasePeakOscillatorWithDivergencesBatchResult {
            outputs: KasePeakOscillatorWithDivergencesDeviceOutputs {
                oscillator,
                histogram,
                max_peak_value,
                min_peak_value,
                market_extreme,
                regular_bullish,
                hidden_bullish,
                regular_bearish,
                hidden_bearish,
                go_long,
                go_short,
            },
            combos,
        })
    }
}
