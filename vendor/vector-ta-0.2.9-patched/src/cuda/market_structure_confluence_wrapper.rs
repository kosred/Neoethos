#![cfg(feature = "cuda")]

//! `market_structure_confluence` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` resolved `market_structure_confluence_batch_f64` — a one-line
//! EMPTY kernel — discarded the function, computed all sixteen output series on
//! the host through `Kernel::ScalarBatch`, and uploaded them.
//!
//! # What it is now
//!
//! A real kernel in `kernels/cuda/market_structure_confluence_kernel.cu`,
//! launched from here. The CPU path is untouched and stays correct with no
//! card; it is unreachable from this file because this type only exists once a
//! device context has been created.

use crate::cuda::f64_launch::{
    checked_mul, plan_slots, scratch_elems, validate_launch, LaunchPlanError, DEFAULT_HEADROOM,
};
use crate::indicators::market_structure_confluence::{
    expand_grid, MarketStructureConfluenceBatchRange, MarketStructureConfluenceParams,
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

const INDICATOR: &str = "market_structure_confluence";
const KERNEL: &str = "market_structure_confluence_batch_f64";

const DEFAULT_SWING_SIZE: usize = 10;
const DEFAULT_BOS_CONFIRMATION: &str = "Candle Close";
const DEFAULT_BASIS_LENGTH: usize = 100;
const DEFAULT_ATR_LENGTH: usize = 14;
const DEFAULT_ATR_SMOOTH: usize = 21;
const DEFAULT_VOL_MULT: f64 = 2.0;

/// `MarketStructureConfluenceBosConfirmation::parse` (:46). Note the CPU match
/// is CASE-SENSITIVE on these exact spellings, so this is too — accepting more
/// here would let a sweep run on the card that the CPU refuses.
fn bos_code(value: &str) -> Result<i32, CudaMarketStructureConfluenceError> {
    match value {
        "Candle Close" | "candle_close" | "candle close" => Ok(0),
        "Wicks" | "wicks" => Ok(1),
        other => Err(CudaMarketStructureConfluenceError::InvalidInput(format!(
            "invalid bos_confirmation: {other}"
        ))),
    }
}

#[derive(Debug, Error)]
pub enum CudaMarketStructureConfluenceError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("market_structure_confluence: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct MarketStructureConfluenceDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MarketStructureConfluenceDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MarketStructureConfluenceDeviceOutputs {
    pub basis: MarketStructureConfluenceDeviceArrayF64,
    pub upper_band: MarketStructureConfluenceDeviceArrayF64,
    pub lower_band: MarketStructureConfluenceDeviceArrayF64,
    pub structure_direction: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_arrow: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_arrow: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_change: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_change: MarketStructureConfluenceDeviceArrayF64,
    pub hh: MarketStructureConfluenceDeviceArrayF64,
    pub lh: MarketStructureConfluenceDeviceArrayF64,
    pub hl: MarketStructureConfluenceDeviceArrayF64,
    pub ll: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_bos: MarketStructureConfluenceDeviceArrayF64,
    pub bullish_choch: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_bos: MarketStructureConfluenceDeviceArrayF64,
    pub bearish_choch: MarketStructureConfluenceDeviceArrayF64,
}

impl MarketStructureConfluenceDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.basis.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.basis.cols
    }
}

pub struct CudaMarketStructureConfluenceBatchResult {
    pub outputs: MarketStructureConfluenceDeviceOutputs,
    pub combos: Vec<MarketStructureConfluenceParams>,
}

pub struct CudaMarketStructureConfluence {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMarketStructureConfluence {
    pub fn new(device_id: usize) -> Result<Self, CudaMarketStructureConfluenceError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("market_structure_confluence_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMarketStructureConfluenceError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        range: &MarketStructureConfluenceBatchRange,
    ) -> Result<CudaMarketStructureConfluenceBatchResult, CudaMarketStructureConfluenceError> {
        let cols = close.len();
        if cols == 0 || high.is_empty() || low.is_empty() {
            return Err(CudaMarketStructureConfluenceError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != cols || low.len() != cols {
            return Err(CudaMarketStructureConfluenceError::InvalidInput(format!(
                "data length mismatch: high={} low={} close={}",
                high.len(),
                low.len(),
                cols
            )));
        }

        let combos = expand_grid(range)
            .map_err(|e| CudaMarketStructureConfluenceError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaMarketStructureConfluenceError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        let mut swing_sizes = Vec::with_capacity(rows);
        let mut bos_confirmations = Vec::with_capacity(rows);
        let mut basis_lengths = Vec::with_capacity(rows);
        let mut atr_lengths = Vec::with_capacity(rows);
        let mut atr_smooths = Vec::with_capacity(rows);
        let mut vol_mults = Vec::with_capacity(rows);
        let mut basis_cap = 1usize;
        let mut smooth_cap = 1usize;
        let mut pivot_cap = 3usize;

        for combo in &combos {
            // resolve_params (:1038) with `data_len = cols`.
            let swing_size = combo.swing_size.unwrap_or(DEFAULT_SWING_SIZE);
            let bos = bos_code(
                combo
                    .bos_confirmation
                    .as_deref()
                    .unwrap_or(DEFAULT_BOS_CONFIRMATION),
            )?;
            let basis_length = combo.basis_length.unwrap_or(DEFAULT_BASIS_LENGTH);
            let atr_length = combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
            let atr_smooth = combo.atr_smooth.unwrap_or(DEFAULT_ATR_SMOOTH);
            let vol_mult = combo.vol_mult.unwrap_or(DEFAULT_VOL_MULT);

            if swing_size < 2 || swing_size * 2 + 1 > cols {
                return Err(CudaMarketStructureConfluenceError::InvalidInput(format!(
                    "invalid swing_size: {swing_size} (data_len={cols})"
                )));
            }
            for (name, value) in [
                ("basis_length", basis_length),
                ("atr_length", atr_length),
                ("atr_smooth", atr_smooth),
            ] {
                if value == 0 || value > cols {
                    return Err(CudaMarketStructureConfluenceError::InvalidInput(format!(
                        "invalid {name}: {value} (data_len={cols})"
                    )));
                }
            }
            if !vol_mult.is_finite() || vol_mult < 0.0 {
                return Err(CudaMarketStructureConfluenceError::InvalidInput(format!(
                    "invalid vol_mult: {vol_mult}"
                )));
            }

            basis_cap = basis_cap.max(basis_length);
            smooth_cap = smooth_cap.max(atr_smooth);
            // The detector holds `2 * swing_size + 1` entries at its peak; one
            // spare slot keeps push-then-pop from wrapping onto itself.
            pivot_cap = pivot_cap.max(2 * swing_size + 2);

            swing_sizes.push(swing_size as i32);
            bos_confirmations.push(bos);
            basis_lengths.push(basis_length as i32);
            atr_lengths.push(atr_length as i32);
            atr_smooths.push(atr_smooth as i32);
            vol_mults.push(vol_mult);
        }

        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();
        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;

        let doubles_per_slot = basis_cap
            .checked_add(smooth_cap)
            .and_then(|value| value.checked_add(2 * pivot_cap))
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "doubles/slot",
            })?;
        let ints_per_slot = checked_mul(INDICATOR, "pivot indices/slot", 2, pivot_cap)?;
        let bytes_per_slot = checked_mul(INDICATOR, "double bytes/slot", doubles_per_slot, f64_size)?
            .checked_add(checked_mul(INDICATOR, "int bytes/slot", ints_per_slot, i32_size)?)
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "bytes/slot",
            })?;
        let fixed_bytes = checked_mul(INDICATOR, "output bytes", output_elems, 16 * f64_size)?
            .checked_add(cols * 3 * f64_size)
            .and_then(|b| {
                rows.checked_mul(5 * i32_size + f64_size)
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
            .map_err(|_| CudaMarketStructureConfluenceError::MissingKernelSymbol { name: KERNEL })?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_swing = DeviceBuffer::from_slice(&swing_sizes)?;
        let d_bos = DeviceBuffer::from_slice(&bos_confirmations)?;
        let d_basis = DeviceBuffer::from_slice(&basis_lengths)?;
        let d_atr = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_smooth = DeviceBuffer::from_slice(&atr_smooths)?;
        let d_vol = DeviceBuffer::from_slice(&vol_mults)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_doubles.max(1))? };
        let d_iscratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_ints.max(1))? };

        let mut outs = Vec::with_capacity(16);
        for _ in 0..16 {
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
                d_swing.as_device_ptr(),
                d_bos.as_device_ptr(),
                d_basis.as_device_ptr(),
                d_atr.as_device_ptr(),
                d_smooth.as_device_ptr(),
                d_vol.as_device_ptr(),
                rows as i32,
                plan.slots as i32,
                basis_cap as i32,
                smooth_cap as i32,
                pivot_cap as i32,
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
                outs[12].as_device_ptr(),
                outs[13].as_device_ptr(),
                outs[14].as_device_ptr(),
                outs[15].as_device_ptr()
            ))
            .map_err(|source| CudaMarketStructureConfluenceError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaMarketStructureConfluenceError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;

        outs.reverse();
        let mut next = move || -> Result<DeviceBuffer<f64>, CudaMarketStructureConfluenceError> {
            outs.pop().ok_or_else(|| {
                CudaMarketStructureConfluenceError::InvalidInput(
                    "internal: fewer output buffers than outputs".into(),
                )
            })
        };
        let shape =
            |buf: DeviceBuffer<f64>| MarketStructureConfluenceDeviceArrayF64 { buf, rows, cols };

        let basis = shape(next()?);
        let upper_band = shape(next()?);
        let lower_band = shape(next()?);
        let structure_direction = shape(next()?);
        let bullish_arrow = shape(next()?);
        let bearish_arrow = shape(next()?);
        let bullish_change = shape(next()?);
        let bearish_change = shape(next()?);
        let hh = shape(next()?);
        let lh = shape(next()?);
        let hl = shape(next()?);
        let ll = shape(next()?);
        let bullish_bos = shape(next()?);
        let bullish_choch = shape(next()?);
        let bearish_bos = shape(next()?);
        let bearish_choch = shape(next()?);

        Ok(CudaMarketStructureConfluenceBatchResult {
            outputs: MarketStructureConfluenceDeviceOutputs {
                basis,
                upper_band,
                lower_band,
                structure_direction,
                bullish_arrow,
                bearish_arrow,
                bullish_change,
                bearish_change,
                hh,
                lh,
                hl,
                ll,
                bullish_bos,
                bullish_choch,
                bearish_bos,
                bearish_choch,
            },
            combos,
        })
    }
}
