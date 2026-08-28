#![cfg(feature = "cuda-build-native")]

//! `insync_index` on the card.
//!
//! # What this used to be
//!
//! `batch_dev` called `get_function("insync_index_batch_f64")` on a one-line
//! EMPTY kernel, discarded the function it got back, computed the indicator on
//! the host through `insync_index_batch_with_kernel(.., Kernel::ScalarBatch)`,
//! and uploaded the host answer with `DeviceBuffer::from_slice` — so the caller
//! received a device pointer and had no way to tell the card had run nothing.
//!
//! # What it is now
//!
//! `kernels/cuda/insync_index_kernel.cu` carries a real `insync_index_batch_f64`
//! written against the CPU reference, and this wrapper launches it. There is no
//! host-compute branch left in this file.
//!
//! # Where the CPU path went
//!
//! Nowhere — it is still `insync_index_batch_with_kernel`, and it is still the
//! CORRECT path on a machine with no card. It is simply not reachable from
//! here: a `CudaInsyncIndex` only exists after `Context::new` succeeded on a
//! real device, so by the time `batch_dev` runs, a card is present and the
//! kernel MUST be what produces the numbers. A failed launch is an `Err` naming
//! the indicator, never a quiet recomputation on the host.

use crate::cuda::f64_launch::{
    DEFAULT_HEADROOM, LaunchPlanError, checked_mul, plan_slots, scratch_elems, validate_launch,
};
use crate::indicators::insync_index::{
    InsyncIndexBatchRange, InsyncIndexParams, expand_grid_insync_index,
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

const INDICATOR: &str = "insync_index";
const KERNEL: &str = "insync_index_batch_f64";

/// Ring-buffer segments the kernel lays out per slot. Must match the
/// `SEG_*` / `ISEG_*` block in `insync_index_kernel.cu`.
const DOUBLE_SEGMENTS: usize = 16;
const INT_SEGMENTS: usize = 4;
/// `DpoState::delayed_components` capacity in the kernel (`DPO_DELAY + 2`).
const DPO_DELAY_CAP: usize = 12;

const DEFAULT_EMO_DIVISOR: usize = 10_000;
const DEFAULT_EMO_LENGTH: usize = 14;
const DEFAULT_FAST_LENGTH: usize = 12;
const DEFAULT_SLOW_LENGTH: usize = 26;
const DEFAULT_MFI_LENGTH: usize = 20;
const DEFAULT_BB_LENGTH: usize = 20;
const DEFAULT_BB_MULTIPLIER: f64 = 2.0;
const DEFAULT_CCI_LENGTH: usize = 14;
const DEFAULT_DPO_LENGTH: usize = 18;
const DEFAULT_ROC_LENGTH: usize = 10;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_STOCH_LENGTH: usize = 14;
const DEFAULT_STOCH_D_LENGTH: usize = 3;
const DEFAULT_STOCH_K_LENGTH: usize = 1;
const DEFAULT_SMA_LENGTH: usize = 10;

#[derive(Debug, Error)]
pub enum CudaInsyncIndexError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    /// A device is present and a kernel exists, so the kernel is what must
    /// produce the numbers. This is the error that used to be a silent host
    /// computation.
    #[error("insync_index: CUDA kernel `{kernel}` failed to launch: {source}")]
    LaunchFailed {
        kernel: &'static str,
        #[source]
        source: cust::error::CudaError,
    },
    #[error(transparent)]
    Plan(#[from] LaunchPlanError),
}

pub struct InsyncIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl InsyncIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct InsyncIndexDeviceOutputs {
    pub values: InsyncIndexDeviceArrayF64,
}

impl InsyncIndexDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.values.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.values.cols
    }
}

pub struct CudaInsyncIndexBatchResult {
    pub outputs: InsyncIndexDeviceOutputs,
    pub combos: Vec<InsyncIndexParams>,
}

pub struct CudaInsyncIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaInsyncIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaInsyncIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("insync_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaInsyncIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        sweep: &InsyncIndexBatchRange,
    ) -> Result<CudaInsyncIndexBatchResult, CudaInsyncIndexError> {
        let cols = close.len();
        if cols == 0 {
            return Err(CudaInsyncIndexError::InvalidInput("empty input".into()));
        }
        if high.len() != cols || low.len() != cols || volume.len() != cols {
            return Err(CudaInsyncIndexError::InvalidInput(
                "high/low/close/volume length mismatch".into(),
            ));
        }

        let combos = expand_grid_insync_index(sweep)
            .map_err(|e| CudaInsyncIndexError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaInsyncIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }
        let rows = combos.len();

        // Per-row parameter vectors, in the same order the kernel reads them.
        let mut emo_divisor = Vec::with_capacity(rows);
        let mut emo_length = Vec::with_capacity(rows);
        let mut fast_length = Vec::with_capacity(rows);
        let mut slow_length = Vec::with_capacity(rows);
        let mut mfi_length = Vec::with_capacity(rows);
        let mut bb_length = Vec::with_capacity(rows);
        let mut bb_multiplier = Vec::with_capacity(rows);
        let mut cci_length = Vec::with_capacity(rows);
        let mut dpo_length = Vec::with_capacity(rows);
        let mut roc_length = Vec::with_capacity(rows);
        let mut rsi_length = Vec::with_capacity(rows);
        let mut stoch_length = Vec::with_capacity(rows);
        let mut stoch_d_length = Vec::with_capacity(rows);
        let mut stoch_k_length = Vec::with_capacity(rows);
        let mut sma_length = Vec::with_capacity(rows);

        // `seg` is the widest ring any row asks for. Every ring is laid out at
        // this stride so the kernel's offsets are row-independent.
        let mut seg = DPO_DELAY_CAP;

        for combo in &combos {
            let get = |value: Option<usize>,
                       default: usize,
                       name: &str|
             -> Result<usize, CudaInsyncIndexError> {
                let value = value.unwrap_or(default);
                if value == 0 {
                    return Err(CudaInsyncIndexError::InvalidInput(format!(
                        "invalid {name}: 0"
                    )));
                }
                Ok(value)
            };

            let emo_div = get(combo.emo_divisor, DEFAULT_EMO_DIVISOR, "emo_divisor")?;
            let emo_len = get(combo.emo_length, DEFAULT_EMO_LENGTH, "emo_length")?;
            let fast_len = get(combo.fast_length, DEFAULT_FAST_LENGTH, "fast_length")?;
            let slow_len = get(combo.slow_length, DEFAULT_SLOW_LENGTH, "slow_length")?;
            let mfi_len = get(combo.mfi_length, DEFAULT_MFI_LENGTH, "mfi_length")?;
            let bb_len = get(combo.bb_length, DEFAULT_BB_LENGTH, "bb_length")?;
            let cci_len = get(combo.cci_length, DEFAULT_CCI_LENGTH, "cci_length")?;
            let dpo_len = get(combo.dpo_length, DEFAULT_DPO_LENGTH, "dpo_length")?;
            let roc_len = get(combo.roc_length, DEFAULT_ROC_LENGTH, "roc_length")?;
            let rsi_len = get(combo.rsi_length, DEFAULT_RSI_LENGTH, "rsi_length")?;
            let stoch_len = get(combo.stoch_length, DEFAULT_STOCH_LENGTH, "stoch_length")?;
            let stoch_d = get(
                combo.stoch_d_length,
                DEFAULT_STOCH_D_LENGTH,
                "stoch_d_length",
            )?;
            let stoch_k = get(
                combo.stoch_k_length,
                DEFAULT_STOCH_K_LENGTH,
                "stoch_k_length",
            )?;
            let sma_len = get(combo.sma_length, DEFAULT_SMA_LENGTH, "sma_length")?;

            let bb_mult = combo.bb_multiplier.unwrap_or(DEFAULT_BB_MULTIPLIER);
            if !bb_mult.is_finite() || bb_mult <= 0.0 {
                return Err(CudaInsyncIndexError::InvalidInput(format!(
                    "invalid bb_multiplier: {bb_mult}"
                )));
            }

            // `DpoState::sma_history` holds `barsback + 1` entries at its peak;
            // the kernel's ring is `barsback + 2`.
            let hist_cap = dpo_len / 2 + 3;
            // The stoch monotonic deques hold `stoch_len + 1` between push and
            // expiry.
            let stoch_cap = stoch_len + 1;
            for width in [
                emo_len, mfi_len, bb_len, cci_len, dpo_len, roc_len, sma_len, stoch_d, stoch_k,
                stoch_cap, hist_cap,
            ] {
                if width > seg {
                    seg = width;
                }
            }

            emo_divisor.push(emo_div as i32);
            emo_length.push(emo_len as i32);
            fast_length.push(fast_len as i32);
            slow_length.push(slow_len as i32);
            mfi_length.push(mfi_len as i32);
            bb_length.push(bb_len as i32);
            bb_multiplier.push(bb_mult);
            cci_length.push(cci_len as i32);
            dpo_length.push(dpo_len as i32);
            roc_length.push(roc_len as i32);
            rsi_length.push(rsi_len as i32);
            stoch_length.push(stoch_len as i32);
            stoch_d_length.push(stoch_d as i32);
            stoch_k_length.push(stoch_k as i32);
            sma_length.push(sma_len as i32);
        }

        let output_elems = checked_mul(INDICATOR, "rows*cols", rows, cols)?;
        let f64_size = std::mem::size_of::<f64>();
        let i32_size = std::mem::size_of::<i32>();

        let doubles_per_slot = checked_mul(INDICATOR, "double scratch/slot", DOUBLE_SEGMENTS, seg)?;
        let ints_per_slot = checked_mul(INDICATOR, "int scratch/slot", INT_SEGMENTS, seg)?;
        let bytes_per_slot = doubles_per_slot
            .checked_mul(f64_size)
            .and_then(|b| {
                ints_per_slot
                    .checked_mul(i32_size)
                    .and_then(|c| b.checked_add(c))
            })
            .ok_or(LaunchPlanError::SizeOverflow {
                indicator: INDICATOR,
                what: "bytes/slot",
            })?;

        let fixed_bytes = output_elems
            .checked_mul(f64_size)
            .and_then(|b| {
                cols.checked_mul(4 * f64_size)
                    .and_then(|c| b.checked_add(c))
            })
            .and_then(|b| {
                rows.checked_mul(14 * i32_size + f64_size)
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
        let scratch_doubles =
            scratch_elems(INDICATOR, "double scratch", plan.slots, doubles_per_slot)?;
        let scratch_ints = scratch_elems(INDICATOR, "int scratch", plan.slots, ints_per_slot)?;

        let func = self
            .module
            .get_function(KERNEL)
            .map_err(|_| CudaInsyncIndexError::MissingKernelSymbol { name: KERNEL })?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_emo_divisor = DeviceBuffer::from_slice(&emo_divisor)?;
        let d_emo_length = DeviceBuffer::from_slice(&emo_length)?;
        let d_fast_length = DeviceBuffer::from_slice(&fast_length)?;
        let d_slow_length = DeviceBuffer::from_slice(&slow_length)?;
        let d_mfi_length = DeviceBuffer::from_slice(&mfi_length)?;
        let d_bb_length = DeviceBuffer::from_slice(&bb_length)?;
        let d_bb_multiplier = DeviceBuffer::from_slice(&bb_multiplier)?;
        let d_cci_length = DeviceBuffer::from_slice(&cci_length)?;
        let d_dpo_length = DeviceBuffer::from_slice(&dpo_length)?;
        let d_roc_length = DeviceBuffer::from_slice(&roc_length)?;
        let d_rsi_length = DeviceBuffer::from_slice(&rsi_length)?;
        let d_stoch_length = DeviceBuffer::from_slice(&stoch_length)?;
        let d_stoch_d_length = DeviceBuffer::from_slice(&stoch_d_length)?;
        let d_stoch_k_length = DeviceBuffer::from_slice(&stoch_k_length)?;
        let d_sma_length = DeviceBuffer::from_slice(&sma_length)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_doubles.max(1))? };
        let d_iscratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_ints.max(1))? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        validate_launch(self.device_id, plan.grid, plan.block)?;
        let stream = &self.stream;
        let grid = plan.grid;
        let block = plan.block;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_emo_divisor.as_device_ptr(),
                d_emo_length.as_device_ptr(),
                d_fast_length.as_device_ptr(),
                d_slow_length.as_device_ptr(),
                d_mfi_length.as_device_ptr(),
                d_bb_length.as_device_ptr(),
                d_bb_multiplier.as_device_ptr(),
                d_cci_length.as_device_ptr(),
                d_dpo_length.as_device_ptr(),
                d_roc_length.as_device_ptr(),
                d_rsi_length.as_device_ptr(),
                d_stoch_length.as_device_ptr(),
                d_stoch_d_length.as_device_ptr(),
                d_stoch_k_length.as_device_ptr(),
                d_sma_length.as_device_ptr(),
                rows as i32,
                plan.slots as i32,
                seg as i32,
                d_scratch.as_device_ptr(),
                d_iscratch.as_device_ptr(),
                d_out.as_device_ptr()
            ))
            .map_err(|source| CudaInsyncIndexError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|source| CudaInsyncIndexError::LaunchFailed {
                kernel: KERNEL,
                source,
            })?;

        Ok(CudaInsyncIndexBatchResult {
            outputs: InsyncIndexDeviceOutputs {
                values: InsyncIndexDeviceArrayF64 {
                    buf: d_out,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
