//! Exact semantic-v3 SMC producer for the incremental resident feature store.
//!
//! This module borrows the already-admitted run's primary context and stream,
//! uploads canonical parent OHLCV/clock arrays once, and emits the 46 canonical
//! SMC value/validity columns plus the 11 native-evaluator SMC slots. It never
//! creates a context or stream and never materializes a host feature matrix.

use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionV3, ResidentF64FeatureBatchV3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreAssemblerV3, ResidentFeatureStoreCudaErrorV3,
    ResidentParentDatasetSourceV3, ResidentProducerReadyEventV3,
};
use cust::context::{Context, CurrentContext};
use cust::event::{Event, EventFlags, EventStatus};
use cust::memory::{
    AsyncCopyDestination, CopyDestination, DeviceBuffer, DeviceCopy, GpuBuffer, LockedBuffer,
};
use cust::stream::Stream;
use cust::sys::CUstream;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentParentDatasetLayoutV4, ResidentProducerCapabilityV3,
    ResidentWorkingSetBoundV3,
};
use sha2::{Digest, Sha256};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

const SHA256_BYTES: usize = 32;
const GENERATED_PARENT_HASH_BYTES_V3: usize = 3 * SHA256_BYTES;
const SMC_SEMANTIC_VERSION_V3: u32 = 3;
const MIN_CANONICAL_MARKET_TIMESTAMP_MS_V3: i64 = 946_684_800_000;
const MAX_CANONICAL_MARKET_TIMESTAMP_MS_V3: i64 = 32_503_680_000_000;
const SMC_LOGICAL_VALIDITY_SCHEMA_V3: &str =
    "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3";
const SMC_PHYSICAL_VALIDITY_SCHEMA_V3: &str =
    "neoethos.resident-validity.physical-u4.low-nibble-first.v3";

pub const RESIDENT_SMC_COLUMN_NAMES_V3: [&str; 46] = [
    "smc_ob",
    "smc_fvg",
    "smc_ifvg",
    "smc_liq_sweep",
    "smc_pd_array",
    "smc_killzone",
    "smc_displacement",
    "smc_breaker_block",
    "smc_mitigation_block",
    "smc_mss",
    "smc_volume_imbalance",
    "smc_bos",
    "smc_eqh",
    "smc_eql",
    "smc_inducement",
    "smc_asian_range",
    "smc_silver_bullet",
    "smc_judas_swing",
    "smc_nwog",
    "smc_ndog",
    "smc_ict_macro",
    "smc_fvg_strength",
    "smc_dealing_range_width",
    "smc_swing_range_pct",
    "smc_ob_strength",
    "smc_trend_bias",
    "smc_unicorn_model",
    "smc_rejection_block",
    "smc_propulsion_block",
    "smc_fib_time_ratio",
    "smc_fib_236",
    "smc_fib_382",
    "smc_fib_500",
    "smc_fib_618",
    "smc_fib_705",
    "smc_fib_786",
    "smc_fib_886",
    "smc_fib_1272",
    "smc_fib_1414",
    "smc_fib_1618",
    "smc_fib_2000",
    "smc_fib_2618",
    "smc_fvg_magnet_dist",
    "smc_fvg_magnet_age",
    "smc_fvg_inside",
    "smc_fvg_open_count",
];

pub const RESIDENT_SMC_PARENT_SLOT_NAMES_V3: [&str; 11] = [
    "ob",
    "fvg",
    "liquidity",
    "trend",
    "premium",
    "inducement",
    "bos",
    "choch",
    "eqh",
    "eql",
    "displacement",
];

pub const SMC_SLOT_ORDER_V3: [&str; 11] = RESIDENT_SMC_PARENT_SLOT_NAMES_V3;

unsafe extern "C" {
    fn neoethos_resident_smc_parent_features_f64_v3(
        open: *const f64,
        high: *const f64,
        low: *const f64,
        close: *const f64,
        timestamps: *const i64,
        rows: usize,
        smc_feature_values: *mut f64,
        smc_feature_validity_u8: *mut u8,
        months: *mut i64,
        days: *mut i64,
        smc_parent_rows: *mut i8,
        generated_parent_hashes: *mut u8,
        device_error: *mut u32,
        stream: CUstream,
    ) -> i32;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentSmcCapabilityReceiptV3 {
    pub semantic_version: u32,
    pub row_count: usize,
    pub feature_column_count: usize,
    pub parent_smc_slot_count: usize,
    pub retained_parent_device_bytes: usize,
    pub retained_feature_device_bytes: usize,
    pub transient_device_bytes: usize,
    pub peak_device_bytes: usize,
    pub one_time_input_h2d_bytes: usize,
    pub logical_validity_u8_bytes: usize,
    pub sealed_validity_u4_logical_bytes: usize,
    pub sealed_validity_u4_padded_device_bytes: usize,
    pub expected_incremental_u4_pack_launch_count: usize,
    pub producer_ready_event_count: usize,
    pub compact_control_plane_d2h_bytes: usize,
    pub logical_validity_schema: &'static str,
    pub physical_validity_schema: &'static str,
}

/// Owner-derived memory authority for the SMC schema span before a CUDA
/// context, stream, allocation, or producer event exists. The generic resident
/// planner accounts the `46 * rows` value/validity staging bytes; this receipt
/// supplies only SMC's retained parent and launch scratch extents.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentSmcPreDeviceMemoryReceiptV4 {
    row_count: usize,
    feature_column_count: usize,
    additional_retained_bytes: usize,
    scratch_bytes: usize,
}

impl ResidentSmcPreDeviceMemoryReceiptV4 {
    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    pub const fn feature_column_count(&self) -> usize {
        self.feature_column_count
    }

    pub const fn additional_retained_bytes(&self) -> usize {
        self.additional_retained_bytes
    }

    pub const fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
}

pub fn preflight_resident_smc_memory_v4(
    rows: usize,
) -> Result<ResidentSmcPreDeviceMemoryReceiptV4, ResidentFeatureStoreCudaErrorV3> {
    if rows == 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident SMC pre-device memory requires at least one row".into(),
        ));
    }
    let additional_retained_bytes = checked_parent_bytes_v3(rows)?;
    let scratch_bytes = GENERATED_PARENT_HASH_BYTES_V3
        .checked_add(std::mem::size_of::<u32>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident SMC pre-device scratch bytes",
        ))?;
    Ok(ResidentSmcPreDeviceMemoryReceiptV4 {
        row_count: rows,
        feature_column_count: RESIDENT_SMC_COLUMN_NAMES_V3.len(),
        additional_retained_bytes,
        scratch_bytes,
    })
}

pub fn resident_smc_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-smc-parent-features.f64.semantic-v3");
    implementation.update(include_bytes!("resident_smc_v3.rs"));
    implementation.update(include_bytes!("../native/resident_smc_v3.cu"));
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::Smc,
        "neoethos.gpu-cuda.resident-smc-parent-features.f64.semantic-v3",
        implementation_sha256,
        "smc-semantic-v3;smc_log1p_exact_v1;f64-fixed-order;fmad=false;ftz=false;prec-div=true;prec-sqrt=true",
    )
    .map_err(Into::into)
}

#[derive(Debug)]
struct StreamOrderedSmcBufferV3<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedSmcBufferV3<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: the owner retains this exact context and stream, and its
        // destructor uses only stream-ordered `drop_async`.
        let buffer = unsafe { DeviceBuffer::uninitialized_async(len, stream.as_ref())? };
        Ok(Self {
            buffer: Some(buffer),
            context,
            stream,
        })
    }

    fn is_owned_by_stream(&self, stream: &Stream) -> bool {
        !stream.as_inner().is_null() && self.stream.as_inner() == stream.as_inner()
    }
}

impl<T: DeviceCopy> Deref for StreamOrderedSmcBufferV3<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live SMC stream owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedSmcBufferV3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live SMC stream owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedSmcBufferV3<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if CurrentContext::set_current(self.context.as_ref()).is_ok() {
            let _ = buffer.drop_async(self.stream.as_ref());
        } else {
            // Never invoke DeviceBuffer's legacy synchronous destructor when
            // the admitted primary context cannot be restored.
            std::mem::forget(buffer);
        }
    }
}

struct PinnedSmcInputsV3 {
    open: LockedBuffer<f64>,
    high: LockedBuffer<f64>,
    low: LockedBuffer<f64>,
    close: LockedBuffer<f64>,
    volume: LockedBuffer<f64>,
    timestamps: LockedBuffer<i64>,
}

struct SmcDeviceAllocationsV3 {
    open: StreamOrderedSmcBufferV3<f64>,
    high: StreamOrderedSmcBufferV3<f64>,
    low: StreamOrderedSmcBufferV3<f64>,
    close: StreamOrderedSmcBufferV3<f64>,
    volume: StreamOrderedSmcBufferV3<f64>,
    timestamps: StreamOrderedSmcBufferV3<i64>,
    feature_values: StreamOrderedSmcBufferV3<f64>,
    feature_validity_u8: StreamOrderedSmcBufferV3<u8>,
    months: StreamOrderedSmcBufferV3<i64>,
    days: StreamOrderedSmcBufferV3<i64>,
    smc_parent_rows: StreamOrderedSmcBufferV3<i8>,
    generated_parent_hashes: StreamOrderedSmcBufferV3<u8>,
    device_error: StreamOrderedSmcBufferV3<u32>,
}

/// Progressive async-launch guard. Once the first H2D copy is queued, every
/// pinned host pointer and device allocation is leaked on an uncertain error
/// path rather than risking an early ordinary free. Event-ready completion
/// disarms the guard and restores stream-ordered destruction.
struct SmcLaunchTransactionV3 {
    pinned: Option<PinnedSmcInputsV3>,
    device: Option<SmcDeviceAllocationsV3>,
    completion_event: Option<Event>,
    producer_event: Option<ResidentProducerReadyEventV3>,
    in_flight: bool,
}

impl Drop for SmcLaunchTransactionV3 {
    fn drop(&mut self) {
        if self.in_flight {
            if let Some(pinned) = self.pinned.take() {
                std::mem::forget(pinned);
            }
            if let Some(device) = self.device.take() {
                std::mem::forget(device);
            }
            if let Some(event) = self.completion_event.take() {
                std::mem::forget(event);
            }
            if let Some(event) = self.producer_event.take() {
                std::mem::forget(event);
            }
        }
    }
}

#[derive(Debug)]
struct ResidentSmcParentOwnerV3 {
    open: StreamOrderedSmcBufferV3<f64>,
    close: StreamOrderedSmcBufferV3<f64>,
    high: StreamOrderedSmcBufferV3<f64>,
    low: StreamOrderedSmcBufferV3<f64>,
    volume: StreamOrderedSmcBufferV3<f64>,
    timestamps: StreamOrderedSmcBufferV3<i64>,
    months: StreamOrderedSmcBufferV3<i64>,
    days: StreamOrderedSmcBufferV3<i64>,
    smc_rows: StreamOrderedSmcBufferV3<i8>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: Arc<ResidentProducerReadyEventV3>,
    layout: ResidentParentDatasetLayoutV4,
    retained_device_bytes: usize,
}

#[derive(Debug)]
struct ResidentSmcFeatureBatchV3 {
    feature_values: StreamOrderedSmcBufferV3<f64>,
    feature_validity_u8: StreamOrderedSmcBufferV3<u8>,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: Arc<ResidentProducerReadyEventV3>,
    retained_device_bytes: usize,
}

#[derive(Debug)]
pub struct ResidentSmcMaterializationV3 {
    parent: Option<ResidentSmcParentOwnerV3>,
    batch: Option<ResidentSmcFeatureBatchV3>,
    receipt: ResidentSmcCapabilityReceiptV3,
}

/// Opaque one-shot SMC batch continuation. Data can append it to the exact
/// assembler, but no caller can extract producer trait objects or raw buffers.
#[derive(Debug)]
pub struct PendingResidentSmcBatchV3 {
    batch: Option<ResidentSmcFeatureBatchV3>,
}

impl ResidentSmcMaterializationV3 {
    pub fn receipt(&self) -> &ResidentSmcCapabilityReceiptV3 {
        &self.receipt
    }

    /// Consume the opaque SMC owner into HTF's crate-internal direct-parent
    /// capture. Concrete SMC buffer types remain private and no feature or
    /// validity array crosses the host boundary.
    pub(crate) fn into_higher_timeframe_parent_parts_v3(
        mut self,
    ) -> Result<
        (
            Box<dyn ResidentParentDatasetSourceV3>,
            Box<dyn ResidentF64FeatureBatchV3>,
        ),
        ResidentFeatureStoreCudaErrorV3,
    > {
        let parent = match self.parent.take() {
            Some(parent) => parent,
            None => {
                if let Some(batch) = self.batch.take() {
                    std::mem::forget(batch);
                }
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "opaque SMC materialization lost its HTF parent source".into(),
                ));
            }
        };
        let batch = match self.batch.take() {
            Some(batch) => batch,
            None => {
                // The parent owns live CUDA allocations. Do not let an invalid
                // one-shot carrier enter a legacy destructor on this path.
                std::mem::forget(parent);
                return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "opaque SMC materialization lost its HTF feature batch".into(),
                ));
            }
        };
        Ok((Box::new(parent), Box::new(batch)))
    }
}

/// Consume the opaque SMC materialization into the generic resident assembler
/// without exposing the internal parent or batch trait objects across crates.
pub fn begin_resident_smc_store_v3(
    run_device: GpuOnlyRunDeviceAdmissionV3,
    expected_column_bindings: Vec<ResidentFeatureColumnBindingV3>,
    working_set: &ResidentWorkingSetBoundV3,
    mut materialization: ResidentSmcMaterializationV3,
) -> Result<
    (ResidentFeatureStoreAssemblerV3, PendingResidentSmcBatchV3),
    ResidentFeatureStoreCudaErrorV3,
> {
    let parent = materialization.parent.take().ok_or_else(|| {
        ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "opaque SMC materialization lost its parent source".into(),
        )
    })?;
    let batch = materialization.batch.take().ok_or_else(|| {
        ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "opaque SMC materialization lost its feature batch".into(),
        )
    })?;
    let assembler = ResidentFeatureStoreAssemblerV3::new(
        run_device,
        expected_column_bindings,
        Box::new(parent),
        working_set,
    )?;
    Ok((assembler, PendingResidentSmcBatchV3 { batch: Some(batch) }))
}

impl PendingResidentSmcBatchV3 {
    pub fn append_to(
        mut self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        let batch = self.batch.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "opaque SMC batch continuation was already consumed".into(),
            )
        })?;
        assembler.append_batch(Box::new(batch))
    }
}

pub fn prepare_resident_smc_parent_v3(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    bindings: Vec<ResidentFeatureColumnBindingV3>,
) -> Result<ResidentSmcMaterializationV3, ResidentFeatureStoreCudaErrorV3> {
    validate_inputs_v3(open, high, low, close, volume, timestamps, &bindings)?;
    let rows = close.len();
    let open_sha256 = hash_f64_bits_le_v3(open);
    let high_sha256 = hash_f64_bits_le_v3(high);
    let low_sha256 = hash_f64_bits_le_v3(low);
    let close_sha256 = hash_f64_bits_le_v3(close);
    let volume_sha256 = hash_f64_bits_le_v3(volume);
    let timestamps_sha256 = hash_i64_le_v3(timestamps);
    let feature_cells = rows.checked_mul(RESIDENT_SMC_COLUMN_NAMES_V3.len()).ok_or(
        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident SMC feature cells"),
    )?;
    let parent_smc_cells = rows.checked_mul(SMC_SLOT_ORDER_V3.len()).ok_or(
        ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("resident parent SMC cells"),
    )?;
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let device_ordinal = run_device.device_identity().ordinal();
    CurrentContext::set_current(context.as_ref())?;

    let pinned = PinnedSmcInputsV3 {
        open: LockedBuffer::from_slice(open)?,
        high: LockedBuffer::from_slice(high)?,
        low: LockedBuffer::from_slice(low)?,
        close: LockedBuffer::from_slice(close)?,
        volume: LockedBuffer::from_slice(volume)?,
        timestamps: LockedBuffer::from_slice(timestamps)?,
    };
    let device = SmcDeviceAllocationsV3 {
        open: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        high: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        low: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        close: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        volume: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        timestamps: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        feature_values: StreamOrderedSmcBufferV3::uninitialized_async(
            feature_cells,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        feature_validity_u8: StreamOrderedSmcBufferV3::uninitialized_async(
            feature_cells,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        months: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        days: StreamOrderedSmcBufferV3::uninitialized_async(
            rows,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        smc_parent_rows: StreamOrderedSmcBufferV3::uninitialized_async(
            parent_smc_cells,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        generated_parent_hashes: StreamOrderedSmcBufferV3::uninitialized_async(
            GENERATED_PARENT_HASH_BYTES_V3,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
        device_error: StreamOrderedSmcBufferV3::uninitialized_async(
            1,
            Arc::clone(&context),
            Arc::clone(&stream),
        )?,
    };
    let mut transaction = SmcLaunchTransactionV3 {
        pinned: Some(pinned),
        device: Some(device),
        completion_event: None,
        producer_event: None,
        in_flight: true,
    };

    {
        let pinned = transaction.pinned.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident SMC launch lost its pinned inputs".into(),
            )
        })?;
        let device = transaction.device.as_mut().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident SMC launch lost its device allocations".into(),
            )
        })?;
        // SAFETY: every destination has the exact source extent, and the
        // armed transaction retains all pinned sources through event-ready.
        unsafe {
            device.open.async_copy_from(&pinned.open, stream.as_ref())?;
            device.high.async_copy_from(&pinned.high, stream.as_ref())?;
            device.low.async_copy_from(&pinned.low, stream.as_ref())?;
            device
                .close
                .async_copy_from(&pinned.close, stream.as_ref())?;
            device
                .volume
                .async_copy_from(&pinned.volume, stream.as_ref())?;
            device
                .timestamps
                .async_copy_from(&pinned.timestamps, stream.as_ref())?;
        }
        // SAFETY: all pointers reference exact live device extents in the same
        // admitted primary context and remain owned through the ready event.
        let status = unsafe {
            neoethos_resident_smc_parent_features_f64_v3(
                device.open.as_device_ptr().as_ptr(),
                device.high.as_device_ptr().as_ptr(),
                device.low.as_device_ptr().as_ptr(),
                device.close.as_device_ptr().as_ptr(),
                device.timestamps.as_device_ptr().as_ptr(),
                rows,
                device.feature_values.as_device_ptr().as_mut_ptr(),
                device.feature_validity_u8.as_device_ptr().as_mut_ptr(),
                device.months.as_device_ptr().as_mut_ptr(),
                device.days.as_device_ptr().as_mut_ptr(),
                device.smc_parent_rows.as_device_ptr().as_mut_ptr(),
                device.generated_parent_hashes.as_device_ptr().as_mut_ptr(),
                device.device_error.as_device_ptr().as_mut_ptr(),
                stream.as_inner(),
            )
        };
        if status != 0 {
            return Err(ResidentFeatureStoreCudaErrorV3::Native {
                operation: "neoethos_resident_smc_parent_features_f64_v3",
                status,
            });
        }
    }

    let completion_event = Event::new(EventFlags::DISABLE_TIMING)?;
    completion_event.record(stream.as_ref())?;
    transaction.completion_event = Some(completion_event);
    transaction.producer_event = Some(ResidentProducerReadyEventV3::record(
        context.as_ref(),
        stream.as_ref(),
        device_ordinal,
    )?);
    loop {
        match transaction
            .completion_event
            .as_ref()
            .ok_or_else(|| {
                ResidentFeatureStoreCudaErrorV3::InvalidInput(
                    "resident SMC completion event is missing".into(),
                )
            })?
            .query()?
        {
            EventStatus::Ready => break,
            EventStatus::NotReady => std::thread::yield_now(),
        }
    }
    transaction.in_flight = false;

    let mut device_error = [u32::MAX; 1];
    let mut generated_parent_hashes = [0_u8; GENERATED_PARENT_HASH_BYTES_V3];
    {
        let device = transaction.device.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident SMC completed without device allocations".into(),
            )
        })?;
        device.device_error.copy_to(&mut device_error)?;
        device
            .generated_parent_hashes
            .copy_to(&mut generated_parent_hashes)?;
    }
    if device_error[0] != 0 {
        let status = match i32::try_from(device_error[0]) {
            Ok(status) => status,
            Err(_) => i32::MAX,
        };
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "resident SMC device validation",
            status,
        });
    }

    let producer_event = Arc::new(transaction.producer_event.take().ok_or_else(|| {
        ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident SMC completed without a producer event".into(),
        )
    })?);
    drop(transaction.completion_event.take());
    drop(transaction.pinned.take());
    let SmcDeviceAllocationsV3 {
        open: transient_open,
        high,
        low,
        close,
        volume: retained_volume,
        timestamps,
        feature_values,
        feature_validity_u8,
        months,
        days,
        smc_parent_rows,
        generated_parent_hashes: transient_hashes,
        device_error: transient_device_error,
    } = transaction.device.take().ok_or_else(|| {
        ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident SMC completed without owned outputs".into(),
        )
    })?;
    drop(transient_hashes);
    drop(transient_device_error);

    let mut months_sha256 = [0_u8; SHA256_BYTES];
    months_sha256.copy_from_slice(&generated_parent_hashes[0..SHA256_BYTES]);
    let mut days_sha256 = [0_u8; SHA256_BYTES];
    days_sha256.copy_from_slice(&generated_parent_hashes[SHA256_BYTES..2 * SHA256_BYTES]);
    let mut smc_rows_sha256 = [0_u8; SHA256_BYTES];
    smc_rows_sha256.copy_from_slice(&generated_parent_hashes[2 * SHA256_BYTES..]);
    let layout = ResidentParentDatasetLayoutV4::new(
        rows,
        open_sha256,
        high_sha256,
        low_sha256,
        close_sha256,
        volume_sha256,
        timestamps_sha256,
        months_sha256,
        days_sha256,
        smc_rows_sha256,
    )?;
    let retained_parent_device_bytes = checked_parent_bytes_v3(rows)?;
    let retained_feature_device_bytes = feature_cells
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<u8>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident SMC retained feature bytes",
        ))?;
    let one_time_input_h2d_bytes = rows
        .checked_mul(5 * std::mem::size_of::<f64>() + std::mem::size_of::<i64>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident SMC one-time input H2D bytes",
        ))?;
    let transient_device_bytes = GENERATED_PARENT_HASH_BYTES_V3
        .checked_add(std::mem::size_of::<u32>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident SMC transient device bytes",
        ))?;
    let peak_device_bytes = retained_parent_device_bytes
        .checked_add(retained_feature_device_bytes)
        .and_then(|bytes| bytes.checked_add(transient_device_bytes))
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident SMC peak device bytes",
        ))?;
    let sealed_validity_u4_logical_bytes = feature_cells.div_ceil(2);
    let sealed_validity_u4_padded_device_bytes = sealed_validity_u4_logical_bytes
        .checked_add(3)
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident SMC padded u4 validity bytes",
        ))?
        & !3;
    let receipt = ResidentSmcCapabilityReceiptV3 {
        semantic_version: SMC_SEMANTIC_VERSION_V3,
        row_count: rows,
        feature_column_count: RESIDENT_SMC_COLUMN_NAMES_V3.len(),
        parent_smc_slot_count: SMC_SLOT_ORDER_V3.len(),
        retained_parent_device_bytes,
        retained_feature_device_bytes,
        transient_device_bytes,
        peak_device_bytes,
        one_time_input_h2d_bytes,
        logical_validity_u8_bytes: feature_cells,
        sealed_validity_u4_logical_bytes,
        sealed_validity_u4_padded_device_bytes,
        expected_incremental_u4_pack_launch_count: 1,
        producer_ready_event_count: 1,
        compact_control_plane_d2h_bytes: std::mem::size_of::<u32>()
            + GENERATED_PARENT_HASH_BYTES_V3,
        logical_validity_schema: SMC_LOGICAL_VALIDITY_SCHEMA_V3,
        physical_validity_schema: SMC_PHYSICAL_VALIDITY_SCHEMA_V3,
    };
    let parent = ResidentSmcParentOwnerV3 {
        open: transient_open,
        close,
        high,
        low,
        volume: retained_volume,
        timestamps,
        months,
        days,
        smc_rows: smc_parent_rows,
        rows,
        device_ordinal,
        context: Arc::clone(&context),
        stream: Arc::clone(&stream),
        ready_event: Arc::clone(&producer_event),
        layout,
        retained_device_bytes: retained_parent_device_bytes,
    };
    let batch = ResidentSmcFeatureBatchV3 {
        feature_values,
        feature_validity_u8,
        bindings,
        rows,
        device_ordinal,
        context,
        stream,
        ready_event: producer_event,
        retained_device_bytes: retained_feature_device_bytes,
    };
    Ok(ResidentSmcMaterializationV3 {
        parent: Some(parent),
        batch: Some(batch),
        receipt,
    })
}

fn validate_inputs_v3(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    let rows = close.len();
    if rows == 0
        || open.len() != rows
        || high.len() != rows
        || low.len() != rows
        || volume.len() != rows
        || timestamps.len() != rows
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident SMC requires equal nonempty OHLCV/timestamp extents".into(),
        ));
    }
    for row in 0..rows {
        let row_open = open[row];
        let row_high = high[row];
        let row_low = low[row];
        let row_close = close[row];
        let row_volume = volume[row];
        if !row_open.is_finite()
            || !row_high.is_finite()
            || !row_low.is_finite()
            || !row_close.is_finite()
            || !row_volume.is_finite()
            || row_open <= 0.0
            || row_high <= 0.0
            || row_low <= 0.0
            || row_close <= 0.0
            || row_volume < 0.0
            || row_low > row_open.min(row_close)
            || row_high < row_open.max(row_close)
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident SMC OHLCV row {row} is non-canonical"
            )));
        }
        let timestamp = timestamps[row];
        let timestamp_in_range = (MIN_CANONICAL_MARKET_TIMESTAMP_MS_V3
            ..=MAX_CANONICAL_MARKET_TIMESTAMP_MS_V3)
            .contains(&timestamp);
        let timestamp_is_strictly_increasing = row == 0 || timestamp > timestamps[row - 1];
        if !timestamp_in_range || !timestamp_is_strictly_increasing {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident SMC timestamp row {row} is non-canonical"
            )));
        }
    }
    if bindings.len() != RESIDENT_SMC_COLUMN_NAMES_V3.len() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
            "resident SMC expected {} exact bindings, received {}",
            RESIDENT_SMC_COLUMN_NAMES_V3.len(),
            bindings.len()
        )));
    }
    for (index, (binding, expected_name)) in bindings
        .iter()
        .zip(RESIDENT_SMC_COLUMN_NAMES_V3)
        .enumerate()
    {
        if binding.feature_name != expected_name
            || (index > 0 && binding.ordinal != bindings[index - 1].ordinal + 1)
            || binding
                .canonical_parameter_tuple_sha256
                .iter()
                .all(|byte| *byte == 0)
            || binding.route_receipt_sha256.iter().all(|byte| *byte == 0)
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident SMC binding {index} does not match canonical order/identity"
            )));
        }
    }
    Ok(())
}

fn checked_parent_bytes_v3(rows: usize) -> Result<usize, ResidentFeatureStoreCudaErrorV3> {
    rows.checked_mul(
        5 * std::mem::size_of::<f64>()
            + 3 * std::mem::size_of::<i64>()
            + SMC_SLOT_ORDER_V3.len() * std::mem::size_of::<i8>(),
    )
    .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
        "resident SMC parent bytes",
    ))
}

fn hash_f64_bits_le_v3(values: &[f64]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

fn hash_i64_le_v3(values: &[i64]) -> [u8; SHA256_BYTES] {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

unsafe impl ResidentParentDatasetSourceV3 for ResidentSmcParentOwnerV3 {
    fn open(&self) -> &DeviceBuffer<f64> {
        &self.open
    }
    fn close(&self) -> &DeviceBuffer<f64> {
        &self.close
    }
    fn high(&self) -> &DeviceBuffer<f64> {
        &self.high
    }
    fn low(&self) -> &DeviceBuffer<f64> {
        &self.low
    }
    fn volume(&self) -> &DeviceBuffer<f64> {
        &self.volume
    }
    fn timestamps(&self) -> &DeviceBuffer<i64> {
        &self.timestamps
    }
    fn months(&self) -> &DeviceBuffer<i64> {
        &self.months
    }
    fn days(&self) -> &DeviceBuffer<i64> {
        &self.days
    }
    fn smc_rows(&self) -> &DeviceBuffer<i8> {
        &self.smc_rows
    }
    fn rows(&self) -> usize {
        self.rows
    }
    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }
    fn producer_context(&self) -> &Context {
        self.context.as_ref()
    }
    fn producer_stream(&self) -> &Stream {
        self.stream.as_ref()
    }
    fn producer_ready_event(&self) -> &ResidentProducerReadyEventV3 {
        self.ready_event.as_ref()
    }
    fn retained_device_bytes(&self) -> usize {
        self.retained_device_bytes
    }
    fn parent_dataset_layout(&self) -> &ResidentParentDatasetLayoutV4 {
        &self.layout
    }
    fn enqueue_nonblocking_release(
        self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self.open.is_owned_by_stream(release_stream)
            || !self.close.is_owned_by_stream(release_stream)
            || !self.high.is_owned_by_stream(release_stream)
            || !self.low.is_owned_by_stream(release_stream)
            || !self.volume.is_owned_by_stream(release_stream)
            || !self.timestamps.is_owned_by_stream(release_stream)
            || !self.months.is_owned_by_stream(release_stream)
            || !self.days.is_owned_by_stream(release_stream)
            || !self.smc_rows.is_owned_by_stream(release_stream)
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        drop(self);
        Ok(())
    }
}

unsafe impl ResidentF64FeatureBatchV3 for ResidentSmcFeatureBatchV3 {
    fn column_bindings(&self) -> &[ResidentFeatureColumnBindingV3] {
        &self.bindings
    }
    fn value_buffer(&self, _column: usize) -> &DeviceBuffer<f64> {
        &self.feature_values
    }
    fn validity_buffer(&self, _column: usize) -> &DeviceBuffer<u8> {
        &self.feature_validity_u8
    }
    fn value_offset(&self, column: usize) -> usize {
        column * self.rows
    }
    fn validity_offset(&self, column: usize) -> usize {
        column * self.rows
    }
    fn rows(&self) -> usize {
        self.rows
    }
    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }
    fn producer_context(&self) -> &Context {
        self.context.as_ref()
    }
    fn producer_stream(&self) -> &Stream {
        self.stream.as_ref()
    }
    fn producer_ready_event(&self) -> &ResidentProducerReadyEventV3 {
        self.ready_event.as_ref()
    }
    fn retained_device_bytes(&self) -> usize {
        self.retained_device_bytes
    }
    fn retained_scratch_bytes(&self) -> usize {
        0
    }
    fn enqueue_nonblocking_release(
        self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self.feature_values.is_owned_by_stream(release_stream)
            || !self.feature_validity_u8.is_owned_by_stream(release_stream)
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        drop(self);
        Ok(())
    }
}
