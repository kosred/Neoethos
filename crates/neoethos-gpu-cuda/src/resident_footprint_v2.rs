//! Resident CUDA implementation of the complete Footprint semantic-v2 family.
//!
//! The producer reads the retained one-upload OHLCV/timestamp parent on the
//! admitted primary context and non-default run stream. It emits all seven
//! feature-major f64/u8 columns, retains prefix scratch through the downstream
//! pack-ready event, and never downloads feature values or validity bytes.

use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionV3, ResidentF64FeatureBatchV3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreCudaErrorV3, ResidentParentDatasetSourceV3, ResidentProducerReadyEventV3,
};
use cust::context::{Context, CurrentContext};
use cust::memory::{DeviceBuffer, DeviceCopy, GpuBuffer};
use cust::stream::Stream;
use cust::sys::CUstream;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityV3,
};
use sha2::{Digest, Sha256};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

const SHA256_BYTES: usize = 32;
pub const FOOTPRINT_SEMANTIC_VERSION_V2: u32 = 2;
const FOOTPRINT_PREFIX_SERIES_V2: usize = 8;
const FOOTPRINT_LOGICAL_VALIDITY_SCHEMA_V2: &str =
    "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3";
const FOOTPRINT_EXACT_MATH_AUTHORITY_V2: &str = "footprint-cpu-oracle-semantic-v2;f64-fixed-prefix-order;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";

pub const RESIDENT_FOOTPRINT_COLUMN_NAMES_V2: [&str; 7] = [
    "fp_volume_z",
    "fp_absorption",
    "fp_effort_result_div",
    "fp_climax",
    "fp_delta_proxy",
    "fp_volprice_corr",
    "fp_fix_window",
];

unsafe extern "C" {
    fn neoethos_resident_footprint_f64_v2(
        open: *const f64,
        high: *const f64,
        low: *const f64,
        close: *const f64,
        volume: *const f64,
        timestamps_ms: *const i64,
        rows: usize,
        feature_values: *mut f64,
        feature_validity_u8: *mut u8,
        prefix_scratch: *mut f64,
        stream: CUstream,
    ) -> i32;
}

pub fn resident_footprint_capability_v2()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-footprint.f64.semantic-v2");
    implementation.update(include_bytes!("resident_footprint_v2.rs"));
    implementation.update(include_bytes!("../native/resident_footprint_v2.cu"));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/footprint_features.rs"
    ));
    implementation.update(include_bytes!("../../neoethos-data/src/core/timestamps.rs"));
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::Footprint,
        "neoethos.gpu-cuda.resident-footprint.f64.semantic-v2",
        implementation_sha256,
        FOOTPRINT_EXACT_MATH_AUTHORITY_V2,
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentFootprintRuntimeReceiptV2 {
    semantic_version: u32,
    row_count: usize,
    feature_column_count: usize,
    retained_feature_device_bytes: usize,
    prefix_scratch_device_bytes: usize,
    parent_input_h2d_bytes: usize,
    feature_value_d2h_bytes: usize,
    producer_ready_event_count: usize,
    native_launch_count: usize,
    logical_validity_schema: &'static str,
}

/// Owner-derived pre-device memory for the fixed seven-column Footprint-v2
/// producer. Value/validity staging is generic; the only producer scratch is
/// the exact `8 * (rows + 1)` f64 prefix buffer used by the native kernel.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentFootprintPreDeviceMemoryReceiptV4 {
    row_count: usize,
    feature_column_count: usize,
    additional_retained_bytes: usize,
    scratch_bytes: usize,
}

impl ResidentFootprintPreDeviceMemoryReceiptV4 {
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

pub fn preflight_resident_footprint_memory_v4(
    rows: usize,
) -> Result<ResidentFootprintPreDeviceMemoryReceiptV4, ResidentFeatureStoreCudaErrorV3> {
    if rows == 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Footprint pre-device memory requires at least one row".into(),
        ));
    }
    let scratch_bytes = rows
        .checked_add(1)
        .and_then(|extent| extent.checked_mul(FOOTPRINT_PREFIX_SERIES_V2))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f64>()))
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Footprint pre-device prefix scratch bytes",
        ))?;
    Ok(ResidentFootprintPreDeviceMemoryReceiptV4 {
        row_count: rows,
        feature_column_count: RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len(),
        additional_retained_bytes: 0,
        scratch_bytes,
    })
}

impl ResidentFootprintRuntimeReceiptV2 {
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }

    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    pub const fn feature_column_count(&self) -> usize {
        self.feature_column_count
    }

    pub const fn retained_feature_device_bytes(&self) -> usize {
        self.retained_feature_device_bytes
    }

    pub const fn prefix_scratch_device_bytes(&self) -> usize {
        self.prefix_scratch_device_bytes
    }

    pub const fn parent_input_h2d_bytes(&self) -> usize {
        self.parent_input_h2d_bytes
    }

    pub const fn feature_value_d2h_bytes(&self) -> usize {
        self.feature_value_d2h_bytes
    }

    pub const fn producer_ready_event_count(&self) -> usize {
        self.producer_ready_event_count
    }

    pub const fn native_launch_count(&self) -> usize {
        self.native_launch_count
    }

    pub const fn logical_validity_schema(&self) -> &'static str {
        self.logical_validity_schema
    }
}

#[derive(Debug)]
struct StreamOrderedFootprintBufferV2<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedFootprintBufferV2<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: the returned owner retains the exact context and stream, and
        // its destructor releases only with stream-ordered drop_async.
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

impl<T: DeviceCopy> Deref for StreamOrderedFootprintBufferV2<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live Footprint stream owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedFootprintBufferV2<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live Footprint stream owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedFootprintBufferV2<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if CurrentContext::set_current(self.context.as_ref()).is_ok() {
            let _ = buffer.drop_async(self.stream.as_ref());
        } else {
            // Never invoke DeviceBuffer's legacy synchronizing destructor when
            // the admitted primary context cannot be restored.
            std::mem::forget(buffer);
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResidentFootprintFeatureBatchV2 {
    feature_values: StreamOrderedFootprintBufferV2<f64>,
    feature_validity_u8: StreamOrderedFootprintBufferV2<u8>,
    prefix_scratch: StreamOrderedFootprintBufferV2<f64>,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    receipt: ResidentFootprintRuntimeReceiptV2,
}

impl ResidentFootprintFeatureBatchV2 {
    pub(crate) fn receipt(&self) -> &ResidentFootprintRuntimeReceiptV2 {
        &self.receipt
    }
}

pub(crate) fn launch_resident_footprint_v2(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &dyn ResidentParentDatasetSourceV3,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
) -> Result<ResidentFootprintFeatureBatchV2, ResidentFeatureStoreCudaErrorV3> {
    validate_bindings_v2(&bindings)?;
    let rows = parent.rows();
    if rows == 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Footprint requires at least one parent row".into(),
        ));
    }
    validate_parent_extents_v2(parent, rows)?;

    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let device_ordinal = run_device.device_identity().ordinal();
    CurrentContext::set_current(context.as_ref())?;
    if parent.producer_context().as_raw() != context.as_raw()
        || parent.producer_stream().as_inner() != stream.as_inner()
        || parent.device_ordinal() != device_ordinal
    {
        return Err(ResidentFeatureStoreCudaErrorV3::PrimaryContextMismatch);
    }
    if stream.as_inner().is_null() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "default/null CUDA streams are not admitted".into(),
        ));
    }
    parent.producer_ready_event().wait_before_read(
        context.as_ref(),
        stream.as_ref(),
        device_ordinal,
    )?;

    let feature_cells = rows
        .checked_mul(RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Footprint feature cells",
        ))?;
    let retained_feature_device_bytes = feature_cells
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<u8>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Footprint retained feature bytes",
        ))?;
    let prefix_elements = rows
        .checked_add(1)
        .and_then(|extent| extent.checked_mul(FOOTPRINT_PREFIX_SERIES_V2))
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Footprint prefix elements",
        ))?;
    let prefix_scratch_device_bytes = prefix_elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Footprint prefix scratch bytes",
        ))?;

    let feature_values = StreamOrderedFootprintBufferV2::<f64>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let feature_validity_u8 = StreamOrderedFootprintBufferV2::<u8>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let prefix_scratch = StreamOrderedFootprintBufferV2::<f64>::uninitialized_async(
        prefix_elements,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;

    // SAFETY: parent and output/scratch owners prove exact extents in this
    // primary context and retain every pointer through the producer event.
    let status = unsafe {
        neoethos_resident_footprint_f64_v2(
            parent.open().as_device_ptr().as_ptr(),
            parent.high().as_device_ptr().as_ptr(),
            parent.low().as_device_ptr().as_ptr(),
            parent.close().as_device_ptr().as_ptr(),
            parent.volume().as_device_ptr().as_ptr(),
            parent.timestamps().as_device_ptr().as_ptr(),
            rows,
            feature_values.as_device_ptr().as_mut_ptr(),
            feature_validity_u8.as_device_ptr().as_mut_ptr(),
            prefix_scratch.as_device_ptr().as_mut_ptr(),
            stream.as_inner(),
        )
    };
    if status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_footprint_f64_v2",
            status,
        });
    }
    let ready_event =
        ResidentProducerReadyEventV3::record(context.as_ref(), stream.as_ref(), device_ordinal)?;
    let receipt = ResidentFootprintRuntimeReceiptV2 {
        semantic_version: FOOTPRINT_SEMANTIC_VERSION_V2,
        row_count: rows,
        feature_column_count: RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len(),
        retained_feature_device_bytes,
        prefix_scratch_device_bytes,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        producer_ready_event_count: 1,
        native_launch_count: 2,
        logical_validity_schema: FOOTPRINT_LOGICAL_VALIDITY_SCHEMA_V2,
    };
    Ok(ResidentFootprintFeatureBatchV2 {
        feature_values,
        feature_validity_u8,
        prefix_scratch,
        bindings,
        rows,
        device_ordinal,
        context,
        stream,
        ready_event,
        receipt,
    })
}

fn validate_bindings_v2(
    bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    if bindings.len() != RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
            "resident Footprint requires exactly {} bindings, received {}",
            RESIDENT_FOOTPRINT_COLUMN_NAMES_V2.len(),
            bindings.len()
        )));
    }
    for (binding, expected_name) in bindings.iter().zip(RESIDENT_FOOTPRINT_COLUMN_NAMES_V2) {
        if binding.feature_name != expected_name {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Footprint schema expected `{expected_name}`, received `{}`",
                binding.feature_name
            )));
        }
    }
    Ok(())
}

fn validate_parent_extents_v2(
    parent: &dyn ResidentParentDatasetSourceV3,
    rows: usize,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    for (name, len) in [
        ("open", parent.open().len()),
        ("high", parent.high().len()),
        ("low", parent.low().len()),
        ("close", parent.close().len()),
        ("volume", parent.volume().len()),
        ("timestamps", parent.timestamps().len()),
    ] {
        if len != rows {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Footprint parent {name} extent {len} differs from {rows} rows"
            )));
        }
    }
    Ok(())
}

unsafe impl ResidentF64FeatureBatchV3 for ResidentFootprintFeatureBatchV2 {
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
        &self.ready_event
    }

    fn retained_device_bytes(&self) -> usize {
        self.receipt.retained_feature_device_bytes
    }

    fn retained_scratch_bytes(&self) -> usize {
        self.receipt.prefix_scratch_device_bytes
    }

    fn enqueue_nonblocking_release(
        self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self.feature_values.is_owned_by_stream(release_stream)
            || !self.feature_validity_u8.is_owned_by_stream(release_stream)
            || !self.prefix_scratch.is_owned_by_stream(release_stream)
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        drop(self);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_device_prefix_scratch_matches_the_eight_runtime_series() {
        let rows = 1_200;
        let receipt =
            preflight_resident_footprint_memory_v4(rows).expect("Footprint memory preflight");
        assert_eq!(
            receipt.scratch_bytes(),
            (rows + 1) * FOOTPRINT_PREFIX_SERIES_V2 * std::mem::size_of::<f64>()
        );
    }
}
