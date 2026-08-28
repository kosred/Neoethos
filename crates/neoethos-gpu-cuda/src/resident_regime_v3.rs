//! Resident CUDA Regime semantic-v3 producer.
//!
//! The one admitted parent upload, primary context and non-default run stream
//! are borrowed in place. Two native launches emit fourteen f64/u8 columns;
//! neither feature payload nor validity is materialized on the host.

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
pub const RESIDENT_REGIME_SEMANTIC_VERSION_V3: u32 = 3;
pub const RESIDENT_REGIME_OPERATION_SCHEDULE_V1: &str =
    "neoethos.regime.semantic-v3.f64-rn-fixed-order-log49-neumaier-v1";
pub const RESIDENT_REGIME_FIXTURE_SHA256_V1: &str =
    "f0f89c26727e90206bb85bdb4b3f6e11f59652176f7ba8475e9fbaa301548a93";
pub const RESIDENT_REGIME_LOG49_OPERATION_TOKENS_SHA256_V1: &str =
    "73002b6761d1ca425250a761fa4411cf3ae0d26c862caa964e93063c69c32080";
pub const RESIDENT_REGIME_LOG49_RUST_MIRROR_SHA256_V1: &str =
    "f7d83af4d95a95c38cb360abcee96a223f4010aba2e3c679145c091e56db8fea";
pub const RESIDENT_REGIME_LOG49_CUDA_MIRROR_SHA256_V1: &str =
    "ec8299d718d7a3d5a189287380f042df603fde7bbed87b7378845d7ce73618fe";
const REGIME_LOGICAL_VALIDITY_SCHEMA_V3: &str =
    "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3";
const REGIME_EXACT_MATH_AUTHORITY_V3: &str = "neoethos.regime.semantic-v3.f64-rn-fixed-order-log49-neumaier-v1;log49-token-sha256=73002b6761d1ca425250a761fa4411cf3ae0d26c862caa964e93063c69c32080;rust-mirror-sha256=f7d83af4d95a95c38cb360abcee96a223f4010aba2e3c679145c091e56db8fea;cuda-mirror-sha256=ec8299d718d7a3d5a189287380f042df603fde7bbed87b7378845d7ce73618fe;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";

pub const RESIDENT_REGIME_COLUMN_NAMES_V3: [&str; 14] = [
    "neoethos_custom_gk_vol_ratio_state_10_50_v3",
    "neoethos_custom_gk_vol_ratio_offset_10_50_v3",
    "regime_wilder_adx_14_v3",
    "neoethos_custom_wilder_di_dominance_direction_14_v3",
    "neoethos_custom_wilder_adx_direction_state_14_25_v3",
    "neoethos_custom_bollinger_keltner_squeeze_state_20_2_1p5_v3",
    "neoethos_custom_bollinger_midline_atr_deviation_20_v3",
    "neoethos_custom_directional_persistence_balance_20_v3",
    "neoethos_custom_candle_body_range_balance_8_v3",
    "regime_dreiss_choppiness_index_14_v3",
    "neoethos_custom_standardized_cusum_up_50_0p5_3_v3",
    "neoethos_custom_standardized_cusum_down_50_0p5_3_v3",
    "neoethos_custom_standardized_cusum_signal_50_0p5_3_v3",
    "neoethos_custom_equal_width_log_return_entropy_30_10_v3",
];

unsafe extern "C" {
    fn neoethos_resident_regime_independent_f64_v3(
        open: *const f64,
        high: *const f64,
        low: *const f64,
        close: *const f64,
        rows: usize,
        scale_anchor: f64,
        feature_values: *mut f64,
        feature_validity_u8: *mut u8,
        stream: CUstream,
    ) -> i32;
    fn neoethos_resident_regime_recurrence_f64_v3(
        high: *const f64,
        low: *const f64,
        close: *const f64,
        rows: usize,
        scale_anchor: f64,
        feature_values: *mut f64,
        feature_validity_u8: *mut u8,
        stream: CUstream,
    ) -> i32;
}

pub fn resident_regime_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-regime.f64.semantic-v3");
    implementation.update(include_bytes!("resident_regime_v3.rs"));
    implementation.update(include_bytes!("../native/resident_regime_v3.cu"));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/regime_detection.rs"
    ));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/regime_exact_math_v1.rs"
    ));
    implementation.update(REGIME_EXACT_MATH_AUTHORITY_V3.as_bytes());
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::Regime,
        "neoethos.gpu-cuda.resident-regime.f64.semantic-v3",
        implementation_sha256,
        REGIME_EXACT_MATH_AUTHORITY_V3,
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentRegimeRuntimeReceiptV3 {
    semantic_version: u32,
    row_count: usize,
    feature_column_count: usize,
    scale_anchor_bits: u64,
    retained_feature_device_bytes: usize,
    additional_retained_device_bytes: usize,
    scratch_device_bytes: usize,
    pointer_table_device_bytes: usize,
    isolated_pointer_schema_metadata_bytes: usize,
    parent_input_h2d_bytes: usize,
    feature_value_d2h_bytes: usize,
    producer_ready_event_count: usize,
    native_launch_count: usize,
    logical_validity_schema: &'static str,
}

/// Owner-derived pre-device memory for the fixed fourteen-column Regime-v3
/// producer. The generic planner accounts value/validity staging; Regime owns
/// no additional retained state or scratch allocation.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentRegimePreDeviceMemoryReceiptV4 {
    row_count: usize,
    feature_column_count: usize,
    additional_retained_bytes: usize,
    scratch_bytes: usize,
}

impl ResidentRegimePreDeviceMemoryReceiptV4 {
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

pub fn preflight_resident_regime_memory_v4(
    rows: usize,
) -> Result<ResidentRegimePreDeviceMemoryReceiptV4, ResidentFeatureStoreCudaErrorV3> {
    if rows == 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Regime pre-device memory requires at least one row".into(),
        ));
    }
    Ok(ResidentRegimePreDeviceMemoryReceiptV4 {
        row_count: rows,
        feature_column_count: RESIDENT_REGIME_COLUMN_NAMES_V3.len(),
        additional_retained_bytes: 0,
        scratch_bytes: 0,
    })
}

impl ResidentRegimeRuntimeReceiptV3 {
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }
    pub const fn row_count(&self) -> usize {
        self.row_count
    }
    pub const fn feature_column_count(&self) -> usize {
        self.feature_column_count
    }
    pub const fn scale_anchor_bits(&self) -> u64 {
        self.scale_anchor_bits
    }
    pub const fn retained_feature_device_bytes(&self) -> usize {
        self.retained_feature_device_bytes
    }
    pub const fn additional_retained_device_bytes(&self) -> usize {
        self.additional_retained_device_bytes
    }
    pub const fn scratch_device_bytes(&self) -> usize {
        self.scratch_device_bytes
    }
    pub const fn pointer_table_device_bytes(&self) -> usize {
        self.pointer_table_device_bytes
    }
    pub const fn isolated_pointer_schema_metadata_bytes(&self) -> usize {
        self.isolated_pointer_schema_metadata_bytes
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
struct StreamOrderedRegimeBufferV3<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedRegimeBufferV3<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: this owner retains the exact context/stream and releases the
        // allocation only with stream-ordered drop_async.
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

impl<T: DeviceCopy> Deref for StreamOrderedRegimeBufferV3<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live Regime owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedRegimeBufferV3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live Regime owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedRegimeBufferV3<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if CurrentContext::set_current(self.context.as_ref()).is_ok() {
            let _ = buffer.drop_async(self.stream.as_ref());
        } else {
            std::mem::forget(buffer);
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResidentRegimeFeatureBatchV3 {
    feature_values: StreamOrderedRegimeBufferV3<f64>,
    feature_validity_u8: StreamOrderedRegimeBufferV3<u8>,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    receipt: ResidentRegimeRuntimeReceiptV3,
}

impl ResidentRegimeFeatureBatchV3 {
    pub(crate) fn receipt(&self) -> &ResidentRegimeRuntimeReceiptV3 {
        &self.receipt
    }
}

fn exact_power_of_two_v3(value: f64) -> bool {
    if !value.is_finite() || value <= 0.0 {
        return false;
    }
    let bits = value.to_bits();
    let exponent = bits & 0x7ff0_0000_0000_0000;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0 {
        fraction.count_ones() == 1
    } else {
        fraction == 0
    }
}

pub(crate) fn launch_resident_regime_v3(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &dyn ResidentParentDatasetSourceV3,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    scale_anchor: f64,
) -> Result<ResidentRegimeFeatureBatchV3, ResidentFeatureStoreCudaErrorV3> {
    validate_bindings_v3(&bindings)?;
    if !exact_power_of_two_v3(scale_anchor) {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Regime scale anchor is not an exact positive power of two".into(),
        ));
    }
    let rows = parent.rows();
    if rows == 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Regime requires at least one parent row".into(),
        ));
    }
    validate_parent_extents_v3(parent, rows)?;

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
        .checked_mul(RESIDENT_REGIME_COLUMN_NAMES_V3.len())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Regime feature cells",
        ))?;
    let retained_feature_device_bytes =
        rows.checked_mul(126)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident Regime retained feature bytes",
            ))?;
    let computed_retained_bytes = feature_cells
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<u8>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Regime value/validity bytes",
        ))?;
    if computed_retained_bytes != retained_feature_device_bytes {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Regime 126N byte authority drifted".into(),
        ));
    }

    let feature_values = StreamOrderedRegimeBufferV3::<f64>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let feature_validity_u8 = StreamOrderedRegimeBufferV3::<u8>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;

    // SAFETY: the retained parent and output owners prove exact extents in the
    // same admitted context/stream through the producer-ready event.
    let independent_status = unsafe {
        neoethos_resident_regime_independent_f64_v3(
            parent.open().as_device_ptr().as_ptr(),
            parent.high().as_device_ptr().as_ptr(),
            parent.low().as_device_ptr().as_ptr(),
            parent.close().as_device_ptr().as_ptr(),
            rows,
            scale_anchor,
            feature_values.as_device_ptr().as_mut_ptr(),
            feature_validity_u8.as_device_ptr().as_mut_ptr(),
            stream.as_inner(),
        )
    };
    if independent_status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_regime_independent_f64_v3",
            status: independent_status,
        });
    }
    // SAFETY: stream order makes launch two observe the same retained parent
    // and disjoint Regime slots after launch one without host synchronization.
    let recurrence_status = unsafe {
        neoethos_resident_regime_recurrence_f64_v3(
            parent.high().as_device_ptr().as_ptr(),
            parent.low().as_device_ptr().as_ptr(),
            parent.close().as_device_ptr().as_ptr(),
            rows,
            scale_anchor,
            feature_values.as_device_ptr().as_mut_ptr(),
            feature_validity_u8.as_device_ptr().as_mut_ptr(),
            stream.as_inner(),
        )
    };
    if recurrence_status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_regime_recurrence_f64_v3",
            status: recurrence_status,
        });
    }
    let ready_event =
        ResidentProducerReadyEventV3::record(context.as_ref(), stream.as_ref(), device_ordinal)?;
    let receipt = ResidentRegimeRuntimeReceiptV3 {
        semantic_version: RESIDENT_REGIME_SEMANTIC_VERSION_V3,
        row_count: rows,
        feature_column_count: RESIDENT_REGIME_COLUMN_NAMES_V3.len(),
        scale_anchor_bits: scale_anchor.to_bits(),
        retained_feature_device_bytes,
        additional_retained_device_bytes: 0,
        scratch_device_bytes: 0,
        pointer_table_device_bytes: 448,
        isolated_pointer_schema_metadata_bytes: 1235,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        producer_ready_event_count: 1,
        native_launch_count: 2,
        logical_validity_schema: REGIME_LOGICAL_VALIDITY_SCHEMA_V3,
    };
    Ok(ResidentRegimeFeatureBatchV3 {
        feature_values,
        feature_validity_u8,
        bindings,
        rows,
        device_ordinal,
        context,
        stream,
        ready_event,
        receipt,
    })
}

fn validate_bindings_v3(
    bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    if bindings.len() != RESIDENT_REGIME_COLUMN_NAMES_V3.len() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
            "resident Regime requires exactly {} bindings, received {}",
            RESIDENT_REGIME_COLUMN_NAMES_V3.len(),
            bindings.len()
        )));
    }
    for (binding, expected_name) in bindings.iter().zip(RESIDENT_REGIME_COLUMN_NAMES_V3) {
        if binding.feature_name != expected_name {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Regime schema expected `{expected_name}`, received `{}`",
                binding.feature_name
            )));
        }
    }
    Ok(())
}

fn validate_parent_extents_v3(
    parent: &dyn ResidentParentDatasetSourceV3,
    rows: usize,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    for (name, len) in [
        ("open", parent.open().len()),
        ("high", parent.high().len()),
        ("low", parent.low().len()),
        ("close", parent.close().len()),
    ] {
        if len != rows {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Regime parent {name} extent {len} differs from {rows} rows"
            )));
        }
    }
    Ok(())
}

unsafe impl ResidentF64FeatureBatchV3 for ResidentRegimeFeatureBatchV3 {
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
        self.receipt.scratch_device_bytes
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
