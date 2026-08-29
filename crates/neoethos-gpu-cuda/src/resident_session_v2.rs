//! Resident CUDA implementation of the atomic Session semantic-v2 family.
//!
//! One sequential kernel consumes the retained canonical OHLCV/millisecond
//! parent and emits all twenty-three feature-major f64/u8 columns. The typed
//! source closure is mandatory before either capability or launch authority
//! can exist; production never downloads feature values or validity bytes.

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
pub const RESIDENT_SESSION_SEMANTIC_VERSION_V2: u32 = 2;
pub const RESIDENT_SESSION_IMPLEMENTATION_ID_V2: &str =
    "neoethos.cuda.resident-session.semantic-v2";
pub const RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2: &str = "neoethos.session.cpu-oracle.semantic-v2;single-thread-sequential-f64;dual-clock=value-first16-nonzero-75pct-magnitude-inference+validity-canonical-ms;asian=00:00-08:00;london=07:00-16:00;new-york=12:00-21:00;overlap=12:00-16:00;cumulative-atr;canonical-qnan=0x7ff8000000000000;validity-codes=0,1,5;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";
pub const RESIDENT_SESSION_RETAINED_BYTES_PER_ROW_V2: usize = 207;
pub const RESIDENT_SESSION_POINTER_TABLE_DEVICE_BYTES_V2: usize = 736;
pub const RESIDENT_SESSION_ISOLATED_POINTER_SCHEMA_BYTES_V2: usize = 1_377;
pub const RESIDENT_SESSION_INVALID_NAN_BITS_V2: u64 = 0x7ff8_0000_0000_0000;
const RESIDENT_SESSION_LOGICAL_VALIDITY_SCHEMA_V2: &str =
    "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3";
const RESIDENT_SESSION_RELEASE_RECEIPT_V2: &str =
    include_str!("../tests/fixtures/resident_session_v2_device_parity.release.txt");
const RESIDENT_SESSION_RELEASE_EXACT_FIELDS_V2: [&str; 15] = [
    "verified=true",
    "semantic_version=2",
    "feature_column_count=23",
    "cpu_cuda_value_bit_mismatches=0",
    "cpu_cuda_validity_mismatches=0",
    "compute_sanitizer_errors=0",
    "compute_sanitizer_leaked_bytes=0",
    "racecheck_errors=0",
    "kernel_launch_count=257",
    "kernel_rows=4096",
    "kernel_feature_columns=23",
    "kernel_median_ns=35152055",
    "kernel_p95_ns=35166075",
    "input_h2d_copy_count=6",
    "feature_d2h_bytes=0",
];
const RESIDENT_SESSION_RELEASE_SHA256_KEYS_V2: [&str; 5] = [
    "device_identity_sha256",
    "parity_log_sha256",
    "sanitizer_log_sha256",
    "racecheck_log_sha256",
    "nsys_report_sha256",
];

pub const RESIDENT_SESSION_COLUMN_NAMES_V2: [&str; 23] = [
    "session_london_open_dist",
    "session_london_high_dist",
    "session_london_low_dist",
    "session_london_range",
    "session_london_vwap_dist",
    "session_ny_open_dist",
    "session_ny_high_dist",
    "session_ny_low_dist",
    "session_ny_range",
    "session_ny_vwap_dist",
    "session_asian_open_dist",
    "session_asian_close_dist",
    "session_asian_range_norm",
    "session_london_ny_overlap",
    "session_vol_ratio",
    "session_prev_close_dist",
    "session_open_gap",
    "daily_range_pct",
    "daily_body_pct",
    "daily_position",
    "daily_high_dist",
    "daily_low_dist",
    "daily_vwap_dist",
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NeoResidentSessionLaunchV2 {
    abi_version: u32,
    semantic_version: u32,
    feature_column_count: u32,
    reserved: u32,
    row_count: u64,
    open: *const f64,
    high: *const f64,
    low: *const f64,
    close: *const f64,
    volume: *const f64,
    timestamps_ms: *const i64,
    feature_values: *mut f64,
    feature_validity_u8: *mut u8,
}

const _: () = {
    assert!(std::mem::size_of::<NeoResidentSessionLaunchV2>() == 88);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, row_count) == 16);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, open) == 24);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, timestamps_ms) == 64);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, feature_values) == 72);
    assert!(std::mem::offset_of!(NeoResidentSessionLaunchV2, feature_validity_u8) == 80);
};

unsafe extern "C" {
    fn neoethos_resident_session_f64_v2(
        launch: *const NeoResidentSessionLaunchV2,
        stream: CUstream,
    ) -> i32;
}

/// Move-only proof that every Rust, C ABI, CUDA and CPU-oracle byte required
/// by Session-v2 was present when its implementation identity was minted.
#[must_use = "Session-v2 source closure must move into one launch authority"]
#[derive(Debug)]
pub struct SealedResidentSessionSourceClosureV2 {
    implementation_sha256: [u8; SHA256_BYTES],
}

impl SealedResidentSessionSourceClosureV2 {
    pub const fn implementation_sha256(&self) -> [u8; SHA256_BYTES] {
        self.implementation_sha256
    }
}

pub fn seal_resident_session_source_closure_v2() -> SealedResidentSessionSourceClosureV2 {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-session.f64.semantic-v2\0");
    implementation.update(include_bytes!("resident_session_v2.rs"));
    implementation.update(include_bytes!("../native/resident_session_v2_abi.cuh"));
    implementation.update(include_bytes!("../native/resident_session_v2.cu"));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/session_features.rs"
    ));
    implementation.update(include_bytes!("../../neoethos-data/src/core/timestamps.rs"));
    implementation.update(include_bytes!("../../neoethos-data/src/core/features.rs"));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/gpu_resident_session_v2.rs"
    ));
    implementation.update(include_bytes!(
        "../tests/fixtures/resident_session_v2_device_parity.release.txt"
    ));
    implementation.update(RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2.as_bytes());
    SealedResidentSessionSourceClosureV2 {
        implementation_sha256: implementation.finalize().into(),
    }
}

fn receipt_has_nonzero_sha256_v2(line: &str, key: &str) -> bool {
    let Some(value) = line
        .strip_prefix(key)
        .and_then(|suffix| suffix.strip_prefix('='))
    else {
        return false;
    };
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn validate_resident_session_release_receipt_v2(
    receipt: &str,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    let mut lines = receipt.lines();
    for expected in RESIDENT_SESSION_RELEASE_EXACT_FIELDS_V2 {
        if lines.next() != Some(expected) {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Session-v2 capability is held closed pending exact required-card field `{expected}`"
            )));
        }
    }
    for key in RESIDENT_SESSION_RELEASE_SHA256_KEYS_V2 {
        if !lines
            .next()
            .is_some_and(|line| receipt_has_nonzero_sha256_v2(line, key))
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Session-v2 capability is held closed pending canonical nonzero SHA-256 field `{key}`"
            )));
        }
    }
    if lines.next().is_some() || !receipt.ends_with('\n') {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Session-v2 capability release receipt has a noncanonical schema".into(),
        ));
    }
    Ok(())
}

pub fn resident_session_capability_v2()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let device_receipt = RESIDENT_SESSION_RELEASE_RECEIPT_V2;
    validate_resident_session_release_receipt_v2(device_receipt)?;
    let closure = seal_resident_session_source_closure_v2();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::Session,
        RESIDENT_SESSION_IMPLEMENTATION_ID_V2,
        closure.implementation_sha256(),
        RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2,
    )
    .map_err(Into::into)
}

/// Data-sealed, move-only authority for one exact admitted input. A caller
/// cannot construct this from raw pointers, row counts or capability flags.
#[must_use = "Session-v2 launch authority must be consumed by its native launch"]
#[derive(Debug)]
pub struct ResidentSessionLaunchAuthorityV2 {
    row_count: usize,
    input_identity_sha256: [u8; SHA256_BYTES],
    semantic_source_sha256: [u8; SHA256_BYTES],
    implementation_sha256: [u8; SHA256_BYTES],
}

impl ResidentSessionLaunchAuthorityV2 {
    pub fn seal(
        row_count: usize,
        input_identity_sha256: [u8; SHA256_BYTES],
        semantic_source_sha256: [u8; SHA256_BYTES],
        closure: SealedResidentSessionSourceClosureV2,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        if row_count == 0
            || input_identity_sha256 == [0; SHA256_BYTES]
            || semantic_source_sha256 == [0; SHA256_BYTES]
            || closure.implementation_sha256() == [0; SHA256_BYTES]
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Session-v2 launch authority is incomplete".into(),
            ));
        }
        Ok(Self {
            row_count,
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256: closure.implementation_sha256(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentSessionRuntimeReceiptV2 {
    semantic_version: u32,
    row_count: usize,
    feature_column_count: usize,
    retained_feature_device_bytes: usize,
    additional_retained_device_bytes: usize,
    scratch_device_bytes: usize,
    pointer_table_device_bytes: usize,
    isolated_pointer_schema_metadata_bytes: usize,
    parent_input_h2d_bytes: usize,
    feature_value_d2h_bytes: usize,
    producer_ready_event_count: usize,
    producer_ready_event_synchronize_count: usize,
    native_launch_count: usize,
    host_synchronize_count: usize,
    logical_validity_schema: &'static str,
    logical_validity_codes: [u8; 3],
    invalid_nan_bits: u64,
    input_identity_sha256: [u8; SHA256_BYTES],
    semantic_source_sha256: [u8; SHA256_BYTES],
    implementation_sha256: [u8; SHA256_BYTES],
}

impl ResidentSessionRuntimeReceiptV2 {
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
    pub const fn producer_ready_event_synchronize_count(&self) -> usize {
        self.producer_ready_event_synchronize_count
    }
    pub const fn native_launch_count(&self) -> usize {
        self.native_launch_count
    }
    pub const fn host_synchronize_count(&self) -> usize {
        self.host_synchronize_count
    }
    pub const fn logical_validity_schema(&self) -> &'static str {
        self.logical_validity_schema
    }
    pub const fn logical_validity_codes(&self) -> [u8; 3] {
        self.logical_validity_codes
    }
    pub const fn invalid_nan_bits(&self) -> u64 {
        self.invalid_nan_bits
    }
    pub const fn input_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.input_identity_sha256
    }
    pub const fn semantic_source_sha256(&self) -> [u8; SHA256_BYTES] {
        self.semantic_source_sha256
    }
    pub const fn implementation_sha256(&self) -> [u8; SHA256_BYTES] {
        self.implementation_sha256
    }
}

#[derive(Debug)]
struct StreamOrderedSessionBufferV2<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedSessionBufferV2<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        // SAFETY: this owner retains the exact context and stream and releases
        // only with stream-ordered drop_async.
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

impl<T: DeviceCopy> Deref for StreamOrderedSessionBufferV2<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live Session-v2 owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedSessionBufferV2<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live Session-v2 owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedSessionBufferV2<T> {
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
pub(crate) struct ResidentSessionFeatureBatchV2 {
    feature_values: StreamOrderedSessionBufferV2<f64>,
    feature_validity_u8: StreamOrderedSessionBufferV2<u8>,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    receipt: ResidentSessionRuntimeReceiptV2,
}

impl ResidentSessionFeatureBatchV2 {
    pub(crate) fn receipt(&self) -> &ResidentSessionRuntimeReceiptV2 {
        &self.receipt
    }
}

pub(crate) fn launch_resident_session_v2(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &dyn ResidentParentDatasetSourceV3,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    authority: ResidentSessionLaunchAuthorityV2,
) -> Result<ResidentSessionFeatureBatchV2, ResidentFeatureStoreCudaErrorV3> {
    validate_bindings_v2(&bindings)?;
    let rows = parent.rows();
    if rows == 0 || rows != authority.row_count {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Session-v2 parent rows disagree with launch authority".into(),
        ));
    }
    validate_parent_extents_v2(parent, rows)?;

    let feature_cells = rows
        .checked_mul(RESIDENT_SESSION_COLUMN_NAMES_V2.len())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Session-v2 feature cells",
        ))?;
    let retained_feature_device_bytes = rows
        .checked_mul(RESIDENT_SESSION_RETAINED_BYTES_PER_ROW_V2)
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Session-v2 retained feature bytes",
        ))?;
    let computed_retained_bytes = feature_cells
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<u8>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Session-v2 value/validity bytes",
        ))?;
    if computed_retained_bytes != retained_feature_device_bytes {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Session-v2 207N byte authority drifted".into(),
        ));
    }

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

    let feature_values = StreamOrderedSessionBufferV2::<f64>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let feature_validity_u8 = StreamOrderedSessionBufferV2::<u8>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let native_launch = NeoResidentSessionLaunchV2 {
        abi_version: 2,
        semantic_version: RESIDENT_SESSION_SEMANTIC_VERSION_V2,
        feature_column_count: RESIDENT_SESSION_COLUMN_NAMES_V2.len() as u32,
        reserved: 0,
        row_count: rows as u64,
        open: parent.open().as_device_ptr().as_ptr(),
        high: parent.high().as_device_ptr().as_ptr(),
        low: parent.low().as_device_ptr().as_ptr(),
        close: parent.close().as_device_ptr().as_ptr(),
        volume: parent.volume().as_device_ptr().as_ptr(),
        timestamps_ms: parent.timestamps().as_device_ptr().as_ptr(),
        feature_values: feature_values.as_device_ptr().as_mut_ptr(),
        feature_validity_u8: feature_validity_u8.as_device_ptr().as_mut_ptr(),
    };
    // SAFETY: parent and output owners prove exact extents in the same
    // admitted primary context and retain all pointers through the ready event.
    let status = unsafe { neoethos_resident_session_f64_v2(&native_launch, stream.as_inner()) };
    if status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_session_f64_v2",
            status,
        });
    }
    let ready_event =
        ResidentProducerReadyEventV3::record(context.as_ref(), stream.as_ref(), device_ordinal)?;
    let receipt = ResidentSessionRuntimeReceiptV2 {
        semantic_version: RESIDENT_SESSION_SEMANTIC_VERSION_V2,
        row_count: rows,
        feature_column_count: RESIDENT_SESSION_COLUMN_NAMES_V2.len(),
        retained_feature_device_bytes,
        additional_retained_device_bytes: 0,
        scratch_device_bytes: 0,
        pointer_table_device_bytes: RESIDENT_SESSION_POINTER_TABLE_DEVICE_BYTES_V2,
        isolated_pointer_schema_metadata_bytes: RESIDENT_SESSION_ISOLATED_POINTER_SCHEMA_BYTES_V2,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        producer_ready_event_count: 1,
        producer_ready_event_synchronize_count: 0,
        native_launch_count: 1,
        host_synchronize_count: 0,
        logical_validity_schema: RESIDENT_SESSION_LOGICAL_VALIDITY_SCHEMA_V2,
        logical_validity_codes: [0, 1, 5],
        invalid_nan_bits: RESIDENT_SESSION_INVALID_NAN_BITS_V2,
        input_identity_sha256: authority.input_identity_sha256,
        semantic_source_sha256: authority.semantic_source_sha256,
        implementation_sha256: authority.implementation_sha256,
    };
    Ok(ResidentSessionFeatureBatchV2 {
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

fn validate_bindings_v2(
    bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    if bindings.len() != RESIDENT_SESSION_COLUMN_NAMES_V2.len() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
            "resident Session-v2 requires exactly {} bindings, received {}",
            RESIDENT_SESSION_COLUMN_NAMES_V2.len(),
            bindings.len()
        )));
    }
    for (binding, expected_name) in bindings.iter().zip(RESIDENT_SESSION_COLUMN_NAMES_V2) {
        if binding.feature_name != expected_name {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Session-v2 schema expected `{expected_name}`, received `{}`",
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
                "resident Session-v2 parent {name} extent {len} differs from {rows} rows"
            )));
        }
    }
    Ok(())
}

unsafe impl ResidentF64FeatureBatchV3 for ResidentSessionFeatureBatchV2 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_receipt_accepts_exact_frozen_evidence() {
        validate_resident_session_release_receipt_v2(RESIDENT_SESSION_RELEASE_RECEIPT_V2)
            .expect("exact required-card Session-v2 receipt");
        let capability = resident_session_capability_v2().expect("verified Session-v2 capability");
        assert_eq!(capability.producer(), ResidentFeatureProducerV3::Session);
        assert_eq!(
            capability.implementation_id(),
            RESIDENT_SESSION_IMPLEMENTATION_ID_V2
        );
        assert_eq!(
            capability.exact_math_authority(),
            RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2
        );
        assert_ne!(capability.implementation_sha256(), [0; SHA256_BYTES]);
    }

    #[test]
    fn release_receipt_rejects_drift_or_zero_hashes() {
        let parity_line = RESIDENT_SESSION_RELEASE_RECEIPT_V2
            .lines()
            .find(|line| line.starts_with("parity_log_sha256="))
            .expect("frozen parity receipt line");
        let zero_parity = format!("parity_log_sha256={}", "0".repeat(64));
        for drifted in [
            RESIDENT_SESSION_RELEASE_RECEIPT_V2.replacen(
                "feature_d2h_bytes=0",
                "feature_d2h_bytes=8",
                1,
            ),
            RESIDENT_SESSION_RELEASE_RECEIPT_V2.replacen(parity_line, &zero_parity, 1),
            format!("{RESIDENT_SESSION_RELEASE_RECEIPT_V2}unexpected_field=true\n"),
            RESIDENT_SESSION_RELEASE_RECEIPT_V2
                .trim_end_matches('\n')
                .to_owned(),
        ] {
            assert!(
                validate_resident_session_release_receipt_v2(&drifted).is_err(),
                "drifted Session-v2 release evidence must fail closed"
            );
        }
    }
}
