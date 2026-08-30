//! Resident CUDA implementation of the complete Quant semantic-v4 family.
//!
//! A source-sealed, move-only launch authority binds one typed intraday grid,
//! the explicit 252-session annualization rule, retained parent buffers and a
//! single non-default stream. The one native launch emits all sixty-three
//! feature-major f64/u8 columns without feature-value D2H.

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
const UTC_DAY_MILLIS_V3: u64 = 86_400_000;
const ASIAN_SESSION_MILLIS_V3: u64 = 28_800_000;
pub const RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4: u32 = 4;
pub const RESIDENT_QUANT_IMPLEMENTATION_ID_V4: &str = "neoethos.cuda.resident-quant.semantic-v4";
pub const RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4: &str = "neoethos.quant.cpu-cuda.semantic-v4;sun-fdlibm-openlibm-e_log;commit=82e90aef0657289192efe77be89791c07dea0775;source-sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD;cpu-cuda-bit-tolerance=zero;real-log-accuracy=bounded-faithful-max-1ulp-reviewed-wide-domain;f64-fixed-order;canonical-ms-fixed-intraday;utc-day-open=00:00;asian-session=00:00-08:00;trading-sessions-per-year=252;annualization=sqrt(252*bars-per-utc-day);orb=asian-session-reset;cumulative-delta-validity=rolling-50-increments;validity=logical-u8-v3;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";
pub const RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY: &str = "neoethos.quant.migration.v2-to-v3;bitwise-preserved-v2-routes=31;migrated-existing-exact-log-routes=10;migrated-annualized-exact-log-routes=8;migrated-temporal-routes=14;changed-routes=32;trading_sessions_per_year=252;v2-artifacts=fail-closed;unversioned-artifacts=fail-closed;never-label-as-bitwise-v2-parity";
pub const RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY: &str = "neoethos.quant.migration.v3-to-v4;changed-routes=1;route=quant_cum_delta_zscore;raw-value-formula=unchanged;materialized-value-bits=change-on-validity-recovery;validity=rolling-50-increments;v3-artifacts=fail-closed;unversioned-artifacts=fail-closed";
pub const RESIDENT_QUANT_RETAINED_BYTES_PER_ROW_V3: usize = 567;
pub const RESIDENT_QUANT_POINTER_TABLE_DEVICE_BYTES_V3: usize = 2_016;
pub const RESIDENT_QUANT_ISOLATED_POINTER_SCHEMA_BYTES_V3: usize = 3_575;
pub const RESIDENT_QUANT_INVALID_NAN_BITS_V3: u64 = 0x7ff8_0000_0000_0000;
pub const RESIDENT_QUANT_VERIFIED_RELEASE_RECEIPT_SHA256_V3: [u8; SHA256_BYTES] = [
    0xcf, 0xa5, 0x7c, 0x76, 0x88, 0x3c, 0xd4, 0x75, 0xf8, 0x7b, 0x9b, 0x7b, 0x82, 0x70, 0xcb, 0x16,
    0xdf, 0x56, 0x87, 0x1e, 0x88, 0xe6, 0x8c, 0x40, 0xf6, 0xe6, 0xd5, 0x08, 0x26, 0x7b, 0xbc, 0x44,
];
const RESIDENT_QUANT_LOGICAL_VALIDITY_SCHEMA_V3: &str =
    "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3";

pub const RESIDENT_QUANT_COLUMN_NAMES_V3: [&str; 63] = [
    "quant_close",
    "quant_return_1",
    "quant_return_2",
    "quant_return_3",
    "quant_return_5",
    "quant_return_8",
    "quant_return_13",
    "quant_return_21",
    "quant_log_return",
    "quant_log_volatility",
    "quant_realized_vol_5",
    "quant_realized_vol_10",
    "quant_realized_vol_20",
    "quant_realized_vol_50",
    "quant_gk_vol_10",
    "quant_gk_vol_20",
    "quant_parkinson_vol_10",
    "quant_parkinson_vol_20",
    "quant_vol_ratio",
    "quant_hurst_100",
    "quant_autocorr_1",
    "quant_autocorr_5",
    "quant_autocorr_10",
    "quant_efficiency_ratio_10",
    "quant_efficiency_ratio_20",
    "quant_skewness_30",
    "quant_kurtosis_30",
    "quant_kyle_lambda",
    "quant_vpin",
    "quant_amihud_illiquidity",
    "quant_roll_spread",
    "quant_consec_up",
    "quant_consec_down",
    "quant_inside_bar",
    "quant_outside_bar",
    "quant_body_ratio",
    "quant_upper_shadow",
    "quant_lower_shadow",
    "quant_prev_day_h_dist",
    "quant_prev_day_l_dist",
    "quant_prev_week_h_dist",
    "quant_prev_week_l_dist",
    "quant_orb_4",
    "quant_orb_8",
    "quant_orb_12",
    "quant_amd_phase",
    "quant_wyckoff",
    "quant_engulfing_vol",
    "quant_pivot_dist",
    "quant_r1_dist",
    "quant_r2_dist",
    "quant_s1_dist",
    "quant_s2_dist",
    "quant_cam_r3_dist",
    "quant_cam_s3_dist",
    "quant_zscore_20",
    "quant_zscore_50",
    "quant_fractal_dim",
    "quant_rvol_10",
    "quant_rvol_20",
    "quant_rvol_50",
    "quant_delta_volume",
    "quant_cum_delta_zscore",
];

#[cfg(feature = "cuda-device-fixtures")]
#[path = "resident_quant_v3_device_fixture.rs"]
mod resident_quant_v3_device_fixture;
#[cfg(feature = "cuda-device-fixtures")]
pub use resident_quant_v3_device_fixture::{
    ResidentQuantDeviceFixtureOutputV3, run_resident_quant_v3_device_fixture,
    run_resident_quant_v3_device_perf_fixture,
};

#[repr(C)]
#[derive(Debug)]
struct NeoResidentQuantLaunchV3 {
    abi_version: u32,
    semantic_version: u32,
    feature_column_count: u32,
    reserved: u32,
    row_count: u64,
    timeframe_millis: u64,
    bars_per_asian_session: u64,
    bars_per_utc_day: u64,
    bars_per_trading_week: u64,
    trading_sessions_per_year: u64,
    annualization_periods_per_year: u64,
    open: *const f64,
    high: *const f64,
    low: *const f64,
    close: *const f64,
    volume: *const f64,
    timestamps: *const i64,
    feature_values: *mut f64,
    feature_validity_u8: *mut u8,
}

const _: () = {
    assert!(std::mem::size_of::<NeoResidentQuantLaunchV3>() == 136);
    assert!(std::mem::offset_of!(NeoResidentQuantLaunchV3, open) == 72);
    assert!(std::mem::offset_of!(NeoResidentQuantLaunchV3, feature_validity_u8) == 128);
};

unsafe extern "C" {
    fn neoethos_resident_quant_f64_v3(
        launch: *const NeoResidentQuantLaunchV3,
        stream: CUstream,
    ) -> i32;
}

/// Move-only proof that the exact Quant-v3 Rust, ABI, CUDA, CPU-oracle and
/// immutable OpenLibm authority bytes were closed before capability minting.
#[must_use = "Quant-v3 migration closure must move into one launch authority"]
#[derive(Debug)]
pub struct SealedResidentQuantMigrationClosureV3 {
    implementation_sha256: [u8; SHA256_BYTES],
}

impl SealedResidentQuantMigrationClosureV3 {
    pub const fn implementation_sha256(&self) -> [u8; SHA256_BYTES] {
        self.implementation_sha256
    }
}

pub fn seal_resident_quant_migration_closure_v3() -> SealedResidentQuantMigrationClosureV3 {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-quant.f64.semantic-v4\0");
    implementation.update(include_bytes!("resident_quant_v3.rs"));
    implementation.update(include_bytes!("resident_quant_v3_census.rs"));
    implementation.update(include_bytes!("../native/resident_quant_v3_abi.cuh"));
    implementation.update(include_bytes!("../native/resident_exact_log_v3.cuh"));
    implementation.update(include_bytes!("../native/resident_quant_v3.cu"));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/quant_features.rs"
    ));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/quant_exact_math_v3.rs"
    ));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/gpu_resident_quant_v3.rs"
    ));
    implementation.update(include_bytes!(
        "../../../vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.c"
    ));
    implementation.update(include_bytes!(
        "../../../vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.receipt.txt"
    ));
    implementation.update(include_bytes!(
        "../tests/fixtures/resident_quant_v3_device_parity.release.txt"
    ));
    implementation.update(RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4.as_bytes());
    implementation.update(RESIDENT_QUANT_V2_TO_V3_MIGRATION_POLICY.as_bytes());
    implementation.update(RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY.as_bytes());
    SealedResidentQuantMigrationClosureV3 {
        implementation_sha256: implementation.finalize().into(),
    }
}

fn receipt_has_nonzero_sha256_v3(receipt: &str, key: &str) -> bool {
    let prefix = format!("{key}=");
    receipt
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .is_some_and(|value| {
            value.len() == 64
                && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                && value.bytes().any(|byte| byte != b'0')
        })
}

pub fn resident_quant_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let device_receipt =
        include_str!("../tests/fixtures/resident_quant_v3_device_parity.release.txt");
    let observed_receipt_sha256: [u8; SHA256_BYTES] =
        Sha256::digest(device_receipt.as_bytes()).into();
    if observed_receipt_sha256 != RESIDENT_QUANT_VERIFIED_RELEASE_RECEIPT_SHA256_V3 {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Quant-v3 capability refuses an unpinned release receipt".into(),
        ));
    }
    for required in [
        "verified=true",
        "semantic_version=4",
        "feature_column_count=63",
        "cpu_cuda_value_bit_mismatches=0",
        "cpu_cuda_validity_mismatches=0",
        "compute_sanitizer_errors=0",
        "compute_sanitizer_leaked_bytes=0",
        "feature_d2h_bytes=0",
    ] {
        if !device_receipt.lines().any(|line| line == required) {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Quant-v3 capability refuses a release receipt missing required token `{required}`"
            )));
        }
    }
    for required_sha256 in [
        "device_identity_sha256",
        "parity_log_sha256",
        "sanitizer_log_sha256",
        "racecheck_log_sha256",
        "nsys_report_sha256",
    ] {
        if !receipt_has_nonzero_sha256_v3(device_receipt, required_sha256) {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Quant-v3 capability refuses a release receipt without nonzero 64-hex field `{required_sha256}`"
            )));
        }
    }
    let closure = seal_resident_quant_migration_closure_v3();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::Quant,
        RESIDENT_QUANT_IMPLEMENTATION_ID_V4,
        closure.implementation_sha256(),
        RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4,
    )
    .map_err(Into::into)
}

/// Data-sealed, move-only authority for one exact typed temporal grid.
#[must_use = "Quant-v3 launch authority must be consumed by its native launch"]
#[derive(Debug)]
pub struct ResidentQuantLaunchAuthorityV3 {
    row_count: usize,
    timeframe_millis: u64,
    bars_per_asian_session: u64,
    bars_per_utc_day: u64,
    bars_per_trading_week: u64,
    trading_sessions_per_year: u64,
    annualization_periods_per_year: u64,
    input_identity_sha256: [u8; SHA256_BYTES],
    semantic_source_sha256: [u8; SHA256_BYTES],
    implementation_sha256: [u8; SHA256_BYTES],
}

impl ResidentQuantLaunchAuthorityV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        row_count: usize,
        timeframe_millis: u64,
        bars_per_asian_session: u64,
        bars_per_utc_day: u64,
        bars_per_trading_week: u64,
        trading_sessions_per_year: u64,
        annualization_periods_per_year: u64,
        input_identity_sha256: [u8; SHA256_BYTES],
        semantic_source_sha256: [u8; SHA256_BYTES],
        closure: SealedResidentQuantMigrationClosureV3,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let expected_annualization = trading_sessions_per_year
            .checked_mul(bars_per_utc_day)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "resident Quant-v3 annualization periods",
            ))?;
        if row_count == 0
            || timeframe_millis == 0
            || UTC_DAY_MILLIS_V3 % timeframe_millis != 0
            || ASIAN_SESSION_MILLIS_V3 % timeframe_millis != 0
            || bars_per_utc_day != UTC_DAY_MILLIS_V3 / timeframe_millis
            || bars_per_asian_session != ASIAN_SESSION_MILLIS_V3 / timeframe_millis
            || bars_per_asian_session < 12
            || bars_per_trading_week != bars_per_utc_day.saturating_mul(5)
            || trading_sessions_per_year != 252
            || annualization_periods_per_year != expected_annualization
            || input_identity_sha256 == [0; SHA256_BYTES]
            || semantic_source_sha256 == [0; SHA256_BYTES]
            || closure.implementation_sha256() == [0; SHA256_BYTES]
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Quant-v3 typed launch authority is incomplete".into(),
            ));
        }
        Ok(Self {
            row_count,
            timeframe_millis,
            bars_per_asian_session,
            bars_per_utc_day,
            bars_per_trading_week,
            trading_sessions_per_year,
            annualization_periods_per_year,
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256: closure.implementation_sha256(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentQuantRuntimeReceiptV3 {
    semantic_version: u32,
    row_count: usize,
    feature_column_count: usize,
    timeframe_millis: u64,
    bars_per_asian_session: u64,
    bars_per_utc_day: u64,
    bars_per_trading_week: u64,
    trading_sessions_per_year: u64,
    annualization_periods_per_year: u64,
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
    logical_validity_codes: [u8; 4],
    invalid_nan_bits: u64,
    input_identity_sha256: [u8; SHA256_BYTES],
    semantic_source_sha256: [u8; SHA256_BYTES],
    implementation_sha256: [u8; SHA256_BYTES],
}

impl ResidentQuantRuntimeReceiptV3 {
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }
    pub const fn row_count(&self) -> usize {
        self.row_count
    }
    pub const fn feature_column_count(&self) -> usize {
        self.feature_column_count
    }
    pub const fn timeframe_millis(&self) -> u64 {
        self.timeframe_millis
    }
    pub const fn bars_per_asian_session(&self) -> u64 {
        self.bars_per_asian_session
    }
    pub const fn bars_per_utc_day(&self) -> u64 {
        self.bars_per_utc_day
    }
    pub const fn bars_per_trading_week(&self) -> u64 {
        self.bars_per_trading_week
    }
    pub const fn trading_sessions_per_year(&self) -> u64 {
        self.trading_sessions_per_year
    }
    pub const fn annualization_periods_per_year(&self) -> u64 {
        self.annualization_periods_per_year
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
    pub const fn logical_validity_codes(&self) -> [u8; 4] {
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
struct StreamOrderedQuantBufferV3<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> StreamOrderedQuantBufferV3<T> {
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

impl<T: DeviceCopy> Deref for StreamOrderedQuantBufferV3<T> {
    type Target = DeviceBuffer<T>;
    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live Quant-v3 owner retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for StreamOrderedQuantBufferV3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live Quant-v3 owner retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for StreamOrderedQuantBufferV3<T> {
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
pub(crate) struct ResidentQuantFeatureBatchV3 {
    feature_values: StreamOrderedQuantBufferV3<f64>,
    feature_validity_u8: StreamOrderedQuantBufferV3<u8>,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    rows: usize,
    device_ordinal: u32,
    context: Arc<Context>,
    stream: Arc<Stream>,
    ready_event: ResidentProducerReadyEventV3,
    receipt: ResidentQuantRuntimeReceiptV3,
}

impl ResidentQuantFeatureBatchV3 {
    pub(crate) fn receipt(&self) -> &ResidentQuantRuntimeReceiptV3 {
        &self.receipt
    }
}

pub(crate) fn launch_resident_quant_v3(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    parent: &dyn ResidentParentDatasetSourceV3,
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    authority: ResidentQuantLaunchAuthorityV3,
) -> Result<ResidentQuantFeatureBatchV3, ResidentFeatureStoreCudaErrorV3> {
    validate_bindings_v3(&bindings)?;
    let rows = parent.rows();
    if rows == 0 || rows != authority.row_count {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Quant-v3 parent rows disagree with launch authority".into(),
        ));
    }
    validate_parent_extents_v3(parent, rows)?;

    let feature_cells = rows
        .checked_mul(RESIDENT_QUANT_COLUMN_NAMES_V3.len())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Quant-v3 feature cells",
        ))?;
    let retained_feature_device_bytes = rows
        .checked_mul(RESIDENT_QUANT_RETAINED_BYTES_PER_ROW_V3)
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Quant-v3 retained feature bytes",
        ))?;
    let computed_retained_bytes = feature_cells
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<u8>())
        .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
            "resident Quant-v3 value/validity bytes",
        ))?;
    if computed_retained_bytes != retained_feature_device_bytes {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident Quant-v3 567N byte authority drifted".into(),
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

    let feature_values = StreamOrderedQuantBufferV3::<f64>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let feature_validity_u8 = StreamOrderedQuantBufferV3::<u8>::uninitialized_async(
        feature_cells,
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;
    let native_launch = NeoResidentQuantLaunchV3 {
        abi_version: 3,
        semantic_version: RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4,
        feature_column_count: RESIDENT_QUANT_COLUMN_NAMES_V3.len() as u32,
        reserved: 0,
        row_count: rows as u64,
        timeframe_millis: authority.timeframe_millis,
        bars_per_asian_session: authority.bars_per_asian_session,
        bars_per_utc_day: authority.bars_per_utc_day,
        bars_per_trading_week: authority.bars_per_trading_week,
        trading_sessions_per_year: authority.trading_sessions_per_year,
        annualization_periods_per_year: authority.annualization_periods_per_year,
        open: parent.open().as_device_ptr().as_ptr(),
        high: parent.high().as_device_ptr().as_ptr(),
        low: parent.low().as_device_ptr().as_ptr(),
        close: parent.close().as_device_ptr().as_ptr(),
        volume: parent.volume().as_device_ptr().as_ptr(),
        timestamps: parent.timestamps().as_device_ptr().as_ptr(),
        feature_values: feature_values.as_device_ptr().as_mut_ptr(),
        feature_validity_u8: feature_validity_u8.as_device_ptr().as_mut_ptr(),
    };
    // SAFETY: retained parent and output owners prove exact extents in the
    // admitted primary context and keep every pointer live through the event.
    let status = unsafe { neoethos_resident_quant_f64_v3(&native_launch, stream.as_inner()) };
    if status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_quant_f64_v3",
            status,
        });
    }
    let ready_event =
        ResidentProducerReadyEventV3::record(context.as_ref(), stream.as_ref(), device_ordinal)?;
    let receipt = ResidentQuantRuntimeReceiptV3 {
        semantic_version: RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4,
        row_count: rows,
        feature_column_count: RESIDENT_QUANT_COLUMN_NAMES_V3.len(),
        timeframe_millis: authority.timeframe_millis,
        bars_per_asian_session: authority.bars_per_asian_session,
        bars_per_utc_day: authority.bars_per_utc_day,
        bars_per_trading_week: authority.bars_per_trading_week,
        trading_sessions_per_year: authority.trading_sessions_per_year,
        annualization_periods_per_year: authority.annualization_periods_per_year,
        retained_feature_device_bytes,
        additional_retained_device_bytes: 0,
        scratch_device_bytes: 0,
        pointer_table_device_bytes: RESIDENT_QUANT_POINTER_TABLE_DEVICE_BYTES_V3,
        isolated_pointer_schema_metadata_bytes: RESIDENT_QUANT_ISOLATED_POINTER_SCHEMA_BYTES_V3,
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        producer_ready_event_count: 1,
        producer_ready_event_synchronize_count: 0,
        native_launch_count: 1,
        host_synchronize_count: 0,
        logical_validity_schema: RESIDENT_QUANT_LOGICAL_VALIDITY_SCHEMA_V3,
        logical_validity_codes: [0, 1, 5, 8],
        invalid_nan_bits: RESIDENT_QUANT_INVALID_NAN_BITS_V3,
        input_identity_sha256: authority.input_identity_sha256,
        semantic_source_sha256: authority.semantic_source_sha256,
        implementation_sha256: authority.implementation_sha256,
    };
    Ok(ResidentQuantFeatureBatchV3 {
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
    if bindings.len() != RESIDENT_QUANT_COLUMN_NAMES_V3.len() {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
            "resident Quant-v3 requires exactly {} bindings, received {}",
            RESIDENT_QUANT_COLUMN_NAMES_V3.len(),
            bindings.len()
        )));
    }
    for (binding, expected_name) in bindings.iter().zip(RESIDENT_QUANT_COLUMN_NAMES_V3) {
        if binding.feature_name != expected_name {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Quant-v3 schema expected `{expected_name}`, received `{}`",
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
        ("volume", parent.volume().len()),
        ("timestamps", parent.timestamps().len()),
    ] {
        if len != rows {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
                "resident Quant-v3 parent {name} extent {len} differs from {rows} rows"
            )));
        }
    }
    Ok(())
}

unsafe impl ResidentF64FeatureBatchV3 for ResidentQuantFeatureBatchV3 {
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
    fn verified_release_receipt_mints_exact_quant_capability_v3() {
        let device_receipt =
            include_str!("../tests/fixtures/resident_quant_v3_device_parity.release.txt");
        let observed_receipt_sha256: [u8; SHA256_BYTES] =
            Sha256::digest(device_receipt.as_bytes()).into();
        assert_eq!(
            observed_receipt_sha256,
            RESIDENT_QUANT_VERIFIED_RELEASE_RECEIPT_SHA256_V3
        );

        let expected_implementation_sha256 =
            seal_resident_quant_migration_closure_v3().implementation_sha256();
        let capability = resident_quant_capability_v3()
            .expect("the exact verified required-card receipt must mint Quant-v3 capability");
        assert_eq!(capability.producer(), ResidentFeatureProducerV3::Quant);
        assert_eq!(
            capability.implementation_id(),
            RESIDENT_QUANT_IMPLEMENTATION_ID_V4
        );
        assert_eq!(
            capability.implementation_sha256(),
            expected_implementation_sha256
        );
        assert_eq!(
            capability.exact_math_authority(),
            RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4
        );
    }
}
