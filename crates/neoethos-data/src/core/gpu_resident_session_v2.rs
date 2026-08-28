//! Data-owned resident Session semantic-v2 producer preflight.
//!
//! Session-v2 is intentionally an atomic, sequential 23-column state machine.
//! Its historical dual-clock behavior is part of the identity: the value lane
//! uses first-sixteen-nonzero magnitude inference while validity consumes the
//! original canonical millisecond timestamp. Correcting that quirk requires a
//! semantic-v3 migration rather than a resident optimization.

use anyhow::{Context as _, Result, ensure};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentFeatureStageV3, ResidentProducerCapabilityV3,
};
use neoethos_gpu_cuda::resident_feature_store_v3::{
    ResidentFeatureColumnBindingV3, ResidentFeatureStoreAssemblerV3,
    ResidentFeatureStoreCudaErrorV3,
};
use neoethos_gpu_cuda::resident_session_v2::{
    RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2 as NATIVE_SESSION_EXACT_MATH_AUTHORITY_V2,
    RESIDENT_SESSION_IMPLEMENTATION_ID_V2 as NATIVE_SESSION_IMPLEMENTATION_ID_V2,
    ResidentSessionLaunchAuthorityV2, ResidentSessionRuntimeReceiptV2,
    resident_session_capability_v2, seal_resident_session_source_closure_v2,
};
use sha2::{Digest, Sha256};

use crate::Ohlcv;

use super::gpu_resident_feature_recipe_v4::{
    ResidentCanonicalParameterV4, ResidentCanonicalParameterValueV4, ResidentProducerBatchDraftV4,
    ResidentProducerDraftV4, ResidentRouteDraftV4,
};
use super::timestamps::{
    TimestampUnit, infer_timestamp_unit, validate_canonical_millisecond_timestamps,
};

pub(crate) const RESIDENT_SESSION_SEMANTIC_VERSION_V2: u32 = 2;
pub(crate) const RESIDENT_SESSION_IMPLEMENTATION_ID_V2: &str = NATIVE_SESSION_IMPLEMENTATION_ID_V2;
pub(crate) const RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2: &str =
    NATIVE_SESSION_EXACT_MATH_AUTHORITY_V2;
const RESIDENT_SESSION_ROUTE_DOMAIN_V2: &str = "neoethos.data.resident-session-route.semantic-v2";
const RESIDENT_SESSION_INDICATOR_ID_V2: &str = "neoethos_session_semantic_v2";

pub(crate) const RESIDENT_SESSION_RETAINED_BYTES_PER_ROW_V2: u64 = 207;
pub(crate) const RESIDENT_SESSION_POINTER_TABLE_DEVICE_BYTES_V2: u64 = 736;
pub(crate) const RESIDENT_SESSION_ISOLATED_POINTER_SCHEMA_BYTES_V2: u64 = 1_377;
pub(crate) const RESIDENT_SESSION_INVALID_NAN_BITS_V2: u64 = 0x7ff8_0000_0000_0000;

pub(crate) const RESIDENT_SESSION_COLUMN_NAMES_V2: [&str; 23] = [
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

#[derive(Debug)]
pub(crate) struct ResidentSessionAllocationReceiptV2 {
    row_count: u64,
    feature_column_count: u64,
    retained_feature_device_bytes: u64,
    additional_retained_device_bytes: u64,
    scratch_device_bytes: u64,
    pointer_table_device_bytes: u64,
    isolated_pointer_schema_metadata_bytes: u64,
    parent_input_h2d_bytes: u64,
    feature_value_d2h_bytes: u64,
    native_launch_count: u64,
    producer_ready_event_count: u64,
    producer_ready_event_synchronize_count: u64,
    host_synchronize_count: u64,
    logical_validity_codes: [u8; 3],
    invalid_nan_bits: u64,
}

#[derive(Debug)]
pub(crate) struct ResidentSessionRuntimeAdmissionV2 {
    input_identity_sha256: [u8; 32],
    semantic_source_sha256: [u8; 32],
    implementation_sha256: [u8; 32],
    allocation: ResidentSessionAllocationReceiptV2,
}

/// Owner-derived Session allocation evidence used by HTF recipe preflight
/// before any run-device carrier exists. Private fields prevent the shared
/// materializer from substituting caller-provided byte counts.
#[derive(Debug)]
pub(crate) struct ResidentSessionHigherTimeframeBatchMemoryV2 {
    row_count: u64,
    feature_column_count: u64,
    retained_feature_device_bytes: u64,
    additional_retained_device_bytes: u64,
    scratch_device_bytes: u64,
}

impl ResidentSessionHigherTimeframeBatchMemoryV2 {
    pub(crate) const fn row_count(&self) -> u64 {
        self.row_count
    }

    pub(crate) const fn feature_column_count(&self) -> u64 {
        self.feature_column_count
    }

    pub(crate) const fn retained_feature_device_bytes(&self) -> u64 {
        self.retained_feature_device_bytes
    }

    pub(crate) const fn additional_retained_device_bytes(&self) -> u64 {
        self.additional_retained_device_bytes
    }

    pub(crate) const fn scratch_device_bytes(&self) -> u64 {
        self.scratch_device_bytes
    }
}

impl ResidentSessionRuntimeAdmissionV2 {
    fn validate_native_receipt(&self, receipt: &ResidentSessionRuntimeReceiptV2) -> Result<()> {
        ensure!(
            receipt.semantic_version() == RESIDENT_SESSION_SEMANTIC_VERSION_V2,
            "resident Session-v2 native semantic version drifted"
        );
        ensure!(
            u64::try_from(receipt.row_count()).context("Session native row receipt overflow")?
                == self.allocation.row_count
                && u64::try_from(receipt.feature_column_count())
                    .context("Session native column receipt overflow")?
                    == self.allocation.feature_column_count,
            "resident Session-v2 native shape drifted"
        );
        ensure!(
            u64::try_from(receipt.retained_feature_device_bytes())
                .context("Session native retained-byte receipt overflow")?
                == self.allocation.retained_feature_device_bytes
                && u64::try_from(receipt.additional_retained_device_bytes())
                    .context("Session native additional-byte receipt overflow")?
                    == self.allocation.additional_retained_device_bytes
                && u64::try_from(receipt.scratch_device_bytes())
                    .context("Session native scratch receipt overflow")?
                    == self.allocation.scratch_device_bytes
                && u64::try_from(receipt.pointer_table_device_bytes())
                    .context("Session native pointer-table receipt overflow")?
                    == self.allocation.pointer_table_device_bytes
                && u64::try_from(receipt.isolated_pointer_schema_metadata_bytes())
                    .context("Session native schema receipt overflow")?
                    == self.allocation.isolated_pointer_schema_metadata_bytes,
            "resident Session-v2 native allocation receipt drifted"
        );
        ensure!(
            u64::try_from(receipt.parent_input_h2d_bytes())
                .context("Session native parent-H2D receipt overflow")?
                == self.allocation.parent_input_h2d_bytes
                && u64::try_from(receipt.feature_value_d2h_bytes())
                    .context("Session native feature-D2H receipt overflow")?
                    == self.allocation.feature_value_d2h_bytes
                && u64::try_from(receipt.native_launch_count())
                    .context("Session native launch receipt overflow")?
                    == self.allocation.native_launch_count
                && u64::try_from(receipt.producer_ready_event_count())
                    .context("Session native event receipt overflow")?
                    == self.allocation.producer_ready_event_count
                && u64::try_from(receipt.producer_ready_event_synchronize_count())
                    .context("Session native event-sync receipt overflow")?
                    == self.allocation.producer_ready_event_synchronize_count
                && u64::try_from(receipt.host_synchronize_count())
                    .context("Session native host-sync receipt overflow")?
                    == self.allocation.host_synchronize_count,
            "resident Session-v2 native transfer/launch/event receipt drifted"
        );
        ensure!(
            receipt.logical_validity_codes() == self.allocation.logical_validity_codes
                && receipt.invalid_nan_bits() == self.allocation.invalid_nan_bits
                && receipt.logical_validity_schema()
                    == "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3",
            "resident Session-v2 native validity authority drifted"
        );
        ensure!(
            receipt.input_identity_sha256() == self.input_identity_sha256
                && receipt.semantic_source_sha256() == self.semantic_source_sha256
                && receipt.implementation_sha256() == self.implementation_sha256,
            "resident Session-v2 native source/input identity drifted"
        );
        Ok(())
    }
}

#[must_use = "resident Session-v2 preflight must move into schema and runtime assembly"]
#[derive(Debug)]
pub(crate) struct PreparedResidentSessionProducerV2 {
    draft: ResidentProducerDraftV4,
    runtime_admission: ResidentSessionRuntimeAdmissionV2,
}

#[must_use = "current native Session-v2 preflight must launch or move into recipe assembly"]
#[derive(Debug)]
pub(crate) struct PreparedCurrentNativeResidentSessionProducerV2 {
    prepared: PreparedResidentSessionProducerV2,
    launch_authority: ResidentSessionLaunchAuthorityV2,
}

#[must_use = "prepared Session-v2 runtime must append to the admitted resident assembler"]
#[derive(Debug)]
pub(crate) struct PreparedResidentSessionRuntimeV2 {
    runtime_admission: ResidentSessionRuntimeAdmissionV2,
    launch_authority: ResidentSessionLaunchAuthorityV2,
}

#[must_use = "Session-v2 HTF parent authority and admission must both be consumed"]
#[derive(Debug)]
pub(crate) struct PendingResidentSessionHigherTimeframeParentV2 {
    runtime_admission: Option<ResidentSessionRuntimeAdmissionV2>,
    launch_authority: Option<ResidentSessionLaunchAuthorityV2>,
}

impl PreparedCurrentNativeResidentSessionProducerV2 {
    pub(crate) fn into_parts(
        self,
    ) -> (
        ResidentProducerDraftV4,
        ResidentSessionRuntimeAdmissionV2,
        ResidentSessionLaunchAuthorityV2,
    ) {
        let (draft, runtime_admission) = self.prepared.into_parts();
        (draft, runtime_admission, self.launch_authority)
    }

    pub(crate) fn into_recipe_parts(
        self,
    ) -> (ResidentProducerDraftV4, PreparedResidentSessionRuntimeV2) {
        let (draft, runtime_admission, launch_authority) = self.into_parts();
        (
            draft,
            PreparedResidentSessionRuntimeV2 {
                runtime_admission,
                launch_authority,
            },
        )
    }
}

impl PreparedResidentSessionRuntimeV2 {
    pub(crate) fn into_higher_timeframe_parent_v2(
        self,
    ) -> PendingResidentSessionHigherTimeframeParentV2 {
        PendingResidentSessionHigherTimeframeParentV2 {
            runtime_admission: Some(self.runtime_admission),
            launch_authority: Some(self.launch_authority),
        }
    }

    pub(crate) fn append_to(
        self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> std::result::Result<
        (
            ResidentSessionRuntimeAdmissionV2,
            ResidentSessionRuntimeReceiptV2,
        ),
        ResidentFeatureStoreCudaErrorV3,
    > {
        let Self {
            runtime_admission,
            launch_authority,
        } = self;
        let receipt = assembler.append_resident_session_v2(bindings, launch_authority)?;
        runtime_admission
            .validate_native_receipt(&receipt)
            .map_err(|error| ResidentFeatureStoreCudaErrorV3::InvalidInput(error.to_string()))?;
        Ok((runtime_admission, receipt))
    }
}

impl PendingResidentSessionHigherTimeframeParentV2 {
    pub(crate) fn higher_timeframe_batch_memory_v2(
        &self,
    ) -> std::result::Result<
        ResidentSessionHigherTimeframeBatchMemoryV2,
        ResidentFeatureStoreCudaErrorV3,
    > {
        let admission = self.runtime_admission.as_ref().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Session-v2 HTF admission was already consumed".into(),
            )
        })?;
        Ok(ResidentSessionHigherTimeframeBatchMemoryV2 {
            row_count: admission.allocation.row_count,
            feature_column_count: admission.allocation.feature_column_count,
            retained_feature_device_bytes: admission.allocation.retained_feature_device_bytes,
            additional_retained_device_bytes: admission.allocation.additional_retained_device_bytes,
            scratch_device_bytes: admission.allocation.scratch_device_bytes,
        })
    }

    pub(crate) fn take_launch_authority_v2(
        &mut self,
    ) -> std::result::Result<ResidentSessionLaunchAuthorityV2, ResidentFeatureStoreCudaErrorV3>
    {
        self.launch_authority.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Session-v2 HTF launch authority was already consumed".into(),
            )
        })
    }

    pub(crate) fn validate_captured_parent_receipt_v2(
        mut self,
        receipt: &ResidentSessionRuntimeReceiptV2,
    ) -> std::result::Result<(), ResidentFeatureStoreCudaErrorV3> {
        if self.launch_authority.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Session-v2 HTF receipt cannot be admitted before launch authority moves"
                    .into(),
            ));
        }
        let runtime_admission = self.runtime_admission.take().ok_or_else(|| {
            ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident Session-v2 HTF admission was already consumed".into(),
            )
        })?;
        runtime_admission
            .validate_native_receipt(receipt)
            .map_err(|error| ResidentFeatureStoreCudaErrorV3::InvalidInput(error.to_string()))?;
        Ok(())
    }
}

pub(crate) fn preflight_current_native_resident_session_v2(
    ohlcv: &Ohlcv,
) -> Result<PreparedCurrentNativeResidentSessionProducerV2> {
    let source_closure = seal_resident_session_source_closure_v2();
    let native_capability = resident_session_capability_v2()?;
    ensure!(
        native_capability.implementation_sha256() == source_closure.implementation_sha256(),
        "resident Session-v2 native capability/source closure drifted"
    );
    let prepared = preflight_resident_session_v2(ohlcv, native_capability)?;
    let launch_authority = ResidentSessionLaunchAuthorityV2::seal(
        ohlcv.len(),
        prepared.runtime_admission.input_identity_sha256,
        prepared.runtime_admission.semantic_source_sha256,
        source_closure,
    )?;
    Ok(PreparedCurrentNativeResidentSessionProducerV2 {
        prepared,
        launch_authority,
    })
}

impl PreparedResidentSessionProducerV2 {
    pub(crate) fn into_parts(self) -> (ResidentProducerDraftV4, ResidentSessionRuntimeAdmissionV2) {
        (self.draft, self.runtime_admission)
    }
}

fn preflight_resident_session_v2(
    ohlcv: &Ohlcv,
    native_capability: ResidentProducerCapabilityV3,
) -> Result<PreparedResidentSessionProducerV2> {
    ensure!(
        native_capability.producer() == ResidentFeatureProducerV3::Session,
        "resident Session-v2 native capability has the wrong producer"
    );
    ensure!(
        native_capability.implementation_id() == RESIDENT_SESSION_IMPLEMENTATION_ID_V2,
        "resident Session-v2 native capability has the wrong implementation id"
    );
    ensure!(
        native_capability.exact_math_authority() == RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2,
        "resident Session-v2 native capability has the wrong exact-math authority"
    );
    let (timestamps, volume) = validate_session_parent_v2(ohlcv)?;
    if infer_timestamp_unit(timestamps) != Some(TimestampUnit::Milliseconds) {
        anyhow::bail!(
            "resident Session-v2 canonical millisecond input disagrees with its retained value-clock inference"
        );
    }
    let semantic_source_sha256 = session_semantic_source_sha256_v2();
    let input_identity_sha256 =
        session_input_identity_v2(ohlcv, timestamps, volume, semantic_source_sha256);
    ensure!(
        input_identity_sha256 != [0; 32],
        "resident Session-v2 input identity is zero"
    );
    let implementation_sha256 = native_capability.implementation_sha256();

    let routes = RESIDENT_SESSION_COLUMN_NAMES_V2
        .iter()
        .copied()
        .map(|name| session_route_v2(name, input_identity_sha256, semantic_source_sha256))
        .collect::<Result<Vec<_>>>()?;
    let draft = ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::Session,
        RESIDENT_SESSION_SEMANTIC_VERSION_V2,
        routes,
        vec![ResidentProducerBatchDraftV4::from_owner_preflight(
            0, 23, 0, 0,
        )],
        native_capability,
    )?;
    let rows = u64::try_from(ohlcv.len()).context("resident Session-v2 row count overflow")?;
    let retained_feature_device_bytes = rows
        .checked_mul(RESIDENT_SESSION_RETAINED_BYTES_PER_ROW_V2)
        .context("resident Session-v2 retained output bytes overflow")?;
    Ok(PreparedResidentSessionProducerV2 {
        draft,
        runtime_admission: ResidentSessionRuntimeAdmissionV2 {
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256,
            allocation: ResidentSessionAllocationReceiptV2 {
                row_count: rows,
                feature_column_count: 23,
                retained_feature_device_bytes,
                additional_retained_device_bytes: 0,
                scratch_device_bytes: 0,
                pointer_table_device_bytes: RESIDENT_SESSION_POINTER_TABLE_DEVICE_BYTES_V2,
                isolated_pointer_schema_metadata_bytes:
                    RESIDENT_SESSION_ISOLATED_POINTER_SCHEMA_BYTES_V2,
                parent_input_h2d_bytes: 0,
                feature_value_d2h_bytes: 0,
                native_launch_count: 1,
                producer_ready_event_count: 1,
                producer_ready_event_synchronize_count: 0,
                host_synchronize_count: 0,
                logical_validity_codes: [0, 1, 5],
                invalid_nan_bits: RESIDENT_SESSION_INVALID_NAN_BITS_V2,
            },
        },
    })
}

fn validate_session_parent_v2(ohlcv: &Ohlcv) -> Result<(&[i64], &[f64])> {
    let rows = ohlcv.len();
    ensure!(rows > 0, "resident Session-v2 requires nonempty OHLCV");
    ensure!(
        ohlcv.open.len() == rows && ohlcv.high.len() == rows && ohlcv.low.len() == rows,
        "resident Session-v2 OHLC shape mismatch"
    );
    let timestamps = ohlcv
        .timestamp
        .as_deref()
        .context("resident Session-v2 requires canonical millisecond timestamps")?;
    ensure!(
        timestamps.len() == rows,
        "resident Session-v2 timestamp shape mismatch"
    );
    validate_canonical_millisecond_timestamps(timestamps)?;
    let volume = ohlcv
        .volume
        .as_deref()
        .context("resident Session-v2 requires present volume")?;
    ensure!(
        volume.len() == rows,
        "resident Session-v2 volume shape mismatch"
    );
    for row in 0..rows {
        let (open, high, low, close, volume) = (
            ohlcv.open[row],
            ohlcv.high[row],
            ohlcv.low[row],
            ohlcv.close[row],
            volume[row],
        );
        ensure!(
            open.is_finite()
                && high.is_finite()
                && low.is_finite()
                && close.is_finite()
                && volume.is_finite(),
            "resident Session-v2 row {row} contains non-finite OHLCV"
        );
        ensure!(
            open > 0.0
                && high > 0.0
                && low > 0.0
                && close > 0.0
                && volume >= 0.0
                && low <= open.min(close)
                && high >= open.max(close),
            "resident Session-v2 row {row} violates canonical OHLCV bounds"
        );
    }
    Ok((timestamps, volume))
}

fn session_route_v2(
    name: &str,
    input_identity_sha256: [u8; 32],
    semantic_source_sha256: [u8; 32],
) -> Result<ResidentRouteDraftV4> {
    let parameters = vec![
        parameter("input_identity_sha256", ResidentCanonicalParameterValueV4::Hash(input_identity_sha256))?,
        parameter("semantic_source_sha256", ResidentCanonicalParameterValueV4::Hash(semantic_source_sha256))?,
        parameter("execution_order", ResidentCanonicalParameterValueV4::Text("one-thread-ascending-row-state-machine".to_owned()))?,
        parameter("dual_clock_policy", ResidentCanonicalParameterValueV4::Text("value:first-16-nonzero-magnitude-vote-75pct;validity:original-canonical-ms".to_owned()))?,
        parameter("timestamp_magnitude_boundaries", ResidentCanonicalParameterValueV4::Text("seconds<10_000_000_000;milliseconds<10_000_000_000_000;microseconds<10_000_000_000_000_000;nanoseconds>=10_000_000_000_000_000".to_owned()))?,
        parameter("utc_windows", ResidentCanonicalParameterValueV4::Text("asian=00:00-08:00;london=07:00-16:00;new-york=12:00-21:00;overlap=12:00-16:00".to_owned()))?,
        parameter("atr_policy", ResidentCanonicalParameterValueV4::Text("cumulative-true-range-from-row-1;row-0=high-low".to_owned()))?,
        parameter("output_session_tag", ResidentCanonicalParameterValueV4::Text(session_output_tag_v2(name).to_owned()))?,
        parameter("canonical_invalid_nan_bits", ResidentCanonicalParameterValueV4::U64(RESIDENT_SESSION_INVALID_NAN_BITS_V2))?,
        parameter("logical_validity_codes", ResidentCanonicalParameterValueV4::Text("valid=0;warmup=1;zero-denominator=5".to_owned()))?,
    ];
    ResidentRouteDraftV4::from_typed_parts(
        name,
        Some(RESIDENT_SESSION_INDICATOR_ID_V2),
        Some(name),
        ResidentFeatureStageV3::Derived,
        None,
        parameters,
        RESIDENT_SESSION_ROUTE_DOMAIN_V2,
    )
    .map_err(Into::into)
}

fn session_output_tag_v2(name: &str) -> &'static str {
    if name.starts_with("session_london_") {
        "london"
    } else if name.starts_with("session_ny_") {
        "new-york"
    } else if name.starts_with("session_asian_") {
        "asian"
    } else if name.starts_with("daily_") {
        "utc-day"
    } else {
        "cross-session"
    }
}

fn parameter(
    name: &'static str,
    value: ResidentCanonicalParameterValueV4,
) -> Result<ResidentCanonicalParameterV4> {
    ResidentCanonicalParameterV4::from_typed_value(name, value).map_err(Into::into)
}

fn session_semantic_source_sha256_v2() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-session-semantic-source.v2\0");
    hash.update(include_bytes!("session_features.rs"));
    hash.update(include_bytes!("timestamps.rs"));
    hash.update(include_bytes!("features.rs"));
    hash.update(include_bytes!(
        "../../../neoethos-gpu-cuda/native/resident_session_v2_abi.cuh"
    ));
    hash.update(include_bytes!(
        "../../../neoethos-gpu-cuda/native/resident_session_v2.cu"
    ));
    hash.update(include_bytes!(
        "../../../neoethos-gpu-cuda/src/resident_session_v2.rs"
    ));
    hash.update(RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2.as_bytes());
    hash.finalize().into()
}

fn session_input_identity_v2(
    ohlcv: &Ohlcv,
    timestamps: &[i64],
    volume: &[f64],
    semantic_source_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-session-input.semantic-v2\0");
    hash.update(RESIDENT_SESSION_SEMANTIC_VERSION_V2.to_le_bytes());
    hash.update(semantic_source_sha256);
    hash.update((ohlcv.len() as u64).to_le_bytes());
    for timestamp in timestamps {
        hash.update(timestamp.to_le_bytes());
    }
    for lane in [&ohlcv.open, &ohlcv.high, &ohlcv.low, &ohlcv.close] {
        for value in lane {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    for value in volume {
        hash.update(value.to_bits().to_le_bytes());
    }
    hash.finalize().into()
}
