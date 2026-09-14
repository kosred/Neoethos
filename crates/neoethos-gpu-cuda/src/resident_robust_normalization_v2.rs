//! Exact post-pack resident Search normalization policy-v3.
//!
//! This module owns deterministic planning and the native launch. The feature
//! store owns the final bar-major values/u4 validity, scratch, fit metadata,
//! control word, primary context, stream and completion event. No feature
//! value or validity payload is copied to host.

use std::ops::Range;

use cust::memory::{DeviceBuffer, GpuBuffer};
use cust::stream::Stream;
use cust::sys::CUstream;
use neoethos_gpu_contracts::normalization_v3::{
    SearchNormalizationColumnModeV3, resident_normalization_fit_metadata_sha256_v3,
};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityV3,
};
use sha2::{Digest, Sha256};

use crate::resident_feature_store_v3::ResidentFeatureStoreCudaErrorV3;

const SHA256_BYTES: usize = 32;
pub const RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2: u32 = 3;
pub const RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2: usize = 64;
pub const RESIDENT_ROBUST_NORMALIZATION_FIT_WORDS_V2: usize = 6;
pub const RESIDENT_ROBUST_NORMALIZATION_FIT_BYTES_V2: usize = 48;
pub const VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2: usize = 4;
const CANONICAL_DISCOVERY_NORMALIZATION_MIN_TRAINING_ROWS_V2: usize = 64;
const CANONICAL_DISCOVERY_OOS_HOLDOUT_FRACTION_V2: f64 = 0.2;
const DISABLED_FIT_DIGEST_DOMAIN_V2: &[u8] =
    b"neoethos.resident-robust-normalization.disabled-fit.semantic-v2\0";
const _: () = assert!(
    RESIDENT_ROBUST_NORMALIZATION_FIT_BYTES_V2
        == RESIDENT_ROBUST_NORMALIZATION_FIT_WORDS_V2 * std::mem::size_of::<u64>()
);
pub const RESIDENT_ROBUST_NORMALIZATION_EXACT_AUTHORITY_V2: &str = "cpu-search-normalization-policy-v3;explicit-column-modes;all-row-gate-domain-validation;zero-anchored-identity-gates;training-valid-finite-only;rust-f64-total-cmp;median-mad-1p4826;population-std-fallback-sorted-order;scale-floor-32eps;clip-plus-minus-10;typed-validity-u4-aligned-atomic-u32;canonical-nan;post-pack-pre-sha;device-fit-sha256;fmad=false;ftz=false;prec-div=true;prec-sqrt=true";

unsafe extern "C" {
    fn neoethos_resident_robust_normalize_bar_major_f64_u4_v3(
        bar_major_values: *mut f64,
        bar_major_validity_u4: *mut u8,
        packed_validity_allocated_bytes: usize,
        rows: usize,
        columns: usize,
        training_start: usize,
        training_end: usize,
        padded_training_rows: usize,
        sort_scratch_bits: *mut u64,
        sort_scratch_slots: usize,
        fit_metadata_words: *mut u64,
        fit_metadata_word_count: usize,
        host_column_modes: *const u8,
        column_mode_count: usize,
        control_error: *mut u32,
        stream: CUstream,
    ) -> i32;
}

pub fn resident_robust_normalization_disabled_fit_sha256_v2() -> [u8; SHA256_BYTES] {
    Sha256::digest(DISABLED_FIT_DIGEST_DOMAIN_V2).into()
}

pub fn resident_robust_normalization_capability_v2()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-search-normalization.f64.policy-v3");
    implementation.update(include_bytes!("resident_robust_normalization_v2.rs"));
    implementation.update(include_bytes!(
        "../native/resident_robust_normalization_v2.cu"
    ));
    implementation.update(include_bytes!(
        "../../neoethos-data/src/core/normalization.rs"
    ));
    implementation.update(include_bytes!(
        "../../neoethos-data/tests/robust_normalization_semantic_v2_oracle.rs"
    ));
    implementation.update(include_bytes!(
        "../../neoethos-gpu-contracts/src/normalization_v3.rs"
    ));
    implementation.update(RESIDENT_ROBUST_NORMALIZATION_EXACT_AUTHORITY_V2.as_bytes());
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::RobustNormalization,
        "neoethos.gpu-cuda.resident-search-normalization.f64.policy-v3",
        implementation_sha256,
        RESIDENT_ROBUST_NORMALIZATION_EXACT_AUTHORITY_V2,
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentRobustNormalizationPlanV2 {
    enabled: bool,
    rows: usize,
    columns: usize,
    training_rows: Range<usize>,
    padded_training_rows: usize,
    packed_validity_logical_bytes: usize,
    packed_validity_allocated_bytes: usize,
    batch_count: usize,
    normalization_scratch_slots: usize,
    normalization_scratch_bytes: usize,
    fit_metadata_words: usize,
    fit_metadata_bytes: usize,
    native_launch_count: usize,
    column_modes: Option<Vec<SearchNormalizationColumnModeV3>>,
}

impl ResidentRobustNormalizationPlanV2 {
    /// Geometry only. Enabled execution additionally requires an explicit
    /// policy binding from the actual ordered schema, never inferred defaults.
    pub fn preflight(
        rows: usize,
        columns: usize,
        training_rows: Range<usize>,
        enabled: bool,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        let canonical_training_end =
            ((rows as f64) * (1.0 - CANONICAL_DISCOVERY_OOS_HOLDOUT_FRACTION_V2)).floor() as usize;
        if rows == 0
            || columns == 0
            || training_rows.start >= training_rows.end
            || training_rows.end > rows
            || training_rows != (0..canonical_training_end)
            || canonical_training_end < CANONICAL_DISCOVERY_NORMALIZATION_MIN_TRAINING_ROWS_V2
            || canonical_training_end >= rows
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident robust normalization requires nonempty rows/columns and the exact canonical 80/20 training range with at least 64 rows and a nonempty holdout"
                    .into(),
            ));
        }
        let cells = rows.checked_mul(columns).ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization feature cells",
            ),
        )?;
        let packed_validity_logical_bytes = cells.div_ceil(2);
        let packed_validity_allocated_bytes = packed_validity_logical_bytes
            .checked_add(VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2 - 1)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization packed-validity alignment",
            ))?
            / VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2
            * VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2;
        if !enabled {
            return Ok(Self {
                enabled,
                rows,
                columns,
                training_rows,
                padded_training_rows: 0,
                packed_validity_logical_bytes,
                packed_validity_allocated_bytes,
                batch_count: 0,
                normalization_scratch_slots: 0,
                normalization_scratch_bytes: 0,
                fit_metadata_words: 0,
                fit_metadata_bytes: 0,
                native_launch_count: 0,
                column_modes: None,
            });
        }
        let training_len = training_rows.len();
        let padded_training_rows = training_len.checked_next_power_of_two().ok_or(
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization padded training rows",
            ),
        )?;
        let batch_count = columns
            .checked_add(RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2 - 1)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization batch count",
            ))?
            / RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2;
        let max_batch_columns = columns.min(RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2);
        let normalization_scratch_slots = max_batch_columns
            .checked_mul(padded_training_rows)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization sort scratch slots",
            ))?;
        let normalization_scratch_bytes = normalization_scratch_slots
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization sort scratch bytes",
            ))?;
        let fit_metadata_words = columns
            .checked_mul(RESIDENT_ROBUST_NORMALIZATION_FIT_WORDS_V2)
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization fit metadata words",
            ))?;
        let fit_metadata_bytes = fit_metadata_words
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization fit metadata bytes",
            ))?;
        let log2_padded = usize::try_from(padded_training_rows.trailing_zeros()).map_err(|_| {
            ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization bitonic depth",
            )
        })?;
        let launches_per_batch = log2_padded
            .checked_mul(log2_padded.checked_add(1).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "robust-normalization bitonic launch depth",
                ),
            )?)
            .and_then(|launches| launches.checked_add(5))
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization launches per batch",
            ))?;
        let native_launch_count = batch_count
            .checked_mul(launches_per_batch)
            .and_then(|launches| launches.checked_add(1))
            .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                "robust-normalization native launch count",
            ))?;
        Ok(Self {
            enabled,
            rows,
            columns,
            training_rows,
            padded_training_rows,
            packed_validity_logical_bytes,
            packed_validity_allocated_bytes,
            batch_count,
            normalization_scratch_slots,
            normalization_scratch_bytes,
            fit_metadata_words,
            fit_metadata_bytes,
            native_launch_count,
            column_modes: None,
        })
    }

    pub fn with_column_modes_v3(
        mut self,
        modes: Vec<SearchNormalizationColumnModeV3>,
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        if modes.len() != self.columns || self.column_modes.is_some() {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "normalization modes must bind the exact ordered columns once".into(),
            ));
        }
        if self.enabled {
            let depth = self.padded_training_rows.trailing_zeros() as usize;
            let sort_launches = depth.checked_mul(depth + 1).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow("policy sort launches"),
            )?;
            self.native_launch_count = 1; // Final six-word metadata digest.
            for batch in modes.chunks(RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2) {
                let robust = batch.contains(&SearchNormalizationColumnModeV3::Robust);
                let gates = batch
                    .iter()
                    .any(|mode| *mode != SearchNormalizationColumnModeV3::Robust);
                self.native_launch_count = self
                    .native_launch_count
                    .checked_add(5 + usize::from(gates) + if robust { sort_launches } else { 0 })
                    .ok_or(ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                        "policy launches",
                    ))?;
            }
        }
        self.column_modes = Some(modes);
        Ok(self)
    }

    pub fn column_modes_v3(&self) -> Option<&[SearchNormalizationColumnModeV3]> {
        self.column_modes.as_deref()
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn columns(&self) -> usize {
        self.columns
    }
    pub fn training_rows(&self) -> Range<usize> {
        self.training_rows.clone()
    }
    pub const fn padded_training_rows(&self) -> usize {
        self.padded_training_rows
    }
    pub const fn packed_validity_logical_bytes(&self) -> usize {
        self.packed_validity_logical_bytes
    }
    pub const fn packed_validity_allocated_bytes(&self) -> usize {
        self.packed_validity_allocated_bytes
    }
    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }
    pub const fn normalization_scratch_slots(&self) -> usize {
        self.normalization_scratch_slots
    }
    pub const fn normalization_scratch_bytes(&self) -> usize {
        self.normalization_scratch_bytes
    }
    pub const fn fit_metadata_words(&self) -> usize {
        self.fit_metadata_words
    }
    pub const fn fit_metadata_bytes(&self) -> usize {
        self.fit_metadata_bytes
    }
    pub const fn native_launch_count(&self) -> usize {
        self.native_launch_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentRobustNormalizationRuntimeReceiptV2 {
    semantic_version: u32,
    enabled: bool,
    row_count: usize,
    feature_column_count: usize,
    training_rows: Range<usize>,
    padded_training_rows: usize,
    packed_validity_logical_bytes: usize,
    packed_validity_allocated_bytes: usize,
    normalization_scratch_bytes: usize,
    fit_metadata_bytes: usize,
    fit_metadata_sha256: [u8; SHA256_BYTES],
    batch_count: usize,
    native_launch_count: usize,
    producer_ready_event_count: usize,
    producer_ready_event_synchronize_count: usize,
    control_error_device_bytes: usize,
    control_error_readback_count: usize,
    control_error_d2h_bytes: usize,
    fit_digest_readback_count: usize,
    fit_digest_d2h_bytes: usize,
    fit_words_readback_count: usize,
    fit_words_d2h_bytes: usize,
    fit_metadata_words_v3: Vec<u64>,
    parent_input_h2d_bytes: usize,
    feature_value_d2h_bytes: usize,
    admission_identity_sha256: [u8; SHA256_BYTES],
    primary_context_process_token: [u8; SHA256_BYTES],
    producer_stream_process_token: [u8; SHA256_BYTES],
    ready_event_process_token: [u8; SHA256_BYTES],
}

impl ResidentRobustNormalizationRuntimeReceiptV2 {
    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }
    pub const fn enabled(&self) -> bool {
        self.enabled
    }
    pub const fn row_count(&self) -> usize {
        self.row_count
    }
    pub const fn feature_column_count(&self) -> usize {
        self.feature_column_count
    }
    pub fn training_rows(&self) -> Range<usize> {
        self.training_rows.clone()
    }
    pub const fn padded_training_rows(&self) -> usize {
        self.padded_training_rows
    }
    pub const fn packed_validity_logical_bytes(&self) -> usize {
        self.packed_validity_logical_bytes
    }
    pub const fn packed_validity_allocated_bytes(&self) -> usize {
        self.packed_validity_allocated_bytes
    }
    pub const fn normalization_scratch_bytes(&self) -> usize {
        self.normalization_scratch_bytes
    }
    pub const fn fit_metadata_bytes(&self) -> usize {
        self.fit_metadata_bytes
    }
    pub const fn fit_metadata_sha256(&self) -> [u8; SHA256_BYTES] {
        self.fit_metadata_sha256
    }
    pub const fn batch_count(&self) -> usize {
        self.batch_count
    }
    pub const fn native_launch_count(&self) -> usize {
        self.native_launch_count
    }
    pub const fn producer_ready_event_count(&self) -> usize {
        self.producer_ready_event_count
    }
    pub const fn producer_ready_event_synchronize_count(&self) -> usize {
        self.producer_ready_event_synchronize_count
    }
    pub const fn control_error_device_bytes(&self) -> usize {
        self.control_error_device_bytes
    }
    pub const fn control_error_readback_count(&self) -> usize {
        self.control_error_readback_count
    }
    pub const fn control_error_d2h_bytes(&self) -> usize {
        self.control_error_d2h_bytes
    }
    pub const fn fit_digest_readback_count(&self) -> usize {
        self.fit_digest_readback_count
    }
    pub const fn fit_digest_d2h_bytes(&self) -> usize {
        self.fit_digest_d2h_bytes
    }
    pub const fn fit_words_readback_count(&self) -> usize {
        self.fit_words_readback_count
    }
    pub const fn fit_words_d2h_bytes(&self) -> usize {
        self.fit_words_d2h_bytes
    }
    pub fn fit_metadata_words_v3(&self) -> &[u64] {
        &self.fit_metadata_words_v3
    }
    pub const fn parent_input_h2d_bytes(&self) -> usize {
        self.parent_input_h2d_bytes
    }
    pub const fn feature_value_d2h_bytes(&self) -> usize {
        self.feature_value_d2h_bytes
    }
    pub const fn admission_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.admission_identity_sha256
    }
    pub const fn primary_context_process_token(&self) -> [u8; SHA256_BYTES] {
        self.primary_context_process_token
    }
    pub const fn producer_stream_process_token(&self) -> [u8; SHA256_BYTES] {
        self.producer_stream_process_token
    }
    pub const fn ready_event_process_token(&self) -> [u8; SHA256_BYTES] {
        self.ready_event_process_token
    }

    pub(crate) fn seal_after_ready_event_v2(
        mut self,
        fit_metadata_sha256: [u8; SHA256_BYTES],
        fit_metadata_words: Vec<u64>,
        admission_identity_sha256: [u8; SHA256_BYTES],
        primary_context_process_token: [u8; SHA256_BYTES],
        producer_stream_process_token: [u8; SHA256_BYTES],
        ready_event_process_token: [u8; SHA256_BYTES],
    ) -> Result<Self, ResidentFeatureStoreCudaErrorV3> {
        if !self.enabled
            || self.producer_ready_event_count != 0
            || fit_metadata_sha256 == [0; SHA256_BYTES]
            || admission_identity_sha256 == [0; SHA256_BYTES]
            || primary_context_process_token == [0; SHA256_BYTES]
            || producer_stream_process_token == [0; SHA256_BYTES]
            || ready_event_process_token == [0; SHA256_BYTES]
            || fit_metadata_words.len().checked_mul(8) != Some(self.fit_metadata_bytes)
            || resident_normalization_fit_metadata_sha256_v3(&fit_metadata_words)
                != fit_metadata_sha256
        {
            return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "resident robust-normalization ready-event evidence is incomplete or duplicated"
                    .into(),
            ));
        }
        self.fit_metadata_sha256 = fit_metadata_sha256;
        self.producer_ready_event_count = 1;
        self.producer_ready_event_synchronize_count = 1;
        self.control_error_readback_count = 1;
        self.control_error_d2h_bytes = std::mem::size_of::<u32>();
        self.fit_digest_readback_count = 1;
        self.fit_digest_d2h_bytes = SHA256_BYTES;
        self.fit_words_readback_count = 1;
        self.fit_words_d2h_bytes = self.fit_metadata_bytes;
        self.fit_metadata_words_v3 = fit_metadata_words;
        self.admission_identity_sha256 = admission_identity_sha256;
        self.primary_context_process_token = primary_context_process_token;
        self.producer_stream_process_token = producer_stream_process_token;
        self.ready_event_process_token = ready_event_process_token;
        Ok(self)
    }
}

pub(crate) fn disabled_resident_robust_normalization_receipt_v2(
    plan: &ResidentRobustNormalizationPlanV2,
    admission_identity_sha256: [u8; SHA256_BYTES],
    primary_context_process_token: [u8; SHA256_BYTES],
    producer_stream_process_token: [u8; SHA256_BYTES],
) -> Result<ResidentRobustNormalizationRuntimeReceiptV2, ResidentFeatureStoreCudaErrorV3> {
    if plan.enabled
        || plan.padded_training_rows != 0
        || plan.normalization_scratch_bytes != 0
        || plan.fit_metadata_bytes != 0
        || plan.native_launch_count != 0
        || admission_identity_sha256 == [0; SHA256_BYTES]
        || primary_context_process_token == [0; SHA256_BYTES]
        || producer_stream_process_token == [0; SHA256_BYTES]
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "disabled resident normalization carries nonzero work or incomplete run identity"
                .into(),
        ));
    }
    Ok(ResidentRobustNormalizationRuntimeReceiptV2 {
        semantic_version: RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2,
        enabled: false,
        row_count: plan.rows,
        feature_column_count: plan.columns,
        training_rows: plan.training_rows.clone(),
        padded_training_rows: 0,
        packed_validity_logical_bytes: plan.packed_validity_logical_bytes,
        packed_validity_allocated_bytes: plan.packed_validity_allocated_bytes,
        normalization_scratch_bytes: 0,
        fit_metadata_bytes: 0,
        fit_metadata_sha256: resident_robust_normalization_disabled_fit_sha256_v2(),
        batch_count: 0,
        native_launch_count: 0,
        producer_ready_event_count: 0,
        producer_ready_event_synchronize_count: 0,
        control_error_device_bytes: 0,
        control_error_readback_count: 0,
        control_error_d2h_bytes: 0,
        fit_digest_readback_count: 0,
        fit_digest_d2h_bytes: 0,
        fit_words_readback_count: 0,
        fit_words_d2h_bytes: 0,
        fit_metadata_words_v3: Vec::new(),
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        admission_identity_sha256,
        primary_context_process_token,
        producer_stream_process_token,
        ready_event_process_token: [0; SHA256_BYTES],
    })
}

pub(crate) fn launch_resident_robust_normalization_v2(
    plan: &ResidentRobustNormalizationPlanV2,
    bar_major_values: &mut DeviceBuffer<f64>,
    bar_major_validity_u4: &mut DeviceBuffer<u8>,
    sort_scratch_bits: &mut DeviceBuffer<u64>,
    fit_metadata_words: &mut DeviceBuffer<u64>,
    control_error: &mut DeviceBuffer<u32>,
    stream: &Stream,
) -> Result<ResidentRobustNormalizationRuntimeReceiptV2, ResidentFeatureStoreCudaErrorV3> {
    let column_modes = plan.column_modes_v3().ok_or_else(|| {
        ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "enabled Search normalization requires exact ordered policy modes".into(),
        )
    })?;
    let validity_address = bar_major_validity_u4.as_device_ptr().as_raw();
    if !plan.enabled
        || bar_major_values.len()
            != plan.rows.checked_mul(plan.columns).ok_or(
                ResidentFeatureStoreCudaErrorV3::ArithmeticOverflow(
                    "robust-normalization feature cells",
                ),
            )?
        || bar_major_validity_u4.len() != plan.packed_validity_allocated_bytes
        || plan.packed_validity_allocated_bytes % VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2 != 0
        || validity_address % VALIDITY_ATOMIC_ALIGNMENT_BYTES_V2 as u64 != 0
        || sort_scratch_bits.len() != plan.normalization_scratch_slots
        || sort_scratch_bits.len() < SHA256_BYTES / std::mem::size_of::<u64>()
        || fit_metadata_words.len() != plan.fit_metadata_words
        || control_error.len() != 1
        || stream.as_inner().is_null()
    {
        return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
            "resident robust-normalization runtime extents or u4 alignment differ from preflight"
                .into(),
        ));
    }
    let status = unsafe {
        neoethos_resident_robust_normalize_bar_major_f64_u4_v3(
            bar_major_values.as_device_ptr().as_mut_ptr(),
            bar_major_validity_u4.as_device_ptr().as_mut_ptr(),
            plan.packed_validity_allocated_bytes,
            plan.rows,
            plan.columns,
            plan.training_rows.start,
            plan.training_rows.end,
            plan.padded_training_rows,
            sort_scratch_bits.as_device_ptr().as_mut_ptr(),
            plan.normalization_scratch_slots,
            fit_metadata_words.as_device_ptr().as_mut_ptr(),
            plan.fit_metadata_words,
            column_modes.as_ptr().cast::<u8>(),
            column_modes.len(),
            control_error.as_device_ptr().as_mut_ptr(),
            stream.as_inner(),
        )
    };
    if status != 0 {
        return Err(ResidentFeatureStoreCudaErrorV3::Native {
            operation: "neoethos_resident_robust_normalize_bar_major_f64_u4_v3",
            status,
        });
    }
    Ok(ResidentRobustNormalizationRuntimeReceiptV2 {
        semantic_version: RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2,
        enabled: true,
        row_count: plan.rows,
        feature_column_count: plan.columns,
        training_rows: plan.training_rows.clone(),
        padded_training_rows: plan.padded_training_rows,
        packed_validity_logical_bytes: plan.packed_validity_logical_bytes,
        packed_validity_allocated_bytes: plan.packed_validity_allocated_bytes,
        normalization_scratch_bytes: plan.normalization_scratch_bytes,
        fit_metadata_bytes: plan.fit_metadata_bytes,
        fit_metadata_sha256: [0; SHA256_BYTES],
        batch_count: plan.batch_count,
        native_launch_count: plan.native_launch_count,
        producer_ready_event_count: 0,
        producer_ready_event_synchronize_count: 0,
        control_error_device_bytes: std::mem::size_of::<u32>(),
        control_error_readback_count: 0,
        control_error_d2h_bytes: 0,
        fit_digest_readback_count: 0,
        fit_digest_d2h_bytes: 0,
        fit_words_readback_count: 0,
        fit_words_d2h_bytes: 0,
        fit_metadata_words_v3: Vec::new(),
        parent_input_h2d_bytes: 0,
        feature_value_d2h_bytes: 0,
        admission_identity_sha256: [0; SHA256_BYTES],
        primary_context_process_token: [0; SHA256_BYTES],
        producer_stream_process_token: [0; SHA256_BYTES],
        ready_event_process_token: [0; SHA256_BYTES],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_policy_binding_is_exact_and_counts_actual_mode_dependent_launches() {
        use SearchNormalizationColumnModeV3::{Binary, Robust, SignedContinuous, SignedState};
        let geometry = ResidentRobustNormalizationPlanV2::preflight(100, 65, 0..80, true).unwrap();
        assert!(
            geometry.column_modes_v3().is_none(),
            "geometry is not runtime policy authority"
        );
        assert!(
            geometry
                .clone()
                .with_column_modes_v3(vec![Robust; 64])
                .is_err()
        );
        let robust = geometry
            .clone()
            .with_column_modes_v3(vec![Robust; 65])
            .unwrap();
        assert_eq!(robust.native_launch_count(), 2 * (56 + 5) + 1);
        let gate = geometry
            .clone()
            .with_column_modes_v3(vec![Binary; 65])
            .unwrap();
        assert_eq!(gate.native_launch_count(), 2 * 6 + 1);
        assert!(gate.with_column_modes_v3(vec![Binary; 65]).is_err());
        let mut modes = vec![Robust; 65];
        modes[0] = Binary;
        modes[63] = SignedState;
        modes[64] = SignedContinuous;
        let mixed = geometry.with_column_modes_v3(modes.clone()).unwrap();
        assert_eq!(mixed.column_modes_v3(), Some(modes.as_slice()));
        assert_eq!(mixed.native_launch_count(), (56 + 6) + 6 + 1);
        assert_eq!(
            mixed.normalization_scratch_bytes(),
            robust.normalization_scratch_bytes()
        );
    }

    #[test]
    fn preflight_freezes_enabled_disabled_extents_alignment_and_launches() {
        let plan = ResidentRobustNormalizationPlanV2::preflight(100, 65, 0..80, true)
            .expect("exact normalization plan");
        assert!(plan.enabled());
        assert_eq!(plan.padded_training_rows(), 128);
        assert_eq!(plan.packed_validity_logical_bytes(), 3_250);
        assert_eq!(plan.packed_validity_allocated_bytes(), 3_252);
        assert_eq!(plan.batch_count(), 2);
        assert_eq!(plan.normalization_scratch_slots(), 64 * 128);
        assert_eq!(plan.normalization_scratch_bytes(), 64 * 128 * 8);
        assert_eq!(plan.fit_metadata_words(), 65 * 6);
        assert_eq!(plan.fit_metadata_bytes(), 65 * 48);
        // log2(128)=7: each batch has two 28-stage sorts plus five kernels;
        // the final fit-metadata SHA is one additional launch.
        assert_eq!(plan.native_launch_count(), 2 * (7 * 8 + 5) + 1);

        let disabled = ResidentRobustNormalizationPlanV2::preflight(100, 65, 0..80, false)
            .expect("disabled normalization plan");
        assert!(!disabled.enabled());
        assert_eq!(disabled.padded_training_rows(), 0);
        assert_eq!(disabled.normalization_scratch_bytes(), 0);
        assert_eq!(disabled.fit_metadata_bytes(), 0);
        assert_eq!(disabled.native_launch_count(), 0);
        assert!(ResidentRobustNormalizationPlanV2::preflight(100, 65, 0..79, true).is_err());
        assert!(ResidentRobustNormalizationPlanV2::preflight(79, 65, 0..63, true).is_err());
    }
}
