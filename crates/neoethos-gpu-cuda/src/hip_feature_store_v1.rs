//! One completed HIP feature-major -> bar-major/u4 pack and V3 Merkle seal.
//!
//! This is physical content evidence, not Data source, selection, chronology,
//! normalization, admission, or CPU/GPU parity authority. In particular, the
//! timestamp buffer is checked for ownership/extent and hashed exactly; genuine
//! calendar validation belongs to the Data producer retaining that buffer.
//! All device work calls the existing production packer and Merkle kernels.

use super::{
    HipDeviceBufferV1, HipRunLeaseV1, HipRuntimeErrorV1, HipRuntimeIdentityV1,
    HipRuntimeMemorySnapshotV1, RawHipRuntimeErrorV1,
};
use crate::population::{
    CudaPopulationError, HostPopulationMetricsReceiptV1, PopulationEvaluationViewV1,
    PopulationGeneView, PopulationSession, RawHipResidentFeatureStoreBindV1,
};
use crate::{NeoPopulationSettings, ScenarioDescriptor};
use neoethos_gpu_contracts::normalization_v3::{
    SearchNormalizationColumnModeV3, resident_normalization_fit_metadata_sha256_v3,
};
use neoethos_gpu_contracts::resident_feature_store_v3::CANONICAL_MERKLE_CHUNK_ROWS_V3;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ops::Range;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawHipFeatureColumnV1 {
    values_key: u64,
    values_element_offset: u64,
    validity_key: u64,
    validity_byte_offset: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
struct RawHipFeatureStoreReceiptV4 {
    abi_version: u32,
    backend_kind: u32,
    rows: u64,
    columns: u64,
    value_bytes: u64,
    validity_bytes: u64,
    transient_device_bytes: u64,
    metadata_upload_bytes: u64,
    control: u32,
    readback_count: u32,
    readback_bytes: u64,
    root: [u8; 32],
    normalization_training_start: u64,
    normalization_training_end: u64,
    fit_word_count: u64,
    fit_metadata_digest: [u8; 32],
}

#[repr(C)]
struct RawHipFeatureNormalizationV3 {
    training_row_start: u64,
    training_row_end: u64,
    column_modes: *const u8,
    column_mode_count: u64,
}

unsafe extern "C" {
    fn neoethos_hip_native_build_manifest_sha256_v1() -> *const u8;
    fn neoethos_hip_runtime_pack_feature_store_v4(
        lease: u64,
        rows: u64,
        columns: u64,
        descriptors: *const RawHipFeatureColumnV1,
        timestamps_key: u64,
        name_offsets: *const u64,
        name_bytes: *const u8,
        name_bytes_len: u64,
        allocator_reserve_bytes: u64,
        normalization: *const RawHipFeatureNormalizationV3,
        output_values_key: u64,
        output_validity_key: u64,
        host_fit_words: *mut u64,
        host_fit_word_capacity: u64,
        receipt: *mut RawHipFeatureStoreReceiptV4,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
}

fn invalid(reason: &'static str) -> HipRuntimeErrorV1 {
    HipRuntimeErrorV1::InvalidInput(reason)
}

fn checked_add(a: usize, b: usize) -> Result<usize, HipRuntimeErrorV1> {
    a.checked_add(b)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("HIP feature-store extent overflows"))
}

fn checked_mul(a: usize, b: usize) -> Result<usize, HipRuntimeErrorV1> {
    a.checked_mul(b)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("HIP feature-store extent overflows"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExtentsV1 {
    values: usize,
    validity_logical: usize,
    validity_allocated: usize,
    metadata_upload: usize,
    merkle_leaf_count: usize,
    transient: usize,
    peak: usize,
}

fn extents_v1(rows: usize, columns: usize, names: usize) -> Result<ExtentsV1, HipRuntimeErrorV1> {
    if rows == 0 || columns == 0 || columns > u32::MAX as usize || names == 0 {
        return Err(invalid(
            "HIP feature-store rows, columns, and names must be nonempty",
        ));
    }
    let cells = checked_mul(rows, columns)?;
    let values = checked_mul(cells, 8)?;
    let validity_logical = cells.div_ceil(2);
    // Boundary nibble updates use atomicOr on aligned u32 words. Even a one-cell
    // store needs four initialized bytes, not merely its one logical u4 byte.
    let validity_allocated = checked_mul(cells.div_ceil(8), 4)?;
    let producers = checked_add(columns, 1)?;
    let metadata_upload = checked_add(
        checked_add(checked_mul(columns, 32)?, checked_mul(producers, 8)?)?,
        names,
    )?;
    let merkle_leaf_count = checked_mul(rows.div_ceil(CANONICAL_MERKLE_CHUNK_ROWS_V3), producers)?;
    // Four compact metadata arrays, names, two complete Merkle levels, a device
    // u32 validity verdict and a 32-byte digest. No normalizer slack is reused.
    let transient = checked_add(
        checked_add(metadata_upload, checked_mul(merkle_leaf_count, 64)?)?,
        36,
    )?;
    let peak = checked_add(checked_add(values, validity_allocated)?, transient)?;
    Ok(ExtentsV1 {
        values,
        validity_logical,
        validity_allocated,
        metadata_upload,
        merkle_leaf_count,
        transient,
        peak,
    })
}

/// Explicit policy-v3 request. Data supplies its actual canonical training
/// range and name-derived modes; this checked geometry does not mint provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HipFeatureNormalizationV3 {
    rows: usize,
    training_rows: Range<usize>,
    column_modes: Vec<SearchNormalizationColumnModeV3>,
    scratch_bytes: usize,
    fit_words: usize,
    fit_bytes: usize,
}

impl HipFeatureNormalizationV3 {
    pub fn preflight(
        rows: usize,
        training_rows: Range<usize>,
        column_modes: Vec<SearchNormalizationColumnModeV3>,
    ) -> Result<Self, HipRuntimeErrorV1> {
        let canonical_end = ((rows as f64) * (1.0 - 0.2)).floor() as usize;
        if rows == 0
            || training_rows != (0..canonical_end)
            || canonical_end < 64
            || canonical_end >= rows
            || column_modes.is_empty()
        {
            return Err(invalid(
                "HIP normalization requires the exact canonical training range and ordered modes",
            ));
        }
        let padded = canonical_end
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("HIP normalization padded rows overflow"))?;
        let scratch_bytes = checked_mul(checked_mul(column_modes.len().min(64), padded)?, 8)?;
        let fit_words = checked_mul(column_modes.len(), 6)?;
        let fit_bytes = checked_mul(fit_words, 8)?;
        Ok(Self {
            rows,
            training_rows,
            column_modes,
            scratch_bytes,
            fit_words,
            fit_bytes,
        })
    }
    pub fn training_rows(&self) -> Range<usize> {
        self.training_rows.clone()
    }
    pub fn column_modes(&self) -> &[SearchNormalizationColumnModeV3] {
        &self.column_modes
    }
    pub const fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
    pub const fn fit_metadata_bytes(&self) -> usize {
        self.fit_bytes
    }
}

/// Checked, exact logical allocation requests for this operation only. Existing
/// parent/producer allocations remain live and are not subtracted or re-uploaded.
/// These byte counts are not a device reservation or a claim about pool overhead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HipFeatureStorePlanV1 {
    rows: usize,
    names: Vec<String>,
    name_offsets: Vec<u64>,
    name_bytes: Vec<u8>,
    extents: ExtentsV1,
    normalization: Option<HipFeatureNormalizationV3>,
}

impl HipFeatureStorePlanV1 {
    pub fn preflight(rows: usize, ordered_names: &[String]) -> Result<Self, HipRuntimeErrorV1> {
        let mut unique = HashSet::new();
        unique
            .try_reserve(ordered_names.len())
            .map_err(|_| invalid("HIP feature-name validation allocation failed"))?;
        let mut name_len = 0;
        for name in ordered_names {
            if name.trim().is_empty() || !unique.insert(name.as_str()) {
                return Err(invalid(
                    "HIP feature names must be nonempty and unique in exact order",
                ));
            }
            name_len = checked_add(name_len, name.len())?;
        }
        let extents = extents_v1(rows, ordered_names.len(), name_len)?;
        let mut name_offsets = Vec::new();
        name_offsets
            .try_reserve_exact(ordered_names.len() + 1)
            .map_err(|_| invalid("HIP feature-name offset allocation failed"))?;
        let mut name_bytes = Vec::new();
        name_bytes
            .try_reserve_exact(name_len)
            .map_err(|_| invalid("HIP feature-name allocation failed"))?;
        let mut names = Vec::new();
        names
            .try_reserve_exact(ordered_names.len())
            .map_err(|_| invalid("HIP feature-name ownership allocation failed"))?;
        name_offsets.push(0);
        for name in ordered_names {
            let mut owned = String::new();
            owned
                .try_reserve_exact(name.len())
                .map_err(|_| invalid("HIP owned feature-name allocation failed"))?;
            owned.push_str(name);
            names.push(owned);
            name_bytes.extend_from_slice(name.as_bytes());
            name_offsets.push(name_bytes.len() as u64);
        }
        Ok(Self {
            rows,
            names,
            name_offsets,
            name_bytes,
            extents,
            normalization: None,
        })
    }

    pub fn with_normalization(
        mut self,
        request: HipFeatureNormalizationV3,
    ) -> Result<Self, HipRuntimeErrorV1> {
        if self.normalization.is_some()
            || request.rows != self.rows
            || request.column_modes.len() != self.columns()
        {
            return Err(invalid("HIP normalization must bind this exact store once"));
        }
        // Both Merkle levels remain admitted while the normalizer runs. Its
        // digest reuses the first32 scratch bytes; no fictitious pool slack.
        let extra = checked_add(request.scratch_bytes, request.fit_bytes)?;
        self.extents.transient = checked_add(self.extents.transient, extra)?;
        self.extents.peak = checked_add(self.extents.peak, extra)?;
        self.normalization = Some(request);
        Ok(self)
    }
    pub fn normalization(&self) -> Option<&HipFeatureNormalizationV3> {
        self.normalization.as_ref()
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.names.len()
    }
    pub fn ordered_names(&self) -> &[String] {
        &self.names
    }
    pub const fn values_bytes(&self) -> usize {
        self.extents.values
    }
    /// Actual four-byte-padded allocation, not just the logical u4 extent.
    pub const fn validity_bytes(&self) -> usize {
        self.extents.validity_allocated
    }
    pub const fn logical_validity_bytes(&self) -> usize {
        self.extents.validity_logical
    }
    pub const fn transient_device_bytes(&self) -> usize {
        self.extents.transient
    }
    pub const fn incremental_peak_device_bytes(&self) -> usize {
        self.extents.peak
    }
    pub const fn metadata_upload_bytes(&self) -> usize {
        self.extents.metadata_upload
    }
    pub const fn merkle_leaf_count(&self) -> usize {
        self.extents.merkle_leaf_count
    }

    /// Consume this plan and new output allocations in one logical operation.
    /// Native code completes all counted readbacks and retires every temporary before
    /// success. An error cannot yield a partially accepted or retryable store.
    /// No feature values, parent prices or timestamps are downloaded or uploaded.
    pub fn pack_and_seal<'timestamp, 'lease>(
        self,
        lease: &'lease HipRunLeaseV1,
        timestamps: &'timestamp HipDeviceBufferV1<'lease>,
        columns: &[HipFeatureColumnV1<'_, 'lease>],
        allocator_reserve_bytes: u64,
    ) -> Result<SealedHipFeatureStoreV1<'timestamp, 'lease>, HipRuntimeErrorV1> {
        lease.require_active()?;
        if columns.len() != self.columns() {
            return Err(invalid(
                "HIP feature descriptors do not cover the exact plan",
            ));
        }
        let timestamp_bytes = checked_mul(self.rows, 8)?;
        let timestamps_key = lease.checked_buffer_keys_v1([timestamps], [timestamp_bytes])?[0];
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(columns.len())
            .map_err(|_| invalid("HIP feature descriptor allocation failed"))?;
        for (column, name) in columns.iter().zip(&self.names) {
            if column.name != name {
                return Err(invalid(
                    "HIP feature descriptor name/order differs from the plan",
                ));
            }
            validate_source_extent_v1(
                self.rows,
                column.values.bytes,
                column.values_element_offset,
                column.validity.bytes,
                column.validity_byte_offset,
            )?;
            let keys = lease.checked_buffer_keys_v1(
                [column.values, column.validity],
                [column.values.bytes, column.validity.bytes],
            )?;
            descriptors.push(RawHipFeatureColumnV1 {
                values_key: keys[0],
                values_element_offset: column.values_element_offset as u64,
                validity_key: keys[1],
                validity_byte_offset: column.validity_byte_offset as u64,
            });
        }
        // Probe after the existing parents and producers are resident. Native
        // repeats the transient+reserve check after the two output allocations.
        let memory_at_pack = lease.revalidate()?;
        require_headroom_v1(
            self.extents.peak as u64,
            allocator_reserve_bytes,
            memory_at_pack.free_memory_bytes(),
        )?;
        let mut fit_words = Vec::new();
        let fit_word_count = self.normalization.as_ref().map_or(0, |p| p.fit_words);
        fit_words
            .try_reserve_exact(fit_word_count)
            .map_err(|_| invalid("HIP normalization fit-word allocation failed"))?;
        fit_words.resize(fit_word_count, 0u64);
        let normalization = self
            .normalization
            .as_ref()
            .map(|plan| RawHipFeatureNormalizationV3 {
                training_row_start: plan.training_rows.start as u64,
                training_row_end: plan.training_rows.end as u64,
                column_modes: plan.column_modes.as_ptr().cast::<u8>(),
                column_mode_count: plan.column_modes.len() as u64,
            });
        let values = lease.allocate_bytes(self.values_bytes())?;
        let validity = lease.allocate_bytes(self.validity_bytes())?;
        let mut raw = RawHipFeatureStoreReceiptV4::default();
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: descriptors/names are exact initialized host slices retained
        // for this synchronous logical call. Only private same-lease keys cross
        // FFI. Native code owns any async staging until completion, even on error.
        let status = unsafe {
            neoethos_hip_runtime_pack_feature_store_v4(
                lease.identity().lease_id(),
                self.rows as u64,
                self.columns() as u64,
                descriptors.as_ptr(),
                timestamps_key,
                self.name_offsets.as_ptr(),
                self.name_bytes.as_ptr(),
                self.name_bytes.len() as u64,
                allocator_reserve_bytes,
                normalization
                    .as_ref()
                    .map_or(std::ptr::null(), std::ptr::from_ref),
                values.key.get(),
                validity.key.get(),
                if fit_words.is_empty() {
                    std::ptr::null_mut()
                } else {
                    fit_words.as_mut_ptr()
                },
                fit_word_count as u64,
                &mut raw,
                &mut error,
            )
        };
        lease.resource_status("feature-store completed pack/Merkle seal", status, &error)?;
        let receipt = match validate_receipt_v1(&self, raw, &fit_words) {
            Ok(receipt) => receipt,
            Err(error) => {
                // Malformed native success is not authority to reuse/free the
                // outputs. The native registry retains them with this lease.
                lease.state.set(super::LeaseStateV1::Quarantined);
                return Err(error);
            }
        };
        Ok(SealedHipFeatureStoreV1 {
            plan: self,
            lease,
            timestamps,
            values,
            validity,
            receipt,
            memory_at_pack,
            normalization_fit_words: fit_words,
        })
    }
}

fn require_headroom_v1(peak: u64, reserve: u64, free: u64) -> Result<(), HipRuntimeErrorV1> {
    if peak
        .checked_add(reserve)
        .is_none_or(|required| required > free)
    {
        Err(invalid(
            "HIP feature-store peak plus reserve exceeds the current free-memory snapshot",
        ))
    } else {
        Ok(())
    }
}

fn validate_source_extent_v1(
    rows: usize,
    value_bytes: usize,
    value_offset: usize,
    validity_bytes: usize,
    validity_offset: usize,
) -> Result<(), HipRuntimeErrorV1> {
    let value_end_bytes = checked_mul(checked_add(value_offset, rows)?, 8)?;
    let validity_end = checked_add(validity_offset, rows)?;
    if value_bytes % 8 != 0 || value_end_bytes > value_bytes || validity_end > validity_bytes {
        return Err(invalid(
            "HIP source column span exceeds its owned value/validity allocation",
        ));
    }
    Ok(())
}

/// Borrowed production column span. Construction grants no source-family or
/// normalization authority; Data owns that separate contract. Buffer keys and
/// device addresses are neither accepted nor exposed by this API.
pub struct HipFeatureColumnV1<'source, 'lease> {
    name: &'source str,
    values: &'source HipDeviceBufferV1<'lease>,
    values_element_offset: usize,
    validity: &'source HipDeviceBufferV1<'lease>,
    validity_byte_offset: usize,
}

impl<'source, 'lease> HipFeatureColumnV1<'source, 'lease> {
    pub fn new(
        name: &'source str,
        values: &'source HipDeviceBufferV1<'lease>,
        values_element_offset: usize,
        validity: &'source HipDeviceBufferV1<'lease>,
        validity_byte_offset: usize,
    ) -> Self {
        Self {
            name,
            values,
            values_element_offset,
            validity,
            validity_byte_offset,
        }
    }
}

/// Exact observed physical pack/hash receipt. It deliberately has no caller
/// constructor and no fields labelled Data admission or canonical source proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HipFeatureStoreReceiptV1 {
    root: [u8; 32],
    metadata_upload_bytes: u64,
    transient_device_bytes: u64,
    readback_count: u32,
    readback_bytes: u64,
    fit_metadata_digest: Option<[u8; 32]>,
}

impl HipFeatureStoreReceiptV1 {
    pub const fn canonical_content_merkle_sha256(&self) -> [u8; 32] {
        self.root
    }
    pub const fn metadata_upload_bytes(&self) -> u64 {
        self.metadata_upload_bytes
    }
    /// All of these temporaries were retired before the successful native return.
    pub const fn retired_transient_device_bytes(&self) -> u64 {
        self.transient_device_bytes
    }
    pub const fn readback_count(&self) -> u32 {
        self.readback_count
    }
    pub const fn readback_bytes(&self) -> u64 {
        self.readback_bytes
    }
    /// Native six-word metadata digest, not Data's name-aware fitted-state hash.
    pub const fn normalization_fit_metadata_sha256(&self) -> Option<[u8; 32]> {
        self.fit_metadata_digest
    }
}

fn validate_receipt_v1(
    plan: &HipFeatureStorePlanV1,
    raw: RawHipFeatureStoreReceiptV4,
    fit_words: &[u64],
) -> Result<HipFeatureStoreReceiptV1, HipRuntimeErrorV1> {
    let (count, bytes, start, end, words) =
        plan.normalization.as_ref().map_or((2, 36, 0, 0, 0), |p| {
            (
                5,
                72 + p.fit_bytes as u64,
                p.training_rows.start as u64,
                p.training_rows.end as u64,
                p.fit_words as u64,
            )
        });
    let fit_digest = if words == 0 {
        [0; 32]
    } else {
        resident_normalization_fit_metadata_sha256_v3(fit_words)
    };
    if raw.abi_version != 4
        || raw.backend_kind != 2
        || raw.rows != plan.rows as u64
        || raw.columns != plan.columns() as u64
        || raw.value_bytes != plan.values_bytes() as u64
        || raw.validity_bytes != plan.validity_bytes() as u64
        || raw.transient_device_bytes != plan.transient_device_bytes() as u64
        || raw.metadata_upload_bytes != plan.metadata_upload_bytes() as u64
        || raw.control != 0
        || raw.readback_count != count
        || raw.readback_bytes != bytes
        || raw.root == [0; 32]
        || raw.normalization_training_start != start
        || raw.normalization_training_end != end
        || raw.fit_word_count != words
        || fit_words.len() as u64 != words
        || raw.fit_metadata_digest != fit_digest
    {
        return Err(HipRuntimeErrorV1::InvalidNativeFacts(
            "HIP feature-store seal or accounting differs from its exact plan",
        ));
    }
    Ok(HipFeatureStoreReceiptV1 {
        root: raw.root,
        metadata_upload_bytes: raw.metadata_upload_bytes,
        transient_device_bytes: raw.transient_device_bytes,
        readback_count: raw.readback_count,
        readback_bytes: raw.readback_bytes,
        fit_metadata_digest: (words != 0).then_some(raw.fit_metadata_digest),
    })
}

/// Immutable physical output owner. No public method returns its output buffers
/// or raw keys, so a successful seal cannot be rewritten through a safe alias.
/// The actual timestamp allocation and lease stay borrowed for the whole lifetime.
#[must_use]
pub struct SealedHipFeatureStoreV1<'timestamp, 'lease> {
    plan: HipFeatureStorePlanV1,
    lease: &'lease HipRunLeaseV1,
    timestamps: &'timestamp HipDeviceBufferV1<'lease>,
    values: HipDeviceBufferV1<'lease>,
    validity: HipDeviceBufferV1<'lease>,
    receipt: HipFeatureStoreReceiptV1,
    memory_at_pack: HipRuntimeMemorySnapshotV1,
    normalization_fit_words: Vec<u64>,
}

impl std::fmt::Debug for SealedHipFeatureStoreV1<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SealedHipFeatureStoreV1")
            .field("plan", &self.plan)
            .field("runtime_identity", self.lease.identity())
            .field("receipt", &self.receipt)
            .field("memory_at_pack", &self.memory_at_pack)
            .finish_non_exhaustive()
    }
}

impl SealedHipFeatureStoreV1<'_, '_> {
    pub fn plan(&self) -> &HipFeatureStorePlanV1 {
        &self.plan
    }
    pub fn receipt(&self) -> &HipFeatureStoreReceiptV1 {
        &self.receipt
    }
    pub fn runtime_identity(&self) -> &HipRuntimeIdentityV1 {
        self.lease.identity()
    }
    pub const fn memory_at_pack(&self) -> HipRuntimeMemorySnapshotV1 {
        self.memory_at_pack
    }
    pub const fn canonical_content_merkle_sha256(&self) -> [u8; 32] {
        self.receipt.root
    }
    /// Actual device-produced fits, six words per ordered column. Disabled
    /// normalization has no fit scope or synthesized identity parameters.
    pub fn normalization_fit_words(&self) -> Option<&[u64]> {
        self.plan
            .normalization
            .as_ref()
            .map(|_| self.normalization_fit_words.as_slice())
    }
    /// Retained source-clock bytes, not a chronology assertion or raw buffer view.
    pub fn retained_timestamp_bytes(&self) -> usize {
        self.timestamps.len_bytes()
    }

    pub fn try_close(self) -> Result<(), HipRuntimeErrorV1> {
        let mut first = None;
        for buffer in [self.values, self.validity] {
            if let Err(error) = buffer.try_close() {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

/// Failure from the HIP physical-parent boundary. The shared population error
/// type keeps its historical name; it does not confer CUDA identity or authority.
#[derive(Debug, thiserror::Error)]
pub enum HipPopulationParentErrorV1 {
    #[error(transparent)]
    Runtime(#[from] HipRuntimeErrorV1),
    #[error(transparent)]
    Population(#[from] CudaPopulationError),
}

/// Explicit decomposition of future Search workspace and untouched allocator
/// headroom. The adaptive view is admitted before preparation, then excluded from
/// the post-view reserve because the native free-memory snapshot already includes
/// its allocation. This host plan is not a reservation or Data authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HipSearchEvaluationBudgetV3 {
    allocator_headroom_bytes: u64,
    retained_capacity: usize,
    month_capacity: u32,
    adaptive_view_rows: usize,
    metrics_workspace_bytes: u64,
    adaptive_view_bytes: u64,
    total_reserve_bytes: u64,
}
impl HipSearchEvaluationBudgetV3 {
    pub fn checked_v3(
        allocator_headroom_bytes: u64,
        retained_capacity: usize,
        month_capacity: u32,
        adaptive_view_rows: usize,
    ) -> Result<Self, HipPopulationParentErrorV1> {
        if allocator_headroom_bytes == 0
            || retained_capacity == 0
            || retained_capacity > i32::MAX as usize
            || month_capacity == 0
            || month_capacity > i32::MAX as u32
            || adaptive_view_rows > i32::MAX as usize
            || (adaptive_view_rows != 0 && adaptive_view_rows < 101)
        {
            return Err(invalid("HIP Search evaluation budget has invalid extents").into());
        }
        let metrics = crate::PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            retained_capacity,
            month_capacity,
        )?;
        let adaptive_view_bytes = (adaptive_view_rows as u64)
            .checked_mul(8)
            .ok_or_else(|| invalid("HIP retained adaptive view bytes overflow"))?;
        let metrics_workspace_bytes = metrics.total_device_bytes();
        let total_reserve_bytes = allocator_headroom_bytes
            .checked_add(metrics_workspace_bytes)
            .filter(|bytes| *bytes <= isize::MAX as u64)
            .ok_or_else(|| invalid("HIP Search reserve decomposition overflows"))?;
        total_reserve_bytes
            .checked_add(adaptive_view_bytes)
            .filter(|bytes| *bytes <= isize::MAX as u64)
            .ok_or_else(|| invalid("HIP Search pre-view peak overflows"))?;
        Ok(Self {
            allocator_headroom_bytes,
            retained_capacity,
            month_capacity,
            adaptive_view_rows,
            metrics_workspace_bytes,
            adaptive_view_bytes,
            total_reserve_bytes,
        })
    }
    pub const fn allocator_headroom_bytes(&self) -> u64 {
        self.allocator_headroom_bytes
    }
    pub const fn retained_capacity(&self) -> usize {
        self.retained_capacity
    }
    pub const fn month_capacity(&self) -> u32 {
        self.month_capacity
    }
    pub const fn adaptive_view_rows(&self) -> usize {
        self.adaptive_view_rows
    }
    pub const fn metrics_workspace_bytes(&self) -> u64 {
        self.metrics_workspace_bytes
    }
    pub const fn adaptive_view_bytes(&self) -> u64 {
        self.adaptive_view_bytes
    }
    pub const fn total_reserve_bytes(&self) -> u64 {
        self.total_reserve_bytes
    }
    pub const fn pre_view_required_bytes(&self) -> u64 {
        // The constructor checked this sum separately from the post-view reserve.
        self.total_reserve_bytes + self.adaptive_view_bytes
    }
    pub(crate) fn covers_v3(&self, capacity: u64, months: u32, adaptive_rows: u64) -> bool {
        capacity != 0
            && capacity <= self.retained_capacity as u64
            && months != 0
            && months <= self.month_capacity
            && adaptive_rows <= self.adaptive_view_rows as u64
    }
}

fn hip_parent_extents_v1(rows: usize, features: usize) -> Result<[usize; 9], HipRuntimeErrorV1> {
    if rows == 0 || rows > i32::MAX as usize || features == 0 || features > i32::MAX as usize {
        return Err(invalid(
            "HIP population parent dimensions must fit positive native signed indices",
        ));
    }
    let lane = checked_mul(rows, 8)?;
    let cells = checked_mul(rows, features)?;
    let values = checked_mul(cells, 8)?;
    let validity = checked_mul(cells / 8 + usize::from(cells % 8 != 0), 4)?;
    let smc = checked_mul(rows, crate::SMC_SLOTS)?;
    Ok([lane, lane, lane, values, validity, lane, lane, lane, smc])
}

fn hip_physical_parent_binding_sha256_v1(
    identity: &HipRuntimeIdentityV1,
    raw: &RawHipResidentFeatureStoreBindV1,
    extents: [usize; 9],
    build: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.hip-population.physical-parent-binding.v1");
    hash.update(raw.abi_version.to_le_bytes());
    hash.update(raw.backend_kind.to_le_bytes());
    hash.update(raw.lease_id.to_le_bytes());
    hash.update(identity.device_ordinal().to_le_bytes());
    hash.update(identity.device_uuid());
    hash.update(identity.stream_id().to_le_bytes());
    // Parent-module private handles are genuine lease facts, never caller inputs.
    hash.update(identity.stream_handle.to_le_bytes());
    hash.update(identity.current_pool_handle.to_le_bytes());
    hash.update(identity.default_pool_handle.to_le_bytes());
    hash.update(identity.runtime_version().to_le_bytes());
    hash.update(identity.driver_version().to_le_bytes());
    hash.update(identity.warp_size().to_le_bytes());
    hash.update(identity.total_memory_bytes().to_le_bytes());
    hash.update((identity.architecture().len() as u64).to_le_bytes());
    hash.update(identity.architecture().as_bytes());
    hash.update(raw.row_count.to_le_bytes());
    hash.update(raw.feature_count.to_le_bytes());
    hash.update(raw.smc_slots.to_le_bytes());
    hash.update(raw.allocator_reserve_bytes.to_le_bytes());
    hash.update(raw.canonical_content_merkle);
    hash.update(build);
    for (key, extent) in raw.buffer_keys.iter().zip(extents) {
        hash.update(key.to_le_bytes());
        hash.update((extent as u64).to_le_bytes());
    }
    hash.finalize().into()
}

impl<'timestamp, 'lease> SealedHipFeatureStoreV1<'timestamp, 'lease> {
    /// Bind the actual immutable packed store and six typed parent lanes to the
    /// shared native population evaluator, without copying parent data to host.
    /// Native pins all nine initialized same-lease buffers until checked close.
    ///
    /// This proves physical binding only. Data must retain its separate genuine
    /// OHLCV/SMC provenance and selection/holdout/config authority; arbitrary
    /// uploaded lanes do not become canonical Data through this method.
    pub fn bind_population_parent_for_search_v3<'parent>(
        &'parent self,
        close: &'parent HipDeviceBufferV1<'lease>,
        high: &'parent HipDeviceBufferV1<'lease>,
        low: &'parent HipDeviceBufferV1<'lease>,
        months: &'parent HipDeviceBufferV1<'lease>,
        days: &'parent HipDeviceBufferV1<'lease>,
        smc: &'parent HipDeviceBufferV1<'lease>,
        budget: HipSearchEvaluationBudgetV3,
    ) -> Result<HipPopulationParentV1<'parent, 'timestamp, 'lease>, HipPopulationParentErrorV1>
    {
        if budget.adaptive_view_rows > self.plan.rows() {
            return Err(invalid("HIP Search adaptive budget exceeds physical parent").into());
        }
        // Parent bind itself retains one gap byte per parent row. Existing
        // packed/source buffers are already live, so they are not charged again.
        let required = budget
            .pre_view_required_bytes()
            .checked_add(self.plan.rows() as u64)
            .ok_or_else(|| invalid("HIP Search parent reserve overflows"))?;
        let observed = self.lease.revalidate()?;
        if required > observed.free_memory_bytes() {
            return Err(invalid(
                "HIP Search parent reserve exceeds current same-lease free memory",
            )
            .into());
        }
        self.bind_population_parent_impl_v3(
            close,
            high,
            low,
            months,
            days,
            smc,
            budget.total_reserve_bytes,
            Some(budget),
        )
    }

    pub fn bind_population_parent_v1<'parent>(
        &'parent self,
        close: &'parent HipDeviceBufferV1<'lease>,
        high: &'parent HipDeviceBufferV1<'lease>,
        low: &'parent HipDeviceBufferV1<'lease>,
        months: &'parent HipDeviceBufferV1<'lease>,
        days: &'parent HipDeviceBufferV1<'lease>,
        smc: &'parent HipDeviceBufferV1<'lease>,
        allocator_reserve_bytes: u64,
    ) -> Result<HipPopulationParentV1<'parent, 'timestamp, 'lease>, HipPopulationParentErrorV1>
    {
        self.bind_population_parent_impl_v3(
            close,
            high,
            low,
            months,
            days,
            smc,
            allocator_reserve_bytes,
            None,
        )
    }

    fn bind_population_parent_impl_v3<'parent>(
        &'parent self,
        close: &'parent HipDeviceBufferV1<'lease>,
        high: &'parent HipDeviceBufferV1<'lease>,
        low: &'parent HipDeviceBufferV1<'lease>,
        months: &'parent HipDeviceBufferV1<'lease>,
        days: &'parent HipDeviceBufferV1<'lease>,
        smc: &'parent HipDeviceBufferV1<'lease>,
        allocator_reserve_bytes: u64,
        search_budget: Option<HipSearchEvaluationBudgetV3>,
    ) -> Result<HipPopulationParentV1<'parent, 'timestamp, 'lease>, HipPopulationParentErrorV1>
    {
        if allocator_reserve_bytes == 0 {
            return Err(invalid("HIP population allocator reserve must be nonzero").into());
        }
        let extents = hip_parent_extents_v1(self.plan.rows(), self.plan.columns())?;
        let keys = self.lease.checked_buffer_keys_v1(
            [
                close,
                high,
                low,
                &self.values,
                &self.validity,
                months,
                days,
                self.timestamps,
                smc,
            ],
            extents,
        )?;
        let manifest = super::hip_native_build_manifest_v1()
            .ok_or_else(|| invalid("HIP native build manifest is missing"))?;
        let expected_build: [u8; 32] = Sha256::digest(manifest.as_bytes()).into();
        // SAFETY: the linked accessor returns a static host digest, not device data.
        let linked = unsafe { neoethos_hip_native_build_manifest_sha256_v1() };
        if linked.is_null() {
            return Err(invalid("HIP linked build digest is missing").into());
        }
        let linked = unsafe { std::slice::from_raw_parts(linked, 32) };
        if linked != expected_build {
            return Err(invalid("HIP Rust and native build manifests differ").into());
        }
        let identity = self.lease.identity();
        let mut raw = RawHipResidentFeatureStoreBindV1 {
            abi_version: 1,
            backend_kind: 2,
            lease_id: identity.lease_id(),
            row_count: self.plan.rows() as u64,
            feature_count: self.plan.columns() as u32,
            smc_slots: crate::SMC_SLOTS as u32,
            buffer_keys: keys,
            allocator_reserve_bytes,
            admission_identity_sha256: [0; 32],
            canonical_content_merkle: self.canonical_content_merkle_sha256(),
            run_stream_process_token: [0; 32],
        };
        // The native field name is historical. Its actual producer here seals
        // only physical device binding, not a fabricated Data admission receipt.
        let binding =
            hip_physical_parent_binding_sha256_v1(identity, &raw, extents, expected_build);
        raw.admission_identity_sha256 = binding;
        let mut process = Sha256::new();
        process.update(b"neoethos.hip-population.physical-parent-process.v1");
        process.update(binding);
        raw.run_stream_process_token = process.finalize().into();
        // SAFETY: this returned guard retains the actual lease, physical store
        // and all six extra buffers. It never exposes the bare native session.
        let core = unsafe {
            PopulationSession::bind_hip_physical_parent_v1(
                &raw,
                identity.device_ordinal(),
                expected_build,
            )
        }
        .map_err(|error| {
            self.lease.state.set(super::LeaseStateV1::Quarantined);
            HipPopulationParentErrorV1::Population(error)
        })?;
        let native_session_identity = core.resident_search_native_handle_v2() as usize;
        Ok(HipPopulationParentV1 {
            core,
            store: self,
            _parent_lanes: [close, high, low, months, days, smc],
            physical_binding_sha256: binding,
            native_build_sha256: expected_build,
            native_session_identity,
            search_detached: false,
            search_budget,
        })
    }
}

/// Lifetime-bound physical HIP parent. Only bounded strict metric evaluation is
/// exposed: no raw session, pointer, key, CUDA receipt, or selection authority.
#[must_use]
pub struct HipPopulationParentV1<'parent, 'timestamp, 'lease> {
    core: PopulationSession,
    store: &'parent SealedHipFeatureStoreV1<'timestamp, 'lease>,
    _parent_lanes: [&'parent HipDeviceBufferV1<'lease>; 6],
    physical_binding_sha256: [u8; 32],
    native_build_sha256: [u8; 32],
    native_session_identity: usize,
    search_detached: bool,
    search_budget: Option<HipSearchEvaluationBudgetV3>,
}

/// Private capture of the already-bound physical owner. No raw constructor is
/// exposed and the Data borrow remains outside the detached common Search state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HipSearchParentIdentityV3 {
    pub(crate) runtime: HipRuntimeIdentityV1,
    pub(crate) physical_binding_sha256: [u8; 32],
    pub(crate) content_merkle: [u8; 32],
    pub(crate) native_build_sha256: [u8; 32],
    pub(crate) rows: usize,
    pub(crate) search_budget: Option<HipSearchEvaluationBudgetV3>,
    pub(crate) features: usize,
    native_session_identity: usize,
}
impl HipSearchParentIdentityV3 {
    pub(crate) fn matches_core_v3(&self, core: &PopulationSession) -> bool {
        core.matches_hip_physical_parent_v1(
            self.native_session_identity,
            self.rows,
            self.features,
            self.runtime.device_ordinal(),
            self.physical_binding_sha256,
            self.native_build_sha256,
        )
    }
}

impl std::fmt::Debug for HipPopulationParentV1<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HipPopulationParentV1")
            .field("runtime_identity", self.runtime_identity())
            .field("physical_binding_sha256", &self.physical_binding_sha256)
            .finish_non_exhaustive()
    }
}

/// Immutable metric evidence with explicit HIP physical identity. It is not a
/// financial validation, selection/holdout admission, or GPU parity certificate.
#[derive(Debug)]
pub struct HipPhysicalPopulationMetricsV1 {
    identity: HipRuntimeIdentityV1,
    physical_binding_sha256: [u8; 32],
    metrics: HostPopulationMetricsReceiptV1,
}

impl HipPhysicalPopulationMetricsV1 {
    pub fn runtime_identity(&self) -> &HipRuntimeIdentityV1 {
        &self.identity
    }
    pub const fn physical_binding_sha256(&self) -> [u8; 32] {
        self.physical_binding_sha256
    }
    pub fn metrics(&self) -> &HostPopulationMetricsReceiptV1 {
        &self.metrics
    }
}

impl HipPopulationParentV1<'_, '_, '_> {
    pub fn runtime_identity(&self) -> &HipRuntimeIdentityV1 {
        self.store.runtime_identity()
    }
    pub const fn physical_binding_sha256(&self) -> [u8; 32] {
        self.physical_binding_sha256
    }

    fn retained_search_identity_v3(&self) -> HipSearchParentIdentityV3 {
        HipSearchParentIdentityV3 {
            runtime: self.runtime_identity().clone(),
            physical_binding_sha256: self.physical_binding_sha256,
            content_merkle: self.store.canonical_content_merkle_sha256(),
            native_build_sha256: self.native_build_sha256,
            rows: self.store.plan.rows(),
            search_budget: self.search_budget,
            features: self.store.plan.columns(),
            native_session_identity: self.native_session_identity,
        }
    }

    pub(crate) fn capture_search_identity_v3(
        &mut self,
    ) -> Result<HipSearchParentIdentityV3, HipPopulationParentErrorV1> {
        self.store.lease.require_active()?;
        let identity = self.retained_search_identity_v3();
        if self.search_detached || !identity.matches_core_v3(&self.core) {
            return Err(invalid("HIP Search requires the original idle physical parent").into());
        }
        self.core
            .admit_resident_search_owner_v2(identity.features)?;
        Ok(identity)
    }

    pub(crate) fn take_search_core_v3(
        &mut self,
        expected: &HipSearchParentIdentityV3,
        adaptive_rows: u64,
    ) -> Result<PopulationSession, HipPopulationParentErrorV1> {
        if &self.capture_search_identity_v3()? != expected {
            return Err(invalid("HIP Search parent changed before detachment").into());
        }
        let observed = self.store.lease.revalidate()?;
        let budget = expected
            .search_budget
            .ok_or_else(|| invalid("HIP Search budget is missing"))?;
        let required = adaptive_rows
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_add(budget.total_reserve_bytes))
            .filter(|_| adaptive_rows <= budget.adaptive_view_rows as u64)
            .ok_or_else(|| invalid("HIP Search pre-view extent exceeds its budget"))?;
        if required > observed.free_memory_bytes() {
            return Err(invalid("HIP Search pre-view peak exceeds current free memory").into());
        }
        self.search_detached = true;
        let mut core = std::mem::replace(&mut self.core, PopulationSession::detached_resident_v3());
        core.arm_resident_session_leak_only_v3();
        Ok(core)
    }

    pub(crate) fn restore_search_core_v3(
        &mut self,
        mut core: PopulationSession,
        expected: &HipSearchParentIdentityV3,
    ) -> Result<(), HipPopulationParentErrorV1> {
        let valid = self.search_detached
            && &self.retained_search_identity_v3() == expected
            && expected.matches_core_v3(&core);
        if !valid {
            core.poison_resident_search_owner_v2();
            std::mem::forget(core);
            self.quarantine_search_v3();
            return Err(
                invalid("HIP Search terminal owner differs from the original parent").into(),
            );
        }
        if let Err(error) = self.store.lease.revalidate() {
            core.poison_resident_search_owner_v2();
            std::mem::forget(core);
            self.quarantine_search_v3();
            return Err(error.into());
        }
        core.arm_resident_session_leak_only_v3();
        self.core = core;
        self.search_detached = false;
        Ok(())
    }

    pub(crate) fn quarantine_search_v3(&mut self) {
        self.store.lease.state.set(super::LeaseStateV1::Quarantined);
        self.core.poison_resident_search_owner_v2();
    }

    /// Uses the existing strict whole-cohort evaluator. The caller's checked
    /// parent-local view is not widened or silently replaced with a full view.
    pub fn evaluate_metrics_v1(
        &mut self,
        view: PopulationEvaluationViewV1,
        genes: PopulationGeneView<'_>,
        scenarios: &[ScenarioDescriptor],
        settings: &NeoPopulationSettings,
    ) -> Result<HipPhysicalPopulationMetricsV1, HipPopulationParentErrorV1> {
        // Ordinary native view/upload/evaluation APIs share the kernels but do
        // not acquire Search's identity lease. Requery the real owning device,
        // stream, UUID and default pool before touching those APIs. This memory
        // observation is not a reservation or a combined Search admission.
        self.store.lease.revalidate()?;
        let result = self
            .core
            .evaluate_hip_physical_parent_v1(view, genes, scenarios, settings);
        match result {
            Ok(metrics) => {
                // Publication also requires the same still-live native identity
                // after the actual terminal synchronization and metric readback.
                self.store.lease.revalidate()?;
                Ok(HipPhysicalPopulationMetricsV1 {
                    identity: self.runtime_identity().clone(),
                    physical_binding_sha256: self.physical_binding_sha256,
                    metrics,
                })
            }
            Err(error) => {
                if self.core.hip_physical_parent_is_poisoned_v1() {
                    self.store.lease.state.set(super::LeaseStateV1::Quarantined);
                }
                Err(error.into())
            }
        }
    }

    fn close(&mut self) -> Result<(), HipPopulationParentErrorV1> {
        if self.search_detached {
            self.quarantine_search_v3();
        }
        self.core
            .close_hip_physical_parent_v1(self.store.lease.require_active().is_ok())
            .map_err(|error| {
                self.store.lease.state.set(super::LeaseStateV1::Quarantined);
                error.into()
            })
    }

    /// Completes checked native deletion. An ambiguous failure quarantines the
    /// lease and retains native pins; Drop cannot retry the disarmed handle.
    pub fn try_close(mut self) -> Result<(), HipPopulationParentErrorV1> {
        self.close()
    }
}

impl Drop for HipPopulationParentV1<'_, '_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("HIP physical population owner retained: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_search_budget_preserves_headroom_and_actual_month_workspace() {
        let budget = HipSearchEvaluationBudgetV3::checked_v3(1024, 5, 240, 120).unwrap();
        let metrics =
            crate::PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(5, 240).unwrap();
        assert_eq!(
            budget.metrics_workspace_bytes(),
            metrics.total_device_bytes()
        );
        assert_eq!(budget.adaptive_view_bytes(), 960);
        assert_eq!(budget.metrics_workspace_bytes(), 20_000);
        assert_eq!(budget.total_reserve_bytes(), 21_024);
        assert_eq!(budget.pre_view_required_bytes(), 21_984);
        assert!(budget.covers_v3(5, 240, 120));
        assert!(budget.covers_v3(3, 120, 0));
        for (capacity, months, rows) in [(6, 240, 120), (5, 241, 120), (5, 240, 121), (0, 240, 0)] {
            assert!(!budget.covers_v3(capacity, months, rows));
        }
        assert!(HipSearchEvaluationBudgetV3::checked_v3(u64::MAX, 5, 240, 0).is_err());
        assert!(HipSearchEvaluationBudgetV3::checked_v3(1024, 1, 1, 100).is_err());
        assert!(HipSearchEvaluationBudgetV3::checked_v3(0, 1, 1, 0).is_err());
    }

    #[test]
    fn hip_physical_parent_abi_and_exact_nine_buffer_extents() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawHipResidentFeatureStoreBindV1>(), 208);
        assert_eq!(align_of::<RawHipResidentFeatureStoreBindV1>(), 8);
        assert_eq!(offset_of!(RawHipResidentFeatureStoreBindV1, lease_id), 8);
        assert_eq!(
            offset_of!(RawHipResidentFeatureStoreBindV1, buffer_keys),
            32
        );
        assert_eq!(
            offset_of!(RawHipResidentFeatureStoreBindV1, allocator_reserve_bytes),
            104
        );
        assert_eq!(
            offset_of!(RawHipResidentFeatureStoreBindV1, admission_identity_sha256),
            112
        );
        assert_eq!(
            hip_parent_extents_v1(1, 1).unwrap(),
            [8, 8, 8, 8, 4, 8, 8, 8, 11]
        );
        assert_eq!(
            hip_parent_extents_v1(23, 5).unwrap(),
            [184, 184, 184, 920, 60, 184, 184, 184, 253]
        );
        for (rows, features) in [
            (0, 1),
            (1, 0),
            (i32::MAX as usize + 1, 1),
            (1, i32::MAX as usize + 1),
            (i32::MAX as usize, i32::MAX as usize),
        ] {
            assert!(hip_parent_extents_v1(rows, features).is_err());
        }
    }

    #[test]
    fn hip_physical_binding_is_backend_owner_content_and_build_specific() {
        // Pure hash test data only: no native owner or Data authority is minted.
        let identity = HipRuntimeIdentityV1 {
            lease_id: std::num::NonZeroU64::new(7).unwrap(),
            device_ordinal: 0,
            uuid: [8; 16],
            architecture: "gfx942".into(),
            runtime_version: 70_200_000,
            driver_version: 70_200_000,
            warp_size: 64,
            stream_id: 11,
            stream_handle: 13,
            total_memory_bytes: 1 << 30,
            current_pool_handle: 17,
            default_pool_handle: 17,
        };
        let raw = RawHipResidentFeatureStoreBindV1 {
            abi_version: 1,
            backend_kind: 2,
            lease_id: 7,
            row_count: 23,
            feature_count: 5,
            smc_slots: 11,
            buffer_keys: std::array::from_fn(|i| i as u64 + 1),
            allocator_reserve_bytes: 4096,
            admission_identity_sha256: [0; 32],
            canonical_content_merkle: [21; 32],
            run_stream_process_token: [0; 32],
        };
        let extents = hip_parent_extents_v1(23, 5).unwrap();
        let build = [22; 32];
        let digest = hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, build);
        for change in 0..9 {
            let mut changed = identity.clone();
            match change {
                0 => changed.device_ordinal += 1,
                1 => changed.uuid[0] ^= 1,
                2 => changed.stream_id += 1,
                3 => changed.stream_handle += 1,
                4 => changed.current_pool_handle += 1,
                5 => changed.default_pool_handle += 1,
                6 => changed.architecture.push_str(":sramecc+"),
                7 => changed.warp_size = 32,
                _ => changed.runtime_version += 1,
            }
            assert_ne!(
                digest,
                hip_physical_parent_binding_sha256_v1(&changed, &raw, extents, build)
            );
        }
        let mut raw = raw;
        for slot in 0..9 {
            raw.buffer_keys[slot] += 100;
            assert_ne!(
                digest,
                hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, build)
            );
            raw.buffer_keys[slot] -= 100;
        }
        raw.lease_id += 1;
        assert_ne!(
            digest,
            hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, build)
        );
        raw.lease_id -= 1;
        raw.allocator_reserve_bytes += 1;
        assert_ne!(
            digest,
            hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, build)
        );
        raw.allocator_reserve_bytes -= 1;
        raw.canonical_content_merkle[0] ^= 1;
        assert_ne!(
            digest,
            hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, build)
        );
        raw.canonical_content_merkle[0] ^= 1;
        assert_ne!(
            digest,
            hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, [23; 32])
        );
        raw.backend_kind = 0;
        assert_ne!(
            digest,
            hip_physical_parent_binding_sha256_v1(&identity, &raw, extents, build)
        );
    }

    fn names() -> Vec<String> {
        vec!["a".into(), "β".into(), "xyz".into()]
    }

    fn receipt(plan: &HipFeatureStorePlanV1) -> RawHipFeatureStoreReceiptV4 {
        RawHipFeatureStoreReceiptV4 {
            abi_version: 4,
            backend_kind: 2,
            rows: plan.rows as u64,
            columns: plan.columns() as u64,
            value_bytes: plan.values_bytes() as u64,
            validity_bytes: plan.validity_bytes() as u64,
            transient_device_bytes: plan.transient_device_bytes() as u64,
            metadata_upload_bytes: plan.metadata_upload_bytes() as u64,
            control: 0,
            readback_count: 2,
            readback_bytes: 36,
            root: [7; 32],
            normalization_training_start: 0,
            normalization_training_end: 0,
            fit_word_count: 0,
            fit_metadata_digest: [0; 32],
        }
    }

    #[test]
    fn hip_feature_store_native_layout_and_exact_memory_are_checked() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawHipFeatureColumnV1>(), 32);
        assert_eq!(offset_of!(RawHipFeatureColumnV1, validity_key), 16);
        assert_eq!(size_of::<RawHipFeatureStoreReceiptV4>(), 160);
        assert_eq!(align_of::<RawHipFeatureStoreReceiptV4>(), 8);
        assert_eq!(offset_of!(RawHipFeatureStoreReceiptV4, control), 56);
        assert_eq!(offset_of!(RawHipFeatureStoreReceiptV4, root), 72);
        assert_eq!(
            offset_of!(RawHipFeatureStoreReceiptV4, normalization_training_start),
            104
        );
        assert_eq!(
            offset_of!(RawHipFeatureStoreReceiptV4, fit_metadata_digest),
            128
        );
        assert_eq!(size_of::<RawHipFeatureNormalizationV3>(), 32);
        assert_eq!(offset_of!(RawHipFeatureNormalizationV3, column_modes), 16);
        let plan = HipFeatureStorePlanV1::preflight(1, &names()).unwrap();
        assert_eq!(plan.name_offsets, [0, 1, 3, 6]);
        assert_eq!(plan.name_bytes, "aβxyz".as_bytes());
        assert_eq!(plan.values_bytes(), 24);
        assert_eq!(plan.logical_validity_bytes(), 2);
        assert_eq!(plan.validity_bytes(), 4);
        assert_eq!(plan.metadata_upload_bytes(), 96 + 32 + 6);
        assert_eq!(plan.merkle_leaf_count(), 4);
        assert_eq!(plan.transient_device_bytes(), 134 + 256 + 36);
        assert_eq!(plan.incremental_peak_device_bytes(), 24 + 4 + 426);
        for rows in [4095usize, 4096, 4097] {
            let plan = HipFeatureStorePlanV1::preflight(rows, &names()).unwrap();
            assert_eq!(plan.merkle_leaf_count(), if rows <= 4096 { 4 } else { 8 });
            assert_eq!(plan.validity_bytes(), (rows * 3).div_ceil(8) * 4);
        }
    }

    #[test]
    fn hip_feature_store_rejects_names_offsets_and_overflow_before_device_work() {
        for names in [
            vec![],
            vec!["".into()],
            vec![" \t".into()],
            vec!["a".into(), "a".into()],
        ] {
            assert!(HipFeatureStorePlanV1::preflight(1, &names).is_err());
        }
        assert!(HipFeatureStorePlanV1::preflight(0, &names()).is_err());
        assert!(extents_v1(usize::MAX, 1, 1).is_err());
        assert!(extents_v1(1, usize::MAX, 1).is_err());
        assert!(extents_v1(1, 1, usize::MAX).is_err());
        assert!(validate_source_extent_v1(5, 120, 10, 15, 10).is_ok());
        for (vbytes, voffset, qbytes, qoffset) in [
            (119, 10, 15, 10),
            (120, 11, 15, 10),
            (120, 10, 14, 10),
            (120, 10, 15, 11),
            (120, usize::MAX, 15, 0),
            (120, 0, 15, usize::MAX),
        ] {
            assert!(validate_source_extent_v1(5, vbytes, voffset, qbytes, qoffset).is_err());
        }
        assert!(require_headroom_v1(100, 20, 120).is_ok());
        assert!(require_headroom_v1(100, 20, 119).is_err());
        assert!(require_headroom_v1(u64::MAX, 1, u64::MAX).is_err());
    }

    #[test]
    fn hip_feature_store_rejects_every_malformed_seal_field() {
        let plan = HipFeatureStorePlanV1::preflight(17, &names()).unwrap();
        let valid = receipt(&plan);
        assert_eq!(
            validate_receipt_v1(&plan, valid, &[])
                .unwrap()
                .readback_bytes(),
            36
        );
        let mutations: &[fn(&mut RawHipFeatureStoreReceiptV4)] = &[
            |r| r.abi_version = 2,
            |r| r.backend_kind = 1,
            |r| r.rows += 1,
            |r| r.columns -= 1,
            |r| r.value_bytes -= 1,
            |r| r.validity_bytes -= 1,
            |r| r.transient_device_bytes -= 1,
            |r| r.metadata_upload_bytes += 1,
            |r| r.control = 1,
            |r| r.readback_count = 1,
            |r| r.readback_bytes = 32,
            |r| r.root = [0; 32],
            |r| r.normalization_training_start = 1,
            |r| r.normalization_training_end = 1,
            |r| r.fit_word_count = 6,
            |r| r.fit_metadata_digest = [1; 32],
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut changed = valid;
            mutate(&mut changed);
            assert!(
                validate_receipt_v1(&plan, changed, &[]).is_err(),
                "field mutation {index}"
            );
        }
    }

    #[test]
    fn hip_policy3_normalization_charges_all_live_bytes_and_verifies_actual_fit_transport() {
        use SearchNormalizationColumnModeV3::{Binary, Robust, SignedState};
        let plain = HipFeatureStorePlanV1::preflight(100, &names()).unwrap();
        let request =
            HipFeatureNormalizationV3::preflight(100, 0..80, vec![Robust, Binary, SignedState])
                .unwrap();
        assert_eq!(request.scratch_bytes(), 3 * 128 * 8);
        assert_eq!(request.fit_metadata_bytes(), 3 * 48);
        let plan = plain.clone().with_normalization(request.clone()).unwrap();
        assert_eq!(
            plan.transient_device_bytes(),
            plain.transient_device_bytes() + 3 * 128 * 8 + 3 * 48
        );
        assert_eq!(
            plan.incremental_peak_device_bytes(),
            plain.incremental_peak_device_bytes() + 3 * 128 * 8 + 3 * 48
        );
        assert_eq!(plan.metadata_upload_bytes(), plain.metadata_upload_bytes());
        assert!(plan.clone().with_normalization(request.clone()).is_err());
        assert!(
            HipFeatureStorePlanV1::preflight(99, &names())
                .unwrap()
                .with_normalization(request)
                .is_err()
        );
        for (rows, range, modes) in [
            (100, 1..81, vec![Robust]),
            (100, 0..79, vec![Robust]),
            (79, 0..63, vec![Robust]),
            (100, 0..80, vec![]),
        ] {
            assert!(HipFeatureNormalizationV3::preflight(rows, range, modes).is_err());
        }
        let fits = [0, 80, 0, 1f64.to_bits(), 80, 0].repeat(3);
        let mut raw = receipt(&plan);
        raw.readback_count = 5;
        raw.readback_bytes = 72 + 3 * 48;
        raw.normalization_training_end = 80;
        raw.fit_word_count = 18;
        raw.fit_metadata_digest = resident_normalization_fit_metadata_sha256_v3(&fits);
        assert_eq!(
            validate_receipt_v1(&plan, raw, &fits)
                .unwrap()
                .readback_bytes(),
            216
        );
        assert!(validate_receipt_v1(&plan, raw, &fits[..17]).is_err());
        let mut changed_fit = fits.clone();
        changed_fit[2] ^= 1;
        assert!(validate_receipt_v1(&plan, raw, &changed_fit).is_err());
        for mutate in [
            (|r: &mut RawHipFeatureStoreReceiptV4| r.abi_version = 3)
                as fn(&mut RawHipFeatureStoreReceiptV4),
            |r| r.readback_count = 4,
            |r| r.readback_bytes -= 4,
            |r| r.normalization_training_end = 79,
            |r| r.fit_word_count = 12,
            |r| r.fit_metadata_digest[31] ^= 1,
        ] {
            let mut changed = raw;
            mutate(&mut changed);
            assert!(validate_receipt_v1(&plan, changed, &fits).is_err());
        }
    }

    #[test]
    #[cfg(feature = "hip-device-fixtures")]
    fn hip_policy3_actual_normalization_known_answers_gates_and_heldout_refusal() {
        use SearchNormalizationColumnModeV3::{Binary, Robust, SignedContinuous, SignedState};
        use neoethos_gpu_contracts::resident_feature_store_v3::canonical_feature_merkle_sha256_host_oracle_v3;
        assert_eq!(
            std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
            Ok("1"),
            "mandatory real HIP device fixture; no skip"
        );
        let ordinal = std::env::var("NEOETHOS_HIP_DEVICE")
            .ok()
            .map(|value| value.parse::<u32>().expect("invalid HIP ordinal"))
            .unwrap_or(0);
        let lease = HipRunLeaseV1::acquire(ordinal).expect("real HIP owner required");
        let rows = 100usize;
        let columns = 65usize;
        let names: Vec<String> = (0..columns).map(|c| format!("column_{c}")).collect();
        let mut modes = vec![Robust; columns];
        modes[2] = Binary;
        modes[3] = SignedState;
        modes[4] = SignedContinuous;
        modes[63] = Binary;
        modes[64] = SignedState;
        let mut source = vec![0u64; rows * columns];
        let mut validity = vec![0u8; rows * columns];
        let mut expected = vec![0u64; rows * columns];
        let mut expected_u4 = vec![0u8; (rows * columns).div_ceil(2)];
        // Literal independent median/MAD and fallback checkpoints from the
        // existing semantic-v2 oracle. No CPU normalizer is called here.
        let mad_inputs = [-2f64, -1., 0., 1., 2.];
        let mad_outputs = [
            0xbff5_956d_a52c_ff6a,
            0xbfe5_956d_a52c_ff6a,
            0,
            0x3fe5_956d_a52c_ff6a,
            0x3ff5_956d_a52c_ff6a,
        ];
        for c in 0..columns {
            for r in 0..rows {
                let (value, output) = match modes[c] {
                    Binary => {
                        if r == 0 {
                            (-0.0, (-0.0f64).to_bits())
                        } else {
                            (0.0, 0)
                        }
                    }
                    SignedState => (1.0, 1f64.to_bits()),
                    SignedContinuous => {
                        if r % 2 == 0 {
                            (-20., (-10f64).to_bits())
                        } else {
                            (20., 10f64.to_bits())
                        }
                    }
                    Robust if c == 1 => {
                        if r % 4 == 3 {
                            (1., 0x4002_79a7_4590_331d)
                        } else {
                            (0., 0)
                        }
                    }
                    Robust => (mad_inputs[r % 5], mad_outputs[r % 5]),
                };
                source[c * rows + r] = value.to_bits();
                expected[r * columns + c] = output;
            }
        }
        source[99] = 1000f64.to_bits();
        expected[99 * columns] = 10f64.to_bits();
        source[3 * rows + 7] = 0x7ff8_0000_0000_0042;
        validity[3 * rows + 7] = 2;
        let hole = 7 * columns + 3;
        expected[hole] = f64::NAN.to_bits();
        expected_u4[hole / 2] |= 2 << ((hole & 1) * 4);
        let timestamps: Vec<i64> = (0..rows)
            .map(|r| 1_700_000_000_000 + r as i64 * 60_000)
            .collect();
        let ts = lease
            .upload_bytes(
                &timestamps
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let values = lease
            .upload_bytes(
                &source
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let valid = lease.upload_bytes(&validity).unwrap();
        let spans: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(c, name)| HipFeatureColumnV1::new(name, &values, c * rows, &valid, c * rows))
            .collect();
        let request = HipFeatureNormalizationV3::preflight(rows, 0..80, modes.clone()).unwrap();
        let plan = HipFeatureStorePlanV1::preflight(rows, &names)
            .unwrap()
            .with_normalization(request)
            .unwrap();
        let store = plan.clone().pack_and_seal(&lease, &ts, &spans, 0).unwrap();
        assert_eq!(
            store.values.read_bytes().unwrap(),
            expected
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        );
        let actual_u4 = store.validity.read_bytes().unwrap();
        assert_eq!(&actual_u4[..expected_u4.len()], expected_u4);
        assert!(actual_u4[expected_u4.len()..].iter().all(|&v| v == 0));
        let fits = store.normalization_fit_words().unwrap();
        for c in 0..columns {
            let (scale, count) = if modes[c] != Robust {
                (1f64.to_bits(), if c == 3 { 79 } else { 80 })
            } else if c == 1 {
                (0x3fdb_b67a_e858_4caa, 80)
            } else {
                (0x3ff7_b8ba_c710_cb29, 80)
            };
            assert_eq!(
                &fits[c * 6..c * 6 + 6],
                &[0, 80, 0, scale, count, 0],
                "fit column {c}"
            );
        }
        assert_eq!(store.receipt().readback_count(), 5);
        assert_eq!(store.receipt().readback_bytes(), 72 + 48 * columns as u64);
        assert_eq!(
            store.canonical_content_merkle_sha256(),
            canonical_feature_merkle_sha256_host_oracle_v3(
                &timestamps,
                &names,
                &expected,
                &expected_u4
            )
            .unwrap()
        );
        store.try_close().unwrap();
        drop(spans);
        // Validity-valid binary state2 in held-out row99 must fail the same
        // actual GPU operation, even though all training observations are0.
        source[2 * rows + 99] = 2f64.to_bits();
        let malformed = lease
            .upload_bytes(
                &source
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let bad: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(c, name)| HipFeatureColumnV1::new(name, &malformed, c * rows, &valid, c * rows))
            .collect();
        assert!(matches!(
            plan.pack_and_seal(&lease, &ts, &bad, 0),
            Err(HipRuntimeErrorV1::Native { status: -8, .. })
        ));
        assert!(
            !lease.is_quarantined(),
            "completed semantic refusal is not a device fault"
        );
        drop(bad);
        malformed.try_close().unwrap();
        valid.try_close().unwrap();
        values.try_close().unwrap();
        ts.try_close().unwrap();
        lease.try_close().unwrap();
    }

    #[test]
    #[cfg(feature = "hip-device-fixtures")]
    fn hip_feature_store_actual_pack_merkle_exact_bits_and_semantic_refusal() {
        use neoethos_gpu_contracts::resident_feature_store_v3::canonical_feature_merkle_sha256_host_oracle_v3;
        assert_eq!(
            std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
            Ok("1"),
            "mandatory real HIP device fixture; no silent skip"
        );
        let ordinal = std::env::var("NEOETHOS_HIP_DEVICE")
            .ok()
            .map(|v| v.parse::<u32>().expect("invalid HIP ordinal"))
            .unwrap_or(0);
        let lease = HipRunLeaseV1::acquire(ordinal).expect("real HIP owner required");
        for rows in [1usize, 23, 4095, 4096, 4097] {
            let names = names();
            let patterns = [
                0u64,
                1,
                (-0.0f64).to_bits(),
                1.0f64.to_bits(),
                f64::INFINITY.to_bits(),
                f64::NEG_INFINITY.to_bits(),
                0x7ff8_0000_0000_0042,
            ];
            // Prefix offsets are deliberately nonzero and differ between value
            // elements and validity bytes, so a key-only/offset-blind pack fails.
            let mut source = vec![0u64; rows * 3 + 7];
            let mut source_validity = vec![0u8; rows * 3 + 5];
            let mut expected_bits = vec![0u64; rows * 3];
            let mut expected_u4 = vec![0u8; (rows * 3).div_ceil(2)];
            for c in 0..3 {
                for r in 0..rows {
                    let cell = r * 3 + c;
                    let bits = patterns[(r + c) % patterns.len()];
                    let code = ((r + 3 * c) % 10) as u8;
                    source[7 + c * rows + r] = bits;
                    source_validity[5 + c * rows + r] = code;
                    expected_bits[cell] = bits;
                    expected_u4[cell / 2] |= code << ((cell & 1) * 4);
                }
            }
            let timestamps: Vec<i64> = (0..rows)
                .map(|r| 1_700_000_000_000 + r as i64 * 60_000)
                .collect();
            let ts_bytes: Vec<u8> = timestamps.iter().flat_map(|v| v.to_le_bytes()).collect();
            let value_bytes: Vec<u8> = source.iter().flat_map(|v| v.to_le_bytes()).collect();
            let ts = lease.upload_bytes(&ts_bytes).unwrap();
            let values = lease.upload_bytes(&value_bytes).unwrap();
            let validity = lease.upload_bytes(&source_validity).unwrap();
            let columns: Vec<_> = names
                .iter()
                .enumerate()
                .map(|(c, name)| {
                    HipFeatureColumnV1::new(name, &values, 7 + c * rows, &validity, 5 + c * rows)
                })
                .collect();
            let store = HipFeatureStorePlanV1::preflight(rows, &names)
                .unwrap()
                .pack_and_seal(&lease, &ts, &columns, 0)
                .unwrap();
            let actual_values = store.values.read_bytes().unwrap();
            let actual_u4 = store.validity.read_bytes().unwrap();
            let expected_bytes: Vec<u8> =
                expected_bits.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(actual_values, expected_bytes);
            assert_eq!(&actual_u4[..expected_u4.len()], expected_u4.as_slice());
            assert!(actual_u4[expected_u4.len()..].iter().all(|&v| v == 0));
            assert_eq!(
                store.canonical_content_merkle_sha256(),
                canonical_feature_merkle_sha256_host_oracle_v3(
                    &timestamps,
                    &names,
                    &expected_bits,
                    &expected_u4
                )
                .unwrap()
            );
            assert_eq!(store.receipt().readback_count(), 2);
            assert_eq!(store.receipt().readback_bytes(), 36);
            if rows == 23 {
                // A 23-row timestamp buffer is exactly the 184-byte value
                // output of a one-row Session producer. All shape checks pass;
                // only the native write seal must reject this safe-API alias.
                let lane = lease.upload_bytes(&1.0f64.to_le_bytes()).unwrap();
                let clock = lease.upload_bytes(&ts_bytes[..8]).unwrap();
                let session_validity = lease.allocate_bytes(23).unwrap();
                assert!(matches!(
                    lease.launch_session_f64_v2(
                        1,
                        [&lane, &lane, &lane, &lane, &lane, &clock],
                        &ts,
                        &session_validity
                    ),
                    Err(HipRuntimeErrorV1::Native { status: -1, .. })
                ));
                assert_eq!(ts.read_bytes().unwrap(), ts_bytes);
                assert!(!lease.is_quarantined());
                session_validity.try_close().unwrap();
                lane.try_close().unwrap();
                clock.try_close().unwrap();
            }
            store.try_close().unwrap();
            drop(columns);
            // A genuine completed invalid validity verdict must never become a
            // carrier; it is not converted to a valid low nibble or CPU fallback.
            source_validity[5] = 10;
            let malformed = lease.upload_bytes(&source_validity).unwrap();
            let bad: Vec<_> = names
                .iter()
                .enumerate()
                .map(|(c, name)| {
                    HipFeatureColumnV1::new(name, &values, 7 + c * rows, &malformed, 5 + c * rows)
                })
                .collect();
            assert!(
                HipFeatureStorePlanV1::preflight(rows, &names)
                    .unwrap()
                    .pack_and_seal(&lease, &ts, &bad, 0)
                    .is_err()
            );
            assert!(
                !lease.is_quarantined(),
                "completed semantic rejection is not a runtime fault"
            );
            drop(bad);
            malformed.try_close().unwrap();
            validity.try_close().unwrap();
            values.try_close().unwrap();
            ts.try_close().unwrap();
        }
        lease.try_close().unwrap();
    }
}
