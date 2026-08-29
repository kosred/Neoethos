//! Node-name-independent identity for one exact pinned canonical source set.
//!
//! CPU and resident feature graphs use different source-node vocabularies.
//! This projection deliberately excludes those graph-local names while
//! retaining every immutable generation, manifest, content and consumed-row
//! fact needed to bind a financial-value authority before device allocation.

use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_feature_contracts::SourceArtifactBindingV1;
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};

use super::pinned_canonical_series_v1::MaterializedPinnedResidentCanonicalSourcesV1;

pub const CANONICAL_PINNED_SOURCE_PROJECTION_SCHEMA_VERSION_V1: u16 = 1;
const CANONICAL_PINNED_SOURCE_PROJECTION_HASH_DOMAIN_V1: &[u8] =
    b"neoethos.data.canonical-pinned-source-projection.v1\0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalPinnedSourceProjectionErrorV1 {
    detail: String,
}

impl CanonicalPinnedSourceProjectionErrorV1 {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for CanonicalPinnedSourceProjectionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "canonical pinned-source projection failed: {}",
            self.detail
        )
    }
}

impl Error for CanonicalPinnedSourceProjectionErrorV1 {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalPinnedSourceSegmentFactsV1 {
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
}

impl CanonicalPinnedSourceSegmentFactsV1 {
    pub fn checked_new(
        row_start: u64,
        row_end: u64,
        timestamp_start_ms: i64,
        timestamp_end_ms: i64,
    ) -> Result<Self, CanonicalPinnedSourceProjectionErrorV1> {
        if row_start >= row_end {
            return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                "source segment row range is empty or reversed",
            ));
        }
        if timestamp_start_ms > timestamp_end_ms {
            return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                "source segment timestamp range is reversed",
            ));
        }
        Ok(Self {
            row_start,
            row_end,
            timestamp_start_ms,
            timestamp_end_ms,
        })
    }

    pub const fn row_start(&self) -> u64 {
        self.row_start
    }

    pub const fn row_end(&self) -> u64 {
        self.row_end
    }

    pub const fn timestamp_start_ms(&self) -> i64 {
        self.timestamp_start_ms
    }

    pub const fn timestamp_end_ms(&self) -> i64 {
        self.timestamp_end_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalPinnedSourceBindingFactsV1 {
    dataset_identity: CanonicalDatasetIdentity,
    manifest_schema_id: String,
    manifest_sha256: [u8; 32],
    generation_id: String,
    vortex_sha256: [u8; 32],
    bar_timestamp_convention: BarTimestampConvention,
    segments: Vec<CanonicalPinnedSourceSegmentFactsV1>,
}

impl CanonicalPinnedSourceBindingFactsV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn checked_new(
        dataset_identity: CanonicalDatasetIdentity,
        manifest_schema_id: impl Into<String>,
        manifest_sha256: [u8; 32],
        generation_id: impl Into<String>,
        vortex_sha256: [u8; 32],
        bar_timestamp_convention: BarTimestampConvention,
        mut segments: Vec<CanonicalPinnedSourceSegmentFactsV1>,
    ) -> Result<Self, CanonicalPinnedSourceProjectionErrorV1> {
        let manifest_schema_id = manifest_schema_id.into();
        let generation_id = generation_id.into();
        if manifest_schema_id.trim().is_empty()
            || manifest_schema_id.trim() != manifest_schema_id
            || generation_id.trim().is_empty()
            || generation_id.trim() != generation_id
            || manifest_sha256 == [0; 32]
            || vortex_sha256 == [0; 32]
            || dataset_identity.bar_timestamp_convention() != bar_timestamp_convention
            || segments.is_empty()
        {
            return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                "source binding has empty, zero, noncanonical, or convention-mismatched facts",
            ));
        }
        segments.sort_by_key(|segment| (segment.row_start, segment.row_end));
        for pair in segments.windows(2) {
            if pair[0].row_end > pair[1].row_start
                || pair[0].timestamp_end_ms >= pair[1].timestamp_start_ms
            {
                return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                    "source binding segments overlap or are timestamp-disordered",
                ));
            }
        }
        Ok(Self {
            dataset_identity,
            manifest_schema_id,
            manifest_sha256,
            generation_id,
            vortex_sha256,
            bar_timestamp_convention,
            segments,
        })
    }

    pub fn checked_from_source_artifact_binding_v1(
        binding: &SourceArtifactBindingV1,
    ) -> Result<Self, CanonicalPinnedSourceProjectionErrorV1> {
        let segments = binding
            .segments()
            .iter()
            .map(|segment| {
                CanonicalPinnedSourceSegmentFactsV1::checked_new(
                    segment.row_start(),
                    segment.row_end(),
                    segment.timestamp_start_ms(),
                    segment.timestamp_end_ms(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::checked_new(
            binding.dataset_identity().clone(),
            binding.manifest_schema_id(),
            *binding.manifest_hash(),
            binding.generation_id(),
            *binding.vortex_hash(),
            binding.bar_timestamp_convention(),
            segments,
        )
    }

    pub const fn dataset_identity(&self) -> &CanonicalDatasetIdentity {
        &self.dataset_identity
    }

    pub fn manifest_schema_id(&self) -> &str {
        &self.manifest_schema_id
    }

    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }

    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub const fn vortex_sha256(&self) -> [u8; 32] {
        self.vortex_sha256
    }

    pub const fn bar_timestamp_convention(&self) -> BarTimestampConvention {
        self.bar_timestamp_convention
    }

    pub fn segments(&self) -> &[CanonicalPinnedSourceSegmentFactsV1] {
        &self.segments
    }
}

/// Exact source-generation projection carried by the move-only prepared Data
/// token. The identity never includes a graph-local source-node name and never
/// aliases CPU feature-content receipts to resident GPU content receipts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalPinnedSourceProjectionV1 {
    schema_version: u16,
    anchor_dataset_identity: CanonicalDatasetIdentity,
    base_timeframe: CanonicalTimeframe,
    parent_row_count: u64,
    bindings: Vec<CanonicalPinnedSourceBindingFactsV1>,
    identity_sha256: [u8; 32],
}

impl CanonicalPinnedSourceProjectionV1 {
    pub fn checked_from_binding_facts_v1(
        anchor_dataset_identity: CanonicalDatasetIdentity,
        parent_row_count: u64,
        mut bindings: Vec<CanonicalPinnedSourceBindingFactsV1>,
    ) -> Result<Self, CanonicalPinnedSourceProjectionErrorV1> {
        if parent_row_count == 0 || bindings.is_empty() {
            return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                "source projection has no parent rows or bindings",
            ));
        }
        bindings.sort_by(|left, right| left.dataset_identity.cmp(&right.dataset_identity));
        let mut anchor_count = 0_usize;
        let mut previous_identity: Option<&CanonicalDatasetIdentity> = None;
        let mut previous_timeframe: Option<CanonicalTimeframe> = None;
        for binding in &bindings {
            let identity = binding.dataset_identity();
            if identity.scope() != anchor_dataset_identity.scope()
                || identity.symbol_name() != anchor_dataset_identity.symbol_name()
                || identity.bar_timestamp_convention()
                    != anchor_dataset_identity.bar_timestamp_convention()
                || previous_identity.is_some_and(|previous| previous == identity)
                || previous_timeframe.is_some_and(|previous| previous == identity.timeframe())
            {
                return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                    "source projection crosses series or repeats a dataset/timeframe",
                ));
            }
            previous_identity = Some(identity);
            previous_timeframe = Some(identity.timeframe());
            if identity == &anchor_dataset_identity {
                anchor_count += 1;
                if !segments_cover_parent_rows_v1(binding.segments(), parent_row_count) {
                    return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                        "anchor segments do not exactly cover prepared parent rows",
                    ));
                }
            }
        }
        if anchor_count != 1 {
            return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                "source projection must contain exactly one anchor binding",
            ));
        }
        let base_timeframe = anchor_dataset_identity.timeframe();
        let identity_sha256 = projection_identity_sha256_v1(
            &anchor_dataset_identity,
            base_timeframe,
            parent_row_count,
            &bindings,
        )?;
        if identity_sha256 == [0; 32] {
            return Err(CanonicalPinnedSourceProjectionErrorV1::new(
                "source projection identity is zero",
            ));
        }
        Ok(Self {
            schema_version: CANONICAL_PINNED_SOURCE_PROJECTION_SCHEMA_VERSION_V1,
            anchor_dataset_identity,
            base_timeframe,
            parent_row_count,
            bindings,
            identity_sha256,
        })
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn anchor_dataset_identity(&self) -> &CanonicalDatasetIdentity {
        &self.anchor_dataset_identity
    }

    pub const fn base_timeframe(&self) -> CanonicalTimeframe {
        self.base_timeframe
    }

    pub const fn parent_row_count(&self) -> u64 {
        self.parent_row_count
    }

    pub fn bindings(&self) -> &[CanonicalPinnedSourceBindingFactsV1] {
        &self.bindings
    }

    pub const fn identity_sha256(&self) -> [u8; 32] {
        self.identity_sha256
    }
}

fn segments_cover_parent_rows_v1(
    segments: &[CanonicalPinnedSourceSegmentFactsV1],
    parent_row_count: u64,
) -> bool {
    segments
        .first()
        .is_some_and(|segment| segment.row_start == 0)
        && segments
            .last()
            .is_some_and(|segment| segment.row_end == parent_row_count)
        && segments
            .windows(2)
            .all(|pair| pair[0].row_end == pair[1].row_start)
}

fn projection_identity_sha256_v1(
    anchor_dataset_identity: &CanonicalDatasetIdentity,
    base_timeframe: CanonicalTimeframe,
    parent_row_count: u64,
    bindings: &[CanonicalPinnedSourceBindingFactsV1],
) -> Result<[u8; 32], CanonicalPinnedSourceProjectionErrorV1> {
    let mut hash = Sha256::new();
    hash.update(CANONICAL_PINNED_SOURCE_PROJECTION_HASH_DOMAIN_V1);
    hash.update(CANONICAL_PINNED_SOURCE_PROJECTION_SCHEMA_VERSION_V1.to_le_bytes());
    update_bytes_v1(
        &mut hash,
        &anchor_dataset_identity.canonical_bytes(),
        "anchor dataset identity",
    )?;
    hash.update([base_timeframe.identity_tag()]);
    hash.update(parent_row_count.to_le_bytes());
    hash.update(
        u64::try_from(bindings.len())
            .map_err(|_| CanonicalPinnedSourceProjectionErrorV1::new("binding count overflow"))?
            .to_le_bytes(),
    );
    for binding in bindings {
        update_bytes_v1(
            &mut hash,
            &binding.dataset_identity.canonical_bytes(),
            "binding dataset identity",
        )?;
        update_bytes_v1(
            &mut hash,
            binding.manifest_schema_id.as_bytes(),
            "manifest schema id",
        )?;
        hash.update(binding.manifest_sha256);
        update_bytes_v1(&mut hash, binding.generation_id.as_bytes(), "generation id")?;
        hash.update(binding.vortex_sha256);
        hash.update([binding.bar_timestamp_convention.identity_tag()]);
        hash.update(
            u64::try_from(binding.segments.len())
                .map_err(|_| CanonicalPinnedSourceProjectionErrorV1::new("segment count overflow"))?
                .to_le_bytes(),
        );
        for segment in &binding.segments {
            hash.update(segment.row_start.to_le_bytes());
            hash.update(segment.row_end.to_le_bytes());
            hash.update(segment.timestamp_start_ms.to_le_bytes());
            hash.update(segment.timestamp_end_ms.to_le_bytes());
        }
    }
    Ok(hash.finalize().into())
}

fn update_bytes_v1(
    hash: &mut Sha256,
    bytes: &[u8],
    field: &'static str,
) -> Result<(), CanonicalPinnedSourceProjectionErrorV1> {
    hash.update(
        u64::try_from(bytes.len())
            .map_err(|_| {
                CanonicalPinnedSourceProjectionErrorV1::new(format!("{field} length overflow"))
            })?
            .to_le_bytes(),
    );
    hash.update(bytes);
    Ok(())
}

pub(crate) fn derive_pinned_source_projection_v1(
    sources: &MaterializedPinnedResidentCanonicalSourcesV1,
) -> Result<CanonicalPinnedSourceProjectionV1, CanonicalPinnedSourceProjectionErrorV1> {
    let parent_row_count = u64::try_from(sources.base().frame().len()).map_err(|_| {
        CanonicalPinnedSourceProjectionErrorV1::new("prepared parent row count overflow")
    })?;
    let bindings = sources
        .all_sources()
        .map(|source| {
            CanonicalPinnedSourceBindingFactsV1::checked_from_source_artifact_binding_v1(
                source.binding(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    CanonicalPinnedSourceProjectionV1::checked_from_binding_facts_v1(
        sources.receipt().anchor().identity().clone(),
        parent_row_count,
        bindings,
    )
}
