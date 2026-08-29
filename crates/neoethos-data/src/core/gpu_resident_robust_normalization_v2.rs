//! Data-owned canonical split and allocation preflight for resident robust
//! normalization semantic-v2.
//!
//! The split is shared Discovery semantics, not Search-owned authority. Data
//! reads rows only from an exact pinned/canonical source and enabled mode only
//! from the startup-installed configuration. No public constructor accepts a
//! replacement range, mode, fit value, feature byte or identity hash.

use std::ops::Range;

use anyhow::{Result, ensure};
use neoethos_dataset_contracts::CanonicalTimeframe;

use super::pinned_canonical_series_v1::PinnedCanonicalSeriesV1;
use crate::sealed_data_runtime_normalization_mode_v2;

pub const RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2: u32 = 2;
pub const CANONICAL_DISCOVERY_NORMALIZATION_MIN_TRAINING_ROWS_V2: usize = 64;
pub const RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2: usize = 64;
pub const RESIDENT_ROBUST_NORMALIZATION_FIT_WORDS_V2: usize = 6;
pub const RESIDENT_ROBUST_NORMALIZATION_FIT_BYTES_PER_COLUMN_V2: usize = 48;
const CANONICAL_DISCOVERY_OOS_HOLDOUT_FRACTION_V2: f64 = 0.2;
const CANONICAL_ROBUST_NORMALIZATION_SPLIT_AUTHORITY_V2: &str =
    "neoethos.data.canonical-robust-normalization-split.semantic-v2";

/// Move-only canonical split derived from Data authority. It is neither
/// `Clone`, serializable nor constructible from caller-supplied evidence.
#[must_use = "the canonical normalization split must be consumed exactly once by Data"]
#[derive(Debug)]
pub(crate) struct SealedCanonicalRobustNormalizationSplitV2 {
    authority: &'static str,
    row_count: usize,
    training_rows: Range<usize>,
    enabled: bool,
}

impl SealedCanonicalRobustNormalizationSplitV2 {
    pub(crate) const fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn is_intact_for_row_count(&self, row_count: usize) -> bool {
        self.authority == CANONICAL_ROBUST_NORMALIZATION_SPLIT_AUTHORITY_V2
            && self.row_count == row_count
            && canonical_training_end_v2(row_count)
                .is_ok_and(|training_end| self.training_rows == (0..training_end))
    }

    fn consume(self) -> Result<ConsumedCanonicalRobustNormalizationSplitV2> {
        ensure!(
            self.authority == CANONICAL_ROBUST_NORMALIZATION_SPLIT_AUTHORITY_V2,
            "canonical robust-normalization split authority drifted"
        );
        let canonical_training_end = canonical_training_end_v2(self.row_count)?;
        ensure!(
            self.training_rows == (0..canonical_training_end),
            "canonical robust-normalization split changed after sealing"
        );
        Ok(ConsumedCanonicalRobustNormalizationSplitV2 {
            row_count: self.row_count,
            training_rows: self.training_rows,
            enabled: self.enabled,
        })
    }
}

#[derive(Debug)]
struct ConsumedCanonicalRobustNormalizationSplitV2 {
    row_count: usize,
    training_rows: Range<usize>,
    enabled: bool,
}

fn canonical_training_end_v2(row_count: usize) -> Result<usize> {
    ensure!(row_count > 0, "canonical normalization parent is empty");
    let split_at =
        ((row_count as f64) * (1.0 - CANONICAL_DISCOVERY_OOS_HOLDOUT_FRACTION_V2)).floor() as usize;
    ensure!(
        split_at > 0,
        "canonical normalization training range is empty"
    );
    ensure!(
        split_at < row_count,
        "canonical normalization holdout suffix is empty"
    );
    ensure!(
        split_at >= CANONICAL_DISCOVERY_NORMALIZATION_MIN_TRAINING_ROWS_V2,
        "canonical normalization training range has {split_at} rows; at least {} are required",
        CANONICAL_DISCOVERY_NORMALIZATION_MIN_TRAINING_ROWS_V2
    );
    Ok(split_at)
}

fn seal_checked_data_split_v2(
    row_count: usize,
    enabled: bool,
) -> Result<SealedCanonicalRobustNormalizationSplitV2> {
    let split_at = canonical_training_end_v2(row_count)?;
    Ok(SealedCanonicalRobustNormalizationSplitV2 {
        authority: CANONICAL_ROBUST_NORMALIZATION_SPLIT_AUTHORITY_V2,
        row_count,
        training_rows: 0..split_at,
        enabled,
    })
}

/// Seal from the metadata-only pinned generation authority and the exact mode
/// installed once by startup. The caller supplies neither rows nor mode.
pub(crate) fn seal_canonical_robust_normalization_split_from_pinned_v2(
    pinned_series: &PinnedCanonicalSeriesV1,
    base_timeframe: CanonicalTimeframe,
) -> Result<SealedCanonicalRobustNormalizationSplitV2> {
    let row_count = pinned_series.row_count(base_timeframe)?;
    let mode = sealed_data_runtime_normalization_mode_v2()?;
    seal_checked_data_split_v2(row_count, mode.enabled())
}

/// Data-owned continuation after the canonical split has been consumed.
/// Private fields and the absence of `Clone` keep the value move-only.
#[derive(Debug)]
pub(crate) struct PreparedResidentRobustNormalizationInputV2 {
    semantic_version: u32,
    row_count: usize,
    feature_column_count: usize,
    training_rows: Range<usize>,
    enabled: bool,
    padded_training_rows: usize,
    normalization_scratch_bytes: usize,
    fit_metadata_bytes: usize,
}

impl PreparedResidentRobustNormalizationInputV2 {
    pub(crate) const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }

    pub(crate) const fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) const fn feature_column_count(&self) -> usize {
        self.feature_column_count
    }

    pub(crate) fn training_rows(&self) -> Range<usize> {
        self.training_rows.clone()
    }

    pub(crate) const fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) const fn padded_training_rows(&self) -> usize {
        self.padded_training_rows
    }

    pub(crate) const fn normalization_scratch_bytes(&self) -> usize {
        self.normalization_scratch_bytes
    }

    pub(crate) const fn fit_metadata_bytes(&self) -> usize {
        self.fit_metadata_bytes
    }
}

/// Consume Data's split exactly once and freeze the exact resident allocation
/// extents. Disabled mode carries semantic/range identity but no allocation or
/// launch extent.
pub(crate) fn prepare_resident_robust_normalization_input_v2(
    split: SealedCanonicalRobustNormalizationSplitV2,
    feature_column_count: usize,
) -> Result<PreparedResidentRobustNormalizationInputV2> {
    ensure!(
        feature_column_count > 0,
        "resident robust normalization requires at least one feature column"
    );
    let consumed = split.consume()?;
    let (padded_training_rows, normalization_scratch_bytes, fit_metadata_bytes) = if consumed
        .enabled
    {
        let padded_training_rows = consumed
            .training_rows
            .len()
            .checked_next_power_of_two()
            .ok_or_else(|| anyhow::anyhow!("robust-normalization padded training rows overflow"))?;
        let scratch_columns =
            feature_column_count.min(RESIDENT_ROBUST_NORMALIZATION_MAX_BATCH_COLUMNS_V2);
        let normalization_scratch_bytes = scratch_columns
            .checked_mul(padded_training_rows)
            .and_then(|slots| slots.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| anyhow::anyhow!("robust-normalization scratch extent overflow"))?;
        let fit_metadata_bytes = feature_column_count
            .checked_mul(RESIDENT_ROBUST_NORMALIZATION_FIT_BYTES_PER_COLUMN_V2)
            .ok_or_else(|| anyhow::anyhow!("robust-normalization fit extent overflow"))?;
        (
            padded_training_rows,
            normalization_scratch_bytes,
            fit_metadata_bytes,
        )
    } else {
        (0, 0, 0)
    };
    Ok(PreparedResidentRobustNormalizationInputV2 {
        semantic_version: RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2,
        row_count: consumed.row_count,
        feature_column_count,
        training_rows: consumed.training_rows,
        enabled: consumed.enabled,
        padded_training_rows,
        normalization_scratch_bytes,
        fit_metadata_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_data_split_freezes_enabled_and_disabled_extents() {
        let prepared = prepare_resident_robust_normalization_input_v2(
            seal_checked_data_split_v2(100, true).expect("canonical enabled split"),
            65,
        )
        .expect("move-only Data input");
        assert_eq!(prepared.semantic_version(), 2);
        assert_eq!(prepared.row_count(), 100);
        assert_eq!(prepared.feature_column_count(), 65);
        assert_eq!(prepared.training_rows(), 0..80);
        assert!(prepared.enabled());
        assert_eq!(prepared.padded_training_rows(), 128);
        assert_eq!(prepared.normalization_scratch_bytes(), 64 * 128 * 8);
        assert_eq!(prepared.fit_metadata_bytes(), 65 * 48);

        let disabled = prepare_resident_robust_normalization_input_v2(
            seal_checked_data_split_v2(100, false).expect("canonical disabled split"),
            65,
        )
        .expect("disabled mode");
        assert!(!disabled.enabled());
        assert_eq!(disabled.padded_training_rows(), 0);
        assert_eq!(disabled.normalization_scratch_bytes(), 0);
        assert_eq!(disabled.fit_metadata_bytes(), 0);
        assert!(seal_checked_data_split_v2(80, true).is_ok());
        assert!(seal_checked_data_split_v2(79, true).is_err());
    }
}
