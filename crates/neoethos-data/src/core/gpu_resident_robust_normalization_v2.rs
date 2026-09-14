//! Data-owned canonical split and allocation preflight for resident robust
//! normalization policy-v3 (continuous transform semantic-v2).
//!
//! The split is shared Discovery semantics, not Search-owned authority. Data
//! reads rows only from an exact pinned/canonical source and enabled mode only
//! from the startup-installed configuration. No public constructor accepts a
//! replacement range, mode, fit value, feature byte or identity hash.

use std::ops::Range;

use anyhow::{Result, ensure};
#[cfg(feature = "gpu-cuda")]
use neoethos_dataset_contracts::CanonicalTimeframe;
use neoethos_gpu_contracts::normalization_v3::SearchNormalizationColumnModeV3;

#[cfg(feature = "gpu-cuda")]
use super::pinned_canonical_series_v1::PinnedCanonicalSeriesV1;
use crate::sealed_data_runtime_normalization_mode_v2;

/// Reuse the CPU policy's exact canonical-name classifier for every backend.
/// This maps typed policy values only; no second SMC name allowlist exists.
pub(crate) fn search_normalization_column_mode_v3(name: &str) -> SearchNormalizationColumnModeV3 {
    use super::normalization::{SmcGateDomain, smc_gate_domain};
    match smc_gate_domain(name) {
        None => SearchNormalizationColumnModeV3::Robust,
        Some(SmcGateDomain::Binary) => SearchNormalizationColumnModeV3::Binary,
        Some(SmcGateDomain::SignedState) => SearchNormalizationColumnModeV3::SignedState,
        Some(SmcGateDomain::SignedContinuous) => SearchNormalizationColumnModeV3::SignedContinuous,
    }
}

/// Decode actual backend fit words through the same portable Data validator.
/// This transport does not replace the name-aware fitted-state identity.
pub(crate) fn fitted_state_from_device_words(
    names: &[String],
    training_rows: Range<usize>,
    words: &[u64],
) -> Result<super::normalization::SearchNormalizationFittedStateV1> {
    use super::normalization::{RobustNormalizationFitF64, SearchNormalizationFittedStateV1};
    ensure!(
        names.len().checked_mul(6) == Some(words.len()),
        "resident normalization fit words do not cover the exact selected schema"
    );
    let fits = words
        .chunks_exact(6)
        .map(|fit| {
            let start = usize::try_from(fit[0])?;
            let end = usize::try_from(fit[1])?;
            ensure!(
                (start..end) == training_rows && fit[5] <= 1,
                "resident normalization fit range or degeneracy flag drift"
            );
            Ok(RobustNormalizationFitF64 {
                training_rows: start..end,
                median: f64::from_bits(fit[2]),
                scale: f64::from_bits(fit[3]),
                valid_training_cells: usize::try_from(fit[4])?,
                degenerate: fit[5] != 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // Checks support, finite location, scale and exact zero-anchored gate fits.
    SearchNormalizationFittedStateV1::new(names.to_vec(), fits)
}

// Compatibility name; the dispatched resident policy is v3 (transform v2
// plus zero-anchored SMC gates), never the legacy gate-centering path.
pub const RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2: u32 = 3;
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
    #[cfg(feature = "gpu-cuda")]
    pub(crate) const fn enabled(&self) -> bool {
        self.enabled
    }

    #[cfg(feature = "gpu-cuda")]
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
#[cfg(feature = "gpu-cuda")]
pub(crate) fn seal_canonical_robust_normalization_split_from_pinned_v2(
    pinned_series: &PinnedCanonicalSeriesV1,
    base_timeframe: CanonicalTimeframe,
) -> Result<SealedCanonicalRobustNormalizationSplitV2> {
    let row_count = pinned_series.row_count(base_timeframe)?;
    let mode = sealed_data_runtime_normalization_mode_v2()?;
    seal_checked_data_split_v2(row_count, mode.enabled())
}

/// Enabled HIP normalization consumes the same Data split as CUDA. The rows
/// come from the actual canonical source-backed shared input, never a caller
/// supplied training range or a later capped Search selection view. Disabled
/// assembly does not call this function and remains valid for short inputs.
#[cfg(feature = "gpu-hip-smc")]
pub(crate) fn seal_canonical_robust_normalization_split_from_hip_v1(
    input: &super::gpu_hip_ohlcv_v1::ResidentHipOhlcvV1<'_>,
) -> Result<SealedCanonicalRobustNormalizationSplitV2> {
    let source = input.canonical_source()?;
    let rows = input.memory_plan().row_count();
    let segments = source.binding().segments();
    ensure!(
        segments.len() == 1
            && segments[0].row_end().checked_sub(segments[0].row_start())
                == u64::try_from(rows).ok(),
        "HIP normalization source segment differs from the retained input"
    );
    let mode = sealed_data_runtime_normalization_mode_v2()?;
    ensure!(
        mode.enabled(),
        "disabled HIP normalization has no fitted split"
    );
    seal_checked_data_split_v2(rows, mode.enabled())
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

/// Immutable Data-owned replay recipe. Unlike the one-shot prepared input,
/// this contains no allocation authority for a particular feature width; it
/// can only mint width-specific inputs under the already sealed row split and
/// startup mode.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(feature = "gpu-cuda")]
pub(crate) struct ResidentRobustNormalizationReplayRecipeV2 {
    semantic_version: u32,
    row_count: usize,
    training_rows: Range<usize>,
    enabled: bool,
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

    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn replay_recipe_v2(&self) -> ResidentRobustNormalizationReplayRecipeV2 {
        ResidentRobustNormalizationReplayRecipeV2 {
            semantic_version: self.semantic_version,
            row_count: self.row_count,
            training_rows: self.training_rows.clone(),
            enabled: self.enabled,
        }
    }
}

#[cfg(feature = "gpu-cuda")]
impl ResidentRobustNormalizationReplayRecipeV2 {
    pub(crate) fn prepare_for_feature_width_v2(
        &self,
        feature_column_count: usize,
    ) -> Result<PreparedResidentRobustNormalizationInputV2> {
        ensure!(
            self.semantic_version == RESIDENT_ROBUST_NORMALIZATION_SEMANTIC_VERSION_V2
                && self.row_count > 0
                && self.training_rows == (0..canonical_training_end_v2(self.row_count)?),
            "resident robust-normalization replay recipe drifted from its canonical split"
        );
        prepare_resident_robust_normalization_extents_v2(
            self.row_count,
            self.training_rows.clone(),
            self.enabled,
            feature_column_count,
        )
    }
}

/// Consume Data's split exactly once and freeze the exact resident allocation
/// extents. Disabled mode carries semantic/range identity but no allocation or
/// launch extent.
pub(crate) fn prepare_resident_robust_normalization_input_v2(
    split: SealedCanonicalRobustNormalizationSplitV2,
    feature_column_count: usize,
) -> Result<PreparedResidentRobustNormalizationInputV2> {
    let consumed = split.consume()?;
    prepare_resident_robust_normalization_extents_v2(
        consumed.row_count,
        consumed.training_rows,
        consumed.enabled,
        feature_column_count,
    )
}

fn prepare_resident_robust_normalization_extents_v2(
    row_count: usize,
    training_rows: Range<usize>,
    enabled: bool,
    feature_column_count: usize,
) -> Result<PreparedResidentRobustNormalizationInputV2> {
    ensure!(
        feature_column_count > 0 && training_rows == (0..canonical_training_end_v2(row_count)?),
        "resident robust normalization requires at least one feature column and the canonical split"
    );
    let (padded_training_rows, normalization_scratch_bytes, fit_metadata_bytes) = if enabled {
        let padded_training_rows = training_rows
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
        row_count,
        feature_column_count,
        training_rows,
        enabled,
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
        assert_eq!(prepared.semantic_version(), 3);
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

#[cfg(all(test, feature = "gpu-cuda-device-fixtures"))]
#[path = "../../tests/fixtures/resident_robust_normalization_v3_device.rs"]
mod device_tests;
