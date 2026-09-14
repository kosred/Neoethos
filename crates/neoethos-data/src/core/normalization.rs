//! Per-column feature normalization.
//!
//! Raw indicator outputs span wildly different scales — a price-level
//! feature like `vwap` is on the order of 1.10 (EURUSD) or 165 (EURJPY)
//! or 2400 (XAUUSD), while an oscillator like `rsi` is bounded 0..100,
//! and SMC binary flags are 0/1. Mixing them in a weighted sum (the
//! GA's "combined" signal) means the largest-magnitude column always
//! dominates regardless of weight, and the GA's `long_threshold ≈ 0.45`
//! never triggers on small-scale columns or always triggers on
//! large-scale ones. The result is the empty-portfolio bug we observed
//! on EURJPY (feature magnitudes ±3.5e11) and XAUUSD.
//!
//! The fix is a robust per-column z-score:
//! - Compute median + MAD (median absolute deviation) per column.
//! - `z = (x - median) / (1.4826 * MAD)` — Gaussian-equivalent scale.
//! - Preserve invalid cells as typed validity plus canonical NaN; an undefined
//!   indicator value is never silently converted into a valid numeric zero.
//! - Clip to ±10 (1-in-billion under Gaussian) so a single outlier
//!   can't blow up the GA's combined sum.
//!
//! The caller supplies the exact training-row range. This module never infers
//! a train/test split and therefore cannot fit on future rows accidentally.

use std::ops::Range;

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::core::features::{FeatureCellValidity, FeatureColumnF64};

pub const Z_CLIP_F64: f64 = 10.0;
pub const MAD_TO_SIGMA_F64: f64 = 1.4826;

/// Search-frame policy v3 leaves semantic SMC gate states zero-anchored.
/// The continuous robust transform itself remains semantic v2 below.
pub const SEARCH_NORMALIZATION_POLICY_VERSION: u32 = 3;

/// These are the SMC producer's gate inputs, not a substring-based exemption
/// for the whole SMC family. Distances, strength and Fibonacci features still
/// use a train-only robust fit. A direct higher-TF copy has the same domain.
pub(crate) fn smc_gate_domain(name: &str) -> Option<SmcGateDomain> {
    let canonical = name.to_ascii_lowercase().replace(['-', ' '], "_");
    let base = if canonical.starts_with("smc_") {
        canonical.as_str()
    } else {
        let (timeframe, feature) = canonical.split_once('_')?;
        timeframe
            .to_ascii_uppercase()
            .parse::<crate::CanonicalTimeframe>()
            .ok()?;
        feature
    };
    match base {
        "smc_eqh" | "smc_eql" => Some(SmcGateDomain::Binary),
        // ATR-scaled MA difference: zero denotes no directional bias. It is
        // already dimensionless; clipping bounds its mixing weight without
        // translating zero or reversing the direction relative to price.
        "smc_trend_bias" | "smc_trend" => Some(SmcGateDomain::SignedContinuous),
        "smc_ob" | "smc_fvg" | "smc_liq_sweep" | "smc_liq" | "smc_pd_array" | "smc_premium"
        | "smc_inducement" | "smc_bos" | "smc_mss" | "smc_choch" | "smc_displacement" => {
            Some(SmcGateDomain::SignedState)
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SmcGateDomain {
    Binary,
    SignedState,
    SignedContinuous,
}

/// Normalize an actual search column without changing the meaning of a gate.
///
/// In particular, a common +1 event must not become zero after centering and a
/// constant valid "no event" column must not become an undefined indicator.
/// The identity fit is explicit (median=0, scale=1), fits the same declared
/// training scope, and is included in the frame's existing fitted-state hash.
pub fn normalize_search_feature_column_f64(
    column: &mut FeatureColumnF64,
    training_rows: Range<usize>,
) -> Result<RobustNormalizationFitF64> {
    let Some(domain) = smc_gate_domain(&column.name) else {
        return normalize_feature_column_f64(column, training_rows);
    };
    ensure!(
        training_rows.start < training_rows.end && training_rows.end <= column.len(),
        "feature column `{}` normalization range {:?} is outside 0..{}",
        column.name,
        training_rows,
        column.len()
    );
    let mut valid_training_cells = 0;
    // Validate before mutation, including the held-out values. This checks
    // representation only; it does not estimate anything from future rows.
    for row in 0..column.len() {
        if !column.validity[row].is_valid() {
            continue;
        }
        let value = column.values[row];
        let in_domain = value.is_finite()
            && match domain {
                SmcGateDomain::Binary => value == 0.0 || value == 1.0,
                SmcGateDomain::SignedState => value == -1.0 || value == 0.0 || value == 1.0,
                SmcGateDomain::SignedContinuous => true,
            };
        ensure!(
            in_domain,
            "SMC gate column `{}` row {row} has an invalid raw state {value}; a centered column cannot be treated as raw",
            column.name
        );
        valid_training_cells += usize::from(training_rows.contains(&row));
    }
    ensure!(
        valid_training_cells > 0,
        "feature column `{}` has no valid cells in normalization training range {:?}",
        column.name,
        training_rows
    );
    for row in 0..column.len() {
        column.values[row] = if column.validity[row].is_valid() {
            column.values[row].clamp(-Z_CLIP_F64, Z_CLIP_F64)
        } else {
            f64::NAN
        };
    }
    Ok(RobustNormalizationFitF64 {
        training_rows,
        median: 0.0,
        scale: 1.0,
        valid_training_cells,
        degenerate: false,
    })
}

/// Immutable fitted state for the explicit-validity f64 normalization lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustNormalizationFitF64 {
    pub training_rows: Range<usize>,
    // Hexadecimal IEEE-754 bits preserve the exact fitted-state identity even
    // through JSON readers without decimal float-roundtrip support. A valid
    // degenerate fit may have infinite scale; it must not become JSON null.
    #[serde(with = "exact_f64_bits")]
    pub median: f64,
    #[serde(with = "exact_f64_bits")]
    pub scale: f64,
    pub valid_training_cells: usize,
    pub degenerate: bool,
}

impl PartialEq for RobustNormalizationFitF64 {
    fn eq(&self, other: &Self) -> bool {
        self.training_rows == other.training_rows
            && self.median.to_bits() == other.median.to_bits()
            && self.scale.to_bits() == other.scale.to_bits()
            && self.valid_training_cells == other.valid_training_cells
            && self.degenerate == other.degenerate
    }
}

impl Eq for RobustNormalizationFitF64 {}

mod exact_f64_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{:016x}", value.to_bits()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.len() != 16 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(serde::de::Error::custom("expected 16 hexadecimal f64 bits"));
        }
        let bits = u64::from_str_radix(&text, 16).map_err(serde::de::Error::custom)?;
        Ok(f64::from_bits(bits))
    }
}

/// Portable, immutable training fit for the CPU search feature representation.
///
/// Keep the complete ordered schema when selecting strategy columns: the
/// normalization node seals this full fit, not a refitted/projected subset.
/// Row coordinates remain relative to the original base source frame even
/// when the feature frame is later sliced to OOS or selected source rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchNormalizationFittedStateV1 {
    policy_version: u32,
    transform_semantic_version: u32,
    column_names: Vec<String>,
    fits: Vec<RobustNormalizationFitF64>,
}

impl SearchNormalizationFittedStateV1 {
    pub fn new(column_names: Vec<String>, fits: Vec<RobustNormalizationFitF64>) -> Result<Self> {
        let state = Self {
            policy_version: SEARCH_NORMALIZATION_POLICY_VERSION,
            transform_semantic_version: NORMALIZATION_TRANSFORM_SEMANTIC_VERSION,
            column_names,
            fits,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    pub fn fits(&self) -> &[RobustNormalizationFitF64] {
        &self.fits
    }

    pub fn training_rows(&self) -> Result<Range<usize>> {
        self.validate()?;
        Ok(self.fits[0].training_rows.clone())
    }

    /// Validate deserialized state before it can authorize a production
    /// transform. This checks representation, never estimates from new rows.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.policy_version == SEARCH_NORMALIZATION_POLICY_VERSION
                && self.transform_semantic_version == NORMALIZATION_TRANSFORM_SEMANTIC_VERSION,
            "unsupported fitted search normalization semantics"
        );
        ensure!(
            !self.column_names.is_empty() && self.column_names.len() == self.fits.len(),
            "normalization fit count mismatch or empty schema"
        );
        let mut names = std::collections::HashSet::with_capacity(self.column_names.len());
        let training_rows = &self.fits[0].training_rows;
        for (name, fit) in self.column_names.iter().zip(&self.fits) {
            ensure!(
                !name.is_empty() && names.insert(name),
                "duplicate or empty fitted feature name"
            );
            validate_search_fit(name, fit)?;
            ensure!(
                &fit.training_rows == training_rows,
                "fitted feature `{name}` belongs to a different training row range"
            );
        }
        Ok(())
    }

    /// Same byte domain/order as the pre-existing normalization FeaturePlan
    /// hash. Persistence adds the payload; it does not redefine fitted math.
    pub fn fitted_state_hash(&self) -> Result<[u8; 32]> {
        self.validate()?;
        normalization_fit_hash(&self.column_names, &self.fits)
    }

    pub fn validate_plan(&self, plan: &neoethos_feature_contracts::FeaturePlanV1) -> Result<()> {
        let hash = self.fitted_state_hash()?;
        let nodes = plan
            .nodes()
            .iter()
            .filter(|node| {
                node.operation() == neoethos_feature_contracts::FeatureOperationTagV1::Normalization
            })
            .collect::<Vec<_>>();
        ensure!(
            nodes.len() == 1,
            "fitted search state requires exactly one normalization node"
        );
        let node = nodes[0];
        ensure!(
            node.id() == "normalization:robust-f64"
                && node.semantic_version() == self.policy_version
                && node.fitted_state_hash() == Some(hash),
            "persisted normalization fit does not match the sealed FeaturePlan"
        );
        ensure!(
            node.outputs()
                .iter()
                .map(|output| output.name())
                .eq(self.column_names.iter().map(String::as_str)),
            "persisted normalization schema/order does not match the sealed FeaturePlan"
        );
        self.validate_feature_names(plan.final_outputs())
    }

    /// A strategy may select/reorder a subset, but every name must resolve to
    /// its original fitted column. Positional fallback is never permitted.
    pub fn validate_feature_names(&self, names: &[String]) -> Result<()> {
        self.validate()?;
        ensure!(
            !names.is_empty(),
            "normalization projection needs feature names"
        );
        let fitted = self
            .column_names
            .iter()
            .collect::<std::collections::HashSet<_>>();
        let mut selected = std::collections::HashSet::with_capacity(names.len());
        for name in names {
            ensure!(
                selected.insert(name),
                "duplicate normalization projection `{name}`"
            );
            ensure!(
                fitted.contains(name),
                "feature `{name}` has no persisted training fit"
            );
        }
        Ok(())
    }

    /// Apply saved training parameters to raw columns. No training/OOS split,
    /// median, scale or support is inferred from these new observations.
    pub fn apply_columns(&self, columns: &mut [FeatureColumnF64]) -> Result<()> {
        let names = columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        self.validate_feature_names(&names)?;
        let fits = self
            .column_names
            .iter()
            .zip(&self.fits)
            .collect::<std::collections::HashMap<_, _>>();
        // Check every raw column before mutation, so malformed gate values or
        // structural input errors cannot leave a partly transformed batch.
        for column in columns.iter() {
            ensure!(
                column.values.len() == column.validity.len(),
                "normalization column length mismatch"
            );
            for (&value, &validity) in column.values.iter().zip(&column.validity) {
                validate_raw_search_cell(&column.name, value, validity)?;
            }
        }
        for column in columns {
            let fit = fits[&column.name];
            apply_fitted_column_values(column, fit);
        }
        Ok(())
    }

    pub fn transform_value(
        &self,
        name: &str,
        value: f64,
        validity: FeatureCellValidity,
    ) -> Result<crate::core::features::FeatureCellF64> {
        self.validate()?;
        let index = self
            .column_names
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| anyhow::anyhow!("feature `{name}` has no persisted training fit"))?;
        validate_raw_search_cell(name, value, validity)?;
        Ok(transform_fitted_cell(value, validity, &self.fits[index]))
    }
}

/// Scalar producer replay seam after the containing payload has been
/// validated once. Validates this column/fit without rescanning every other
/// fit for each worker in a wide feature cube.
pub(crate) fn apply_search_normalization_fit(
    column: &mut FeatureColumnF64,
    fit: &RobustNormalizationFitF64,
) -> Result<()> {
    validate_search_fit(&column.name, fit)?;
    ensure!(
        column.values.len() == column.validity.len(),
        "normalization column length mismatch"
    );
    for (&value, &validity) in column.values.iter().zip(&column.validity) {
        validate_raw_search_cell(&column.name, value, validity)?;
    }
    apply_fitted_column_values(column, fit);
    Ok(())
}

fn apply_fitted_column_values(column: &mut FeatureColumnF64, fit: &RobustNormalizationFitF64) {
    for (value, validity) in column.values.iter_mut().zip(&mut column.validity) {
        let cell = transform_fitted_cell(*value, *validity, fit);
        *value = cell.value;
        *validity = cell.validity;
    }
}

fn validate_search_fit(name: &str, fit: &RobustNormalizationFitF64) -> Result<()> {
    ensure!(
        fit.training_rows.start < fit.training_rows.end
            && fit.valid_training_cells > 0
            && fit.valid_training_cells <= fit.training_rows.len(),
        "feature `{name}` has an invalid fitted training scope/support"
    );
    ensure!(
        fit.median.is_finite()
            && !fit.scale.is_nan()
            && fit.scale >= 0.0
            && (fit.degenerate || (fit.scale.is_finite() && fit.scale > 0.0)),
        "feature `{name}` has invalid fitted location/scale"
    );
    if smc_gate_domain(name).is_some() {
        ensure!(
            fit.median == 0.0 && fit.scale == 1.0 && !fit.degenerate,
            "SMC gate `{name}` requires its zero-anchored identity fit"
        );
    }
    Ok(())
}

fn validate_raw_search_cell(name: &str, value: f64, validity: FeatureCellValidity) -> Result<()> {
    if !validity.is_valid() {
        return Ok(());
    }
    ensure!(
        value.is_finite(),
        "feature `{name}` has a valid non-finite raw cell"
    );
    let in_domain = match smc_gate_domain(name) {
        Some(SmcGateDomain::Binary) => value == 0.0 || value == 1.0,
        Some(SmcGateDomain::SignedState) => value == -1.0 || value == 0.0 || value == 1.0,
        Some(SmcGateDomain::SignedContinuous) | None => true,
    };
    ensure!(
        in_domain,
        "SMC gate `{name}` has an invalid raw state {value}"
    );
    Ok(())
}

fn transform_fitted_cell(
    value: f64,
    validity: FeatureCellValidity,
    fit: &RobustNormalizationFitF64,
) -> crate::core::features::FeatureCellF64 {
    use crate::core::features::FeatureCellF64;
    if !validity.is_valid() {
        return FeatureCellF64 {
            value: f64::NAN,
            validity,
        };
    }
    if fit.degenerate {
        return FeatureCellF64 {
            value: f64::NAN,
            validity: FeatureCellValidity::Degenerate,
        };
    }
    let normalized = (value - fit.median) / fit.scale;
    if normalized.is_finite() {
        FeatureCellF64 {
            value: normalized.clamp(-Z_CLIP_F64, Z_CLIP_F64),
            validity,
        }
    } else {
        FeatureCellF64 {
            value: f64::NAN,
            validity: FeatureCellValidity::NonFinite,
        }
    }
}

pub(crate) fn normalization_fit_hash(
    names: &[String],
    fits: &[RobustNormalizationFitF64],
) -> Result<[u8; 32]> {
    ensure!(
        names.len() == fits.len(),
        "normalization fit count mismatch"
    );
    let mut hash = Sha256::new();
    hash.update(b"neoethos.robust-normalization-fit.f64.v1\0");
    for (name, fit) in names.iter().zip(fits) {
        hash.update((name.len() as u64).to_be_bytes());
        hash.update(name.as_bytes());
        hash.update((fit.training_rows.start as u64).to_be_bytes());
        hash.update((fit.training_rows.end as u64).to_be_bytes());
        hash.update(fit.median.to_bits().to_be_bytes());
        hash.update(fit.scale.to_bits().to_be_bytes());
        hash.update((fit.valid_training_cells as u64).to_be_bytes());
        hash.update([u8::from(fit.degenerate)]);
    }
    Ok(hash.finalize().into())
}

/// Fit robust normalization only on explicitly valid training cells and apply
/// that immutable fit to the full column.
///
/// No split is inferred here: callers must supply the exact training range.
/// Invalid cells retain their reason and canonical NaN payload. A constant
/// training column is marked degenerate rather than emitted as a zero-valued
/// signal. If MAD is zero but the observations are not constant (for example a
/// sparse binary flag), population standard deviation is the deterministic
/// fallback scale so distinct valid values remain distinct.
pub fn normalize_feature_column_f64(
    column: &mut FeatureColumnF64,
    training_rows: Range<usize>,
) -> Result<RobustNormalizationFitF64> {
    if training_rows.start >= training_rows.end || training_rows.end > column.len() {
        bail!(
            "feature column `{}` normalization range {:?} is outside 0..{}",
            column.name,
            training_rows,
            column.len()
        );
    }

    let mut training_values = Vec::with_capacity(training_rows.len());
    for row in training_rows.clone() {
        if column.validity[row].is_valid() {
            let value = column.values[row];
            if !value.is_finite() {
                bail!(
                    "feature column `{}` row {row} is valid but non-finite before normalization",
                    column.name
                );
            }
            training_values.push(value);
        }
    }
    if training_values.is_empty() {
        bail!(
            "feature column `{}` has no valid cells in normalization training range {:?}",
            column.name,
            training_rows
        );
    }

    training_values.sort_by(f64::total_cmp);
    let median = median_sorted_f64(&training_values);
    let mut deviations: Vec<f64> = training_values
        .iter()
        .map(|value| (value - median).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    let mad_scale = median_sorted_f64(&deviations) * MAD_TO_SIGMA_F64;
    let max_abs = training_values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let scale_floor = 32.0 * f64::EPSILON * max_abs.max(1.0);

    let scale = if mad_scale > scale_floor {
        mad_scale
    } else {
        let mean = training_values.iter().sum::<f64>() / training_values.len() as f64;
        let variance = training_values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / training_values.len() as f64;
        variance.sqrt()
    };
    let degenerate = !scale.is_finite() || scale <= scale_floor;

    if degenerate {
        for row in 0..column.len() {
            if column.validity[row].is_valid() {
                column.validity[row] = FeatureCellValidity::Degenerate;
                column.values[row] = f64::NAN;
            }
        }
    } else {
        for row in 0..column.len() {
            if !column.validity[row].is_valid() {
                column.values[row] = f64::NAN;
                continue;
            }
            let normalized = (column.values[row] - median) / scale;
            if normalized.is_finite() {
                column.values[row] = normalized.clamp(-Z_CLIP_F64, Z_CLIP_F64);
            } else {
                column.validity[row] = FeatureCellValidity::NonFinite;
                column.values[row] = f64::NAN;
            }
        }
    }

    Ok(RobustNormalizationFitF64 {
        training_rows,
        median,
        scale,
        valid_training_cells: training_values.len(),
        degenerate,
    })
}

fn median_sorted_f64(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        sorted[mid - 1] * 0.5 + sorted[mid] * 0.5
    } else {
        sorted[mid]
    }
}

/// Version 2 fits only the declared training rows and preserves typed invalid
/// cells instead of rewriting them to numeric zero.
pub const NORMALIZATION_TRANSFORM_SEMANTIC_VERSION: u32 = 2;

#[cfg(test)]
mod fitted_state_tests {
    use super::*;
    use crate::core::features::{
        FeatureCellValidity::{Gap, Valid, Warmup},
        FeatureFrame,
    };

    fn fitted_columns() -> Result<(
        Vec<FeatureColumnF64>,
        Vec<FeatureColumnF64>,
        SearchNormalizationFittedStateV1,
    )> {
        let raw = vec![
            FeatureColumnF64::new(
                "price",
                vec![1.0, 3.0, 5.0, 7.0, 1000.0, -1000.0],
                vec![Valid; 6],
            )?,
            FeatureColumnF64::new(
                "H1_smc_ob",
                vec![1.0, 1.0, 1.0, 0.0, -1.0, 0.0],
                vec![Valid; 6],
            )?,
            FeatureColumnF64::new("constant", vec![2.0; 6], vec![Valid; 6])?,
        ];
        let mut normalized = raw.clone();
        let fits = normalized
            .iter_mut()
            .map(|column| normalize_search_feature_column_f64(column, 0..4))
            .collect::<Result<Vec<_>>>()?;
        let state = SearchNormalizationFittedStateV1::new(
            raw.iter().map(|column| column.name.clone()).collect(),
            fits,
        )?;
        Ok((raw, normalized, state))
    }

    fn assert_column_bits(actual: &FeatureColumnF64, expected: &FeatureColumnF64) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.validity, expected.validity);
        assert_eq!(
            actual
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    fn frame_with_fitted_columns(
        columns: Vec<FeatureColumnF64>,
        state: &SearchNormalizationFittedStateV1,
    ) -> Result<FeatureFrame> {
        use neoethos_feature_contracts::{
            DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1,
            FeatureOutputV1, FeaturePlanV1, SourceArtifactBindingV1, SourceSegmentV1,
        };
        let rows = columns[0].len();
        let names = columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let identity = crate::CanonicalDatasetIdentity::external(
            "fitted-state-test",
            "TESTFX",
            crate::CanonicalTimeframe::M1,
            crate::BarTimestampConvention::BarOpen,
        )?;
        let source = FeatureNodeV1::source(
            "source",
            identity.clone(),
            "f64-ms",
            1,
            vec![FeatureOutputV1::f64("raw", 1)?],
            [1; 32],
        )?;
        let normalized = FeatureNodeV1::transform(
            "normalization:robust-f64",
            FeatureOperationTagV1::Normalization,
            SEARCH_NORMALIZATION_POLICY_VERSION,
            vec!["source".into()],
            names
                .iter()
                .map(|name| FeatureOutputV1::f64(name, SEARCH_NORMALIZATION_POLICY_VERSION))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            Vec::new(),
            [2; 32],
            [3; 32],
            Some(state.fitted_state_hash()?),
        )?;
        let plan = FeaturePlanV1::new(vec![source, normalized], names)?;
        let timestamps = (0..rows)
            .map(|row| 1_700_000_000_000 + row as i64 * 60_000)
            .collect::<Vec<_>>();
        let binding = SourceArtifactBindingV1::new(
            "source",
            identity,
            "manifest",
            [4; 32],
            "test-generation",
            [5; 32],
            crate::BarTimestampConvention::BarOpen,
            vec![SourceSegmentV1::new(
                0,
                rows as u64,
                timestamps[0],
                timestamps[rows - 1],
            )?],
        )?;
        let provenance = DatasetFeatureArtifactProvenanceV1::new(&plan, vec![binding])?;
        FeatureFrame::from_columns(timestamps, columns, plan, provenance)?
            .with_normalization_fitted_state(state.clone())
    }

    #[test]
    fn persisted_fit_replays_identical_oos_bits_without_refitting() -> Result<()> {
        let (raw, expected, state) = fitted_columns()?;
        assert_eq!(state.fits()[0].median, 4.0);
        assert_eq!(state.fits()[0].scale, 2.0 * 1.4826);
        assert_eq!(
            expected[0].values[0].to_bits(),
            (-3.0_f64 / (2.0 * 1.4826)).to_bits()
        );
        // Only two unseen rows remain; the saved training scope has four.
        // Refitting these values would produce +-0.674..., not clipped +-10.
        let mut live = raw
            .into_iter()
            .map(|column| {
                FeatureColumnF64::new(
                    column.name,
                    column.values[4..].to_vec(),
                    column.validity[4..].to_vec(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        state.apply_columns(&mut live)?;
        assert_eq!(live[0].values, [10.0, -10.0]);
        assert_eq!(live[1].values, [-1.0, 0.0]);
        for (actual, expected) in live.iter().zip(expected) {
            assert_column_bits(
                actual,
                &FeatureColumnF64::new(
                    expected.name,
                    expected.values[4..].to_vec(),
                    expected.validity[4..].to_vec(),
                )?,
            );
        }
        assert_eq!(state.training_rows()?, 0..4);
        Ok(())
    }

    #[test]
    fn saved_fit_json_preserves_exact_bits_and_degenerate_infinity() -> Result<()> {
        let (_, _, mut state) = fitted_columns()?;
        state.fits[0].median = f64::from_bits(0x3ff0000000000001);
        state.fits[0].scale = f64::from_bits(1);
        state.fits[2].scale = f64::INFINITY;
        let before = state.fitted_state_hash()?;
        let encoded = serde_json::to_string(&state)?;
        assert!(encoded.contains("3ff0000000000001"));
        assert!(encoded.contains("7ff0000000000000"));
        assert!(!encoded.contains("null"));
        let restored: SearchNormalizationFittedStateV1 = serde_json::from_str(&encoded)?;
        restored.validate()?;
        assert_eq!(state, restored);
        assert_eq!(before, restored.fitted_state_hash()?);
        Ok(())
    }

    #[test]
    fn fitted_state_rejects_invalid_statistics_scopes_schema_and_semantics() -> Result<()> {
        let (_, _, state) = fitted_columns()?;
        for scale in [f64::NAN, -1.0, 0.0, f64::INFINITY] {
            let mut invalid = state.clone();
            invalid.fits[0].scale = scale;
            assert!(invalid.validate().is_err());
        }
        for median in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut invalid = state.clone();
            invalid.fits[0].median = median;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = state.clone();
        invalid.fits[1].median = 1.0;
        assert!(invalid.validate().is_err(), "SMC cannot be centered");
        let mut invalid = state.clone();
        invalid.fits[0].valid_training_cells = 0;
        assert!(invalid.validate().is_err());
        invalid.fits[0].valid_training_cells = 5;
        assert!(invalid.validate().is_err());
        let mut invalid = state.clone();
        invalid.fits[0].training_rows = 1..5;
        assert!(invalid.validate().is_err());
        let mut invalid = state.clone();
        invalid.column_names[1] = invalid.column_names[0].clone();
        assert!(invalid.validate().is_err());
        let mut invalid = state.clone();
        invalid.policy_version += 1;
        assert!(invalid.validate().is_err());
        invalid = state.clone();
        invalid.transform_semantic_version += 1;
        assert!(invalid.validate().is_err());
        assert!(state.validate_feature_names(&["missing".into()]).is_err());
        assert!(
            state
                .validate_feature_names(&["price".into(), "price".into()])
                .is_err()
        );
        assert!(
            serde_json::from_str::<SearchNormalizationFittedStateV1>(
                &serde_json::to_string(&state)?.replace("4010000000000000", "not-f64-bits")
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn frozen_transform_preserves_validity_and_rejects_malformed_smc_before_mutation() -> Result<()>
    {
        let (_, _, state) = fitted_columns()?;
        let mut columns = vec![FeatureColumnF64::new(
            "price",
            vec![3.0, 4.0, 5.0],
            vec![Warmup, Gap, Valid],
        )?];
        state.apply_columns(&mut columns)?;
        assert_eq!(columns[0].validity, [Warmup, Gap, Valid]);
        assert!(columns[0].values[0].is_nan() && columns[0].values[1].is_nan());
        assert_eq!(
            columns[0].values[2].to_bits(),
            (1.0_f64 / (2.0 * 1.4826)).to_bits()
        );
        let mut malformed = vec![
            FeatureColumnF64::new("price", vec![1.0], vec![Valid])?,
            FeatureColumnF64::new("H1_smc_ob", vec![0.5], vec![Valid])?,
        ];
        let original = malformed.clone();
        assert!(state.apply_columns(&mut malformed).is_err());
        assert_eq!(malformed, original);
        Ok(())
    }

    #[test]
    fn fitted_frame_views_keep_original_scope_full_schema_and_plan_binding() -> Result<()> {
        let (_, columns, state) = fitted_columns()?;
        let frame = frame_with_fitted_columns(columns, &state)?.with_feature_build_options(
            crate::FeatureBuildOptions {
                prefix_base_features: true,
                normalization_training_rows: Some(0..4),
                ..Default::default()
            },
        );
        for selected in [
            frame.select_columns(&[1, 0])?,
            frame.row_window(4, 6)?,
            frame.row_slice(4, 6)?,
            frame.select_rows(&[1, 4, 5])?,
        ] {
            let carried = selected.normalization_fitted_state().expect("fit retained");
            assert_eq!(carried, &state);
            assert_eq!(carried.training_rows()?, 0..4);
            assert_eq!(carried.column_names().len(), 3);
            carried.validate_plan(selected.plan())?;
            assert_eq!(
                selected.feature_build_options(),
                frame.feature_build_options()
            );
        }
        let mut tampered = state.clone();
        tampered.fits[0].median += 1.0;
        assert!(tampered.validate_plan(frame.plan()).is_err());
        let mut reordered = state.clone();
        reordered.column_names.swap(0, 2);
        reordered.fits.swap(0, 2);
        assert!(reordered.validate_plan(frame.plan()).is_err());
        Ok(())
    }
}
