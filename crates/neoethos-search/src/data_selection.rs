//! Exact canonical-data boundary for discovery and historical evaluation.
//!
//! The anchor is not a display symbol. It is the complete immutable dataset
//! identity selected by the caller. Every base/higher-timeframe generation and
//! every feature-provenance binding is checked against that anchor before the
//! search receives a row. Missing higher timeframes fail: this module never
//! derives or resamples one from M1.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};

use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1, CanonicalOhlcvFrame,
    CanonicalTimeframe, FeatureBuildOptions, FeatureFrame, IndicatorComputePolicy, Ohlcv,
    ResolvedCanonicalFeatureExecutionAuthorityV1, ResolvedCanonicalFeatureMathLaneV1,
    SymbolDataset, VECTOR_TA_CPU_F64_MATH_AUTHORITY_V1, VECTOR_TA_CUDA_F64_MATH_AUTHORITY_V1,
    load_exact_dataset_series_receipt, prepare_multitimeframe_features_with_options,
    require_direct_timeframes, resolved_canonical_feature_execution_authority_v1,
};
use neoethos_dataset_contracts::CanonicalDatasetScope;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CANONICAL_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V2: u16 = 2;
const CANONICAL_SEARCH_INPUT_RECEIPT_HASH_DOMAIN_V2: &[u8] =
    b"neoethos.canonical-search-input-receipt.v2\0";
const CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V1: &[u8] = b"neoethos.canonical-feature-content.v1\0";
const CANONICAL_FEATURE_EXECUTION_SCHEMA_VERSION_V1: u16 = 1;
const CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2: u16 = 2;
const CANONICAL_SEARCH_ARTIFACT_SCOPE_HASH_DOMAIN_V2: &[u8] =
    b"neoethos.canonical-search-artifact-scope.v2\0";
const CANONICAL_SEARCH_ARTIFACT_ENVELOPE_SCHEMA_VERSION_V2: u16 = 2;

pub const CANONICAL_VECTOR_TA_CPU_MATH_AUTHORITY_V1: &str = VECTOR_TA_CPU_F64_MATH_AUTHORITY_V1;
pub const CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1: &str = VECTOR_TA_CUDA_F64_MATH_AUTHORITY_V1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CanonicalDataSelectionError {
    InventoryFailed {
        requested_symbol: String,
        detail: String,
    },
    AnchorUnavailable {
        anchor_id: String,
        candidate_ids: Vec<String>,
    },
    MissingDirectTimeframe {
        anchor_id: String,
        requested_symbol: String,
        requested_timeframe: CanonicalTimeframe,
        candidate_ids: Vec<String>,
    },
    AmbiguousDirectTimeframe {
        anchor_id: String,
        requested_symbol: String,
        requested_timeframe: CanonicalTimeframe,
        candidate_ids: Vec<String>,
    },
    DatasetOpenFailed {
        anchor_id: String,
        detail: String,
    },
    FeatureBuildFailed {
        anchor_id: String,
        detail: String,
    },
    ProvenanceMismatch {
        anchor_id: String,
        detail: String,
    },
    NoDirectTimeframeRequested {
        anchor_id: String,
        requested_symbol: String,
    },
    InvalidReceipt {
        detail: String,
    },
}

impl fmt::Display for CanonicalDataSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InventoryFailed {
                requested_symbol,
                detail,
            } => write!(
                formatter,
                "canonical dataset inventory failed for {requested_symbol}: {detail}"
            ),
            Self::AnchorUnavailable {
                anchor_id,
                candidate_ids,
            } => write!(
                formatter,
                "selected canonical dataset anchor {anchor_id} is not current; candidates: {}",
                display_candidates(candidate_ids)
            ),
            Self::MissingDirectTimeframe {
                anchor_id,
                requested_symbol,
                requested_timeframe,
                candidate_ids,
            } => write!(
                formatter,
                "selected series {anchor_id} has no direct canonical {requested_symbol} \
                 {requested_timeframe} generation; non-matching candidates: {}. Resampling is \
                 forbidden",
                display_candidates(candidate_ids)
            ),
            Self::AmbiguousDirectTimeframe {
                anchor_id,
                requested_symbol,
                requested_timeframe,
                candidate_ids,
            } => write!(
                formatter,
                "selected series {anchor_id} has multiple direct canonical {requested_symbol} \
                 {requested_timeframe} generations: {}",
                display_candidates(candidate_ids)
            ),
            Self::DatasetOpenFailed { anchor_id, detail } => write!(
                formatter,
                "failed to open exact canonical dataset series {anchor_id}: {detail}"
            ),
            Self::FeatureBuildFailed { anchor_id, detail } => write!(
                formatter,
                "failed to build features from exact canonical dataset series {anchor_id}: \
                 {detail}"
            ),
            Self::ProvenanceMismatch { anchor_id, detail } => write!(
                formatter,
                "canonical search input provenance disagrees with selected series {anchor_id}: \
                 {detail}"
            ),
            Self::NoDirectTimeframeRequested {
                anchor_id,
                requested_symbol,
            } => write!(
                formatter,
                "no direct timeframe preference was supplied for related symbol \
                 {requested_symbol} under anchor {anchor_id}"
            ),
            Self::InvalidReceipt { detail } => {
                write!(
                    formatter,
                    "invalid canonical search input receipt: {detail}"
                )
            }
        }
    }
}

impl Error for CanonicalDataSelectionError {}

fn display_candidates(candidate_ids: &[String]) -> String {
    if candidate_ids.is_empty() {
        "<none>".to_owned()
    } else {
        candidate_ids.join(", ")
    }
}

/// One exact source/account series anchored by a current canonical identity.
#[derive(Clone, Debug)]
pub struct ExactCanonicalSeries {
    root: PathBuf,
    anchor: CanonicalDatasetIdentity,
}

impl ExactCanonicalSeries {
    pub fn open(
        root: impl Into<PathBuf>,
        anchor: CanonicalDatasetIdentity,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let root = root.into();
        let candidates = inventory_for_symbol(&root, anchor.symbol_name())?;
        let exact_count = candidates
            .iter()
            .filter(|candidate| *candidate == &anchor)
            .count();
        if exact_count != 1 {
            return Err(CanonicalDataSelectionError::AnchorUnavailable {
                anchor_id: anchor.to_path_component(),
                candidate_ids: candidate_ids(candidates.iter()),
            });
        }
        Ok(Self { root, anchor })
    }

    pub const fn anchor_identity(&self) -> &CanonicalDatasetIdentity {
        &self.anchor
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Load an exact base + direct higher-timeframe feature cube.
    ///
    /// Every requested timeframe must already have a canonical generation in
    /// the selected series. No resampling or source/account substitution is
    /// reachable through this API.
    pub fn load_search_input(
        &self,
        higher_timeframes: &[CanonicalTimeframe],
    ) -> Result<CanonicalSearchInput, CanonicalDataSelectionError> {
        let mut requested = BTreeSet::from([self.anchor.timeframe()]);
        requested.extend(higher_timeframes.iter().copied());
        for timeframe in &requested {
            self.select_same_series_direct(*timeframe)?;
        }

        let timeframe_names = requested
            .iter()
            .map(|timeframe| timeframe.as_str())
            .collect::<Vec<_>>();
        let dataset = neoethos_data::load_dataset_for_identity_with_timeframes(
            &self.root,
            &self.anchor,
            &timeframe_names,
        )
        .map_err(|error| CanonicalDataSelectionError::DatasetOpenFailed {
            anchor_id: self.anchor.to_path_component(),
            detail: error.to_string(),
        })?;
        verify_direct_artifacts(&self.anchor, &dataset, &requested)?;

        let base_name = self.anchor.timeframe().as_str();
        let base_frame = dataset.canonical_frame(base_name).map_err(|error| {
            CanonicalDataSelectionError::DatasetOpenFailed {
                anchor_id: self.anchor.to_path_component(),
                detail: error.to_string(),
            }
        })?;
        if base_frame.artifact().identity() != &self.anchor {
            return Err(CanonicalDataSelectionError::ProvenanceMismatch {
                anchor_id: self.anchor.to_path_component(),
                detail: format!(
                    "base frame resolved to {}",
                    base_frame.artifact().identity().to_path_component()
                ),
            });
        }

        let higher_names = higher_timeframes
            .iter()
            .copied()
            .filter(|timeframe| *timeframe != self.anchor.timeframe())
            .map(CanonicalTimeframe::as_str)
            .collect::<Vec<_>>();
        let feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        let features =
            neoethos_data::prepare_multitimeframe_features(&dataset, base_name, &higher_names)
                .map_err(|error| CanonicalDataSelectionError::FeatureBuildFailed {
                    anchor_id: self.anchor.to_path_component(),
                    detail: error.to_string(),
                })?;
        let execution_after_build = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        if execution_after_build != feature_execution {
            return Err(CanonicalDataSelectionError::FeatureBuildFailed {
                anchor_id: self.anchor.to_path_component(),
                detail: "canonical feature execution authority changed during the build".to_owned(),
            });
        }
        verify_search_input_provenance(&self.anchor, &dataset, &base_frame, &features)?;

        Ok(CanonicalSearchInput {
            anchor: self.anchor.clone(),
            base_frame,
            features,
            feature_execution,
        })
    }

    /// Select a direct generation for a related symbol in the same data source
    /// (external namespace) or broker environment/server/account.
    ///
    /// cTrader `symbol_id` deliberately differs between the traded pair and a
    /// bridge pair. Two matching IDs for the same broker account are ambiguous
    /// and fail with both opaque candidate identities.
    pub fn select_related_direct(
        &self,
        requested_symbol: &str,
        timeframe_preference: &[CanonicalTimeframe],
    ) -> Result<CanonicalDatasetIdentity, CanonicalDataSelectionError> {
        if timeframe_preference.is_empty() {
            return Err(CanonicalDataSelectionError::NoDirectTimeframeRequested {
                anchor_id: self.anchor.to_path_component(),
                requested_symbol: requested_symbol.to_owned(),
            });
        }
        let inventory = inventory_for_symbol(&self.root, requested_symbol)?;
        let mut all_preferred_candidates = Vec::new();
        for timeframe in timeframe_preference {
            let at_timeframe = inventory
                .iter()
                .filter(|candidate| {
                    candidate.symbol_name() == requested_symbol
                        && candidate.timeframe() == *timeframe
                })
                .collect::<Vec<_>>();
            all_preferred_candidates.extend(at_timeframe.iter().copied());
            let matching = at_timeframe
                .into_iter()
                .filter(|candidate| same_source_account(candidate, &self.anchor))
                .collect::<Vec<_>>();
            match matching.as_slice() {
                [identity] => return Ok((**identity).clone()),
                [] => {}
                _ => {
                    return Err(CanonicalDataSelectionError::AmbiguousDirectTimeframe {
                        anchor_id: self.anchor.to_path_component(),
                        requested_symbol: requested_symbol.to_owned(),
                        requested_timeframe: *timeframe,
                        candidate_ids: candidate_ids(matching),
                    });
                }
            }
        }
        Err(CanonicalDataSelectionError::MissingDirectTimeframe {
            anchor_id: self.anchor.to_path_component(),
            requested_symbol: requested_symbol.to_owned(),
            requested_timeframe: timeframe_preference[0],
            candidate_ids: candidate_ids(all_preferred_candidates),
        })
    }

    pub fn load_related_direct(
        &self,
        requested_symbol: &str,
        timeframe_preference: &[CanonicalTimeframe],
    ) -> Result<CanonicalOhlcvFrame, CanonicalDataSelectionError> {
        let identity = self.select_related_direct(requested_symbol, timeframe_preference)?;
        neoethos_data::load_canonical_timeframe(&self.root, &identity).map_err(|error| {
            CanonicalDataSelectionError::DatasetOpenFailed {
                anchor_id: self.anchor.to_path_component(),
                detail: format!(
                    "related identity {} failed verification: {error}",
                    identity.to_path_component()
                ),
            }
        })
    }

    fn select_same_series_direct(
        &self,
        timeframe: CanonicalTimeframe,
    ) -> Result<CanonicalDatasetIdentity, CanonicalDataSelectionError> {
        let inventory = inventory_for_symbol(&self.root, self.anchor.symbol_name())?;
        let at_timeframe = inventory
            .iter()
            .filter(|candidate| candidate.timeframe() == timeframe)
            .collect::<Vec<_>>();
        let matching = at_timeframe
            .iter()
            .copied()
            .filter(|candidate| same_exact_series(candidate, &self.anchor))
            .collect::<Vec<_>>();
        match matching.as_slice() {
            [identity] => Ok((**identity).clone()),
            [] => Err(CanonicalDataSelectionError::MissingDirectTimeframe {
                anchor_id: self.anchor.to_path_component(),
                requested_symbol: self.anchor.symbol_name().to_owned(),
                requested_timeframe: timeframe,
                candidate_ids: candidate_ids(at_timeframe),
            }),
            _ => Err(CanonicalDataSelectionError::AmbiguousDirectTimeframe {
                anchor_id: self.anchor.to_path_component(),
                requested_symbol: self.anchor.symbol_name().to_owned(),
                requested_timeframe: timeframe,
                candidate_ids: candidate_ids(matching),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalSearchInput {
    anchor: CanonicalDatasetIdentity,
    base_frame: CanonicalOhlcvFrame,
    features: FeatureFrame,
    feature_execution: CanonicalFeatureExecutionReceiptV1,
}

impl CanonicalSearchInput {
    /// Own an already-built canonical CPU input only after recomputing its
    /// runtime math authority and revalidating the exact receipt/base-frame
    /// provenance. This is the CPU factory boundary used after a sealed
    /// cross-vendor physical-GPU absence admission.
    pub fn from_prepared_canonical_frame(
        anchor: CanonicalDatasetIdentity,
        base_frame: CanonicalOhlcvFrame,
        features: FeatureFrame,
    ) -> Result<CanonicalSearchInput, CanonicalDataSelectionError> {
        if base_frame.artifact().identity() != &anchor {
            return Err(provenance_mismatch(
                &anchor,
                "prepared canonical base frame does not match the selected anchor identity",
            ));
        }
        let feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame_with_execution(
            &anchor,
            &features,
            feature_execution.clone(),
        )?;
        CanonicalSearchRunInputV2::new(receipt, &features, &base_frame)?;
        Ok(CanonicalSearchInput {
            anchor,
            base_frame,
            features,
            feature_execution,
        })
    }

    /// Build the search cube from one explicitly selected immutable series.
    /// Every base/higher timeframe is reopened by its generation receipt; no
    /// inventory or current-generation selection participates in this path.
    pub fn from_exact_series_receipt(
        root: impl AsRef<Path>,
        series: &CanonicalDatasetSeriesReceiptV1,
        base_timeframe: CanonicalTimeframe,
        options: &FeatureBuildOptions,
    ) -> Result<CanonicalSearchInput, CanonicalDataSelectionError> {
        series
            .validate()
            .map_err(|error| CanonicalDataSelectionError::DatasetOpenFailed {
                anchor_id: series.anchor().identity().to_path_component(),
                detail: error.to_string(),
            })?;
        let requested = std::iter::once(Ok(base_timeframe))
            .chain(options.higher_tfs.iter().map(|timeframe| {
                timeframe
                    .parse::<CanonicalTimeframe>()
                    .map_err(|error| error.to_string())
            }))
            .collect::<Result<BTreeSet<_>, _>>()
            .map_err(|detail| CanonicalDataSelectionError::DatasetOpenFailed {
                anchor_id: series.anchor().identity().to_path_component(),
                detail: format!("non-canonical requested feature timeframe: {detail}"),
            })?;
        let selected = series
            .direct_timeframes()
            .iter()
            .map(|receipt| (receipt.identity().timeframe(), receipt))
            .collect::<std::collections::BTreeMap<_, _>>();
        for timeframe in &requested {
            if !selected.contains_key(timeframe) {
                return Err(CanonicalDataSelectionError::MissingDirectTimeframe {
                    anchor_id: series.anchor().identity().to_path_component(),
                    requested_symbol: series.anchor().identity().symbol_name().to_owned(),
                    requested_timeframe: *timeframe,
                    candidate_ids: series
                        .direct_timeframes()
                        .iter()
                        .map(|receipt| receipt.identity().to_path_component())
                        .collect(),
                });
            }
        }

        let dataset = load_exact_dataset_series_receipt(root, series).map_err(|error| {
            CanonicalDataSelectionError::DatasetOpenFailed {
                anchor_id: series.anchor().identity().to_path_component(),
                detail: error.to_string(),
            }
        })?;
        let base_selected = selected
            .get(&base_timeframe)
            .expect("requested base timeframe was proved present");
        let anchor = base_selected.identity().clone();
        let base_name = base_timeframe.as_str();
        let base_frame = dataset.canonical_frame(base_name).map_err(|error| {
            CanonicalDataSelectionError::DatasetOpenFailed {
                anchor_id: anchor.to_path_component(),
                detail: error.to_string(),
            }
        })?;
        if base_frame.artifact().identity() != &anchor {
            return Err(CanonicalDataSelectionError::ProvenanceMismatch {
                anchor_id: anchor.to_path_component(),
                detail: "exact base generation reopened with a different identity".to_owned(),
            });
        }
        let feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        let features = prepare_multitimeframe_features_with_options(&dataset, base_name, options)
            .map_err(|error| CanonicalDataSelectionError::FeatureBuildFailed {
            anchor_id: anchor.to_path_component(),
            detail: error.to_string(),
        })?;
        let execution_after_build = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        if execution_after_build != feature_execution {
            return Err(CanonicalDataSelectionError::FeatureBuildFailed {
                anchor_id: anchor.to_path_component(),
                detail: "canonical feature execution authority changed during the build".to_owned(),
            });
        }
        verify_search_input_provenance(&anchor, &dataset, &base_frame, &features)?;
        Ok(CanonicalSearchInput {
            anchor,
            base_frame,
            features,
            feature_execution,
        })
    }

    pub const fn anchor_identity(&self) -> &CanonicalDatasetIdentity {
        &self.anchor
    }

    pub const fn base_frame(&self) -> &CanonicalOhlcvFrame {
        &self.base_frame
    }

    pub const fn features(&self) -> &FeatureFrame {
        &self.features
    }

    /// Serializable, content-addressed proof of the exact data consumed by the
    /// search. Values stay opaque; no display symbol is promoted to identity.
    pub fn receipt(&self) -> Result<CanonicalSearchInputReceiptV2, CanonicalDataSelectionError> {
        CanonicalSearchInputReceiptV2::from_feature_frame_with_execution(
            &self.anchor,
            &self.features,
            self.feature_execution.clone(),
        )
    }

    pub fn as_run_input(
        &self,
    ) -> Result<CanonicalSearchRunInputV2<'_>, CanonicalDataSelectionError> {
        CanonicalSearchRunInputV2::new(self.receipt()?, &self.features, &self.base_frame)
    }
}

/// The only data shape accepted by production discovery entrypoints.
///
/// The receipt is owned so the exact source generations and semantic feature
/// identities cannot disappear while borrowed values are evaluated. Creation
/// checks both the receipt-to-FeatureFrame binding and exact OHLCV timestamp
/// alignment; equal row counts alone are never treated as provenance.
#[derive(Debug)]
pub struct CanonicalSearchRunInputV2<'a> {
    receipt: CanonicalSearchInputReceiptV2,
    anchor: CanonicalDatasetIdentity,
    features: &'a FeatureFrame,
    ohlcv: &'a Ohlcv,
}

impl<'a> CanonicalSearchRunInputV2<'a> {
    pub fn new(
        receipt: CanonicalSearchInputReceiptV2,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let ohlcv = base_frame.ohlcv();
        let anchor = Self::validate_values(&receipt, features, ohlcv)?;
        if base_frame.artifact().identity() != &anchor {
            return Err(provenance_mismatch(
                &anchor,
                format!(
                    "base frame identity {} does not match receipt anchor {}",
                    base_frame.artifact().identity().to_path_component(),
                    anchor.to_path_component()
                ),
            ));
        }
        let anchor_id = anchor.to_path_component();
        let receipt_binding = receipt
            .source_bindings()
            .iter()
            .find(|binding| binding.dataset_identity() == anchor_id)
            .expect("validate_values requires exactly one anchor binding");
        let frame_binding = base_frame
            .source_binding(receipt_binding.source_node_id())
            .map_err(|error| {
                provenance_mismatch(&anchor, format!("binding exact base frame: {error}"))
            })?;
        if receipt_binding.dataset_identity()
            != frame_binding.dataset_identity().to_path_component()
            || receipt_binding.manifest_schema_id() != frame_binding.manifest_schema_id()
            || receipt_binding.manifest_sha256() != hex(frame_binding.manifest_hash())
            || receipt_binding.generation_id() != frame_binding.generation_id()
            || receipt_binding.vortex_sha256() != hex(frame_binding.vortex_hash())
            || receipt_binding.bar_timestamp_convention()
                != frame_binding.bar_timestamp_convention().to_string()
            || receipt_binding.segments().len() != frame_binding.segments().len()
            || receipt_binding
                .segments()
                .iter()
                .zip(frame_binding.segments())
                .any(|(receipt_segment, frame_segment)| {
                    receipt_segment.row_start() != frame_segment.row_start()
                        || receipt_segment.row_end() != frame_segment.row_end()
                        || receipt_segment.timestamp_start_ms()
                            != frame_segment.timestamp_start_ms()
                        || receipt_segment.timestamp_end_ms() != frame_segment.timestamp_end_ms()
                })
        {
            return Err(provenance_mismatch(
                &anchor,
                "base frame immutable artifact/segment does not match the receipt anchor binding",
            ));
        }
        Ok(Self {
            receipt,
            anchor,
            features,
            ohlcv,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_test_values(
        receipt: CanonicalSearchInputReceiptV2,
        features: &'a FeatureFrame,
        ohlcv: &'a Ohlcv,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let anchor = Self::validate_values(&receipt, features, ohlcv)?;
        Ok(Self {
            receipt,
            anchor,
            features,
            ohlcv,
        })
    }

    fn validate_values(
        receipt: &CanonicalSearchInputReceiptV2,
        features: &FeatureFrame,
        ohlcv: &Ohlcv,
    ) -> Result<CanonicalDatasetIdentity, CanonicalDataSelectionError> {
        let anchor = receipt.validate()?;
        receipt.validate_against(&anchor, features)?;
        let timestamps = ohlcv.timestamp.as_deref().ok_or_else(|| {
            provenance_mismatch(&anchor, "base OHLCV has no canonical timestamps")
        })?;
        if ohlcv.open.len() != timestamps.len()
            || ohlcv.high.len() != timestamps.len()
            || ohlcv.low.len() != timestamps.len()
            || ohlcv.close.len() != timestamps.len()
            || ohlcv
                .volume
                .as_ref()
                .is_some_and(|volume| volume.len() != timestamps.len())
        {
            return Err(provenance_mismatch(
                &anchor,
                "base OHLCV column lengths disagree",
            ));
        }
        if features.n_samples() != timestamps.len() {
            return Err(provenance_mismatch(
                &anchor,
                format!(
                    "feature/OHLCV row-count mismatch: {} vs {}",
                    features.n_samples(),
                    timestamps.len()
                ),
            ));
        }
        if features.timestamps.as_slice() != timestamps {
            return Err(provenance_mismatch(
                &anchor,
                "feature/OHLCV timestamp mismatch",
            ));
        }
        let anchor_id = anchor.to_path_component();
        let anchor_bindings = receipt
            .source_bindings
            .iter()
            .filter(|binding| binding.dataset_identity == anchor_id)
            .collect::<Vec<_>>();
        if anchor_bindings.len() != 1 {
            return Err(provenance_mismatch(
                &anchor,
                format!(
                    "receipt must contain exactly one anchor source binding; found {}",
                    anchor_bindings.len()
                ),
            ));
        }
        let segments = &anchor_bindings[0].segments;
        let consumed_rows = segments.iter().try_fold(0_u64, |total, segment| {
            total
                .checked_add(segment.row_end - segment.row_start)
                .ok_or_else(|| provenance_mismatch(&anchor, "anchor segment row-count overflow"))
        })?;
        if consumed_rows != timestamps.len() as u64
            || segments.first().map(|segment| segment.timestamp_start_ms)
                != timestamps.first().copied()
            || segments.last().map(|segment| segment.timestamp_end_ms) != timestamps.last().copied()
        {
            return Err(provenance_mismatch(
                &anchor,
                "anchor consumed segments do not cover the exact OHLCV row/timestamp range",
            ));
        }
        Ok(anchor)
    }

    pub const fn receipt(&self) -> &CanonicalSearchInputReceiptV2 {
        &self.receipt
    }

    pub const fn anchor_identity(&self) -> &CanonicalDatasetIdentity {
        &self.anchor
    }

    pub const fn features(&self) -> &FeatureFrame {
        self.features
    }

    pub const fn ohlcv(&self) -> &Ohlcv {
        self.ohlcv
    }
}

/// Versioned identity of the exact feature payload consumed by search.
///
/// V2 binds ordered timestamps and names, every f64 payload bit, every typed
/// validity code, the vector-ta math authority and selected execution lane,
/// plus the immutable source/plan provenance. V1 has no active alias or
/// defaulting decoder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchInputReceiptV2 {
    schema_version: u16,
    anchor_dataset_identity: String,
    feature_plan_identity: String,
    feature_provenance_identity: String,
    feature_content_sha256: String,
    feature_execution: CanonicalFeatureExecutionReceiptV1,
    source_bindings: Vec<CanonicalSearchSourceBindingReceiptV1>,
}

/// Exclusive production policy recorded by a canonical feature receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalFeatureComputePolicyV1 {
    Auto,
    CpuOnly,
    GpuOnly,
}

/// Exact arithmetic lane selected by vector-ta or the strict CUDA graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanonicalFeatureMathLaneV1 {
    #[serde(rename = "cpu_scalar")]
    CpuScalar,
    #[serde(rename = "cpu_avx2_fma")]
    CpuAvx2Fma,
    #[serde(rename = "cpu_avx512f_dq_vl_bw_avx2_fma")]
    CpuAvx512FDqVlBwAvx2Fma,
    #[serde(rename = "gpu_cuda_f64_strict")]
    GpuCudaF64Strict,
}

/// Versioned producer authority for the exact feature payload bits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFeatureExecutionReceiptV1 {
    schema_version: u16,
    compute_policy: CanonicalFeatureComputePolicyV1,
    vector_ta_math_authority: String,
    selected_lane: CanonicalFeatureMathLaneV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchSourceBindingReceiptV1 {
    source_node_id: String,
    dataset_identity: String,
    manifest_schema_id: String,
    manifest_sha256: String,
    generation_id: String,
    vortex_sha256: String,
    bar_timestamp_convention: String,
    segments: Vec<CanonicalSearchSourceSegmentReceiptV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchSourceSegmentReceiptV1 {
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
}

/// Semantic role of one exact evaluated window inside a receipt-bound search.
///
/// Roles are serialized as part of the artifact identity so in-sample,
/// holdout, and later validation evidence cannot be substituted for each
/// other even when their row boundaries happen to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalSearchWindowRoleV1 {
    DiscoveryInput,
    InSample,
    Holdout,
    WalkForwardTrain,
    WalkForwardValidation,
    ForwardTest,
    LiveSimulation,
    PropFirmRisk,
}

/// One non-empty half-open source-row window evaluated by a search artifact.
///
/// Row offsets are absolute offsets in the anchor source generation, not
/// offsets reconstructed from a display symbol or from the artifact file's
/// current location. Timestamps name the exact first and last consumed bars.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchEvaluatedWindowV1 {
    role: CanonicalSearchWindowRoleV1,
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
}

/// Durable authority for every search-derived artifact.
///
/// The full canonical receipt is embedded alongside its recomputed digest and
/// an explicit evaluated window. A neighboring sidecar, symbol/timeframe file
/// name, or currently-published dataset can never supply missing authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchArtifactScopeV2 {
    schema_version: u16,
    receipt: CanonicalSearchInputReceiptV2,
    receipt_sha256: String,
    evaluated_window: CanonicalSearchEvaluatedWindowV1,
}

/// Strict self-contained envelope used by search result writers.
///
/// The payload can move between files or machines without losing the exact
/// receipt/window authority because the scope is part of the same serialized
/// object. `artifact_kind` prevents a valid portfolio envelope from being
/// reinterpreted as quality, trade, funnel, or promotion evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchArtifactEnvelopeV2<T> {
    schema_version: u16,
    artifact_kind: String,
    scope: CanonicalSearchArtifactScopeV2,
    search_config_hash: String,
    payload: T,
}

impl CanonicalSearchEvaluatedWindowV1 {
    pub fn new(
        role: CanonicalSearchWindowRoleV1,
        row_start: u64,
        row_end: u64,
        timestamp_start_ms: i64,
        timestamp_end_ms: i64,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let window = Self {
            role,
            row_start,
            row_end,
            timestamp_start_ms,
            timestamp_end_ms,
        };
        window.validate_shape()?;
        Ok(window)
    }

    pub const fn role(&self) -> CanonicalSearchWindowRoleV1 {
        self.role
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

    fn validate_shape(&self) -> Result<(), CanonicalDataSelectionError> {
        if self.row_start >= self.row_end {
            return Err(invalid_receipt(
                "evaluated window has an empty or reversed source-row range",
            ));
        }
        if self.timestamp_start_ms > self.timestamp_end_ms {
            return Err(invalid_receipt(
                "evaluated window has reversed first/last timestamps",
            ));
        }
        Ok(())
    }

    fn validate_against_receipt(
        &self,
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> Result<(), CanonicalDataSelectionError> {
        self.validate_shape()?;
        let anchor = receipt.validate()?;
        let anchor_id = anchor.to_path_component();
        let anchor_bindings = receipt
            .source_bindings()
            .iter()
            .filter(|binding| binding.dataset_identity() == anchor_id)
            .collect::<Vec<_>>();
        if anchor_bindings.len() != 1 {
            return Err(provenance_mismatch(
                &anchor,
                format!(
                    "artifact scope requires exactly one anchor source binding; found {}",
                    anchor_bindings.len()
                ),
            ));
        }
        let segments = anchor_bindings[0].segments();
        let first = segments
            .first()
            .ok_or_else(|| provenance_mismatch(&anchor, "anchor source has no segments"))?;
        let last = segments
            .last()
            .ok_or_else(|| provenance_mismatch(&anchor, "anchor source has no segments"))?;
        if self.timestamp_start_ms < first.timestamp_start_ms()
            || self.timestamp_end_ms > last.timestamp_end_ms()
        {
            return Err(provenance_mismatch(
                &anchor,
                "evaluated timestamps fall outside the receipt anchor segments",
            ));
        }

        let mut cursor = self.row_start;
        for segment in segments {
            if cursor >= self.row_end {
                break;
            }
            if cursor < segment.row_start() || cursor >= segment.row_end() {
                continue;
            }
            cursor = self.row_end.min(segment.row_end());
        }
        if cursor != self.row_end {
            return Err(provenance_mismatch(
                &anchor,
                "evaluated source-row window is not fully covered by contiguous anchor segments",
            ));
        }
        Ok(())
    }
}

impl CanonicalSearchArtifactScopeV2 {
    pub fn new(
        receipt: CanonicalSearchInputReceiptV2,
        evaluated_window: CanonicalSearchEvaluatedWindowV1,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let receipt_sha256 = receipt.identity_sha256()?;
        let scope = Self {
            schema_version: CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2,
            receipt,
            receipt_sha256,
            evaluated_window,
        };
        scope.validate()?;
        Ok(scope)
    }

    pub fn from_run_input(
        role: CanonicalSearchWindowRoleV1,
        input: &CanonicalSearchRunInputV2<'_>,
    ) -> Result<Self, CanonicalDataSelectionError> {
        Self::from_run_input_range(role, input, 0..input.ohlcv().len())
    }

    pub fn for_entire_receipt(
        role: CanonicalSearchWindowRoleV1,
        receipt: CanonicalSearchInputReceiptV2,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let anchor = receipt.validate()?;
        let anchor_id = anchor.to_path_component();
        let anchor_bindings = receipt
            .source_bindings()
            .iter()
            .filter(|binding| binding.dataset_identity() == anchor_id)
            .collect::<Vec<_>>();
        if anchor_bindings.len() != 1 {
            return Err(provenance_mismatch(
                &anchor,
                format!(
                    "artifact scope requires exactly one anchor source binding; found {}",
                    anchor_bindings.len()
                ),
            ));
        }
        let segments = anchor_bindings[0].segments();
        for adjacent in segments.windows(2) {
            if adjacent[0].row_end() != adjacent[1].row_start() {
                return Err(provenance_mismatch(
                    &anchor,
                    "one entire-receipt window cannot represent disjoint anchor segments",
                ));
            }
        }
        let first = segments
            .first()
            .ok_or_else(|| provenance_mismatch(&anchor, "anchor source has no segments"))?;
        let last = segments
            .last()
            .ok_or_else(|| provenance_mismatch(&anchor, "anchor source has no segments"))?;
        let window = CanonicalSearchEvaluatedWindowV1::new(
            role,
            first.row_start(),
            last.row_end(),
            first.timestamp_start_ms(),
            last.timestamp_end_ms(),
        )?;
        Self::new(receipt, window)
    }

    pub fn from_run_input_range(
        role: CanonicalSearchWindowRoleV1,
        input: &CanonicalSearchRunInputV2<'_>,
        range: Range<usize>,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let timestamps = input.ohlcv().timestamp.as_deref().ok_or_else(|| {
            provenance_mismatch(input.anchor_identity(), "base OHLCV has no timestamps")
        })?;
        if range.start >= range.end || range.end > timestamps.len() {
            return Err(provenance_mismatch(
                input.anchor_identity(),
                format!(
                    "evaluated input row range {}..{} is empty or exceeds {} rows",
                    range.start,
                    range.end,
                    timestamps.len()
                ),
            ));
        }
        let anchor_id = input.anchor_identity().to_path_component();
        let anchor_bindings = input
            .receipt()
            .source_bindings()
            .iter()
            .filter(|binding| binding.dataset_identity() == anchor_id)
            .collect::<Vec<_>>();
        if anchor_bindings.len() != 1 {
            return Err(provenance_mismatch(
                input.anchor_identity(),
                format!(
                    "artifact scope requires exactly one anchor source binding; found {}",
                    anchor_bindings.len()
                ),
            ));
        }
        let segments = anchor_bindings[0].segments();
        for adjacent in segments.windows(2) {
            if adjacent[0].row_end() != adjacent[1].row_start() {
                return Err(provenance_mismatch(
                    input.anchor_identity(),
                    "a single evaluated row range cannot represent disjoint anchor segments",
                ));
            }
        }
        let source_row_start = segments
            .first()
            .ok_or_else(|| provenance_mismatch(input.anchor_identity(), "anchor has no segments"))?
            .row_start()
            .checked_add(range.start as u64)
            .ok_or_else(|| provenance_mismatch(input.anchor_identity(), "row-start overflow"))?;
        let source_row_end = segments
            .first()
            .expect("segments checked non-empty")
            .row_start()
            .checked_add(range.end as u64)
            .ok_or_else(|| provenance_mismatch(input.anchor_identity(), "row-end overflow"))?;
        let window = CanonicalSearchEvaluatedWindowV1::new(
            role,
            source_row_start,
            source_row_end,
            timestamps[range.start],
            timestamps[range.end - 1],
        )?;
        Self::new(input.receipt().clone(), window)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn receipt(&self) -> &CanonicalSearchInputReceiptV2 {
        &self.receipt
    }

    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub const fn evaluated_window(&self) -> &CanonicalSearchEvaluatedWindowV1 {
        &self.evaluated_window
    }

    pub fn validate(&self) -> Result<(), CanonicalDataSelectionError> {
        if self.schema_version != CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2 {
            return Err(invalid_receipt(format!(
                "unsupported artifact-scope schema version {}; expected {}",
                self.schema_version, CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2
            )));
        }
        let recomputed = self.receipt.identity_sha256()?;
        if self.receipt_sha256 != recomputed {
            return Err(invalid_receipt(
                "artifact-scope receipt SHA-256 does not match the embedded receipt",
            ));
        }
        self.evaluated_window
            .validate_against_receipt(&self.receipt)
    }

    pub fn validate_against_receipt(
        &self,
        expected_receipt: &CanonicalSearchInputReceiptV2,
    ) -> Result<(), CanonicalDataSelectionError> {
        self.validate()?;
        expected_receipt.validate()?;
        if &self.receipt != expected_receipt {
            let expected = expected_receipt.identity_sha256()?;
            return Err(invalid_receipt(format!(
                "artifact receipt {} does not match expected receipt {expected}",
                self.receipt_sha256
            )));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        expected_receipt: &CanonicalSearchInputReceiptV2,
        expected_window: &CanonicalSearchEvaluatedWindowV1,
    ) -> Result<(), CanonicalDataSelectionError> {
        self.validate_against_receipt(expected_receipt)?;
        if &self.evaluated_window != expected_window {
            return Err(invalid_receipt(
                "artifact evaluated window does not match the expected role/rows/timestamps",
            ));
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String, CanonicalDataSelectionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize artifact scope: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_SEARCH_ARTIFACT_SCOPE_HASH_DOMAIN_V2);
        hasher.update(bytes);
        Ok(hex(&hasher.finalize()))
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, CanonicalDataSelectionError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize artifact scope JSON: {error}")))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, CanonicalDataSelectionError> {
        let scope: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid_receipt(format!("parse artifact scope JSON: {error}")))?;
        scope.validate()?;
        Ok(scope)
    }
}

impl<T> CanonicalSearchArtifactEnvelopeV2<T> {
    pub fn new(
        artifact_kind: impl Into<String>,
        scope: CanonicalSearchArtifactScopeV2,
        search_config_hash: impl Into<String>,
        payload: T,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let envelope = Self {
            schema_version: CANONICAL_SEARCH_ARTIFACT_ENVELOPE_SCHEMA_VERSION_V2,
            artifact_kind: artifact_kind.into(),
            scope,
            search_config_hash: search_config_hash.into(),
            payload,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn artifact_kind(&self) -> &str {
        &self.artifact_kind
    }

    pub const fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.scope
    }

    pub fn search_config_hash(&self) -> &str {
        &self.search_config_hash
    }

    pub const fn payload(&self) -> &T {
        &self.payload
    }

    pub fn into_payload(self) -> T {
        self.payload
    }

    pub fn validate(&self) -> Result<(), CanonicalDataSelectionError> {
        if self.schema_version != CANONICAL_SEARCH_ARTIFACT_ENVELOPE_SCHEMA_VERSION_V2 {
            return Err(invalid_receipt(format!(
                "unsupported artifact-envelope schema version {}; expected {}",
                self.schema_version, CANONICAL_SEARCH_ARTIFACT_ENVELOPE_SCHEMA_VERSION_V2
            )));
        }
        validate_artifact_kind(&self.artifact_kind)?;
        validate_search_config_hash(&self.search_config_hash)?;
        self.scope.validate()
    }

    pub fn validate_against(
        &self,
        expected_kind: &str,
        expected_search_config_hash: &str,
        expected_receipt: &CanonicalSearchInputReceiptV2,
        expected_window: &CanonicalSearchEvaluatedWindowV1,
    ) -> Result<(), CanonicalDataSelectionError> {
        self.validate()?;
        validate_artifact_kind(expected_kind)?;
        if self.artifact_kind != expected_kind {
            return Err(invalid_receipt(format!(
                "artifact kind `{}` does not match expected `{expected_kind}`",
                self.artifact_kind
            )));
        }
        validate_search_config_hash(expected_search_config_hash)?;
        if self.search_config_hash != expected_search_config_hash {
            return Err(invalid_receipt(format!(
                "artifact search config hash `{}` does not match expected `{expected_search_config_hash}`",
                self.search_config_hash
            )));
        }
        self.scope
            .validate_against(expected_receipt, expected_window)
    }
}

impl<T> CanonicalSearchArtifactEnvelopeV2<T>
where
    T: Serialize,
{
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, CanonicalDataSelectionError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize artifact envelope: {error}")))
    }
}

impl<T> CanonicalSearchArtifactEnvelopeV2<T>
where
    T: DeserializeOwned,
{
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, CanonicalDataSelectionError> {
        let envelope: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid_receipt(format!("parse artifact envelope: {error}")))?;
        envelope.validate()?;
        Ok(envelope)
    }
}

impl CanonicalFeatureExecutionReceiptV1 {
    fn from_runtime_authority(authority: ResolvedCanonicalFeatureExecutionAuthorityV1) -> Self {
        let compute_policy = match authority.policy {
            IndicatorComputePolicy::Auto => CanonicalFeatureComputePolicyV1::Auto,
            IndicatorComputePolicy::CpuOnly => CanonicalFeatureComputePolicyV1::CpuOnly,
            IndicatorComputePolicy::GpuOnly => CanonicalFeatureComputePolicyV1::GpuOnly,
        };
        let selected_lane = match authority.selected_lane {
            ResolvedCanonicalFeatureMathLaneV1::CpuScalar => CanonicalFeatureMathLaneV1::CpuScalar,
            ResolvedCanonicalFeatureMathLaneV1::CpuAvx2Fma => {
                CanonicalFeatureMathLaneV1::CpuAvx2Fma
            }
            ResolvedCanonicalFeatureMathLaneV1::CpuAvx512F64Avx2FmaDqVlBw => {
                CanonicalFeatureMathLaneV1::CpuAvx512FDqVlBwAvx2Fma
            }
            ResolvedCanonicalFeatureMathLaneV1::GpuCudaF64Strict => {
                CanonicalFeatureMathLaneV1::GpuCudaF64Strict
            }
        };
        Self {
            schema_version: CANONICAL_FEATURE_EXECUTION_SCHEMA_VERSION_V1,
            compute_policy,
            vector_ta_math_authority: authority.vector_ta_math_authority.to_owned(),
            selected_lane,
        }
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn compute_policy(&self) -> CanonicalFeatureComputePolicyV1 {
        self.compute_policy
    }

    pub fn vector_ta_math_authority(&self) -> &str {
        &self.vector_ta_math_authority
    }

    pub const fn selected_lane(&self) -> CanonicalFeatureMathLaneV1 {
        self.selected_lane
    }

    fn validate(&self) -> Result<(), CanonicalDataSelectionError> {
        if self.schema_version != CANONICAL_FEATURE_EXECUTION_SCHEMA_VERSION_V1 {
            return Err(invalid_receipt(format!(
                "unsupported feature-execution schema version {}; expected {}",
                self.schema_version, CANONICAL_FEATURE_EXECUTION_SCHEMA_VERSION_V1
            )));
        }
        let (expected_authority, lane_is_compatible) = match self.compute_policy {
            CanonicalFeatureComputePolicyV1::Auto | CanonicalFeatureComputePolicyV1::CpuOnly => (
                CANONICAL_VECTOR_TA_CPU_MATH_AUTHORITY_V1,
                matches!(
                    self.selected_lane,
                    CanonicalFeatureMathLaneV1::CpuScalar
                        | CanonicalFeatureMathLaneV1::CpuAvx2Fma
                        | CanonicalFeatureMathLaneV1::CpuAvx512FDqVlBwAvx2Fma
                ),
            ),
            CanonicalFeatureComputePolicyV1::GpuOnly => (
                CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1,
                self.selected_lane == CanonicalFeatureMathLaneV1::GpuCudaF64Strict,
            ),
        };
        if !lane_is_compatible {
            return Err(invalid_receipt(
                "feature compute policy and selected arithmetic lane disagree",
            ));
        }
        if self.vector_ta_math_authority != expected_authority {
            return Err(invalid_receipt(format!(
                "feature math authority `{}` does not match selected lane `{}`",
                self.vector_ta_math_authority,
                canonical_feature_math_lane_name(self.selected_lane)
            )));
        }
        Ok(())
    }
}

impl CanonicalSearchInputReceiptV2 {
    pub fn from_feature_frame(
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
    ) -> Result<Self, CanonicalDataSelectionError> {
        Self::from_feature_frame_with_execution(
            anchor,
            features,
            CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
                resolved_canonical_feature_execution_authority_v1(),
            ),
        )
    }

    fn from_feature_frame_with_execution(
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
        feature_execution: CanonicalFeatureExecutionReceiptV1,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let receipt = Self {
            schema_version: CANONICAL_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V2,
            anchor_dataset_identity: anchor.to_path_component(),
            feature_plan_identity: features.plan_identity().to_hex(),
            feature_provenance_identity: features.provenance_identity().to_hex(),
            feature_content_sha256: canonical_feature_content_sha256(features)?,
            feature_execution,
            source_bindings: source_binding_receipts(features),
        };
        receipt.validate_against(anchor, features)?;
        Ok(receipt)
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn anchor_dataset_identity(&self) -> &str {
        &self.anchor_dataset_identity
    }

    pub fn feature_plan_identity(&self) -> &str {
        &self.feature_plan_identity
    }

    pub fn feature_provenance_identity(&self) -> &str {
        &self.feature_provenance_identity
    }

    pub fn feature_content_sha256(&self) -> &str {
        &self.feature_content_sha256
    }

    pub const fn feature_execution(&self) -> &CanonicalFeatureExecutionReceiptV1 {
        &self.feature_execution
    }

    pub fn source_bindings(&self) -> &[CanonicalSearchSourceBindingReceiptV1] {
        &self.source_bindings
    }

    pub fn validate(&self) -> Result<CanonicalDatasetIdentity, CanonicalDataSelectionError> {
        if self.schema_version != CANONICAL_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V2 {
            return Err(invalid_receipt(format!(
                "unsupported schema version {}; expected {}",
                self.schema_version, CANONICAL_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V2
            )));
        }
        let anchor = CanonicalDatasetIdentity::from_path_component(&self.anchor_dataset_identity)
            .map_err(|error| invalid_receipt(format!("anchor identity: {error}")))?;
        validate_sha256_hex("feature plan identity", &self.feature_plan_identity)?;
        validate_sha256_hex(
            "feature provenance identity",
            &self.feature_provenance_identity,
        )?;
        validate_sha256_hex("feature content SHA-256", &self.feature_content_sha256)?;
        self.feature_execution.validate()?;
        if self.source_bindings.is_empty() {
            return Err(invalid_receipt("source bindings are empty"));
        }
        let mut previous_node: Option<&str> = None;
        let mut contains_anchor = false;
        for binding in &self.source_bindings {
            validate_nonempty("source node id", &binding.source_node_id)?;
            if previous_node.is_some_and(|previous| previous >= binding.source_node_id.as_str()) {
                return Err(invalid_receipt(format!(
                    "source bindings are not strictly ordered or contain duplicate node `{}`",
                    binding.source_node_id
                )));
            }
            previous_node = Some(&binding.source_node_id);
            let identity = CanonicalDatasetIdentity::from_path_component(&binding.dataset_identity)
                .map_err(|error| {
                    invalid_receipt(format!(
                        "source node `{}` dataset identity: {error}",
                        binding.source_node_id
                    ))
                })?;
            contains_anchor |= identity == anchor;
            validate_nonempty("manifest schema id", &binding.manifest_schema_id)?;
            validate_sha256_hex("manifest SHA-256", &binding.manifest_sha256)?;
            validate_nonempty("generation id", &binding.generation_id)?;
            validate_sha256_hex("Vortex SHA-256", &binding.vortex_sha256)?;
            if binding.bar_timestamp_convention != identity.bar_timestamp_convention().to_string() {
                return Err(invalid_receipt(format!(
                    "source node `{}` bar timestamp convention disagrees with its dataset identity",
                    binding.source_node_id
                )));
            }
            validate_segments(&binding.source_node_id, &binding.segments)?;
        }
        if !contains_anchor {
            return Err(invalid_receipt(format!(
                "anchor {} has no exact source binding",
                self.anchor_dataset_identity
            )));
        }
        Ok(anchor)
    }

    pub fn validate_against(
        &self,
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
    ) -> Result<(), CanonicalDataSelectionError> {
        let received_anchor = self.validate()?;
        if &received_anchor != anchor {
            return Err(provenance_mismatch(
                anchor,
                format!(
                    "receipt anchor {} does not match requested anchor {}",
                    received_anchor.to_path_component(),
                    anchor.to_path_component()
                ),
            ));
        }
        if self.feature_plan_identity != features.plan_identity().to_hex() {
            return Err(provenance_mismatch(
                anchor,
                "feature plan identity does not match loaded FeatureFrame",
            ));
        }
        if self.feature_provenance_identity != features.provenance_identity().to_hex() {
            return Err(provenance_mismatch(
                anchor,
                "feature provenance identity does not match loaded FeatureFrame",
            ));
        }
        let expected_feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        if self.feature_execution != expected_feature_execution {
            return Err(provenance_mismatch(
                anchor,
                "feature execution policy/math lane does not match the immutable current build authority",
            ));
        }
        let feature_content_sha256 = canonical_feature_content_sha256(features)?;
        if self.feature_content_sha256 != feature_content_sha256 {
            return Err(provenance_mismatch(
                anchor,
                "feature content SHA-256 does not match exact timestamps/names/value bits/validity codes",
            ));
        }
        let expected = source_binding_receipts(features);
        if self.source_bindings.len() != expected.len() {
            return Err(provenance_mismatch(
                anchor,
                format!(
                    "source binding count {} does not match loaded FeatureFrame count {}",
                    self.source_bindings.len(),
                    expected.len()
                ),
            ));
        }
        for (received, expected) in self.source_bindings.iter().zip(expected.iter()) {
            if received.source_node_id != expected.source_node_id {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node order/id mismatch: received `{}`, expected `{}`",
                        received.source_node_id, expected.source_node_id
                    ),
                ));
            }
            if received.dataset_identity != expected.dataset_identity {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` dataset identity mismatch",
                        received.source_node_id
                    ),
                ));
            }
            if received.manifest_schema_id != expected.manifest_schema_id {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` manifest schema mismatch",
                        received.source_node_id
                    ),
                ));
            }
            if received.manifest_sha256 != expected.manifest_sha256 {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` manifest hash mismatch",
                        received.source_node_id
                    ),
                ));
            }
            if received.generation_id != expected.generation_id {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` generation mismatch",
                        received.source_node_id
                    ),
                ));
            }
            if received.vortex_sha256 != expected.vortex_sha256 {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` Vortex hash mismatch",
                        received.source_node_id
                    ),
                ));
            }
            if received.bar_timestamp_convention != expected.bar_timestamp_convention {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` bar convention mismatch",
                        received.source_node_id
                    ),
                ));
            }
            if received.segments != expected.segments {
                return Err(provenance_mismatch(
                    anchor,
                    format!(
                        "source node `{}` consumed segment mismatch",
                        received.source_node_id
                    ),
                ));
            }
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String, CanonicalDataSelectionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize canonical bytes: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_SEARCH_INPUT_RECEIPT_HASH_DOMAIN_V2);
        hasher.update(bytes);
        Ok(hex(&hasher.finalize()))
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, CanonicalDataSelectionError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize JSON: {error}")))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, CanonicalDataSelectionError> {
        let receipt: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid_receipt(format!("parse JSON: {error}")))?;
        receipt.validate()?;
        Ok(receipt)
    }
}

impl CanonicalSearchSourceBindingReceiptV1 {
    pub fn source_node_id(&self) -> &str {
        &self.source_node_id
    }

    pub fn dataset_identity(&self) -> &str {
        &self.dataset_identity
    }

    pub fn manifest_schema_id(&self) -> &str {
        &self.manifest_schema_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub fn vortex_sha256(&self) -> &str {
        &self.vortex_sha256
    }

    pub fn bar_timestamp_convention(&self) -> &str {
        &self.bar_timestamp_convention
    }

    pub fn segments(&self) -> &[CanonicalSearchSourceSegmentReceiptV1] {
        &self.segments
    }
}

impl CanonicalSearchSourceSegmentReceiptV1 {
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

fn canonical_feature_math_lane_name(lane: CanonicalFeatureMathLaneV1) -> &'static str {
    match lane {
        CanonicalFeatureMathLaneV1::CpuScalar => "cpu_scalar",
        CanonicalFeatureMathLaneV1::CpuAvx2Fma => "cpu_avx2_fma",
        CanonicalFeatureMathLaneV1::CpuAvx512FDqVlBwAvx2Fma => "cpu_avx512f_dq_vl_bw_avx2_fma",
        CanonicalFeatureMathLaneV1::GpuCudaF64Strict => "gpu_cuda_f64_strict",
    }
}

/// Hash the exact payload that search/model consumers observe.
///
/// The framing is deliberately independent of RAM/Vortex/view storage: row
/// and column counts are little-endian u64 values, timestamps are ordered i64
/// little-endian values, each ordered UTF-8 name is length-prefixed, and then
/// every column contributes each f64 `to_bits` followed by its typed validity
/// code. There is no NaN canonicalization, tolerance, column sorting, or
/// current-host reconstruction.
fn canonical_feature_content_sha256(
    features: &FeatureFrame,
) -> Result<String, CanonicalDataSelectionError> {
    let row_count = u64::try_from(features.n_samples())
        .map_err(|_| invalid_receipt("feature row count does not fit u64"))?;
    let column_count = u64::try_from(features.n_features())
        .map_err(|_| invalid_receipt("feature column count does not fit u64"))?;
    if features.timestamps.len() != features.n_samples() {
        return Err(invalid_receipt(
            "feature timestamp count does not match the frame row count",
        ));
    }
    if features.names.len() != features.n_features() {
        return Err(invalid_receipt(
            "feature name count does not match the frame column count",
        ));
    }

    let mut hasher = Sha256::new();
    hasher.update(CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V1);
    hasher.update(row_count.to_le_bytes());
    hasher.update(column_count.to_le_bytes());
    for timestamp in &features.timestamps {
        hasher.update(timestamp.to_le_bytes());
    }

    for (index, name) in features.names.iter().enumerate() {
        let name_len = u64::try_from(name.len())
            .map_err(|_| invalid_receipt("feature name length does not fit u64"))?;
        hasher.update(name_len.to_le_bytes());
        hasher.update(name.as_bytes());

        let column = features.feature_column(index).map_err(|error| {
            invalid_receipt(format!(
                "materialize exact feature column `{name}` for content receipt: {error}"
            ))
        })?;
        if column.name != *name {
            return Err(invalid_receipt(format!(
                "feature column {index} materialized as `{}` instead of `{name}`",
                column.name
            )));
        }
        if column.values.len() != features.n_samples()
            || column.validity.len() != features.n_samples()
        {
            return Err(invalid_receipt(format!(
                "feature column `{name}` value/validity lengths do not match {row_count} rows"
            )));
        }
        for (value, validity) in column.values.iter().zip(column.validity.iter()) {
            hasher.update(value.to_bits().to_le_bytes());
            hasher.update([validity.code()]);
        }
    }
    Ok(hex(&hasher.finalize()))
}

fn source_binding_receipts(features: &FeatureFrame) -> Vec<CanonicalSearchSourceBindingReceiptV1> {
    let mut bindings = features
        .provenance()
        .bindings()
        .iter()
        .map(|binding| CanonicalSearchSourceBindingReceiptV1 {
            source_node_id: binding.source_node_id().to_owned(),
            dataset_identity: binding.dataset_identity().to_path_component(),
            manifest_schema_id: binding.manifest_schema_id().to_owned(),
            manifest_sha256: hex(binding.manifest_hash()),
            generation_id: binding.generation_id().to_owned(),
            vortex_sha256: hex(binding.vortex_hash()),
            bar_timestamp_convention: binding.bar_timestamp_convention().to_string(),
            segments: binding
                .segments()
                .iter()
                .map(|segment| CanonicalSearchSourceSegmentReceiptV1 {
                    row_start: segment.row_start(),
                    row_end: segment.row_end(),
                    timestamp_start_ms: segment.timestamp_start_ms(),
                    timestamp_end_ms: segment.timestamp_end_ms(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    bindings.sort_by(|left, right| left.source_node_id.cmp(&right.source_node_id));
    bindings
}

fn invalid_receipt(detail: impl Into<String>) -> CanonicalDataSelectionError {
    CanonicalDataSelectionError::InvalidReceipt {
        detail: detail.into(),
    }
}

fn provenance_mismatch(
    anchor: &CanonicalDatasetIdentity,
    detail: impl Into<String>,
) -> CanonicalDataSelectionError {
    CanonicalDataSelectionError::ProvenanceMismatch {
        anchor_id: anchor.to_path_component(),
        detail: detail.into(),
    }
}

fn validate_nonempty(label: &str, value: &str) -> Result<(), CanonicalDataSelectionError> {
    if value.trim().is_empty() {
        return Err(invalid_receipt(format!("{label} is empty")));
    }
    Ok(())
}

fn validate_artifact_kind(value: &str) -> Result<(), CanonicalDataSelectionError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-' | b'_')
        })
    {
        return Err(invalid_receipt(
            "artifact kind must be 1..=128 lowercase ASCII letters, digits, dot, dash, or underscore",
        ));
    }
    Ok(())
}

fn validate_search_config_hash(value: &str) -> Result<(), CanonicalDataSelectionError> {
    let Some(hex) = value.strip_prefix("fnv64:") else {
        return Err(invalid_receipt(
            "search config hash must use the canonical fnv64:<16 lowercase hex> form",
        ));
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_receipt(
            "search config hash must use the canonical fnv64:<16 lowercase hex> form",
        ));
    }
    Ok(())
}

fn validate_sha256_hex(label: &str, value: &str) -> Result<(), CanonicalDataSelectionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_receipt(format!(
            "{label} must be exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_segments(
    source_node_id: &str,
    segments: &[CanonicalSearchSourceSegmentReceiptV1],
) -> Result<(), CanonicalDataSelectionError> {
    if segments.is_empty() {
        return Err(invalid_receipt(format!(
            "source node `{source_node_id}` has no consumed segments"
        )));
    }
    for (index, segment) in segments.iter().enumerate() {
        if segment.row_start >= segment.row_end {
            return Err(invalid_receipt(format!(
                "source node `{source_node_id}` segment {index} has an empty/reversed row range"
            )));
        }
        if segment.timestamp_start_ms > segment.timestamp_end_ms {
            return Err(invalid_receipt(format!(
                "source node `{source_node_id}` segment {index} has reversed timestamps"
            )));
        }
        if let Some(previous) = index.checked_sub(1).and_then(|i| segments.get(i)) {
            if previous.row_end > segment.row_start
                || previous.timestamp_end_ms >= segment.timestamp_start_ms
            {
                return Err(invalid_receipt(format!(
                    "source node `{source_node_id}` consumed segments overlap or are out of order"
                )));
            }
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn inventory_for_symbol(
    root: &Path,
    symbol: &str,
) -> Result<Vec<CanonicalDatasetIdentity>, CanonicalDataSelectionError> {
    neoethos_data::discover_canonical_dataset_identities(root, symbol).map_err(|error| {
        CanonicalDataSelectionError::InventoryFailed {
            requested_symbol: symbol.to_owned(),
            detail: error.to_string(),
        }
    })
}

fn candidate_ids<'a>(
    candidates: impl IntoIterator<Item = &'a CanonicalDatasetIdentity>,
) -> Vec<String> {
    let mut ids = candidates
        .into_iter()
        .map(CanonicalDatasetIdentity::to_path_component)
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

fn same_exact_series(
    candidate: &CanonicalDatasetIdentity,
    anchor: &CanonicalDatasetIdentity,
) -> bool {
    candidate.scope() == anchor.scope()
        && candidate.symbol_name() == anchor.symbol_name()
        && candidate.bar_timestamp_convention() == anchor.bar_timestamp_convention()
}

fn same_source_account(
    candidate: &CanonicalDatasetIdentity,
    anchor: &CanonicalDatasetIdentity,
) -> bool {
    if candidate.bar_timestamp_convention() != anchor.bar_timestamp_convention() {
        return false;
    }
    match (candidate.scope(), anchor.scope()) {
        (
            CanonicalDatasetScope::External {
                source_namespace: candidate_namespace,
            },
            CanonicalDatasetScope::External {
                source_namespace: anchor_namespace,
            },
        ) => candidate_namespace == anchor_namespace,
        (
            CanonicalDatasetScope::CTrader {
                environment: candidate_environment,
                server: candidate_server,
                account_id: candidate_account,
                ..
            },
            CanonicalDatasetScope::CTrader {
                environment: anchor_environment,
                server: anchor_server,
                account_id: anchor_account,
                ..
            },
        ) => {
            candidate_environment == anchor_environment
                && candidate_server == anchor_server
                && candidate_account == anchor_account
        }
        _ => false,
    }
}

fn verify_direct_artifacts(
    anchor: &CanonicalDatasetIdentity,
    dataset: &SymbolDataset,
    requested: &BTreeSet<CanonicalTimeframe>,
) -> Result<(), CanonicalDataSelectionError> {
    let required = requested.iter().copied().collect::<Vec<_>>();
    require_direct_timeframes(dataset, anchor, &required).map_err(|error| {
        CanonicalDataSelectionError::ProvenanceMismatch {
            anchor_id: anchor.to_path_component(),
            detail: error.to_string(),
        }
    })?;
    for timeframe in requested {
        let artifact = dataset
            .source_artifacts
            .get(timeframe.as_str())
            .ok_or_else(|| CanonicalDataSelectionError::MissingDirectTimeframe {
                anchor_id: anchor.to_path_component(),
                requested_symbol: anchor.symbol_name().to_owned(),
                requested_timeframe: *timeframe,
                candidate_ids: Vec::new(),
            })?;
        if !same_exact_series(artifact.identity(), anchor) {
            return Err(CanonicalDataSelectionError::ProvenanceMismatch {
                anchor_id: anchor.to_path_component(),
                detail: format!(
                    "{} belongs to source/account {}",
                    timeframe,
                    artifact.identity().to_path_component()
                ),
            });
        }
    }
    Ok(())
}

fn verify_search_input_provenance(
    anchor: &CanonicalDatasetIdentity,
    dataset: &SymbolDataset,
    base_frame: &CanonicalOhlcvFrame,
    features: &FeatureFrame,
) -> Result<(), CanonicalDataSelectionError> {
    let base_timestamps = base_frame.ohlcv().timestamp.as_deref().ok_or_else(|| {
        CanonicalDataSelectionError::ProvenanceMismatch {
            anchor_id: anchor.to_path_component(),
            detail: "base canonical frame has no timestamps".to_owned(),
        }
    })?;
    if features.timestamps.as_slice() != base_timestamps {
        return Err(CanonicalDataSelectionError::ProvenanceMismatch {
            anchor_id: anchor.to_path_component(),
            detail: format!(
                "feature timestamps/rows do not exactly equal base timestamps ({} vs {} rows)",
                features.timestamps.len(),
                base_timestamps.len()
            ),
        });
    }

    let bindings = features.provenance().bindings();
    let mut matched_artifacts = BTreeSet::new();
    for binding in bindings {
        let matching = dataset
            .source_artifacts
            .values()
            .filter(|artifact| artifact.identity() == binding.dataset_identity())
            .collect::<Vec<_>>();
        let [artifact] = matching.as_slice() else {
            return Err(CanonicalDataSelectionError::ProvenanceMismatch {
                anchor_id: anchor.to_path_component(),
                detail: format!(
                    "feature binding {} resolves to {} selected artifacts",
                    binding.dataset_identity().to_path_component(),
                    matching.len()
                ),
            });
        };
        let expected = artifact
            .source_binding(binding.source_node_id())
            .map_err(|error| CanonicalDataSelectionError::ProvenanceMismatch {
                anchor_id: anchor.to_path_component(),
                detail: error.to_string(),
            })?;
        if binding != &expected {
            return Err(CanonicalDataSelectionError::ProvenanceMismatch {
                anchor_id: anchor.to_path_component(),
                detail: format!(
                    "feature binding for {} does not match its pinned manifest/generation",
                    binding.dataset_identity().to_path_component()
                ),
            });
        }
        matched_artifacts.insert(binding.dataset_identity().to_path_component());
    }
    let expected_artifacts = dataset
        .source_artifacts
        .values()
        .map(|artifact| artifact.identity().to_path_component())
        .collect::<BTreeSet<_>>();
    if matched_artifacts != expected_artifacts {
        return Err(CanonicalDataSelectionError::ProvenanceMismatch {
            anchor_id: anchor.to_path_component(),
            detail: format!(
                "feature provenance covers {:?}, selected direct artifacts are {:?}",
                matched_artifacts, expected_artifacts
            ),
        });
    }
    Ok(())
}
