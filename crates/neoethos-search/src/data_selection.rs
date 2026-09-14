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
use std::time::Instant;

use neoethos_data::{
    CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1, CanonicalOhlcvFrame,
    CanonicalTimeframe, FeatureBuildControl, FeatureBuildOptions, FeatureFrame,
    IndicatorComputePolicy, Ohlcv, ResolvedCanonicalFeatureExecutionAuthorityV1,
    ResolvedCanonicalFeatureMathLaneV1, SearchNormalizationFittedStateV1, SymbolDataset,
    VECTOR_TA_CPU_F64_MATH_AUTHORITY_V1, VECTOR_TA_CUDA_F64_MATH_AUTHORITY_V1,
    canonical_feature_execution_authority_for_policy_v1, load_exact_dataset_series_receipt,
    prepare_multitimeframe_features_with_options, require_direct_timeframes,
    resolved_canonical_feature_execution_authority_v1,
};
#[cfg(feature = "gpu-cuda")]
use neoethos_data::{
    CanonicalGpuResidentFeatureExecutionSemanticV1, SealedGpuResidentFeatureStoreV3,
};
use neoethos_dataset_contracts::CanonicalDatasetScope;
use neoethos_feature_contracts::{FeatureOperationTagV1, FeaturePlanV1};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CANONICAL_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V2: u16 = 2;
const CANONICAL_SEARCH_INPUT_RECEIPT_HASH_DOMAIN_V2: &[u8] =
    b"neoethos.canonical-search-input-receipt.v2\0";
const CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V1: &[u8] = b"neoethos.canonical-feature-content.v1\0";
#[cfg(feature = "gpu-cuda")]
const CANONICAL_GPU_RESIDENT_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V3: u16 = 3;
#[cfg(feature = "gpu-cuda")]
const CANONICAL_GPU_RESIDENT_SEARCH_INPUT_RECEIPT_HASH_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-gpu-resident-search-input-receipt.v3\0";
#[cfg(feature = "gpu-cuda")]
const CANONICAL_GPU_RESIDENT_CONTENT_MERKLE_ALGORITHM_V3: &str =
    "neoethos.canonical-feature-content.merkle.v3";
const CANONICAL_FEATURE_EXECUTION_SCHEMA_VERSION_V1: u16 = 1;
const CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2: u16 = 2;
const CANONICAL_SEARCH_ARTIFACT_SCOPE_HASH_DOMAIN_V2: &[u8] =
    b"neoethos.canonical-search-artifact-scope.v2\0";
#[cfg(feature = "gpu-cuda")]
const CANONICAL_GPU_RESIDENT_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V3: u16 = 3;
#[cfg(feature = "gpu-cuda")]
const CANONICAL_GPU_RESIDENT_SEARCH_ARTIFACT_SCOPE_HASH_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-gpu-resident-search-artifact-scope.v3\0";
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
        self.load_search_input_with_builder(
            higher_timeframes,
            &FeatureBuildControl::default(),
            |dataset, base_tf, higher_tfs| {
                neoethos_data::prepare_multitimeframe_features(dataset, base_tf, higher_tfs)
            },
        )
    }

    fn load_search_input_with_builder(
        &self,
        higher_timeframes: &[CanonicalTimeframe],
        control: &FeatureBuildControl,
        build: impl FnOnce(&SymbolDataset, &str, &[&str]) -> anyhow::Result<FeatureFrame>,
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
            canonical_feature_execution_authority_for_policy_v1(
                control.resolved_indicator_compute_policy(),
            ),
        );
        let features = build(&dataset, base_name, &higher_names).map_err(|error| {
            CanonicalDataSelectionError::FeatureBuildFailed {
                anchor_id: self.anchor.to_path_component(),
                detail: error.to_string(),
            }
        })?;
        let execution_after_build = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            canonical_feature_execution_authority_for_policy_v1(
                control.resolved_indicator_compute_policy(),
            ),
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
            prepared_receipt: None,
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
    // A preparation-time observation, not permission to trust mutable disk
    // backing later. The consumer rehashes against this original receipt.
    prepared_receipt: Option<CanonicalSearchInputReceiptV2>,
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
        Self::from_prepared_canonical_frame_with_control(
            anchor,
            base_frame,
            features,
            &FeatureBuildControl::default(),
        )
    }

    /// Same exact data boundary, with per-request progress and cooperative Stop
    /// between bounded projection/hash units. No partial receipt is returned.
    pub fn from_prepared_canonical_frame_with_control(
        anchor: CanonicalDatasetIdentity,
        base_frame: CanonicalOhlcvFrame,
        features: FeatureFrame,
        control: &FeatureBuildControl,
    ) -> Result<CanonicalSearchInput, CanonicalDataSelectionError> {
        if base_frame.artifact().identity() != &anchor {
            return Err(provenance_mismatch(
                &anchor,
                "prepared canonical base frame does not match the selected anchor identity",
            ));
        }
        // Use the same operation policy as the producer, not an unrelated
        // process default that would mislabel this batch and break saved replay.
        let feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            canonical_feature_execution_authority_for_policy_v1(
                control.resolved_indicator_compute_policy(),
            ),
        );
        let prepared_receipt = CanonicalSearchRunInputV2::from_fresh_feature_frame_with_execution(
            &anchor,
            &features,
            &base_frame,
            feature_execution.clone(),
            control,
        )?
        .receipt()
        .clone();
        Ok(CanonicalSearchInput {
            anchor,
            base_frame,
            features,
            feature_execution,
            prepared_receipt: Some(prepared_receipt),
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
        Self::from_exact_series_receipt_with_builder_v3(
            root,
            series,
            base_timeframe,
            options,
            &FeatureBuildControl::default(),
            prepare_multitimeframe_features_with_options,
        )
    }

    /// Rebuild a recorded complete-generation input with its saved recipe and
    /// frozen fit. Pin every selected generation/manifest before feature work;
    /// if the publication has advanced, the current Data retention contract
    /// refuses the stale receipt rather than substituting newer history.
    /// Recorded column projections are restored by the final content binding.
    /// Source cutoffs are not encoded in this recipe and remain unsupported.
    pub fn from_recorded_receipt(
        root: impl AsRef<Path>,
        receipt: CanonicalSearchInputReceiptV2,
        options: &FeatureBuildOptions,
    ) -> Result<Self, CanonicalDataSelectionError> {
        Self::from_recorded_receipt_with_control(
            root,
            receipt,
            options,
            &FeatureBuildControl::default(),
        )
    }

    pub fn from_recorded_receipt_with_control(
        root: impl AsRef<Path>,
        receipt: CanonicalSearchInputReceiptV2,
        options: &FeatureBuildOptions,
        control: &FeatureBuildControl,
    ) -> Result<Self, CanonicalDataSelectionError> {
        control
            .checkpoint()
            .map_err(|error| invalid_receipt(error.to_string()))?;
        let anchor = receipt.validate()?;
        if receipt
            .feature_build_options()
            .is_some_and(|recorded| recorded != options)
        {
            return Err(invalid_receipt(
                "reopen options differ from the recorded producer recipe",
            ));
        }
        // Select the saved policy for this replay only. A process-wide default
        // is not authority to relabel a different recorded computation.
        let replay_control = receipt.feature_execution.replay_control(control)?;
        let control = &replay_control;
        let mut selected = std::collections::BTreeMap::new();
        for binding in receipt.source_bindings() {
            let identity = CanonicalDatasetIdentity::from_path_component(
                binding.dataset_identity(),
            )
            .map_err(|error| invalid_receipt(format!("recorded source identity: {error}")))?;
            if identity.scope() != anchor.scope()
                || identity.symbol_name() != anchor.symbol_name()
                || identity.bar_timestamp_convention() != anchor.bar_timestamp_convention()
            {
                return Err(invalid_receipt(
                    "recorded source is outside the exact anchor series",
                ));
            }
            let generation = neoethos_data::SelectedDatasetGenerationV1::new(
                identity.clone(),
                binding.generation_id(),
                binding.manifest_sha256(),
            )
            .map_err(|error| invalid_receipt(format!("recorded source generation: {error}")))?;
            if let Some(previous) = selected.insert(identity.timeframe(), generation.clone()) {
                if previous != generation {
                    return Err(invalid_receipt(format!(
                        "recorded input has conflicting exact generations for direct timeframe {}; multi-generation concatenation is not supported",
                        identity.timeframe()
                    )));
                }
            }
        }
        let selected_anchor = selected
            .get(&anchor.timeframe())
            .filter(|selected| selected.identity() == &anchor)
            .cloned()
            .ok_or_else(|| invalid_receipt("recorded input has no exact anchor generation"))?;
        let series =
            CanonicalDatasetSeriesReceiptV1::new(selected_anchor, selected.into_values().collect())
                .map_err(|error| invalid_receipt(format!("recorded direct series: {error}")))?;
        let input = Self::from_exact_series_receipt_with_builder_v3(
            root,
            &series,
            anchor.timeframe(),
            options,
            control,
            |dataset, base_tf, options| {
                // A prefix producer needs its original cutoff before computing
                // indicators, not a post-computation slice of full-history values.
                for binding in receipt.source_bindings() {
                    let identity =
                        CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())?;
                    let frame = dataset
                        .timeframe(identity.timeframe().as_str())
                        .ok_or_else(|| anyhow::anyhow!("recorded direct source is missing"))?;
                    let timestamps = frame.timestamp.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("recorded direct source has no timestamps")
                    })?;
                    anyhow::ensure!(
                        matches!(binding.segments(), [segment]
                            if segment.row_start() == 0
                                && segment.row_end() == frame.len() as u64
                                && timestamps.first().copied() == Some(segment.timestamp_start_ms())
                                && timestamps.last().copied() == Some(segment.timestamp_end_ms())),
                        "recorded source row window for {} cannot be reconstructed by the complete-generation recipe; its original source cutoff is not persisted",
                        identity.timeframe()
                    );
                }
                match receipt.normalization_fitted_state() {
                    Some(state) => {
                        neoethos_data::prepare_multitimeframe_features_with_fitted_normalization_and_control(
                            dataset, base_tf, options, state, control,
                        )
                    }
                    None => neoethos_data::prepare_multitimeframe_features_raw_with_options_and_control(
                        dataset, base_tf, options, control,
                    ),
                }
            },
        )?;
        input.bind_recorded_receipt_with_control(receipt, control)
    }

    /// Build the CPU-authored exact-parity contract input from the feature
    /// vocabulary and Quant-v3 semantics admitted by the resident GPU V3
    /// Standard profile. The returned V2 envelope receipt still
    /// records the real CPU/Auto process authority and selected CPU math lane;
    /// the Data boundary refuses this call if the process was installed as
    /// GpuOnly, so CPU values cannot be mislabeled as CUDA output.
    #[cfg(feature = "gpu-cuda")]
    pub fn from_exact_series_receipt_gpu_exact_parity_cpu_reference_v3(
        root: impl AsRef<Path>,
        series: &CanonicalDatasetSeriesReceiptV1,
        base_timeframe: CanonicalTimeframe,
        options: &FeatureBuildOptions,
    ) -> Result<CanonicalSearchInput, CanonicalDataSelectionError> {
        Self::from_exact_series_receipt_with_builder_v3(
            root,
            series,
            base_timeframe,
            options,
            &FeatureBuildControl::default(),
            neoethos_data::prepare_multitimeframe_features_gpu_exact_parity_cpu_reference_v3,
        )
    }

    fn from_exact_series_receipt_with_builder_v3(
        root: impl AsRef<Path>,
        series: &CanonicalDatasetSeriesReceiptV1,
        base_timeframe: CanonicalTimeframe,
        options: &FeatureBuildOptions,
        control: &FeatureBuildControl,
        build_features: impl FnOnce(
            &SymbolDataset,
            &str,
            &FeatureBuildOptions,
        ) -> anyhow::Result<FeatureFrame>,
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
                detail: format!("{error:#}"),
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
            canonical_feature_execution_authority_for_policy_v1(
                control.resolved_indicator_compute_policy(),
            ),
        );
        let features = build_features(&dataset, base_name, options).map_err(|error| {
            CanonicalDataSelectionError::FeatureBuildFailed {
                anchor_id: anchor.to_path_component(),
                detail: error.to_string(),
            }
        })?;
        let execution_after_build = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            canonical_feature_execution_authority_for_policy_v1(
                control.resolved_indicator_compute_policy(),
            ),
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
            prepared_receipt: None,
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

    /// Bind the original persisted receipt to the actual reopened values and
    /// base artifact, preserving its identity (including historical raw V2
    /// receipts without the newer optional recipe fields). The untrusted
    /// boundary restores only the recorded plan's exact column projection and
    /// hashes the current payload exactly once; `receipt()` then returns this
    /// validated original instead of minting a different receipt.
    pub fn bind_recorded_receipt(
        self,
        receipt: CanonicalSearchInputReceiptV2,
    ) -> Result<Self, CanonicalDataSelectionError> {
        self.bind_recorded_receipt_with_control(receipt, &FeatureBuildControl::default())
    }

    pub fn bind_recorded_receipt_with_control(
        mut self,
        receipt: CanonicalSearchInputReceiptV2,
        control: &FeatureBuildControl,
    ) -> Result<Self, CanonicalDataSelectionError> {
        control
            .checkpoint()
            .map_err(|error| invalid_receipt(error.to_string()))?;
        if receipt.feature_execution() != &self.feature_execution {
            return Err(provenance_mismatch(
                &self.anchor,
                "recorded feature execution differs from the authority that built the reopened input",
            ));
        }
        if self.features.plan_identity().to_hex() != receipt.feature_plan_identity {
            let bytes = receipt.feature_plan_canonical_bytes.as_ref().ok_or_else(|| {
                invalid_receipt(
                    "reopened legacy plan differs and has no recorded full plan to authorize a projection",
                )
            })?;
            let recorded = FeaturePlanV1::from_canonical_bytes(bytes)
                .map_err(|error| invalid_receipt(format!("recorded projection plan: {error}")))?;
            if recorded.identity().to_hex() != receipt.feature_plan_identity {
                return Err(invalid_receipt(
                    "recorded projection plan disagrees with its sealed identity",
                ));
            }
            let columns = recorded
                .final_outputs()
                .iter()
                .map(|name| {
                    self.features
                        .names
                        .iter()
                        .position(|actual| actual == name)
                        .ok_or_else(|| {
                            invalid_receipt(format!(
                                "reopened input is missing recorded feature `{name}`"
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let projected = self.features.select_columns(&columns).map_err(|error| {
                invalid_receipt(format!("recorded feature projection: {error}"))
            })?;
            if projected.plan_identity().to_hex() != receipt.feature_plan_identity {
                return Err(invalid_receipt(
                    "reopened feature math/source plan differs after the exact recorded projection",
                ));
            }
            self.features = projected;
        }
        let checked = CanonicalSearchRunInputV2::new_with_execution(
            receipt,
            &self.features,
            &self.base_frame,
            &self.feature_execution,
            control,
        )?;
        if checked.anchor_identity() != &self.anchor {
            return Err(provenance_mismatch(
                &self.anchor,
                "recorded receipt anchor differs from the reopened input anchor",
            ));
        }
        self.prepared_receipt = Some(checked.receipt);
        Ok(self)
    }

    /// Serializable, content-addressed observation of the prepared values.
    /// Reuse an existing preparation receipt instead of decoding the complete
    /// cube again just to publish that same receipt. This does not assert that
    /// externally writable Vortex backing is immutable: `as_run_input` binds
    /// the current values to the original receipt before Search can use them.
    pub fn receipt(&self) -> Result<CanonicalSearchInputReceiptV2, CanonicalDataSelectionError> {
        if let Some(receipt) = &self.prepared_receipt {
            return Ok(receipt.clone());
        }
        CanonicalSearchInputReceiptV2::from_feature_frame_with_execution(
            &self.anchor,
            &self.features,
            self.feature_execution.clone(),
            &FeatureBuildControl::default(),
        )
    }

    pub fn as_run_input(
        &self,
    ) -> Result<CanonicalSearchRunInputV2<'_>, CanonicalDataSelectionError> {
        self.as_run_input_with_control(&FeatureBuildControl::default())
    }

    pub fn as_run_input_with_control(
        &self,
        control: &FeatureBuildControl,
    ) -> Result<CanonicalSearchRunInputV2<'_>, CanonicalDataSelectionError> {
        if let Some(receipt) = &self.prepared_receipt {
            // Keep the full value check. Creating another fresh receipt here
            // would silently relabel changed backing as a new valid input.
            return CanonicalSearchRunInputV2::new_with_execution(
                receipt.clone(),
                &self.features,
                &self.base_frame,
                &self.feature_execution,
                control,
            );
        }
        CanonicalSearchRunInputV2::from_fresh_feature_frame_with_execution(
            &self.anchor,
            &self.features,
            &self.base_frame,
            self.feature_execution.clone(),
            control,
        )
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
    /// Bind a freshly-built feature frame to its canonical base frame while
    /// computing the exact content digest once. This is the production path
    /// for values that are already resident in this process. [`Self::new`]
    /// remains the untrusted-receipt boundary and always recomputes the digest.
    pub fn from_fresh_feature_frame(
        anchor: &CanonicalDatasetIdentity,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
    ) -> Result<Self, CanonicalDataSelectionError> {
        Self::from_fresh_feature_frame_with_execution(
            anchor,
            features,
            base_frame,
            CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
                resolved_canonical_feature_execution_authority_v1(),
            ),
            &FeatureBuildControl::default(),
        )
    }

    fn from_fresh_feature_frame_with_execution(
        anchor: &CanonicalDatasetIdentity,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
        feature_execution: CanonicalFeatureExecutionReceiptV1,
        control: &FeatureBuildControl,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame_with_execution(
            anchor,
            features,
            feature_execution,
            control,
        )?;
        let known_content_sha256 = receipt.feature_content_sha256.clone();
        let validated_anchor = Self::validate_values(
            &receipt,
            features,
            base_frame.ohlcv(),
            Some(&known_content_sha256),
            &receipt.feature_execution,
        )?;
        if &validated_anchor != anchor {
            return Err(provenance_mismatch(
                anchor,
                "fresh receipt anchor changed while binding the canonical run input",
            ));
        }
        Self::bind_base_frame(receipt, validated_anchor, features, base_frame)
    }

    pub fn new(
        receipt: CanonicalSearchInputReceiptV2,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
    ) -> Result<Self, CanonicalDataSelectionError> {
        Self::new_with_control(
            receipt,
            features,
            base_frame,
            &FeatureBuildControl::default(),
        )
    }

    /// A carried receipt is never trusted without hashing the current values.
    pub fn new_with_control(
        receipt: CanonicalSearchInputReceiptV2,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
        control: &FeatureBuildControl,
    ) -> Result<Self, CanonicalDataSelectionError> {
        // A control can request future production, but cannot attest how an
        // unrelated supplied frame was built. Keep this public boundary strict.
        let execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        Self::new_with_execution(receipt, features, base_frame, &execution, control)
    }

    // Only an owned input whose builder captured the actual operation authority
    // may carry it across replay/control lifetimes. All values are still hashed.
    fn new_with_execution(
        receipt: CanonicalSearchInputReceiptV2,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
        execution: &CanonicalFeatureExecutionReceiptV1,
        control: &FeatureBuildControl,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let ohlcv = base_frame.ohlcv();
        receipt.validate()?;
        let content_sha256 = canonical_feature_content_sha256_with_control(features, control)?;
        let anchor =
            Self::validate_values(&receipt, features, ohlcv, Some(&content_sha256), execution)?;
        Self::bind_base_frame(receipt, anchor, features, base_frame)
    }

    fn bind_base_frame(
        receipt: CanonicalSearchInputReceiptV2,
        anchor: CanonicalDatasetIdentity,
        features: &'a FeatureFrame,
        base_frame: &'a CanonicalOhlcvFrame,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let ohlcv = base_frame.ohlcv();
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
        let execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        let anchor = Self::validate_values(&receipt, features, ohlcv, None, &execution)?;
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
        known_content_sha256: Option<&str>,
        execution: &CanonicalFeatureExecutionReceiptV1,
    ) -> Result<CanonicalDatasetIdentity, CanonicalDataSelectionError> {
        let anchor = receipt.validate()?;
        let computed;
        let content_sha256 = match known_content_sha256 {
            Some(content_sha256) => content_sha256,
            None => {
                computed = canonical_feature_content_sha256(features)?;
                &computed
            }
        };
        receipt.validate_against_with_execution(&anchor, features, content_sha256, execution)?;
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
    /// Canonical plan bytes bind the saved fit to its actual transform node,
    /// including when the frame's final outputs are only a projected subset.
    /// Missing fields retain the exact serialization of historical raw V2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    feature_plan_canonical_bytes: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    normalization_fitted_state: Option<SearchNormalizationFittedStateV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    feature_build_options: Option<FeatureBuildOptions>,
}

/// Versioned identity of one exact GPU-resident feature payload.
///
/// V3 deliberately does not alias the CPU V2 linear content hash. It binds
/// the semantic-v3 Merkle root produced by the strict CUDA graph under its
/// own algorithm identifier, together with the exact resident shape and the
/// immutable source/plan provenance. Missing or legacy fields fail closed.
#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalGpuResidentSearchInputReceiptV3 {
    schema_version: u16,
    anchor_dataset_identity: String,
    feature_plan_identity: String,
    feature_provenance_identity: String,
    content_merkle_algorithm: String,
    feature_content_merkle_sha256: String,
    normalization_fit_sha256: String,
    /// Portable recipe and actual device-fitted values, not the transport digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    feature_plan_canonical_bytes: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    normalization_fitted_state: Option<SearchNormalizationFittedStateV1>,
    row_count: u64,
    column_count: u64,
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

#[cfg(all(test, feature = "gpu-cuda"))]
pub(crate) fn canonical_result_maximum_json_receipt_v3_for_test(
    maximum_general_string: &str,
    source_count: usize,
    total_segment_count: usize,
) -> CanonicalGpuResidentSearchInputReceiptV3 {
    assert!(source_count > 0);
    assert!(total_segment_count >= source_count);
    let fixed_sha256 = "f".repeat(64);
    let mut remaining_segments = total_segment_count;
    let source_bindings = (0..source_count)
        .map(|source_index| {
            let remaining_sources = source_count - source_index;
            let segment_count = remaining_segments / remaining_sources;
            remaining_segments -= segment_count;
            CanonicalSearchSourceBindingReceiptV1 {
                source_node_id: maximum_general_string.to_owned(),
                dataset_identity: maximum_general_string.to_owned(),
                manifest_schema_id: maximum_general_string.to_owned(),
                manifest_sha256: fixed_sha256.clone(),
                generation_id: maximum_general_string.to_owned(),
                vortex_sha256: fixed_sha256.clone(),
                bar_timestamp_convention: maximum_general_string.to_owned(),
                segments: (0..segment_count)
                    .map(|_| CanonicalSearchSourceSegmentReceiptV1 {
                        row_start: u64::MAX,
                        row_end: u64::MAX,
                        timestamp_start_ms: i64::MIN,
                        timestamp_end_ms: i64::MIN,
                    })
                    .collect(),
            }
        })
        .collect();
    CanonicalGpuResidentSearchInputReceiptV3 {
        schema_version: u16::MAX,
        anchor_dataset_identity: maximum_general_string.to_owned(),
        feature_plan_identity: fixed_sha256.clone(),
        feature_provenance_identity: fixed_sha256.clone(),
        content_merkle_algorithm: CANONICAL_GPU_RESIDENT_CONTENT_MERKLE_ALGORITHM_V3.to_owned(),
        feature_content_merkle_sha256: fixed_sha256.clone(),
        normalization_fit_sha256: fixed_sha256,
        feature_plan_canonical_bytes: None,
        normalization_fitted_state: None,
        row_count: u64::MAX,
        column_count: u64::MAX,
        feature_execution: CanonicalFeatureExecutionReceiptV1 {
            schema_version: u16::MAX,
            compute_policy: CanonicalFeatureComputePolicyV1::GpuOnly,
            vector_ta_math_authority: CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1.to_owned(),
            selected_lane: CanonicalFeatureMathLaneV1::GpuCudaF64Strict,
        },
        source_bindings,
    }
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
    /// Post-search selection and sizing, never an untouched final test.
    SelectionValidation,
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

/// Compact transport reference to a receipt embedded once in the SAME envelope.
/// This is not standalone authority and never resolves a file or CURRENT. The
/// original V2 scope hash binds the exact receipt, role, rows and timestamps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSearchArtifactScopeRefV1 {
    schema_version: u16,
    receipt_sha256: String,
    scope_sha256: String,
    evaluated_window: CanonicalSearchEvaluatedWindowV1,
}

impl CanonicalSearchArtifactScopeRefV1 {
    pub fn from_scope(
        scope: &CanonicalSearchArtifactScopeV2,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let scope_sha256 = scope.identity_sha256()?;
        Ok(Self {
            schema_version: 1,
            receipt_sha256: scope.receipt_sha256.clone(),
            scope_sha256,
            evaluated_window: scope.evaluated_window.clone(),
        })
    }

    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn evaluated_window(&self) -> &CanonicalSearchEvaluatedWindowV1 {
        &self.evaluated_window
    }

    pub fn attach(
        &self,
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> Result<CanonicalSearchArtifactScopeV2, CanonicalDataSelectionError> {
        if self.schema_version != 1 {
            return Err(invalid_receipt(
                "unsupported shared artifact-scope reference schema",
            ));
        }
        validate_sha256_hex("shared scope receipt SHA-256", &self.receipt_sha256)?;
        validate_sha256_hex("shared scope SHA-256", &self.scope_sha256)?;
        if receipt.identity_sha256()? != self.receipt_sha256 {
            return Err(invalid_receipt(
                "shared artifact-scope reference names a different receipt",
            ));
        }
        self.evaluated_window.validate_against_receipt(receipt)?;
        // The receipt identity and window were checked above in this operation.
        // Construct the same private scope without rehashing that receipt.
        let scope = CanonicalSearchArtifactScopeV2 {
            schema_version: CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2,
            receipt: receipt.clone(),
            receipt_sha256: self.receipt_sha256.clone(),
            evaluated_window: self.evaluated_window.clone(),
        };
        if scope.canonical_identity_sha256()? != self.scope_sha256 {
            return Err(invalid_receipt(
                "shared artifact-scope reference changed its exact window or identity",
            ));
        }
        Ok(scope)
    }
}

/// GPU-native artifact scope. This is a separate schema rather than a V2
/// receipt with a Merkle root smuggled into the linear-hash field.
#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalGpuResidentSearchArtifactScopeV3 {
    schema_version: u16,
    receipt: CanonicalGpuResidentSearchInputReceiptV3,
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
        evaluated_window.validate_against_receipt(&receipt)?;
        Ok(Self {
            schema_version: CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2,
            receipt,
            receipt_sha256,
            evaluated_window,
        })
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
        self.canonical_identity_sha256()
    }

    // Only callers that validated this exact scope in the same operation may
    // use this framing helper. Public identity requests always validate above.
    fn canonical_identity_sha256(&self) -> Result<String, CanonicalDataSelectionError> {
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
    fn replay_control(
        &self,
        control: &FeatureBuildControl,
    ) -> Result<FeatureBuildControl, CanonicalDataSelectionError> {
        self.validate()?;
        let policy = match self.compute_policy {
            CanonicalFeatureComputePolicyV1::Auto => IndicatorComputePolicy::Auto,
            CanonicalFeatureComputePolicyV1::CpuOnly => IndicatorComputePolicy::CpuOnly,
            CanonicalFeatureComputePolicyV1::GpuOnly => {
                return Err(invalid_receipt(
                    "recorded GpuOnly execution cannot be replayed by the host FeatureFrame producer",
                ));
            }
        };
        if control
            .indicator_compute_policy()
            .is_some_and(|requested| requested != policy)
        {
            return Err(invalid_receipt(
                "explicit replay policy conflicts with the recorded feature execution policy",
            ));
        }
        let actual = Self::from_runtime_authority(
            canonical_feature_execution_authority_for_policy_v1(policy),
        );
        if self != &actual {
            return Err(invalid_receipt(
                "recorded feature execution math lane differs from the current operation authority",
            ));
        }
        Ok(control.clone().with_indicator_compute_policy(policy))
    }

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
            &FeatureBuildControl::default(),
        )
    }

    fn from_feature_frame_with_execution(
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
        feature_execution: CanonicalFeatureExecutionReceiptV1,
        control: &FeatureBuildControl,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let feature_content_sha256 =
            canonical_feature_content_sha256_with_control(features, control)?;
        let receipt = Self {
            schema_version: CANONICAL_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V2,
            anchor_dataset_identity: anchor.to_path_component(),
            feature_plan_identity: features.plan_identity().to_hex(),
            feature_provenance_identity: features.provenance_identity().to_hex(),
            feature_content_sha256: feature_content_sha256.clone(),
            feature_execution,
            source_bindings: source_binding_receipts(features),
            feature_plan_canonical_bytes: Some(features.plan().canonical_bytes().to_vec()),
            normalization_fitted_state: features.normalization_fitted_state().cloned(),
            feature_build_options: features.feature_build_options().cloned(),
        };
        receipt.validate_against_with_execution(
            anchor,
            features,
            &feature_content_sha256,
            &receipt.feature_execution,
        )?;
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

    pub fn normalization_fitted_state(&self) -> Option<&SearchNormalizationFittedStateV1> {
        self.normalization_fitted_state.as_ref()
    }

    pub fn feature_build_options(&self) -> Option<&FeatureBuildOptions> {
        self.feature_build_options.as_ref()
    }

    /// Decode the recorded plan and verify its sealed identity/normalization
    /// linkage. This does not replace validation of the complete receipt.
    /// Historical raw receipts can explicitly lack this proof; modern model
    /// preprocessing contracts must require `Some` and compare final outputs.
    pub fn recorded_feature_plan(
        &self,
    ) -> Result<Option<FeaturePlanV1>, CanonicalDataSelectionError> {
        match &self.feature_plan_canonical_bytes {
            Some(bytes) => {
                let plan = FeaturePlanV1::from_canonical_bytes(bytes)
                    .map_err(|error| invalid_receipt(format!("recorded feature plan: {error}")))?;
                if plan.identity().to_hex() != self.feature_plan_identity {
                    return Err(invalid_receipt(
                        "recorded feature plan bytes disagree with the sealed plan identity",
                    ));
                }
                self.validate_normalization_plan(&plan)?;
                Ok(Some(plan))
            }
            None if self.normalization_fitted_state.is_some()
                || self.feature_build_options.is_some() =>
            {
                Err(invalid_receipt(
                    "persisted normalization/producer recipe requires the sealed feature plan bytes",
                ))
            }
            None => Ok(None),
        }
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
        self.recorded_feature_plan()?;
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
        if let Some(options) = &self.feature_build_options {
            if let Some(batch) = &options.classic_ta_working_set {
                batch.validate().map_err(|error| {
                    invalid_receipt(format!("producer working-set recipe: {error}"))
                })?;
                if !batch.replace_base_vocabulary {
                    return Err(invalid_receipt(
                        "producer working-set recipe must record the complete base selection",
                    ));
                }
            }
            let requested = std::iter::once(Ok(anchor.timeframe()))
                .chain(
                    options
                        .higher_tfs
                        .iter()
                        .map(|timeframe| timeframe.parse::<CanonicalTimeframe>()),
                )
                .collect::<Result<BTreeSet<_>, _>>()
                .map_err(|error| invalid_receipt(format!("producer recipe timeframe: {error}")))?;
            let actual = self
                .source_bindings
                .iter()
                .map(|binding| {
                    CanonicalDatasetIdentity::from_path_component(&binding.dataset_identity)
                        .map(|identity| identity.timeframe())
                })
                .collect::<Result<BTreeSet<_>, _>>()
                .map_err(|error| invalid_receipt(format!("producer recipe source: {error}")))?;
            if requested != actual {
                return Err(invalid_receipt(
                    "producer recipe timeframes disagree with exact direct-source bindings",
                ));
            }
        }
        Ok(anchor)
    }

    fn validate_normalization_plan(
        &self,
        plan: &FeaturePlanV1,
    ) -> Result<(), CanonicalDataSelectionError> {
        let has_normalization = plan
            .nodes()
            .iter()
            .any(|node| node.operation() == FeatureOperationTagV1::Normalization);
        match self.normalization_fitted_state.as_ref() {
            Some(state) => {
                state.validate_plan(plan).map_err(|error| {
                    invalid_receipt(format!(
                        "persisted normalization fit/plan mismatch: {error}"
                    ))
                })?;
                let options = self.feature_build_options.as_ref().ok_or_else(|| {
                    invalid_receipt("normalized feature receipt is missing its producer recipe")
                })?;
                if let Some(rows) = &options.normalization_training_rows {
                    let fitted_rows = state.training_rows().map_err(|error| {
                        invalid_receipt(format!("persisted normalization fit range: {error}"))
                    })?;
                    if rows != &fitted_rows {
                        return Err(invalid_receipt(
                            "producer recipe training rows disagree with persisted normalization fit",
                        ));
                    }
                }
            }
            None if has_normalization => {
                return Err(invalid_receipt(
                    "normalized feature plan is missing its persisted training fit",
                ));
            }
            None => {}
        }
        Ok(())
    }

    /// Verify live feature semantics without comparing historical rows or
    /// generation provenance. The canonical plan contains source identities,
    /// formulas, alignment parameters and the frozen fit, not row counts or
    /// timestamps. A projected receipt retains the full graph, so only its
    /// final-output ordering is restored before comparing identities.
    pub fn validate_live_feature_plan(
        &self,
        features: &FeatureFrame,
    ) -> Result<(), CanonicalDataSelectionError> {
        self.validate_normalization_plan(features.plan())?;
        if self.normalization_fitted_state.as_ref() != features.normalization_fitted_state() {
            return Err(invalid_receipt(
                "live feature frame does not carry the recorded normalization fit",
            ));
        }
        // The ordinary full-schema live path reuses the already-computed plan
        // identity; do not decode/hash the saved graph again on every bar.
        if features.plan_identity().to_hex() == self.feature_plan_identity {
            return Ok(());
        }
        let bytes = self.feature_plan_canonical_bytes.as_ref().ok_or_else(|| {
            invalid_receipt(
                "live feature plan differs from the legacy recorded plan; its missing full plan cannot authorize a guessed projection",
            )
        })?;
        let recorded = FeaturePlanV1::from_canonical_bytes(bytes)
            .map_err(|error| invalid_receipt(format!("recorded live feature plan: {error}")))?;
        if recorded.identity().to_hex() != self.feature_plan_identity {
            return Err(invalid_receipt(
                "recorded live feature plan bytes disagree with the sealed plan identity",
            ));
        }
        self.validate_normalization_plan(&recorded)?;
        let projected = FeaturePlanV1::new(
            features.plan().nodes().to_vec(),
            recorded.final_outputs().to_vec(),
        )
        .map_err(|error| invalid_receipt(format!("projecting live semantic plan: {error}")))?;
        if projected.identity().to_hex() != self.feature_plan_identity {
            return Err(invalid_receipt(
                "live feature formulas, source identities, alignment or frozen normalization differ from the recorded search plan",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
    ) -> Result<(), CanonicalDataSelectionError> {
        let feature_content_sha256 = canonical_feature_content_sha256(features)?;
        self.validate_against_with_content_sha256(anchor, features, &feature_content_sha256)
    }

    fn validate_against_with_content_sha256(
        &self,
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
        feature_content_sha256: &str,
    ) -> Result<(), CanonicalDataSelectionError> {
        let execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            resolved_canonical_feature_execution_authority_v1(),
        );
        self.validate_against_with_execution(anchor, features, feature_content_sha256, &execution)
    }

    fn validate_against_with_execution(
        &self,
        anchor: &CanonicalDatasetIdentity,
        features: &FeatureFrame,
        feature_content_sha256: &str,
        execution: &CanonicalFeatureExecutionReceiptV1,
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
        // Historical raw receipts have no saved recipe/plan bytes. They remain
        // readable, but can never validate a normalized frame without its fit.
        self.validate_normalization_plan(features.plan())?;
        if self.normalization_fitted_state.as_ref() != features.normalization_fitted_state() {
            return Err(provenance_mismatch(
                anchor,
                "persisted normalization fit does not match the loaded FeatureFrame fit",
            ));
        }
        if let Some(options) = self.feature_build_options.as_ref() {
            if Some(options) != features.feature_build_options() {
                return Err(provenance_mismatch(
                    anchor,
                    "persisted producer recipe does not match the loaded FeatureFrame recipe",
                ));
            }
        }
        if self.feature_provenance_identity != features.provenance_identity().to_hex() {
            return Err(provenance_mismatch(
                anchor,
                "feature provenance identity does not match loaded FeatureFrame",
            ));
        }
        if &self.feature_execution != execution {
            return Err(provenance_mismatch(
                anchor,
                "feature execution policy/math lane does not match the immutable current build authority",
            ));
        }
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

#[cfg(feature = "gpu-cuda")]
impl CanonicalGpuResidentSearchInputReceiptV3 {
    pub fn from_resident_store(
        anchor: &CanonicalDatasetIdentity,
        store: &SealedGpuResidentFeatureStoreV3,
    ) -> Result<Self, CanonicalDataSelectionError> {
        let row_count = store.contract().layout().row_count();
        let column_count = store.contract().layout().column_count();
        let receipt = Self {
            schema_version: CANONICAL_GPU_RESIDENT_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V3,
            anchor_dataset_identity: anchor.to_path_component(),
            feature_plan_identity: hex(&store.final_feature_plan_v3_sha256()),
            feature_provenance_identity: hex(&store.source_provenance_sha256()),
            content_merkle_algorithm: CANONICAL_GPU_RESIDENT_CONTENT_MERKLE_ALGORITHM_V3.to_owned(),
            feature_content_merkle_sha256: hex(&store
                .contract()
                .canonical_feature_content_merkle_sha256()),
            normalization_fit_sha256: hex(&store.normalization_fit_sha256()),
            feature_plan_canonical_bytes: Some(store.feature_plan().canonical_bytes().to_vec()),
            normalization_fitted_state: store.normalization_fitted_state().cloned(),
            row_count,
            column_count,
            feature_execution: CanonicalFeatureExecutionReceiptV1 {
                schema_version: CANONICAL_FEATURE_EXECUTION_SCHEMA_VERSION_V1,
                compute_policy: CanonicalFeatureComputePolicyV1::GpuOnly,
                vector_ta_math_authority: CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1.to_owned(),
                selected_lane: CanonicalFeatureMathLaneV1::GpuCudaF64Strict,
            },
            source_bindings: source_binding_receipts_from_store_v3(store),
        };
        receipt.validate_against_store(anchor, store)?;
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

    pub fn content_merkle_algorithm(&self) -> &str {
        &self.content_merkle_algorithm
    }

    pub fn feature_content_merkle_sha256(&self) -> &str {
        &self.feature_content_merkle_sha256
    }

    pub fn normalization_fit_sha256(&self) -> &str {
        &self.normalization_fit_sha256
    }

    pub fn feature_plan_canonical_bytes(&self) -> Option<&[u8]> {
        self.feature_plan_canonical_bytes.as_deref()
    }

    pub fn normalization_fitted_state(&self) -> Option<&SearchNormalizationFittedStateV1> {
        self.normalization_fitted_state.as_ref()
    }

    /// Decode the recorded recipe without rebuilding features or refitting.
    pub fn recorded_feature_plan(
        &self,
    ) -> Result<Option<FeaturePlanV1>, CanonicalDataSelectionError> {
        let plan = self
            .feature_plan_canonical_bytes
            .as_deref()
            .map(|bytes| {
                FeaturePlanV1::from_canonical_bytes(bytes).map_err(|error| {
                    invalid_receipt(format!("GPU-resident recorded plan: {error}"))
                })
            })
            .transpose()?;
        if let Some(plan) = &plan {
            if plan.identity().to_hex() != self.feature_plan_identity
                || plan.final_outputs().len() as u64 != self.column_count
            {
                return Err(invalid_receipt(
                    "GPU-resident recorded plan identity/shape differs",
                ));
            }
        }
        self.validate_normalization_metadata_v3(plan.as_ref())?;
        Ok(plan)
    }

    fn validate_normalization_metadata_v3(
        &self,
        plan: Option<&FeaturePlanV1>,
    ) -> Result<(), CanonicalDataSelectionError> {
        match &self.normalization_fitted_state {
            Some(state) => {
                let plan = plan.ok_or_else(|| {
                    invalid_receipt(
                        "GPU-resident fitted normalization requires the recorded feature plan",
                    )
                })?;
                state.validate_plan(plan).map_err(|error| {
                    invalid_receipt(format!("GPU-resident portable normalization: {error}"))
                })?;
                let training = state.training_rows().map_err(|error| {
                    invalid_receipt(format!("GPU-resident normalization training rows: {error}"))
                })?;
                let rows = usize::try_from(self.row_count)
                    .map_err(|_| invalid_receipt("GPU-resident parent rows exceed this process"))?;
                let expected_training =
                    crate::discovery::canonical_discovery_normalization_training_rows(rows)
                        .map_err(|error| {
                            invalid_receipt(format!("GPU-resident canonical fit scope: {error}"))
                        })?;
                if training != expected_training || state.column_names() != plan.final_outputs() {
                    return Err(invalid_receipt(
                        "GPU-resident fitted normalization geometry/schema differs",
                    ));
                }
                let word_count = state
                    .fits()
                    .len()
                    .checked_mul(6)
                    .ok_or_else(|| invalid_receipt("GPU-resident fit word-count overflow"))?;
                let mut words = Vec::new();
                words
                    .try_reserve_exact(word_count)
                    .map_err(|_| invalid_receipt("GPU-resident fit word allocation failed"))?;
                for fit in state.fits() {
                    words.extend_from_slice(&[
                        fit.training_rows.start as u64,
                        fit.training_rows.end as u64,
                        fit.median.to_bits(),
                        fit.scale.to_bits(),
                        fit.valid_training_cells as u64,
                        u64::from(fit.degenerate),
                    ]);
                }
                let transport = neoethos_gpu_contracts::normalization_v3::resident_normalization_fit_metadata_sha256_v3(&words);
                if self.normalization_fit_sha256 != hex(&transport) {
                    return Err(invalid_receipt(
                        "GPU-resident six-word fit transport digest differs from actual fitted state",
                    ));
                }
            }
            None => {
                let disabled = neoethos_gpu_cuda::resident_robust_normalization_v2::resident_robust_normalization_disabled_fit_sha256_v2();
                if self.normalization_fit_sha256 != hex(&disabled)
                    || plan.is_some_and(|plan| {
                        plan.nodes()
                            .iter()
                            .any(|node| node.operation() == FeatureOperationTagV1::Normalization)
                    })
                {
                    return Err(invalid_receipt(
                        "GPU-resident normalization is missing its actual fitted state",
                    ));
                }
                // Opaque legacy raw receipts cannot authorize replay. A supplied
                // live plan is checked again at the store-binding boundary.
            }
        }
        Ok(())
    }

    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    pub const fn column_count(&self) -> u64 {
        self.column_count
    }

    pub const fn feature_execution(&self) -> &CanonicalFeatureExecutionReceiptV1 {
        &self.feature_execution
    }

    pub fn source_bindings(&self) -> &[CanonicalSearchSourceBindingReceiptV1] {
        &self.source_bindings
    }

    pub fn validate(&self) -> Result<CanonicalDatasetIdentity, CanonicalDataSelectionError> {
        if self.schema_version != CANONICAL_GPU_RESIDENT_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V3 {
            return Err(invalid_receipt(format!(
                "unsupported GPU-resident input receipt schema version {}; expected {}",
                self.schema_version, CANONICAL_GPU_RESIDENT_SEARCH_INPUT_RECEIPT_SCHEMA_VERSION_V3
            )));
        }
        let anchor = CanonicalDatasetIdentity::from_path_component(&self.anchor_dataset_identity)
            .map_err(|error| {
            invalid_receipt(format!("GPU-resident anchor identity: {error}"))
        })?;
        validate_sha256_hex(
            "GPU-resident feature plan identity",
            &self.feature_plan_identity,
        )?;
        validate_sha256_hex(
            "GPU-resident feature provenance identity",
            &self.feature_provenance_identity,
        )?;
        validate_sha256_hex(
            "GPU-resident feature content Merkle SHA-256",
            &self.feature_content_merkle_sha256,
        )?;
        validate_sha256_hex(
            "GPU-resident normalization-fit SHA-256",
            &self.normalization_fit_sha256,
        )?;
        if self.content_merkle_algorithm != CANONICAL_GPU_RESIDENT_CONTENT_MERKLE_ALGORITHM_V3 {
            return Err(invalid_receipt(format!(
                "unsupported GPU-resident content Merkle algorithm `{}`",
                self.content_merkle_algorithm
            )));
        }
        if self.row_count == 0 || self.column_count == 0 {
            return Err(invalid_receipt(
                "GPU-resident input receipt has an empty row/column shape",
            ));
        }
        self.recorded_feature_plan()?;
        self.feature_execution.validate()?;
        if self.feature_execution.compute_policy() != CanonicalFeatureComputePolicyV1::GpuOnly
            || self.feature_execution.selected_lane()
                != CanonicalFeatureMathLaneV1::GpuCudaF64Strict
        {
            return Err(invalid_receipt(
                "GPU-resident input receipt is not bound to the strict CUDA lane",
            ));
        }
        validate_source_binding_receipts_v3(&anchor, &self.source_bindings)?;
        let anchor_id = anchor.to_path_component();
        let anchor_bindings = self
            .source_bindings
            .iter()
            .filter(|binding| binding.dataset_identity == anchor_id)
            .collect::<Vec<_>>();
        let anchor_rows = anchor_bindings[0]
            .segments
            .iter()
            .try_fold(0_u64, |rows, segment| {
                rows.checked_add(segment.row_end - segment.row_start)
                    .ok_or_else(|| invalid_receipt("GPU-resident anchor row-count overflow"))
            })?;
        if anchor_rows != self.row_count {
            return Err(provenance_mismatch(
                &anchor,
                format!(
                    "GPU-resident receipt has {} rows but its anchor segments cover {anchor_rows}",
                    self.row_count
                ),
            ));
        }
        Ok(anchor)
    }

    pub fn validate_against_store(
        &self,
        anchor: &CanonicalDatasetIdentity,
        store: &SealedGpuResidentFeatureStoreV3,
    ) -> Result<(), CanonicalDataSelectionError> {
        let received_anchor = self.validate()?;
        self.validate_normalization_metadata_v3(Some(store.feature_plan()))?;
        let device = store.device_identity();
        let execution_authority = store
            .validated_gpu_resident_feature_execution_authority_v1()
            .map_err(|error| {
                provenance_mismatch(
                    anchor,
                    format!("sealed Data store GPU execution authority is invalid: {error}"),
                )
            })?;
        let expected_bindings = source_binding_receipts_from_store_v3(store);
        if &received_anchor != anchor
            || self.feature_plan_identity != hex(&store.final_feature_plan_v3_sha256())
            || self.feature_plan_identity != store.feature_plan().identity().to_hex()
            || execution_authority.semantic()
                != CanonicalGpuResidentFeatureExecutionSemanticV1::GpuCudaF64Strict
            || execution_authority.final_feature_plan_v3_sha256()
                != store.final_feature_plan_v3_sha256()
            || self.feature_plan_identity
                != hex(&execution_authority.final_feature_plan_v3_sha256())
            || execution_authority.classic_ta_implementation_sha256()
                != execution_authority.vector_ta_build_sha256()
            || execution_authority.identity_sha256() == [0; 32]
            || self.feature_provenance_identity != hex(&store.source_provenance_sha256())
            || self.feature_provenance_identity != store.source_provenance().identity().to_hex()
            || self.feature_content_merkle_sha256
                != hex(&store.contract().canonical_feature_content_merkle_sha256())
            || self.normalization_fit_sha256 != hex(&store.normalization_fit_sha256())
            || self.normalization_fitted_state.as_ref() != store.normalization_fitted_state()
            || self.feature_plan_canonical_bytes.as_deref()
                != Some(store.feature_plan().canonical_bytes())
            || self.row_count != store.contract().layout().row_count()
            || self.column_count != store.contract().layout().column_count()
            || self.source_bindings != expected_bindings
            || device.vector_ta_build_sha256() == [0; 32]
        {
            return Err(provenance_mismatch(
                anchor,
                "GPU-resident receipt does not match the sealed Data store",
            ));
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String, CanonicalDataSelectionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize GPU-resident receipt: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_GPU_RESIDENT_SEARCH_INPUT_RECEIPT_HASH_DOMAIN_V3);
        hasher.update(bytes);
        Ok(hex(&hasher.finalize()))
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, CanonicalDataSelectionError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| invalid_receipt(format!("serialize GPU-resident JSON: {error}")))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, CanonicalDataSelectionError> {
        let receipt: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid_receipt(format!("parse GPU-resident JSON: {error}")))?;
        receipt.validate()?;
        Ok(receipt)
    }
}

#[cfg(feature = "gpu-cuda")]
impl CanonicalGpuResidentSearchArtifactScopeV3 {
    pub fn for_entire_receipt(
        role: CanonicalSearchWindowRoleV1,
        receipt: CanonicalGpuResidentSearchInputReceiptV3,
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
                "GPU-resident scope requires exactly one anchor source binding",
            ));
        }
        let segments = anchor_bindings[0].segments();
        for adjacent in segments.windows(2) {
            if adjacent[0].row_end() != adjacent[1].row_start() {
                return Err(provenance_mismatch(
                    &anchor,
                    "one GPU-resident scope cannot represent disjoint anchor segments",
                ));
            }
        }
        let first = segments
            .first()
            .ok_or_else(|| provenance_mismatch(&anchor, "GPU-resident anchor has no segments"))?;
        let last = segments
            .last()
            .ok_or_else(|| provenance_mismatch(&anchor, "GPU-resident anchor has no segments"))?;
        let evaluated_window = CanonicalSearchEvaluatedWindowV1::new(
            role,
            first.row_start(),
            last.row_end(),
            first.timestamp_start_ms(),
            last.timestamp_end_ms(),
        )?;
        let receipt_sha256 = receipt.identity_sha256()?;
        let scope = Self {
            schema_version: CANONICAL_GPU_RESIDENT_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V3,
            receipt,
            receipt_sha256,
            evaluated_window,
        };
        scope.validate()?;
        Ok(scope)
    }

    pub const fn receipt(&self) -> &CanonicalGpuResidentSearchInputReceiptV3 {
        &self.receipt
    }

    pub const fn evaluated_window(&self) -> &CanonicalSearchEvaluatedWindowV1 {
        &self.evaluated_window
    }

    pub fn validate(&self) -> Result<(), CanonicalDataSelectionError> {
        if self.schema_version != CANONICAL_GPU_RESIDENT_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V3 {
            return Err(invalid_receipt(format!(
                "unsupported GPU-resident artifact scope schema {}; expected {}",
                self.schema_version, CANONICAL_GPU_RESIDENT_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V3
            )));
        }
        if self.receipt_sha256 != self.receipt.identity_sha256()? {
            return Err(invalid_receipt(
                "GPU-resident artifact scope receipt SHA-256 drifted",
            ));
        }
        validate_gpu_resident_window_v3(&self.evaluated_window, &self.receipt)
    }

    pub fn identity_sha256(&self) -> Result<String, CanonicalDataSelectionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| {
            invalid_receipt(format!("serialize GPU-resident artifact scope: {error}"))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_GPU_RESIDENT_SEARCH_ARTIFACT_SCOPE_HASH_DOMAIN_V3);
        hasher.update(bytes);
        Ok(hex(&hasher.finalize()))
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
    canonical_feature_content_sha256_with_control(features, &FeatureBuildControl::default())
}

fn canonical_feature_content_sha256_with_control(
    features: &FeatureFrame,
    control: &FeatureBuildControl,
) -> Result<String, CanonicalDataSelectionError> {
    control
        .checkpoint()
        .map_err(|error| invalid_receipt(error.to_string()))?;
    let projection_plan =
        neoethos_data::adaptive_feature_projection_plan(features, rayon::current_num_threads())
            .map_err(|error| {
                invalid_receipt(format!("admit exact feature content projection: {error:#}"))
            })?;
    tracing::info!(
        target: "neoethos_search::canonical_receipt",
        rows = features.n_samples(),
        columns = features.n_features(),
        columns_per_projection = projection_plan.columns_per_batch,
        concurrent_projections = projection_plan.concurrent_batches,
        projection_budget_bytes = projection_plan.budget_bytes,
        "hashing the exact feature payload through adaptive bounded parallel projections"
    );
    canonical_feature_content_sha256_with_controlled_projection_schedule(
        features,
        projection_plan.columns_per_batch,
        projection_plan.concurrent_batches,
        control,
    )
}

#[cfg(test)]
fn canonical_feature_content_sha256_with_batch_columns(
    features: &FeatureFrame,
    batch_columns: usize,
) -> Result<String, CanonicalDataSelectionError> {
    canonical_feature_content_sha256_with_projection_schedule(features, batch_columns, 1)
}

#[cfg(test)]
fn canonical_feature_content_sha256_with_projection_schedule(
    features: &FeatureFrame,
    batch_columns: usize,
    concurrent_batches: usize,
) -> Result<String, CanonicalDataSelectionError> {
    canonical_feature_content_sha256_with_controlled_projection_schedule(
        features,
        batch_columns,
        concurrent_batches,
        &FeatureBuildControl::default(),
    )
}

fn canonical_feature_content_sha256_with_controlled_projection_schedule(
    features: &FeatureFrame,
    batch_columns: usize,
    concurrent_batches: usize,
    control: &FeatureBuildControl,
) -> Result<String, CanonicalDataSelectionError> {
    control
        .report(
            "feature_receipt",
            "hashing exact feature columns",
            0,
            features.n_features(),
        )
        .map_err(|error| invalid_receipt(error.to_string()))?;
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
    const HASH_BUFFER_BYTES: usize = 1024 * 1024;
    let mut byte_buffer = Vec::with_capacity(HASH_BUFFER_BYTES);
    for timestamps in features.timestamps.chunks((HASH_BUFFER_BYTES / 8).max(1)) {
        control
            .checkpoint()
            .map_err(|error| invalid_receipt(error.to_string()))?;
        byte_buffer.clear();
        for timestamp in timestamps {
            byte_buffer.extend_from_slice(&timestamp.to_le_bytes());
        }
        hasher.update(&byte_buffer);
    }

    let batch_columns = batch_columns.max(1);
    let concurrent_batches = concurrent_batches.max(1);
    let wave_columns = batch_columns.saturating_mul(concurrent_batches).max(1);
    let column_indices = (0..features.n_features()).collect::<Vec<_>>();
    let total_waves = features.n_features().div_ceil(wave_columns);
    let started = Instant::now();
    for (wave_index, wave) in column_indices.chunks(wave_columns).enumerate() {
        control
            .checkpoint()
            .map_err(|error| invalid_receipt(error.to_string()))?;
        tracing::info!(
            target: "neoethos_search::canonical_receipt",
            wave = wave_index + 1,
            wave_count = total_waves,
            first_column = wave[0],
            last_column = wave[wave.len() - 1],
            "canonical feature content projection wave started"
        );
        let projections = wave
            .par_chunks(batch_columns)
            .map(|indices| {
                control
                    .checkpoint()
                    .map_err(|error| invalid_receipt(error.to_string()))?;
                let projection = features
                    .project_columns(indices, 0..features.n_samples())
                    .map_err(|error| {
                        invalid_receipt(format!(
                            "materialize exact feature columns {}..{} for content receipt: {error}",
                            indices[0],
                            indices[indices.len() - 1]
                        ))
                    })?;
                control
                    .checkpoint()
                    .map_err(|error| invalid_receipt(error.to_string()))?;
                if projection.timestamps != features.timestamps {
                    return Err(invalid_receipt(
                        "feature projection timestamps changed while hashing the content receipt",
                    ));
                }
                if projection.columns.len() != indices.len() {
                    return Err(invalid_receipt(format!(
                        "feature projection returned {} columns for {} requested indices",
                        projection.columns.len(),
                        indices.len()
                    )));
                }
                Ok(projection)
            })
            .collect::<Result<Vec<_>, CanonicalDataSelectionError>>()?;

        // SHA-256 is intentionally fed in the original column-major order.
        // Only independent Vortex projection/decode work runs concurrently, so
        // RAM, Vortex, and view-backed inputs retain the exact V1 digest.
        for (indices, projection) in wave.chunks(batch_columns).zip(projections) {
            for (&index, column) in indices.iter().zip(&projection.columns) {
                let name = &features.names[index];
                let name_len = u64::try_from(name.len())
                    .map_err(|_| invalid_receipt("feature name length does not fit u64"))?;
                hasher.update(name_len.to_le_bytes());
                hasher.update(name.as_bytes());

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
                for row_range in (0..column.values.len())
                    .step_by((HASH_BUFFER_BYTES / 9).max(1))
                    .map(|start| start..(start + HASH_BUFFER_BYTES / 9).min(column.values.len()))
                {
                    control
                        .checkpoint()
                        .map_err(|error| invalid_receipt(error.to_string()))?;
                    byte_buffer.clear();
                    for row in row_range {
                        byte_buffer.extend_from_slice(&column.values[row].to_bits().to_le_bytes());
                        byte_buffer.push(column.validity[row].code());
                    }
                    hasher.update(&byte_buffer);
                }
                control
                    .report("feature_receipt", name, index + 1, features.n_features())
                    .map_err(|error| invalid_receipt(error.to_string()))?;
            }
        }
        tracing::info!(
            target: "neoethos_search::canonical_receipt",
            wave = wave_index + 1,
            wave_count = total_waves,
            completed_columns = (wave_index + 1)
                .saturating_mul(wave_columns)
                .min(features.n_features()),
            elapsed_ms = started.elapsed().as_millis(),
            "canonical feature content projection wave completed"
        );
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

#[cfg(feature = "gpu-cuda")]
fn source_binding_receipts_from_store_v3(
    store: &neoethos_data::SealedGpuResidentFeatureStoreV3,
) -> Vec<CanonicalSearchSourceBindingReceiptV1> {
    let mut bindings = store
        .source_provenance()
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

#[cfg(feature = "gpu-cuda")]
fn validate_source_binding_receipts_v3(
    anchor: &CanonicalDatasetIdentity,
    bindings: &[CanonicalSearchSourceBindingReceiptV1],
) -> Result<(), CanonicalDataSelectionError> {
    if bindings.is_empty() {
        return Err(invalid_receipt(
            "GPU-resident receipt source bindings are empty",
        ));
    }
    let mut previous_node: Option<&str> = None;
    let mut contains_anchor = false;
    for binding in bindings {
        validate_nonempty("GPU-resident source node id", &binding.source_node_id)?;
        if previous_node.is_some_and(|previous| previous >= binding.source_node_id.as_str()) {
            return Err(invalid_receipt(format!(
                "GPU-resident source bindings are not strictly ordered or repeat `{}`",
                binding.source_node_id
            )));
        }
        previous_node = Some(&binding.source_node_id);
        let identity = CanonicalDatasetIdentity::from_path_component(&binding.dataset_identity)
            .map_err(|error| {
                invalid_receipt(format!(
                    "GPU-resident source node `{}` dataset identity: {error}",
                    binding.source_node_id
                ))
            })?;
        contains_anchor |= &identity == anchor;
        validate_nonempty(
            "GPU-resident manifest schema id",
            &binding.manifest_schema_id,
        )?;
        validate_sha256_hex("GPU-resident manifest SHA-256", &binding.manifest_sha256)?;
        validate_nonempty("GPU-resident generation id", &binding.generation_id)?;
        validate_sha256_hex("GPU-resident Vortex SHA-256", &binding.vortex_sha256)?;
        if binding.bar_timestamp_convention != identity.bar_timestamp_convention().to_string() {
            return Err(invalid_receipt(format!(
                "GPU-resident source node `{}` bar convention disagrees with its identity",
                binding.source_node_id
            )));
        }
        validate_segments(&binding.source_node_id, &binding.segments)?;
    }
    if !contains_anchor {
        return Err(provenance_mismatch(
            anchor,
            "GPU-resident receipt has no exact anchor source binding",
        ));
    }
    let anchor_count = bindings
        .iter()
        .filter(|binding| binding.dataset_identity == anchor.to_path_component())
        .count();
    if anchor_count != 1 {
        return Err(provenance_mismatch(
            anchor,
            format!(
                "GPU-resident receipt requires one anchor source binding; found {anchor_count}"
            ),
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-cuda")]
fn validate_gpu_resident_window_v3(
    window: &CanonicalSearchEvaluatedWindowV1,
    receipt: &CanonicalGpuResidentSearchInputReceiptV3,
) -> Result<(), CanonicalDataSelectionError> {
    window.validate_shape()?;
    let anchor = receipt.validate()?;
    let anchor_id = anchor.to_path_component();
    let anchor_binding = receipt
        .source_bindings()
        .iter()
        .find(|binding| binding.dataset_identity() == anchor_id)
        .ok_or_else(|| {
            provenance_mismatch(&anchor, "GPU-resident scope anchor binding is absent")
        })?;
    let segments = anchor_binding.segments();
    let first = segments
        .first()
        .ok_or_else(|| provenance_mismatch(&anchor, "GPU-resident anchor has no segments"))?;
    let last = segments
        .last()
        .ok_or_else(|| provenance_mismatch(&anchor, "GPU-resident anchor has no segments"))?;
    if window.timestamp_start_ms() < first.timestamp_start_ms()
        || window.timestamp_end_ms() > last.timestamp_end_ms()
    {
        return Err(provenance_mismatch(
            &anchor,
            "GPU-resident evaluated timestamps fall outside the anchor segments",
        ));
    }
    let mut cursor = window.row_start();
    for segment in segments {
        if cursor >= window.row_end() {
            break;
        }
        if cursor < segment.row_start() || cursor >= segment.row_end() {
            continue;
        }
        cursor = window.row_end().min(segment.row_end());
    }
    if cursor != window.row_end() {
        return Err(provenance_mismatch(
            &anchor,
            "GPU-resident evaluated row window is not covered by contiguous anchor segments",
        ));
    }
    Ok(())
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

#[cfg(all(test, feature = "gpu-cuda"))]
pub(crate) fn normalization_receipt_codec_fixture_v3(
    normalized: bool,
) -> CanonicalGpuResidentSearchInputReceiptV3 {
    // Host codec fixture only. It cannot construct a sealed resident Data store,
    // device admission, execution receipt or production runtime authority.
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};
    let timestamps = (0..100)
        .map(|row| 1_704_067_200_000_i64 + row * 60_000)
        .collect();
    let columns = vec![
        FeatureColumnF64::new(
            "ramp",
            (0..100).map(|row| row as f64).collect(),
            vec![FeatureCellValidity::Valid; 100],
        )
        .unwrap(),
        FeatureColumnF64::new(
            "zero_δοκιμή",
            vec![-0.0; 100],
            vec![FeatureCellValidity::Valid; 100],
        )
        .unwrap(),
    ];
    let frame = if normalized {
        neoethos_data::test_fixtures::ctrader_test_normalized_feature_frame_from_columns(
            timestamps,
            columns,
            FeatureBuildOptions {
                normalization_training_rows: Some(0..80),
                ..FeatureBuildOptions::default()
            },
        )
        .unwrap()
    } else {
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(timestamps, columns)
            .unwrap()
    };
    let state = frame.normalization_fitted_state().cloned();
    let transport = match &state {
        Some(state) => {
            let words: Vec<_> = state.fits().iter().flat_map(|fit| [
                fit.training_rows.start as u64, fit.training_rows.end as u64,
                fit.median.to_bits(), fit.scale.to_bits(), fit.valid_training_cells as u64,
                u64::from(fit.degenerate),
            ]).collect();
            neoethos_gpu_contracts::normalization_v3::resident_normalization_fit_metadata_sha256_v3(&words)
        }
        None => neoethos_gpu_cuda::resident_robust_normalization_v2::resident_robust_normalization_disabled_fit_sha256_v2(),
    };
    CanonicalGpuResidentSearchInputReceiptV3 {
        schema_version: 3,
        anchor_dataset_identity: frame.provenance().bindings()[0]
            .dataset_identity()
            .to_path_component(),
        feature_plan_identity: frame.plan_identity().to_hex(),
        feature_provenance_identity: frame.provenance_identity().to_hex(),
        content_merkle_algorithm: CANONICAL_GPU_RESIDENT_CONTENT_MERKLE_ALGORITHM_V3.to_owned(),
        feature_content_merkle_sha256: "a".repeat(64),
        normalization_fit_sha256: hex(&transport),
        feature_plan_canonical_bytes: Some(frame.plan().canonical_bytes().to_vec()),
        normalization_fitted_state: state,
        row_count: 100,
        column_count: 2,
        feature_execution: CanonicalFeatureExecutionReceiptV1 {
            schema_version: 1,
            compute_policy: CanonicalFeatureComputePolicyV1::GpuOnly,
            vector_ta_math_authority: CANONICAL_VECTOR_TA_CUDA_MATH_AUTHORITY_V1.to_owned(),
            selected_lane: CanonicalFeatureMathLaneV1::GpuCudaF64Strict,
        },
        source_bindings: source_binding_receipts(&frame),
    }
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod resident_normalization_receipt_v3_tests {
    use super::*;

    #[test]
    fn actual_fit_bits_and_portable_plan_survive_receipt_and_scope_roundtrip() {
        let receipt = normalization_receipt_codec_fixture_v3(true);
        let state = receipt.normalization_fitted_state().unwrap();
        assert_eq!(state.training_rows().unwrap(), 0..80);
        assert_eq!(state.fits()[1].median.to_bits(), (-0.0_f64).to_bits());
        assert!(state.fits()[1].degenerate);
        assert_ne!(
            receipt.normalization_fit_sha256(),
            hex(&state.fitted_state_hash().unwrap())
        );
        let decoded = CanonicalGpuResidentSearchInputReceiptV3::from_json_bytes(
            &receipt.to_json_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, receipt);
        state
            .validate_plan(&decoded.recorded_feature_plan().unwrap().unwrap())
            .unwrap();
        let scope = CanonicalGpuResidentSearchArtifactScopeV3::for_entire_receipt(
            CanonicalSearchWindowRoleV1::DiscoveryInput,
            receipt.clone(),
        )
        .unwrap();
        let decoded_scope: CanonicalGpuResidentSearchArtifactScopeV3 =
            serde_json::from_slice(&serde_json::to_vec(&scope).unwrap()).unwrap();
        decoded_scope.validate().unwrap();
        assert_eq!(
            decoded_scope.receipt().normalization_fitted_state(),
            Some(state)
        );
    }

    #[test]
    fn missing_or_changed_device_fit_plan_and_transport_fail_closed() {
        let receipt = normalization_receipt_codec_fixture_v3(true);
        let value = serde_json::to_value(&receipt).unwrap();
        for (pointer, replacement) in [
            ("/normalization_fitted_state", serde_json::Value::Null),
            ("/feature_plan_canonical_bytes", serde_json::Value::Null),
            (
                "/normalization_fit_sha256",
                serde_json::json!("b".repeat(64)),
            ),
            (
                "/normalization_fit_sha256",
                serde_json::json!(hex(&receipt
                    .normalization_fitted_state()
                    .unwrap()
                    .fitted_state_hash()
                    .unwrap())),
            ),
            (
                "/normalization_fitted_state/column_names/0",
                serde_json::json!("different"),
            ),
            (
                "/normalization_fitted_state/fits/0/median",
                serde_json::json!("3ff0000000000000"),
            ),
            (
                "/normalization_fitted_state/fits/1/median",
                serde_json::json!("0000000000000000"),
            ),
            (
                "/normalization_fitted_state/fits/0/training_rows/end",
                serde_json::json!(79),
            ),
            (
                "/normalization_fitted_state/fits/0/valid_training_cells",
                serde_json::json!(79),
            ),
            (
                "/normalization_fitted_state/fits/1/degenerate",
                serde_json::json!(false),
            ),
        ] {
            let mut changed = value.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                CanonicalGpuResidentSearchInputReceiptV3::from_json_bytes(
                    &serde_json::to_vec(&changed).unwrap()
                )
                .is_err(),
                "{pointer}"
            );
        }
        let mut missing_both = receipt.clone();
        missing_both.normalization_fitted_state = None;
        missing_both.feature_plan_canonical_bytes = None;
        assert!(missing_both.validate().is_err());
        let mut corrupt_plan = receipt;
        corrupt_plan.feature_plan_canonical_bytes.as_mut().unwrap()[0] ^= 1;
        assert!(corrupt_plan.validate().is_err());
    }

    #[test]
    fn raw_legacy_is_opaque_only_and_cannot_hide_a_normalization_node() {
        let mut raw = normalization_receipt_codec_fixture_v3(false);
        raw.validate().unwrap();
        raw.feature_plan_canonical_bytes = None;
        let bytes = raw.to_json_bytes().unwrap();
        let decoded = CanonicalGpuResidentSearchInputReceiptV3::from_json_bytes(&bytes).unwrap();
        assert!(decoded.recorded_feature_plan().unwrap().is_none());
        assert_eq!(decoded.to_json_bytes().unwrap(), bytes);
        raw.normalization_fit_sha256 = "a".repeat(64);
        assert!(raw.validate().is_err());
        let mut disguised = normalization_receipt_codec_fixture_v3(true);
        disguised.normalization_fitted_state = None;
        disguised.normalization_fit_sha256 = decoded.normalization_fit_sha256;
        assert!(disguised.validate().is_err());
        let source = include_str!("data_selection.rs");
        let binding = source
            .split_once("impl CanonicalGpuResidentSearchInputReceiptV3 {")
            .unwrap()
            .1
            .split_once("pub fn validate_against_store(")
            .unwrap()
            .1
            .split_once("pub fn identity_sha256(")
            .unwrap()
            .0;
        let compact: String = binding.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(compact.contains("self.feature_plan_canonical_bytes.as_deref()!=Some(store.feature_plan().canonical_bytes())"));
        assert!(compact.contains(
            "self.normalization_fitted_state.as_ref()!=store.normalization_fitted_state()"
        ));
    }
}

#[cfg(test)]
mod canonical_feature_content_hash_tests {
    use super::*;
    use neoethos_data::core::feature_run_lease::FeatureRunLease;
    use neoethos_data::core::vortex_feature_store::{
        VortexFeatureStore, VortexFeatureStoreOptions,
    };
    use neoethos_data::test_fixtures::{
        ctrader_test_feature_frame_from_columns, ctrader_test_normalized_feature_frame_from_columns,
    };
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureData};
    use std::sync::Arc;

    fn exact_hash_fixture() -> FeatureFrame {
        use FeatureCellValidity::{
            AlignmentMissing, ComputeFailure, Degenerate, Gap, MissingInput, NonFinite, Stale,
            Valid, Warmup, ZeroDenominator,
        };
        let timestamps = (0..10)
            .map(|row| 1_704_067_200_000_i64 + row * 60_000)
            .collect::<Vec<_>>();
        let validity = vec![
            Valid,
            Warmup,
            MissingInput,
            Gap,
            Stale,
            ZeroDenominator,
            Degenerate,
            NonFinite,
            ComputeFailure,
            AlignmentMissing,
        ];
        let columns = vec![
            FeatureColumnF64::new(
                "exact_bits",
                vec![-0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                validity,
            )
            .expect("all validity codes"),
            FeatureColumnF64::new(
                "unicode_δοκιμή",
                vec![
                    f64::MIN_POSITIVE,
                    0.1,
                    -123_456.789_012_345,
                    1.0 / 3.0,
                    42.0,
                    -7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                ],
                vec![Valid; 10],
            )
            .expect("exact finite values"),
            FeatureColumnF64::new(
                "mixed_validity",
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![
                    Valid, Valid, Gap, Valid, Warmup, Valid, Stale, Valid, Valid, Valid,
                ],
            )
            .expect("mixed validity"),
        ];
        ctrader_test_feature_frame_from_columns(timestamps, columns)
            .expect("canonical exact-hash fixture")
    }

    fn normalized_hash_fixture() -> FeatureFrame {
        let raw = exact_hash_fixture();
        let columns = raw
            .project_columns(
                &(0..raw.n_features()).collect::<Vec<_>>(),
                0..raw.n_samples(),
            )
            .expect("raw fixture columns")
            .columns
            .clone();
        ctrader_test_normalized_feature_frame_from_columns(
            raw.timestamps.clone(),
            columns,
            FeatureBuildOptions {
                normalization_training_rows: Some(0..6),
                ..FeatureBuildOptions::default()
            },
        )
        .expect("train-prefix normalized fixture")
    }

    #[test]
    fn shared_scope_reference_preserves_exact_identity_and_refuses_substitution() {
        for frame in [exact_hash_fixture(), normalized_hash_fixture()] {
            let anchor = frame.provenance().bindings()[0].dataset_identity();
            let receipt =
                CanonicalSearchInputReceiptV2::from_feature_frame(anchor, &frame).unwrap();
            for role in [
                CanonicalSearchWindowRoleV1::DiscoveryInput,
                CanonicalSearchWindowRoleV1::InSample,
                CanonicalSearchWindowRoleV1::Holdout,
                CanonicalSearchWindowRoleV1::WalkForwardTrain,
                CanonicalSearchWindowRoleV1::WalkForwardValidation,
                CanonicalSearchWindowRoleV1::ForwardTest,
                CanonicalSearchWindowRoleV1::LiveSimulation,
                CanonicalSearchWindowRoleV1::PropFirmRisk,
                CanonicalSearchWindowRoleV1::SelectionValidation,
            ] {
                let window = CanonicalSearchEvaluatedWindowV1::new(
                    role,
                    2,
                    8,
                    frame.timestamps[2],
                    frame.timestamps[7],
                )
                .unwrap();
                // The previous constructor built these fields, then performed
                // public validation. Preserve its exact bytes and hash framing.
                let prior = CanonicalSearchArtifactScopeV2 {
                    schema_version: CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2,
                    receipt: receipt.clone(),
                    receipt_sha256: receipt.identity_sha256().unwrap(),
                    evaluated_window: window.clone(),
                };
                prior.validate().unwrap();
                let prior_bytes = serde_json::to_vec(&prior).unwrap();
                let mut expected = Sha256::new();
                expected.update(CANONICAL_SEARCH_ARTIFACT_SCOPE_HASH_DOMAIN_V2);
                expected.update(&prior_bytes);
                let expected = hex(&expected.finalize());
                let actual = CanonicalSearchArtifactScopeV2::new(receipt.clone(), window).unwrap();
                assert_eq!(actual.to_json_bytes().unwrap(), prior_bytes);
                assert_eq!(actual.identity_sha256().unwrap(), expected);
                let attached = CanonicalSearchArtifactScopeRefV1::from_scope(&actual)
                    .unwrap()
                    .attach(&receipt)
                    .unwrap();
                assert_eq!(attached.to_json_bytes().unwrap(), prior_bytes);
                assert_eq!(attached.identity_sha256().unwrap(), expected);
            }
            let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
                CanonicalSearchWindowRoleV1::DiscoveryInput,
                receipt.clone(),
            )
            .unwrap();
            let original_bytes = scope.to_json_bytes().unwrap();
            let compact = CanonicalSearchArtifactScopeRefV1::from_scope(&scope).unwrap();
            let bytes = serde_json::to_vec(&compact).unwrap();
            assert!(!String::from_utf8_lossy(&bytes).contains("feature_plan_canonical_bytes"));
            let compact: CanonicalSearchArtifactScopeRefV1 =
                serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                compact.attach(&receipt).unwrap().to_json_bytes().unwrap(),
                original_bytes
            );
            let mut foreign = receipt.clone();
            foreign.feature_content_sha256 = "00".repeat(32);
            assert!(compact.attach(&foreign).is_err());
            let mut changed = compact.clone();
            changed.evaluated_window.role = CanonicalSearchWindowRoleV1::Holdout;
            assert!(changed.attach(&receipt).is_err());
            let mut changed = compact.clone();
            changed.evaluated_window.timestamp_end_ms -= 1;
            assert!(changed.attach(&receipt).is_err());
            let mut changed = compact.clone();
            changed.evaluated_window.row_end -= 1;
            assert!(changed.attach(&receipt).is_err());
            let mut changed = compact;
            changed.schema_version += 1;
            assert!(changed.attach(&receipt).is_err());
            let json = String::from_utf8(bytes).unwrap();
            let unknown = json.replacen('{', "{\"unknown\":true,", 1);
            assert!(serde_json::from_str::<CanonicalSearchArtifactScopeRefV1>(&unknown).is_err());
            let duplicate = json.replacen('{', "{\"schema_version\":1,", 1);
            assert!(serde_json::from_str::<CanonicalSearchArtifactScopeRefV1>(&duplicate).is_err());
        }
    }

    #[test]
    fn shared_scope_construction_and_attachment_preserve_prior_validation_errors() {
        fn prior_new(
            receipt: CanonicalSearchInputReceiptV2,
            evaluated_window: CanonicalSearchEvaluatedWindowV1,
        ) -> Result<CanonicalSearchArtifactScopeV2, CanonicalDataSelectionError> {
            let receipt_sha256 = receipt.identity_sha256()?;
            let scope = CanonicalSearchArtifactScopeV2 {
                schema_version: CANONICAL_SEARCH_ARTIFACT_SCOPE_SCHEMA_VERSION_V2,
                receipt,
                receipt_sha256,
                evaluated_window,
            };
            scope.validate()?;
            Ok(scope)
        }
        fn prior_attach(
            reference: &CanonicalSearchArtifactScopeRefV1,
            receipt: &CanonicalSearchInputReceiptV2,
        ) -> Result<CanonicalSearchArtifactScopeV2, CanonicalDataSelectionError> {
            if reference.schema_version != 1 {
                return Err(invalid_receipt(
                    "unsupported shared artifact-scope reference schema",
                ));
            }
            validate_sha256_hex("shared scope receipt SHA-256", &reference.receipt_sha256)?;
            validate_sha256_hex("shared scope SHA-256", &reference.scope_sha256)?;
            if receipt.identity_sha256()? != reference.receipt_sha256 {
                return Err(invalid_receipt(
                    "shared artifact-scope reference names a different receipt",
                ));
            }
            let scope = prior_new(receipt.clone(), reference.evaluated_window.clone())?;
            if scope.identity_sha256()? != reference.scope_sha256 {
                return Err(invalid_receipt(
                    "shared artifact-scope reference changed its exact window or identity",
                ));
            }
            Ok(scope)
        }
        let frame = exact_hash_fixture();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            frame.provenance().bindings()[0].dataset_identity(),
            &frame,
        )
        .unwrap();
        let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
            CanonicalSearchWindowRoleV1::Holdout,
            receipt.clone(),
        )
        .unwrap();
        let reference = CanonicalSearchArtifactScopeRefV1::from_scope(&scope).unwrap();
        let window_mutations: &[fn(&mut CanonicalSearchEvaluatedWindowV1)] = &[
            |w| w.row_end = w.row_start,
            |w| w.row_start = w.row_end + 1,
            |w| w.row_end += 1,
            |w| w.timestamp_start_ms = w.timestamp_end_ms + 1,
            |w| w.timestamp_start_ms -= 1,
            |w| w.timestamp_end_ms += 1,
        ];
        for mutate in window_mutations {
            let mut changed = reference.clone();
            mutate(&mut changed.evaluated_window);
            assert_eq!(
                CanonicalSearchArtifactScopeV2::new(
                    receipt.clone(),
                    changed.evaluated_window.clone()
                )
                .unwrap_err(),
                prior_new(receipt.clone(), changed.evaluated_window.clone()).unwrap_err(),
            );
            assert_eq!(
                changed.attach(&receipt).unwrap_err(),
                prior_attach(&changed, &receipt).unwrap_err()
            );
        }
        let mut corrupt = receipt.clone();
        corrupt.feature_plan_canonical_bytes.as_mut().unwrap()[0] ^= 1;
        assert_eq!(
            CanonicalSearchArtifactScopeV2::new(corrupt.clone(), scope.evaluated_window.clone())
                .unwrap_err(),
            prior_new(corrupt.clone(), scope.evaluated_window.clone()).unwrap_err(),
        );
        assert_eq!(
            reference.attach(&corrupt).unwrap_err(),
            prior_attach(&reference, &corrupt).unwrap_err()
        );
        let mut gapped = receipt.clone();
        let mut first = gapped.source_bindings[0].segments[0].clone();
        first.row_end = 4;
        first.timestamp_end_ms = frame.timestamps[3];
        let mut last = gapped.source_bindings[0].segments[0].clone();
        last.row_start = 5;
        last.timestamp_start_ms = frame.timestamps[5];
        gapped.source_bindings[0].segments = vec![first, last];
        let mut gap_reference = reference.clone();
        gap_reference.receipt_sha256 = gapped.identity_sha256().unwrap();
        assert_eq!(
            CanonicalSearchArtifactScopeV2::new(gapped.clone(), scope.evaluated_window.clone())
                .unwrap_err(),
            prior_new(gapped.clone(), scope.evaluated_window.clone()).unwrap_err(),
        );
        assert_eq!(
            gap_reference.attach(&gapped).unwrap_err(),
            prior_attach(&gap_reference, &gapped).unwrap_err()
        );
        let reference_mutations: &[fn(&mut CanonicalSearchArtifactScopeRefV1)] = &[
            |r| r.schema_version += 1,
            |r| r.receipt_sha256 = "invalid".into(),
            |r| r.receipt_sha256 = "00".repeat(32),
            |r| r.scope_sha256 = "invalid".into(),
            |r| r.scope_sha256 = "00".repeat(32),
            |r| r.evaluated_window.role = CanonicalSearchWindowRoleV1::SelectionValidation,
        ];
        for mutate in reference_mutations {
            let mut changed = reference.clone();
            mutate(&mut changed);
            assert_eq!(
                changed.attach(&receipt).unwrap_err(),
                prior_attach(&changed, &receipt).unwrap_err()
            );
        }
    }

    #[test]
    fn shared_scope_public_identity_revalidates_after_prior_success() {
        let frame = normalized_hash_fixture();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            frame.provenance().bindings()[0].dataset_identity(),
            &frame,
        )
        .unwrap();
        let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
            CanonicalSearchWindowRoleV1::DiscoveryInput,
            receipt,
        )
        .unwrap();
        scope.identity_sha256().unwrap();
        let mutations: &[fn(&mut CanonicalSearchArtifactScopeV2)] = &[
            |s| s.schema_version += 1,
            |s| s.receipt_sha256 = "00".repeat(32),
            |s| s.receipt.feature_content_sha256 = "00".repeat(32),
            |s| s.receipt.feature_plan_canonical_bytes.as_mut().unwrap()[0] ^= 1,
            |s| s.evaluated_window.row_end += 1,
            |s| s.evaluated_window.timestamp_start_ms -= 1,
        ];
        for mutate in mutations {
            let mut changed = scope.clone();
            mutate(&mut changed);
            let expected = changed.validate().unwrap_err();
            assert_eq!(changed.identity_sha256().unwrap_err(), expected);
            assert_eq!(
                CanonicalSearchArtifactScopeRefV1::from_scope(&changed).unwrap_err(),
                expected
            );
            assert_eq!(
                CanonicalSearchArtifactScopeV2::from_json_bytes(
                    &serde_json::to_vec(&changed).unwrap()
                )
                .unwrap_err(),
                expected,
            );
        }
    }

    fn canonical_input_fixture() -> (tempfile::TempDir, CanonicalSearchInput) {
        use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
        use neoethos_data::{
            BarTimestampConvention, CanonicalOhlcvPublishRequest, CanonicalVolumeRef,
            publish_canonical_ohlcv_generation,
        };
        use neoethos_feature_contracts::{
            DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1,
        };
        let root = tempfile::tempdir().expect("isolated canonical receipt fixture");
        let anchor = CanonicalDatasetIdentity::external(
            "recorded-receipt-test",
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("fixture identity");
        let ohlcv = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
        let provenance = ProducerProvenanceEnvelopeV1::new(
            "neoethos.recorded-receipt-test.v1",
            anchor.canonical_bytes(),
        )
        .expect("fixture provenance");
        publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
            configured_root: root.path(),
            identity: &anchor,
            expected_generation: None,
            provenance: &provenance,
            ohlcv: &ohlcv,
            volume: CanonicalVolumeRef::Absent,
            rows_per_chunk: 128,
        })
        .expect("publish bounded canonical fixture");
        let dataset =
            neoethos_data::load_dataset_for_identity_with_timeframes(root.path(), &anchor, &["M1"])
                .expect("open published canonical fixture");
        let base = dataset.canonical_frame("M1").expect("canonical base");
        let source_id = "source:recorded-receipt-test";
        let source = FeatureNodeV1::source(
            source_id,
            anchor.clone(),
            "neoethos.recorded-receipt-test.close.v1",
            1,
            vec![
                FeatureOutputV1::f64("close", 1).unwrap(),
                FeatureOutputV1::f64("open", 1).unwrap(),
            ],
            [1; 32],
        )
        .expect("fixture source node");
        let plan = FeaturePlanV1::new(vec![source], vec!["close".to_owned(), "open".to_owned()])
            .expect("fixture feature plan");
        let provenance = DatasetFeatureArtifactProvenanceV1::new(
            &plan,
            vec![base.source_binding(source_id).expect("exact base binding")],
        )
        .expect("fixture feature provenance");
        let features = FeatureFrame::from_columns(
            base.ohlcv().timestamp.clone().expect("base timestamps"),
            vec![
                FeatureColumnF64::new(
                    "close",
                    base.ohlcv().close.clone(),
                    vec![FeatureCellValidity::Valid; base.len()],
                )
                .expect("fixture close column"),
                FeatureColumnF64::new(
                    "open",
                    base.ohlcv().open.clone(),
                    vec![FeatureCellValidity::Valid; base.len()],
                )
                .expect("fixture open column"),
            ],
            plan,
            provenance,
        )
        .expect("bounded raw feature frame");
        let input = CanonicalSearchInput::from_prepared_canonical_frame(anchor, base, features)
            .expect("exact canonical fixture input");
        (root, input)
    }

    fn scalar_reference_hash(features: &FeatureFrame) -> String {
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V1);
        hasher.update((features.n_samples() as u64).to_le_bytes());
        hasher.update((features.n_features() as u64).to_le_bytes());
        for timestamp in &features.timestamps {
            hasher.update(timestamp.to_le_bytes());
        }
        for (index, name) in features.names.iter().enumerate() {
            hasher.update((name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            let column = features.feature_column(index).expect("reference column");
            for (value, validity) in column.values.iter().zip(&column.validity) {
                hasher.update(value.to_bits().to_le_bytes());
                hasher.update([validity.code()]);
            }
        }
        hex(&hasher.finalize())
    }

    fn vortex_copy(features: &FeatureFrame, scratch_root: &Path) -> FeatureFrame {
        let FeatureData::InMemory(columns) = &features.data else {
            panic!("fixture must start in RAM")
        };
        let lease = Arc::new(
            FeatureRunLease::create(scratch_root, "canonical-hash-parity")
                .expect("Vortex run lease"),
        );
        let store = VortexFeatureStore::create(
            lease,
            &features.timestamps,
            columns,
            VortexFeatureStoreOptions::default(),
        )
        .expect("persist hash fixture");
        FeatureFrame::from_vortex(
            features.timestamps.clone(),
            store,
            features.plan().clone(),
            features.provenance().clone(),
        )
        .expect("Vortex-backed hash fixture")
    }

    #[test]
    fn controlled_receipt_hash_reports_real_columns_without_changing_any_digest_bit() {
        let frame = exact_hash_fixture();
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&events);
        let control = FeatureBuildControl::default().with_observer(move |event| {
            observed.lock().expect("progress lock").push(event);
        });
        let actual = canonical_feature_content_sha256_with_controlled_projection_schedule(
            &frame, 1, 2, &control,
        )
        .expect("controlled hash");
        assert_eq!(actual, scalar_reference_hash(&frame));
        let events = events.lock().expect("progress lock");
        assert_eq!(
            events.iter().map(|e| e.completed).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(
            events
                .iter()
                .all(|e| e.stage == "feature_receipt" && e.total == 3)
        );
    }

    #[test]
    fn controlled_receipt_hash_stops_between_columns_and_never_returns_a_partial_digest() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let frame = exact_hash_fixture();
        let cancelled = Arc::new(AtomicBool::new(false));
        let observer_cancel = Arc::clone(&cancelled);
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&events);
        let control =
            FeatureBuildControl::new(Arc::clone(&cancelled)).with_observer(move |event| {
                let completed = event.completed;
                observed.lock().expect("progress lock").push(completed);
                if completed == 1 {
                    observer_cancel.store(true, Ordering::Release);
                }
            });
        let error = canonical_feature_content_sha256_with_controlled_projection_schedule(
            &frame, 1, 2, &control,
        )
        .expect_err("cancelled content is not a digest");
        assert!(neoethos_data::FeatureBuildCancelled::matches(
            &anyhow::Error::new(error)
        ));
        assert_eq!(*events.lock().expect("progress lock"), vec![0, 1]);
        assert!(canonical_feature_content_sha256_with_control(&frame, &control).is_err());
        assert_eq!(*events.lock().expect("progress lock"), vec![0, 1]);
    }

    #[test]
    fn batched_hash_preserves_the_exact_v1_digest_for_ram_and_vortex() {
        let ram = exact_hash_fixture();
        let expected = scalar_reference_hash(&ram);
        assert_eq!(
            expected,
            "223b4c46cc7daf4633fc39d05b1ea9bf545e469266e9889a30822c6beba6b384"
        );
        for batch_columns in [1, 2, ram.n_features(), usize::MAX] {
            assert_eq!(
                canonical_feature_content_sha256_with_batch_columns(&ram, batch_columns)
                    .expect("RAM batched digest"),
                expected
            );
            assert_eq!(
                canonical_feature_content_sha256_with_projection_schedule(&ram, batch_columns, 3,)
                    .expect("RAM parallel digest"),
                expected
            );
        }

        let temp = tempfile::tempdir().expect("temporary Vortex root");
        let vortex = vortex_copy(&ram, temp.path());
        for batch_columns in [1, 2, vortex.n_features(), usize::MAX] {
            assert_eq!(
                canonical_feature_content_sha256_with_batch_columns(&vortex, batch_columns)
                    .expect("Vortex batched digest"),
                expected
            );
            assert_eq!(
                canonical_feature_content_sha256_with_projection_schedule(
                    &vortex,
                    batch_columns,
                    3,
                )
                .expect("Vortex parallel digest"),
                expected
            );
        }
    }

    #[test]
    fn hash_chunks_preserve_exact_digest_across_both_one_mib_buffer_boundaries() {
        // The production timestamp buffer holds 131_072 i64s; its value/code
        // buffer holds 116_508 nine-byte cells. Cross both boundaries and keep
        // an incomplete final chunk, with more columns than one projection wave.
        let rows = 131_089;
        let timestamps = (0..rows)
            .map(|row| 1_704_067_200_000_i64 + row as i64 * 60_000)
            .collect::<Vec<_>>();
        let columns = (0..3)
            .map(|column| {
                let values = (0..rows)
                    .map(|row| match row % 4 {
                        0 => -0.0,
                        1 => f64::MIN_POSITIVE,
                        2 => -(row as f64 + column as f64 + 0.125),
                        _ => f64::from_bits(1.0_f64.to_bits() + row as u64 + column as u64),
                    })
                    .collect();
                let validity = (0..rows)
                    .map(|row| {
                        if row % 17 == 0 {
                            FeatureCellValidity::Warmup
                        } else {
                            FeatureCellValidity::Valid
                        }
                    })
                    .collect();
                FeatureColumnF64::new(format!("long_δοκιμή_{column}"), values, validity)
                    .expect("long exact-bit column")
            })
            .collect();
        let ram = ctrader_test_feature_frame_from_columns(timestamps, columns)
            .expect("long exact-hash fixture");
        let expected = scalar_reference_hash(&ram);
        let temp = tempfile::tempdir().expect("temporary Vortex root");
        let vortex = vortex_copy(&ram, temp.path());
        for (label, frame) in [("RAM", &ram), ("Vortex", &vortex)] {
            for (batch_columns, concurrent_batches) in [(1, 2), (2, 3)] {
                assert_eq!(
                    canonical_feature_content_sha256_with_projection_schedule(
                        frame,
                        batch_columns,
                        concurrent_batches,
                    )
                    .expect("chunked parallel content digest"),
                    expected,
                    "{label}: chunk framing or projection order changed at schedule \
                     {batch_columns}x{concurrent_batches}"
                );
            }
        }
    }

    #[test]
    fn known_content_digest_does_not_bypass_receipt_metadata_revalidation() {
        let frame = exact_hash_fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
            .expect("fresh receipt");
        let known_digest = receipt.feature_content_sha256().to_owned();
        receipt
            .validate_against_with_content_sha256(&anchor, &frame, &known_digest)
            .expect("the exact fresh metadata and digest must validate");

        let mutations: &[(&str, fn(&mut CanonicalSearchInputReceiptV2))] = &[
            ("schema", |r| r.schema_version += 1),
            ("anchor", |r| {
                r.anchor_dataset_identity = CanonicalDatasetIdentity::external(
                    "foreign-receipt-test",
                    "EURUSD",
                    CanonicalTimeframe::M1,
                    neoethos_data::BarTimestampConvention::BarOpen,
                )
                .expect("foreign test identity")
                .to_path_component();
            }),
            ("feature plan", |r| {
                r.feature_plan_identity = "00".repeat(32)
            }),
            ("feature provenance", |r| {
                r.feature_provenance_identity = "00".repeat(32);
            }),
            ("feature content", |r| {
                r.feature_content_sha256 = "00".repeat(32)
            }),
            ("execution schema", |r| {
                r.feature_execution.schema_version += 1
            }),
            ("execution policy", |r| {
                r.feature_execution.compute_policy = match r.feature_execution.compute_policy {
                    CanonicalFeatureComputePolicyV1::Auto => {
                        CanonicalFeatureComputePolicyV1::CpuOnly
                    }
                    _ => CanonicalFeatureComputePolicyV1::Auto,
                };
            }),
            ("math lane", |r| {
                r.feature_execution.selected_lane = match r.feature_execution.selected_lane {
                    CanonicalFeatureMathLaneV1::CpuScalar => CanonicalFeatureMathLaneV1::CpuAvx2Fma,
                    _ => CanonicalFeatureMathLaneV1::CpuScalar,
                };
            }),
            ("math authority", |r| {
                r.feature_execution
                    .vector_ta_math_authority
                    .push_str("-changed");
            }),
            ("source node", |r| {
                r.source_bindings[0].source_node_id.push_str("-changed")
            }),
            ("source count", |r| r.source_bindings.clear()),
            ("manifest schema", |r| {
                r.source_bindings[0].manifest_schema_id.push_str("-changed");
            }),
            ("manifest digest", |r| {
                r.source_bindings[0].manifest_sha256 = "00".repeat(32)
            }),
            ("generation", |r| {
                r.source_bindings[0].generation_id.push_str("-changed")
            }),
            ("Vortex digest", |r| {
                r.source_bindings[0].vortex_sha256 = "00".repeat(32)
            }),
            ("bar convention", |r| {
                r.source_bindings[0]
                    .bar_timestamp_convention
                    .push_str("-changed");
            }),
            ("row start", |r| {
                r.source_bindings[0].segments[0].row_start += 1
            }),
            ("row end", |r| r.source_bindings[0].segments[0].row_end -= 1),
            ("timestamp start", |r| {
                r.source_bindings[0].segments[0].timestamp_start_ms += 1;
            }),
            ("timestamp end", |r| {
                r.source_bindings[0].segments[0].timestamp_end_ms -= 1;
            }),
        ];
        for (label, mutate) in mutations {
            let mut changed = receipt.clone();
            mutate(&mut changed);
            assert!(
                changed
                    .validate_against_with_content_sha256(&anchor, &frame, &known_digest)
                    .is_err(),
                "a known content digest must not bypass {label} revalidation"
            );
        }
    }

    #[test]
    fn untrusted_receipt_still_rehashes_and_rejects_tampered_content() {
        let frame = exact_hash_fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let mut receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
            .expect("fresh receipt");
        receipt.feature_content_sha256 = "00".repeat(32);
        let error = receipt
            .validate_against(&anchor, &frame)
            .expect_err("tampered digest must fail closed");
        assert!(
            error
                .to_string()
                .contains("feature content SHA-256 does not match")
        );
    }

    #[test]
    fn legacy_raw_receipt_keeps_its_json_and_identity_without_optional_recipe_fields() {
        let frame = exact_hash_fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let mut legacy = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
            .expect("raw receipt");
        legacy.feature_plan_canonical_bytes = None;
        let bytes = legacy.to_json_bytes().expect("legacy raw JSON");
        let fields: serde_json::Value = serde_json::from_slice(&bytes).expect("receipt JSON");
        for absent in [
            "feature_plan_canonical_bytes",
            "normalization_fitted_state",
            "feature_build_options",
        ] {
            assert!(
                fields.get(absent).is_none(),
                "old raw field set changed: {absent}"
            );
        }
        let restored = CanonicalSearchInputReceiptV2::from_json_bytes(&bytes)
            .expect("old raw V2 remains readable");
        assert_eq!(restored.to_json_bytes().expect("unchanged JSON"), bytes);
        let mut expected = Sha256::new();
        expected.update(CANONICAL_SEARCH_INPUT_RECEIPT_HASH_DOMAIN_V2);
        expected.update(&bytes);
        assert_eq!(
            restored.identity_sha256().expect("legacy identity"),
            hex(&expected.finalize())
        );
        restored
            .validate_against(&anchor, &frame)
            .expect("exact raw frame still validates");
    }

    #[test]
    fn receipt_rejects_corrupted_plan_proof_and_unbound_producer_recipe() {
        let frame = exact_hash_fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
            .expect("raw receipt with plan proof");
        let mut corrupt = receipt.clone();
        corrupt
            .feature_plan_canonical_bytes
            .as_mut()
            .expect("plan proof")[0] ^= 1;
        assert!(
            corrupt.validate().is_err(),
            "canonical plan bytes must be decoded, not trusted"
        );

        let mut unbound = receipt.clone();
        unbound.feature_plan_canonical_bytes = None;
        unbound.feature_build_options = Some(FeatureBuildOptions::default());
        assert!(
            unbound.validate().is_err(),
            "recipe cannot bypass the plan proof"
        );

        let mut wrong_timeframes = receipt;
        wrong_timeframes.feature_build_options = Some(FeatureBuildOptions {
            higher_tfs: vec!["H1".to_owned()],
            ..FeatureBuildOptions::default()
        });
        assert!(
            wrong_timeframes.validate().is_err(),
            "recipe must match exact direct-source timeframes"
        );
    }

    #[test]
    fn receipt_checks_mutated_working_set_recipes_without_a_serde_roundtrip() {
        let frame = exact_hash_fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let mut receipt =
            CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame).unwrap();
        let mut batch = neoethos_data::search_working_set_batch_seeded(0, 2, true, 73);
        batch.replace_base_vocabulary = false;
        receipt.feature_build_options = Some(FeatureBuildOptions {
            classic_ta_working_set: Some(batch.clone()),
            ..FeatureBuildOptions::default()
        });
        assert!(
            receipt
                .validate()
                .unwrap_err()
                .to_string()
                .contains("working-set recipe")
        );
        batch.replace_base_vocabulary = true;
        batch.next_cursor = usize::MAX;
        receipt
            .feature_build_options
            .as_mut()
            .unwrap()
            .classic_ta_working_set = Some(batch);
        assert!(
            receipt
                .validate()
                .unwrap_err()
                .to_string()
                .contains("working-set recipe")
        );
    }

    #[test]
    fn normalized_receipt_roundtrip_preserves_full_fit_after_column_projection() {
        let full = normalized_hash_fixture();
        let frame = full
            .select_columns(&[2, 1])
            .expect("reordered feature subset");
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
            .expect("normalized projected receipt");
        let bytes = receipt.to_json_bytes().expect("normalized JSON");
        eprintln!(
            "normalized receipt fixture: {} columns, {} saved fit columns, {} JSON bytes, {} canonical plan bytes",
            frame.n_features(),
            full.n_features(),
            bytes.len(),
            frame.plan().canonical_bytes().len(),
        );
        let restored = CanonicalSearchInputReceiptV2::from_json_bytes(&bytes)
            .expect("exact-bit fitted-state roundtrip");
        assert_eq!(restored, receipt);
        assert_eq!(
            restored.normalization_fitted_state(),
            full.normalization_fitted_state()
        );
        assert_eq!(
            restored.feature_build_options(),
            full.feature_build_options()
        );
        assert_eq!(
            restored
                .normalization_fitted_state()
                .expect("saved fit")
                .column_names(),
            full.names
        );
        restored
            .validate_against(&anchor, &frame)
            .expect("full fit remains sealed to projected plan");
    }

    #[test]
    fn normalized_receipt_rejects_missing_fit_recipe_and_changed_fitted_bits() {
        let frame = normalized_hash_fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity().clone();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &frame)
            .expect("normalized receipt");
        let mut missing_fit = receipt.clone();
        missing_fit.normalization_fitted_state = None;
        assert!(
            missing_fit.validate().is_err(),
            "normalization cannot be mislabeled as raw"
        );

        let mut legacy_shape = missing_fit;
        legacy_shape.feature_plan_canonical_bytes = None;
        legacy_shape.feature_build_options = None;
        legacy_shape
            .validate()
            .expect("opaque legacy receipt remains decodable");
        assert!(
            legacy_shape.validate_against(&anchor, &frame).is_err(),
            "legacy shape cannot validate a normalized frame without its fit"
        );

        let mut missing_recipe = receipt.clone();
        missing_recipe.feature_build_options = None;
        assert!(
            missing_recipe.validate().is_err(),
            "normalized replay requires actual producer options"
        );

        let mut changed: serde_json::Value =
            serde_json::to_value(&receipt).expect("receipt fields");
        changed["normalization_fitted_state"]["fits"][1]["median"] =
            serde_json::Value::String("3ff0000000000000".to_owned());
        assert!(
            CanonicalSearchInputReceiptV2::from_json_bytes(
                &serde_json::to_vec(&changed).expect("mutated JSON")
            )
            .is_err(),
            "fitted value bits must match the sealed normalization node, not merely valid JSON"
        );

        let mut wrong_prefix = receipt;
        wrong_prefix
            .feature_build_options
            .as_mut()
            .expect("recipe")
            .prefix_base_features = true;
        assert!(
            wrong_prefix.validate_against(&anchor, &frame).is_err(),
            "same content hash cannot authorize a different producer recipe"
        );
    }

    #[test]
    fn raw_receipt_rejects_a_normalization_fit_without_its_transform() {
        let raw = exact_hash_fixture();
        let normalized = normalized_hash_fixture();
        let anchor = raw.provenance().bindings()[0].dataset_identity().clone();
        let mut receipt =
            CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &raw).expect("raw receipt");
        receipt.normalization_fitted_state = normalized.normalization_fitted_state().cloned();
        receipt.feature_build_options = normalized.feature_build_options().cloned();
        assert!(
            receipt.validate().is_err(),
            "a valid fit cannot be attached to an unrelated raw plan"
        );
    }

    fn publish_recorded_generation(
        root: &Path,
        identity: &CanonicalDatasetIdentity,
        expected_generation: Option<&str>,
        offset: f64,
    ) -> String {
        use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
        use neoethos_data::{
            CanonicalOhlcvPublishRequest, CanonicalVolumeRef, publish_canonical_ohlcv_generation,
        };
        let rows = 512;
        let period = identity.timeframe().fixed_duration_ms().unwrap();
        let end = 1_704_067_200_000_i64;
        let close = (0..rows)
            .map(|row| 1.12 + offset + (row as f64 * 0.19).sin() * 0.001 + row as f64 * 0.000_005)
            .collect::<Vec<_>>();
        let frame = Ohlcv {
            timestamp: Some(
                (0..rows)
                    .map(|row| end - (rows - 1 - row) as i64 * period)
                    .collect(),
            ),
            open: close.iter().map(|value| value - 0.000_02).collect(),
            high: close.iter().map(|value| value + 0.000_05).collect(),
            low: close.iter().map(|value| value - 0.000_05).collect(),
            close,
            volume: None,
        };
        let producer = ProducerProvenanceEnvelopeV1::new(
            "neoethos.recorded-input-reopen-test.v1",
            identity.canonical_bytes(),
        )
        .unwrap();
        publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
            configured_root: root,
            identity,
            expected_generation,
            provenance: &producer,
            ohlcv: &frame,
            volume: CanonicalVolumeRef::Absent,
            rows_per_chunk: 128,
        })
        .expect("publish independently authored direct timeframe")
        .generation()
        .to_owned()
    }

    #[test]
    fn recorded_receipt_reopens_exact_current_raw_and_frozen_fit_before_refusing_advanced_data() {
        use neoethos_data::BarTimestampConvention;
        use neoethos_data::core::normalization::normalize_search_feature_column_f64;

        // A fresh process proves replay under the ordinary Auto default without
        // changing the immutable global policy of this or any parallel test.
        const CHILD: &str = "NEOETHOS_TEST_RECORDED_POLICY_CHILD";
        let completion = std::env::var_os(CHILD).map(PathBuf::from);
        if completion.is_none() {
            let child_root = tempfile::tempdir().expect("owned replay child marker");
            let marker = child_root.path().join("completed");
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "data_selection::canonical_feature_content_hash_tests::recorded_receipt_reopens_exact_current_raw_and_frozen_fit_before_refusing_advanced_data",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, &marker)
                .stdin(std::process::Stdio::null())
                .spawn()
                .expect("isolated policy replay child");
            let started = Instant::now();
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        assert!(status.success(), "policy replay child failed: {status}");
                        assert_eq!(
                            std::fs::read_to_string(&marker).expect("the selected child test ran"),
                            "recorded-policy-replay-pass",
                        );
                        return;
                    }
                    Ok(None) if started.elapsed() < std::time::Duration::from_secs(240) => {
                        std::thread::sleep(std::time::Duration::from_millis(25));
                    }
                    state => {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("policy replay child did not finish within 240 seconds: {state:?}");
                    }
                }
            }
        }
        let process_authority = resolved_canonical_feature_execution_authority_v1();
        assert_eq!(process_authority.policy, IndicatorComputePolicy::Auto);
        let production_control = FeatureBuildControl::default()
            .with_indicator_compute_policy(IndicatorComputePolicy::CpuOnly);
        let root = tempfile::tempdir().expect("isolated exact-reopen store");
        let identities = [
            CanonicalTimeframe::M1,
            CanonicalTimeframe::M5,
            CanonicalTimeframe::H1,
        ]
        .map(|timeframe| {
            CanonicalDatasetIdentity::external(
                "recorded-input-reopen-test",
                "EURUSD",
                timeframe,
                BarTimestampConvention::BarOpen,
            )
            .unwrap()
        });
        let generations = identities
            .each_ref()
            .map(|identity| publish_recorded_generation(root.path(), identity, None, 0.0));
        let raw_options = FeatureBuildOptions {
            prefix_base_features: true,
            higher_tfs: vec!["M5".to_owned(), "H1".to_owned()],
            ..Default::default()
        };
        let exact = ExactCanonicalSeries::open(root.path(), identities[0].clone()).unwrap();
        let higher = [CanonicalTimeframe::M5, CanonicalTimeframe::H1];
        let mut raw = exact
            .load_search_input_with_builder(&higher, &production_control, |dataset, base_tf, _| {
                neoethos_data::prepare_multitimeframe_features_raw_with_options_and_control(
                    dataset,
                    base_tf,
                    &raw_options,
                    &production_control,
                )
            })
            .expect("actual raw direct-source producer");
        assert_eq!(raw.features().provenance().bindings().len(), 3);
        let training_rows = 0..384;
        let mut fit_names = Vec::new();
        let mut fits = Vec::new();
        for identity in &identities {
            let prefix = format!("{}_", identity.timeframe());
            let column = raw
                .features()
                .names
                .iter()
                .enumerate()
                .filter(|(_, name)| name.starts_with(&prefix))
                .find_map(|(index, _)| {
                    let batch = raw
                        .features()
                        .project_columns(&[index], 0..raw.features().n_samples())
                        .unwrap();
                    let column = &batch.columns[0];
                    (column.validity[training_rows.clone()]
                        .iter()
                        .filter(|validity| validity.is_valid())
                        .count()
                        >= 2)
                        .then(|| column.clone())
                })
                .expect("one train-supported actual feature per direct timeframe");
            let mut column = column;
            fit_names.push(column.name.clone());
            fits.push(
                normalize_search_feature_column_f64(&mut column, training_rows.clone())
                    .expect("fit only the declared training prefix"),
            );
        }
        let fitted = SearchNormalizationFittedStateV1::new(fit_names, fits).unwrap();
        let normalized_options = FeatureBuildOptions {
            normalization_training_rows: Some(training_rows),
            drop_columns_without_normalization_training_support: true,
            ..raw_options.clone()
        };
        let mut normalized = exact
            .load_search_input_with_builder(&higher, &production_control, |dataset, base_tf, _| {
                neoethos_data::prepare_multitimeframe_features_with_fitted_normalization_and_control(
                    dataset,
                    base_tf,
                    &normalized_options,
                    &fitted,
                    &production_control,
                )
            })
            .expect("actual normalized producer using the frozen per-timeframe fits");
        assert_eq!(normalized.features().n_features(), 3);
        // These are output projections, not fabricated replacement plans. The
        // normalized frame retains the complete three-column fitted schema.
        raw.features = raw
            .features
            .select_columns(&[raw.features.n_features() - 1, 0])
            .unwrap();
        normalized.features = normalized.features.select_columns(&[2, 0]).unwrap();
        let inputs = [raw, normalized];
        let options = [raw_options, normalized_options];
        let mut receipts = Vec::new();
        for (original, options) in inputs.iter().zip(&options) {
            let receipt = original.receipt().unwrap();
            let receipt =
                CanonicalSearchInputReceiptV2::from_json_bytes(&receipt.to_json_bytes().unwrap())
                    .expect("persist and decode the entire receipt, recipe and exact fit bits");
            let reopened =
                CanonicalSearchInput::from_recorded_receipt(root.path(), receipt.clone(), options)
                    .expect("reopen the exact still-current direct generations");
            assert_eq!(
                receipt.feature_execution().compute_policy(),
                CanonicalFeatureComputePolicyV1::CpuOnly
            );
            assert_eq!(reopened.receipt().unwrap(), receipt);
            // The owned input retains actual production authority after its
            // local replay control is gone, without suppressing a content hash.
            assert_eq!(reopened.as_run_input().unwrap().receipt(), &receipt);
            assert_eq!(
                reopened
                    .clone()
                    .as_run_input_with_control(&FeatureBuildControl::default())
                    .unwrap()
                    .receipt(),
                &receipt,
            );
            assert_eq!(
                resolved_canonical_feature_execution_authority_v1(),
                process_authority
            );
            assert!(
                CanonicalSearchRunInputV2::new_with_control(
                    receipt.clone(),
                    reopened.features(),
                    reopened.base_frame(),
                    &production_control,
                )
                .is_err(),
                "an override alone cannot attest an unrelated caller-supplied frame"
            );
            let actual_base = reopened.base_frame().ohlcv();
            let expected_base = original.base_frame().ohlcv();
            assert_eq!(actual_base.timestamp, expected_base.timestamp);
            assert_eq!(actual_base.volume, expected_base.volume);
            for (actual, expected) in [
                (&actual_base.open, &expected_base.open),
                (&actual_base.high, &expected_base.high),
                (&actual_base.low, &expected_base.low),
                (&actual_base.close, &expected_base.close),
            ] {
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            assert_eq!(reopened.features().names, original.features().names);
            assert_eq!(
                reopened.features().normalization_fitted_state(),
                original.features().normalization_fitted_state()
            );
            let expected = original
                .features()
                .project_columns(&[0, 1], 0..512)
                .unwrap();
            let actual = reopened
                .features()
                .project_columns(&[0, 1], 0..512)
                .unwrap();
            assert_eq!(actual.row_ids, expected.row_ids);
            for (actual, expected) in actual.columns.iter().zip(&expected.columns) {
                assert_eq!(actual.validity, expected.validity);
                assert!(
                    actual
                        .values
                        .iter()
                        .zip(&expected.values)
                        .all(|(a, b)| { a.to_bits() == b.to_bits() })
                );
            }
            receipts.push(receipt);
        }
        // Advancing even one direct higher-TF source must fail at dataset pinning,
        // before a feature builder can consume CURRENT or refit against it.
        let newer =
            publish_recorded_generation(root.path(), &identities[1], Some(&generations[1]), 0.75);
        assert_ne!(newer, generations[1]);
        for (receipt, options) in receipts.into_iter().zip(&options) {
            let error = CanonicalSearchInput::from_recorded_receipt(root.path(), receipt, options)
                .expect_err("a stale selected generation cannot fall forward to CURRENT");
            assert!(matches!(
                error,
                CanonicalDataSelectionError::DatasetOpenFailed { .. }
            ));
            let detail = error.to_string();
            assert!(detail.contains(&generations[1]), "{detail}");
            assert!(detail.contains(&newer), "{detail}");
        }
        std::fs::write(
            completion.expect("isolated child completion path"),
            b"recorded-policy-replay-pass",
        )
        .expect("record actual child completion, never a zero-test exit");
    }

    #[test]
    fn recorded_policy_preflight_refuses_wrong_lane_gpu_and_explicit_conflict_before_production() {
        let (root, input) = canonical_input_fixture();
        let receipt = input.receipt().unwrap();
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&events);
        let control = FeatureBuildControl::default().with_observer(move |event| {
            observed.lock().unwrap().push(event);
        });
        let mut wrong_lane = receipt.clone();
        wrong_lane.feature_execution.selected_lane =
            match wrong_lane.feature_execution.selected_lane {
                CanonicalFeatureMathLaneV1::CpuScalar => CanonicalFeatureMathLaneV1::CpuAvx2Fma,
                _ => CanonicalFeatureMathLaneV1::CpuScalar,
            };
        let mut gpu = receipt.clone();
        gpu.feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            canonical_feature_execution_authority_for_policy_v1(IndicatorComputePolicy::GpuOnly),
        );
        let conflicting_policy = match receipt.feature_execution.compute_policy() {
            CanonicalFeatureComputePolicyV1::CpuOnly => IndicatorComputePolicy::Auto,
            _ => IndicatorComputePolicy::CpuOnly,
        };
        // An absent root distinguishes early policy refusal from any attempted
        // source read, and no Data observer event may have been emitted.
        let absent = root.path().join("must-not-open");
        for (changed, request, message) in [
            (wrong_lane, control.clone(), "math lane differs"),
            (gpu, control.clone(), "GpuOnly execution cannot be replayed"),
            (
                receipt,
                control
                    .clone()
                    .with_indicator_compute_policy(conflicting_policy),
                "explicit replay policy conflicts",
            ),
        ] {
            let error = CanonicalSearchInput::from_recorded_receipt_with_control(
                &absent,
                changed,
                &FeatureBuildOptions::default(),
                &request,
            )
            .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
            assert!(
                events.lock().unwrap().is_empty(),
                "feature producer must not run"
            );
            assert!(
                !absent.exists(),
                "policy preflight must not create source state"
            );
        }
    }

    #[test]
    fn a_recorded_policy_cannot_relabel_an_unrelated_already_built_input() {
        let (_root, input) = canonical_input_fixture();
        let mut receipt = input.receipt().unwrap();
        let other = match receipt.feature_execution.compute_policy() {
            CanonicalFeatureComputePolicyV1::CpuOnly => IndicatorComputePolicy::Auto,
            _ => IndicatorComputePolicy::CpuOnly,
        };
        receipt.feature_execution = CanonicalFeatureExecutionReceiptV1::from_runtime_authority(
            canonical_feature_execution_authority_for_policy_v1(other),
        );
        let request = FeatureBuildControl::default().with_indicator_compute_policy(other);
        assert!(
            receipt
                .validate_against(input.anchor_identity(), input.features())
                .is_err()
        );
        assert!(
            CanonicalSearchRunInputV2::new_with_control(
                receipt.clone(),
                input.features(),
                input.base_frame(),
                &request,
            )
            .is_err()
        );
        let error = input
            .bind_recorded_receipt_with_control(receipt, &request)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("recorded feature execution differs"),
            "{error}"
        );
    }

    #[test]
    fn recorded_receipt_rejects_conflicting_generations_for_one_direct_timeframe() {
        let (root, input) = canonical_input_fixture();
        let mut receipt = input.receipt().unwrap();
        let mut conflicting = receipt.source_bindings[0].clone();
        conflicting.source_node_id.push_str(":conflicting");
        conflicting.generation_id = format!("g1-{}.vortex", "a".repeat(64));
        receipt.source_bindings.push(conflicting);
        receipt
            .validate()
            .expect("structurally valid multi-source receipt");
        let error = CanonicalSearchInput::from_recorded_receipt(
            root.path(),
            receipt,
            &FeatureBuildOptions::default(),
        )
        .expect_err("one direct timeframe cannot silently select one of two generations");
        assert!(
            error.to_string().contains("conflicting exact generations"),
            "{error}"
        );
    }

    #[test]
    fn recorded_receipt_refuses_unrecorded_source_cutoffs_before_feature_computation() {
        let (root, input) = canonical_input_fixture();
        let mut receipt = input.receipt().unwrap();
        let segment = &mut receipt.source_bindings[0].segments[0];
        segment.row_end -= 1;
        segment.timestamp_end_ms = input.features.timestamps[input.features.n_samples() - 2];
        receipt
            .validate()
            .expect("structurally valid source-prefix receipt");
        let error = CanonicalSearchInput::from_recorded_receipt(
            root.path(),
            receipt,
            &FeatureBuildOptions::default(),
        )
        .expect_err("an unrecorded cutoff must not be reconstructed as a full generation");
        assert!(
            error
                .to_string()
                .contains("original source cutoff is not persisted"),
            "{error}"
        );
    }

    #[test]
    fn binding_recorded_receipt_preserves_legacy_identity_and_rechecks_actual_values() {
        let (_root, input) = canonical_input_fixture();
        let mut legacy = input.receipt().expect("prepared receipt");
        legacy.feature_plan_canonical_bytes = None;
        let legacy_hash = legacy.identity_sha256().expect("legacy hash");
        let bound = input
            .clone()
            .bind_recorded_receipt(legacy.clone())
            .expect("bind original receipt without upgrading its JSON identity");
        assert_eq!(bound.receipt().unwrap(), legacy);
        assert_eq!(
            bound.receipt().unwrap().identity_sha256().unwrap(),
            legacy_hash
        );
        assert_eq!(bound.as_run_input().unwrap().receipt(), &legacy);

        let mut changed = input;
        let mut columns = changed
            .features
            .project_columns(&[0, 1], 0..changed.features.n_samples())
            .unwrap()
            .columns
            .clone();
        columns[0].values[0] = f64::from_bits(columns[0].values[0].to_bits() + 1);
        changed.features = FeatureFrame::from_columns(
            changed.features.timestamps.clone(),
            columns,
            changed.features.plan().clone(),
            changed.features.provenance().clone(),
        )
        .unwrap();
        assert!(
            changed.bind_recorded_receipt(legacy).is_err(),
            "binding a recorded receipt must rehash values, not relabel a changed frame"
        );
    }

    #[test]
    fn controlled_recorded_binding_cancels_inside_the_final_content_digest() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (root, input) = canonical_input_fixture();
        let receipt = input.receipt().unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let observed_cancel = Arc::clone(&cancelled);
        let control =
            FeatureBuildControl::new(Arc::clone(&cancelled)).with_observer(move |event| {
                if event.stage == "feature_receipt" && event.completed == 1 {
                    observed_cancel.store(true, Ordering::Release);
                }
            });
        let error = input
            .bind_recorded_receipt_with_control(receipt.clone(), &control)
            .unwrap_err();
        assert!(neoethos_data::FeatureBuildCancelled::matches(
            &anyhow::Error::new(error)
        ));
        let error = CanonicalSearchInput::from_recorded_receipt_with_control(
            root.path(),
            receipt,
            &FeatureBuildOptions::default(),
            &control,
        )
        .unwrap_err();
        assert!(neoethos_data::FeatureBuildCancelled::matches(
            &anyhow::Error::new(error)
        ));
    }

    #[test]
    fn binding_recorded_receipt_restores_only_the_saved_exact_column_projection() {
        let (_root, input) = canonical_input_fixture();
        let selected = input.features().select_columns(&[1]).unwrap();
        let receipt =
            CanonicalSearchInputReceiptV2::from_feature_frame(input.anchor_identity(), &selected)
                .unwrap();
        let bound = input
            .clone()
            .bind_recorded_receipt(receipt.clone())
            .expect("restore the exact saved feature subset before content validation");
        assert_eq!(bound.features().names, vec!["open"]);
        assert_eq!(bound.receipt().unwrap(), receipt);
        assert_eq!(bound.features().plan_identity(), selected.plan_identity());
        bound
            .as_run_input()
            .expect("projected canonical base still binds");

        let mut legacy = receipt;
        legacy.feature_plan_canonical_bytes = None;
        assert!(
            input.bind_recorded_receipt(legacy).is_err(),
            "without saved plan bytes a different legacy projection cannot be guessed"
        );
    }

    #[test]
    fn live_semantic_plan_allows_new_rows_and_recorded_output_projection_not_new_math() {
        let full = normalized_hash_fixture();
        let projected = full.select_columns(&[2, 1]).unwrap();
        let anchor = full.provenance().bindings()[0].dataset_identity().clone();
        let receipt =
            CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &projected).unwrap();
        let live_tail = full.row_window(6, 10).unwrap();
        receipt
            .validate_live_feature_plan(&live_tail)
            .expect("row count and final-output projection do not change the frozen math");
        let raw = exact_hash_fixture();
        assert!(
            receipt.validate_live_feature_plan(&raw).is_err(),
            "matching names with raw values cannot satisfy the frozen normalized plan"
        );
    }

    #[test]
    fn live_semantic_plan_rejects_changed_formulas_even_when_names_and_values_match() {
        use neoethos_feature_contracts::{
            DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOutputV1,
        };
        let raw = exact_hash_fixture();
        let binding = raw.provenance().bindings()[0].clone();
        let anchor = binding.dataset_identity().clone();
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &raw).unwrap();
        let changed_source = FeatureNodeV1::source(
            binding.source_node_id(),
            anchor,
            "neoethos.test-fixture-derived-features.f64.v1",
            1,
            raw.names
                .iter()
                .map(|name| FeatureOutputV1::f64(name, 1))
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            [7; 32],
        )
        .unwrap();
        let plan = FeaturePlanV1::new(vec![changed_source], raw.names.clone()).unwrap();
        let provenance = DatasetFeatureArtifactProvenanceV1::new(&plan, vec![binding]).unwrap();
        let columns = raw
            .project_columns(
                &(0..raw.n_features()).collect::<Vec<_>>(),
                0..raw.n_samples(),
            )
            .unwrap()
            .columns
            .clone();
        let changed =
            FeatureFrame::from_columns(raw.timestamps.clone(), columns, plan, provenance).unwrap();
        assert_eq!(changed.names, raw.names);
        assert_eq!(
            canonical_feature_content_sha256(&changed).unwrap(),
            canonical_feature_content_sha256(&raw).unwrap()
        );
        assert!(
            receipt.validate_live_feature_plan(&changed).is_err(),
            "equal names and payloads are insufficient when the formula/source semantic hash changed"
        );

        let mut legacy = receipt;
        legacy.feature_plan_canonical_bytes = None;
        legacy
            .validate_live_feature_plan(&raw.row_window(6, 10).unwrap())
            .unwrap();
        assert!(
            legacy.validate_live_feature_plan(&changed).is_err(),
            "legacy raw compatibility cannot relax semantic identity"
        );
    }
}
