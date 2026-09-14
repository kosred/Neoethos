//! The model producer's numeric input contract, persisted once per symbol/TF.
//! Search and model fits are deliberately independent: models fit on their
//! purged training prefix, before dense/L1 projections. Inference only replays
//! this fitted transformation; it never estimates statistics from live rows.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use neoethos_data::{CanonicalDatasetIdentity, FeatureFrame, SymbolDataset};
use neoethos_search::data_selection::CanonicalSearchInputReceiptV2;
use serde::{Deserialize, Serialize};

use crate::promotion_candidate_training_v1::PromotionCandidateTrainingHandoffV1;

pub const MODEL_FEATURE_INPUT_FILE_V1: &str = "model_feature_input.v1.json";
const MODEL_FEATURE_INPUT_MAX_BYTES_V1: usize = 8 * 1024 * 1024;

fn read_bounded_model_json(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("read required model metadata {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= MODEL_FEATURE_INPUT_MAX_BYTES_V1 as u64,
        "model metadata must be a regular file no larger than 8 MiB"
    );
    let mut bytes = Vec::new();
    file.take(MODEL_FEATURE_INPUT_MAX_BYTES_V1 as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MODEL_FEATURE_INPUT_MAX_BYTES_V1,
        "model metadata exceeds 8 MiB"
    );
    Ok(bytes)
}

struct BoundedJson(Vec<u8>);
impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.0.len().saturating_add(bytes.len()) > MODEL_FEATURE_INPUT_MAX_BYTES_V1 {
            return Err(std::io::Error::other("model feature input exceeds 8 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PurgedTrainingPrefixV1 {
    cutoff_ms: i64,
    purge_bars: usize,
    in_sample_rows: usize,
    source_row_start: u64,
    source_row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFeatureInputV1 {
    schema_version: u16,
    /// This receipt is captured from MODEL features, not the Search frame.
    producer_receipt: CanonicalSearchInputReceiptV2,
    feature_columns: Vec<String>,
    purged_training_prefix: Option<PurgedTrainingPrefixV1>,
}

impl ModelFeatureInputV1 {
    /// Capture the original complete producer frame, before row/column views.
    /// Reading one bounded column batch proves actual row identities without
    /// materializing another dense feature matrix or saving every row ID.
    pub fn from_training_frame(
        anchor: &CanonicalDatasetIdentity,
        frame: &FeatureFrame,
        cutoff_ms: Option<i64>,
        purge_bars: usize,
    ) -> Result<Self> {
        let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(anchor, frame)?;
        let binding = receipt
            .source_bindings()
            .iter()
            .find(|binding| binding.dataset_identity() == anchor.to_path_component())
            .context("model producer has no anchor binding")?;
        let segments = binding.segments();
        let first = segments.first().context("model anchor has no segments")?;
        let last = segments.last().context("model anchor has no segments")?;
        ensure!(
            segments
                .windows(2)
                .all(|pair| pair[0].row_end() == pair[1].row_start()),
            "model input capture requires contiguous original source segments"
        );
        ensure!(
            last.row_end().checked_sub(first.row_start()) == Some(frame.n_samples() as u64)
                && frame.timestamps.first() == Some(&first.timestamp_start_ms())
                && frame.timestamps.last() == Some(&last.timestamp_end_ms()),
            "model input must be captured before original source rows are projected"
        );
        for start in (0..frame.n_samples()).step_by(4096) {
            let end = start.saturating_add(4096).min(frame.n_samples());
            let batch = frame.project_columns(&[0], start..end)?;
            ensure!(
                batch
                    .row_ids
                    .iter()
                    .copied()
                    .eq((start as u64)..(end as u64)),
                "model input contains shifted or selected source row identities"
            );
        }
        let prefix = cutoff_ms
            .map(|cutoff_ms| -> Result<_> {
                ensure!(purge_bars > 0, "model input purge must be positive");
                let in_sample_rows = frame
                    .timestamps
                    .partition_point(|timestamp| *timestamp < cutoff_ms);
                let keep = in_sample_rows
                    .checked_sub(purge_bars)
                    .filter(|keep| *keep > 0)
                    .context("model input has no purged training rows")?;
                if let Some(fit) = frame.normalization_fitted_state() {
                    ensure!(
                        fit.training_rows()? == (0..keep),
                        "model normalization was not fitted on the exact purged training prefix"
                    );
                }
                Ok(PurgedTrainingPrefixV1 {
                    cutoff_ms,
                    purge_bars,
                    in_sample_rows,
                    source_row_start: first.row_start(),
                    source_row_end: first
                        .row_start()
                        .checked_add(keep as u64)
                        .context("model fitted source row overflow")?,
                    timestamp_start_ms: frame.timestamps[0],
                    timestamp_end_ms: frame.timestamps[keep - 1],
                })
            })
            .transpose()?;
        let contract = Self {
            schema_version: 1,
            producer_receipt: receipt,
            feature_columns: frame.names.clone(),
            purged_training_prefix: prefix,
        };
        contract.validate()?;
        Ok(contract)
    }

    pub fn producer_receipt(&self) -> &CanonicalSearchInputReceiptV2 {
        &self.producer_receipt
    }

    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    pub fn base_feature_name(&self, unprefixed: &str) -> Result<String> {
        let anchor = self.producer_receipt.validate()?;
        let options = self
            .producer_receipt
            .feature_build_options()
            .context("model input recipe missing")?;
        let name = if options.prefix_base_features {
            format!("{}_{}", anchor.timeframe().as_str(), unprefixed)
        } else {
            unprefixed.to_owned()
        };
        ensure!(
            self.feature_columns.contains(&name),
            "model base feature `{name}` is absent"
        );
        Ok(name)
    }

    pub fn read_from_path(path: &Path) -> Result<Self> {
        let bytes = read_bounded_model_json(path)?;
        Self::from_json_bytes(&bytes)
    }

    /// Verify this expert's actual persisted training projection, not merely
    /// matching names. A partial retrain may leave an older expert beside a
    /// newer shared contract; its sidecar must reject that fit mismatch.
    pub(crate) fn validate_expert_runtime_artifact(
        &self,
        artifact_dir: &Path,
        expert_name: &str,
        columns: &[String],
    ) -> Result<()> {
        use super::profile::{
            MODEL_RUNTIME_ARTIFACT_FILE_NAME, TrainingRuntimeProfile,
            validate_training_runtime_profile,
        };
        use super::training_artifact::{
            model_feature_availability_hash, model_feature_schema_hash,
        };
        self.validate()?;
        let anchor = self.producer_receipt.validate()?;
        let options = self
            .producer_receipt
            .feature_build_options()
            .context("model input producer recipe missing")?;
        let path = artifact_dir.join(MODEL_RUNTIME_ARTIFACT_FILE_NAME);
        let artifact: neoethos_core::ModelRuntimeArtifact<TrainingRuntimeProfile> =
            serde_json::from_slice(&read_bounded_model_json(&path)?)
                .with_context(|| format!("parse model input sidecar {}", path.display()))?;
        let artifact =
            neoethos_core::ModelRuntimeArtifact::new(artifact.provenance, artifact.payload)?;
        let profile = &artifact.payload;
        validate_training_runtime_profile(profile)?;
        ensure!(
            profile.model_name == expert_name
                && profile.symbol == anchor.symbol_name()
                && profile.base_timeframe == anchor.timeframe().as_str()
                && profile.feature_count == columns.len()
                && profile.base_features_prefixed == options.prefix_base_features
                && profile.higher_timeframes == options.higher_tfs,
            "expert '{expert_name}' sidecar identity/schema/recipe differs from its selected model input"
        );
        ensure!(
            artifact.provenance.feature_schema_hash == model_feature_schema_hash(columns)?,
            "expert '{expert_name}' persisted feature order differs from its loader metadata"
        );
        // Match precisely the registry's `<canonical>_<digits>` replica rule.
        // An arbitrary name beginning with `swarm` cannot bypass its fit.
        let raw_price = expert_name == "swarm_forecaster"
            || expert_name
                .strip_prefix("swarm_forecaster_")
                .is_some_and(|suffix| {
                    !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
                });
        let plan = self.expert_training_plan(columns, raw_price)?;
        let fit_hash = if raw_price {
            None
        } else {
            self.producer_receipt
                .normalization_fitted_state()
                .map(|fit| fit.fitted_state_hash())
                .transpose()?
        };
        let expected = model_feature_availability_hash(
            profile,
            columns,
            &plan.identity().to_hex(),
            fit_hash.as_ref(),
            Some(options),
        )?;
        ensure!(
            artifact.provenance.feature_availability_policy_hash == expected,
            "expert '{expert_name}' persisted plan/fit/recipe differs from the active model input; partial retraining cannot reinterpret an older expert"
        );
        Ok(())
    }

    fn expert_training_plan(
        &self,
        columns: &[String],
        raw_price: bool,
    ) -> Result<neoethos_feature_contracts::FeaturePlanV1> {
        use neoethos_feature_contracts::{FeatureOperationTagV1, FeaturePlanV1};
        ensure!(
            !columns.is_empty()
                && columns
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    == columns.len()
                && columns
                    .iter()
                    .all(|name| self.feature_columns.contains(name)),
            "expert projection must contain unique columns from its model input"
        );
        let recorded = self
            .producer_receipt
            .recorded_feature_plan()?
            .context("model input recorded plan missing")?;
        let nodes = if raw_price {
            ensure!(
                columns == [self.base_feature_name("quant_close")?],
                "Swarm input must be exactly its recorded raw base-price column"
            );
            if self.producer_receipt.normalization_fitted_state().is_some() {
                // Undo only the lossless GRAPH namespace wrapper. No values
                // are inverted or recovered from clipped normalized cells.
                let mut removed = 0;
                let mut restored = 0;
                let mut nodes = Vec::new();
                for node in recorded.nodes() {
                    if node.operation() == FeatureOperationTagV1::Normalization {
                        ensure!(
                            node.id() == "normalization:robust-f64",
                            "unexpected model normalization node in raw-price contract"
                        );
                        removed += 1;
                    } else {
                        let names = node
                            .outputs()
                            .iter()
                            .map(|output| {
                                if let Some(raw) = output.name().strip_prefix("model-input:raw:") {
                                    restored += 1;
                                    raw.to_owned()
                                } else {
                                    output.name().to_owned()
                                }
                            })
                            .collect();
                        nodes.push(node.with_output_names(names)?);
                    }
                }
                ensure!(
                    removed == 1 && restored > 0,
                    "raw-price input requires the preserved model raw producer graph"
                );
                nodes
            } else {
                recorded.nodes().to_vec()
            }
        } else {
            recorded.nodes().to_vec()
        };
        Ok(FeaturePlanV1::new(nodes, columns.to_vec())?)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported model feature input schema"
        );
        let anchor = self.producer_receipt.validate()?;
        ensure!(
            self.producer_receipt.feature_build_options().is_some(),
            "model feature input is missing its exact producer recipe"
        );
        let recorded = self
            .producer_receipt
            .recorded_feature_plan()?
            .context("model feature input has no sealed producer plan")?;
        ensure!(
            recorded.final_outputs() == self.feature_columns,
            "model input ordered schema differs from its sealed producer plan"
        );
        ensure!(
            !self.feature_columns.is_empty()
                && self
                    .feature_columns
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    == self.feature_columns.len(),
            "model feature input columns are empty or duplicated"
        );
        if let Some(fit) = self.producer_receipt.normalization_fitted_state() {
            ensure!(
                fit.column_names() == self.feature_columns,
                "model input schema must retain the complete original fitted columns"
            );
        }
        if let Some(prefix) = &self.purged_training_prefix {
            let binding = self
                .producer_receipt
                .source_bindings()
                .iter()
                .find(|binding| binding.dataset_identity() == anchor.to_path_component())
                .context("model input has no anchor binding")?;
            let segments = binding.segments();
            let first = segments
                .first()
                .context("model input has no source segments")?;
            let last = segments
                .last()
                .context("model input has no source segments")?;
            ensure!(
                segments
                    .windows(2)
                    .all(|pair| pair[0].row_end() == pair[1].row_start()),
                "model input purged source mapping must be contiguous"
            );
            let keep = prefix
                .in_sample_rows
                .checked_sub(prefix.purge_bars)
                .filter(|keep| *keep > 0)
                .context("invalid model purged prefix row count")?;
            ensure!(
                prefix.purge_bars > 0
                    && prefix.source_row_start == first.row_start()
                    && prefix.source_row_start.checked_add(keep as u64)
                        == Some(prefix.source_row_end)
                    && prefix
                        .source_row_start
                        .checked_add(prefix.in_sample_rows as u64)
                        .is_some_and(|end| end <= last.row_end())
                    && prefix.timestamp_start_ms == first.timestamp_start_ms()
                    && prefix.timestamp_start_ms <= prefix.timestamp_end_ms
                    && prefix.timestamp_end_ms < prefix.cutoff_ms,
                "model fitted source rows/timestamps do not identify a purged training prefix"
            );
            if let Some(fit) = self.producer_receipt.normalization_fitted_state() {
                ensure!(
                    fit.training_rows()? == (0..keep),
                    "model fit rows differ from the recorded purged source prefix"
                );
            }
        }
        Ok(())
    }

    pub fn validate_for_handoff(
        &self,
        handoff: &PromotionCandidateTrainingHandoffV1,
    ) -> Result<()> {
        self.validate()?;
        let expected = handoff.search_input_receipt();
        ensure!(
            self.producer_receipt.anchor_dataset_identity() == expected.anchor_dataset_identity()
                && self.producer_receipt.source_bindings() == expected.source_bindings(),
            "model preprocessing source generations differ from the candidate handoff"
        );
        self.validate_candidate_boundary(handoff.oos_cutoff_ms(), handoff.purge_bars())
    }

    fn validate_candidate_boundary(&self, cutoff_ms: i64, purge_bars: usize) -> Result<()> {
        let prefix = self
            .purged_training_prefix
            .as_ref()
            .context("candidate model preprocessing has no purged training boundary")?;
        ensure!(
            prefix.cutoff_ms == cutoff_ms && prefix.purge_bars == purge_bars,
            "model preprocessing cutoff/purge differs from the candidate handoff"
        );
        Ok(())
    }

    /// Check numeric semantics, not historical generation IDs, on fresh data.
    pub fn validate_features(&self, frame: &FeatureFrame) -> Result<()> {
        self.producer_receipt.validate_live_feature_plan(frame)?;
        ensure!(
            frame.names == self.feature_columns,
            "model inference requires its complete ordered producer schema"
        );
        Ok(())
    }

    pub fn prepare_features(&self, dataset: &SymbolDataset) -> Result<FeatureFrame> {
        self.prepare_features_with_control(dataset, &neoethos_data::FeatureBuildControl::default())
    }

    pub fn prepare_features_with_control(
        &self,
        dataset: &SymbolDataset,
        control: &neoethos_data::FeatureBuildControl,
    ) -> Result<FeatureFrame> {
        control.checkpoint()?;
        let anchor = self.producer_receipt.validate()?;
        let options = self
            .producer_receipt
            .feature_build_options()
            .context("model preprocessing recipe missing")?;
        let raw = Arc::new(
            neoethos_data::prepare_multitimeframe_features_raw_with_options_and_control(
                dataset,
                anchor.timeframe().as_str(),
                options,
                control,
            )?,
        );
        control.checkpoint()?;
        let frame = self.apply_to_raw_features(raw)?;
        control.checkpoint()?;
        Ok(frame)
    }

    pub(crate) fn apply_to_raw_features(&self, raw: Arc<FeatureFrame>) -> Result<FeatureFrame> {
        let frame = match self.producer_receipt.normalization_fitted_state() {
            Some(fit) => raw.with_fitted_normalization(fit)?,
            None => raw.shared_view()?,
        };
        self.validate_features(&frame)?;
        Ok(frame)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut output = BoundedJson(Vec::new());
        serde_json::to_writer(&mut output, self)?;
        Ok(output.0)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MODEL_FEATURE_INPUT_MAX_BYTES_V1,
            "model feature input exceeds 8 MiB"
        );
        let contract: Self = serde_json::from_slice(bytes)?;
        contract.validate()?;
        Ok(contract)
    }
}

fn contract_path(root: &Path, anchor: &CanonicalDatasetIdentity) -> PathBuf {
    root.join(anchor.symbol_name())
        .join(anchor.timeframe().as_str())
        .join(MODEL_FEATURE_INPUT_FILE_V1)
}

pub fn write_model_feature_input_v1(root: &Path, contract: &ModelFeatureInputV1) -> Result<()> {
    let anchor = contract.producer_receipt.validate()?;
    let path = contract_path(root, &anchor);
    neoethos_core::storage::json::write_bytes_atomic(&path, &contract.to_json_bytes()?)
        .with_context(|| format!("persist model-owned feature input {}", path.display()))
}

pub fn load_model_feature_input_for_handoff_v1(
    root: &Path,
    handoff: &PromotionCandidateTrainingHandoffV1,
) -> Result<ModelFeatureInputV1> {
    let path = contract_path(root, handoff.canonical_series().anchor().identity());
    let contract = ModelFeatureInputV1::read_from_path(&path)?;
    contract.validate_for_handoff(handoff)?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

    fn fixture() -> FeatureFrame {
        let raw = Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
                neoethos_data::test_fixtures::canonical_test_timestamps(12),
                vec![
                    FeatureColumnF64::new(
                        "f1",
                        (0..12).map(|row| row as f64).collect(),
                        vec![FeatureCellValidity::Valid; 12],
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        );
        let fit = raw
            .fit_normalization(0..6, true, &neoethos_data::FeatureBuildControl::default())
            .unwrap();
        raw.with_fitted_normalization(&fit).unwrap()
    }

    #[test]
    fn model_input_roundtrip_preserves_own_fit_and_accepts_short_live_window() {
        let frame = fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let contract =
            ModelFeatureInputV1::from_training_frame(anchor, &frame, Some(frame.timestamps[8]), 2)
                .unwrap();
        let reopened =
            ModelFeatureInputV1::from_json_bytes(&contract.to_json_bytes().unwrap()).unwrap();
        assert_eq!(reopened, contract);
        reopened
            .validate_features(&frame.row_window(10, 12).unwrap())
            .unwrap();
        let mut different_fit = frame.normalization_fitted_state().unwrap().clone();
        let mut encoded = serde_json::to_value(&different_fit).unwrap();
        encoded["fits"][0]["median"] = serde_json::json!("0000000000000000");
        different_fit = serde_json::from_value(encoded).unwrap();
        assert_ne!(
            Some(&different_fit),
            contract.producer_receipt().normalization_fitted_state()
        );
    }

    #[test]
    fn model_input_preparation_observes_cancellation_before_building_any_cube() {
        let frame = fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let contract =
            ModelFeatureInputV1::from_training_frame(anchor, &frame, Some(frame.timestamps[8]), 2)
                .unwrap();
        let dataset = SymbolDataset {
            symbol: "EURUSD".into(),
            frames: Default::default(),
            source_artifacts: Default::default(),
        };
        let control = neoethos_data::FeatureBuildControl::new(Arc::new(
            std::sync::atomic::AtomicBool::new(true),
        ));
        let error = contract
            .prepare_features_with_control(&dataset, &control)
            .unwrap_err();
        assert!(neoethos_data::FeatureBuildCancelled::matches(&error));
    }

    #[test]
    fn capture_refuses_projected_source_rows_and_oos_fitted_statistics() {
        let frame = fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        assert!(
            ModelFeatureInputV1::from_training_frame(
                anchor,
                &frame.row_window(2, 12).unwrap(),
                Some(frame.timestamps[8]),
                2
            )
            .unwrap_err()
            .to_string()
            .contains("before original")
        );
        assert!(
            ModelFeatureInputV1::from_training_frame(anchor, &frame, Some(frame.timestamps[8]), 1)
                .unwrap_err()
                .to_string()
                .contains("exact purged")
        );
        assert!(
            ModelFeatureInputV1::from_training_frame(
                anchor,
                &frame.select_rows(&[0, 2, 4, 6, 8, 10]).unwrap(),
                Some(frame.timestamps[8]),
                2
            )
            .is_err()
        );
    }

    #[test]
    fn model_input_rejects_forged_fit_scope_and_missing_recipe() {
        let frame = fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let contract =
            ModelFeatureInputV1::from_training_frame(anchor, &frame, Some(frame.timestamps[8]), 2)
                .unwrap();
        let mut tampered = contract.clone();
        tampered
            .purged_training_prefix
            .as_mut()
            .unwrap()
            .source_row_end += 1;
        assert!(tampered.validate().is_err());
        let mut encoded = serde_json::to_value(&contract).unwrap();
        encoded["producer_receipt"]["feature_build_options"] = serde_json::Value::Null;
        assert!(
            ModelFeatureInputV1::from_json_bytes(&serde_json::to_vec(&encoded).unwrap()).is_err()
        );
    }

    #[test]
    fn model_input_io_rejects_oversize_before_parse_and_roundtrips_one_contract() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "neoethos-model-input-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join(MODEL_FEATURE_INPUT_FILE_V1);
        let frame = fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let contract =
            ModelFeatureInputV1::from_training_frame(anchor, &frame, Some(frame.timestamps[8]), 2)
                .unwrap();
        neoethos_core::storage::json::write_bytes_atomic(&path, &contract.to_json_bytes().unwrap())
            .unwrap();
        assert_eq!(
            ModelFeatureInputV1::read_from_path(&path).unwrap(),
            contract
        );
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MODEL_FEATURE_INPUT_MAX_BYTES_V1 as u64 + 1)
            .unwrap();
        assert!(
            ModelFeatureInputV1::read_from_path(&path)
                .unwrap_err()
                .to_string()
                .contains("8 MiB")
        );
        let mut output = BoundedJson(vec![0; MODEL_FEATURE_INPUT_MAX_BYTES_V1]);
        assert!(output.write_all(&[1]).is_err());
        assert_eq!(output.0.len(), MODEL_FEATURE_INPUT_MAX_BYTES_V1);
    }

    #[test]
    fn ordinary_model_input_cannot_masquerade_as_a_purged_candidate() {
        let frame = fixture();
        let anchor = frame.provenance().bindings()[0].dataset_identity();
        let ordinary = ModelFeatureInputV1::from_training_frame(anchor, &frame, None, 2).unwrap();
        assert!(
            ordinary
                .validate_candidate_boundary(frame.timestamps[8], 2)
                .is_err()
        );
        let candidate =
            ModelFeatureInputV1::from_training_frame(anchor, &frame, Some(frame.timestamps[8]), 2)
                .unwrap();
        candidate
            .validate_candidate_boundary(frame.timestamps[8], 2)
            .unwrap();
        assert!(
            candidate
                .validate_candidate_boundary(frame.timestamps[8], 1)
                .is_err()
        );
        assert!(
            candidate
                .validate_candidate_boundary(frame.timestamps[9], 2)
                .is_err()
        );
    }
}
