//! `SoftVotingEnsemble` — the ensemble aggregator.
//!
//! Phase D1.3, Stage 2. This aggregator runs every loaded expert's
//! [`super::ExpertModel::predict`] in turn and combines their
//! Classification3 outputs by **weighted-average** of the
//! `[p_neutral, p_buy, p_sell]` vectors (canonical order — see
//! `base.rs` lines 128-135), through [`SoftVotingEnsemble::predict_with_roles`].
//! The result is one [`super::EnsembleDecision`] per input row — a direction
//! vote plus two bounded gate factors — ready for the producer's
//! `dispatch_auto_trade_signal` gate chain.
//!
//! **2026-08-09 (batch D4)**: the earlier flat average over EVERY
//! Classification3 expert is gone. It let `hmm_regime` and
//! `isolation_forest` vote on direction, which is exactly what
//! `predict_with_roles` was written to stop, and no production caller
//! had used it since. See the note on [`super::EnsemblePredictor`].
//!
//! ## Why "soft voting" and not "MoE"
//!
//! Following the user's research-backed direction (2026-05-17
//! correspondence + the 2025 ensemble-learning survey):
//!
//! - The combiner receives one canonical [`neoethos_data::FeatureFrame`]. Each
//!   expert adapter projects the exact ordered feature columns recorded in its
//!   artifact without copying or privately recomputing market features.
//! - Each expert produces a Classification3 vote.
//! - Soft voting averages those votes — equivalent to assuming
//!   every expert is equally trustworthy. This is the simplest
//!   diversity-aware combiner and ships TODAY against whatever
//!   experts are already trained.
//! - A future MoE gate may replace this layer only through an explicit,
//!   versioned selector. Missing artifacts never trigger an implicit change of
//!   inference semantics.
//!
//! Soft voting is **not a scaffold**: it is a real production
//! aggregation strategy used by widely-deployed ensembles (sklearn
//! `VotingClassifier`, Kaggle competitions, etc.). The MoE will
//! often outperform it, but soft voting alone is a meaningful
//! baseline that lets the bot generate real signals from real
//! trained models from day one.
//!
//! ## Honest limitations
//!
//! - **Heterogeneous output kinds**: experts that emit
//!   `Forecast1` / `AnomalyScore` / `ActionValues3` cannot be
//!   averaged with Classification3 directly. SoftVotingEnsemble
//!   silently SKIPS them — they sit unused at the voting layer,
//!   counted in `experts_unused_for_voting()`. The MoE (D1.5+)
//!   is the right consumer for those signal types.
//! - **No confidence calibration**: averaging produces sharper
//!   distributions when experts agree and flatter ones when they
//!   disagree, but the resulting probabilities are NOT calibrated
//!   to long-run accuracy. The producer's gate chain converts the
//!   argmax + raw confidence to a trade decision; downstream
//!   prop-firm gates and the operator's confidence floor handle
//!   the rest.
//! - **No abstention gate**: unlike `MetaDecisionStack`'s conformal
//!   prediction layer, SoftVoting always votes. There used to be a
//!   `SoftVotingEnsembleConfig::abstain_below_confidence` knob here;
//!   it was removed 2026-08-09 (batch D4) because it was doubly
//!   unreachable — its only reader was `maybe_abstain`, called only
//!   from the flat-average `EnsemblePredictor::predict`, which no
//!   production path ever invoked. Restoring abstention means adding
//!   it to `predict_with_roles`, the combiner that actually runs.

use std::collections::HashSet;

use anyhow::Result;
use neoethos_data::{FeatureCellValidity, FeatureFrame};
use neoethos_execution_budget::CpuLease;

use super::{
    EnsembleDecision, EnsemblePredictor, ExpertLoadOutcome, ExpertOutputKind, ExpertPrediction,
    ExpertRole, anomaly_scale_from_score, expert_role, regime_gate_from_posterior,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Tunables for [`SoftVotingEnsemble`]. All have sensible defaults.
#[derive(Debug, Clone)]
pub struct SoftVotingEnsembleConfig {
    /// Optional per-expert weight (`name` → `weight`). Experts not
    /// listed get weight 1.0. Weights normalise per-row at predict
    /// time so the output is always a valid probability vector.
    /// Useful when the operator has validation accuracy data and
    /// wants to bias the average toward better-performing experts.
    pub expert_weights: std::collections::HashMap<String, f64>,
    /// Expert canonical names that must NOT participate in voting
    /// even when present in the load outcome.
    ///
    /// **History (F-319, 2026-05-29)**: this field used to hold a
    /// default exclusion set of `{"genetic", "neuro_evo"}` because
    /// those adapters were architecturally misplaced — they wrapped
    /// strategy-discovery algorithms (GA / CR-FM-NES neuroevolution)
    /// from `neoethos-search` as if they were inference experts. The
    /// 2026-05-17 operator correction added the exclusion to prevent
    /// double-counting. F-319 removed the adapters entirely (the
    /// discoverers run in the search crate; only trained models vote
    /// here), so the default exclusion set is now empty — there are
    /// no built-in non-voters left to skip. Operators can still
    /// populate this field manually to drop a specific expert from
    /// voting (e.g., A/B testing whether a deep model contributes).
    pub excluded_names: std::collections::HashSet<String>,
    /// v0.5 ML-integration Stage 2: anomaly-score lower knee. Raw
    /// `isolation_forest` scores below this get no size penalty (scale 1.0).
    pub anomaly_lo: f64,
    /// Anomaly-score upper knee — at/above this the anomaly scale hard-vetoes
    /// to 0.0. Default 0.9 (matches the trained ~0.95-quantile threshold).
    pub anomaly_hi: f64,
}

impl Default for SoftVotingEnsembleConfig {
    fn default() -> Self {
        Self {
            expert_weights: std::collections::HashMap::new(),
            excluded_names: std::collections::HashSet::new(),
            anomaly_lo: 0.5,
            anomaly_hi: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// SoftVotingEnsemble
// ---------------------------------------------------------------------------

/// Weighted-average aggregator over a set of loaded experts.
pub struct SoftVotingEnsemble {
    outcome: ExpertLoadOutcome,
    config: SoftVotingEnsembleConfig,
    /// Names of experts whose output_kind is NOT Classification3.
    /// Cached at construction so the aggregator doesn't pay the
    /// per-predict cost of re-checking. Surfaced via
    /// [`Self::experts_unused_for_voting`] for the chrome banner.
    unused_for_voting: HashSet<String>,
    model_feature_input: Option<crate::runtime::feature_input::ModelFeatureInputV1>,
}

/// An immutable numerical input binding. Only windows of this exact shared
/// frame may reach its ensemble without repeating the full fitted-state check.
pub struct BoundModelInference<'a> {
    ensemble: &'a SoftVotingEnsemble,
    frame: &'a std::sync::Arc<FeatureFrame>,
    projections: Vec<neoethos_data::BoundFeatureColumnProjection>,
    expert_projections: Vec<Option<usize>>,
}

impl BoundModelInference<'_> {
    /// Infer causally using at most `history_rows` ending immediately before
    /// `end_exclusive`. Future rows cannot enter an expert's history window.
    pub fn last_row(
        &self,
        end_exclusive: usize,
        history_rows: usize,
        lease: &CpuLease,
    ) -> Result<EnsembleDecision> {
        anyhow::ensure!(
            history_rows > 0 && end_exclusive > 0 && end_exclusive <= self.frame.n_samples(),
            "bound model window must have positive history and a valid exclusive end"
        );
        let start = end_exclusive.saturating_sub(history_rows);
        self.ensemble
            .predict_with_roles_selected(end_exclusive - start, |index, expert| {
                let projection = self.expert_projections[index].ok_or_else(|| {
                    anyhow::anyhow!("active expert has no bound column projection")
                })?;
                let window = self.projections[projection].row_window(start..end_exclusive)?;
                expert.predict(&window, lease)
            })?
            .into_iter()
            .next_back()
            .ok_or_else(|| anyhow::anyhow!("bound model window returned no final decision"))
    }
}

impl SoftVotingEnsemble {
    /// Build from a load outcome + config. Errors if NO loaded
    /// expert can contribute to voting after applying both filters
    /// (output_kind == Classification3 AND name not in excluded).
    pub fn new(outcome: ExpertLoadOutcome, config: SoftVotingEnsembleConfig) -> Result<Self> {
        let mut unused = HashSet::new();
        let mut votable = 0;
        for e in &outcome.loaded {
            let name = e.name();
            // An expert is "unused" if EITHER its output kind isn't
            // Classification3 (Forecast1, AnomalyScore, ExitDecision3,
            // ActionValues3) OR its name is in the operator's
            // exclusion list (strategy discoverers like genetic,
            // neuro_evo by default).
            let wrong_kind = e.output_kind() != ExpertOutputKind::Classification3;
            let excluded = config.excluded_names.contains(name);
            if wrong_kind || excluded {
                unused.insert(name.to_string());
            } else {
                votable += 1;
            }
        }
        if votable == 0 {
            anyhow::bail!(
                "SoftVotingEnsemble requires at least one votable Classification3 expert in \
                 the load outcome AFTER applying the exclusion list. Loaded {} experts, all \
                 of which were either heterogeneous-output-kind or excluded by name. Unused: \
                 {:?}",
                outcome.loaded.len(),
                unused
            );
        }
        Ok(Self {
            outcome,
            config,
            unused_for_voting: unused,
            model_feature_input: None,
        })
    }

    pub(crate) fn bind_model_feature_input(
        mut self,
        input: crate::runtime::feature_input::ModelFeatureInputV1,
    ) -> Result<Self> {
        input.validate()?;
        let columns = input
            .feature_columns()
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        for expert in &self.outcome.loaded {
            anyhow::ensure!(
                expert
                    .feature_columns()
                    .iter()
                    .all(|name| columns.contains(name.as_str())),
                "expert '{}' requests columns absent from its model producer contract",
                expert.name()
            );
        }
        self.model_feature_input = Some(input);
        Ok(self)
    }

    pub fn model_feature_input(
        &self,
    ) -> Option<&crate::runtime::feature_input::ModelFeatureInputV1> {
        self.model_feature_input.as_ref()
    }

    /// Produce the saved model's own numerical input, independently of Search.
    pub fn prepare_model_features(
        &self,
        dataset: &neoethos_data::SymbolDataset,
    ) -> Result<FeatureFrame> {
        self.prepare_model_features_with_control(
            dataset,
            &neoethos_data::FeatureBuildControl::default(),
        )
    }

    pub fn prepare_model_features_with_control(
        &self,
        dataset: &neoethos_data::SymbolDataset,
        control: &neoethos_data::FeatureBuildControl,
    ) -> Result<FeatureFrame> {
        self.model_feature_input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ensemble has no persisted model-owned preprocessing"))?
            .prepare_features_with_control(dataset, control)
    }

    pub fn bind_model_features<'a>(
        &'a self,
        frame: &'a std::sync::Arc<FeatureFrame>,
    ) -> Result<BoundModelInference<'a>> {
        self.model_feature_input
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("verified model inference requires persisted preprocessing")
            })?
            .validate_features(frame)?;
        let indices_by_name = frame
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<std::collections::HashMap<_, _>>();
        let mut schemas = std::collections::HashMap::<Vec<usize>, usize>::new();
        let mut projections = Vec::new();
        let mut expert_projections = Vec::with_capacity(self.outcome.loaded.len());
        for expert in &self.outcome.loaded {
            if self.config.excluded_names.contains(expert.name())
                || expert.output_kind() != ExpertOutputKind::Classification3
            {
                expert_projections.push(None);
                continue;
            }
            anyhow::ensure!(
                !expert.feature_columns().is_empty(),
                "expert '{}' has no recorded feature columns",
                expert.name()
            );
            let indices = expert
                .feature_columns()
                .iter()
                .map(|name| {
                    indices_by_name.get(name.as_str()).copied().ok_or_else(|| {
                        anyhow::anyhow!(
                            "bound expert '{}' feature '{name}' is absent",
                            expert.name()
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let projection = if let Some(&existing) = schemas.get(&indices) {
                existing
            } else {
                let index = projections.len();
                projections.push(frame.bind_column_projection(&indices)?);
                schemas.insert(indices, index);
                index
            };
            expert_projections.push(Some(projection));
        }
        Ok(BoundModelInference {
            ensemble: self,
            frame,
            projections,
            expert_projections,
        })
    }

    /// Names of loaded experts whose `output_kind` is not
    /// Classification3 — they're held in the outcome (so the
    /// chrome can list them) but the soft-voting layer doesn't use
    /// their predictions. The MoE will (D1.5+).
    pub fn experts_unused_for_voting(&self) -> Vec<&str> {
        self.unused_for_voting.iter().map(String::as_str).collect()
    }

    /// Count of experts that actually participate in voting.
    pub fn voting_expert_count(&self) -> usize {
        self.outcome.loaded.len() - self.unused_for_voting.len()
    }

    /// v0.5 ML-integration Stage 2 — role-aware combiner. THE aggregator:
    /// both production consumers (`bootstrap.rs:246` replay, `bootstrap.rs:275`
    /// live) call this, and since 2026-08-09 it is the only one that exists.
    ///
    /// It replaced a flat average over EVERY Classification3 expert, which let
    /// `hmm_regime`/`isolation_forest` pollute the direction vote. This
    /// partitions the loaded experts by [`ExpertRole`] and returns one
    /// [`EnsembleDecision`] per row:
    /// - direction vote = weighted average of the genuine directional
    ///   classifiers + `dqn` (confirm), with `hmm_regime` / `isolation_forest`
    ///   REMOVED from the vote;
    /// - `regime_gate` ∈ [0,1] from `hmm_regime` (1.0 when absent);
    /// - `anomaly_scale` ∈ [0,1] from `isolation_forest` (1.0 when absent,
    ///   0.0 hard-veto at an extreme score).
    ///
    /// FAILS LOUD if a loaded, non-excluded expert name is unmapped (a new
    /// expert must be assigned a role in [`expert_role`]) or if the direction
    /// pool ends up empty (re-roling must never strip every directional voter).
    /// The two gate factors are bounded [0,1], so the ensemble can only SHRINK
    /// conviction or veto — never flip direction or manufacture a trade.
    pub fn predict_with_roles(
        &self,
        frame: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<EnsembleDecision>> {
        if let Some(input) = &self.model_feature_input {
            input.validate_features(frame)?;
        }
        self.predict_with_roles_validated(frame, lease)
    }

    fn predict_with_roles_validated(
        &self,
        frame: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Vec<EnsembleDecision>> {
        self.predict_with_roles_selected(frame.n_samples(), |_, expert| {
            expert.predict(frame, lease)
        })
    }

    fn predict_with_roles_selected(
        &self,
        n_rows: usize,
        predict: impl Fn(usize, &dyn super::ExpertModel) -> Result<Vec<ExpertPrediction>>,
    ) -> Result<Vec<EnsembleDecision>> {
        if n_rows == 0 {
            return Ok(Vec::new());
        }

        // Direction vote accumulator (weighted, per row).
        let mut dir_sums: Vec<[f64; 3]> = vec![[0.0; 3]; n_rows];
        let mut dir_weight_totals: Vec<f64> = vec![0.0; n_rows];
        let mut row_validity = vec![FeatureCellValidity::Valid; n_rows];
        let mut direction_voters = 0usize;
        // Optional per-row regime posterior + anomaly score.
        let mut regime_posterior: Option<Vec<[f64; 3]>> = None;
        let mut regime_validity: Option<Vec<FeatureCellValidity>> = None;
        let mut anomaly_scores: Option<Vec<f64>> = None;
        let mut anomaly_validity: Option<Vec<FeatureCellValidity>> = None;

        for (expert_index, expert) in self.outcome.loaded.iter().enumerate() {
            let name = expert.name();
            if self.config.excluded_names.contains(name) {
                continue;
            }
            // Non-Classification3 kinds (Forecast1, ExitDecision3, …) are not
            // consumed by this combiner; the role map only covers voting kinds.
            if expert.output_kind() != ExpertOutputKind::Classification3 {
                continue;
            }
            let Some(role) = expert_role(name) else {
                anyhow::bail!(
                    "role-aware combiner: loaded expert '{}' has no role mapping; add it to \
                     `expert_role` (Direction / DirectionalConfirm / RegimeGate / AnomalyScale)",
                    name
                );
            };

            let preds: Vec<ExpertPrediction> = predict(expert_index, expert.as_ref())?;
            if preds.len() != n_rows {
                anyhow::bail!(
                    "expert '{}' returned {} predictions for a {}-row FeatureFrame",
                    name,
                    preds.len(),
                    n_rows
                );
            }

            match role {
                ExpertRole::Direction | ExpertRole::DirectionalConfirm => {
                    let weight = self.config.expert_weights.get(name).copied().unwrap_or(1.0);
                    if !weight.is_finite() || weight < 0.0 {
                        anyhow::bail!("expert '{name}' has invalid voting weight {weight}");
                    }
                    if weight == 0.0 {
                        continue;
                    }
                    direction_voters += 1;
                    for (row_idx, p) in preds.iter().enumerate() {
                        p.validate()?;
                        if !p.validity.is_valid() {
                            if row_validity[row_idx].is_valid() {
                                row_validity[row_idx] = p.validity;
                            }
                            continue;
                        }
                        if p.kind != ExpertOutputKind::Classification3 || p.values.len() != 3 {
                            anyhow::bail!(
                                "direction expert '{name}' returned incompatible output at row {row_idx}"
                            );
                        }
                        dir_sums[row_idx][0] += weight * p.values[0];
                        dir_sums[row_idx][1] += weight * p.values[1];
                        dir_sums[row_idx][2] += weight * p.values[2];
                        dir_weight_totals[row_idx] += weight;
                    }
                }
                ExpertRole::RegimeGate => {
                    // hmm_regime posterior: col0=P(range), col1=P(buy), col2=P(sell).
                    let mut post = vec![[f64::NAN; 3]; n_rows];
                    let mut validity = vec![FeatureCellValidity::ComputeFailure; n_rows];
                    for (row_idx, p) in preds.iter().enumerate() {
                        p.validate()?;
                        validity[row_idx] = p.validity;
                        if p.validity.is_valid() {
                            if p.values.len() != 3 {
                                anyhow::bail!(
                                    "regime expert '{name}' returned incompatible output at row {row_idx}"
                                );
                            }
                            post[row_idx] = [p.values[0], p.values[1], p.values[2]];
                        }
                    }
                    regime_posterior = Some(post);
                    regime_validity = Some(validity);
                }
                ExpertRole::AnomalyScale => {
                    // isolation_forest emits [anomaly, (1-a)/2, (1-a)/2] -> col0
                    // is the raw anomaly score (no retrain / new artifact needed).
                    let mut scores = vec![f64::NAN; n_rows];
                    let mut validity = vec![FeatureCellValidity::ComputeFailure; n_rows];
                    for (row_idx, p) in preds.iter().enumerate() {
                        p.validate()?;
                        validity[row_idx] = p.validity;
                        if p.validity.is_valid() {
                            if p.values.is_empty() {
                                anyhow::bail!(
                                    "anomaly expert '{name}' returned an empty output at row {row_idx}"
                                );
                            }
                            scores[row_idx] = p.values[0];
                        }
                    }
                    anomaly_scores = Some(scores);
                    anomaly_validity = Some(validity);
                }
            }
        }

        if direction_voters == 0 {
            anyhow::bail!(
                "role-aware combiner: no directional voters remained after re-roling \
                 (hmm_regime/isolation_forest are gates, not voters). At least one genuine \
                 Classification3 directional expert must be loaded."
            );
        }

        let mut out = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            if row_validity[row_idx].is_valid()
                && let Some(validity) = &regime_validity
                && !validity[row_idx].is_valid()
            {
                row_validity[row_idx] = validity[row_idx];
            }
            if row_validity[row_idx].is_valid()
                && let Some(validity) = &anomaly_validity
                && !validity[row_idx].is_valid()
            {
                row_validity[row_idx] = validity[row_idx];
            }
            if !row_validity[row_idx].is_valid() {
                out.push(EnsembleDecision::invalid(row_validity[row_idx]));
                continue;
            }
            let total = dir_weight_totals[row_idx];
            if !total.is_finite() || total <= 0.0 {
                out.push(EnsembleDecision::invalid(
                    FeatureCellValidity::ComputeFailure,
                ));
                continue;
            }
            let dir_probs = [
                dir_sums[row_idx][0] / total,
                dir_sums[row_idx][1] / total,
                dir_sums[row_idx][2] / total,
            ];
            let regime_gate = match &regime_posterior {
                Some(post) => regime_gate_from_posterior(dir_probs, post[row_idx]),
                None => 1.0,
            };
            let anomaly_scale = match &anomaly_scores {
                Some(scores) => anomaly_scale_from_score(
                    scores[row_idx],
                    self.config.anomaly_lo,
                    self.config.anomaly_hi,
                ),
                None => 1.0,
            };
            out.push(EnsembleDecision {
                dir_probs,
                regime_gate,
                anomaly_scale,
                validity: FeatureCellValidity::Valid,
            });
        }
        Ok(out)
    }
}

impl EnsemblePredictor for SoftVotingEnsemble {
    // NOTE (2026-08-09, batch D4): the flat-average `predict` that used to live
    // here — a weighted mean over EVERY Classification3 expert — is gone, and
    // so is the trait method it implemented. It had no production caller:
    // `bootstrap.rs:246` (replay) and `bootstrap.rs:275` (live) both call
    // `predict_with_roles`, which exists precisely because the flat average let
    // `hmm_regime` and `isolation_forest` vote on DIRECTION. It went with
    // `maybe_abstain` and `SoftVotingEnsembleConfig::abstain_below_confidence`,
    // which were reachable only through it.
    fn load_outcome(&self) -> &ExpertLoadOutcome {
        &self.outcome
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble_inference::{
        ExpertLoadOutcome, ExpertModel, ExpertOutputKind, ExpertPrediction,
    };
    use crate::runtime::capabilities::ModelFamily;
    use anyhow::Result;
    use neoethos_data::{FeatureColumnF64, FeatureFrame};
    use neoethos_execution_budget::{CpuLease, CpuPermitBroker, CpuPermitRequest, WorkerLimit};

    /// In-test ExpertModel that emits a constant Classification3
    /// prediction for every row.
    struct ConstantClassifier {
        name: String,
        probs: [f64; 3],
    }

    impl ExpertModel for ConstantClassifier {
        fn name(&self) -> &str {
            &self.name
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Tree
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Classification3
        }
        fn feature_columns(&self) -> &[String] {
            &[]
        }
        fn predict(
            &self,
            frame: &FeatureFrame,
            _lease: &CpuLease,
        ) -> Result<Vec<ExpertPrediction>> {
            let n = frame.n_samples();
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(ExpertPrediction::valid(
                    ExpertOutputKind::Classification3,
                    self.probs.to_vec(),
                )?);
            }
            Ok(out)
        }
    }

    /// Forecast1 expert — should be IGNORED by SoftVoting.
    struct ForecastEmitter;
    impl ExpertModel for ForecastEmitter {
        fn name(&self) -> &str {
            "forecaster"
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Forecasting
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Forecast1
        }
        fn feature_columns(&self) -> &[String] {
            &[]
        }
        fn predict(
            &self,
            frame: &FeatureFrame,
            _lease: &CpuLease,
        ) -> Result<Vec<ExpertPrediction>> {
            (0..frame.n_samples())
                .map(|_| ExpertPrediction::valid(ExpertOutputKind::Forecast1, vec![0.5]))
                .collect()
        }
    }

    struct InvalidClassifier {
        name: String,
        reason: FeatureCellValidity,
    }

    impl ExpertModel for InvalidClassifier {
        fn name(&self) -> &str {
            &self.name
        }

        fn family(&self) -> ModelFamily {
            ModelFamily::Meta
        }

        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Classification3
        }

        fn feature_columns(&self) -> &[String] {
            &[]
        }

        fn predict(
            &self,
            frame: &FeatureFrame,
            _lease: &CpuLease,
        ) -> Result<Vec<ExpertPrediction>> {
            (0..frame.n_samples())
                .map(|_| ExpertPrediction::invalid(ExpertOutputKind::Classification3, self.reason))
                .collect()
        }
    }

    fn outcome_with(experts: Vec<Box<dyn ExpertModel>>) -> ExpertLoadOutcome {
        ExpertLoadOutcome {
            loaded: experts,
            missing: vec![],
            degraded: vec![],
        }
    }

    fn small_frame(rows: usize) -> FeatureFrame {
        assert!(rows > 0, "canonical FeatureFrame cannot be empty");
        let column = FeatureColumnF64::new(
            "f1",
            (0..rows).map(|row| row as f64 + 1.0).collect(),
            vec![FeatureCellValidity::Valid; rows],
        )
        .expect("valid f64 test column");
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            vec![column],
        )
        .expect("valid canonical test frame")
    }

    fn one_worker_lease() -> CpuLease {
        let width = WorkerLimit::new(1).expect("one worker");
        CpuPermitBroker::new(width)
            .acquire(CpuPermitRequest::local(width))
            .expect("test CPU lease")
    }

    fn predict_rows(ensemble: &SoftVotingEnsemble, rows: usize) -> Result<Vec<EnsembleDecision>> {
        let frame = small_frame(rows);
        let lease = one_worker_lease();
        ensemble.predict_with_roles(&frame, &lease)
    }

    /// Replaces the deleted `SoftVotingEnsemble::with_default_config`, which
    /// existed only for these tests.
    fn default_ensemble(outcome: ExpertLoadOutcome) -> Result<SoftVotingEnsemble> {
        SoftVotingEnsemble::new(outcome, SoftVotingEnsembleConfig::default())
    }

    // -- Construction invariants ---------------------------------------

    #[test]
    fn new_rejects_empty_classification3_set() {
        let outcome = outcome_with(vec![Box::new(ForecastEmitter)]);
        // Cannot use expect_err — SoftVotingEnsemble holds Box<dyn ExpertModel>
        // which does not implement Debug. Match on the result instead.
        match default_ensemble(outcome) {
            Ok(_) => panic!("must reject empty Classification3 set"),
            Err(err) => assert!(err.to_string().contains("Classification3")),
        }
    }

    #[test]
    fn new_accepts_when_at_least_one_classification3() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.2, 0.6, 0.2],
            }),
            Box::new(ForecastEmitter),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        assert_eq!(ens.voting_expert_count(), 1);
        assert_eq!(ens.experts_unused_for_voting(), vec!["forecaster"]);
    }

    // -- Stage 2: role-aware combiner ---------------------------------

    #[test]
    fn role_map_covers_all_bootstrap_experts() {
        for name in crate::ensemble_inference::bootstrap::DEFAULT_BOOTSTRAP_EXPERT_NAMES {
            assert!(
                expert_role(name).is_some(),
                "bootstrap expert '{name}' has no role in expert_role(); the role-aware \
                 combiner would fail loud on it in production"
            );
        }
        assert_eq!(expert_role("hmm_regime"), Some(ExpertRole::RegimeGate));
        assert_eq!(
            expert_role("isolation_forest"),
            Some(ExpertRole::AnomalyScale)
        );
        assert_eq!(expert_role("dqn"), Some(ExpertRole::DirectionalConfirm));
        assert_eq!(expert_role("sac"), Some(ExpertRole::DirectionalConfirm));
        assert_eq!(expert_role("xgboost"), Some(ExpertRole::Direction));
        assert_eq!(expert_role("not_a_real_model"), None);
    }

    #[test]
    fn regime_gate_pure_math() {
        // Voted long; HMM strong buy-trend -> gate near 1.
        let g = regime_gate_from_posterior([0.1, 0.8, 0.1], [0.05, 0.9, 0.05]);
        assert!(g > 0.8, "agreeing trend should keep size, got {g}");
        // Voted long; HMM says range -> gate near 0.
        let g = regime_gate_from_posterior([0.1, 0.8, 0.1], [0.95, 0.03, 0.02]);
        assert!(g < 0.1, "range regime should shrink, got {g}");
        // Voted long; HMM says sell-trend (disagree) -> small gate.
        let g = regime_gate_from_posterior([0.1, 0.8, 0.1], [0.05, 0.05, 0.9]);
        assert!(g < 0.1, "disagreeing trend should shrink, got {g}");
    }

    #[test]
    fn anomaly_scale_pure_math() {
        assert_eq!(anomaly_scale_from_score(0.3, 0.5, 0.9), 1.0); // below lo
        assert_eq!(anomaly_scale_from_score(0.9, 0.5, 0.9), 0.0); // at hi -> veto
        assert_eq!(anomaly_scale_from_score(0.95, 0.5, 0.9), 0.0); // above hi
        let mid = anomaly_scale_from_score(0.7, 0.5, 0.9); // halfway -> 0.5
        assert!(
            (mid - 0.5).abs() < 1e-6,
            "mid ramp should be 0.5, got {mid}"
        );
    }

    #[test]
    fn predict_with_roles_excludes_gates_from_direction() {
        // Directional voter votes strong buy; the regime + anomaly experts must
        // NOT pollute dir_probs — they only set the gate factors.
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.8, 0.1],
            }),
            Box::new(ConstantClassifier {
                name: "hmm_regime".into(),
                probs: [0.05, 0.9, 0.05], // P(range)=.05, P(buy)=.9
            }),
            Box::new(ConstantClassifier {
                name: "isolation_forest".into(),
                probs: [0.3, 0.35, 0.35], // col0=anomaly score 0.3 (< lo)
            }),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        let decisions = predict_rows(&ens, 2).expect("roles");
        assert_eq!(decisions.len(), 2);
        for d in &decisions {
            // dir_probs == the SOLE directional voter, gates removed.
            assert!(
                (d.dir_probs[1] - 0.8).abs() < 1e-6,
                "dir vote polluted: {d:?}"
            );
            assert!(d.regime_gate > 0.8, "agreeing regime gate: {d:?}");
            assert_eq!(d.anomaly_scale, 1.0, "low anomaly -> no penalty: {d:?}");
        }
    }

    #[test]
    fn invalid_gate_invalidates_row_instead_of_becoming_neutral_numeric_data() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.8, 0.1],
            }),
            Box::new(InvalidClassifier {
                name: "hmm_regime".into(),
                reason: FeatureCellValidity::Warmup,
            }),
        ]);
        let ensemble = default_ensemble(outcome).expect("valid ensemble");
        let decisions = predict_rows(&ensemble, 2).expect("role-aware inference");

        for decision in decisions {
            assert_eq!(decision.validity, FeatureCellValidity::Warmup);
            assert!(decision.dir_probs.iter().all(|value| value.is_nan()));
            assert!(decision.regime_gate.is_nan());
            assert!(decision.anomaly_scale.is_nan());
        }
    }

    #[test]
    fn predict_with_roles_bails_when_no_direction_voter() {
        // Only gates loaded -> construction succeeds (they are Classification3)
        // but the role-aware combiner must refuse (no directional voter).
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "hmm_regime".into(),
                probs: [0.2, 0.4, 0.4],
            }),
            Box::new(ConstantClassifier {
                name: "isolation_forest".into(),
                probs: [0.1, 0.45, 0.45],
            }),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        match predict_rows(&ens, 1) {
            Ok(_) => panic!("must bail when no directional voter remains"),
            Err(err) => assert!(err.to_string().contains("no directional voters")),
        }
    }

    #[test]
    fn predict_with_roles_bails_on_unmapped_expert() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.2, 0.6, 0.2],
            }),
            Box::new(ConstantClassifier {
                name: "mystery_model".into(),
                probs: [0.3, 0.4, 0.3],
            }),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        match predict_rows(&ens, 1) {
            Ok(_) => panic!("must fail loud on an unmapped expert"),
            Err(err) => assert!(err.to_string().contains("no role mapping")),
        }
    }

    // -- Vote arithmetic ----------------------------------------------
    //
    // 2026-08-09 (batch D4): these used to exercise the flat-average
    // `EnsemblePredictor::predict`, which production never called. They now
    // exercise `predict_with_roles` — the combiner that actually runs — so the
    // weighting / exclusion / skipping mechanisms stay pinned on the live path
    // instead of on a dead one. Expert names must carry a role mapping, hence
    // `xgboost` / `lightgbm` rather than `a` / `b`.

    #[test]
    fn single_expert_pass_through() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "xgboost".into(),
            probs: [0.1, 0.7, 0.2],
        })]);
        let ens = default_ensemble(outcome).expect("ok");
        let decisions = predict_rows(&ens, 3).expect("roles");
        assert_eq!(decisions.len(), 3);
        for d in &decisions {
            assert!((d.dir_probs[0] - 0.1).abs() < 1e-6);
            assert!((d.dir_probs[1] - 0.7).abs() < 1e-6);
            assert!((d.dir_probs[2] - 0.2).abs() < 1e-6);
            // No hmm_regime / isolation_forest loaded -> both gates neutral.
            assert_eq!(d.regime_gate, 1.0);
            assert_eq!(d.anomaly_scale, 1.0);
        }
    }

    #[test]
    fn two_experts_equal_weight_averaged() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.8, 0.1, 0.1],
            }),
            Box::new(ConstantClassifier {
                name: "lightgbm".into(),
                probs: [0.2, 0.6, 0.2],
            }),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        let decisions = predict_rows(&ens, 2).expect("roles");
        // Average of (0.8,0.1,0.1) + (0.2,0.6,0.2) = (0.5,0.35,0.15)
        for d in &decisions {
            assert!((d.dir_probs[0] - 0.5).abs() < 1e-5);
            assert!((d.dir_probs[1] - 0.35).abs() < 1e-5);
            assert!((d.dir_probs[2] - 0.15).abs() < 1e-5);
        }
    }

    #[test]
    fn per_expert_weights_bias_average() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.8, 0.1, 0.1],
            }),
            Box::new(ConstantClassifier {
                name: "lightgbm".into(),
                probs: [0.2, 0.6, 0.2],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.expert_weights.insert("xgboost".into(), 3.0);
        cfg.expert_weights.insert("lightgbm".into(), 1.0);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let decisions = predict_rows(&ens, 1).expect("roles");
        // Weighted: (3*0.8 + 1*0.2)/4, (3*0.1+1*0.6)/4, (3*0.1+1*0.2)/4
        //         = (0.65, 0.225, 0.125)
        let d = decisions[0];
        assert!((d.dir_probs[0] - 0.65).abs() < 1e-5);
        assert!((d.dir_probs[1] - 0.225).abs() < 1e-5);
        assert!((d.dir_probs[2] - 0.125).abs() < 1e-5);
    }

    #[test]
    fn forecast_experts_are_skipped() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ForecastEmitter),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        let decisions = predict_rows(&ens, 1).expect("roles");
        // ForecastEmitter must not have contributed — and must not have
        // tripped the unmapped-expert bail either.
        let d = decisions[0];
        assert!((d.dir_probs[0] - 0.1).abs() < 1e-6);
        assert!((d.dir_probs[1] - 0.7).abs() < 1e-6);
        assert!((d.dir_probs[2] - 0.2).abs() < 1e-6);
    }

    // -- Load outcome surfacing --------------------------------------

    #[test]
    fn load_outcome_round_trips_through_trait() {
        let outcome = ExpertLoadOutcome {
            loaded: vec![Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.2, 0.6, 0.2],
            })],
            missing: vec!["lightgbm".into(), "transformer".into()],
            degraded: vec![],
        };
        let ens = default_ensemble(outcome).expect("ok");
        let lo = ens.load_outcome();
        assert_eq!(lo.loaded_count(), 1);
        assert_eq!(lo.missing_count(), 2);
        assert_eq!(lo.loaded_names(), vec!["xgboost"]);
    }

    #[test]
    fn default_config_excludes_strategy_discoverers() {
        // F-319 (2026-05-29): the strategy-discoverer adapters
        // (genetic / neuro_evo / neat) were removed from the inference
        // layer entirely, so the default exclusion set is now EMPTY —
        // there are no built-in non-voters left to skip. Operators can
        // still populate `excluded_names` manually to drop a specific
        // expert from voting (see the explicit-exclusion tests below).
        let cfg = SoftVotingEnsembleConfig::default();
        assert!(cfg.excluded_names.is_empty());
    }

    #[test]
    fn excluded_expert_is_skipped_at_voting_layer() {
        // A regular voter plus a second one the operator excludes by name. The
        // excluded expert must not contribute to the direction vote even though
        // its output_kind is Classification3.
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantClassifier {
                name: "neuro_evo".into(),
                probs: [0.8, 0.1, 0.1],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.excluded_names.insert("neuro_evo".to_string());
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        // 2 loaded but only 1 votes — neuro_evo excluded.
        assert_eq!(ens.voting_expert_count(), 1);
        assert!(ens.experts_unused_for_voting().contains(&"neuro_evo"));
        // The vote must reflect ONLY the regular expert, not an average.
        let d = predict_rows(&ens, 1).expect("roles")[0];
        assert!((d.dir_probs[0] - 0.1).abs() < 1e-6);
        assert!((d.dir_probs[1] - 0.7).abs() < 1e-6);
        assert!((d.dir_probs[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn operator_can_clear_exclusion_to_include_strategy_discoverers() {
        // Operator override: someone WANTS neuro_evo in the vote (e.g. a
        // sanity check during validation). They clear the exclusion list and
        // it participates.
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "neuro_evo".into(),
            probs: [0.1, 0.7, 0.2],
        })]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.excluded_names.clear();
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        assert_eq!(ens.voting_expert_count(), 1);
        let d = predict_rows(&ens, 1).expect("roles")[0];
        assert!((d.dir_probs[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn zero_weight_expert_drops_out_of_the_vote() {
        // Weight 0 is the operator's "load it but do not let it vote" lever.
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantClassifier {
                name: "lightgbm".into(),
                probs: [0.9, 0.05, 0.05],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.expert_weights.insert("lightgbm".into(), 0.0);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let d = predict_rows(&ens, 1).expect("roles")[0];
        assert!(
            (d.dir_probs[1] - 0.7).abs() < 1e-6,
            "a zero-weight expert must not pull the vote, got {:?}",
            d.dir_probs
        );
    }

    #[test]
    fn canonical_feature_frame_rejects_empty_rows_before_ensemble_inference() {
        let column = FeatureColumnF64::new("f1", Vec::new(), Vec::new())
            .expect("empty column shape is internally consistent");
        let error = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            Vec::new(),
            vec![column],
        )
        .expect_err("canonical feature frames cannot be empty");
        assert!(error.to_string().contains("must not be empty"));
    }

    #[test]
    fn candidate_ensemble_rejects_raw_or_refitted_inputs_before_expert_prediction() {
        let raw = small_frame(12);
        let normalized =
            neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
                &raw,
                0..6,
                None,
            )
            .unwrap();
        let anchor = normalized.provenance().bindings()[0].dataset_identity();
        let input = crate::runtime::feature_input::ModelFeatureInputV1::from_training_frame(
            anchor,
            &normalized,
            Some(raw.timestamps[8]),
            2,
        )
        .unwrap();
        let ensemble = default_ensemble(outcome_with(vec![Box::new(ConstantClassifier {
            name: "xgboost".into(),
            probs: [0.1, 0.8, 0.1],
        })]))
        .unwrap()
        .bind_model_feature_input(input)
        .unwrap();
        let lease = one_worker_lease();
        assert_eq!(
            ensemble
                .predict_with_roles(&normalized, &lease)
                .unwrap()
                .len(),
            12
        );
        assert!(ensemble.predict_with_roles(&raw, &lease).is_err());
        let refitted = neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
            &raw,
            0..10,
            None,
        )
        .unwrap();
        assert!(ensemble.predict_with_roles(&refitted, &lease).is_err());
        assert!(
            ensemble
                .predict_with_roles(&normalized.row_window(10, 12).unwrap(), &lease)
                .is_ok()
        );
    }

    struct WindowMeanClassifier;
    impl ExpertModel for WindowMeanClassifier {
        fn name(&self) -> &str {
            "xgboost"
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Tree
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Classification3
        }
        fn feature_columns(&self) -> &[String] {
            static COLUMNS: std::sync::LazyLock<Vec<String>> =
                std::sync::LazyLock::new(|| vec!["f1".into()]);
            &COLUMNS
        }
        fn predict(&self, frame: &FeatureFrame, _: &CpuLease) -> Result<Vec<ExpertPrediction>> {
            let column = frame.feature_column(0)?;
            let mean = column.values.iter().sum::<f64>() / column.len() as f64;
            let buy = 0.5 + 0.4 * mean.tanh();
            (0..frame.n_samples())
                .map(|_| {
                    ExpertPrediction::valid(
                        ExpertOutputKind::Classification3,
                        vec![0.0, buy, 1.0 - buy],
                    )
                })
                .collect()
        }
    }

    struct ProjectedMeanClassifier {
        name: &'static str,
        columns: Vec<String>,
    }

    impl ExpertModel for ProjectedMeanClassifier {
        fn name(&self) -> &str {
            self.name
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Tree
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::Classification3
        }
        fn feature_columns(&self) -> &[String] {
            &self.columns
        }
        fn predict(&self, frame: &FeatureFrame, _: &CpuLease) -> Result<Vec<ExpertPrediction>> {
            let projected = super::super::project_expert_frame(frame, &self.columns, self.name)?;
            let column = projected.feature_column(0)?;
            let mean = column.values.iter().sum::<f64>() / column.len() as f64;
            let buy = 0.5 + 0.4 * mean.tanh();
            (0..frame.n_samples())
                .map(|_| {
                    ExpertPrediction::valid(
                        ExpertOutputKind::Classification3,
                        vec![0.0, buy, 1.0 - buy],
                    )
                })
                .collect()
        }
    }

    #[test]
    fn bound_expert_projections_are_deduplicated_and_match_general_causal_inference() {
        use crate::runtime::feature_input::ModelFeatureInputV1;
        use std::sync::Arc;
        let raw = Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns_with_options(
                neoethos_data::test_fixtures::canonical_test_timestamps(12),
                ["f1", "f2"]
                    .into_iter()
                    .enumerate()
                    .map(|(column, name)| {
                        FeatureColumnF64::new(
                            name,
                            (0..12)
                                .map(|row| (row + 1) as f64 * (column + 1) as f64)
                                .collect(),
                            vec![FeatureCellValidity::Valid; 12],
                        )
                        .unwrap()
                    })
                    .collect(),
                neoethos_data::FeatureBuildOptions::default(),
            )
            .unwrap(),
        );
        let fit = raw
            .fit_normalization(0..6, true, &neoethos_data::FeatureBuildControl::default())
            .unwrap();
        let normalized = Arc::new(raw.with_fitted_normalization(&fit).unwrap());
        let lease = one_worker_lease();
        for frame in [&raw, &normalized] {
            let input = ModelFeatureInputV1::from_training_frame(
                frame.provenance().bindings()[0].dataset_identity(),
                frame,
                Some(raw.timestamps[8]),
                2,
            )
            .unwrap();
            let ensemble = default_ensemble(outcome_with(vec![
                Box::new(ProjectedMeanClassifier {
                    name: "xgboost",
                    columns: vec!["f2".into(), "f1".into()],
                }),
                Box::new(ProjectedMeanClassifier {
                    name: "lightgbm",
                    columns: vec!["f2".into(), "f1".into()],
                }),
                Box::new(ProjectedMeanClassifier {
                    name: "logistic",
                    columns: vec!["f1".into()],
                }),
            ]))
            .unwrap()
            .bind_model_feature_input(input)
            .unwrap();
            let bound = ensemble.bind_model_features(frame).unwrap();
            assert_eq!(bound.projections.len(), 2);
            assert_eq!(bound.expert_projections, vec![Some(0), Some(0), Some(1)]);
            for end in 4..=12 {
                let direct = ensemble
                    .predict_with_roles(&frame.shared_row_window(end - 4..end).unwrap(), &lease)
                    .unwrap();
                assert_eq!(bound.last_row(end, 4, &lease).unwrap(), direct[3]);
            }
        }
    }

    #[test]
    fn bound_swarm_uses_exact_raw_prices_and_matches_the_real_loaded_adapter() {
        use crate::ensemble_inference::ExpertLoader;
        use crate::ensemble_inference::swarm_adapter::SwarmForecasterAdapterLoader;
        use crate::forecasting::swarm_impl::SwarmForecaster;
        use crate::runtime::feature_input::{MODEL_FEATURE_INPUT_FILE_V1, ModelFeatureInputV1};
        use std::sync::Arc;
        struct OwnedDirectory(std::path::PathBuf);
        impl Drop for OwnedDirectory {
            fn drop(&mut self) {
                if let Err(error) = std::fs::remove_dir_all(&self.0) {
                    eprintln!("owned bound-Swarm fixture cleanup failed: {error}");
                }
            }
        }
        let directory = OwnedDirectory(std::env::temp_dir().join(format!(
            "neoethos-bound-swarm-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        )));
        std::fs::create_dir(&directory.0).unwrap();
        let raw = Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns_with_options(
                neoethos_data::test_fixtures::canonical_test_timestamps(64),
                vec![
                    FeatureColumnF64::new(
                        "quant_close",
                        (0..64).map(|row| 1.1 + row as f64 * 0.0001).collect(),
                        vec![FeatureCellValidity::Valid; 64],
                    )
                    .unwrap(),
                    FeatureColumnF64::new(
                        "f1",
                        (0..64).map(|row| row as f64).collect(),
                        vec![FeatureCellValidity::Valid; 64],
                    )
                    .unwrap(),
                ],
                neoethos_data::FeatureBuildOptions::default(),
            )
            .unwrap(),
        );
        let fit = raw
            .fit_normalization(0..40, true, &neoethos_data::FeatureBuildControl::default())
            .unwrap();
        let normalized = Arc::new(raw.with_fitted_normalization(&fit).unwrap());
        let input = ModelFeatureInputV1::from_training_frame(
            raw.provenance().bindings()[0].dataset_identity(),
            &normalized,
            Some(raw.timestamps[44]),
            4,
        )
        .unwrap();
        neoethos_core::storage::json::write_bytes_atomic(
            &directory.0.join(MODEL_FEATURE_INPUT_FILE_V1),
            &input.to_json_bytes().unwrap(),
        )
        .unwrap();
        let artifact_dir = directory.0.join("swarm_forecaster");
        let lease = one_worker_lease();
        let mut model = SwarmForecaster::new(64.0);
        model.config.horizon = 2;
        model
            .fit_from_frame(&normalized.row_window(0, 40).unwrap(), "EURUSD", &lease)
            .unwrap();
        model.save(&artifact_dir).unwrap();
        let adapter = SwarmForecasterAdapterLoader.load(&artifact_dir).unwrap();
        let expected_raw = adapter
            .predict(&raw.shared_row_window(16..48).unwrap(), &lease)
            .unwrap();
        let ensemble = default_ensemble(outcome_with(vec![adapter]))
            .unwrap()
            .bind_model_feature_input(input.clone())
            .unwrap();
        let bound = ensemble.bind_model_features(&normalized).unwrap();
        assert_eq!(bound.projections.len(), 1);
        let projected = bound.projections[0].row_window(16..48).unwrap();
        assert_eq!(projected.names, vec!["quant_close"]);
        assert_eq!(
            projected.raw_model_column("quant_close").unwrap().values,
            raw.feature_column(0).unwrap().values[16..48]
        );
        let actual = bound.last_row(48, 32, &lease).unwrap();
        let direct = ensemble
            .predict_with_roles(&normalized.shared_row_window(16..48).unwrap(), &lease)
            .unwrap();
        assert_eq!(actual, direct[31]);
        assert!(actual.validity.is_valid());
        let raw_last = &expected_raw[31];
        let direct_adapter = ensemble.outcome.loaded[0]
            .predict(&projected, &lease)
            .unwrap();
        assert_eq!(direct_adapter[31].values, raw_last.values);
        assert_eq!(direct_adapter[31].validity, raw_last.validity);
        let mut columns = raw.project_columns(&[0, 1], 0..64).unwrap().columns.clone();
        for column in &mut columns {
            for value in &mut column.values[48..] {
                *value += 1000.0;
            }
        }
        let changed = Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns_with_options(
                raw.timestamps.clone(),
                columns,
                neoethos_data::FeatureBuildOptions::default(),
            )
            .unwrap(),
        );
        let changed = Arc::new(input.apply_to_raw_features(changed).unwrap());
        assert_eq!(
            actual,
            ensemble
                .bind_model_features(&changed)
                .unwrap()
                .last_row(48, 32, &lease)
                .unwrap()
        );
    }

    #[test]
    fn saved_model_input_is_independent_of_raw_search_or_a_different_search_fit() {
        use crate::runtime::feature_input::ModelFeatureInputV1;
        use std::sync::Arc;
        let raw = Arc::new(small_frame(12));
        let fit = raw
            .fit_normalization(0..6, true, &neoethos_data::FeatureBuildControl::default())
            .unwrap();
        let model = raw.with_fitted_normalization(&fit).unwrap();
        let input = ModelFeatureInputV1::from_training_frame(
            model.provenance().bindings()[0].dataset_identity(),
            &model,
            Some(raw.timestamps[8]),
            2,
        )
        .unwrap();
        let input = ModelFeatureInputV1::from_json_bytes(&input.to_json_bytes().unwrap()).unwrap();
        let ensemble = default_ensemble(outcome_with(vec![Box::new(WindowMeanClassifier)]))
            .unwrap()
            .bind_model_feature_input(input.clone())
            .unwrap();
        let search_fit = raw
            .fit_normalization(0..9, true, &neoethos_data::FeatureBuildControl::default())
            .unwrap();
        let normalized_search = Arc::new(raw.with_fitted_normalization(&search_fit).unwrap());
        assert!(ensemble.bind_model_features(&raw).is_err());
        assert!(ensemble.bind_model_features(&normalized_search).is_err());
        let lease = one_worker_lease();
        for search in [&raw, &normalized_search] {
            // Search values exist independently but never enter the model helper.
            assert_eq!(search.n_samples(), 12);
            let prepared = Arc::new(input.apply_to_raw_features(Arc::clone(&raw)).unwrap());
            assert_eq!(prepared.normalization_fitted_state(), Some(&fit));
            assert_eq!(
                prepared.feature_column(0).unwrap().values,
                model.feature_column(0).unwrap().values
            );
            let bound = ensemble.bind_model_features(&prepared).unwrap();
            let observed = bound.last_row(12, 4, &lease).unwrap();
            let direct = ensemble
                .predict_with_roles(&prepared.shared_row_window(8..12).unwrap(), &lease)
                .unwrap();
            assert_eq!(observed, direct[3]);
        }
    }

    #[test]
    fn borrowed_model_windows_ignore_future_perturbations_and_refuse_drift_on_binding() {
        use crate::runtime::feature_input::ModelFeatureInputV1;
        use std::sync::Arc;
        let raw = Arc::new(small_frame(12));
        let fit = raw
            .fit_normalization(0..6, true, &neoethos_data::FeatureBuildControl::default())
            .unwrap();
        let model = Arc::new(raw.with_fitted_normalization(&fit).unwrap());
        let input = ModelFeatureInputV1::from_training_frame(
            model.provenance().bindings()[0].dataset_identity(),
            &model,
            Some(raw.timestamps[8]),
            2,
        )
        .unwrap();
        let ensemble = default_ensemble(outcome_with(vec![Box::new(WindowMeanClassifier)]))
            .unwrap()
            .bind_model_feature_input(input.clone())
            .unwrap();
        let altered = Arc::new(
            neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
                raw.timestamps.clone(),
                vec![
                    FeatureColumnF64::new(
                        "f1",
                        (0..12)
                            .map(|r| if r < 8 { r as f64 + 1.0 } else { -10_000.0 })
                            .collect(),
                        vec![FeatureCellValidity::Valid; 12],
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        );
        let altered = Arc::new(input.apply_to_raw_features(altered).unwrap());
        let first = ensemble.bind_model_features(&model).unwrap();
        let second = ensemble.bind_model_features(&altered).unwrap();
        let lease = one_worker_lease();
        assert_eq!(
            first.last_row(8, 4, &lease).unwrap(),
            second.last_row(8, 4, &lease).unwrap()
        );
        assert_ne!(
            first.last_row(12, 4, &lease).unwrap(),
            second.last_row(12, 4, &lease).unwrap()
        );
        for (end, history) in [(0, 4), (13, 4), (8, 0)] {
            assert!(first.last_row(end, history, &lease).is_err());
        }
        let mut drift = model.shared_view().unwrap();
        drift.names[0] = "wrong_schema".into();
        assert!(ensemble.bind_model_features(&Arc::new(drift)).is_err());
        let legacy = default_ensemble(outcome_with(vec![Box::new(WindowMeanClassifier)])).unwrap();
        assert!(legacy.bind_model_features(&model).is_err());
    }
}
