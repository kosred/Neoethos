//! `SoftVotingEnsemble` — first concrete [`super::EnsemblePredictor`].
//!
//! Phase D1.3. This aggregator runs every loaded expert's
//! [`super::ExpertModel::predict`] in turn and combines their
//! Classification3 outputs by **weighted-average** of the
//! `[p_neutral, p_buy, p_sell]` vectors (canonical order — see
//! `base.rs` lines 128-135). The result is one
//! `[p_neutral, p_buy, p_sell]` per input row, ready for the
//! producer's `dispatch_auto_trade_signal` gate chain.
//!
//! ## Why "soft voting" and not "MoE"
//!
//! Following the user's research-backed direction (2026-05-17
//! correspondence + the 2025 ensemble-learning survey):
//!
//! - All experts see the same input features. Diversity comes from
//!   their distinct architectures and learning algorithms, NOT
//!   from artificial feature restrictions.
//! - Each expert produces a Classification3 vote.
//! - Soft voting averages those votes — equivalent to assuming
//!   every expert is equally trustworthy. This is the simplest
//!   diversity-aware combiner and ships TODAY against whatever
//!   experts are already trained.
//! - The MoE gate (D1.5+) replaces this layer with a learnt
//!   gating network that decides who-to-trust-when. SoftVoting
//!   stays as a fallback when the MoE artifact isn't on disk.
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
//!   prediction layer, SoftVoting always votes. If you need
//!   "predict only when confident enough", set
//!   [`SoftVotingEnsembleConfig::abstain_below_confidence`].

use std::collections::HashSet;

use anyhow::Result;
use ndarray::Array2;
use polars::prelude::DataFrame;

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
    pub expert_weights: std::collections::HashMap<String, f32>,
    /// Optional minimum-confidence abstention threshold in `[0, 1]`.
    /// When set, predictions whose max-class probability is below
    /// this threshold are flattened to a uniform `[1/3, 1/3, 1/3]`
    /// so the downstream gate chain interprets them as "no signal"
    /// (the producer's [`super::ExpertOutputKind`] mapping treats
    /// uniform outputs as Flat). When `None`, every prediction
    /// passes through verbatim.
    pub abstain_below_confidence: Option<f32>,
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
    pub anomaly_lo: f32,
    /// Anomaly-score upper knee — at/above this the anomaly scale hard-vetoes
    /// to 0.0. Default 0.9 (matches the trained ~0.95-quantile threshold).
    pub anomaly_hi: f32,
}

impl Default for SoftVotingEnsembleConfig {
    fn default() -> Self {
        Self {
            expert_weights: std::collections::HashMap::new(),
            abstain_below_confidence: None,
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
        })
    }

    /// Convenience: build with default config.
    pub fn with_default_config(outcome: ExpertLoadOutcome) -> Result<Self> {
        Self::new(outcome, SoftVotingEnsembleConfig::default())
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

    /// v0.5 ML-integration Stage 2 — role-aware combiner.
    ///
    /// Unlike [`EnsemblePredictor::predict`] (a flat average of EVERY
    /// Classification3 expert, which lets `hmm_regime`/`isolation_forest`
    /// pollute the direction vote), this partitions the loaded experts by
    /// [`ExpertRole`] and returns one [`EnsembleDecision`] per row:
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
    pub fn predict_with_roles(&self, df: &DataFrame) -> Result<Vec<EnsembleDecision>> {
        let n_rows = df.height();
        if n_rows == 0 {
            return Ok(Vec::new());
        }

        // Direction vote accumulator (weighted, per row).
        let mut dir_sums: Vec<[f32; 3]> = vec![[0.0; 3]; n_rows];
        let mut dir_weight_totals: Vec<f32> = vec![0.0; n_rows];
        let mut direction_voters = 0usize;
        // Optional per-row regime posterior + anomaly score.
        let mut regime_posterior: Option<Vec<[f32; 3]>> = None;
        let mut anomaly_scores: Option<Vec<f32>> = None;

        for expert in &self.outcome.loaded {
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

            let preds: Vec<ExpertPrediction> = expert.predict(df)?;
            if preds.len() != n_rows {
                anyhow::bail!(
                    "expert '{}' returned {} predictions for a {}-row DataFrame",
                    name,
                    preds.len(),
                    n_rows
                );
            }

            match role {
                ExpertRole::Direction | ExpertRole::DirectionalConfirm => {
                    let weight = self
                        .config
                        .expert_weights
                        .get(name)
                        .copied()
                        .unwrap_or(1.0);
                    if weight <= 0.0 {
                        continue;
                    }
                    direction_voters += 1;
                    for (row_idx, p) in preds.iter().enumerate() {
                        if p.kind != ExpertOutputKind::Classification3 || p.values.len() != 3 {
                            continue;
                        }
                        dir_sums[row_idx][0] += weight * p.values[0];
                        dir_sums[row_idx][1] += weight * p.values[1];
                        dir_sums[row_idx][2] += weight * p.values[2];
                        dir_weight_totals[row_idx] += weight;
                    }
                }
                ExpertRole::RegimeGate => {
                    // hmm_regime posterior: col0=P(range), col1=P(buy), col2=P(sell).
                    let mut post = vec![[1.0 / 3.0_f32; 3]; n_rows];
                    for (row_idx, p) in preds.iter().enumerate() {
                        if p.values.len() == 3 {
                            post[row_idx] = [p.values[0], p.values[1], p.values[2]];
                        }
                    }
                    regime_posterior = Some(post);
                }
                ExpertRole::AnomalyScale => {
                    // isolation_forest emits [anomaly, (1-a)/2, (1-a)/2] -> col0
                    // is the raw anomaly score (no retrain / new artifact needed).
                    let mut scores = vec![0.0_f32; n_rows];
                    for (row_idx, p) in preds.iter().enumerate() {
                        if !p.values.is_empty() {
                            scores[row_idx] = p.values[0];
                        }
                    }
                    anomaly_scores = Some(scores);
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
            let total = dir_weight_totals[row_idx];
            let dir_probs = if total <= 0.0 {
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            } else {
                [
                    dir_sums[row_idx][0] / total,
                    dir_sums[row_idx][1] / total,
                    dir_sums[row_idx][2] / total,
                ]
            };
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
            });
        }
        Ok(out)
    }

    /// Apply the per-row probability vector through the optional
    /// abstention threshold. Returns a (possibly flattened) vector.
    fn maybe_abstain(&self, row: [f32; 3]) -> [f32; 3] {
        let Some(threshold) = self.config.abstain_below_confidence else {
            return row;
        };
        let max_p = row[0].max(row[1]).max(row[2]);
        if max_p < threshold {
            // Flat / "no signal" → uniform.
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        } else {
            row
        }
    }
}

impl EnsemblePredictor for SoftVotingEnsemble {
    fn predict(&self, df: &DataFrame) -> Result<Array2<f32>> {
        let n_rows = df.height();
        if n_rows == 0 {
            return Ok(Array2::<f32>::zeros((0, 3)));
        }
        // Per-row accumulator: sum of (weight × probabilities) and
        // total weight (for normalisation if some experts emit NaN
        // or fail mid-batch).
        let mut sums: Vec<[f32; 3]> = vec![[0.0; 3]; n_rows];
        let mut weight_totals: Vec<f32> = vec![0.0; n_rows];

        for expert in &self.outcome.loaded {
            if expert.output_kind() != ExpertOutputKind::Classification3 {
                continue;
            }
            // Skip strategy-discovery / operator-excluded experts.
            // They're in the load outcome (so the chrome can render
            // them) but don't contribute to the direction vote.
            if self.config.excluded_names.contains(expert.name()) {
                continue;
            }
            let weight = self
                .config
                .expert_weights
                .get(expert.name())
                .copied()
                .unwrap_or(1.0);
            if weight <= 0.0 {
                continue;
            }
            let preds: Vec<ExpertPrediction> = expert.predict(df)?;
            if preds.len() != n_rows {
                anyhow::bail!(
                    "expert '{}' returned {} predictions for a {}-row DataFrame",
                    expert.name(),
                    preds.len(),
                    n_rows
                );
            }
            for (row_idx, p) in preds.iter().enumerate() {
                if p.kind != ExpertOutputKind::Classification3 || p.values.len() != 3 {
                    // Skip — defensive; output_kind says Classification3
                    // but the prediction itself doesn't match. This
                    // is a programmer error in the adapter; the
                    // tree adapters' validator should have caught it.
                    continue;
                }
                sums[row_idx][0] += weight * p.values[0];
                sums[row_idx][1] += weight * p.values[1];
                sums[row_idx][2] += weight * p.values[2];
                weight_totals[row_idx] += weight;
            }
        }

        // Normalise + apply abstention.
        let mut out = Array2::<f32>::zeros((n_rows, 3));
        for row_idx in 0..n_rows {
            let total = weight_totals[row_idx];
            if total <= 0.0 {
                // No expert contributed — flat output. This can
                // happen if every expert errored or all weights were
                // zero. The producer treats uniform output as Flat
                // (no signal), which is the correct safe default.
                out[(row_idx, 0)] = 1.0 / 3.0;
                out[(row_idx, 1)] = 1.0 / 3.0;
                out[(row_idx, 2)] = 1.0 / 3.0;
                continue;
            }
            let row = [
                sums[row_idx][0] / total,
                sums[row_idx][1] / total,
                sums[row_idx][2] / total,
            ];
            let row = self.maybe_abstain(row);
            out[(row_idx, 0)] = row[0];
            out[(row_idx, 1)] = row[1];
            out[(row_idx, 2)] = row[2];
        }
        Ok(out)
    }

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
    use polars::prelude::*;

    /// In-test ExpertModel that emits a constant Classification3
    /// prediction for every row.
    struct ConstantClassifier {
        name: String,
        probs: [f32; 3],
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
        fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
            let n = df.height();
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(ExpertPrediction {
                    kind: ExpertOutputKind::Classification3,
                    values: self.probs.to_vec(),
                });
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
        fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
            Ok((0..df.height())
                .map(|_| ExpertPrediction {
                    kind: ExpertOutputKind::Forecast1,
                    values: vec![0.5],
                })
                .collect())
        }
    }

    fn outcome_with(experts: Vec<Box<dyn ExpertModel>>) -> ExpertLoadOutcome {
        ExpertLoadOutcome {
            loaded: experts,
            missing: vec![],
            degraded: vec![],
        }
    }

    fn small_df(rows: usize) -> DataFrame {
        let v: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        df!("f1" => v).expect("df")
    }

    // -- Construction invariants ---------------------------------------

    #[test]
    fn new_rejects_empty_classification3_set() {
        let outcome = outcome_with(vec![Box::new(ForecastEmitter)]);
        // Can't expect_err — SoftVotingEnsemble holds Box<dyn ExpertModel>
        // which doesn't implement Debug. Match on the result instead.
        match SoftVotingEnsemble::with_default_config(outcome) {
            Ok(_) => panic!("must reject empty Classification3 set"),
            Err(err) => assert!(err.to_string().contains("Classification3")),
        }
    }

    #[test]
    fn new_accepts_when_at_least_one_classification3() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "a".into(),
                probs: [0.2, 0.6, 0.2],
            }),
            Box::new(ForecastEmitter),
        ]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
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
        assert_eq!(expert_role("isolation_forest"), Some(ExpertRole::AnomalyScale));
        assert_eq!(expert_role("dqn"), Some(ExpertRole::DirectionalConfirm));
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
        assert!((mid - 0.5).abs() < 1e-6, "mid ramp should be 0.5, got {mid}");
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
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        let decisions = ens.predict_with_roles(&small_df(2)).expect("roles");
        assert_eq!(decisions.len(), 2);
        for d in &decisions {
            // dir_probs == the SOLE directional voter, gates removed.
            assert!((d.dir_probs[1] - 0.8).abs() < 1e-6, "dir vote polluted: {d:?}");
            assert!(d.regime_gate > 0.8, "agreeing regime gate: {d:?}");
            assert_eq!(d.anomaly_scale, 1.0, "low anomaly -> no penalty: {d:?}");
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
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        match ens.predict_with_roles(&small_df(1)) {
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
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        match ens.predict_with_roles(&small_df(1)) {
            Ok(_) => panic!("must fail loud on an unmapped expert"),
            Err(err) => assert!(err.to_string().contains("no role mapping")),
        }
    }

    // -- Vote arithmetic ----------------------------------------------

    #[test]
    fn single_expert_pass_through() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "a".into(),
            probs: [0.1, 0.7, 0.2],
        })]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        let probs = ens.predict(&small_df(3)).expect("predict");
        assert_eq!(probs.shape(), &[3, 3]);
        for row in probs.outer_iter() {
            assert!((row[0] - 0.1).abs() < 1e-6);
            assert!((row[1] - 0.7).abs() < 1e-6);
            assert!((row[2] - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn two_experts_equal_weight_averaged() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "a".into(),
                probs: [0.8, 0.1, 0.1],
            }),
            Box::new(ConstantClassifier {
                name: "b".into(),
                probs: [0.2, 0.6, 0.2],
            }),
        ]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        let probs = ens.predict(&small_df(2)).expect("predict");
        // Average of (0.8,0.1,0.1) + (0.2,0.6,0.2) = (0.5,0.35,0.15)
        for row in probs.outer_iter() {
            assert!((row[0] - 0.5).abs() < 1e-5);
            assert!((row[1] - 0.35).abs() < 1e-5);
            assert!((row[2] - 0.15).abs() < 1e-5);
        }
    }

    #[test]
    fn per_expert_weights_bias_average() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "strong".into(),
                probs: [0.8, 0.1, 0.1],
            }),
            Box::new(ConstantClassifier {
                name: "weak".into(),
                probs: [0.2, 0.6, 0.2],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.expert_weights.insert("strong".into(), 3.0);
        cfg.expert_weights.insert("weak".into(), 1.0);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let probs = ens.predict(&small_df(1)).expect("predict");
        // Weighted: (3*0.8 + 1*0.2)/4, (3*0.1+1*0.6)/4, (3*0.1+1*0.2)/4
        //         = (2.6/4, 0.9/4, 0.5/4)
        //         = (0.65, 0.225, 0.125)
        let row = probs.row(0);
        assert!((row[0] - 0.65).abs() < 1e-5);
        assert!((row[1] - 0.225).abs() < 1e-5);
        assert!((row[2] - 0.125).abs() < 1e-5);
    }

    #[test]
    fn forecast_experts_are_skipped() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "a".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ForecastEmitter),
        ]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        let probs = ens.predict(&small_df(1)).expect("predict");
        // ForecastEmitter must not have contributed.
        let row = probs.row(0);
        assert!((row[0] - 0.1).abs() < 1e-6);
        assert!((row[1] - 0.7).abs() < 1e-6);
        assert!((row[2] - 0.2).abs() < 1e-6);
    }

    // -- Abstention ---------------------------------------------------

    #[test]
    fn abstain_threshold_flattens_low_confidence() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "a".into(),
            probs: [0.4, 0.35, 0.25],
        })]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.abstain_below_confidence = Some(0.5);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let probs = ens.predict(&small_df(1)).expect("predict");
        let row = probs.row(0);
        // Max=0.4 < 0.5 → flatten.
        assert!((row[0] - 1.0 / 3.0).abs() < 1e-5);
        assert!((row[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((row[2] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn abstain_threshold_passes_high_confidence() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "a".into(),
            probs: [0.7, 0.2, 0.1],
        })]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.abstain_below_confidence = Some(0.5);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let probs = ens.predict(&small_df(1)).expect("predict");
        let row = probs.row(0);
        // Max=0.7 >= 0.5 → pass through.
        assert!((row[0] - 0.7).abs() < 1e-6);
        assert!((row[1] - 0.2).abs() < 1e-6);
        assert!((row[2] - 0.1).abs() < 1e-6);
    }

    // -- Load outcome surfacing --------------------------------------

    #[test]
    fn load_outcome_round_trips_through_trait() {
        let outcome = ExpertLoadOutcome {
            loaded: vec![Box::new(ConstantClassifier {
                name: "a".into(),
                probs: [0.2, 0.6, 0.2],
            })],
            missing: vec!["xgboost".into(), "transformer".into()],
            degraded: vec![],
        };
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        let lo = ens.load_outcome();
        assert_eq!(lo.loaded_count(), 1);
        assert_eq!(lo.missing_count(), 2);
        assert_eq!(lo.loaded_names(), vec!["a"]);
    }

    #[test]
    fn default_config_excludes_strategy_discoverers() {
        // genetic + neuro_evo are excluded by default.
        let cfg = SoftVotingEnsembleConfig::default();
        assert!(cfg.excluded_names.contains("genetic"));
        assert!(cfg.excluded_names.contains("neuro_evo"));
        assert_eq!(cfg.excluded_names.len(), 2);
    }

    #[test]
    fn genetic_expert_is_skipped_at_voting_layer() {
        // Construct an outcome with a regular voter + a "genetic"
        // expert. With the default exclusion list, the genetic
        // expert must not contribute to the average even though
        // its output_kind is Classification3.
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "regular".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantClassifier {
                name: "genetic".into(),
                probs: [0.8, 0.1, 0.1],
            }),
        ]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        // 2 loaded but only 1 votes — genetic excluded.
        assert_eq!(ens.voting_expert_count(), 1);
        assert!(ens.experts_unused_for_voting().contains(&"genetic"));
        // The output must reflect ONLY the regular expert, not an
        // average of the two.
        let probs = ens.predict(&small_df(1)).expect("predict");
        let row = probs.row(0);
        assert!((row[0] - 0.1).abs() < 1e-6);
        assert!((row[1] - 0.7).abs() < 1e-6);
        assert!((row[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn neuro_evo_expert_is_also_skipped_by_default() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "voter".into(),
                probs: [0.2, 0.6, 0.2],
            }),
            Box::new(ConstantClassifier {
                name: "neuro_evo".into(),
                probs: [0.9, 0.05, 0.05],
            }),
        ]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        assert_eq!(ens.voting_expert_count(), 1);
        let probs = ens.predict(&small_df(1)).expect("predict");
        let row = probs.row(0);
        assert!(
            (row[1] - 0.6).abs() < 1e-6,
            "neuro_evo must not pull p_neutral toward 0.9"
        );
    }

    #[test]
    fn operator_can_clear_exclusion_to_include_strategy_discoverers() {
        // Operator override: someone WANTS to vote on genetic
        // outputs (e.g. for sanity-check during validation). They
        // clear the exclusion list and genetic participates.
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "genetic".into(),
            probs: [0.1, 0.7, 0.2],
        })]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.excluded_names.clear();
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        assert_eq!(ens.voting_expert_count(), 1);
        let probs = ens.predict(&small_df(1)).expect("predict");
        let row = probs.row(0);
        assert!((row[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn empty_dataframe_returns_empty_predictions() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "a".into(),
            probs: [0.2, 0.6, 0.2],
        })]);
        let ens = SoftVotingEnsemble::with_default_config(outcome).expect("ok");
        let probs = ens.predict(&small_df(0)).expect("predict");
        assert_eq!(probs.shape(), &[0, 3]);
    }
}
