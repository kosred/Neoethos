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
//! ## The exit-side chain (audit #174 + #310, 2026-08-10)
//!
//! `predict_with_roles` runs a SECOND combiner in the same pass, over the
//! [`ExpertOutputKind::ExitDecision3`] experts, with its own role map
//! ([`super::exit_expert_role`]) and its own gate
//! ([`super::ExitOpinion::keep_fraction`]). Its result lands in
//! [`EnsembleDecision::exit`].
//!
//! Until this landed, `soft_voting.rs:230-233` dropped every non-Classification3
//! expert with a bare `continue`, so `exit_agent` trained on every run and ZERO
//! RL EXIT OUTPUT HAD EVER REACHED A TRADE. The fix is deliberately NOT "remove
//! the `continue`": `ExitDecision3` is shape-compatible with `Classification3`
//! but is an EXIT opinion (hold / close / scale), not an entry direction, and
//! averaging the two axes together would be worse than dropping one.
//!
//! Two invariants make this a SAFE addition:
//! - the exit chain writes ONLY `EnsembleDecision::exit`; `dir_probs`,
//!   `regime_gate` and `anomaly_scale` are byte-identical to what they were
//!   before it existed;
//! - a broken/mismatched exit expert is REFUSED (loudly, at `error`) and the
//!   exit opinion becomes `None` — it never turns into an `Err` from the whole
//!   combiner. That matters for money: the trader's response to an ensemble
//!   error is to fall back to GENE-ONLY sizing, i.e. without the ML shrink, so
//!   a new way to fail the ensemble would be a LOOSER change, not a safer one.
//!
//! ## Honest limitations
//!
//! - **Heterogeneous output kinds**: experts that emit
//!   `Forecast1` / `AnomalyScore` / `ActionValues3` cannot be
//!   averaged with Classification3 directly. SoftVotingEnsemble
//!   silently SKIPS them — they sit unused at the voting layer,
//!   counted in `experts_unused_for_voting()`. The MoE (D1.5+)
//!   is the right consumer for those signal types. `ExitDecision3`
//!   is no longer one of them — see above.
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
use polars::prelude::DataFrame;

use super::{
    EnsembleDecision, EnsemblePredictor, ExitOpinion, ExpertLoadOutcome, ExpertOutputKind,
    ExpertPrediction, ExpertRole, anomaly_scale_from_score, exit_expert_role, expert_role,
    regime_gate_from_posterior,
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
    /// Names of experts that reach NEITHER chain — wrong output kind for
    /// both, or excluded by the operator. Cached at construction so the
    /// aggregator doesn't pay the per-predict cost of re-checking. Surfaced
    /// via [`Self::experts_unused_for_voting`] for the chrome banner.
    ///
    /// **Audit #174/#310**: `ExitDecision3` experts are NO LONGER counted
    /// here. They were, and the banner therefore told the operator that
    /// `exit_agent` was unused — which was true then and would be a lie now.
    unused_for_voting: HashSet<String>,
    /// Names of loaded, non-excluded experts that feed the EXIT-side chain.
    exit_chain: HashSet<String>,
}

impl SoftVotingEnsemble {
    /// Build from a load outcome + config. Errors if NO loaded
    /// expert can contribute to the DIRECTION vote after applying both
    /// filters (output_kind == Classification3 AND name not in excluded).
    ///
    /// An empty exit chain is NOT an error: the exit side is an addition,
    /// and an operator with no trained `exit_agent` must keep getting exactly
    /// the ensemble he had before.
    pub fn new(outcome: ExpertLoadOutcome, config: SoftVotingEnsembleConfig) -> Result<Self> {
        let mut unused = HashSet::new();
        let mut exit_chain = HashSet::new();
        let mut votable = 0;
        for e in &outcome.loaded {
            let name = e.name();
            let excluded = config.excluded_names.contains(name);
            if excluded {
                // The operator's exclusion list drops an expert from BOTH
                // chains — it is a per-name "do not use this model", not a
                // per-axis one.
                unused.insert(name.to_string());
                continue;
            }
            match e.output_kind() {
                ExpertOutputKind::Classification3 => votable += 1,
                ExpertOutputKind::ExitDecision3 => {
                    exit_chain.insert(name.to_string());
                }
                // Forecast1 / AnomalyScore / ActionValues3 producers still
                // reach neither combiner.
                _ => {
                    unused.insert(name.to_string());
                }
            }
        }
        if votable == 0 {
            anyhow::bail!(
                "SoftVotingEnsemble requires at least one votable Classification3 expert in \
                 the load outcome AFTER applying the exclusion list. Loaded {} experts, all \
                 of which were either heterogeneous-output-kind or excluded by name. Unused: \
                 {:?}; exit-side only: {:?}",
                outcome.loaded.len(),
                unused,
                exit_chain
            );
        }
        Ok(Self {
            outcome,
            config,
            unused_for_voting: unused,
            exit_chain,
        })
    }

    /// Names of loaded experts that reach NEITHER combiner — they're held in
    /// the outcome (so the chrome can list them) but nothing consumes their
    /// predictions. The MoE will (D1.5+).
    pub fn experts_unused_for_voting(&self) -> Vec<&str> {
        self.unused_for_voting.iter().map(String::as_str).collect()
    }

    /// Names of loaded experts feeding the EXIT-side chain (audit #174/#310).
    /// Empty when no exit expert is trained/loaded.
    pub fn experts_in_exit_chain(&self) -> Vec<&str> {
        self.exit_chain.iter().map(String::as_str).collect()
    }

    /// Count of experts that actually participate in the DIRECTION vote.
    pub fn voting_expert_count(&self) -> usize {
        self.outcome.loaded.len() - self.unused_for_voting.len() - self.exit_chain.len()
    }

    /// Count of experts that participate in the EXIT-side chain.
    pub fn exit_expert_count(&self) -> usize {
        self.exit_chain.len()
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
    /// It ALSO runs the parallel EXIT-side chain (audit #174/#310) over the
    /// `ExitDecision3` experts and puts its verdict in
    /// [`EnsembleDecision::exit`]. That chain never touches `dir_probs` /
    /// `regime_gate` / `anomaly_scale`.
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
            // Only Classification3 experts vote on DIRECTION. `ExitDecision3`
            // is handled by `exit_opinions` below — it is a different axis, and
            // that is exactly why it is a second pass and not a relaxed filter
            // here. Forecast1 / AnomalyScore producers reach neither.
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

        // The parallel exit-side chain. Runs after the direction vote and
        // cannot influence it — see `exit_opinions`.
        let exit = self.exit_opinions(df, n_rows);

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
                exit: exit.as_ref().and_then(|e| e[row_idx]),
            });
        }
        Ok(out)
    }

    /// THE EXIT-SIDE COMBINER (audit #174 + #310, operator decision 2026-08-10
    /// "CONNECT IT, visible even in the back/forward test process").
    ///
    /// Weighted average of the loaded [`super::ExitRole::CloseVote`] experts'
    /// `[p_hold, p_neutral, p_close]` vectors, using the SAME
    /// `models.ensemble_voting.expert_weights` the direction vote uses — one
    /// config, one place, and `exit_agent: 0.0` is already a working "load it
    /// but do not let it speak" lever without a new knob existing anywhere.
    ///
    /// Returns `None` when the chain did not run at all, and `None` in an
    /// individual row when no expert produced a usable vector for it. Never an
    /// `Err`: see the module doc — turning an exit-expert fault into a whole-
    /// ensemble error would drop the trader to gene-only sizing, i.e. remove
    /// the ML shrink, which moves money the WRONG way. A refusal is logged at
    /// `error` with the expert named, so it is loud without being contagious.
    fn exit_opinions(&self, df: &DataFrame, n_rows: usize) -> Option<Vec<Option<ExitOpinion>>> {
        if self.exit_chain.is_empty() {
            return None;
        }
        let mut sums: Vec<[f32; 3]> = vec![[0.0; 3]; n_rows];
        let mut weight_totals: Vec<f32> = vec![0.0; n_rows];
        let mut voters: Vec<u16> = vec![0; n_rows];
        let mut contributors = 0usize;

        for expert in &self.outcome.loaded {
            let name = expert.name();
            if !self.exit_chain.contains(name) {
                continue;
            }
            // Same fail-loud contract as the direction map: a new
            // ExitDecision3 expert MUST be given a role, or it is refused by
            // name rather than quietly averaged in.
            if exit_expert_role(name).is_none() {
                tracing::error!(
                    target: "neoethos_models::ensemble",
                    expert = %name,
                    "exit chain: loaded ExitDecision3 expert has no role mapping; add it to \
                     `exit_expert_role`. REFUSED — it will not contribute to the exit opinion."
                );
                continue;
            }
            let weight = self.config.expert_weights.get(name).copied().unwrap_or(1.0);
            if weight <= 0.0 {
                continue;
            }
            let preds: Vec<ExpertPrediction> = match expert.predict(df) {
                Ok(preds) => preds,
                Err(error) => {
                    tracing::error!(
                        target: "neoethos_models::ensemble",
                        expert = %name,
                        %error,
                        "exit chain: expert REFUSED at predict (feature-column mismatch or a \
                         broken artifact). The exit opinion abstains for this frame; the \
                         direction vote is unaffected."
                    );
                    continue;
                }
            };
            if preds.len() != n_rows {
                tracing::error!(
                    target: "neoethos_models::ensemble",
                    expert = %name,
                    returned = preds.len(),
                    expected = n_rows,
                    "exit chain: expert returned the wrong number of predictions. REFUSED."
                );
                continue;
            }
            contributors += 1;
            for (row_idx, p) in preds.iter().enumerate() {
                if p.kind != ExpertOutputKind::ExitDecision3 || p.values.len() != 3 {
                    continue;
                }
                sums[row_idx][0] += weight * p.values[0];
                sums[row_idx][1] += weight * p.values[1];
                sums[row_idx][2] += weight * p.values[2];
                weight_totals[row_idx] += weight;
                voters[row_idx] = voters[row_idx].saturating_add(1);
            }
        }

        if contributors == 0 {
            return None;
        }
        Some(
            (0..n_rows)
                .map(|row_idx| {
                    let total = weight_totals[row_idx];
                    // No contributor for THIS row -> honest absence, not a
                    // neutral-looking vector a consumer could mistake for
                    // "hold". Ambiguous sentinels are defects.
                    if total <= 0.0 || voters[row_idx] == 0 {
                        return None;
                    }
                    Some(ExitOpinion {
                        close_probs: [
                            sums[row_idx][0] / total,
                            sums[row_idx][1] / total,
                            sums[row_idx][2] / total,
                        ],
                        voters: voters[row_idx],
                    })
                })
                .collect(),
        )
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

    /// In-test ExpertModel emitting a constant ExitDecision3 vector
    /// `[p_hold, p_neutral, p_close]` for every row — the exit-side twin of
    /// [`ConstantClassifier`].
    struct ConstantExitAgent {
        name: String,
        probs: [f32; 3],
    }

    impl ExpertModel for ConstantExitAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Exit
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::ExitDecision3
        }
        fn feature_columns(&self) -> &[String] {
            &[]
        }
        fn predict(&self, df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
            Ok((0..df.height())
                .map(|_| ExpertPrediction {
                    kind: ExpertOutputKind::ExitDecision3,
                    values: self.probs.to_vec(),
                })
                .collect())
        }
    }

    /// An exit expert that FAILS at predict — the stale-artifact /
    /// column-mismatch case. Must be contained, never contagious.
    struct BrokenExitAgent;
    impl ExpertModel for BrokenExitAgent {
        fn name(&self) -> &str {
            "exit_agent"
        }
        fn family(&self) -> ModelFamily {
            ModelFamily::Exit
        }
        fn output_kind(&self) -> ExpertOutputKind {
            ExpertOutputKind::ExitDecision3
        }
        fn feature_columns(&self) -> &[String] {
            &[]
        }
        fn predict(&self, _df: &DataFrame) -> Result<Vec<ExpertPrediction>> {
            anyhow::bail!("exit-agent prediction feature-column mismatch: expected [..], got [..]")
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
        use crate::ensemble_inference::exit_expert_role;
        for name in crate::ensemble_inference::bootstrap::DEFAULT_BOOTSTRAP_EXPERT_NAMES {
            // Every bootstrap expert must carry a role in ONE of the two maps.
            // Audit #174/#310 added the exit-side map; `exit_agent` lives there
            // and deliberately has NO direction role — that separation is the
            // whole point, so this asserts coverage across both, not just one.
            assert!(
                expert_role(name).is_some() || exit_expert_role(name).is_some(),
                "bootstrap expert '{name}' has no role in expert_role() OR \
                 exit_expert_role(); the combiners would refuse it in production"
            );
        }
        assert_eq!(expert_role("exit_agent"), None, "exit_agent must never vote on direction");
        assert_eq!(expert_role("hmm_regime"), Some(ExpertRole::RegimeGate));
        assert_eq!(expert_role("isolation_forest"), Some(ExpertRole::AnomalyScale));
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
        let ens = default_ensemble(outcome).expect("ok");
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
        let ens = default_ensemble(outcome).expect("ok");
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
        let ens = default_ensemble(outcome).expect("ok");
        match ens.predict_with_roles(&small_df(1)) {
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
        let decisions = ens.predict_with_roles(&small_df(3)).expect("roles");
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
        let decisions = ens.predict_with_roles(&small_df(2)).expect("roles");
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
        let decisions = ens.predict_with_roles(&small_df(1)).expect("roles");
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
        let decisions = ens.predict_with_roles(&small_df(1)).expect("roles");
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
        let d = ens.predict_with_roles(&small_df(1)).expect("roles")[0];
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
        let d = ens.predict_with_roles(&small_df(1)).expect("roles")[0];
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
        let d = ens.predict_with_roles(&small_df(1)).expect("roles")[0];
        assert!(
            (d.dir_probs[1] - 0.7).abs() < 1e-6,
            "a zero-weight expert must not pull the vote, got {:?}",
            d.dir_probs
        );
    }

    // -- The EXIT-side chain (audit #174 + #310) ----------------------
    //
    // THE regression these pin: `soft_voting.rs:230-233` dropped every
    // non-Classification3 expert with a bare `continue`, so ZERO RL EXIT
    // OUTPUT HAD EVER REACHED A TRADE while `exit_agent` trained on every run.

    /// THE test for #174 + #310. An ExitDecision3 expert must reach the exit
    /// chain — and must NOT reach the direction vote, which is the reason the
    /// `continue` was not simply deleted.
    #[test]
    fn exit_expert_reaches_the_exit_chain_and_never_the_direction_vote() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantExitAgent {
                name: "exit_agent".into(),
                probs: [0.2, 0.1, 0.7], // strong CLOSE
            }),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        assert_eq!(ens.exit_expert_count(), 1, "exit expert must be in the chain");
        assert_eq!(ens.voting_expert_count(), 1, "exit expert must not be a direction voter");
        assert!(
            !ens.experts_unused_for_voting().contains(&"exit_agent"),
            "a wired exit expert must not be reported as unused: {:?}",
            ens.experts_unused_for_voting()
        );
        assert!(ens.experts_in_exit_chain().contains(&"exit_agent"));

        let d = ens.predict_with_roles(&small_df(3)).expect("roles");
        assert_eq!(d.len(), 3);
        for row in &d {
            // Direction vote is EXACTLY the sole classifier — the exit vector
            // [0.2, 0.1, 0.7] must not have moved it toward "sell".
            assert!((row.dir_probs[0] - 0.1).abs() < 1e-6, "{row:?}");
            assert!((row.dir_probs[1] - 0.7).abs() < 1e-6, "{row:?}");
            assert!((row.dir_probs[2] - 0.2).abs() < 1e-6, "{row:?}");
            let exit = row.exit.expect("the exit chain must produce an opinion");
            assert_eq!(exit.voters, 1);
            assert!((exit.close_pressure() - 0.7).abs() < 1e-6, "{exit:?}");
            assert!((exit.keep_fraction() - 0.3).abs() < 1e-6, "{exit:?}");
        }
    }

    /// No exit expert loaded ⇒ honest absence, not a neutral-looking vector a
    /// consumer could read as "hold". Ambiguous sentinels are defects.
    #[test]
    fn no_exit_expert_means_none_not_a_neutral_opinion() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "xgboost".into(),
            probs: [0.1, 0.7, 0.2],
        })]);
        let ens = default_ensemble(outcome).expect("ok");
        assert_eq!(ens.exit_expert_count(), 0);
        let d = ens.predict_with_roles(&small_df(2)).expect("roles");
        for row in &d {
            assert!(row.exit.is_none(), "absence must be None, got {row:?}");
        }
    }

    /// A broken/mis-columned exit expert must NOT fail the whole combiner.
    /// If it did, the trader would fall back to gene-only sizing — i.e. drop
    /// the ML shrink — which moves money the WRONG way. Containment is the
    /// safe behaviour here, and it is loud (an `error!` names the expert).
    #[test]
    fn a_broken_exit_expert_abstains_and_leaves_the_direction_vote_intact() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(BrokenExitAgent),
        ]);
        let ens = default_ensemble(outcome).expect("ok");
        let d = ens
            .predict_with_roles(&small_df(2))
            .expect("a broken EXIT expert must never fail the whole combiner");
        for row in &d {
            assert!(row.exit.is_none(), "refused exit expert must abstain: {row:?}");
            assert!((row.dir_probs[1] - 0.7).abs() < 1e-6, "direction damaged: {row:?}");
            assert_eq!(row.regime_gate, 1.0);
            assert_eq!(row.anomaly_scale, 1.0);
        }
    }

    /// The operator's ONE config reaches the exit chain too: the same
    /// `expert_weights` map weights exit voters, and 0.0 silences one — no new
    /// knob exists or is needed.
    #[test]
    fn operator_weights_and_exclusions_reach_the_exit_chain() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantExitAgent {
                name: "exit_agent".into(),
                probs: [0.1, 0.1, 0.8],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.expert_weights.insert("exit_agent".into(), 0.0);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let d = ens.predict_with_roles(&small_df(1)).expect("roles")[0];
        assert!(d.exit.is_none(), "weight 0 must silence the exit voter, got {d:?}");

        // And the exclusion list drops it from the chain entirely.
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantExitAgent {
                name: "exit_agent".into(),
                probs: [0.1, 0.1, 0.8],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.excluded_names.insert("exit_agent".to_string());
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        assert_eq!(ens.exit_expert_count(), 0);
        assert!(ens.experts_unused_for_voting().contains(&"exit_agent"));
        assert!(
            ens.predict_with_roles(&small_df(1)).expect("roles")[0]
                .exit
                .is_none()
        );
    }

    /// Two exit voters average, weighted, exactly like the direction pool.
    #[test]
    fn two_exit_voters_average_with_weights() {
        let outcome = outcome_with(vec![
            Box::new(ConstantClassifier {
                name: "xgboost".into(),
                probs: [0.1, 0.7, 0.2],
            }),
            Box::new(ConstantExitAgent {
                name: "exit_agent".into(),
                probs: [0.8, 0.1, 0.1],
            }),
            Box::new(ConstantExitAgent {
                name: "exit_agent_02".into(), // replica -> same canonical role
                probs: [0.2, 0.2, 0.6],
            }),
        ]);
        let mut cfg = SoftVotingEnsembleConfig::default();
        cfg.expert_weights.insert("exit_agent".into(), 3.0);
        cfg.expert_weights.insert("exit_agent_02".into(), 1.0);
        let ens = SoftVotingEnsemble::new(outcome, cfg).expect("ok");
        let d = ens.predict_with_roles(&small_df(1)).expect("roles")[0];
        let exit = d.exit.expect("two exit voters must produce an opinion");
        assert_eq!(exit.voters, 2);
        // (3*0.8 + 1*0.2)/4 = 0.65 ; (3*0.1+1*0.2)/4 = 0.125 ; (3*0.1+1*0.6)/4 = 0.225
        assert!((exit.close_probs[0] - 0.65).abs() < 1e-5, "{exit:?}");
        assert!((exit.close_probs[1] - 0.125).abs() < 1e-5, "{exit:?}");
        assert!((exit.close_probs[2] - 0.225).abs() < 1e-5, "{exit:?}");
        assert!((exit.keep_fraction() - 0.775).abs() < 1e-5, "{exit:?}");
    }

    /// The exit gate is bounded ABOVE by 1.0 for every possible input, which is
    /// what makes wiring this chain a SAFER change: a consumer multiplying its
    /// held size by `keep_fraction` can only ever hold less, never more.
    #[test]
    fn the_exit_gate_can_never_increase_exposure() {
        for close in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let opinion = ExitOpinion {
                close_probs: [1.0 - close, 0.0, close],
                voters: 1,
            };
            let keep = opinion.keep_fraction();
            assert!(
                (0.0..=1.0).contains(&keep),
                "keep_fraction {keep} out of [0,1] for p_close {close}"
            );
            assert!(keep <= 1.0, "the exit gate must never exceed 1.0");
        }
    }

    /// The exit-side role map must cover every exit expert the bootstrap loads,
    /// or the combiner refuses it by name at runtime.
    #[test]
    fn exit_role_map_covers_the_bootstrap_exit_experts() {
        use crate::ensemble_inference::{ExitRole, exit_expert_role};
        assert_eq!(exit_expert_role("exit_agent"), Some(ExitRole::CloseVote));
        assert_eq!(exit_expert_role("exit_agent_07"), Some(ExitRole::CloseVote));
        assert_eq!(exit_expert_role("xgboost"), None);
        assert_eq!(exit_expert_role("not_a_real_model"), None);
        // And the two maps must not overlap — one expert, one axis.
        for name in crate::ensemble_inference::bootstrap::DEFAULT_BOOTSTRAP_EXPERT_NAMES {
            assert!(
                expert_role(name).is_some() != exit_expert_role(name).is_some(),
                "'{name}' must belong to EXACTLY ONE chain (direction or exit)"
            );
        }
    }

    #[test]
    fn empty_dataframe_returns_empty_predictions() {
        let outcome = outcome_with(vec![Box::new(ConstantClassifier {
            name: "xgboost".into(),
            probs: [0.2, 0.6, 0.2],
        })]);
        let ens = default_ensemble(outcome).expect("ok");
        let decisions = ens.predict_with_roles(&small_df(0)).expect("roles");
        assert!(decisions.is_empty());
    }
}
