//! Frozen Risky walk-forward survivors replayed concurrently on one account.
use super::*;
use neoethos_search::data_selection::CanonicalSearchArtifactScopeRefV1;

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
struct ArchivedGoalWindow {
    start_ms: i64,
    end_ms: i64,
}

#[derive(Deserialize)]
struct ArchivedGoalWindows {
    window_days: usize,
    windows: Vec<ArchivedGoalWindow>,
}

pub fn evaluate_walkforward_joint_risky(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let source = research.discovery_result();
        let (policy, mut evaluation, _) = research_recipe_for_mode(research, true)?;
        ensure!(
            source.effective_smc_gate_threshold.is_finite()
                && source.effective_smc_gate_threshold >= 0.0,
            "invalid final SMC threshold in the archived search"
        );
        evaluation.smc_gate_threshold = source.effective_smc_gate_threshold;
        let goal = validate_risky_recipe(&evaluation)?;
        let calibration = source
            .calibration_scope()?
            .context("missing saved calibration")?;
        let funnel = source
            .funnel_profile
            .as_ref()
            .context("missing archived funnel")?;
        let wf = funnel
            .walkforward_selection_cohort
            .as_ref()
            .context("missing walk-forward trials")?;
        let calibrated = funnel
            .selection_calibration_cohort
            .as_ref()
            .context("missing calibration trials")?;
        ensure!(
            wf.search_config_hash == source.search_config_hash
                && calibrated.search_config_hash == source.search_config_hash
                && wf.scope
                    == CanonicalSearchArtifactScopeRefV1::from_scope(&source.selection_scope)?
                && calibrated.scope == CanonicalSearchArtifactScopeRefV1::from_scope(calibration)?,
            "trial cohorts do not match the saved search and its exact scopes"
        );

        let survivors = wf
            .trials
            .iter()
            .filter(|trial| trial.passed)
            .collect::<Vec<_>>();
        ensure!(
            !survivors.is_empty()
                && survivors.len() <= 256
                && survivors.len() == calibrated.trials.len(),
            "joint replay requires the complete nonempty archived cohort (at most 256 lanes)"
        );
        let by_archive_index = calibrated
            .trials
            .iter()
            .map(|trial| (trial.candidate_archive_index, trial))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut indexes = BTreeSet::new();
        let mut genes = Vec::with_capacity(survivors.len());
        let mut members = Vec::with_capacity(survivors.len());
        let mut archived_windows: Option<Vec<ArchivedGoalWindow>> = None;
        for trial in survivors {
            let index = trial.candidate_archive_index;
            ensure!(
                trial.tested && indexes.insert(index),
                "duplicate or unmeasured walk-forward survivor"
            );
            let gene = source
                .candidates
                .get(index)
                .context("walk-forward archive index is out of range")?;
            trial.strategy_identity.validate_against(gene)?;
            let saved = by_archive_index
                .get(&index)
                .context("survivor is missing its calibration record")?;
            saved.strategy_identity.validate_against(gene)?;
            let windows = ArchivedGoalWindows::deserialize(
                saved
                    .risky_goal_windows
                    .as_ref()
                    .context("missing Risky goal windows")?,
            )?;
            ensure!(
                windows.window_days as f64 == goal.horizon_days
                    && !windows.windows.is_empty()
                    && windows
                        .windows
                        .windows(2)
                        .all(|w| w[0].start_ms < w[1].start_ms)
                    && archived_windows
                        .as_ref()
                        .is_none_or(|expected| *expected == windows.windows),
                "survivors do not share identical chronological Risky windows"
            );
            archived_windows.get_or_insert(windows.windows);
            ensure!(
                gene.stop_vol_mult == 0.0,
                "archived joint replay requires fixed entry brackets"
            );
            genes.push(gene.clone());
            members.push(json!({
                "candidate_archive_index": index,
                "strategy_identity": trial.strategy_identity,
                "walkforward_positive_folds": trial.positive_folds,
                "walkforward_splits": trial.reported_splits,
                "fixed_stop_pips": gene.sl_pips,
                "fixed_target_pips": gene.tp_pips,
            }));
        }

        let account_policy = JointBarAccountPolicyV1 {
            max_target_pips: genes.iter().map(|gene| gene.tp_pips).fold(0.0, f64::max),
            ..risky_account_policy(&evaluation)?
        };
        let cal_window = calibration.evaluated_window();
        let first_row = usize::try_from(cal_window.row_start())?;
        let last_row = usize::try_from(cal_window.row_end())?;
        let bars = neoethos_data::slice_ohlcv(input.ohlcv(), first_row, last_row, None);
        let features = project_features_to_effective(
            &input.features().row_window(first_row, last_row)?,
            &source.effective_feature_names,
        )?;
        let times = bars
            .timestamp
            .as_deref()
            .context("missing calibration timestamps")?;
        let votes = ArchivedGeneVotes::prepare(
            &genes,
            &features,
            &bars,
            &evaluation,
            policy.sealed_smc_gate_disabled()?,
            policy.sealed_adaptive_stops_policy()?,
        )?;
        // All lanes receive their archived signals across the entire calibration.
        // Preparing them once preserves the warm-up across later window starts.
        let active = vec![Some(0); times.len()];
        let tapes = (0..genes.len())
            .map(|index| Tape::routed(&votes, &active, [Some(index); STATES.len()]))
            .collect::<Result<Vec<_>>>()?;
        drop(votes);
        drop(features);
        let allocation = vec![1.0; times.len()];
        let mut results = Vec::new();
        let archived_windows = archived_windows.context("missing common calibration windows")?;
        for (index, window) in archived_windows.iter().enumerate() {
            let start = times
                .binary_search(&window.start_ms)
                .map_err(|_| anyhow::anyhow!("archived window start is absent from calibration"))?;
            let archived_last = times
                .binary_search(&window.end_ms)
                .map_err(|_| anyhow::anyhow!("archived window end is absent from calibration"))?;
            let deadline = window
                .start_ms
                .checked_add((goal.horizon_days * 86_400_000.0) as i64)
                .context("joint replay deadline overflow")?;
            ensure!(
                start < archived_last && window.end_ms <= deadline,
                "invalid archived window bounds"
            );
            let end = (times.partition_point(|&stamp| stamp <= deadline) + 1).min(times.len());
            let account_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
                CanonicalSearchWindowRoleV1::SelectionValidation,
                input,
                first_row + start..first_row + end,
            )?;
            let account = PreparedNettedCanonicalBarResearchV1::from_training_scope(
                input,
                &account_scope,
                research.execution_contract(),
                &evaluation,
            )?;
            let lanes = genes
                .iter()
                .zip(&tapes)
                .map(|(gene, tape)| JointBarLaneV1 {
                    expert_id: &gene.strategy_id,
                    tape: NettedBarDecisionTapeV1 {
                        signals: &tape.signals[start..end],
                        confidences: &tape.confidence[start..end],
                        sl_pips: &tape.stops[start..end],
                        tp_pips: &tape.targets[start..end],
                        ml_multipliers: &allocation[start..end],
                    },
                })
                .collect::<Vec<_>>();
            let result = account.evaluate_joint_risky_archived_brackets(
                &lanes,
                account_policy,
                None,
                true,
                goal,
            )?;
            println!(
                "joint_walkforward_window={} lanes={} closed_trades={} ending_balance={:.6}",
                index + 1,
                lanes.len(),
                result.closed_trades.len(),
                result.valuation.ending_realized_balance
            );
            results.push(json!({
                "window": index + 1,
                "archived_bounds": window,
                "coverage_last_bar_ms": times[end - 1],
                "shared_account": ledger(result, account.timestamps())?,
            }));
        }
        Ok(json!({
            "schema": "neoethos.walkforward-joint-risky.v1",
            "backend": "canonical_cpu",
            "combined_portfolio_evaluated": true,
            "promotion_eligible": false,
            "source_evidence_sha256": research.evidence_identity_sha256(),
            "training_and_selection_config_hash": source.search_config_hash,
            "calibration_scope_sha256": calibration.identity_sha256()?,
            "calibration_window": cal_window,
            "goal": goal,
            "effective_smc_gate_threshold": evaluation.smc_gate_threshold,
            "policy": account_policy,
            "allocation": "All archived lanes request the sealed confidence risk from common marked equity. The account scales admitted orders to available stop-risk and gross-notional caps. Equal-time capacity ties use strategy ID.",
            "members": members,
            "windows": results,
        }))
    })
}
