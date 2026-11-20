//! Frozen Risky walk-forward survivors replayed concurrently on one account.
use super::*;
use neoethos_search::data_selection::CanonicalSearchArtifactScopeRefV1;
use neoethos_search::scoring::weekly_cadence::complete_week_entry_counts_v1;
use std::collections::BTreeMap;
use std::ops::Range;

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

struct FrozenTrial<'a> {
    gene: &'a Gene,
    member: Value,
    goal_windows: Option<&'a Value>,
}

fn frozen_walkforward_trials(
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Vec<FrozenTrial<'_>>> {
    let source = research.discovery_result();
    let funnel = source
        .funnel_profile
        .as_ref()
        .context("missing archived funnel")?;
    let Some(wf) = funnel.walkforward_selection_cohort.as_ref() else {
        ensure!(
            source.quality_metrics.is_empty(),
            "missing walk-forward trials"
        );
        return Ok(Vec::new());
    };
    ensure!(
        wf.search_config_hash == source.search_config_hash
            && wf.scope == CanonicalSearchArtifactScopeRefV1::from_scope(&source.selection_scope)?,
        "walk-forward trials do not match the saved search selection"
    );
    let survivors: Vec<_> = wf.trials.iter().filter(|trial| trial.passed).collect();
    if survivors.is_empty() {
        return Ok(Vec::new());
    }
    ensure!(
        survivors.len() <= 256,
        "joint replay supports at most 256 frozen lanes"
    );
    let calibrated = match (
        source.calibration_scope()?,
        &funnel.selection_calibration_cohort,
    ) {
        (Some(scope), Some(cohort)) => {
            ensure!(
                survivors.len() == cohort.trials.len()
                    && cohort.search_config_hash == source.search_config_hash
                    && cohort.scope == CanonicalSearchArtifactScopeRefV1::from_scope(scope)?,
                "calibration must match the complete frozen walk-forward cohort"
            );
            Some(
                cohort
                    .trials
                    .iter()
                    .map(|trial| (trial.candidate_archive_index, trial))
                    .collect::<BTreeMap<_, _>>(),
            )
        }
        (None, None) => None,
        _ => anyhow::bail!("calibration trials and scope disagree"),
    };
    let mut indexes = BTreeSet::new();
    survivors
        .into_iter()
        .map(|trial| {
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
            let goal_windows = if let Some(calibrated) = &calibrated {
                let saved = calibrated
                    .get(&index)
                    .context("survivor is missing its calibration record")?;
                saved.strategy_identity.validate_against(gene)?;
                saved.risky_goal_windows.as_ref()
            } else {
                None
            };
            ensure!(
                gene.stop_vol_mult == 0.0,
                "joint replay requires fixed entry brackets"
            );
            Ok(FrozenTrial {
                gene,
                member: json!({
                    "candidate_archive_index": index,
                    "strategy_identity": trial.strategy_identity,
                    "walkforward_positive_folds": trial.positive_folds,
                    "walkforward_splits": trial.reported_splits,
                    "fixed_stop_pips": gene.sl_pips,
                    "fixed_target_pips": gene.tp_pips,
                }),
                goal_windows,
            })
        })
        .collect()
}

fn frozen_risky_recipe(
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<(&LiveTradingPolicyV1, EvaluationConfig, RiskyGrowthGoal)> {
    let (policy, mut evaluation, _) = research_recipe_for_mode(research, true)?;
    let threshold = research.discovery_result().effective_smc_gate_threshold;
    ensure!(
        threshold.is_finite() && threshold >= 0.0,
        "invalid archived final SMC threshold"
    );
    evaluation.smc_gate_threshold = threshold;
    let goal = validate_risky_recipe(&evaluation)?;
    Ok((policy, evaluation, goal))
}

struct DirectTapes {
    times: Vec<i64>,
    tapes: Vec<Tape>,
    allocation: Vec<f64>,
}

impl DirectTapes {
    fn prepare(
        input: &CanonicalSearchRunInputV2<'_>,
        research: &CanonicalTrendbarResearchDiscoveryResultV3,
        scope: &CanonicalSearchArtifactScopeV2,
        trials: &[FrozenTrial<'_>],
        policy: &LiveTradingPolicyV1,
        evaluation: &EvaluationConfig,
    ) -> Result<Self> {
        let window = scope.evaluated_window();
        let start = usize::try_from(window.row_start())?;
        let end = usize::try_from(window.row_end())?;
        let bars = neoethos_data::slice_ohlcv(input.ohlcv(), start, end, None);
        let features = project_features_to_effective(
            &input.features().row_window(start, end)?,
            &research.discovery_result().effective_feature_names,
        )?;
        let genes = trials
            .iter()
            .map(|trial| trial.gene.clone())
            .collect::<Vec<_>>();
        let votes = ArchivedGeneVotes::prepare(
            &genes,
            &features,
            &bars,
            evaluation,
            policy.sealed_smc_gate_disabled()?,
            policy.sealed_adaptive_stops_policy()?,
        )?;
        let times = bars.timestamp.context("missing replay timestamps")?;
        let active = vec![Some(0); times.len()];
        Ok(Self {
            tapes: (0..genes.len())
                .map(|index| Tape::routed(&votes, &active, [Some(index); STATES.len()]))
                .collect::<Result<_>>()?,
            allocation: vec![1.0; times.len()],
            times,
        })
    }

    fn lanes<'a>(
        &'a self,
        trials: &'a [FrozenTrial<'_>],
        range: Range<usize>,
    ) -> Vec<JointBarLaneV1<'a>> {
        trials
            .iter()
            .zip(&self.tapes)
            .map(|(trial, tape)| JointBarLaneV1 {
                expert_id: &trial.gene.strategy_id,
                tape: NettedBarDecisionTapeV1 {
                    signals: &tape.signals[range.clone()],
                    confidences: &tape.confidence[range.clone()],
                    sl_pips: &tape.stops[range.clone()],
                    tp_pips: &tape.targets[range.clone()],
                    ml_multipliers: &self.allocation[range.clone()],
                },
            })
            .collect()
    }
}

pub fn evaluate_walkforward_joint_risky(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let source = research.discovery_result();
        let (policy, evaluation, goal) = frozen_risky_recipe(research)?;
        let calibration = source
            .calibration_scope()?
            .context("missing saved calibration")?;
        let trials = frozen_walkforward_trials(research)?;
        ensure!(
            !trials.is_empty(),
            "joint replay requires a nonempty frozen cohort"
        );
        let mut archived_windows: Option<Vec<ArchivedGoalWindow>> = None;
        for trial in &trials {
            let windows = ArchivedGoalWindows::deserialize(
                trial.goal_windows.context("missing Risky goal windows")?,
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
        }

        let account_policy = JointBarAccountPolicyV1 {
            max_target_pips: maximum_target(&trials),
            ..risky_account_policy(&evaluation)?
        };
        let cal_window = calibration.evaluated_window();
        let first_row = usize::try_from(cal_window.row_start())?;
        let tapes =
            DirectTapes::prepare(input, research, calibration, &trials, policy, &evaluation)?;
        let times = &tapes.times;
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
            let lanes = tapes.lanes(&trials, start..end);
            let result =
                account.evaluate_joint_risky_archived_brackets(&lanes, account_policy, goal)?;
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
            "members": trials.into_iter().map(|trial| trial.member).collect::<Vec<_>>(),
            "windows": results,
        }))
    })
}

fn maximum_target(trials: &[FrozenTrial<'_>]) -> f64 {
    trials
        .iter()
        .map(|trial| trial.gene.tp_pips)
        .fold(0.0, f64::max)
}

pub fn evaluate_weekly_joint_risky_forward(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let source = research.discovery_result();
        ensure!(
            source.calibration_scope()?.is_none(),
            "weekly Risky forward requires the full suffix to be reserved from search"
        );
        let holdout = source
            .holdout_scope()?
            .context("missing locked forward scope")?;
        let trials = frozen_walkforward_trials(research)?;
        let mut report = json!({
            "schema": "neoethos.weekly-joint-risky-forward.v1",
            "backend": "canonical_cpu",
            "promotion_eligible": false,
            "source_evidence_sha256": research.evidence_identity_sha256(),
            "search_config_hash": source.search_config_hash,
            "training_window": source.selection_scope.evaluated_window(),
            "forward_window": holdout.evaluated_window(),
            "members": trials.iter().map(|trial| &trial.member).collect::<Vec<_>>(),
            "combined_portfolio_evaluated": false,
        });
        if trials.is_empty() {
            report["status"] = json!("no_frozen_walkforward_survivors");
            return Ok(report);
        }
        let (policy, evaluation, goal) = frozen_risky_recipe(research)?;
        let training = &source.selection_scope;
        let training_goals = {
            let tapes =
                DirectTapes::prepare(input, research, training, &trials, policy, &evaluation)?;
            joint_goal_windows(
                &tapes,
                &trials,
                input,
                research,
                training,
                &evaluation,
                goal,
            )?
        };
        let tapes = DirectTapes::prepare(input, research, holdout, &trials, policy, &evaluation)?;
        let times = &tapes.times;
        let account = PreparedNettedCanonicalBarResearchV1::from_input(
            input,
            holdout,
            research.execution_contract(),
            &evaluation,
        )?;
        let target_ceiling = maximum_target(&trials);
        let continuous = account.evaluate_joint_exact_risky(
            &tapes.lanes(&trials, 0..times.len()),
            target_ceiling,
            true,
            None,
        )?;
        let weekly =
            forward_weekly_admissions(&continuous, &trials, times[0], times[times.len() - 1])?;
        let goals =
            joint_goal_windows(&tapes, &trials, input, research, holdout, &evaluation, goal)?;
        report["combined_portfolio_evaluated"] = json!(true);
        report["status"] = json!("evaluated");
        report["goal"] = json!(goal);
        report["weekly_admissions"] = weekly;
        report["continuous_shared_account"] = ledger(continuous, times)?;
        report["training_goal_windows"] = json!(training_goals);
        report["goal_windows"] = json!(goals);
        Ok(report)
    })
}

fn forward_weekly_admissions(
    result: &JointBarAccountResultV1,
    trials: &[FrozenTrial<'_>],
    start_ms: i64,
    end_ms: i64,
) -> Result<Value> {
    let mut entries: BTreeMap<&str, Vec<i64>> = BTreeMap::new();
    for trade in &result.closed_trades {
        entries
            .entry(&trade.expert_id)
            .or_default()
            .push(trade.trade.entry_time);
    }
    for mark in &result.terminal_open {
        entries
            .entry(&mark.expert_id)
            .or_default()
            .push(mark.entry_timestamp_ms);
    }
    let mut passing = 0;
    let mut by_strategy = Vec::with_capacity(trials.len());
    for trial in trials {
        let id = trial.gene.strategy_id.as_str();
        let stamps = entries.remove(id).unwrap_or_default();
        let counts = complete_week_entry_counts_v1(start_ms, end_ms, stamps.iter().copied())?;
        let missing = counts.iter().filter(|&&count| count == 0).count();
        let passed = !counts.is_empty() && missing == 0;
        passing += usize::from(passed);
        by_strategy.push(json!({
            "strategy_id": id,
            "total_entries": stamps.len(),
            "entries_per_complete_week": counts,
            "missed_weeks": missing,
            "passed": passed,
        }));
    }
    ensure!(
        entries.is_empty(),
        "forward entries include an unfrozen strategy"
    );
    Ok(json!({
        "week_start_ms": start_ms,
        "available_through_ms": end_ms,
        "passing_strategies": passing,
        "all_strategies_passed": passing == trials.len(),
        "by_strategy": by_strategy,
    }))
}

fn joint_goal_windows(
    tapes: &DirectTapes,
    trials: &[FrozenTrial<'_>],
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    scope: &CanonicalSearchArtifactScopeV2,
    evaluation: &EvaluationConfig,
    goal: RiskyGrowthGoal,
) -> Result<Vec<Value>> {
    let first_row = usize::try_from(scope.evaluated_window().row_start())?;
    let role = scope.evaluated_window().role();
    let prepare = match role {
        CanonicalSearchWindowRoleV1::Holdout => PreparedNettedCanonicalBarResearchV1::from_input,
        CanonicalSearchWindowRoleV1::InSample => {
            PreparedNettedCanonicalBarResearchV1::from_training_scope
        }
        _ => anyhow::bail!("shared goal windows require training or holdout input"),
    };
    let times = &tapes.times;
    let target_ceiling = maximum_target(trials);
    let duration = (goal.horizon_days * 86_400_000.0) as i64;
    let mut start = 0;
    let mut windows = Vec::new();
    while start < times.len() {
        let deadline = times[start]
            .checked_add(duration)
            .context("goal deadline overflow")?;
        if times.last().is_none_or(|&stamp| stamp < deadline) {
            break;
        }
        let next = times.partition_point(|&stamp| stamp <= deadline);
        // A following bar can witness coverage without being traded by the earlier account.
        let end = (next + 1).min(times.len());
        let window_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            role,
            input,
            first_row + start..first_row + end,
        )?;
        let account = prepare(
            input,
            &window_scope,
            research.execution_contract(),
            evaluation,
        )?;
        let result = account.evaluate_joint_exact_risky(
            &tapes.lanes(trials, start..end),
            target_ceiling,
            true,
            Some(goal),
        )?;
        windows.push(json!({
            "window": windows.len() + 1,
            "start_ms": times[start],
            "deadline_ms": deadline,
            "shared_account": ledger(result, account.timestamps())?,
        }));
        start = next;
    }
    Ok(windows)
}
