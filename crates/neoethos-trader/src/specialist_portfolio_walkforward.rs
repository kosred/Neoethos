//! Frozen Risky walk-forward survivors replayed concurrently on one account.
use super::*;
use neoethos_search::data_selection::CanonicalSearchArtifactScopeRefV1;
use neoethos_search::eval::{
    BarReplayLotGridV1, JointBarFixedNotionalLeverageV1, JointBarPriceStopLevelsV1,
    JointBarRiskyOrderSizingV1,
};
use neoethos_search::scoring::weekly_cadence::complete_week_entry_counts_v1;
use std::collections::BTreeMap;
use std::mem::size_of;
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
    lane_limit: Option<usize>,
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
    if let Some(limit) = lane_limit {
        ensure!(
            survivors.len() <= limit,
            "joint replay supports at most {limit} frozen lanes"
        );
    }
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
    price_stop_levels: Option<Vec<f64>>,
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
        if trials.len() > 256 {
            // Vote synthesis holds one signal and one confidence per row. The
            // routed tapes add a signal, confidence, stop and target per row.
            let cell_bytes = 2 * size_of::<i8>() + 4 * size_of::<f64>();
            let rows = end.checked_sub(start).context("invalid frozen window")?;
            let estimated_bytes = rows
                .checked_mul(trials.len())
                .and_then(|cells| cells.checked_mul(cell_bytes))
                .context("frozen replay tape size overflow")?;
            let tape_budget = neoethos_core::allocation_headroom_bytes() / 2;
            ensure!(
                estimated_bytes as u64 <= tape_budget,
                "{} frozen lanes require {estimated_bytes} bytes of vote and decision tapes, above the {tape_budget}-byte budget",
                trials.len()
            );
            tracing::info!(
                lanes = trials.len(),
                rows,
                estimated_bytes,
                tape_budget,
                "expanded frozen cohort admitted for shared-account replay"
            );
        }
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
            price_stop_levels: None,
            times,
        })
    }

    fn lanes<'a>(&'a self, expert_ids: &[&'a str], range: Range<usize>) -> Vec<JointBarLaneV1<'a>> {
        expert_ids
            .iter()
            .zip(&self.tapes)
            .map(|(&expert_id, tape)| JointBarLaneV1 {
                expert_id,
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
        let trials = frozen_walkforward_trials(research, Some(256))?;
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
        let expert_ids: Vec<_> = trials
            .iter()
            .map(|trial| trial.gene.strategy_id.as_str())
            .collect();
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
            let lanes = tapes.lanes(&expert_ids, start..end);
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

const FAST_ATR: usize = 14;
const SLOW_ATR: usize = 96;
const ATR_SCALE: [f64; 2] = [0.5, 2.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WeeklyRiskyScenario {
    Fixed,
    AdaptiveRisk,
    AdaptiveTp,
    AdaptiveSl,
    BrokerGrid,
    BrokerGridFitMargin,
    BrokerGridSingle,
    PriceAction,
    PriceActionRiskTen,
    PriceActionRiskFive,
    PriceActionRiskFiveTargetTwelve,
}

impl WeeklyRiskyScenario {
    const ALL: [Self; 4] = [
        Self::Fixed,
        Self::AdaptiveRisk,
        Self::AdaptiveTp,
        Self::AdaptiveSl,
    ];

    const EXECUTION: [Self; 5] = [
        Self::Fixed,
        Self::BrokerGrid,
        Self::BrokerGridFitMargin,
        Self::BrokerGridSingle,
        Self::PriceAction,
    ];

    const PRICE_ACTION_RISK: [Self; 4] = [
        Self::PriceAction,
        Self::PriceActionRiskTen,
        Self::PriceActionRiskFive,
        Self::PriceActionRiskFiveTargetTwelve,
    ];

    fn uses_price_action(self) -> bool {
        Self::PRICE_ACTION_RISK.contains(&self)
    }

    fn price_action_target(self) -> f64 {
        if self == Self::PriceActionRiskFiveTargetTwelve {
            12.0
        } else {
            PRICE_ACTION_TARGET_PIPS
        }
    }

    fn tapes<'a>(
        self,
        frozen: Option<&'a DirectTapes>,
        price: Option<&'a mut DirectTapes>,
    ) -> Result<&'a DirectTapes> {
        if self.uses_price_action() {
            let price = price.context("missing price action tape")?;
            price.tapes[0].targets.fill(self.price_action_target());
            Ok(price)
        } else {
            frozen.context("missing frozen specialist tapes")
        }
    }

    fn order_sizing<'a>(
        self,
        leverage: Option<JointBarFixedNotionalLeverageV1>,
        lot_grid: Option<BarReplayLotGridV1>,
        price_stop_bounds: Option<[f64; 2]>,
        tapes: &'a DirectTapes,
    ) -> JointBarRiskyOrderSizingV1<'a> {
        let single_position = self == Self::BrokerGridSingle || self.uses_price_action();
        let fit_to_buying_power = single_position || self == Self::BrokerGridFitMargin;
        let lot_grid = lot_grid.filter(|_| self != Self::Fixed);
        JointBarRiskyOrderSizingV1 {
            leverage,
            confidence_risk_floor: (self == Self::AdaptiveRisk).then_some(
                neoethos_core::domain::risky_mode::RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION,
            ),
            lot_grid,
            fit_to_buying_power,
            max_open_positions: if single_position { 1 } else { 3 },
            price_stops: price_stop_bounds.and_then(|distance_bounds_pips| {
                tapes.price_stop_levels.as_deref().map(|signal_row_levels| {
                    JointBarPriceStopLevelsV1 {
                        signal_row_levels,
                        distance_bounds_pips,
                    }
                })
            }),
            price_action_risk_ceiling: match self {
                Self::PriceActionRiskTen => Some(0.10),
                Self::PriceActionRiskFive | Self::PriceActionRiskFiveTargetTwelve => Some(0.05),
                _ => None,
            },
        }
    }

    fn target_ceiling(self, trials: &[FrozenTrial<'_>]) -> f64 {
        let target = maximum_target(trials);
        if self.uses_price_action() {
            self.price_action_target()
        } else if self == Self::AdaptiveTp {
            target * ATR_SCALE[1]
        } else {
            target
        }
    }
}

pub struct BrokerLotComparisonV1 {
    pub grid: BarReplayLotGridV1,
    pub symbol_contract_sha256: String,
    pub price_action_stop_bounds_pips: [f64; 2],
    pub price_action_risk_comparison: bool,
}

const PRICE_ACTION_ID: &str = "h1-trend-m15-pullback-v1";
const PRICE_ACTION_TARGET_PIPS: f64 = 20.0;
const M15_MS: i64 = 15 * 60_000;
const H1_MS: i64 = 4 * M15_MS;

struct PriceActionTape {
    signals: Vec<i8>,
    invalidation_prices: Vec<f64>,
}

impl PriceActionTape {
    fn from_input(input: &CanonicalSearchRunInputV2<'_>) -> Result<Self> {
        let bars = input.ohlcv();
        let times = bars
            .timestamp
            .as_deref()
            .context("price action requires M15 timestamps")?;
        let mut hours = Vec::new();
        // An H1 candle exists only after all four consecutive M15 bars.
        for (row, stamps) in times.windows(4).enumerate() {
            if stamps[0].rem_euclid(H1_MS) != 0
                || !stamps.windows(2).all(|pair| pair[1] - pair[0] == M15_MS)
            {
                continue;
            }
            let high = bars.high[row..row + 4]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let low = bars.low[row..row + 4]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            hours.push((stamps[0] + H1_MS, high, low));
        }
        let mut signals = vec![0; times.len()];
        let mut invalidation_prices = vec![0.0; times.len()];
        for row in 1..times.len() {
            let previous = row - 1;
            if times[row] - times[previous] != M15_MS {
                continue;
            }
            // Both trend bars must be known before the pullback bar opens.
            let completed = hours.partition_point(|&(end, _, _)| end <= times[previous]);
            if completed < 2 {
                continue;
            }
            let (older_end, older_high, older_low) = hours[completed - 2];
            let (recent_end, recent_high, recent_low) = hours[completed - 1];
            if recent_end - older_end != H1_MS || times[previous] - recent_end >= H1_MS {
                continue;
            }
            if recent_high > older_high
                && recent_low > older_low
                && bars.close[previous] < bars.open[previous]
                && bars.close[row] > bars.high[previous]
            {
                signals[row] = 1;
                invalidation_prices[row] = bars.low[row].min(bars.low[previous]);
            } else if recent_high < older_high
                && recent_low < older_low
                && bars.close[previous] > bars.open[previous]
                && bars.close[row] < bars.low[previous]
            {
                signals[row] = -1;
                invalidation_prices[row] = bars.high[row].max(bars.high[previous]);
            }
        }
        Ok(Self {
            signals,
            invalidation_prices,
        })
    }

    fn for_scope(
        &self,
        input: &CanonicalSearchRunInputV2<'_>,
        scope: &CanonicalSearchArtifactScopeV2,
        pip: f64,
    ) -> Result<DirectTapes> {
        let window = scope.evaluated_window();
        let range = usize::try_from(window.row_start())?..usize::try_from(window.row_end())?;
        let bars = input.ohlcv();
        let times = bars
            .timestamp
            .as_deref()
            .and_then(|values| values.get(range.clone()))
            .context("price action scope is outside the M15 input")?
            .to_vec();
        let signals = self.signals[range.clone()].to_vec();
        let invalidation_prices = self.invalidation_prices[range.clone()].to_vec();
        let stops = signals
            .iter()
            .zip(&bars.close[range])
            .zip(&invalidation_prices)
            .map(|((&direction, &close), &level)| f64::from(direction) * (close - level) / pip)
            .collect();
        Ok(DirectTapes {
            allocation: vec![1.0; times.len()],
            tapes: vec![Tape {
                signals,
                stops,
                confidence: vec![1.0; times.len()],
                targets: vec![PRICE_ACTION_TARGET_PIPS; times.len()],
            }],
            times,
            price_stop_levels: Some(invalidation_prices),
        })
    }
}

struct AdaptiveWeeklyBrackets {
    multipliers: Vec<f64>,
    stop_bounds: [f64; 2],
}

impl AdaptiveWeeklyBrackets {
    fn from_input(input: &CanonicalSearchRunInputV2<'_>, stop_bounds: [f64; 2]) -> Result<Self> {
        ensure!(
            stop_bounds.iter().all(|x| x.is_finite() && *x > 0.0)
                && stop_bounds[0] <= stop_bounds[1],
            "adaptive SL comparison requires ordered positive bounds"
        );
        let bars = input.ohlcv();
        let fast =
            neoethos_search::stop_target::compute_atr(&bars.high, &bars.low, &bars.close, FAST_ATR);
        let slow =
            neoethos_search::stop_target::compute_atr(&bars.high, &bars.low, &bars.close, SLOW_ATR);
        let multipliers = fast
            .into_iter()
            .zip(slow)
            .enumerate()
            .map(|(row, (fast, slow))| {
                if row < SLOW_ATR {
                    return Ok(1.0);
                }
                ensure!(
                    fast.is_finite() && fast > 0.0 && slow.is_finite() && slow > 0.0,
                    "invalid ATR scale on completed bar {row}"
                );
                Ok((fast / slow).clamp(ATR_SCALE[0], ATR_SCALE[1]))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            multipliers,
            stop_bounds,
        })
    }

    fn apply(
        &self,
        tapes: &mut DirectTapes,
        trades: &[FrozenTrial<'_>],
        scope: &CanonicalSearchArtifactScopeV2,
        scenario: WeeklyRiskyScenario,
    ) -> Result<()> {
        let window = scope.evaluated_window();
        let range = usize::try_from(window.row_start())?..usize::try_from(window.row_end())?;
        let multipliers = self
            .multipliers
            .get(range)
            .context("ATR scale does not cover the replay scope")?;
        // Reuse the prepared tapes across scenarios. Restore the other leg from
        // the frozen gene so each arm changes one bracket on the signal row.
        for (tape, trial) in tapes.tapes.iter_mut().zip(trades) {
            for (row, &scale) in multipliers.iter().enumerate() {
                if tape.signals[row] == 0 {
                    continue;
                }
                tape.targets[row] = trial.gene.tp_pips;
                tape.stops[row] = trial.gene.sl_pips;
                match scenario {
                    WeeklyRiskyScenario::AdaptiveTp => tape.targets[row] *= scale,
                    WeeklyRiskyScenario::AdaptiveSl => {
                        tape.stops[row] = (tape.stops[row] * scale)
                            .clamp(self.stop_bounds[0], self.stop_bounds[1]);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

pub fn evaluate_weekly_joint_risky_forward(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    leverage: Option<JointBarFixedNotionalLeverageV1>,
    adaptive_stop_bounds: Option<[f64; 2]>,
    broker_lots: Option<&BrokerLotComparisonV1>,
) -> Result<Value> {
    let price_risk = broker_lots.is_some_and(|options| options.price_action_risk_comparison);
    ensure!(
        broker_lots.is_none()
            || (adaptive_stop_bounds.is_none() && (price_risk || leverage.is_some())),
        "lot comparison requires fixed leverage and its own replay mode"
    );
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let source = research.discovery_result();
        ensure!(
            source.calibration_scope()?.is_none(),
            "weekly Risky forward requires the full suffix to be reserved from search"
        );
        let holdout = source
            .holdout_scope()?
            .context("missing locked forward scope")?;
        let trials = frozen_walkforward_trials(research, None)?;
        let mut report = json!({
            "schema": if broker_lots.is_some() {
                "neoethos.weekly-joint-risky-forward.lot-comparison.v1"
            } else if adaptive_stop_bounds.is_some() {
                "neoethos.weekly-joint-risky-forward.adaptive-comparison.v1"
            } else if leverage.is_some() {
                "neoethos.weekly-joint-risky-forward.leverage.v1"
            } else {
                "neoethos.weekly-joint-risky-forward.v1"
            },
            "backend": "canonical_cpu",
            "promotion_eligible": false,
            "source_evidence_sha256": research.evidence_identity_sha256(),
            "search_config_hash": source.search_config_hash,
            "training_window": source.selection_scope.evaluated_window(),
            "forward_window": holdout.evaluated_window(),
            "members": trials.iter().map(|trial| &trial.member).collect::<Vec<_>>(),
            "combined_portfolio_evaluated": false,
        });
        if trials.is_empty() && !price_risk {
            report["status"] = json!("no_frozen_walkforward_survivors");
            return Ok(report);
        }
        let (policy, evaluation, goal) = frozen_risky_recipe(research)?;
        let bound_policy = evaluation.risky_account_experiment.as_ref();
        let bound_leverage = bound_policy.and_then(|policy| policy.fixed_leverage);
        let mut lot_grid = broker_lots.map(|options| options.grid);
        if bound_leverage.is_some() {
            ensure!(
                leverage.is_none()
                    && adaptive_stop_bounds.is_none()
                    && (broker_lots.is_none() || price_risk),
                "the shared account must use the fixed broker profile sealed during search"
            );
            let bound_lots = bound_policy
                .and_then(|policy| policy.lot_constraints.as_ref())
                .context("sealed fixed leverage has no lot grid")?;
            let grid = bound_lots.checked_grid()?;
            if let Some(options) = broker_lots {
                ensure!(
                    options.grid == grid
                        && options.symbol_contract_sha256 == bound_lots.source_artifact_sha256(),
                    "comparison symbol differs from the broker profile sealed during search"
                );
            }
            lot_grid = Some(grid);
            report["schema"] = json!("neoethos.weekly-joint-risky-forward.broker-profile.v1");
            report["broker_volume_source"] = json!({
                "symbol_contract_sha256": bound_lots.source_artifact_sha256(),
                "lot_grid": grid,
                "fixed_leverage": bound_leverage,
            });
        } else if price_risk {
            report["schema"] = json!("neoethos.weekly-joint-risky-forward.price-action-risk.v1");
        }
        let leverage = bound_leverage.or(leverage);
        ensure!(
            !price_risk || leverage.is_some(),
            "price-action risk comparison requires a fixed leverage assumption"
        );
        let price_stop_bounds = broker_lots.map(|options| options.price_action_stop_bounds_pips);
        let training = &source.selection_scope;
        let expert_ids: Vec<_> = trials
            .iter()
            .map(|trial| trial.gene.strategy_id.as_str())
            .collect();
        let price_ids = [PRICE_ACTION_ID];
        let price_action = broker_lots
            .map(|_| PriceActionTape::from_input(input))
            .transpose()?;
        let adaptive = adaptive_stop_bounds
            .map(|bounds| AdaptiveWeeklyBrackets::from_input(input, bounds))
            .transpose()?;
        let mut modes = if price_risk {
            WeeklyRiskyScenario::PRICE_ACTION_RISK.to_vec()
        } else if broker_lots.is_some() {
            WeeklyRiskyScenario::EXECUTION.to_vec()
        } else if adaptive.is_some() {
            WeeklyRiskyScenario::ALL.to_vec()
        } else {
            vec![WeeklyRiskyScenario::Fixed]
        };
        if bound_leverage.is_some() && !trials.is_empty() {
            if price_risk {
                modes.insert(0, WeeklyRiskyScenario::BrokerGridFitMargin);
            } else {
                modes[0] = WeeklyRiskyScenario::BrokerGridFitMargin;
            }
        }
        let run_frozen = modes.iter().any(|mode| !mode.uses_price_action());
        let prepare_frozen = |scope| {
            run_frozen
                .then(|| DirectTapes::prepare(input, research, scope, &trials, policy, &evaluation))
                .transpose()
        };
        let mut results = Vec::with_capacity(modes.len());
        {
            let mut frozen_tapes = prepare_frozen(training)?;
            let mut price_tapes = price_action
                .as_ref()
                .map(|price| price.for_scope(input, training, evaluation.pip_value))
                .transpose()?;
            for &scenario in &modes {
                tracing::info!(?scenario, "shared Risky training scenario started");
                if let (Some(brackets), Some(frozen)) = (&adaptive, &mut frozen_tapes) {
                    brackets.apply(frozen, &trials, training, scenario)?;
                }
                let tapes = scenario.tapes(frozen_tapes.as_ref(), price_tapes.as_mut())?;
                let ids = if scenario.uses_price_action() {
                    price_ids.as_slice()
                } else {
                    expert_ids.as_slice()
                };
                let training_goals = joint_goal_windows(
                    tapes,
                    ids,
                    input,
                    research,
                    training,
                    &evaluation,
                    goal,
                    scenario.target_ceiling(&trials),
                    scenario.order_sizing(leverage, lot_grid, price_stop_bounds, tapes),
                )?;
                results.push((scenario, json!({"training_goal_windows": training_goals})));
            }
        }
        let mut frozen_tapes = prepare_frozen(holdout)?;
        let mut price_tapes = price_action
            .as_ref()
            .map(|price| price.for_scope(input, holdout, evaluation.pip_value))
            .transpose()?;
        let account = PreparedNettedCanonicalBarResearchV1::from_input(
            input,
            holdout,
            research.execution_contract(),
            &evaluation,
        )?;
        for (scenario, outcome) in &mut results {
            tracing::info!(?scenario, "shared Risky forward scenario started");
            if let (Some(brackets), Some(frozen)) = (&adaptive, &mut frozen_tapes) {
                brackets.apply(frozen, &trials, holdout, *scenario)?;
            }
            let tapes = scenario.tapes(frozen_tapes.as_ref(), price_tapes.as_mut())?;
            let ids = if scenario.uses_price_action() {
                price_ids.as_slice()
            } else {
                expert_ids.as_slice()
            };
            let times = &tapes.times;
            let order_sizing = scenario.order_sizing(leverage, lot_grid, price_stop_bounds, tapes);
            let target_ceiling = scenario.target_ceiling(&trials);
            let continuous = account.evaluate_joint_risky_full_size(
                &tapes.lanes(ids, 0..times.len()),
                target_ceiling,
                true,
                None,
                order_sizing,
            )?;
            outcome["weekly_admissions"] =
                forward_weekly_admissions(&continuous, ids, times[0], times[times.len() - 1])?;
            outcome["continuous_shared_account"] = ledger(continuous, times)?;
            outcome["goal_windows"] = json!(joint_goal_windows(
                tapes,
                ids,
                input,
                research,
                holdout,
                &evaluation,
                goal,
                target_ceiling,
                order_sizing,
            )?);
        }
        report["combined_portfolio_evaluated"] = json!(run_frozen);
        report["status"] = json!("evaluated");
        report["goal"] = json!(goal);
        if let Some(config) = broker_lots {
            if bound_leverage.is_none() {
                report["broker_volume_source"] = json!({
                    "symbol_contract_sha256": config.symbol_contract_sha256,
                    "lot_grid": config.grid,
                });
            }
            report["price_action_recipe"] = json!({
                "strategy_id": PRICE_ACTION_ID,
                "trend": "two consecutive complete H1 ranges with higher highs and lows or lower highs and lows, known before the pullback starts",
                "signal": "M15 countertrend bar followed by a close beyond its high or low in the trend direction",
                "stop": "the extreme of the pullback and signal bars, checked against the modeled entry price",
                "stop_distance_bounds_pips": config.price_action_stop_bounds_pips,
                "target_pips": PRICE_ACTION_TARGET_PIPS,
            });
            if price_risk {
                report["price_action_recipe"]["short_target_pips"] = json!(
                    WeeklyRiskyScenario::PriceActionRiskFiveTargetTwelve.price_action_target()
                );
            }
        }
        if let Some(brackets) = adaptive {
            report["confidence_full_risk_threshold"] = json!(evaluation.high_quality_confidence);
            report["adaptive_brackets"] = json!({
                "source": "causal_ema_true_range_ratio_on_signal_bar",
                "fast_period": FAST_ATR, "slow_period": SLOW_ATR,
                "scale_bounds": ATR_SCALE, "warmup_rows_at_fixed_brackets": SLOW_ATR,
                "stop_bounds_pips": brackets.stop_bounds,
            });
        }
        if modes.len() > 1 {
            report["scenarios"] = json!(
                results
                    .into_iter()
                    .map(|(scenario, mut outcome)| {
                        outcome["scenario"] = json!(scenario);
                        outcome
                    })
                    .collect::<Vec<_>>()
            );
        } else {
            let (_, outcome) = results
                .pop()
                .context("fixed Risky replay result is absent")?;
            let Value::Object(mut fields) = outcome else {
                anyhow::bail!("invalid Risky replay report");
            };
            report
                .as_object_mut()
                .context("invalid joint replay document")?
                .append(&mut fields);
        }
        Ok(report)
    })
}

fn forward_weekly_admissions(
    result: &JointBarAccountResultV1,
    expert_ids: &[&str],
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
    let mut by_strategy = Vec::with_capacity(expert_ids.len());
    for &id in expert_ids {
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
        "all_strategies_passed": passing == expert_ids.len(),
        "by_strategy": by_strategy,
    }))
}

#[allow(clippy::too_many_arguments)]
fn joint_goal_windows(
    tapes: &DirectTapes,
    expert_ids: &[&str],
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    scope: &CanonicalSearchArtifactScopeV2,
    evaluation: &EvaluationConfig,
    goal: RiskyGrowthGoal,
    target_ceiling: f64,
    order_sizing: JointBarRiskyOrderSizingV1<'_>,
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
        let mut window_sizing = order_sizing;
        if let Some(stops) = &mut window_sizing.price_stops {
            stops.signal_row_levels = &stops.signal_row_levels[start..end];
        }
        let result = account.evaluate_joint_risky_full_size(
            &tapes.lanes(expert_ids, start..end),
            target_ceiling,
            true,
            Some(goal),
            window_sizing,
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
