//! Conditional expert selection and cost-charged replay on one canonical account.
use anyhow::{Context, Result, ensure};
use neoethos_data::{FeatureFrame, Ohlcv};
use neoethos_search::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchRunInputV2, CanonicalSearchWindowRoleV1,
    CanonicalTrendbarResearchDiscoveryResultV3, EvaluationConfig, Gene,
    eval::{
        NettedBarDecisionTapeV1, NettedBarResearchResultV1, PreparedNettedCanonicalBarResearchV1,
    },
    live_portfolio::{LiveTradingPolicyV1, project_features_to_effective},
    quality::monthly_closed_trade_metrics_v1,
};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::gene_signal::ArchivedGeneVotes;

#[path = "specialist_portfolio_joint.rs"]
mod joint;
pub use joint::evaluate as evaluate_joint;
pub use joint::evaluate_continual;
pub use joint::evaluate_risky;
pub use joint::evaluate_walkforward_joint_risky;

const STATES: [&str; 4] = [
    "hourly_up",
    "hourly_down",
    "hourly_range",
    "hourly_volatility",
];
const HOUR_MS: i64 = 3_600_000;
const CONTEXT_HOURS: usize = 9;
const SEED: u64 = 20261111;
const PARENTS: usize = 16;
const MIN_EXPERT_TRADES: usize = 20;
const RISK_CAP: f64 = 0.01;
type Assignments = [Option<usize>; STATES.len()];

#[derive(Clone, Copy)]
struct Hour {
    close: f64,
    high: f64,
    low: f64,
}

fn market_state(history: &std::collections::VecDeque<Hour>) -> Option<usize> {
    if history.len() != CONTEXT_HOURS {
        return None;
    }
    let last = history.back()?;
    let first = history.front()?;
    let range_sum: f64 = history
        .iter()
        .take(CONTEXT_HOURS - 1)
        .map(|h| h.high - h.low)
        .sum();
    if last.high - last.low > 2.0 * range_sum / (CONTEXT_HOURS - 1) as f64 {
        return Some(3);
    }
    let distance: f64 = history
        .iter()
        .zip(history.iter().skip(1))
        .map(|(a, b)| (b.close - a.close).abs())
        .sum();
    let movement = last.close - first.close;
    if distance == 0.0 || movement.abs() < 0.35 * distance {
        Some(2)
    } else {
        Some(usize::from(movement < 0.0))
    }
}

/// A context becomes visible only in the next clock hour. Missing candles
/// invalidate the rolling history, including weekend and partial-hour gaps.
fn market_states(bars: &Ohlcv, period_ms: i64) -> Result<Vec<Option<usize>>> {
    ensure!(
        period_ms > 0 && HOUR_MS % period_ms == 0,
        "expert routing needs bars that partition a UTC hour"
    );
    let times = bars
        .timestamp
        .as_deref()
        .context("missing market timestamps")?;
    ensure!(
        times.len() == bars.close.len()
            && times.len() == bars.high.len()
            && times.len() == bars.low.len(),
        "unaligned market context"
    );
    let mut history = std::collections::VecDeque::with_capacity(CONTEXT_HOURS);
    let mut states = Vec::with_capacity(times.len());
    let mut start = 0;
    let mut state = None;
    for (row, &time) in times.iter().enumerate() {
        ensure!(
            time % period_ms == 0 && (row == 0 || times[row - 1] < time),
            "invalid decision clock"
        );
        if row > 0 && time - times[row - 1] != period_ms {
            history.clear();
            state = None;
            start = row;
        }
        if time / HOUR_MS != times[start] / HOUR_MS {
            let complete = times[start] % HOUR_MS == 0
                && row - start == (HOUR_MS / period_ms) as usize
                && time - times[row - 1] == period_ms;
            if complete {
                if history.len() == CONTEXT_HOURS {
                    history.pop_front();
                }
                history.push_back(Hour {
                    close: bars.close[row - 1],
                    high: bars.high[start..row]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                    low: bars.low[start..row]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min),
                });
            } else {
                history.clear();
            }
            state = market_state(&history);
            start = row;
        }
        states.push(state);
    }
    Ok(states)
}

/// Frozen model. Indices address the embedded genes in the exact feature order.
/// No future prices or individual holdout scores participate in routing.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialistPolicyV1 {
    pub schema: String,
    pub source_evidence_sha256: String,
    pub feature_names: Vec<String>,
    pub genes: Vec<Gene>,
    assignments: Assignments,
    pub base_period_ms: i64,
}

struct Window {
    features: FeatureFrame,
    bars: Ohlcv,
    states: Vec<Option<usize>>,
    account: PreparedNettedCanonicalBarResearchV1,
}

impl Window {
    fn open(
        input: &CanonicalSearchRunInputV2<'_>,
        research: &CanonicalTrendbarResearchDiscoveryResultV3,
        scope: &CanonicalSearchArtifactScopeV2,
        evaluation: &EvaluationConfig,
        period: i64,
    ) -> Result<Self> {
        let contract = research.execution_contract();
        let window = scope.evaluated_window();
        let account = if window.role() == CanonicalSearchWindowRoleV1::Holdout {
            PreparedNettedCanonicalBarResearchV1::from_input(input, scope, contract, evaluation)?
        } else {
            PreparedNettedCanonicalBarResearchV1::from_training_scope(
                input, scope, contract, evaluation,
            )?
        };
        let start = usize::try_from(window.row_start())?;
        let end = usize::try_from(window.row_end())?;
        let bars = neoethos_data::slice_ohlcv(input.ohlcv(), start, end, None);
        let states = market_states(&bars, period)?;
        let features = project_features_to_effective(
            &input.features().row_window(start, end)?,
            &research.discovery_result().effective_feature_names,
        )?;
        Ok(Self {
            features,
            bars,
            states,
            account,
        })
    }

    fn votes<'a>(
        &self,
        genes: &'a [Gene],
        policy: &LiveTradingPolicyV1,
    ) -> Result<ArchivedGeneVotes<'a>> {
        ArchivedGeneVotes::prepare(
            genes,
            &self.features,
            &self.bars,
            &policy.sealed_evaluation_config()?,
            policy.sealed_smc_gate_disabled()?,
            policy.sealed_adaptive_stops_policy()?,
        )
    }
}

/// Select before netting. All entry fields are taken from the same expert.
/// A silent expert or an unassigned state explicitly emits no new entry.
struct Tape {
    signals: Vec<i8>,
    confidence: Vec<f64>,
    stops: Vec<f64>,
    targets: Vec<f64>,
}

impl Tape {
    fn routed(
        votes: &ArchivedGeneVotes<'_>,
        states: &[Option<usize>],
        assignments: Assignments,
    ) -> Result<Self> {
        let n = states.len();
        let mut tape = Self {
            signals: vec![0; n],
            confidence: vec![0.; n],
            stops: vec![0.; n],
            targets: vec![0.; n],
        };
        for (row, state) in states.iter().enumerate() {
            if let Some(gene) = state.and_then(|s| assignments[s])
                && let Some((signal, [confidence, stop, target])) = votes.decision(gene, row)?
            {
                tape.signals[row] = signal;
                tape.confidence[row] = confidence;
                tape.stops[row] = stop;
                tape.targets[row] = target;
            }
        }
        Ok(tape)
    }

    fn evaluate(
        &self,
        account: &PreparedNettedCanonicalBarResearchV1,
    ) -> Result<NettedBarResearchResultV1> {
        account.evaluate(
            NettedBarDecisionTapeV1 {
                signals: &self.signals,
                confidences: &self.confidence,
                sl_pips: &self.stops,
                tp_pips: &self.targets,
                ml_multipliers: &vec![1.0; self.signals.len()],
            },
            None,
        )
    }
}

fn ledger_months(
    result: &NettedBarResearchResultV1,
    initial: f64,
    times: &[i64],
) -> Result<Vec<neoethos_search::quality::MonthlyStrategyMetricsV1>> {
    monthly_closed_trade_metrics_v1(
        &result.closed_trades,
        initial,
        times[0],
        times[times.len() - 1],
    )
}

fn account_score(result: &NettedBarResearchResultV1, initial: f64, times: &[i64]) -> Result<f64> {
    let monthly = ledger_months(result, initial, times)?;
    let mut sum = 0.0;
    for month in &monthly {
        if month.starting_balance <= 0.0 || month.ending_balance <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        let growth = (month.ending_balance / month.starting_balance).ln();
        sum += growth + growth.min(0.0);
    }
    Ok((sum - result.metrics.max_drawdown) / monthly.len() as f64)
}

fn trade_states(
    result: &NettedBarResearchResultV1,
    window: &Window,
) -> Result<[usize; STATES.len()]> {
    let mut counts = [0; STATES.len()];
    for trade in &result.closed_trades {
        let entry = window
            .account
            .timestamps()
            .binary_search(&trade.entry_time)
            .map_err(|_| anyhow::anyhow!("account trade has no entry bar"))?;
        if let Some(state) = entry.checked_sub(1).and_then(|row| window.states[row]) {
            counts[state] += 1;
        }
    }
    Ok(counts)
}

// Coordinate ascent on the actual shared ledger. An inactive market state
// contributes cash. Adding an expert must improve the existing portfolio.
fn fit(
    window: &Window,
    genes: &[Gene],
    policy: &LiveTradingPolicyV1,
    initial: f64,
    mut assignments: Assignments,
) -> Result<(Assignments, f64, Vec<Value>)> {
    let votes = window.votes(genes, policy)?;
    let mut trials = Vec::new();
    let mut score = |a| -> Result<_> {
        let result = Tape::routed(&votes, &window.states, a)?.evaluate(&window.account)?;
        let value = account_score(&result, initial, window.account.timestamps())?;
        let counts = trade_states(&result, window)?;
        trials.push(
            json!({ "assignments": a, "score": value, "score_finite": value.is_finite(),
            "ending_realized_balance": result.ending_realized_balance,
            "trade_count": result.closed_trades.len(), "entries_by_state": counts,
            "max_drawdown": result.metrics.max_drawdown }),
        );
        Ok((value, counts))
    };
    let (mut best, _) = score(assignments)?;
    for state in 0..STATES.len() {
        let mut chosen = assignments[state];
        for expert in std::iter::once(None).chain((0..genes.len()).map(Some)) {
            if expert == assignments[state] {
                continue;
            }
            let mut trial = assignments;
            trial[state] = expert;
            let (value, counts) = score(trial)?;
            if value > best && (expert.is_none() || counts[state] >= MIN_EXPERT_TRADES) {
                best = value;
                chosen = expert;
            }
        }
        assignments[state] = chosen;
    }
    Ok((assignments, best, trials))
}

fn used(assignments: Assignments) -> Vec<usize> {
    assignments
        .into_iter()
        .flatten()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Cash may win admission without stopping the bounded genetic exploration.
fn breed_training_children(genes: &mut Vec<Gene>, assignments: Assignments) -> Result<usize> {
    let n = genes.len();
    ensure!(n >= 2, "breeding requires two archived training parents");
    let assigned = used(assignments);
    let parents = if assigned.len() >= 2 {
        assigned
    } else {
        (0..n).collect()
    };
    let mut rng = StdRng::seed_from_u64(SEED);
    for (rank, &parent) in parents.iter().enumerate() {
        for displacement in [1, n / 2] {
            let mate = (parent + displacement + rank) % n;
            if parent != mate {
                let generation = genes[parent].generation.max(genes[mate].generation) + 1;
                genes.push(neoethos_search::genetic::crossover(
                    &genes[parent],
                    &genes[mate],
                    generation,
                    &mut rng,
                ));
            }
        }
    }
    Ok(genes.len() - n)
}

fn ledger(result: NettedBarResearchResultV1, window: &Window, initial: f64) -> Result<Value> {
    let monthly = ledger_months(&result, initial, window.account.timestamps())?;
    let counts = trade_states(&result, window)?;
    Ok(json!({ "account": result, "monthly": monthly, "entries_by_state": counts }))
}

fn compare(
    window: &Window,
    frozen: &SpecialistPolicyV1,
    policy: &LiveTradingPolicyV1,
    initial: f64,
) -> Result<Value> {
    let votes = window.votes(&frozen.genes, policy)?;
    let routed =
        Tape::routed(&votes, &window.states, frozen.assignments)?.evaluate(&window.account)?;
    let netted = votes.net(&used(frozen.assignments))?;
    let ordinary = Tape {
        signals: netted.directions.iter().map(|d| d.sign() as i8).collect(),
        confidence: netted.confidences,
        stops: netted.sl_pips,
        targets: netted.tp_pips,
    }
    .evaluate(&window.account)?;
    Ok(
        json!({ "specialists": ledger(routed, window, initial)?, "same_genes_majority_vote": ledger(ordinary, window, initial)?,
        "cash_ending_balance": initial }),
    )
}

/// Fit exclusively on the recorded selection interval. The two subsequent
/// intervals evaluate the same locked genes, assignments, costs and sizing.
pub fn evaluate(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let result = research.discovery_result();
        let policy = result
            .funnel_profile
            .as_ref()
            .and_then(|f| f.live_trading_policy_v1())
            .context("missing sealed gene policy")?;
        policy.validate()?;
        ensure!(
            policy.source_search_config_hash == result.search_config_hash,
            "mismatched search policy"
        );
        let sealed_evaluation = policy.sealed_evaluation_config()?;
        let mut evaluation = sealed_evaluation.clone();
        let original_risk = [evaluation.risk_per_trade_min, evaluation.risk_per_trade_max];
        evaluation.risk_per_trade_max = evaluation.risk_per_trade_max.min(RISK_CAP);
        evaluation.risk_per_trade_min = evaluation
            .risk_per_trade_min
            .min(evaluation.risk_per_trade_max);
        let initial = evaluation.initial_equity;
        let calibration = result
            .calibration_scope
            .as_ref()
            .context("missing calibration scope")?;
        let holdout = result
            .holdout_scope
            .as_ref()
            .context("missing final holdout scope")?;
        let selection = &result.selection_scope;
        ensure!(
            selection.evaluated_window().role() == CanonicalSearchWindowRoleV1::InSample
                && calibration.evaluated_window().role()
                    == CanonicalSearchWindowRoleV1::SelectionValidation
                && holdout.evaluated_window().role() == CanonicalSearchWindowRoleV1::Holdout
                && selection.evaluated_window().row_end()
                    <= calibration.evaluated_window().row_start()
                && calibration.evaluated_window().row_end()
                    <= holdout.evaluated_window().row_start(),
            "overlapping research scopes"
        );
        let times = input
            .ohlcv()
            .timestamp
            .as_ref()
            .context("missing canonical timestamps")?;
        let train_end = usize::try_from(selection.evaluated_window().row_end())?;
        ensure!(
            train_end >= 2 && train_end <= times.len(),
            "invalid selection boundary"
        );
        let period = times[..train_end]
            .windows(2)
            .map(|w| w[1] - w[0])
            .min()
            .context("missing base cadence")?;
        let train = Window::open(input, research, selection, &evaluation, period)?;
        let n = PARENTS.min(result.candidates.len());
        ensure!(n >= 2, "at least two archived genetic candidates required");
        let mut genes: Vec<_> = (0..n)
            .map(|i| result.candidates[i * result.candidates.len() / n].clone())
            .collect();
        let (first, first_score, parent_trials) =
            fit(&train, &genes, policy, initial, [None; STATES.len()])?;
        breed_training_children(&mut genes, first)?;
        let (routes, score, child_trials) = fit(&train, &genes, policy, initial, first)?;
        let selected = used(routes);
        let frozen = SpecialistPolicyV1 {
            schema: "neoethos.specialist-router.v1".into(),
            source_evidence_sha256: research.evidence_identity_sha256().to_owned(),
            base_period_ms: period,
            feature_names: result.effective_feature_names.clone(),
            assignments: routes
                .map(|g| g.map(|i| selected.binary_search(&i).expect("selected assignment"))),
            genes: selected.iter().map(|&i| genes[i].clone()).collect(),
        };
        drop(train);
        let evaluate_window = |scope| {
            let window = Window::open(input, research, scope, &evaluation, period)?;
            compare(&window, &frozen, policy, initial)
        };
        let calibration_result = evaluate_window(calibration)?;
        let final_result = evaluate_window(holdout)?;
        // Evaluate the already-frozen router at the archived risk as a separate
        // diagnostic. Neither outcome may refit genes or change assignments.
        let sealed_risk_window = |scope| {
            let window = Window::open(input, research, scope, &sealed_evaluation, period)?;
            compare(&window, &frozen, policy, initial)
        };
        let sealed_risk_calibration = sealed_risk_window(calibration)?;
        let sealed_risk_holdout = sealed_risk_window(holdout)?;
        Ok(
            json!({ "schema": "neoethos.specialist-portfolio-research.v1", "research_only": true,
            "training_semantics_version": 2,
            "promotion_eligible": false, "backend": "canonical_cpu", "combined_portfolio_evaluated": true,
            "initial_balance": initial, "account_currency": evaluation.account_currency, "original_risk_band": original_risk,
            "evaluated_risk_band": [evaluation.risk_per_trade_min, evaluation.risk_per_trade_max],
            "risk_policy": "fit and primary comparison cap the archived risk at 1%; archived-risk comparison reuses the frozen policy without refitting",
            "financial_limits": "scalar reference costs and account conversion; lot_grid=null means fractional research lots; no historical broker margin or stop-out evidence",
            "holdout_status": "chronological out-of-selection evaluation; this research envelope does not attest that the historical tail was previously unseen",
            "market_states": STATES, "training_source_candidates": result.candidates.len(), "parents_tested": n,
            "children_tested": genes.len()-n, "generation_scores": [first_score, score],
            "training_trials": { "parents": parent_trials, "with_children": child_trials },
            "training_candidate_ids": genes.iter().map(|gene| &gene.strategy_id).collect::<Vec<_>>(),
            "frozen_policy": frozen, "training_window": selection.evaluated_window(),
            "calibration_window": calibration.evaluated_window(), "holdout_window": holdout.evaluated_window(),
            "calibration": calibration_result, "holdout": final_result,
            "archived_risk_diagnostic": { "calibration": sealed_risk_calibration, "holdout": sealed_risk_holdout } }),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUARTER_HOUR: i64 = HOUR_MS / 4;

    fn rising_hours(hours: usize) -> Ohlcv {
        let close: Vec<_> = (0..hours * 4)
            .map(|row| 1.0 + (row / 4) as f64 * 0.001)
            .collect();
        Ohlcv {
            timestamp: Some(
                (0..close.len())
                    .map(|row| row as i64 * QUARTER_HOUR)
                    .collect(),
            ),
            open: close.clone(),
            high: close.iter().map(|value| value + 0.0001).collect(),
            low: close.iter().map(|value| value - 0.0001).collect(),
            close,
            volume: None,
        }
    }

    #[test]
    fn cash_admission_does_not_stop_training_exploration() {
        let first = Gene {
            indices: vec![0],
            weights: vec![1.],
            ..Default::default()
        };
        let mut second = first.clone();
        second.weights = vec![-1.];
        let mut genes = vec![first.clone(), second];
        let children = breed_training_children(&mut genes, [None; STATES.len()]).unwrap();
        assert!(children > 0 && children <= 4);
        assert_eq!(genes.len(), 2 + children);
        assert_eq!(genes[0].indices, first.indices);
        assert_eq!(genes[0].weights, first.weights);
        assert!(genes[2..].iter().all(|g| g.generation == 1));
    }

    #[test]
    fn partial_current_hour_cannot_change_its_own_route() {
        let mut bars = rising_hours(12);
        let before = market_states(&bars, QUARTER_HOUR).unwrap();
        assert!(before[..36].iter().all(Option::is_none));
        assert_eq!(before[36..40], [Some(0); 4]);
        for row in 36..40 {
            bars.high[row] += 0.5;
            bars.low[row] -= 0.5;
        }
        let after = market_states(&bars, QUARTER_HOUR).unwrap();
        assert_eq!(before[..40], after[..40]);
        assert_eq!(after[40], Some(3));
    }

    #[test]
    fn routes_are_identical_for_every_observed_prefix() {
        let bars = rising_hours(13);
        let complete = market_states(&bars, QUARTER_HOUR).unwrap();
        for end in 2..bars.close.len() {
            let prefix = neoethos_data::slice_ohlcv(&bars, 0, end, None);
            assert_eq!(
                market_states(&prefix, QUARTER_HOUR).unwrap(),
                complete[..end]
            );
        }
    }

    #[test]
    fn a_missing_bar_invalidates_context_immediately() {
        let mut bars = rising_hours(22);
        bars.timestamp.as_mut().unwrap().remove(41);
        for column in [
            &mut bars.open,
            &mut bars.high,
            &mut bars.low,
            &mut bars.close,
        ] {
            column.remove(41);
        }
        let states = market_states(&bars, QUARTER_HOUR).unwrap();
        assert_eq!(states[40], Some(0));
        assert_eq!(
            states[41], None,
            "10:30 cannot reuse context after missing 10:15"
        );
        assert_eq!(states[78], None);
        assert_eq!(
            states[79],
            Some(0),
            "nine complete hours restore context at 20:00"
        );
    }

    #[test]
    fn routing_refuses_an_unsupported_or_duplicate_clock() {
        let mut bars = rising_hours(2);
        assert!(market_states(&bars, 700_000).is_err());
        bars.timestamp.as_mut().unwrap()[1] = 0;
        assert!(market_states(&bars, QUARTER_HOUR).is_err());
    }

    #[test]
    fn specialist_selects_before_votes_cancel_and_keeps_its_own_brackets() {
        let features = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
            neoethos_data::test_fixtures::canonical_test_timestamps(4),
            vec!["up_setup".into(), "down_setup".into()],
            ndarray::array![[2., -4.], [0., -4.], [2., -4.], [2., -4.]],
        )
        .unwrap();
        let mut bars = rising_hours(1);
        bars.timestamp = Some(features.timestamps.clone());
        let long = Gene {
            indices: vec![0],
            weights: vec![1.],
            long_threshold: 1.,
            short_threshold: -1.,
            sl_pips: 10.,
            tp_pips: 30.,
            stop_vol_mult: 0.,
            ..Default::default()
        };
        let mut short = long.clone();
        short.indices = vec![1];
        short.sl_pips = 20.;
        short.tp_pips = 40.;
        let genes = [long, short];
        let evaluation = EvaluationConfig {
            pip_value: 0.0001,
            ..Default::default()
        };
        let adaptive = neoethos_search::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
            neoethos_search::stop_target::StopTargetSettings {
                vol_estimator: "parkinson".into(),
                ..Default::default()
            },
            false,
            2.,
        )
        .unwrap();
        let votes =
            ArchivedGeneVotes::prepare(&genes, &features, &bars, &evaluation, true, &adaptive)
                .unwrap();
        assert_eq!(votes.net(&[0, 1]).unwrap().directions[0].sign(), 0.0);
        let routed = Tape::routed(
            &votes,
            &[Some(0), Some(0), None, Some(1)],
            [Some(0), Some(1), None, None],
        )
        .unwrap();
        assert_eq!(routed.signals, [1, 0, 0, -1]);
        assert_eq!(routed.confidence, [0.5, 0., 0., 1.]);
        assert_eq!(routed.stops, [10., 0., 0., 20.]);
        assert_eq!(routed.targets, [30., 0., 0., 40.]);
    }
}
