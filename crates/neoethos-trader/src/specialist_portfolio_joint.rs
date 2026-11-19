//! Training-only genetic search of whole specialist books, followed by frozen
//! chronological replay. Historical archive reuse is explicit, not fresh OOS.
use super::*;
use neoethos_search::eval::{JointBarAccountPolicyV1, JointBarAccountResultV1, JointBarLaneV1};
use rand::Rng;
use std::collections::BTreeSet;

const EPISODES: usize = 4;
const EPISODE_DAYS: i64 = 180;
const POPULATION: usize = 16;
const GENERATIONS: usize = 6;
const MAX_EXPERTS_PER_STATE: usize = 4;

#[path = "specialist_portfolio_walkforward.rs"]
mod walkforward;
pub use walkforward::evaluate_walkforward_joint_risky;

#[path = "specialist_portfolio_continual.rs"]
mod continual;
pub use continual::evaluate as evaluate_continual;
pub use continual::evaluate_risky;
use neoethos_search::scoring::{RiskyGrowthGoal, ga_fitness_goal};

#[derive(Clone, Copy, Debug, Serialize)]
struct FitBudget {
    population: usize,
    generations: usize,
    seed: u64,
    reject_behavior_clones: bool,
}

/// Entries in each state allocate a share of ONE account risk band. Zero is
/// inactive. Several specialists may act together and keep different exits.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Genome {
    weights: [Vec<u8>; STATES.len()],
}

impl Genome {
    fn cash(n: usize) -> Self {
        Self {
            weights: std::array::from_fn(|_| vec![0; n]),
        }
    }
    fn validate(&self, n: usize) -> Result<()> {
        ensure!(
            self.weights.iter().all(|w| w.len() == n
                && w.iter().all(|&v| v <= 4)
                && w.iter().filter(|&&v| v > 0).count() <= MAX_EXPERTS_PER_STATE),
            "invalid joint portfolio activation/allocation genome"
        );
        Ok(())
    }
    fn active(&self) -> Vec<usize> {
        (0..self.weights[0].len())
            .filter(|&i| self.weights.iter().any(|w| w[i] > 0))
            .collect()
    }
    fn normalize(&self) -> [Vec<f64>; STATES.len()] {
        std::array::from_fn(|s| {
            let sum: f64 = self.weights[s].iter().map(|&v| f64::from(v)).sum();
            self.weights[s]
                .iter()
                .map(|&v| if sum > 0.0 { f64::from(v) / sum } else { 0.0 })
                .collect()
        })
    }
    fn random(n: usize, rng: &mut StdRng) -> Self {
        let mut value = Self::cash(n);
        for state in 0..STATES.len() {
            for _ in 0..rng.random_range(1..=MAX_EXPERTS_PER_STATE) {
                value.weights[state][rng.random_range(0..n)] = rng.random_range(1..=4);
            }
        }
        value
    }
    fn child(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        let mut child = a.clone();
        for s in 0..STATES.len() {
            if rng.random_bool(0.5) {
                child.weights[s].clone_from(&b.weights[s]);
            }
        }
        // Structural mutation changes activation and allocation together.
        let state = rng.random_range(0..STATES.len());
        let index = rng.random_range(0..child.weights[state].len());
        if child.weights[state][index] > 0
            || child.weights[state].iter().filter(|&&v| v > 0).count() < MAX_EXPERTS_PER_STATE
        {
            child.weights[state][index] = rng.random_range(0..=4);
        } else {
            let old = child.weights[state]
                .iter()
                .position(|&v| v > 0)
                .expect("full active state");
            child.weights[state][old] = 0;
            child.weights[state][index] = rng.random_range(1..=4);
        }
        child
    }
}

struct CachedWindow {
    window: Window,
    tapes: Vec<Tape>,
    expert_ids: Vec<String>,
    goal: Option<RiskyGrowthGoal>,
}

impl CachedWindow {
    fn new(window: Window, genes: &[Gene], policy: &LiveTradingPolicyV1) -> Result<Self> {
        Self::with_goal(window, genes, policy, None)
    }
    fn with_goal(
        window: Window,
        genes: &[Gene],
        policy: &LiveTradingPolicyV1,
        goal: Option<RiskyGrowthGoal>,
    ) -> Result<Self> {
        let votes = window.votes(genes, policy)?;
        let mut tapes = Vec::with_capacity(genes.len());
        for gene in 0..genes.len() {
            tapes.push(Tape::routed(
                &votes,
                &window.states,
                [Some(gene); STATES.len()],
            )?);
        }
        Ok(Self {
            goal,
            window,
            tapes,
            expert_ids: genes
                .iter()
                .enumerate()
                .map(|(i, g)| format!("expert_{i}:{}", g.strategy_id))
                .collect(),
        })
    }
    fn replay(
        &self,
        genome: &Genome,
        account_policy: JointBarAccountPolicyV1,
        trace: bool,
    ) -> Result<JointBarAccountResultV1> {
        genome.validate(self.tapes.len())?;
        let active = genome.active();
        let weights = genome.normalize();
        let allocation: Vec<Vec<f64>> = active
            .iter()
            .map(|&i| {
                self.window
                    .states
                    .iter()
                    .map(|state| state.map_or(0.0, |s| weights[s][i]))
                    .collect()
            })
            .collect();
        self.replay_allocations(&active, &allocation, account_policy, trace)
    }

    fn replay_allocations(
        &self,
        active: &[usize],
        allocation: &[Vec<f64>],
        account_policy: JointBarAccountPolicyV1,
        trace: bool,
    ) -> Result<JointBarAccountResultV1> {
        ensure!(
            active.len() == allocation.len(),
            "expert allocations must align"
        );
        let lanes: Vec<_> = active
            .iter()
            .zip(allocation)
            .map(|(&i, allocation)| {
                let tape = &self.tapes[i];
                JointBarLaneV1 {
                    expert_id: &self.expert_ids[i],
                    tape: NettedBarDecisionTapeV1 {
                        signals: &tape.signals,
                        confidences: &tape.confidence,
                        sl_pips: &tape.stops,
                        tp_pips: &tape.targets,
                        ml_multipliers: allocation,
                    },
                }
            })
            .collect();
        match self.goal {
            Some(goal) => {
                self.window
                    .account
                    .evaluate_joint_risky(&lanes, account_policy, None, trace, goal)
            }
            None => self
                .window
                .account
                .evaluate_joint(&lanes, account_policy, None, trace),
        }
    }
}

fn episode_ranges(times: &[i64], start: usize, end: usize) -> Result<Vec<std::ops::Range<usize>>> {
    ensure!(
        start < end && end <= times.len(),
        "invalid joint training range"
    );
    let duration = EPISODE_DAYS * 86_400_000;
    let mut ranges = Vec::new();
    for i in 1..=EPISODES {
        let stop = start + (end - start) * i / EPISODES;
        let begin = times[..stop]
            .partition_point(|&t| t < times[stop - 1] - duration)
            .max(start);
        ensure!(
            stop - begin >= 2
                && ranges
                    .last()
                    .is_none_or(|r: &std::ops::Range<usize>| r.end <= begin),
            "joint training needs four disjoint 180-day episodes"
        );
        ranges.push(begin..stop);
    }
    Ok(ranges)
}

fn monthly(
    result: &JointBarAccountResultV1,
    times: &[i64],
) -> Result<Vec<neoethos_search::quality::MonthlyStrategyMetricsV1>> {
    let trades: Vec<_> = result
        .closed_trades
        .iter()
        .map(|v| v.trade.clone())
        .collect();
    monthly_closed_trade_metrics_v1(
        &trades,
        result.valuation.initial_balance,
        times[0],
        result
            .risky_goal
            .as_ref()
            .map_or(*times.last().unwrap(), |g| {
                (*times.last().unwrap()).min(g.deadline_timestamp_ms)
            }),
    )
}

fn score(result: &JointBarAccountResultV1, times: &[i64]) -> Result<f64> {
    if result.valuation.drawdown.capital_bound_breached
        || result.valuation.ending_modeled_net_equity <= 0.0
    {
        return Ok(-1e6);
    }
    if let Some(goal) = &result.risky_goal {
        // Cash is a zero-progress control. Risky uses the SAME configured
        // goal-pace scorer as discovery, without PropFirm monthly/DD weights.
        // Pending gains cannot boost fitness; pending losses reduce it.
        if result.closed_trades.is_empty() {
            return Ok(0.0);
        }
        let mut metrics = [0.0; 11];
        metrics[0] = result
            .valuation
            .ending_realized_balance
            .min(result.valuation.ending_modeled_net_equity)
            - result.valuation.initial_balance;
        metrics[3] = result.valuation.drawdown.max_close_equity_drawdown;
        metrics[8] = result.closed_trades.len() as f64;
        let span_days =
            (result.valuation.last_processed_timestamp_ms - times[0]) as f64 / 86_400_000.0;
        let value = ga_fitness_goal(
            &metrics,
            result.valuation.initial_balance,
            span_days,
            goal.goal,
        );
        return Ok(if value.is_finite() { value } else { -1e6 });
    }
    let months = monthly(result, times)?;
    let mut growth = 0.0;
    for month in &months {
        if month.ending_balance <= 0.0 || month.starting_balance <= 0.0 {
            return Ok(-1e6);
        }
        let change = (month.ending_balance / month.starting_balance).ln();
        growth += change + change.min(0.0);
    }
    // Pending open losses count; unrealized gains cannot improve fitness.
    growth += ((result.valuation.ending_modeled_net_equity
        / result.valuation.ending_realized_balance)
        .ln())
    .min(0.0);
    Ok((growth
        - result
            .valuation
            .drawdown
            .max_adverse_bar_drawdown_bound_from_close_peak)
        / months.len() as f64)
}

fn fit(
    windows: &[CachedWindow],
    n: usize,
    policy: JointBarAccountPolicyV1,
) -> Result<(Genome, Vec<Value>)> {
    fit_with_budget(
        windows,
        n,
        policy,
        FitBudget {
            population: POPULATION,
            generations: GENERATIONS,
            seed: SEED + 1,
            reject_behavior_clones: false,
        },
        &[],
    )
}

fn fit_with_budget(
    windows: &[CachedWindow],
    n: usize,
    policy: JointBarAccountPolicyV1,
    budget: FitBudget,
    warm_starts: &[Genome],
) -> Result<(Genome, Vec<Value>)> {
    ensure!(
        (8..=32).contains(&budget.population)
            && (1..=8).contains(&budget.generations)
            && !windows.is_empty()
            && n > 0,
        "invalid bounded portfolio fit"
    );
    let mut rng = StdRng::seed_from_u64(budget.seed);
    let behavior_classes = budget
        .reject_behavior_clones
        .then(|| continual::behavior_classes(windows, n));
    let mut population = seed_population(n, budget.population, warm_starts, &mut rng)?;
    let mut trials = Vec::new();
    let mut cache = std::collections::BTreeMap::<String, f64>::new();
    let mut best = Genome::cash(n);
    for generation in 0..budget.generations {
        let mut ranked = Vec::new();
        for genome in population {
            let key = serde_json::to_string(&genome)?;
            let fitness = if let Some(value) = cache.get(&key) {
                *value
            } else {
                let mut episode_scores = Vec::new();
                let mut counts = Vec::new();
                let mut money = Vec::new();
                let behavior_clone = behavior_classes
                    .as_ref()
                    .is_some_and(|classes| continual::has_behavior_clones(classes, &genome));
                for window in windows {
                    let result = window.replay(&genome, policy, false)?;
                    episode_scores.push(score(&result, window.window.account.timestamps())?);
                    counts.push(result.closed_trades.len());
                    money.push(json!({"net_profit":result.valuation.ending_realized_balance-result.valuation.initial_balance,
                        "ending_realized_balance":result.valuation.ending_realized_balance,
                        "pending_cost_net_equity":result.valuation.ending_modeled_net_equity,
                        "entry_outcomes":result.admission.entry_outcomes,
                        "risky_goal":result.risky_goal,
                        "max_adverse_bar_dd_bound":result.valuation.drawdown.max_adverse_bar_drawdown_bound_from_close_peak,
                        "capital_bound_breached":result.valuation.drawdown.capital_bound_breached}));
                }
                let average = episode_scores.iter().sum::<f64>() / windows.len() as f64;
                let spread = (episode_scores
                    .iter()
                    .map(|v| (v - average).powi(2))
                    .sum::<f64>()
                    / windows.len() as f64)
                    .sqrt();
                let value = if behavior_clone {
                    -1e6
                } else {
                    average - 0.25 * spread
                };
                trials.push(
                    json!({"generation":generation,"genome":genome,"episode_scores":episode_scores,
                    "episode_closed_trades":counts,"episode_money":money,
                    "behavior_clone_rejected":behavior_clone,"fitness":value}),
                );
                cache.insert(key, value);
                value
            };
            ranked.push((fitness, genome));
        }
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        best = ranked[0].1.clone();
        eprintln!(
            "joint_portfolio generation={generation} unique_books={} best_training_fitness={:.8} active_experts={}",
            cache.len(),
            ranked[0].0,
            best.active().len()
        );
        population = ranked.iter().take(4).map(|v| v.1.clone()).collect();
        while population.len() < budget.population {
            population.push(Genome::child(
                &ranked[rng.random_range(0..8)].1,
                &ranked[rng.random_range(0..8)].1,
                &mut rng,
            ));
        }
    }
    Ok((best, trials))
}

fn seed_population(
    n: usize,
    capacity: usize,
    warm_starts: &[Genome],
    rng: &mut StdRng,
) -> Result<Vec<Genome>> {
    let mut population = vec![Genome::cash(n)];
    for book in warm_starts {
        book.validate(n)?;
        if !population.contains(book) {
            population.push(book.clone());
        }
    }
    ensure!(
        population.len() <= capacity,
        "warm starts exceed the fixed fit budget"
    );
    while population.len() < capacity {
        population.push(Genome::random(n, rng));
    }
    Ok(population)
}

fn ledger(result: JointBarAccountResultV1, times: &[i64]) -> Result<Value> {
    let months = monthly(&result, times)?;
    let days: BTreeSet<_> = result
        .closed_trades
        .iter()
        .map(|t| t.trade.entry_time / 86_400_000)
        .chain(
            result
                .terminal_open
                .iter()
                .map(|t| t.entry_timestamp_ms / 86_400_000),
        )
        .collect();
    let mut streak = 0;
    let mut longest = 0;
    for trade in &result.closed_trades {
        if trade.trade.pnl < 0.0 {
            streak += 1;
            longest = longest.max(streak);
        } else {
            streak = 0;
        }
    }
    let summary = json!({"net_return_pct":100.*(result.valuation.ending_realized_balance/result.valuation.initial_balance-1.),
        "profitable_months":months.iter().filter(|m|m.net_profit>0.0).count(),"calendar_months":months.len(),
        "worst_month_return_pct":months.iter().filter_map(|m|m.return_pct).min_by(f64::total_cmp),
        "active_entry_days":days.len(),"longest_closed_trade_losing_streak":longest,
        "distinct_experts_traded":result.closed_trades.iter().map(|t|&t.expert_id).collect::<BTreeSet<_>>().len()});
    Ok(json!({"summary":summary,"monthly":months,"account":result}))
}

fn research_recipe(
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<(&LiveTradingPolicyV1, EvaluationConfig, [f64; 2])> {
    research_recipe_for_mode(research, false)
}

fn research_recipe_for_mode(
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    risky: bool,
) -> Result<(&LiveTradingPolicyV1, EvaluationConfig, [f64; 2])> {
    let source = research.discovery_result();
    let policy = source
        .funnel_profile
        .as_ref()
        .and_then(|f| f.live_trading_policy_v1())
        .context("missing sealed specialist policy")?;
    policy.validate()?;
    ensure!(
        policy.source_search_config_hash == source.search_config_hash,
        "mismatched specialist recipe"
    );
    let mut evaluation = policy.sealed_evaluation_config()?.clone();
    let original = [evaluation.risk_per_trade_min, evaluation.risk_per_trade_max];
    if risky {
        validate_risky_recipe(&evaluation)?;
    } else {
        evaluation.risk_per_trade_max = evaluation.risk_per_trade_max.min(RISK_CAP);
        evaluation.risk_per_trade_min = evaluation
            .risk_per_trade_min
            .min(evaluation.risk_per_trade_max);
    }
    Ok((policy, evaluation, original))
}

fn validate_risky_recipe(evaluation: &EvaluationConfig) -> Result<RiskyGrowthGoal> {
    ensure!(
        evaluation.growth_objective,
        "Risky specialists require a sealed Risky growth recipe"
    );
    let goal = evaluation
        .growth_goal
        .context("Risky specialists require a configured goal/deadline")?;
    goal.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        evaluation.initial_equity == goal.start_balance,
        "Risky specialists must replay at the configured goal's starting capital"
    );
    ensure!(
        evaluation.risk_per_trade_min.is_finite()
            && evaluation.risk_per_trade_max.is_finite()
            && evaluation.risk_per_trade_min >= 0.0
            && evaluation.risk_per_trade_max >= evaluation.risk_per_trade_min
            && evaluation.risk_per_trade_max > 0.0
            && evaluation.risk_per_trade_max <= 1.0,
        "invalid sealed Risky sizing band"
    );
    Ok(goal)
}

fn risky_account_policy(evaluation: &EvaluationConfig) -> Result<JointBarAccountPolicyV1> {
    validate_risky_recipe(evaluation)?;
    let policy = JointBarAccountPolicyV1 {
        // One aggregate budget, not four copies of the per-trade ceiling.
        max_entry_stop_risk_fraction: evaluation.risk_per_trade_max,
        ..joint_account_policy()
    };
    policy.validate()?;
    Ok(policy)
}

fn joint_account_policy() -> JointBarAccountPolicyV1 {
    JointBarAccountPolicyV1 {
        max_open_positions: 4,
        max_entry_stop_risk_fraction: RISK_CAP,
        max_gross_notional_multiple: 30.0,
        max_target_pips: 20.0,
    }
}

pub fn evaluate(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let source = research.discovery_result();
        let (policy, evaluation, original_risk) = research_recipe(research)?;
        let account_policy = joint_account_policy();
        let calibration = source
            .calibration_scope
            .as_ref()
            .context("missing calibration")?;
        let holdout = source.holdout_scope.as_ref().context("missing holdout")?;
        let selection = &source.selection_scope;
        ensure!(
            selection.evaluated_window().role() == CanonicalSearchWindowRoleV1::InSample
                && calibration.evaluated_window().role()
                    == CanonicalSearchWindowRoleV1::SelectionValidation
                && holdout.evaluated_window().role() == CanonicalSearchWindowRoleV1::Holdout
                && selection.evaluated_window().row_end()
                    <= calibration.evaluated_window().row_start()
                && calibration.evaluated_window().row_end()
                    <= holdout.evaluated_window().row_start(),
            "overlapping joint scopes"
        );
        let times = input
            .ohlcv()
            .timestamp
            .as_deref()
            .context("missing timestamps")?;
        let start = usize::try_from(selection.evaluated_window().row_start())?;
        let end = usize::try_from(selection.evaluated_window().row_end())?;
        let period = times[start..end]
            .windows(2)
            .map(|w| w[1] - w[0])
            .min()
            .context("missing base cadence")?;
        let eligible: Vec<_> = source
            .candidates
            .iter()
            .filter(|g| g.tp_pips.is_finite() && g.tp_pips > 0.0 && g.tp_pips <= 20.0)
            .collect();
        let parents = PARENTS.min(eligible.len());
        ensure!(
            parents >= 2,
            "at least two training experts within normal TP<=20 required"
        );
        let mut genes: Vec<_> = (0..parents)
            .map(|i| eligible[i * eligible.len() / parents].clone())
            .collect();
        breed_training_children(&mut genes, [None; STATES.len()])?;
        let ranges = episode_ranges(times, start, end)?;
        let scopes: Vec<_> = ranges
            .into_iter()
            .map(|range| {
                CanonicalSearchArtifactScopeV2::from_run_input_range(
                    CanonicalSearchWindowRoleV1::InSample,
                    input,
                    range,
                )
            })
            .collect::<std::result::Result<_, _>>()?;
        let training: Vec<_> = scopes
            .iter()
            .map(|scope| {
                CachedWindow::new(
                    Window::open(input, research, scope, &evaluation, period)?,
                    &genes,
                    policy,
                )
            })
            .collect::<Result<_>>()?;
        let (genome, trials) = fit(&training, genes.len(), account_policy)?;
        drop(training);
        // Lock the whole policy BEFORE opening either later window. The
        // historical archive and tail were seen before: diagnostic evidence.
        let frozen = json!({"schema":"neoethos.joint-specialist-policy.v2","source_evidence_sha256":research.evidence_identity_sha256(),
            "base_period_ms":period,"feature_names":source.effective_feature_names,"market_states":STATES,
            "genes":genes,"activation_allocation":genome,"account_admission":account_policy,
            "evaluated_risk_band":[evaluation.risk_per_trade_min,evaluation.risk_per_trade_max]});
        let mut baseline = Genome::cash(genes.len());
        for weights in &mut baseline.weights {
            for weight in weights.iter_mut().take(4) {
                *weight = 1;
            }
        }
        let compare = |scope| -> Result<Value> {
            let window = CachedWindow::new(
                Window::open(input, research, scope, &evaluation, period)?,
                &genes,
                policy,
            )?;
            let times = window.window.account.timestamps();
            Ok(
                json!({"learned_joint_portfolio":ledger(window.replay(&genome,account_policy,true)?,times)?,
                "predeclared_first_four_equal_risk_diagnostic":ledger(window.replay(&baseline,account_policy,true)?,times)?,
                "cash_ending_balance":evaluation.initial_equity}),
            )
        };
        let calibration_result = compare(calibration)?;
        let holdout_result = compare(holdout)?;
        Ok(
            json!({"schema":"neoethos.joint-specialist-portfolio-research.v2","research_only":true,
            "promotion_eligible":false,"backend":"canonical_cpu","combined_portfolio_evaluated":true,
            "original_risk_band":original_risk,"training_episodes":scopes,"training_trials":trials,
            "seed":SEED+1,"population":POPULATION,"generations":GENERATIONS,"source_candidates":source.candidates.len(),
            "eligible_normal_tp_candidates":eligible.len(),"archived_parents":parents,"expert_crossover_children":genes.len()-parents,
            "fitness":"mean_episode((sum(month_log_growth+negative_month_log_growth)+negative_pending_open_log_growth-adverse_bar_dd_bound)/calendar_months)-0.25*episode_score_stddev; cash=0",
            "training_limit":"bounded crossover of archived training experts plus genetic joint activation/allocation; four disjoint training episodes are stability checks, not independent inner OOS",
            "holdout_status":"previously inspected historical tail; chronological diagnostic, not fresh independent OOS or promotion",
            "financial_limits":"reference spread/commission/carry/conversion; fractional lots unless sealed checked grid; separate hedged sleeves; no broker quotes, margin, stopout or challenge pass claim",
            "frozen_policy":frozen,"calibration":calibration_result,"holdout":holdout_result}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn risky_recipe_preserves_its_sizing_and_refuses_propfirm_or_other_capital() {
        let mut evaluation = EvaluationConfig {
            growth_objective: true,
            initial_equity: 100.,
            growth_goal: Some(RiskyGrowthGoal {
                start_balance: 100.,
                target_balance: 50000.,
                horizon_days: 180.,
            }),
            risk_per_trade_min: 0.23,
            risk_per_trade_max: 0.30,
            ..Default::default()
        };
        let goal = validate_risky_recipe(&evaluation).unwrap();
        assert_eq!(goal.horizon_days, 180.);
        assert_eq!(goal.target_balance, 50000.);
        let policy = risky_account_policy(&evaluation).unwrap();
        assert_eq!(policy.max_entry_stop_risk_fraction, 0.30);
        assert_eq!(policy.max_gross_notional_multiple, 30.);
        assert_eq!(policy.max_target_pips, 20.);
        evaluation.initial_equity = 10000.;
        assert!(validate_risky_recipe(&evaluation).is_err());
        evaluation.initial_equity = 100.;
        evaluation.growth_objective = false;
        assert!(validate_risky_recipe(&evaluation).is_err());
    }
    #[test]
    fn joint_allocations_allow_many_experts_but_only_one_risk_budget() {
        let genome = Genome {
            weights: [vec![1, 3], vec![0, 2], vec![0, 0], vec![4, 4]],
        };
        genome.validate(2).unwrap();
        assert_eq!(genome.active(), [0, 1]);
        assert_eq!(
            genome.normalize(),
            [vec![0.25, 0.75], vec![0., 1.], vec![0., 0.], vec![0.5, 0.5]]
        );
    }
    #[test]
    fn joint_evolution_is_seeded_and_preserves_bounded_activation() {
        let generate = || {
            let mut rng = StdRng::seed_from_u64(42);
            let a = Genome::random(17, &mut rng);
            let b = Genome::random(17, &mut rng);
            let mut child = a;
            for _ in 0..100 {
                child = Genome::child(&child, &b, &mut rng);
                child.validate(17).unwrap();
            }
            child
        };
        assert_eq!(generate(), generate());
        assert!(
            Genome {
                weights: std::array::from_fn(|_| vec![1; 5])
            }
            .validate(5)
            .is_err()
        );
    }
    #[test]
    fn episodes_are_disjoint_and_never_enter_later_windows() {
        let times: Vec<_> = (0..4000).map(|d| d * 86_400_000).collect();
        let ranges = episode_ranges(&times, 100, 3600).unwrap();
        assert_eq!(ranges.len(), 4);
        assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));
        assert!(ranges.iter().all(|r| r.start >= 100 && r.end <= 3600));
        assert!(episode_ranges(&times, 100, 500).is_err());
    }
}
