//! Versioned prequential research: train on older history, validate on a
//! separate recent interval, then freeze entry allocations for the next month.
//! Open positions retain their original protection across allocation updates.
use super::*;
use neoethos_search::genetic::{
    build_rolling_regime_windows, gene_signature_hash, month_day_indices,
};
use std::{
    hash::{Hash, Hasher},
    ops::Range,
};

const BANK_LIMIT: usize = 256;
const TRAIN_DAYS: i64 = 180;
const VALIDATION_DAYS: i64 = 60;
const NEW_CHILDREN_PER_UPDATE: usize = 2;
const MAX_UPDATES_PER_FORWARD: usize = 24;

pub(super) fn behavior_classes(
    windows: &[CachedWindow],
    n: usize,
) -> Vec<[Option<u64>; STATES.len()]> {
    (0..n)
        .map(|gene| {
            std::array::from_fn(|state| {
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                let mut observations = 0usize;
                for (segment, window) in windows.iter().enumerate() {
                    let tape = &window.tapes[gene];
                    for (row, current) in window.window.states.iter().enumerate() {
                        if *current == Some(state) && tape.signals[row] != 0 {
                            observations += 1;
                            (
                                segment,
                                row,
                                tape.signals[row],
                                tape.stops[row].to_bits(),
                                tape.targets[row].to_bits(),
                            )
                                .hash(&mut hash);
                        }
                    }
                }
                (observations > 0).then(|| hash.finish())
            })
        })
        .collect()
}

pub(super) fn has_behavior_clones(
    classes: &[[Option<u64>; STATES.len()]],
    genome: &Genome,
) -> bool {
    genome.weights.iter().enumerate().any(|(state, weights)| {
        let mut seen = BTreeSet::new();
        weights.iter().enumerate().any(|(gene, &weight)| {
            weight > 0 && classes[gene][state].is_some_and(|class| !seen.insert(class))
        })
    })
}

/// All ranges exclude the decision bar and a complete maximum-hold embargo.
/// The gap between fit and validation also purges overlapping outcome spans.
fn past_ranges(
    times: &[i64],
    decision: usize,
    max_hold: usize,
) -> Result<(Range<usize>, Range<usize>)> {
    ensure!(
        decision < times.len() && max_hold > 0,
        "continual research requires a finite positive holding horizon"
    );
    let embargo = max_hold
        .checked_add(1)
        .context("holding embargo overflow")?;
    let end = decision
        .checked_sub(embargo)
        .context("insufficient matured history")?;
    let validation_time = times[decision]
        .checked_sub(VALIDATION_DAYS * 86_400_000)
        .context("validation time overflow")?;
    let start_time = validation_time
        .checked_sub(TRAIN_DAYS * 86_400_000)
        .context("training time overflow")?;
    let validation_start = times[..end].partition_point(|&t| t < validation_time);
    let train_start = times[..validation_start].partition_point(|&t| t < start_time);
    let train_end = validation_start
        .checked_sub(embargo)
        .context("missing fit-validation embargo")?;
    ensure!(
        train_start < train_end
            && train_end < validation_start
            && validation_start < end
            && end < decision,
        "invalid chronological continual fit/validation ranges"
    );
    ensure!(
        times[0] <= start_time,
        "archive does not cover complete rolling history"
    );
    Ok((train_start..train_end, validation_start..end))
}

fn month_ranges(times: &[i64], forward: Range<usize>) -> Result<Vec<Range<usize>>> {
    ensure!(
        forward.start < forward.end && forward.end <= times.len(),
        "invalid forward interval"
    );
    let (months, _) = month_day_indices(&times[forward.clone()]);
    let mut starts = vec![forward.start];
    starts.extend(
        months
            .windows(2)
            .enumerate()
            .filter(|(_, m)| m[0] != m[1])
            .map(|(row, _)| forward.start + row + 1),
    );
    starts.push(forward.end);
    Ok(starts.windows(2).map(|p| p[0]..p[1]).collect())
}

fn padded(genome: &Genome, n: usize) -> Result<Genome> {
    let old = genome.weights[0].len();
    ensure!(
        old <= n,
        "an archived expert cannot disappear from a scheduled book"
    );
    genome.validate(old)?;
    let mut value = genome.clone();
    for weights in &mut value.weights {
        weights.resize(n, 0);
    }
    Ok(value)
}

fn renew_bank(genes: &mut Vec<Gene>, trained: &Genome, seed: u64) -> Result<Vec<Value>> {
    let mut parents = trained.active();
    if parents.len() < 2 {
        parents = (0..genes.len()).collect();
    }
    ensure!(
        parents.len() >= 2 && genes.len() <= BANK_LIMIT,
        "invalid bounded specialist bank"
    );
    let mut seen: BTreeSet<_> = genes.iter().map(gene_signature_hash).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut births = Vec::new();
    for _ in 0..32 {
        if births.len() == NEW_CHILDREN_PER_UPDATE || genes.len() == BANK_LIMIT {
            break;
        }
        let a = parents[rng.random_range(0..parents.len())];
        let b = parents[rng.random_range(0..parents.len())];
        if a == b {
            continue;
        }
        let generation = genes[a]
            .generation
            .max(genes[b].generation)
            .checked_add(1)
            .context("expert generation overflow")?;
        let child = neoethos_search::genetic::crossover(&genes[a], &genes[b], generation, &mut rng);
        ensure!(
            child.tp_pips > 0.0 && child.tp_pips <= 20.0,
            "renewal cannot inflate a normal target"
        );
        if seen.insert(gene_signature_hash(&child)) {
            births.push(json!({"index":genes.len(),"strategy_id":child.strategy_id,
                "parent_indices":[a,b],"generation":generation}));
            genes.push(child);
        }
    }
    Ok(births)
}

fn prepare_training(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    range: Range<usize>,
    evaluation: &EvaluationConfig,
    period: i64,
    genes: &[Gene],
    policy: &LiveTradingPolicyV1,
    goal: Option<RiskyGrowthGoal>,
) -> Result<Vec<CachedWindow>> {
    let times = input
        .ohlcv()
        .timestamp
        .as_deref()
        .context("missing timestamps")?;
    let windows = build_rolling_regime_windows(&times[range.clone()], 45, 45, 500);
    ensure!(
        (3..=4).contains(&windows.len()),
        "rolling training requires three or four non-overlapping subwindows"
    );
    windows
        .iter()
        .map(|window| {
            let rows = range.start + window.start_idx..range.start + window.end_idx;
            let scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
                CanonicalSearchWindowRoleV1::InSample,
                input,
                rows,
            )?;
            CachedWindow::with_goal(
                Window::open(input, research, &scope, evaluation, period)?,
                genes,
                policy,
                goal,
            )
        })
        .collect()
}

fn choose_on_validation(
    window: &CachedWindow,
    challenger: &Genome,
    incumbent: &Genome,
    account_policy: JointBarAccountPolicyV1,
) -> Result<(Genome, Value)> {
    let minimum = neoethos_search::genetic::RegimeLabelPolicy::default().min_trades_per_window;
    ensure!(
        minimum.is_finite() && minimum >= 2.0 && minimum <= usize::MAX as f64,
        "invalid existing regime evidence floor"
    );
    let minimum = minimum.ceil() as usize;
    let n = window.tapes.len();
    let cash = Genome::cash(n);
    let incumbent = padded(incumbent, n)?;
    let challenger = padded(challenger, n)?;
    let mut chosen = cash.clone();
    let mut best = 0.0;
    let mut scores = Vec::new();
    // Stable ties favor cash, then the incumbent; churn must improve validation.
    for (name, book) in [("incumbent", &incumbent), ("challenger", &challenger)] {
        let result = window.replay(book, account_policy, false)?;
        let value = score(&result, window.window.account.timestamps())?;
        scores.push(json!({"candidate":name,"score":value,"net_profit":result.valuation.ending_realized_balance-result.valuation.initial_balance,
            "closed_trades":result.closed_trades.len(),"minimum_closed_trades":minimum,
            "evidence_floor_met":result.closed_trades.len()>=minimum,"capital_bound_breached":result.valuation.drawdown.capital_bound_breached}));
        if candidate_can_replace(best, value, result.closed_trades.len(), minimum) {
            best = value;
            chosen = book.clone();
        }
    }
    Ok((
        chosen,
        json!({"cash_score":0.0,"candidates":scores,"chosen_score":best}),
    ))
}

fn candidate_can_replace(best: f64, score: f64, trades: usize, minimum: usize) -> bool {
    trades >= minimum && score > best + 1e-12
}

fn scheduled_allocations(
    states: &[Option<usize>],
    schedule: &[(usize, Genome)],
    n: usize,
) -> Result<(Vec<usize>, Vec<Vec<f64>>)> {
    ensure!(
        !schedule.is_empty()
            && schedule[0].0 == 0
            && schedule.windows(2).all(|s| s[0].0 < s[1].0)
            && schedule.last().unwrap().0 < states.len(),
        "invalid monthly book schedule"
    );
    let mut allocation = vec![vec![0.0; states.len()]; n];
    for (index, (start, book)) in schedule.iter().enumerate() {
        let weights = padded(book, n)?.normalize();
        let end = schedule.get(index + 1).map_or(states.len(), |next| next.0);
        for row in *start..end {
            if let Some(state) = states[row] {
                ensure!(state < STATES.len(), "unknown market state");
                for gene in 0..n {
                    allocation[gene][row] = weights[state][gene];
                }
            }
        }
    }
    let active: Vec<_> = (0..n)
        .filter(|&gene| allocation[gene].iter().any(|&v| v > 0.0))
        .collect();
    let lanes = active
        .iter()
        .map(|&gene| std::mem::take(&mut allocation[gene]))
        .collect();
    Ok((active, lanes))
}

#[allow(clippy::too_many_arguments)]
fn forward(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    forward_scope: &CanonicalSearchArtifactScopeV2,
    evaluation: &EvaluationConfig,
    period: i64,
    genes: &mut Vec<Gene>,
    policy: &LiveTradingPolicyV1,
    account_policy: JointBarAccountPolicyV1,
    goal: Option<RiskyGrowthGoal>,
) -> Result<Value> {
    let times = input
        .ohlcv()
        .timestamp
        .as_deref()
        .context("missing timestamps")?;
    let bounds = forward_scope.evaluated_window();
    let start = usize::try_from(bounds.row_start())?;
    let end = usize::try_from(bounds.row_end())?;
    let intervals = month_ranges(times, start..end)?;
    let intervals: Vec<_> = intervals
        .into_iter()
        .filter(|r| {
            goal.is_none_or(|g| {
                ((times[r.start] - times[start]) as f64) < g.horizon_days * 86_400_000.0
            })
        })
        .collect();
    ensure!(
        intervals.len() <= MAX_UPDATES_PER_FORWARD,
        "forward interval exceeds predeclared monthly update budget"
    );
    let mut schedule = Vec::new();
    let mut updates = Vec::new();
    let mut incumbent = Genome::cash(genes.len());
    for (update, interval) in intervals.iter().enumerate() {
        let (fit_range, validation_range) =
            past_ranges(times, interval.start, evaluation.max_hold_bars)?;
        let budget = FitBudget {
            population: 8,
            generations: 3,
            seed: (times[interval.start] as u64) ^ (SEED + 5),
            reject_behavior_clones: true,
        };
        let training = prepare_training(
            input,
            research,
            fit_range.clone(),
            evaluation,
            period,
            genes,
            policy,
            goal,
        )?;
        let (first, first_trials) = fit_with_budget(
            &training,
            genes.len(),
            account_policy,
            budget,
            &[padded(&incumbent, genes.len())?],
        )?;
        drop(training);
        // New experts are derived only from the older fit; validation cannot
        // choose their parents, generate offspring or change the fit budget.
        let births = renew_bank(genes, &first, budget.seed ^ 0x9e37)?;
        let training = prepare_training(
            input,
            research,
            fit_range.clone(),
            evaluation,
            period,
            genes,
            policy,
            goal,
        )?;
        let (challenger, child_trials) = fit_with_budget(
            &training,
            genes.len(),
            account_policy,
            FitBudget {
                seed: budget.seed ^ 0x51b7,
                ..budget
            },
            &[
                padded(&first, genes.len())?,
                padded(&incumbent, genes.len())?,
            ],
        )?;
        drop(training);
        let scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::InSample,
            input,
            validation_range.clone(),
        )?;
        let validation = CachedWindow::with_goal(
            Window::open(input, research, &scope, evaluation, period)?,
            genes,
            policy,
            goal,
        )?;
        let (chosen, validation_result) =
            choose_on_validation(&validation, &challenger, &incumbent, account_policy)?;
        let changed = chosen != padded(&incumbent, genes.len())?;
        incumbent = chosen.clone();
        schedule.push((interval.start - start, chosen.clone()));
        updates.push(json!({"decision_row":interval.start,"decision_timestamp_ms":times[interval.start],
            "forward_rows":[interval.start,interval.end],"fit_rows":[fit_range.start,fit_range.end],
            "validation_rows":[validation_range.start,validation_range.end],"max_label_row_exclusive":validation_range.end,
            "embargo_bars":evaluation.max_hold_bars+1,"budget":budget,"bank_size":genes.len(),"births":births,
            "selected_book":chosen,"changed":changed,"validation":validation_result,
            "training_trials":{"parents":first_trials,"with_children":child_trials}}));
        eprintln!(
            "continual_specialists update={update} asof_ms={} bank={} active={} changed={changed}",
            times[interval.start],
            genes.len(),
            incumbent.active().len()
        );
    }
    let window = CachedWindow::with_goal(
        Window::open(input, research, forward_scope, evaluation, period)?,
        genes,
        policy,
        goal,
    )?;
    let (active, allocations) =
        scheduled_allocations(&window.window.states, &schedule, genes.len())?;
    let adaptive = window.replay_allocations(&active, &allocations, account_policy, true)?;
    let first = &schedule[0].1;
    let frozen = window.replay(&padded(first, genes.len())?, account_policy, true)?;
    Ok(
        json!({"updates":updates,"adaptive":ledger(adaptive,window.window.account.timestamps())?,
        "frozen_first_book":ledger(frozen,window.window.account.timestamps())?,"forward_window":bounds,
        "cash_ending_balance":evaluation.initial_equity}),
    )
}

pub fn evaluate(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    evaluate_mode(input, research, false)
}

pub fn evaluate_risky(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
) -> Result<Value> {
    evaluate_mode(input, research, true)
}

/// Keep explicit V3 reference semantics; V4 binds mode, money and deadline.
fn evaluate_mode(
    input: &CanonicalSearchRunInputV2<'_>,
    research: &CanonicalTrendbarResearchDiscoveryResultV3,
    risky: bool,
) -> Result<Value> {
    neoethos_search::discovery::with_saved_research_execution(input, research, || {
        let source = research.discovery_result();
        let (policy, evaluation, original_risk) = research_recipe_for_mode(research, risky)?;
        let goal = if risky {
            Some(validate_risky_recipe(&evaluation)?)
        } else {
            None
        };
        let calibration = source
            .calibration_scope
            .as_ref()
            .context("missing calibration")?;
        let holdout = source
            .holdout_scope
            .as_ref()
            .context("missing forward scope")?;
        ensure!(
            source.selection_scope.evaluated_window().row_end()
                <= calibration.evaluated_window().row_start()
                && calibration.evaluated_window().row_end()
                    <= holdout.evaluated_window().row_start(),
            "overlapping source windows"
        );
        ensure!(
            evaluation.max_hold_bars > 0,
            "continual protocol needs a finite holding horizon"
        );
        let times = input
            .ohlcv()
            .timestamp
            .as_deref()
            .context("missing timestamps")?;
        // A separate account starts at each predeclared forward scope. V4
        // evaluates exactly one goal horizon there, never fourteen months
        // relabeled as a six-month challenge. A coverage-only sentinel bar
        // after the deadline may be included; its prices are not replayed.
        let bounded_scope =
            |scope: &CanonicalSearchArtifactScopeV2| -> Result<CanonicalSearchArtifactScopeV2> {
                let bounds = scope.evaluated_window();
                let start = usize::try_from(bounds.row_start())?;
                let mut end = usize::try_from(bounds.row_end())?;
                if let Some(goal) = goal {
                    let duration = goal.horizon_days * 86_400_000.0;
                    ensure!(
                        duration < i64::MAX as f64,
                        "Risky deadline overflows milliseconds"
                    );
                    let deadline = times[start]
                        .checked_add(duration as i64)
                        .context("Risky deadline overflow")?;
                    let after = times[..end].partition_point(|&t| t <= deadline);
                    end = end.min(after.saturating_add(1));
                }
                Ok(CanonicalSearchArtifactScopeV2::from_run_input_range(
                    bounds.role(),
                    input,
                    start..end,
                )?)
            };
        let calibration = bounded_scope(calibration)?;
        let holdout = bounded_scope(holdout)?;
        let source_end = usize::try_from(source.selection_scope.evaluated_window().row_end())?;
        let period = times[..source_end]
            .windows(2)
            .map(|w| w[1] - w[0])
            .min()
            .context("missing base cadence")?;
        let mut seen = BTreeSet::new();
        let mut genes: Vec<_> = source
            .candidates
            .iter()
            .filter(|g| g.tp_pips.is_finite() && g.tp_pips > 0.0 && g.tp_pips <= 20.0)
            .filter(|g| seen.insert(gene_signature_hash(g)))
            .cloned()
            .collect();
        ensure!(
            (2..=BANK_LIMIT).contains(&genes.len()),
            "normal-target specialist bank must contain2..256 distinct expert signatures"
        );
        let starting_bank = genes.len();
        let account_policy = if risky {
            risky_account_policy(&evaluation)?
        } else {
            joint_account_policy()
        };
        eprintln!(
            "specialist_mode={} account_currency={} initial={} target={:?} horizon_days={:?} risk_band={:?} aggregate_stop_risk={} notional_cap={}",
            if risky {
                "risky-v4"
            } else {
                "continual-v3-reference"
            },
            evaluation.account_currency,
            evaluation.initial_equity,
            goal.map(|g| g.target_balance),
            goal.map(|g| g.horizon_days),
            original_risk,
            account_policy.max_entry_stop_risk_fraction,
            account_policy.max_gross_notional_multiple
        );
        let calibration_result = forward(
            input,
            research,
            &calibration,
            &evaluation,
            period,
            &mut genes,
            policy,
            account_policy,
            goal,
        )?;
        let holdout_result = forward(
            input,
            research,
            &holdout,
            &evaluation,
            period,
            &mut genes,
            policy,
            account_policy,
            goal,
        )?;
        Ok(
            json!({"schema":if risky { "neoethos.risky-specialist-research.v4" } else { "neoethos.continual-specialist-research.v3" },"research_only":true,"promotion_eligible":false,
            "backend":"canonical_cpu","combined_portfolio_evaluated":true,"source_evidence_sha256":research.evidence_identity_sha256(),
            "evaluated_mode":if risky {"risky"} else {"reference_risk_overlay"},"account_currency":evaluation.account_currency,
            "goal":goal,"goal_scoring":if risky { Some("existing canonical ga_fitness_goal pace on min(realized, pending-cost net equity); cash=0; mean minus 0.25 episode score stddev; proxy is not target success; banked flat realized target before deadline reported separately") } else { None },
            "notional_cap_source":"existing live_trading.rs MAX_NOTIONAL_MULTIPLE=30; no 400x assumption or broker margin claim",
            "original_risk_band":original_risk,"evaluated_risk_band":[evaluation.risk_per_trade_min,evaluation.risk_per_trade_max],
            "account_policy":account_policy,"initial_bank_size":starting_bank,"final_bank_size":genes.len(),"final_genes":genes,
            "protocol":{"protocol_revision":if risky {4} else {3},"behavior_semantics_revision":2,"inactive_specialists_are_not_clones":true,
                "validation_min_closed_trades":neoethos_search::genetic::RegimeLabelPolicy::default().min_trades_per_window,
                "evidence_floor_status":"existing regime-policy floor; necessary screening only, not a significance or profitability guarantee",
                "train_days":TRAIN_DAYS,"validation_days":VALIDATION_DAYS,"rebalance":"monthly after decision bar closes",
                "embargo_bars":evaluation.max_hold_bars+1,"max_children_per_update":NEW_CHILDREN_PER_UPDATE,"bank_limit":BANK_LIMIT,
                "fit":"8 books x3 generations before and after training-only crossover; incumbent and earlier fit champion retained as warm starts; behavioral clones rejected",
                "scope_roles":"rolling fit and validation are explicitly reclassified as InSample relative to each as-of decision; prior forward labels may be consumed only after maturity",
                "incumbent":"recent validation compares incumbent, challenger and cash; strict improvement required; open positions retain protection"},
            "validation_status":"predeclared adaptive/prequential historical diagnostic using past matured labels, not a static untouched OOS; historical tail already inspected",
            "financial_limits":"same reference costs; fractional lots unless sealed checked grid; no broker quote/margin/stopout, GPU or live promotion evidence",
            "discovery_limits":"archive-seeded crossover renewal and joint-book learning; original upstream discovery gates and supervised ML ensemble unchanged",
            "calibration":calibration_result,"holdout":holdout_result}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_profitable_validation_trades_cannot_promote_a_book() {
        let minimum = neoethos_search::genetic::RegimeLabelPolicy::default()
            .min_trades_per_window
            .ceil() as usize;
        assert!(minimum > 2);
        assert!(!candidate_can_replace(0.0, 0.01, 2, minimum));
        assert!(!candidate_can_replace(0.001, 0.5, 2, minimum));
        assert!(candidate_can_replace(0.0, 0.01, minimum, minimum));
        assert!(!candidate_can_replace(0.0, -0.01, minimum, minimum));
    }
    #[test]
    fn inactive_specialists_are_not_behavioral_clones() {
        let classes = [
            [None, Some(11), Some(12), Some(13)],
            [None, Some(21), Some(22), Some(23)],
        ];
        let mut book = Genome::cash(2);
        book.weights[0] = vec![1, 1];
        assert!(
            !has_behavior_clones(&classes, &book),
            "absence of signals must not disqualify a specialist book"
        );
    }
    #[test]
    fn renewal_keeps_incumbent_and_fit_champion_in_the_fixed_population() {
        let incumbent = Genome {
            weights: std::array::from_fn(|_| vec![1, 0, 0]),
        };
        let champion = Genome {
            weights: std::array::from_fn(|_| vec![0, 2, 1]),
        };
        let mut rng = StdRng::seed_from_u64(7);
        let population =
            seed_population(3, 8, &[incumbent.clone(), champion.clone()], &mut rng).unwrap();
        assert_eq!(population.len(), 8);
        assert_eq!(population[0], Genome::cash(3));
        assert!(population.contains(&incumbent) && population.contains(&champion));
        assert!(seed_population(3, 2, &[incumbent, champion], &mut rng).is_err());
    }
    #[test]
    fn cash_does_not_end_seeded_expert_renewal_and_targets_stay_normal() {
        let a = Gene {
            indices: vec![0, 1],
            weights: vec![1., 2.],
            sl_pips: 2.,
            tp_pips: 6.,
            stop_vol_mult: 0.,
            ..Default::default()
        };
        let b = Gene {
            indices: vec![2, 3],
            weights: vec![2., 1.],
            sl_pips: 3.,
            tp_pips: 9.,
            stop_vol_mult: 0.,
            ..Default::default()
        };
        let original = vec![a, b];
        let generate = || {
            let mut genes = original.clone();
            let births = renew_bank(&mut genes, &Genome::cash(2), 42).unwrap();
            assert!(!births.is_empty() && births.len() <= 2);
            assert!(genes.iter().all(|g| g.tp_pips > 0.0 && g.tp_pips <= 20.0));
            let signatures: Vec<_> = genes.iter().map(gene_signature_hash).collect();
            assert_eq!(
                signatures.iter().collect::<BTreeSet<_>>().len(),
                signatures.len()
            );
            signatures
        };
        assert_eq!(generate(), generate());
        assert_eq!(
            generate()[..2],
            original.iter().map(gene_signature_hash).collect::<Vec<_>>()
        );
    }
    #[test]
    fn fit_validation_and_decision_have_real_holding_embargoes() {
        let times: Vec<_> = (0..500 * 24)
            .map(|hour| 1_704_067_200_000 + hour * 3_600_000)
            .collect();
        let (fit, val) = past_ranges(&times, 450 * 24, 16).unwrap();
        assert_eq!(fit.end + 17, val.start);
        assert_eq!(val.end + 17, 450 * 24);
        assert!(fit.end < val.start && val.end < 450 * 24);
        assert!(past_ranges(&times, 200 * 24, 16).is_err());
        assert!(past_ranges(&times, 450 * 24, 0).is_err());
    }
    #[test]
    fn future_timestamps_cannot_change_the_asof_training_ranges() {
        let mut times: Vec<_> = (0..500 * 24)
            .map(|hour| 1_704_067_200_000 + hour * 3_600_000)
            .collect();
        let before = past_ranges(&times, 450 * 24, 16).unwrap();
        for t in &mut times[450 * 24 + 1..] {
            *t += 999_000_000;
        }
        assert_eq!(before, past_ranges(&times, 450 * 24, 16).unwrap());
    }
    #[test]
    fn scheduled_books_preserve_prior_allocations_and_exclude_unborn_experts() {
        let first = Genome {
            weights: std::array::from_fn(|_| vec![1, 1]),
        };
        let next = Genome {
            weights: std::array::from_fn(|_| vec![0, 1, 1]),
        };
        let (active, weights) =
            scheduled_allocations(&[Some(0); 4], &[(0, first.clone()), (2, next)], 3).unwrap();
        assert_eq!(active, [0, 1, 2]);
        assert_eq!(
            weights,
            [vec![0.5, 0.5, 0., 0.], vec![0.5; 4], vec![0., 0., 0.5, 0.5]]
        );
        let (_, prefix) = scheduled_allocations(&[Some(0); 2], &[(0, first)], 3).unwrap();
        assert_eq!(weights[0][..2], prefix[0]);
        assert_eq!(weights[1][..2], prefix[1]);
    }
    #[test]
    fn observed_behavior_clones_are_local_to_their_declared_state() {
        let classes = [[11, 12, 13, 14], [11, 22, 23, 24]].map(|row| row.map(Some));
        let mut book = Genome::cash(2);
        book.weights[0] = vec![1, 1];
        assert!(has_behavior_clones(&classes, &book));
        book.weights[0] = vec![1, 0];
        book.weights[1] = vec![1, 1];
        assert!(!has_behavior_clones(&classes, &book));
    }
    #[test]
    fn bank_growth_pads_with_zero_and_cannot_rewrite_old_gene_indices() {
        let book = Genome {
            weights: std::array::from_fn(|_| vec![2, 1]),
        };
        let extended = padded(&book, 4).unwrap();
        assert_eq!(extended.weights[0], [2, 1, 0, 0]);
        assert!(padded(&extended, 2).is_err());
    }
    #[test]
    fn month_updates_cover_the_entire_forward_span_once() {
        let times: Vec<_> = (0..100)
            .map(|day| 1_704_067_200_000 + day * 86_400_000)
            .collect();
        let ranges = month_ranges(&times, 4..95).unwrap();
        assert_eq!(ranges, [4..31, 31..60, 60..91, 91..95]);
        assert!(ranges.windows(2).all(|w| w[0].end == w[1].start));
    }
}
