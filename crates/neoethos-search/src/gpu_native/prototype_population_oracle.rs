//! Validation-only semantic oracle for the common Prototype B/C contracts.
//!
//! This module intentionally re-expresses the canonical signal, causal entry,
//! first-hit, cost and metric rules without calling Prototype A or the full
//! population evaluator. It is an untimed validation reference: production
//! benchmark paths must prepare these results before entering their timed loop.

use crate::eval::{BacktestSettings, current_backtest_runtime_overrides};
use crate::gpu_native::prototype_population::PrototypePopulationWorkload;
use neoethos_gpu_contracts::device::{
    NeoPopulationCounters, NeoPopulationEvent, NeoPopulationMetricRow, NeoPopulationOutcome,
    NeoPopulationSettings,
};
use neoethos_gpu_contracts::{
    ABI_VERSION, POPULATION_DIRECTION_LONG, POPULATION_EXIT_GAP, POPULATION_EXIT_MAX_HOLD,
    POPULATION_EXIT_NONE, POPULATION_EXIT_STOP, POPULATION_EXIT_TARGET,
    POPULATION_PRECEDENCE_STOP_FIRST, POPULATION_SETTINGS_FLAG_RISK_BASED_SIZING,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct OraclePopulationEvaluation {
    pub settings: NeoPopulationSettings,
    /// Candidate-major signal events. Outcomes are positionally aligned.
    pub events: Vec<NeoPopulationEvent>,
    pub outcomes: Vec<NeoPopulationOutcome>,
    pub metrics: Vec<NeoPopulationMetricRow>,
    pub counters: NeoPopulationCounters,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PopulationOracleError {
    UnsupportedTrailingState,
    FixedWidthOverflow(&'static str),
    OutcomeCountMismatch {
        events: usize,
        outcomes: usize,
    },
    OutcomeIdentityMismatch {
        index: usize,
        event_candidate_id: u64,
        event_scenario_id: u64,
        outcome_candidate_id: u64,
        outcome_scenario_id: u64,
    },
    OutcomeSemanticMismatch {
        index: usize,
        candidate_id: u64,
        scenario_id: u64,
        expected_exit_bar: i32,
        expected_exit_reason: i32,
        actual_exit_bar: i32,
        actual_exit_reason: i32,
    },
}

impl fmt::Display for PopulationOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTrailingState => {
                write!(f, "Prototype B/C oracle does not support trailing state")
            }
            Self::FixedWidthOverflow(field) => {
                write!(f, "{field} does not fit the fixed-width population ABI")
            }
            Self::OutcomeCountMismatch { events, outcomes } => write!(
                f,
                "population outcome count {outcomes} does not match event count {events}"
            ),
            Self::OutcomeIdentityMismatch {
                index,
                event_candidate_id,
                event_scenario_id,
                outcome_candidate_id,
                outcome_scenario_id,
            } => write!(
                f,
                "population outcome {index} identity ({outcome_candidate_id}, {outcome_scenario_id}) \
                 does not match event identity ({event_candidate_id}, {event_scenario_id})"
            ),
            Self::OutcomeSemanticMismatch {
                index,
                candidate_id,
                scenario_id,
                expected_exit_bar,
                expected_exit_reason,
                actual_exit_bar,
                actual_exit_reason,
            } => write!(
                f,
                "population outcome {index} for ({candidate_id}, {scenario_id}) has exit \
                 ({actual_exit_bar}, {actual_exit_reason}), expected \
                 ({expected_exit_bar}, {expected_exit_reason}) for its positional event"
            ),
        }
    }
}

impl Error for PopulationOracleError {}

#[derive(Debug, Clone)]
struct CandidateSignals {
    values: Vec<i8>,
    confidences: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct OpenPosition {
    event: NeoPopulationEvent,
    outcome: NeoPopulationOutcome,
    entry_price: f64,
    lots: f64,
}

#[derive(Debug, Clone, Copy)]
struct CandidateReduction {
    row: NeoPopulationMetricRow,
    accepted_trades: u64,
}

pub fn population_settings(
    workload: &PrototypePopulationWorkload,
) -> Result<NeoPopulationSettings, PopulationOracleError> {
    let source = workload.dataset.settings.to_settings();
    let runtime = current_backtest_runtime_overrides();
    Ok(NeoPopulationSettings {
        abi_version: ABI_VERSION,
        flags: if source.risk_based_sizing {
            POPULATION_SETTINGS_FLAG_RISK_BASED_SIZING
        } else {
            0
        },
        max_hold_bars: u32::try_from(source.max_hold_bars)
            .map_err(|_| PopulationOracleError::FixedWidthOverflow("max_hold_bars"))?,
        min_hold_bars: u32::try_from(source.min_hold_bars)
            .map_err(|_| PopulationOracleError::FixedWidthOverflow("min_hold_bars"))?,
        max_trades_per_day: u32::try_from(source.max_trades_per_day)
            .map_err(|_| PopulationOracleError::FixedWidthOverflow("max_trades_per_day"))?,
        month_capacity: u32::try_from(runtime.month_capacity)
            .map_err(|_| PopulationOracleError::FixedWidthOverflow("month_capacity"))?,
        gap_threshold_ms: source.gap_threshold_ms,
        initial_equity: runtime.initial_equity,
        pip_value: source.pip_value,
        spread_pips: source.spread_pips,
        commission_per_trade: source.commission_per_trade,
        pip_value_per_lot: source.pip_value_per_lot,
        swap_long_pips_per_day: source.swap_long_pips_per_day,
        swap_short_pips_per_day: source.swap_short_pips_per_day,
        pnl_conversion_fee_rate: source.pnl_conversion_fee_rate,
        risk_per_trade_min: source.risk_per_trade_min,
        risk_per_trade_max: source.risk_per_trade_max,
        high_quality_confidence: source.high_quality_confidence,
        adaptive_rr: source.adaptive_rr,
    })
}

pub fn evaluate_population_oracle(
    workload: &PrototypePopulationWorkload,
) -> Result<OraclePopulationEvaluation, PopulationOracleError> {
    let settings = population_settings(workload)?;
    let signals = synthesize_population_signals(workload);
    let events = emit_population_events_with_signals(workload, &signals)?;
    let outcomes = resolve_population_outcomes(workload, &events)?;
    let (metrics, accepted_trade_count) =
        reduce_population_outcomes_with_signals(workload, &signals, &events, &outcomes)?;
    Ok(OraclePopulationEvaluation {
        settings,
        counters: NeoPopulationCounters {
            event_count: events.len() as u64,
            accepted_trade_count,
            ..NeoPopulationCounters::default()
        },
        events,
        outcomes,
        metrics,
    })
}

pub fn emit_population_events(
    workload: &PrototypePopulationWorkload,
) -> Result<Vec<NeoPopulationEvent>, PopulationOracleError> {
    let signals = synthesize_population_signals(workload);
    emit_population_events_with_signals(workload, &signals)
}

pub fn resolve_population_outcomes(
    workload: &PrototypePopulationWorkload,
    events: &[NeoPopulationEvent],
) -> Result<Vec<NeoPopulationOutcome>, PopulationOracleError> {
    let settings = workload.dataset.settings.to_settings();
    if settings.trailing_enabled {
        return Err(PopulationOracleError::UnsupportedTrailingState);
    }
    let bars = workload.dataset.bars();
    i32::try_from(bars.saturating_sub(1))
        .map_err(|_| PopulationOracleError::FixedWidthOverflow("outcome exit_bar"))?;
    Ok(events
        .iter()
        .map(|event| {
            let mut outcome = NeoPopulationOutcome {
                candidate_id: event.candidate_id,
                scenario_id: event.scenario_id,
                exit_bar: -1,
                exit_reason: POPULATION_EXIT_NONE,
            };
            let entry_bar = event.entry_bar as usize;
            let last_bar = (event.last_bar as usize).min(bars.saturating_sub(1));
            if entry_bar >= last_bar {
                return outcome;
            }

            for bar in entry_bar + 1..=last_bar {
                if is_gap_exit(workload, &settings, bar) {
                    outcome.exit_bar = bar as i32;
                    outcome.exit_reason = POPULATION_EXIT_GAP;
                    break;
                }
                let bars_held = bar - entry_bar;
                let past_min_hold =
                    settings.min_hold_bars == 0 || bars_held >= settings.min_hold_bars;
                if !past_min_hold {
                    continue;
                }

                let high = workload.dataset.high[bar];
                let low = workload.dataset.low[bar];
                let (stop_hit, target_hit) = if event.direction == POPULATION_DIRECTION_LONG {
                    (low <= event.stop_price, high >= event.target_price)
                } else {
                    (high >= event.stop_price, low <= event.target_price)
                };
                if stop_hit {
                    outcome.exit_bar = bar as i32;
                    outcome.exit_reason = POPULATION_EXIT_STOP;
                    break;
                }
                if target_hit {
                    outcome.exit_bar = bar as i32;
                    outcome.exit_reason = POPULATION_EXIT_TARGET;
                    break;
                }
                if settings.max_hold_bars > 0 && bars_held >= settings.max_hold_bars {
                    outcome.exit_bar = bar as i32;
                    outcome.exit_reason = POPULATION_EXIT_MAX_HOLD;
                    break;
                }
            }
            outcome
        })
        .collect())
}

pub fn reduce_population_outcomes(
    workload: &PrototypePopulationWorkload,
    events: &[NeoPopulationEvent],
    outcomes: &[NeoPopulationOutcome],
) -> Result<Vec<NeoPopulationMetricRow>, PopulationOracleError> {
    let signals = synthesize_population_signals(workload);
    reduce_population_outcomes_with_signals(workload, &signals, events, outcomes)
        .map(|(rows, _)| rows)
}

fn emit_population_events_with_signals(
    workload: &PrototypePopulationWorkload,
    signals: &[CandidateSignals],
) -> Result<Vec<NeoPopulationEvent>, PopulationOracleError> {
    let settings = workload.dataset.settings.to_settings();
    if settings.trailing_enabled {
        return Err(PopulationOracleError::UnsupportedTrailingState);
    }
    let bars = workload.dataset.bars();
    let last_dataset_bar = bars.saturating_sub(1);
    u32::try_from(last_dataset_bar)
        .map_err(|_| PopulationOracleError::FixedWidthOverflow("dataset bars"))?;
    i32::try_from(last_dataset_bar)
        .map_err(|_| PopulationOracleError::FixedWidthOverflow("outcome exit_bar"))?;
    let pip = guarded_pip(settings.pip_value);
    let half_spread_price = settings.spread_pips * 0.5 * pip;
    let mut events = Vec::new();

    for (gene_index, candidate_signals) in signals.iter().enumerate() {
        let candidate_id = workload.genes.candidate_ids[gene_index];
        let scenario_id = workload.scenarios.scenarios[gene_index].scenario_id;
        for entry_bar in 1..bars {
            let direction = candidate_signals.values[entry_bar - 1] as i32;
            if direction == 0 {
                continue;
            }
            let entry_price =
                workload.dataset.close[entry_bar] + direction as f64 * half_spread_price;
            let (stop_pips, target_pips) = entry_stop_target_pips(
                workload,
                &settings,
                gene_index,
                entry_bar.saturating_sub(1),
            );
            let (stop_price, target_price) = if direction == POPULATION_DIRECTION_LONG {
                (
                    entry_price - stop_pips * pip,
                    entry_price + target_pips * pip,
                )
            } else {
                (
                    entry_price + stop_pips * pip,
                    entry_price - target_pips * pip,
                )
            };
            let last_bar = if settings.max_hold_bars > 0 {
                entry_bar
                    .saturating_add(settings.max_hold_bars.max(settings.min_hold_bars))
                    .min(last_dataset_bar)
            } else {
                last_dataset_bar
            };
            events.push(NeoPopulationEvent {
                candidate_id,
                scenario_id,
                entry_bar: u32::try_from(entry_bar)
                    .map_err(|_| PopulationOracleError::FixedWidthOverflow("entry_bar"))?,
                last_bar: u32::try_from(last_bar)
                    .map_err(|_| PopulationOracleError::FixedWidthOverflow("last_bar"))?,
                direction,
                precedence: POPULATION_PRECEDENCE_STOP_FIRST,
                stop_price,
                target_price,
            });
        }
    }
    Ok(events)
}

fn reduce_population_outcomes_with_signals(
    workload: &PrototypePopulationWorkload,
    signals: &[CandidateSignals],
    events: &[NeoPopulationEvent],
    outcomes: &[NeoPopulationOutcome],
) -> Result<(Vec<NeoPopulationMetricRow>, u64), PopulationOracleError> {
    validate_outcome_alignment(workload, events, outcomes)?;
    let settings = workload.dataset.settings.to_settings();
    if settings.trailing_enabled {
        return Err(PopulationOracleError::UnsupportedTrailingState);
    }

    let mut rows = Vec::with_capacity(workload.genes.population());
    let mut accepted_trade_count = 0_u64;
    for gene_index in 0..workload.genes.population() {
        let candidate_id = workload.genes.candidate_ids[gene_index];
        let scenario_id = workload.scenarios.scenarios[gene_index].scenario_id;
        let candidate_events = events
            .iter()
            .copied()
            .zip(outcomes.iter().copied())
            .filter(|(event, _)| {
                event.candidate_id == candidate_id && event.scenario_id == scenario_id
            })
            .collect::<Vec<_>>();
        let reduction = reduce_candidate(
            workload,
            &settings,
            &signals[gene_index],
            candidate_id,
            scenario_id,
            &candidate_events,
        );
        rows.push(reduction.row);
        accepted_trade_count = accepted_trade_count.saturating_add(reduction.accepted_trades);
    }
    Ok((rows, accepted_trade_count))
}

fn validate_outcome_alignment(
    workload: &PrototypePopulationWorkload,
    events: &[NeoPopulationEvent],
    outcomes: &[NeoPopulationOutcome],
) -> Result<(), PopulationOracleError> {
    if events.len() != outcomes.len() {
        return Err(PopulationOracleError::OutcomeCountMismatch {
            events: events.len(),
            outcomes: outcomes.len(),
        });
    }
    for (index, (event, outcome)) in events.iter().zip(outcomes).enumerate() {
        if event.candidate_id != outcome.candidate_id || event.scenario_id != outcome.scenario_id {
            return Err(PopulationOracleError::OutcomeIdentityMismatch {
                index,
                event_candidate_id: event.candidate_id,
                event_scenario_id: event.scenario_id,
                outcome_candidate_id: outcome.candidate_id,
                outcome_scenario_id: outcome.scenario_id,
            });
        }
    }
    let expected = resolve_population_outcomes(workload, events)?;
    for (index, ((event, expected), actual)) in
        events.iter().zip(expected.iter()).zip(outcomes).enumerate()
    {
        if expected.exit_bar != actual.exit_bar || expected.exit_reason != actual.exit_reason {
            return Err(PopulationOracleError::OutcomeSemanticMismatch {
                index,
                candidate_id: event.candidate_id,
                scenario_id: event.scenario_id,
                expected_exit_bar: expected.exit_bar,
                expected_exit_reason: expected.exit_reason,
                actual_exit_bar: actual.exit_bar,
                actual_exit_reason: actual.exit_reason,
            });
        }
    }
    Ok(())
}

fn synthesize_population_signals(workload: &PrototypePopulationWorkload) -> Vec<CandidateSignals> {
    (0..workload.genes.population())
        .map(|gene_index| synthesize_candidate_signals(workload, gene_index))
        .collect()
}

fn synthesize_candidate_signals(
    workload: &PrototypePopulationWorkload,
    gene_index: usize,
) -> CandidateSignals {
    let bars = workload.dataset.bars();
    let mut combined = vec![0.0_f32; bars];
    let start = workload.genes.offsets[gene_index] as usize;
    let end = workload.genes.offsets[gene_index + 1] as usize;
    for term in start..end {
        let feature = workload.genes.indices[term] as usize;
        let weight = workload.genes.weights[term];
        let row_start = feature * bars;
        for (bar, combined_value) in combined.iter_mut().enumerate() {
            *combined_value += weight * workload.dataset.indicators[row_start + bar];
        }
    }

    let mut values = vec![0_i8; bars];
    let mut confidences = vec![0.0_f32; bars];
    let long_threshold = workload.genes.long_thresholds[gene_index];
    let short_threshold = workload.genes.short_thresholds[gene_index];
    let gap = (long_threshold - short_threshold).abs().max(1.0e-6);
    let flags = workload.genes.smc_flags[gene_index];
    let active_sum = flags
        .iter()
        .enumerate()
        .map(|(index, &flag)| {
            if flag != 0 {
                workload.genes.smc_weights[index]
            } else {
                0.0
            }
        })
        .sum::<f32>();
    let active_sum = if crate::genetic::smc_gate_disabled() {
        0.0
    } else {
        active_sum
    };
    let gate = workload.genes.gate_threshold.min(active_sum);

    for bar in 0..bars {
        let combined_value = combined[bar];
        let signal = if combined_value >= long_threshold {
            1
        } else if combined_value <= short_threshold {
            -1
        } else {
            0
        };
        if signal == 0 {
            continue;
        }
        let margin = if signal == 1 {
            combined_value - long_threshold
        } else {
            short_threshold - combined_value
        };
        let confidence = (margin / gap).clamp(0.0, 1.0);
        let passes_gate = if active_sum > 0.0 {
            let score = flags
                .iter()
                .enumerate()
                .map(|(index, &flag)| {
                    if flag == 0 {
                        0.0
                    } else if index == 5 {
                        if workload.dataset.smc_data[bar][index] == 1 {
                            workload.genes.smc_weights[index]
                        } else {
                            0.0
                        }
                    } else if workload.dataset.smc_data[bar][index] == signal {
                        workload.genes.smc_weights[index]
                    } else {
                        0.0
                    }
                })
                .sum::<f32>();
            score >= gate
        } else {
            true
        };
        if passes_gate {
            values[bar] = signal;
            confidences[bar] = confidence;
        }
    }
    CandidateSignals {
        values,
        confidences,
    }
}

fn reduce_candidate(
    workload: &PrototypePopulationWorkload,
    settings: &BacktestSettings,
    signals: &CandidateSignals,
    candidate_id: u64,
    scenario_id: u64,
    events: &[(NeoPopulationEvent, NeoPopulationOutcome)],
) -> CandidateReduction {
    let initial_equity = current_backtest_runtime_overrides().initial_equity;
    let month_capacity = current_backtest_runtime_overrides().month_capacity;
    let mut equity = initial_equity;
    let mut peak_equity = initial_equity;
    let mut max_drawdown = 0.0_f64;
    let mut trade_count = 0_usize;
    let mut wins = 0_usize;
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;
    let mut accepted_trades = 0_u64;

    let mut last_month = -1_i64;
    let mut current_month_pnl = 0.0_f64;
    let mut monthly_pnls = vec![0.0; month_capacity];
    let mut month_start_equities = vec![initial_equity; month_capacity];
    let mut current_month_start_equity = initial_equity;
    let mut month_ptr = -1_i64;

    let mut last_day = -1_i64;
    let mut day_peak = equity;
    let mut day_low = equity;
    let mut max_daily_drawdown = 0.0_f64;
    let mut day_trade_count = 0_usize;
    let mut open_position: Option<OpenPosition> = None;

    let pip = guarded_pip(settings.pip_value);
    let half_spread_cost = settings.spread_pips * 0.5 * settings.pip_value_per_lot;
    let bars = workload.dataset.bars();
    for bar in 1..bars {
        let month = workload.dataset.months[bar];
        if month != last_month {
            if last_month != -1 {
                month_ptr += 1;
                if month_ptr < month_capacity as i64 {
                    monthly_pnls[month_ptr as usize] = current_month_pnl;
                    month_start_equities[month_ptr as usize] = current_month_start_equity;
                }
            }
            current_month_pnl = 0.0;
            current_month_start_equity = equity;
            last_month = month;
        }

        let day = workload.dataset.days[bar];
        if day != last_day {
            if last_day != -1 && day_peak > 0.0 {
                let drawdown = (day_peak - day_low) / day_peak;
                if drawdown > max_daily_drawdown {
                    max_daily_drawdown = drawdown;
                }
            }
            last_day = day;
            day_peak = equity;
            day_low = equity;
            day_trade_count = 0;
        }

        if let Some(position) = open_position {
            let exited_on_gap = position.outcome.exit_bar == bar as i32
                && position.outcome.exit_reason == POPULATION_EXIT_GAP;
            if exited_on_gap {
                realize_position(
                    workload,
                    settings,
                    position,
                    bar,
                    pip,
                    half_spread_cost,
                    &mut equity,
                    &mut current_month_pnl,
                    &mut trade_count,
                    &mut wins,
                    &mut gross_profit,
                    &mut gross_loss,
                );
                update_realized_risk(equity, &mut peak_equity, &mut day_low, &mut max_drawdown);
                open_position = None;
            } else {
                let (worst_float_pnl, best_float_pnl) =
                    floating_pnl(workload, settings, position, bar, pip);
                if equity + worst_float_pnl < day_low {
                    day_low = equity + worst_float_pnl;
                }
                if equity + best_float_pnl > peak_equity {
                    peak_equity = equity + best_float_pnl;
                }
                if peak_equity > 0.0 {
                    let drawdown = (peak_equity - (equity + worst_float_pnl)) / peak_equity;
                    if drawdown > max_drawdown {
                        max_drawdown = drawdown;
                    }
                }

                if position.outcome.exit_bar == bar as i32
                    && position.outcome.exit_reason != POPULATION_EXIT_NONE
                {
                    realize_position(
                        workload,
                        settings,
                        position,
                        bar,
                        pip,
                        half_spread_cost,
                        &mut equity,
                        &mut current_month_pnl,
                        &mut trade_count,
                        &mut wins,
                        &mut gross_profit,
                        &mut gross_loss,
                    );
                    update_realized_risk(equity, &mut peak_equity, &mut day_low, &mut max_drawdown);
                    open_position = None;
                }
                // A regular SL/TP/max-hold exit occurs inside the open-position
                // branch, so canonical evaluation cannot re-enter on this bar.
                continue;
            }
            // Gap handling precedes the open/flat branch in the canonical
            // evaluator. Once it closes the position, the same bar is flat and
            // may consume the prior bar's signal below.
        }

        if let Some((event, outcome)) = events
            .iter()
            .copied()
            .find(|(event, _)| event.entry_bar as usize == bar)
        {
            if settings.max_trades_per_day > 0 && day_trade_count >= settings.max_trades_per_day {
                continue;
            }
            let signal_bar = bar - 1;
            let stop_pips =
                ((event.stop_price - event_entry_price(workload, settings, event)).abs() / pip)
                    .abs();
            let lots = if settings.risk_based_sizing && !signals.confidences.is_empty() {
                risk_based_position_lots(
                    signals.confidences[signal_bar] as f64,
                    equity,
                    stop_pips,
                    settings,
                )
            } else {
                1.0
            };
            open_position = Some(OpenPosition {
                event,
                outcome,
                entry_price: event_entry_price(workload, settings, event),
                lots,
            });
            day_trade_count += 1;
            accepted_trades = accepted_trades.saturating_add(1);
        }
    }

    let net_profit = equity - initial_equity;
    let win_rate = if trade_count > 0 {
        wins as f64 / trade_count as f64
    } else {
        0.0
    };
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        10.0
    } else {
        0.0
    };
    let expectancy = if trade_count > 0 {
        net_profit / trade_count as f64
    } else {
        0.0
    };
    let month_returns = completed_month_pnls(&monthly_pnls, month_ptr, month_capacity);
    let (monthly_mean, monthly_std) = neoethos_core::utils::mean_std(&month_returns);
    let (monthly_mean, monthly_std) = if monthly_mean.is_finite() && monthly_std.is_finite() {
        (monthly_mean, monthly_std)
    } else {
        (0.0, 0.0)
    };
    let sharpe = if monthly_std > 0.0 {
        (monthly_mean / monthly_std) * 3.4641
    } else {
        0.0
    };
    let consistency = if monthly_std > 0.0 {
        (monthly_mean / monthly_std).clamp(0.0, 1.0)
    } else if monthly_mean > 0.0 && month_returns.len() < 2 {
        1.0
    } else {
        0.0
    };
    let monthly_target_hit_rate = monthly_target_hit_rate(
        &monthly_pnls,
        &month_start_equities,
        month_ptr,
        month_capacity,
    );
    let sanitize = |value: f64| if value.is_finite() { value } else { 0.0 };
    CandidateReduction {
        row: NeoPopulationMetricRow {
            candidate_id,
            scenario_id,
            values: [
                sanitize(net_profit),
                sanitize(sharpe),
                sanitize(peak_equity),
                sanitize(max_drawdown),
                sanitize(win_rate),
                sanitize(profit_factor),
                sanitize(expectancy),
                sanitize(monthly_target_hit_rate),
                trade_count as f64,
                sanitize(consistency),
                sanitize(max_daily_drawdown),
            ],
        },
        accepted_trades,
    }
}

#[allow(clippy::too_many_arguments)]
fn realize_position(
    workload: &PrototypePopulationWorkload,
    settings: &BacktestSettings,
    position: OpenPosition,
    exit_bar: usize,
    pip: f64,
    half_spread_cost: f64,
    equity: &mut f64,
    current_month_pnl: &mut f64,
    trade_count: &mut usize,
    wins: &mut usize,
    gross_profit: &mut f64,
    gross_loss: &mut f64,
) {
    let exit_price = match position.outcome.exit_reason {
        POPULATION_EXIT_STOP => position.event.stop_price,
        POPULATION_EXIT_TARGET => position.event.target_price,
        POPULATION_EXIT_MAX_HOLD | POPULATION_EXIT_GAP => workload.dataset.close[exit_bar],
        _ => return,
    };
    let price_pnl = if position.event.direction == POPULATION_DIRECTION_LONG {
        (exit_price - position.entry_price) / pip * settings.pip_value_per_lot
    } else {
        (position.entry_price - exit_price) / pip * settings.pip_value_per_lot
    };
    let gross_scaled = price_pnl * position.lots
        - (settings.commission_per_trade + half_spread_cost) * position.lots;
    let entry_bar = position.event.entry_bar as usize;
    let entry_timestamp = workload
        .dataset
        .timestamps
        .get(entry_bar)
        .copied()
        .unwrap_or(0);
    let exit_timestamp = workload
        .dataset
        .timestamps
        .get(exit_bar)
        .copied()
        .unwrap_or(0);
    let pnl = apply_carry_and_conversion(
        gross_scaled,
        position.lots,
        position.event.direction,
        entry_timestamp,
        exit_timestamp,
        settings,
    );
    *equity += pnl;
    *current_month_pnl += pnl;
    *trade_count += 1;
    if pnl > 0.0 {
        *wins += 1;
        *gross_profit += pnl;
    } else {
        *gross_loss += pnl.abs();
    }
}

fn update_realized_risk(
    equity: f64,
    peak_equity: &mut f64,
    day_low: &mut f64,
    max_drawdown: &mut f64,
) {
    if equity > *peak_equity {
        *peak_equity = equity;
    }
    if equity < *day_low {
        *day_low = equity;
    }
    if *peak_equity > 0.0 {
        let drawdown = (*peak_equity - equity) / *peak_equity;
        if drawdown > *max_drawdown {
            *max_drawdown = drawdown;
        }
    }
}

fn floating_pnl(
    workload: &PrototypePopulationWorkload,
    settings: &BacktestSettings,
    position: OpenPosition,
    bar: usize,
    pip: f64,
) -> (f64, f64) {
    let low = workload.dataset.low[bar];
    let high = workload.dataset.high[bar];
    let (worst, best) = if position.event.direction == POPULATION_DIRECTION_LONG {
        (
            (low - position.entry_price) / pip * settings.pip_value_per_lot,
            (high - position.entry_price) / pip * settings.pip_value_per_lot,
        )
    } else {
        (
            (position.entry_price - high) / pip * settings.pip_value_per_lot,
            (position.entry_price - low) / pip * settings.pip_value_per_lot,
        )
    };
    (worst * position.lots, best * position.lots)
}

fn event_entry_price(
    workload: &PrototypePopulationWorkload,
    settings: &BacktestSettings,
    event: NeoPopulationEvent,
) -> f64 {
    let pip = guarded_pip(settings.pip_value);
    workload.dataset.close[event.entry_bar as usize]
        + event.direction as f64 * settings.spread_pips * 0.5 * pip
}

fn entry_stop_target_pips(
    workload: &PrototypePopulationWorkload,
    settings: &BacktestSettings,
    gene_index: usize,
    signal_bar: usize,
) -> (f64, f64) {
    let multiplier = workload.genes.stop_vol_multipliers[gene_index];
    if multiplier > 0.0 {
        if let Some(base) = &settings.adaptive_base_pips {
            if let Some(&distance) = base.get(signal_bar) {
                let stop = multiplier * distance;
                let target = settings.adaptive_rr * stop;
                if stop.is_finite() && stop > 0.0 && target.is_finite() && target > 0.0 {
                    return (stop, target);
                }
            }
        }
    }
    (
        workload.genes.stop_pips[gene_index],
        workload.genes.target_pips[gene_index],
    )
}

fn risk_based_position_lots(
    confidence: f64,
    equity: f64,
    stop_pips: f64,
    settings: &BacktestSettings,
) -> f64 {
    let confidence = confidence.clamp(0.0, 1.0);
    let confidence_scale =
        if settings.high_quality_confidence.is_finite() && settings.high_quality_confidence > 0.0 {
            (confidence / settings.high_quality_confidence).min(1.0)
        } else {
            1.0
        };
    let risk = settings.risk_per_trade_min
        + (settings.risk_per_trade_max - settings.risk_per_trade_min) * confidence_scale;
    let denominator = stop_pips.max(1.0) * settings.pip_value_per_lot;
    let lots = if equity > 0.0 && denominator.abs() > 1.0e-12 && denominator.is_finite() {
        risk * equity / denominator
    } else {
        0.0
    };
    if lots.is_finite() {
        lots.clamp(0.0, 100.0)
    } else {
        0.0
    }
}

fn apply_carry_and_conversion(
    gross_pnl_scaled: f64,
    lots: f64,
    direction: i32,
    entry_timestamp: i64,
    exit_timestamp: i64,
    settings: &BacktestSettings,
) -> f64 {
    let overnight_days = if exit_timestamp > entry_timestamp && entry_timestamp > 0 {
        (exit_timestamp - entry_timestamp) as f64 / 86_400_000.0
    } else {
        0.0
    };
    let swap_pips = if direction == POPULATION_DIRECTION_LONG {
        settings.swap_long_pips_per_day
    } else {
        settings.swap_short_pips_per_day
    };
    let with_carry =
        gross_pnl_scaled + swap_pips * overnight_days * settings.pip_value_per_lot * lots;
    if settings.pnl_conversion_fee_rate.is_finite()
        && settings.pnl_conversion_fee_rate > 0.0
        && settings.pnl_conversion_fee_rate < 1.0
    {
        with_carry * (1.0 - settings.pnl_conversion_fee_rate)
    } else {
        with_carry
    }
}

fn is_gap_exit(
    workload: &PrototypePopulationWorkload,
    settings: &BacktestSettings,
    bar: usize,
) -> bool {
    if settings.gap_threshold_ms <= 0 || bar == 0 {
        return false;
    }
    let previous = workload.dataset.timestamps[bar - 1];
    let current = workload.dataset.timestamps[bar];
    current > previous && current - previous >= settings.gap_threshold_ms
}

fn guarded_pip(pip_value: f64) -> f64 {
    if pip_value.abs() < 1.0e-12 {
        1.0e-12
    } else {
        pip_value
    }
}

fn completed_month_pnls(values: &[f64], month_ptr: i64, month_capacity: usize) -> Vec<f64> {
    if month_ptr < 0 || month_capacity == 0 {
        return Vec::new();
    }
    let limit = month_ptr.min(month_capacity.saturating_sub(1) as i64) as usize;
    values[..=limit].to_vec()
}

fn monthly_target_hit_rate(
    monthly_pnls: &[f64],
    month_start_equities: &[f64],
    month_ptr: i64,
    month_capacity: usize,
) -> f64 {
    if month_ptr < 0 || month_capacity == 0 {
        return 0.0;
    }
    let limit = month_ptr.min(month_capacity.saturating_sub(1) as i64) as usize;
    let mut hits = 0_usize;
    let mut counted = 0_usize;
    for index in 0..=limit {
        let base = month_start_equities[index];
        if base > 0.0 {
            counted += 1;
            if monthly_pnls[index] / base >= 0.04 {
                hits += 1;
            }
        }
    }
    if counted > 0 {
        hits as f64 / counted as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{BacktestMetrics, PopulationEvalInputs, validation_backtest_population_cpu};
    use crate::gpu_native::prototype_a::{
        PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeAScenarioUpload,
    };
    use crate::gpu_native::prototype_population::{
        PropFirmRequirement, PrototypeBcRequirements, PrototypePopulationWorkload,
    };
    use crate::gpu_native::snapshot_fixture::SnapshotSettingsDto;
    use ndarray::ArrayView2;
    use neoethos_gpu_contracts::device::ScenarioDescriptor;

    const BARS: usize = 5;

    fn canonical_cost_fixture() -> PrototypePopulationWorkload {
        let start = 1_700_000_000_000_i64;
        let settings = SnapshotSettingsDto {
            max_hold_bars: 0,
            min_hold_bars: 0,
            max_trades_per_day: 0,
            gap_threshold_ms: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: 1.0,
            spread_pips: 2.0,
            commission_per_trade: 10.0,
            pip_value_per_lot: 100.0,
            swap_long_pips_per_day: -1.0,
            swap_short_pips_per_day: -0.5,
            pnl_conversion_fee_rate: 0.01,
            risk_based_sizing: false,
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.03,
            high_quality_confidence: 0.65,
            adaptive_base_pips: None,
            adaptive_rr: 2.0,
        };
        let candidate_ids = vec![101, 202, 303];
        let scenarios = candidate_ids
            .iter()
            .copied()
            .map(|candidate_id| ScenarioDescriptor {
                base_candidate_id: candidate_id,
                scenario_id: candidate_id + 900,
                window_offset: 0,
                window_len: BARS as u32,
                scenario_type: 0,
                ..ScenarioDescriptor::default()
            })
            .collect();

        PrototypePopulationWorkload::from_uploads(
            PrototypeADatasetUpload {
                close: vec![100.0; BARS],
                high: vec![100.0, 100.0, 151.0, 100.0, 100.0],
                low: vec![100.0; BARS],
                indicators: vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                feature_count: 3,
                months: vec![0, 0, 0, 1, 1],
                days: vec![0, 0, 1, 1, 2],
                timestamps: vec![
                    start,
                    start + 86_400_000,
                    start + 3 * 86_400_000,
                    start + 3 * 86_400_000 + 60_000,
                    start + 3 * 86_400_000 + 120_000,
                ],
                smc_data: vec![[0; 11]; BARS],
                settings,
            },
            PrototypeAGeneUpload {
                candidate_ids,
                offsets: vec![0, 1, 2, 3],
                indices: vec![0, 1, 2],
                weights: vec![1.0; 3],
                long_thresholds: vec![0.5; 3],
                short_thresholds: vec![-0.5; 3],
                stop_pips: vec![10.0; 3],
                target_pips: vec![50.0; 3],
                stop_vol_multipliers: vec![0.0; 3],
                smc_flags: vec![[0; 11]; 3],
                smc_weights: [0.0; 11],
                gate_threshold: 0.0,
            },
            PrototypeAScenarioUpload { scenarios },
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap()
    }

    fn full_cpu_metrics(workload: &PrototypePopulationWorkload) -> Vec<[f64; 11]> {
        let dataset = &workload.dataset;
        let genes = &workload.genes;
        let indicators =
            ArrayView2::from_shape((dataset.feature_count, dataset.bars()), &dataset.indicators)
                .unwrap();
        let settings = dataset.settings.to_settings();
        validation_backtest_population_cpu(PopulationEvalInputs {
            close: &dataset.close,
            high: &dataset.high,
            low: &dataset.low,
            indicators,
            gene_offsets: &genes.offsets,
            gene_indices: &genes.indices,
            gene_weights: &genes.weights,
            long_thr: &genes.long_thresholds,
            short_thr: &genes.short_thresholds,
            month_idx: &dataset.months,
            day_idx: &dataset.days,
            timestamps: &dataset.timestamps,
            sl_pips: &genes.stop_pips,
            tp_pips: &genes.target_pips,
            stop_vol_mult: &genes.stop_vol_multipliers,
            smc_data: &dataset.smc_data,
            gene_smc_flags: &genes.smc_flags,
            gate_threshold: genes.gate_threshold,
            weights: &genes.smc_weights,
            settings: &settings,
        })
    }

    #[test]
    fn oracle_emits_canonical_events_and_preserves_positional_outcome_identity() {
        let workload = canonical_cost_fixture();
        let actual = evaluate_population_oracle(&workload).unwrap();

        assert_eq!(actual.events.len(), 2);
        assert_eq!(actual.outcomes.len(), actual.events.len());
        assert_eq!(
            (
                actual.events[0].candidate_id,
                actual.events[0].scenario_id,
                actual.events[0].entry_bar,
                actual.events[0].direction,
                actual.events[0].stop_price,
                actual.events[0].target_price,
            ),
            (101, 1001, 1, 1, 91.0, 151.0)
        );
        assert_eq!(
            (
                actual.events[1].candidate_id,
                actual.events[1].scenario_id,
                actual.events[1].entry_bar,
                actual.events[1].direction,
                actual.events[1].stop_price,
                actual.events[1].target_price,
            ),
            (202, 1102, 1, -1, 109.0, 49.0)
        );
        for (event, outcome) in actual.events.iter().zip(&actual.outcomes) {
            assert_eq!(event.candidate_id, outcome.candidate_id);
            assert_eq!(event.scenario_id, outcome.scenario_id);
            assert_eq!(outcome.exit_bar, 2);
        }
    }

    #[test]
    fn oracle_level_10_matches_full_cpu_for_cost_win_loss_and_no_trade_paths() {
        let workload = canonical_cost_fixture();
        let expected = full_cpu_metrics(&workload);
        let actual = evaluate_population_oracle(&workload).unwrap();

        assert_eq!(actual.metrics.len(), 3);
        for (index, row) in actual.metrics.iter().enumerate() {
            assert_eq!(row.candidate_id, workload.genes.candidate_ids[index]);
            assert_eq!(
                row.scenario_id,
                workload.scenarios.scenarios[index].scenario_id
            );
            for (slot, (oracle, cpu)) in row.values.iter().zip(&expected[index]).enumerate() {
                assert!(
                    (oracle - cpu).abs() <= 1.0e-10,
                    "candidate {index} metric slot {slot}: oracle={oracle}, cpu={cpu}"
                );
            }
        }

        assert!(actual.metrics[0].values[0] > 4_000.0);
        assert!(actual.metrics[1].values[0] < 0.0);
        assert_eq!(actual.metrics[2].values[0], 0.0);
        assert_eq!(actual.metrics[2].values[2], 100_000.0);
        assert_eq!(actual.metrics[2].values[8], 0.0);
        assert!(actual.metrics[0].values[10] > 0.0);
        assert!(actual.metrics[1].values[10] > actual.metrics[0].values[10]);
    }

    #[test]
    fn oracle_metric_row_keeps_raw_monthly_hit_rate_in_slot_seven() {
        let workload = canonical_cost_fixture();
        let row = evaluate_population_oracle(&workload).unwrap().metrics[0];

        assert_eq!(row.values[7], 1.0);
        assert_eq!(
            BacktestMetrics::from_metric_array(row.values).to_metric_array()[7],
            0.0
        );
    }

    #[test]
    fn population_settings_capture_runtime_equity_capacity_and_cost_flags() {
        let workload = canonical_cost_fixture();
        let settings = population_settings(&workload).unwrap();

        assert_eq!(settings.abi_version, neoethos_gpu_contracts::ABI_VERSION);
        assert_eq!(settings.initial_equity, 100_000.0);
        assert_eq!(settings.month_capacity, 240);
        assert_eq!(settings.spread_pips, 2.0);
        assert_eq!(settings.commission_per_trade, 10.0);
        assert_eq!(settings.flags, 0);
    }

    #[test]
    fn gap_exit_allows_canonical_same_bar_reentry() {
        let mut workload = canonical_cost_fixture();
        workload.dataset.settings.gap_threshold_ms = 86_400_000;
        workload.dataset.indicators[1] = 1.0;

        let expected = full_cpu_metrics(&workload);
        let actual = evaluate_population_oracle(&workload).unwrap();

        assert_eq!(actual.counters.event_count, 3);
        assert_eq!(actual.counters.accepted_trade_count, 3);
        for (slot, (oracle, cpu)) in actual.metrics[0]
            .values
            .iter()
            .zip(&expected[0])
            .enumerate()
        {
            assert!(
                (oracle - cpu).abs() <= 1.0e-10,
                "gap/re-entry metric slot {slot}: oracle={oracle}, cpu={cpu}"
            );
        }
    }

    #[test]
    fn reduction_rejects_reordered_same_identity_outcomes() {
        let mut workload = canonical_cost_fixture();
        workload.dataset.settings.gap_threshold_ms = 86_400_000;
        workload.dataset.indicators[1] = 1.0;
        let events = emit_population_events(&workload).unwrap();
        let mut outcomes = resolve_population_outcomes(&workload, &events).unwrap();

        assert_eq!(
            (events[0].candidate_id, events[0].scenario_id),
            (events[1].candidate_id, events[1].scenario_id)
        );
        assert_ne!(
            (outcomes[0].exit_bar, outcomes[0].exit_reason),
            (outcomes[1].exit_bar, outcomes[1].exit_reason)
        );
        assert!(reduce_population_outcomes(&workload, &events, &outcomes).is_ok());

        outcomes.swap(0, 1);
        assert_eq!(
            reduce_population_outcomes(&workload, &events, &outcomes),
            Err(PopulationOracleError::OutcomeSemanticMismatch {
                index: 0,
                candidate_id: 101,
                scenario_id: 1001,
                expected_exit_bar: 2,
                expected_exit_reason: neoethos_gpu_contracts::POPULATION_EXIT_GAP,
                actual_exit_bar: -1,
                actual_exit_reason: neoethos_gpu_contracts::POPULATION_EXIT_NONE,
            })
        );
    }

    #[test]
    fn adaptive_entry_levels_match_full_cpu_and_ignore_fixed_fallback_values() {
        let mut workload = canonical_cost_fixture();
        workload.dataset.settings.adaptive_base_pips = Some(vec![10.0; BARS]);
        workload.dataset.settings.adaptive_rr = 5.0;
        workload.genes.stop_vol_multipliers[0] = 1.0;
        workload.genes.stop_pips[0] = 999.0;
        workload.genes.target_pips[0] = 999.0;

        let expected = full_cpu_metrics(&workload);
        let actual = evaluate_population_oracle(&workload).unwrap();

        assert_eq!(actual.events[0].stop_price, 91.0);
        assert_eq!(actual.events[0].target_price, 151.0);
        for (slot, (oracle, cpu)) in actual.metrics[0]
            .values
            .iter()
            .zip(&expected[0])
            .enumerate()
        {
            assert!(
                (oracle - cpu).abs() <= 1.0e-10,
                "adaptive metric slot {slot}: oracle={oracle}, cpu={cpu}"
            );
        }
    }

    #[test]
    fn minimum_hold_extends_an_earlier_max_hold_boundary() {
        let mut workload = canonical_cost_fixture();
        workload.dataset.settings.max_hold_bars = 1;
        workload.dataset.settings.min_hold_bars = 2;

        let expected = full_cpu_metrics(&workload);
        let actual = evaluate_population_oracle(&workload).unwrap();

        assert_eq!(actual.events[0].last_bar, 3);
        assert_eq!(actual.outcomes[0].exit_bar, 3);
        assert_eq!(
            actual.outcomes[0].exit_reason,
            neoethos_gpu_contracts::POPULATION_EXIT_MAX_HOLD
        );
        for (slot, (oracle, cpu)) in actual.metrics[0]
            .values
            .iter()
            .zip(&expected[0])
            .enumerate()
        {
            assert!(
                (oracle - cpu).abs() <= 1.0e-10,
                "min/max-hold metric slot {slot}: oracle={oracle}, cpu={cpu}"
            );
        }
    }

    #[test]
    fn oracle_matches_risk_sized_multi_trade_population_fixture() {
        use crate::gpu_native::population_fixture::TinyPopulationFixture;

        let fixture = TinyPopulationFixture::new(4, 128, 4);
        let workload = fixture
            .population_workload(PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            })
            .unwrap();
        let expected = full_cpu_metrics(&workload);
        let actual = evaluate_population_oracle(&workload).unwrap();

        for (index, row) in actual.metrics.iter().enumerate() {
            for (slot, (oracle, cpu)) in row.values.iter().zip(&expected[index]).enumerate() {
                assert!(
                    (oracle - cpu).abs() <= 1.0e-8,
                    "candidate {index} metric slot {slot}: oracle={oracle}, cpu={cpu}"
                );
            }
        }
    }
}
