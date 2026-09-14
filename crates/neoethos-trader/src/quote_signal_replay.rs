//! Canonical fixed/adaptive signal lanes over the existing quote execution kernel.
//!
//! Reuses DecisionEngine brackets and Position's trailing arithmetic. Quotes
//! decide fills and position occupancy; OHLC never supplies execution prices.
//! This is research execution, not lot sizing, account economics or promotion.

use anyhow::{Context, Result, ensure};
use neoethos_broker_truth::{
    CanonicalBarSignalResearchDecisionV1, ClosedCanonicalBarTimeExitV1,
    ClosedCanonicalBarTrailingThresholdV1, QuoteValidatedResearchEntryPreviewV1,
    QuoteValidatedResearchNonEntryV1, QuoteValidatedResearchPositionV1,
    QuoteValidatedResearchReplayBindingV1, QuoteValidatedResearchReplayPlanV1,
    QuoteValidatedResearchReplayPolicyV1, ResearchPositionDirectionV1,
    SealedHistoricalBidAskQuoteReplayEvidenceV1, SealedHistoricalQuoteValidatedResearchLedgerV1,
    preview_sealed_quote_validated_research_entry_v1, replay_sealed_quote_validated_research_v1,
};
use rayon::prelude::*;

use crate::contracts::{Direction, LiveBar, Signal, SignalSource, TradeIntent};
use crate::decision::{DecisionConfig, DecisionEngine};
use crate::position::TrailingPolicy;

/// Borrow the exact prepared full-SMC signals; do not regenerate them with
/// default evaluation settings. One lane owns one gene and at most one open
/// position. A separate lane may run concurrently over the same sealed quotes.
pub struct CanonicalSignalQuoteLaneV1<'a> {
    pub bars: &'a [LiveBar],
    pub signals: &'a [i8],
    pub gene: &'a neoethos_search::Gene,
    pub pip_size: f64,
    /// Zero disables time exits; the reviewed quote window still bounds replay.
    pub max_hold_bars: usize,
    pub trailing: Option<TrailingPolicy>,
}

impl CanonicalSignalQuoteLaneV1<'_> {
    pub fn validate(
        &self,
        binding: &QuoteValidatedResearchReplayBindingV1,
        policy: &QuoteValidatedResearchReplayPolicyV1,
    ) -> Result<i64> {
        self.borrowed().validate(binding, policy)
    }

    fn borrowed(&self) -> BorrowedSignalQuoteLane<'_> {
        BorrowedSignalQuoteLane {
            bars: QuoteBars::Live(self.bars),
            signals: self.signals,
            confidences: None, // Explicit geometry-only V1 diagnostic.
            entry_eligibility: None,
            entry_brackets: None,
            gene: self.gene,
            pip_size: self.pip_size,
            max_hold_bars: self.max_hold_bars,
            trailing: self.trailing,
        }
    }
}

/// A row view, not an owned copy of the historical series. In the locked V2
/// route the OHLC columns, timestamps, symbol and timeframe are all borrowed.
#[derive(Clone, Copy)]
struct QuoteBar<'a> {
    symbol: &'a str,
    tf: &'a str,
    ts: i64,
    o: f64,
    h: f64,
    l: f64,
    c: f64,
}

#[derive(Clone, Copy)]
enum QuoteBars<'a> {
    Live(&'a [LiveBar]),
    Canonical {
        ohlcv: &'a neoethos_data::Ohlcv,
        timestamps: &'a [i64],
        symbol: &'a str,
        timeframe: &'a str,
    },
}

impl<'a> QuoteBars<'a> {
    fn len(self) -> usize {
        match self {
            Self::Live(bars) => bars.len(),
            Self::Canonical { timestamps, .. } => timestamps.len(),
        }
    }

    fn get(self, index: usize) -> Option<QuoteBar<'a>> {
        Some(match self {
            Self::Live(bars) => {
                let bar = bars.get(index)?;
                QuoteBar {
                    symbol: &bar.symbol,
                    tf: &bar.tf,
                    ts: bar.ts,
                    o: bar.o,
                    h: bar.h,
                    l: bar.l,
                    c: bar.c,
                }
            }
            Self::Canonical {
                ohlcv,
                timestamps,
                symbol,
                timeframe,
            } => QuoteBar {
                symbol,
                tf: timeframe,
                ts: *timestamps.get(index)?,
                o: *ohlcv.open.get(index)?,
                h: *ohlcv.high.get(index)?,
                l: *ohlcv.low.get(index)?,
                c: *ohlcv.close.get(index)?,
            },
        })
    }

    fn partition_at(self, timestamp: i64) -> usize {
        match self {
            Self::Live(bars) => bars.partition_point(|bar| bar.ts <= timestamp),
            Self::Canonical { timestamps, .. } => timestamps.partition_point(|ts| *ts <= timestamp),
        }
    }
}

struct BorrowedSignalQuoteLane<'a> {
    bars: QuoteBars<'a>,
    signals: &'a [i8],
    confidences: Option<&'a [f64]>,
    entry_eligibility: Option<&'a dyn Fn(usize) -> bool>,
    entry_brackets: Option<&'a dyn Fn(usize) -> Option<(f64, f64)>>,
    gene: &'a neoethos_search::Gene,
    pip_size: f64,
    max_hold_bars: usize,
    trailing: Option<TrailingPolicy>,
}

impl BorrowedSignalQuoteLane<'_> {
    fn validate(
        &self,
        binding: &QuoteValidatedResearchReplayBindingV1,
        policy: &QuoteValidatedResearchReplayPolicyV1,
    ) -> Result<i64> {
        binding.validate_replay_policy_v1(policy)?;
        ensure!(
            self.bars.len() >= 2 && self.bars.len() == self.signals.len(),
            "quote signal lane needs matching closed bars and signals"
        );
        ensure!(
            self.confidences
                .is_none_or(|values| values.len() == self.signals.len()
                    && values
                        .iter()
                        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))),
            "quote signal lane needs finite aligned gene confidence"
        );
        if let QuoteBars::Canonical {
            ohlcv, timestamps, ..
        } = self.bars
        {
            ensure!(
                [
                    ohlcv.open.len(),
                    ohlcv.high.len(),
                    ohlcv.low.len(),
                    ohlcv.close.len()
                ]
                .into_iter()
                .all(|length| length == timestamps.len()),
                "quote signal lane has mismatched OHLC column lengths"
            );
        }
        ensure!(
            self.entry_brackets.is_some() || self.gene.stop_vol_mult == 0.0,
            "fixed-stop quote lane cannot replace adaptive per-bar strategy brackets"
        );
        ensure!(
            [self.pip_size, self.gene.sl_pips, self.gene.tp_pips]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "quote signal lane requires explicit finite positive pip size and gene SL/TP"
        );
        ensure!(
            self.pip_size == policy.pip_size(),
            "strategy and captured quote policy have different pip sizes"
        );
        if let Some(trailing) = self.trailing {
            ensure!(
                trailing.pip_size == self.pip_size
                    && TrailingPolicy::new(
                        trailing.be_trigger_r,
                        trailing.stop_multiplier,
                        trailing.min_lock_pips,
                        trailing.pip_size
                    )
                    .is_some(),
                "quote signal lane has invalid or mismatched trailing geometry"
            );
        }
        let first = self
            .bars
            .get(0)
            .context("quote signal lane has incomplete OHLC columns")?;
        let timeframe = first.tf.parse::<neoethos_data::CanonicalTimeframe>()?;
        let duration = timeframe.fixed_duration_ms().context(
            "calendar timeframes require an explicit broker calendar; no invented bar close",
        )?;
        let window = binding.replay_scope().locked_evaluation_window();
        ensure!(
            first.ts == window.from_unix_ms_inclusive()
                && self
                    .bars
                    .get(self.bars.len() - 1)
                    .and_then(|bar| bar.ts.checked_add(duration))
                    == Some(window.to_unix_ms_exclusive()),
            "quote signal lane must cover the exact locked bar window"
        );
        for (index, signal) in self.signals.iter().enumerate() {
            let bar = self
                .bars
                .get(index)
                .context("quote signal lane has incomplete OHLC columns")?;
            ensure!(
                bar.symbol == binding.symbol_name() && bar.tf == timeframe.as_str(),
                "quote signal lane changes its captured symbol or timeframe at row {index}"
            );
            ensure!(
                (-1..=1).contains(signal),
                "invalid strategy direction at row {index}"
            );
            ensure!(
                bar.ts >= 0
                    && bar.ts % duration == 0
                    && bar
                        .ts
                        .checked_add(duration)
                        .is_some_and(|end| end <= window.to_unix_ms_exclusive())
                    && (index == 0
                        || self
                            .bars
                            .get(index - 1)
                            .expect("previous row was validated")
                            .ts
                            .checked_add(duration)
                            .is_some_and(|end| end <= bar.ts)),
                "quote signal lane has unordered, overlapping or unaligned bars at row {index}"
            );
            ensure!(
                [bar.o, bar.h, bar.l, bar.c]
                    .iter()
                    .all(|price| price.is_finite() && *price > 0.0)
                    && bar.l <= bar.o
                    && bar.l <= bar.c
                    && bar.h >= bar.o
                    && bar.h >= bar.c,
                "quote signal lane has invalid OHLC at row {index}"
            );
        }
        Ok(duration)
    }
}

#[derive(Debug)]
pub struct CanonicalSignalQuoteOutcomeV1 {
    pub decision_bar_index: usize,
    pub risk_pips: f64,
    pub ledger: SealedHistoricalQuoteValidatedResearchLedgerV1,
}

/// The reviewed acquisition consumer calls this after opening its one opaque
/// quote snapshot. No ambient financial gate is bypassed, no broker is called,
/// and no mock execution or points-times-volume P&L enters this route.
pub fn replay_canonical_signal_quote_lane_v1(
    lane: &CanonicalSignalQuoteLaneV1<'_>,
    binding: &QuoteValidatedResearchReplayBindingV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    evidence: &SealedHistoricalBidAskQuoteReplayEvidenceV1,
) -> Result<Vec<CanonicalSignalQuoteOutcomeV1>> {
    evidence.validate_replay_context_v1(binding, policy)?;
    let outcomes = drive_lane(
        lane,
        binding,
        policy,
        |plan| {
            Ok(preview_sealed_quote_validated_research_entry_v1(
                plan, evidence,
            )?)
        },
        |plan| Ok(replay_sealed_quote_validated_research_v1(plan, evidence)?),
    )?;
    Ok(outcomes
        .into_iter()
        .map(
            |(decision_bar_index, ledger)| CanonicalSignalQuoteOutcomeV1 {
                decision_bar_index,
                risk_pips: lane.gene.sl_pips,
                ledger,
            },
        )
        .collect())
}

/// The acquisition hash and the producer consume this SAME immutable plan.
/// Require the caller's exact leased Rayon pool before touching the shared input,
/// then replay independent genes there; causal sequencing stays inside each lane.
pub fn replay_locked_canonical_signal_portfolio_v3(
    cpu: &neoethos_core::execution::BudgetedCpuScope<'_>,
    locked: &neoethos_search::LockedCanonicalSignalPlanV3<'_>,
    binding: &QuoteValidatedResearchReplayBindingV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    evidence: &SealedHistoricalBidAskQuoteReplayEvidenceV1,
) -> Result<Vec<Vec<SealedHistoricalQuoteValidatedResearchLedgerV1>>> {
    cpu.require_current_pool()?;
    locked.validate_replay_binding(binding, policy)?;
    evidence.validate_replay_context_v1(binding, policy)?;
    let bars = QuoteBars::Canonical {
        ohlcv: locked.bars(),
        timestamps: locked.timestamps(),
        symbol: locked.symbol_name(),
        timeframe: locked.timeframe().as_str(),
    };
    let exit = locked.exit_policy();
    let trailing = if exit.trailing_enabled {
        Some(
            TrailingPolicy::new(
                exit.trailing_be_trigger_r,
                exit.trailing_stop_multiplier,
                exit.trailing_min_lock_pips,
                exit.pip_size,
            )
            .context("invalid locked trailing policy")?,
        )
    } else {
        None
    };
    locked
        .portfolio()
        .par_iter()
        .enumerate()
        .map(|(index, gene)| {
            cpu.require_current_pool()?;
            let eligible = |row| locked.entry_eligible(index, row);
            let brackets = |row| locked.entry_stop_target_pips(index, row);
            let lane = BorrowedSignalQuoteLane {
                bars,
                signals: &locked.ordered_signals()[index],
                confidences: Some(&locked.ordered_confidences()[index]),
                entry_eligibility: Some(&eligible),
                entry_brackets: Some(&brackets),
                gene,
                pip_size: exit.pip_size,
                max_hold_bars: exit.max_hold_bars,
                trailing,
            };
            drive_borrowed_lane(
                &lane,
                binding,
                policy,
                |plan| {
                    Ok(preview_sealed_quote_validated_research_entry_v1(
                        plan, evidence,
                    )?)
                },
                |plan| Ok(replay_sealed_quote_validated_research_v1(plan, evidence)?),
            )
            .map(|outcomes| outcomes.into_iter().map(|(_, ledger)| ledger).collect())
        })
        .collect()
}

// This private seam lets tests exercise the SAME lane and quote kernel with
// explicitly untrusted synthetic records, without manufacturing a broker seal.
trait LaneLedger {
    fn positions(&self) -> &[QuoteValidatedResearchPositionV1];
    fn non_entries(&self) -> &[QuoteValidatedResearchNonEntryV1];
}

impl LaneLedger for SealedHistoricalQuoteValidatedResearchLedgerV1 {
    fn positions(&self) -> &[QuoteValidatedResearchPositionV1] {
        self.positions()
    }
    fn non_entries(&self) -> &[QuoteValidatedResearchNonEntryV1] {
        self.entry_unavailable()
    }
}

fn bracket_decision(
    engine: &mut DecisionEngine,
    signal: &Signal,
    bar: &QuoteBar<'_>,
    duration: i64,
    entry_price: f64,
) -> Result<(CanonicalBarSignalResearchDecisionV1, f64, f64)> {
    let Some(TradeIntent::Open {
        sl: Some(stop),
        tp: Some(target),
        ..
    }) = engine.intent(signal, &[], entry_price)
    else {
        anyhow::bail!("directional strategy signal did not produce its own bracket");
    };
    let direction = match signal.dir {
        Direction::Long => ResearchPositionDirectionV1::Long,
        Direction::Short => ResearchPositionDirectionV1::Short,
        Direction::Flat => anyhow::bail!("flat signal cannot own an entry decision"),
    };
    Ok((
        CanonicalBarSignalResearchDecisionV1::new(
            bar.ts,
            bar.ts.checked_add(duration).context("bar close overflow")?,
            direction,
            stop,
            target,
        )?,
        stop,
        target,
    ))
}

fn drive_lane<'a, L: LaneLedger>(
    lane: &CanonicalSignalQuoteLaneV1<'_>,
    binding: &QuoteValidatedResearchReplayBindingV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    preview: impl FnMut(
        &QuoteValidatedResearchReplayPlanV1,
    ) -> Result<Option<QuoteValidatedResearchEntryPreviewV1<'a>>>,
    replay: impl FnMut(&QuoteValidatedResearchReplayPlanV1) -> Result<L>,
) -> Result<Vec<(usize, L)>> {
    drive_borrowed_lane(&lane.borrowed(), binding, policy, preview, replay)
}

fn drive_borrowed_lane<'a, L: LaneLedger>(
    lane: &BorrowedSignalQuoteLane<'_>,
    binding: &QuoteValidatedResearchReplayBindingV1,
    policy: &QuoteValidatedResearchReplayPolicyV1,
    mut preview: impl FnMut(
        &QuoteValidatedResearchReplayPlanV1,
    ) -> Result<Option<QuoteValidatedResearchEntryPreviewV1<'a>>>,
    mut replay: impl FnMut(&QuoteValidatedResearchReplayPlanV1) -> Result<L>,
) -> Result<Vec<(usize, L)>> {
    let duration = lane.validate(binding, policy)?;
    let window = binding.replay_scope().locked_evaluation_window();
    let mut engine = DecisionEngine::new(DecisionConfig::gene_parity(lane.pip_size));
    let mut occupied_until = None;
    let mut outcomes = Vec::new();
    for index in 0..lane.bars.len() {
        let bar = lane.bars.get(index).expect("all lane rows were validated");
        let decision_at = bar.ts.checked_add(duration).context("bar close overflow")?;
        if decision_at >= window.to_unix_ms_exclusive() {
            break;
        }
        // A signal while entry is pending or a position is open is not a new
        // trade. Equal-time close/reopen has no reviewed ordering: wait.
        if occupied_until.is_some_and(|until| decision_at <= until) {
            continue;
        }
        if lane
            .entry_eligibility
            .is_some_and(|eligible| !eligible(index))
        {
            continue;
        }
        let dir = match lane.signals[index] {
            1 => Direction::Long,
            -1 => Direction::Short,
            _ => continue,
        };
        let (sl_pips, tp_pips) = match lane.entry_brackets {
            Some(resolve) => match resolve(index) {
                Some(brackets) => brackets,
                None => continue,
            },
            None => (lane.gene.sl_pips, lane.gene.tp_pips),
        };
        let signal = Signal {
            symbol: bar.symbol.to_owned(),
            dir,
            // The V3 route preserves the exact Search pair. V1 computes only
            // geometry and carries no claim of a maximum-quality signal.
            confidence: lane.confidences.map_or(0.0, |values| values[index]),
            source: SignalSource::Strategy,
            sl_pips,
            tp_pips,
        };
        let (decision, _, _) = bracket_decision(&mut engine, &signal, &bar, duration, bar.c)?;
        let mut plan = QuoteValidatedResearchReplayPlanV1::new(
            binding.clone(),
            policy.clone(),
            vec![decision],
            Vec::new(),
        )?;
        if let Some(entry) = preview(&plan)? {
            let entry_index = lane
                .bars
                .partition_at(entry.timestamp_unix_ms())
                .checked_sub(1)
                .context("modeled entry precedes the locked bars")?;
            let entry_bar = lane
                .bars
                .get(entry_index)
                .expect("entry index belongs to validated bars");
            let entry_bar_end = entry_bar
                .ts
                .checked_add(duration)
                .context("entry bar close overflow")?;
            ensure!(
                entry.timestamp_unix_ms() < entry_bar_end,
                "modeled entry falls in a canonical bar gap; cannot invent a holding clock"
            );
            let (decision, stop, _) = bracket_decision(
                &mut engine,
                &signal,
                &bar,
                duration,
                entry.modeled_entry_price(),
            )?;
            let mut trail_stop = None;
            let last_hold_index = if lane.max_hold_bars == 0 {
                lane.bars.len()
            } else {
                entry_index
                    .checked_add(lane.max_hold_bars - 1)
                    .context("max-hold index overflow")?
            };
            let mut thresholds = Vec::new();
            if let Some(trailing) = &lane.trailing {
                // At most max_hold_bars closed bars per accepted position, not
                // a new full-history signal/feature pass per candidate signal.
                for trail_index in entry_index..last_hold_index.min(lane.bars.len()) {
                    let source = lane
                        .bars
                        .get(trail_index)
                        .expect("trailing index belongs to validated bars");
                    let end = source
                        .ts
                        .checked_add(duration)
                        .context("trailing bar close overflow")?;
                    let (high, low) = if trail_index == entry_index {
                        entry.bid_extrema_before(end)?
                    } else {
                        (source.h, source.l)
                    };
                    // Exactly the same policy consumed by Position::ratchet_trail,
                    // without constructing owned LiveBar strings or a fake position.
                    if let Some(next_stop) = trailing.next_stop_price(
                        entry.modeled_entry_price(),
                        stop,
                        lane.signals[index],
                        high,
                        low,
                        trail_stop,
                    ) {
                        trail_stop = Some(next_stop);
                        thresholds.push(ClosedCanonicalBarTrailingThresholdV1::new(
                            source.ts,
                            end,
                            if dir == Direction::Long {
                                ResearchPositionDirectionV1::Long
                            } else {
                                ResearchPositionDirectionV1::Short
                            },
                            next_stop,
                        )?);
                    }
                }
            }
            plan = QuoteValidatedResearchReplayPlanV1::new(
                binding.clone(),
                policy.clone(),
                vec![decision],
                thresholds,
            )?;
            if let Some(exit_bar) = lane.bars.get(last_hold_index) {
                plan = plan.with_time_exit(ClosedCanonicalBarTimeExitV1::new(
                    exit_bar.ts,
                    exit_bar
                        .ts
                        .checked_add(duration)
                        .context("time exit close overflow")?,
                )?)?;
            }
        }
        let ledger = replay(&plan)?;
        let terminal_open = match (ledger.positions(), ledger.non_entries()) {
            ([position], []) => {
                occupied_until = position
                    .exit_reference()
                    .map(|exit| exit.timestamp_unix_ms());
                occupied_until.is_none()
            }
            ([], [unavailable]) => {
                occupied_until = Some(unavailable.deadline_unix_ms());
                false
            }
            _ => anyhow::bail!(
                "quote kernel must return exactly one position or non-entry per decision"
            ),
        };
        outcomes.push((index, ledger));
        if terminal_open {
            break;
        }
    }
    Ok(outcomes)
}

#[cfg(test)]
mod tests;
