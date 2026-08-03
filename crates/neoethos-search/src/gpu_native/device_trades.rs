//! Turning the device's settled outcomes back into the trade list the rest of
//! discovery works with.
//!
//! The quality screen needs `Trade` values, not aggregate metrics: the analyzer
//! derives `StrategyMetrics` from them and a capped sample is exported. That is
//! the only reason its backtest stayed on the CPU while everything around it
//! moved to the card. With the kernel now settling P&L, excursion and
//! r-multiple per trade, the remaining gap is purely a translation — bars to
//! timestamps, and the two host-side fields the device has no business knowing
//! about (duration and P&L as a fraction of the starting balance).
//!
//! The conventions here are copied from `eval.rs` deliberately, down to
//! returning `None` for a duration when the exit timestamp precedes the entry.
//! Diverging would show up as a parity failure that looks like a kernel bug.

use neoethos_gpu_contracts::device::NeoPopulationOutcome;
use neoethos_gpu_contracts::POPULATION_EXIT_NONE;

use crate::quality::Trade;

/// Milliseconds in an hour, matching the CPU walk's divisor exactly.
const MS_PER_HOUR: f64 = 3_600_000.0;

/// Convert one candidate's settled outcomes into its trade list.
///
/// Outcomes that never resolved are skipped rather than reported as
/// zero-valued trades: an unresolved position is not a trade, and emitting one
/// would inflate the trade count that downstream gates read.
pub fn trades_from_outcomes(
    outcomes: &[NeoPopulationOutcome],
    candidate_id: u64,
    timestamps: &[i64],
    initial_balance: f64,
) -> Vec<Trade> {
    outcomes
        .iter()
        .filter(|outcome| {
            outcome.candidate_id == candidate_id && outcome.exit_reason != POPULATION_EXIT_NONE
        })
        .filter_map(|outcome| {
            let entry_bar = usize::try_from(outcome.entry_bar).ok()?;
            let exit_bar = usize::try_from(outcome.exit_bar).ok()?;
            let entry_time = timestamps.get(entry_bar).copied()?;
            let exit_time = timestamps.get(exit_bar).copied()?;
            let duration_hours = (exit_time >= entry_time)
                .then(|| (exit_time - entry_time) as f64 / MS_PER_HOUR);
            Some(Trade {
                entry_time,
                exit_time: Some(exit_time),
                pnl: outcome.pnl,
                // A zero starting balance would make this meaningless rather
                // than infinite, so it is reported as absent.
                pnl_pct: (initial_balance != 0.0).then(|| outcome.pnl / initial_balance),
                duration_hours,
                mfe: outcome.mfe,
                mae: outcome.mae,
                r_multiple: outcome.r_multiple,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_gpu_contracts::POPULATION_EXIT_TARGET;

    fn outcome(candidate_id: u64, entry_bar: i32, exit_bar: i32, pnl: f64) -> NeoPopulationOutcome {
        NeoPopulationOutcome {
            candidate_id,
            scenario_id: 1,
            exit_bar,
            exit_reason: POPULATION_EXIT_TARGET,
            entry_bar,
            _pad: 0,
            mfe: 12.0,
            mae: 4.0,
            exit_price: 0.0,
            pnl,
            r_multiple: pnl / 10.0,
        }
    }

    fn timestamps() -> Vec<i64> {
        // One bar an hour.
        (0..6).map(|bar| bar as i64 * 3_600_000).collect()
    }

    #[test]
    fn settled_outcomes_become_trades_with_cpu_conventions() {
        let ts = timestamps();
        let trades = trades_from_outcomes(&[outcome(7, 1, 4, 250.0)], 7, &ts, 10_000.0);
        assert_eq!(trades.len(), 1);
        let trade = &trades[0];
        assert_eq!(trade.entry_time, 3_600_000);
        assert_eq!(trade.exit_time, Some(4 * 3_600_000));
        assert_eq!(trade.duration_hours, Some(3.0));
        assert_eq!(trade.pnl, 250.0);
        assert_eq!(trade.pnl_pct, Some(0.025));
        assert_eq!(trade.mfe, 12.0);
        assert_eq!(trade.mae, 4.0);
    }

    #[test]
    fn other_candidates_and_unresolved_positions_are_skipped() {
        let ts = timestamps();
        let mut open = outcome(7, 1, -1, 0.0);
        open.exit_reason = POPULATION_EXIT_NONE;
        let outcomes = [outcome(7, 1, 3, 10.0), outcome(8, 1, 3, 99.0), open];
        let trades = trades_from_outcomes(&outcomes, 7, &ts, 1_000.0);
        assert_eq!(trades.len(), 1, "only candidate 7's settled trade");
        assert_eq!(trades[0].pnl, 10.0);
    }

    /// A bar index outside the series is a contract violation, not a trade to
    /// invent a timestamp for.
    #[test]
    fn out_of_range_bars_drop_the_trade_rather_than_guessing() {
        let ts = timestamps();
        let trades = trades_from_outcomes(&[outcome(7, 1, 99, 10.0)], 7, &ts, 1_000.0);
        assert!(trades.is_empty());
    }

    #[test]
    fn a_zero_starting_balance_reports_no_percentage_instead_of_infinity() {
        let ts = timestamps();
        let trades = trades_from_outcomes(&[outcome(7, 0, 2, 5.0)], 7, &ts, 0.0);
        assert_eq!(trades[0].pnl_pct, None);
    }
}
