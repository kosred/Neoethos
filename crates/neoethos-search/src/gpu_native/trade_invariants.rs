//! Properties a trade record must satisfy whoever computed it.
//!
//! Parity against the CPU proves that the card *reproduces* the CPU. It does
//! not prove either one is right, and the difference is not academic: the CPU
//! cost model has a documented `estimate_pip_value_per_lot` fallback that
//! returns quote-currency units as if they were account currency and logs
//! itself as "silently wrong". A parity suite would carry that across to the
//! GPU and report success.
//!
//! These checks are therefore written against arithmetic and definitions, not
//! against a reference implementation. They cannot tell which side is wrong,
//! but unlike a parity test they can tell that *both* are — which is the case
//! parity is structurally blind to.

use neoethos_gpu_contracts::device::{NeoPopulationEvent, NeoPopulationOutcome};
use neoethos_gpu_contracts::{POPULATION_DIRECTION_LONG, POPULATION_EXIT_NONE};

/// A property that did not hold, with enough context to locate the trade.
#[derive(Debug, Clone, PartialEq)]
pub struct TradeInvariantViolation {
    pub index: usize,
    pub candidate_id: u64,
    pub property: &'static str,
    pub detail: String,
}

impl std::fmt::Display for TradeInvariantViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trade {} (candidate {}) violates {}: {}",
            self.index, self.candidate_id, self.property, self.detail
        )
    }
}

/// Bars the excursion was measured over, used to bound MFE/MAE against the
/// prices that actually occurred.
pub struct PriceSeries<'a> {
    pub high: &'a [f64],
    pub low: &'a [f64],
}

/// Check every settled trade against properties that hold by definition.
///
/// Returns every violation rather than the first: one broken property usually
/// breaks several, and the pattern says more than the earliest failure does.
pub fn check_trade_invariants(
    events: &[NeoPopulationEvent],
    outcomes: &[NeoPopulationOutcome],
    prices: PriceSeries<'_>,
    pip_value: f64,
    pip_value_per_lot: f64,
) -> Vec<TradeInvariantViolation> {
    let mut violations = Vec::new();
    let mut push = |index: usize, candidate_id: u64, property: &'static str, detail: String| {
        violations.push(TradeInvariantViolation {
            index,
            candidate_id,
            property,
            detail,
        });
    };

    for (index, outcome) in outcomes.iter().enumerate() {
        if outcome.exit_reason == POPULATION_EXIT_NONE {
            // Never opened or never resolved: nothing was settled, so there is
            // no trade to hold to account.
            continue;
        }
        let candidate = outcome.candidate_id;

        // Excursion is a distance from the entry in one direction. It cannot be
        // negative in either sense — the CPU floors both at zero and so does the
        // kernel, so a negative value means one of them stopped doing that.
        if !(outcome.mfe >= 0.0) {
            push(index, candidate, "mfe_non_negative", format!("mfe={}", outcome.mfe));
        }
        if !(outcome.mae >= 0.0) {
            push(index, candidate, "mae_non_negative", format!("mae={}", outcome.mae));
        }

        // A trade cannot close better than the best price it ever saw. Compared
        // per lot, because that is the scale excursion is measured on while P&L
        // is scaled by position size.
        if outcome.pnl > 0.0 {
            let realised_per_lot = outcome.pnl;
            if outcome.mfe + 1e-6 < realised_per_lot.min(f64::MAX) && outcome.mfe > 0.0 {
                push(
                    index,
                    candidate,
                    "mfe_covers_realised_gain",
                    format!("mfe={} < pnl={}", outcome.mfe, outcome.pnl),
                );
            }
        }

        // Time runs forward.
        if outcome.exit_bar >= 0 && outcome.exit_bar <= outcome.entry_bar {
            push(
                index,
                candidate,
                "exit_after_entry",
                format!("entry_bar={} exit_bar={}", outcome.entry_bar, outcome.exit_bar),
            );
        }

        // R-multiple is P&L divided by a positive risk, so it carries P&L's
        // sign. A mismatch means the denominator went negative or the two were
        // settled from different trades.
        if outcome.pnl != 0.0
            && outcome.r_multiple != 0.0
            && outcome.pnl.signum() != outcome.r_multiple.signum()
        {
            push(
                index,
                candidate,
                "r_multiple_sign_matches_pnl",
                format!("pnl={} r_multiple={}", outcome.pnl, outcome.r_multiple),
            );
        }

        // Excursion must be reachable from the prices in the window. This is
        // the check that does not depend on either implementation's arithmetic:
        // it recomputes the bound straight from the series.
        let Some(event) = events.get(index) else {
            continue;
        };
        let first = (outcome.entry_bar + 1).max(0) as usize;
        let last = if outcome.exit_bar >= 0 {
            outcome.exit_bar as usize
        } else {
            continue;
        };
        if first > last || last >= prices.high.len() || last >= prices.low.len() {
            continue;
        }
        let pip = if pip_value.abs() < 1e-12 { 1e-12 } else { pip_value };
        let (mut max_fav, mut max_adv) = (0.0_f64, 0.0_f64);
        for bar in first..=last {
            let (fav, adv) = if event.direction == POPULATION_DIRECTION_LONG {
                (prices.high[bar] - event.entry_price, event.entry_price - prices.low[bar])
            } else {
                (event.entry_price - prices.low[bar], prices.high[bar] - event.entry_price)
            };
            max_fav = max_fav.max(fav / pip * pip_value_per_lot);
            max_adv = max_adv.max(adv / pip * pip_value_per_lot);
        }
        let tolerance = 1e-6 * max_fav.abs().max(max_adv.abs()).max(1.0);
        if (outcome.mfe - max_fav).abs() > tolerance {
            push(
                index,
                candidate,
                "mfe_matches_window_prices",
                format!("reported={} from prices={}", outcome.mfe, max_fav),
            );
        }
        if (outcome.mae - max_adv).abs() > tolerance {
            push(
                index,
                candidate,
                "mae_matches_window_prices",
                format!("reported={} from prices={}", outcome.mae, max_adv),
            );
        }
    }

    violations
}

/// Do the settled trades add up to the aggregate the reducer reported?
///
/// The two are produced by different code inside the same kernel — one
/// accumulates as it walks, the other is written per trade — so agreement is
/// evidence, not tautology. Disagreement means a trade was settled against the
/// wrong slot or counted twice, which no per-trade property would reveal.
pub fn check_pnl_reconciles(
    outcomes: &[NeoPopulationOutcome],
    candidate_id: u64,
    aggregate_net_profit: f64,
) -> Option<String> {
    let summed: f64 = outcomes
        .iter()
        .filter(|o| o.candidate_id == candidate_id && o.exit_reason != POPULATION_EXIT_NONE)
        .map(|o| o.pnl)
        .sum();
    let tolerance = 1e-6 * summed.abs().max(aggregate_net_profit.abs()).max(1.0);
    ((summed - aggregate_net_profit).abs() > tolerance).then(|| {
        format!(
            "candidate {candidate_id}: per-trade P&L sums to {summed} but the reducer \
             reported {aggregate_net_profit}"
        )
    })
}

/// The whole net over one device readback, as one call.
///
/// `aggregates` is `(candidate_id, net_profit)` straight off the reducer's
/// metric rows — `values[0]`. Every candidate that appears there is reconciled
/// against the trades that were settled for it, and every settled trade is held
/// to the per-trade properties above.
///
/// Returns rendered complaints rather than the typed violations because the two
/// checks produce different shapes and every caller so far wants the same
/// thing: a list to refuse the run with. An empty vector means the readback
/// held.
///
/// `events` may be empty. The device does not maintain an event stream — the
/// reduce opens positions from the signal, and `read_diagnostics` passes a null
/// events pointer deliberately (`neoethos-gpu-cuda/src/population.rs:700`) — so
/// on that path the two price-window properties do not fire and the arithmetic
/// properties do. Callers that *do* have the entries (a host reference walk)
/// get the window bound checked as well.
pub fn audit_device_outcomes(
    events: &[NeoPopulationEvent],
    outcomes: &[NeoPopulationOutcome],
    aggregates: &[(u64, f64)],
    prices: PriceSeries<'_>,
    pip_value: f64,
    pip_value_per_lot: f64,
) -> Vec<String> {
    let mut complaints: Vec<String> = check_trade_invariants(
        events,
        outcomes,
        prices,
        pip_value,
        pip_value_per_lot,
    )
    .into_iter()
    .map(|violation| violation.to_string())
    .collect();
    for (candidate_id, net_profit) in aggregates {
        if let Some(complaint) = check_pnl_reconciles(outcomes, *candidate_id, *net_profit) {
            complaints.push(complaint);
        }
    }
    complaints
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_gpu_contracts::POPULATION_EXIT_TARGET;

    fn event(direction: i32, entry_price: f64) -> NeoPopulationEvent {
        NeoPopulationEvent {
            candidate_id: 1,
            scenario_id: 1,
            entry_bar: 0,
            last_bar: 4,
            direction,
            precedence: 0,
            stop_price: entry_price - 1.0,
            target_price: entry_price + 1.0,
            entry_price,
        }
    }

    fn outcome(mfe: f64, mae: f64, pnl: f64, r_multiple: f64) -> NeoPopulationOutcome {
        NeoPopulationOutcome {
            candidate_id: 1,
            scenario_id: 1,
            exit_bar: 3,
            exit_reason: POPULATION_EXIT_TARGET,
            entry_bar: 0,
            _pad: 0,
            mfe,
            mae,
            exit_price: 0.0,
            pnl,
            r_multiple,
        }
    }

    /// The bound is recomputed from the series, so a hand-checked window is the
    /// one place the whole stack can be caught agreeing on a wrong answer.
    #[test]
    fn excursion_is_checked_against_the_prices_that_occurred() {
        // Entry at 100. Bars 1..=3 reach a high of 102 and a low of 99, so with
        // pip 0.01 and 1 unit per pip: favourable 200, adverse 100.
        let high = [100.0, 101.0, 102.0, 100.5, 100.0];
        let low = [100.0, 99.5, 100.0, 99.0, 100.0];
        let events = [event(POPULATION_DIRECTION_LONG, 100.0)];

        let honest = [outcome(200.0, 100.0, 50.0, 1.0)];
        assert!(
            check_trade_invariants(
                &events,
                &honest,
                PriceSeries { high: &high, low: &low },
                0.01,
                1.0,
            )
            .is_empty()
        );

        // Both implementations agreeing on 150 would pass any parity suite.
        let agreed_but_wrong = [outcome(150.0, 100.0, 50.0, 1.0)];
        let violations = check_trade_invariants(
            &events,
            &agreed_but_wrong,
            PriceSeries { high: &high, low: &low },
            0.01,
            1.0,
        );
        assert!(
            violations
                .iter()
                .any(|v| v.property == "mfe_matches_window_prices"),
            "{violations:?}"
        );
    }

    #[test]
    fn negative_excursion_and_backwards_time_are_rejected() {
        let high = [100.0, 101.0];
        let low = [100.0, 99.0];
        let events = [event(POPULATION_DIRECTION_LONG, 100.0)];
        let mut bad = outcome(-1.0, -2.0, 10.0, -1.0);
        bad.exit_bar = 0;
        let violations = check_trade_invariants(
            &events,
            &[bad],
            PriceSeries { high: &high, low: &low },
            0.01,
            1.0,
        );
        for property in [
            "mfe_non_negative",
            "mae_non_negative",
            "exit_after_entry",
            "r_multiple_sign_matches_pnl",
        ] {
            assert!(
                violations.iter().any(|v| v.property == property),
                "missing {property} in {violations:?}"
            );
        }
    }

    #[test]
    fn unresolved_trades_are_not_held_to_account() {
        let high = [100.0];
        let low = [100.0];
        let mut open = outcome(0.0, 0.0, 0.0, 0.0);
        open.exit_reason = POPULATION_EXIT_NONE;
        open.exit_bar = -1;
        assert!(
            check_trade_invariants(
                &[event(POPULATION_DIRECTION_LONG, 100.0)],
                &[open],
                PriceSeries { high: &high, low: &low },
                0.01,
                1.0,
            )
            .is_empty()
        );
    }

    #[test]
    fn per_trade_pnl_must_reconcile_with_the_reducer_aggregate() {
        let trades = [outcome(10.0, 5.0, 30.0, 1.0), outcome(10.0, 5.0, -12.0, -0.4)];
        assert!(check_pnl_reconciles(&trades, 1, 18.0).is_none());
        let complaint = check_pnl_reconciles(&trades, 1, 25.0).expect("mismatch must be reported");
        assert!(complaint.contains("18"), "{complaint}");
    }

    /// The device path supplies no events. The arithmetic properties and the
    /// reconciliation must still fire, or wiring the net there buys nothing.
    #[test]
    fn the_audit_still_bites_when_the_device_supplies_no_event_stream() {
        let high = [100.0, 101.0];
        let low = [100.0, 99.0];
        let mut bad = outcome(-1.0, 0.0, 10.0, -1.0);
        bad.exit_bar = 1;
        let complaints = audit_device_outcomes(
            &[],
            &[bad],
            &[(1, 999.0)],
            PriceSeries { high: &high, low: &low },
            0.01,
            1.0,
        );
        assert!(
            complaints.iter().any(|c| c.contains("mfe_non_negative")),
            "{complaints:?}"
        );
        assert!(
            complaints
                .iter()
                .any(|c| c.contains("r_multiple_sign_matches_pnl")),
            "{complaints:?}"
        );
        assert!(
            complaints.iter().any(|c| c.contains("reducer reported 999")),
            "{complaints:?}"
        );
    }

    #[test]
    fn an_honest_readback_produces_no_complaints() {
        let high = [100.0, 101.0, 102.0, 100.5, 100.0];
        let low = [100.0, 99.5, 100.0, 99.0, 100.0];
        let honest = [outcome(200.0, 100.0, 50.0, 1.0)];
        assert!(
            audit_device_outcomes(
                &[event(POPULATION_DIRECTION_LONG, 100.0)],
                &honest,
                &[(1, 50.0)],
                PriceSeries { high: &high, low: &low },
                0.01,
                1.0,
            )
            .is_empty()
        );
    }
}
